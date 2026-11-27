"""Two-stage T4 runner for pinned Qwen-Image-Layered GGUF assets."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def encode(args):
    from diffusers import QwenImageLayeredPipeline
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

    base = args.root / 'models/qwen-base'
    caption = args.caption_file.read_text(encoding='utf8').strip()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(8)
    encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base / 'text_encoder', torch_dtype=torch.float32, device_map={'': 'cpu'},
        local_files_only=True, attn_implementation='sdpa')
    encoder.eval()
    tokenizer = Qwen2Tokenizer.from_pretrained(base / 'tokenizer', local_files_only=True)
    pipe = QwenImageLayeredPipeline(scheduler=None, vae=None, text_encoder=encoder,
                                   tokenizer=tokenizer, processor=None, transformer=None)
    with torch.inference_mode():
        positive, positive_mask = pipe.encode_prompt(caption, device=torch.device('cpu'))
        negative, negative_mask = pipe.encode_prompt(' ', device=torch.device('cpu'))
    if positive_mask is None:
        positive_mask = torch.ones(positive.shape[:2], dtype=torch.int64)
    if negative_mask is None:
        negative_mask = torch.ones(negative.shape[:2], dtype=torch.int64)
    save_file({'positive': positive.contiguous(), 'positive_mask': positive_mask.contiguous(),
               'negative': negative.contiguous(), 'negative_mask': negative_mask.contiguous()},
              str(args.output / 'embeddings.safetensors'))
    write_json(args.output / 'caption.json', {'caption': caption,
               'caption_sha256': hashlib.sha256(caption.encode()).hexdigest(),
               'encoder_dtype': 'float32', 'encoder_device': 'cpu',
               'process_peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
               'elapsed_s': round(time.monotonic() - started, 3)})
    print('CPU prompt encoding completed', flush=True)


def infer(args):
    from accelerate import init_empty_weights
    from diffusers import (AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
                           GGUFQuantizationConfig, QwenImageLayeredPipeline,
                           QwenImageTransformer2DModel)
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    compute_dtype = {'float16': torch.float16, 'float32': torch.float32}[args.compute_dtype]

    class T4LayeredPipeline(QwenImageLayeredPipeline):
        @property
        def _execution_device(self):
            return torch.device('cuda:0')

        def _encode_vae_image(self, image, generator):
            latents = super()._encode_vae_image(image.to(self.vae.dtype), generator)
            if not torch.isfinite(latents).all().item():
                raise FloatingPointError('Non-finite VAE input latents')
            print('VAE input latents are finite', flush=True)
            return latents.to(compute_dtype)

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    base = args.root / 'models/qwen-base'
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(0.92)
    if torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError('This experiment is intended for the selected T4 GPU')
    caption_record = json.loads((args.embeddings / 'caption.json').read_text())
    cached = load_file(str(args.embeddings / 'embeddings.safetensors'))
    cached = {k: v.to('cuda', dtype=compute_dtype if 'mask' not in k else torch.int64)
              for k, v in cached.items()}
    # The pinned pipeline reads text_encoder.dtype even with precomputed embeds.
    # Preserve the real encoder type on meta, without allocating its weights.
    config = Qwen2_5_VLConfig.from_pretrained(base / 'text_encoder', local_files_only=True)
    with init_empty_weights():
        encoder = Qwen2_5_VLForConditionalGeneration(config)
    encoder.to(dtype=compute_dtype)
    transformer = QwenImageTransformer2DModel.from_single_file(
        str(args.root / 'models/qwen-gguf/qwen-image-layered-Q3_K_M.gguf'),
        config=str(base), subfolder='transformer', local_files_only=True,
        quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
        torch_dtype=compute_dtype)
    restored_parameters = []
    # Diffusers 0.37 leaves GGUF parameters outside Linear modules compressed.
    # Layered's additional timestep Embedding must hold decoded floating values.
    for module_name, module in transformer.named_modules():
        if isinstance(module, GGUFLinear):
            continue
        for parameter_name, parameter in list(module.named_parameters(recurse=False)):
            if isinstance(parameter, GGUFParameter):
                decoded = dequantize_gguf_tensor(parameter).to(dtype=compute_dtype)
                if decoded.numel() > 1_000_000:
                    raise ValueError('Unexpected large non-Linear GGUF parameter')
                module._parameters[parameter_name] = torch.nn.Parameter(decoded, requires_grad=False)
                restored_parameters.append(module_name + '.' + parameter_name)
    print(json.dumps({'restored_non_linear_parameters': restored_parameters}), flush=True)
    transformer.to('cuda')
    vae = AutoencoderKLQwenImage.from_pretrained(base / 'vae',
        torch_dtype=torch.float32, local_files_only=True).to('cuda')
    vae.enable_slicing()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base / 'scheduler', local_files_only=True)
    pipe = T4LayeredPipeline(scheduler=scheduler, vae=vae, text_encoder=encoder,
                             tokenizer=None, processor=None, transformer=transformer)
    loaded = time.monotonic()
    with Image.open(args.input) as original:
        source = ImageOps.exif_transpose(original).convert('RGBA')
    source.save(args.output / 'source.png')
    torch.cuda.reset_peak_memory_stats()

    def progress(pipeline, step, timestep, callback_kwargs):
        if not torch.isfinite(callback_kwargs['latents']).all().item():
            raise FloatingPointError('Non-finite diffusion latents in the T4 compute dtype')
        print(json.dumps({'step': step + 1, 'total': args.steps,
                          'elapsed_s': round(time.monotonic() - loaded, 2)}), flush=True)
        return callback_kwargs

    with torch.inference_mode():
        result = pipe(image=source, prompt=caption_record['caption'],
            prompt_embeds=cached['positive'], prompt_embeds_mask=cached['positive_mask'],
            negative_prompt_embeds=cached['negative'], negative_prompt_embeds_mask=cached['negative_mask'],
            true_cfg_scale=4.0, cfg_normalize=True, num_inference_steps=args.steps,
            layers=args.layers, resolution=args.resolution,
            generator=torch.Generator(device='cuda').manual_seed(42),
            callback_on_step_end=progress)
    inference_s = time.monotonic() - loaded
    layers = result.images[0]
    composite = Image.new('RGBA', layers[0].size, (0, 0, 0, 0))
    records = []
    for index, layer in enumerate(layers):
        layer = layer.convert('RGBA')
        filename = f'layer-{index:02d}.png'
        layer.save(args.output / filename)
        composite.alpha_composite(layer)
        alpha = np.asarray(layer.getchannel('A'))
        records.append({'file': filename, 'z': index,
                        'alpha_zero_fraction': float((alpha == 0).mean()),
                        'alpha_bbox': layer.getchannel('A').getbbox()})
    composite.save(args.output / 'composite.png')
    reference = source.resize(composite.size, Image.Resampling.LANCZOS)
    reference.save(args.output / 'source-at-output-size.png')
    delta = np.asarray(reference.convert('RGB'), dtype=np.float32) - np.asarray(composite.convert('RGB'), dtype=np.float32)
    report = {'model': 'Qwen-Image-Layered', 'quantization': 'Unsloth Q3_K_M',
        'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'model_revisions': {r['repo']: r['revision'] for r in
                            json.loads((args.root / 'asset-manifest.json').read_text())['files']},
        'loader_adaptations': {'decoded_non_linear_gguf_parameters': restored_parameters},
        'compute_dtype': args.compute_dtype, 'vae_dtype': 'float32', 'resolution_bucket': args.resolution,
        'steps': args.steps, 'requested_layers': args.layers, 'layers': records,
        'original_size': list(source.size), 'output_size': list(composite.size),
        'load_s': round(loaded - started, 3), 'inference_s': round(inference_s, 3),
        'gpu_peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
        'gpu_peak_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1),
        'process_peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'composite_mae_255': float(np.abs(delta).mean()),
        'native_text': False, 'review_status': 'REVIEW_REQUIRED',
        'caption_source': str(args.embeddings / 'caption.json')}
    write_json(args.output / 'manifest.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['encode', 'infer'])
    parser.add_argument('--root', type=Path, default=Path('/workspace'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--caption-file', type=Path)
    parser.add_argument('--embeddings', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--resolution', type=int, choices=[640, 1024], default=640)
    parser.add_argument('--compute-dtype', choices=['float16', 'float32'], default='float32')
    args = parser.parse_args()
    if not 1 <= args.steps <= 100 or not 1 <= args.layers <= 8:
        parser.error('Use 1..100 steps and 1..8 layers')
    encode(args) if args.stage == 'encode' else infer(args)


if __name__ == '__main__':
    main()
