"""Explicit offline SAM3 worker. Imports do not load weights or start inference."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .plan_schema import validate_plan


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')


def prompt_box(box):
    a, b, c, d = box
    return [(a+c)/2000, (b+d)/2000, (c-a)/1000, (d-b)/1000]


def pixel_box(box, size):
    """Original model xyxy box, rounded outward to whole source pixels only."""
    w, h = size
    a, b, c, d = box
    x0, y0 = max(0, math.floor(a*w/1000)), max(0, math.floor(b*h/1000))
    x1, y1 = min(w, math.ceil(c*w/1000)), min(h, math.ceil(d*h/1000))
    if x1 <= x0 or y1 <= y0:
        raise ValueError('empty SAM3 crop box')
    return [x0, y0, x1, y1]


def restore_crop(masks, boxes, crop_box, size):
    """Embed crop-local predictions into the page with strictly zero outside."""
    w, h = size; x0, y0, x1, y1 = crop_box
    if masks.shape[1:] != (y1-y0, x1-x0):
        raise ValueError('SAM3 crop mask dimensions mismatch')
    restored = np.zeros((len(masks), h, w), dtype=bool)
    restored[:, y0:y1, x0:x1] = masks
    boxes = boxes.copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, x1-x0)+x0
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, y1-y0)+y0
    return restored, boxes


def infer(processor, image, rows, out, precision, box_policy='prompt'):
    import torch
    from contextlib import nullcontext
    context = torch.autocast('cuda', dtype=torch.float16) if precision == 'fp16' else nullcontext()
    records = []
    with torch.inference_mode(), context:
        state = processor.set_image(image) if box_policy == 'prompt' else None
        # Detect numerical failures before thresholding can hide NaN predictions.
        def finite(value):
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                raise FloatingPointError('nonfinite SAM3 output')
            if isinstance(value, dict):
                for v in value.values(): finite(v)
            elif isinstance(value, (tuple, list)):
                for v in value: finite(v)
        if state is not None:
            finite(state['backbone_out'])
        original = processor.model.forward_grounding
        def checked(*args, **kwargs):
            value = original(*args, **kwargs); finite(value); return value
        processor.model.forward_grounding = checked
        try:
            for row in rows:
                started = time.monotonic()
                crop_box = None
                box = prompt_box(row['bbox'])
                if box_policy == 'crop-hard':
                    crop_box = pixel_box(row['bbox'], image.size)
                    state = processor.set_image(image.crop(crop_box))
                    finite(state['backbone_out'])
                    box = [.5, .5, 1., 1.]
                processor.reset_all_prompts(state)
                processor.set_text_prompt(row['sam_prompt_en'], state)
                result = processor.add_geometric_prompt(box, True, state)
                finite(result)
                masks = result['masks'].detach().cpu().numpy()[:, 0].astype(bool)
                scores = result['scores'].float().detach().cpu().numpy()
                boxes = result['boxes'].float().detach().cpu().numpy()
                if crop_box is not None:
                    masks, boxes = restore_crop(masks, boxes, crop_box, image.size)
                file = out/(row['id']+'.npz')
                np.savez_compressed(file, masks=masks, scores=scores, boxes=boxes)
                records.append({'id': row['id'], 'sam_prompt_en': row['sam_prompt_en'],
                    'bbox': row['bbox'], 'segmentation_mode': row['segmentation_mode'],
                    'box_policy': box_policy, 'crop_bbox_px': crop_box,
                    'mask_file': file.name, 'sha256': sha(file), 'candidates': len(masks),
                    'seconds': round(time.monotonic()-started, 3)})
        finally:
            processor.model.forward_grounding = original
    return records


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', required=True); p.add_argument('--plans')
    p.add_argument('--output', required=True); p.add_argument('--smoke', action='store_true')
    p.add_argument('--box-policy', choices=('prompt', 'crop-hard'), default='prompt')
    args = p.parse_args(argv)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(os.environ.get('SAM3_CHECKPOINT', '/models/sam3.pt'))
    manifest = json.loads(checkpoint.with_name('manifest.json').read_text())
    if sha(checkpoint) != manifest['sha256']: raise ValueError('checkpoint hash mismatch')
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False,
                                   device='cuda', eval_mode=True, compile=False,
                                   enable_inst_interactivity=False)
    processor = Sam3Processor(model, device='cuda', confidence_threshold=.5)
    run = {'schema': 'SAM3Run/1', 'checkpoint': manifest, 'torch': torch.__version__,
           'gpu': torch.cuda.get_device_name(), 'resolution': 1008, 'box_policy': args.box_policy, 'pages': {},
           'human_acceptance': False}
    if args.smoke:
        source = out/'smoke.png'; image = Image.new('RGB', (768, 512), 'white')
        draw = ImageDraw.Draw(image); draw.ellipse((160, 110, 430, 380), fill='red'); image.save(source)
        sources = [source]
    else:
        sources = sorted(Path(args.inputs).glob('*.png'))
    for source in sources:
        target = out/source.stem; target.mkdir()
        if args.smoke:
            payload = {'smoke': True}; rows = [{'id': 'S1', 'bbox': [200, 200, 580, 780],
                'sam_prompt_en': 'red circle', 'segmentation_mode': 'object'}]
        else:
            planfile = Path(args.plans)/(source.stem+'.json')
            payload = json.loads(planfile.read_text()); validate_plan(payload['plan'])
            if payload['plan']['schema'] != 'SegmentationPlan/2' or payload['source_sha256'] != sha(source):
                raise ValueError('SAM3 requires a source-bound v2 plan')
            parents = {r.get('parent_id') for r in payload['plan']['regions']}
            rows = [r for r in payload['plan']['regions'] if r['id'] not in parents and r['action'] == 'keep_pixels']
        image = Image.open(source).convert('RGB'); retries = []
        for precision in ('fp16', 'fp32'):
            attempt = target/precision; attempt.mkdir()
            try:
                records = infer(processor, image, rows, attempt, precision, args.box_policy)
                break
            except FloatingPointError as error:
                retries.append({'precision': precision, 'error': str(error)})
                if precision == 'fp32': raise
                torch.cuda.empty_cache()
        value = {'schema': 'SAM3Page/1', 'source_sha256': sha(source), 'image_size': list(image.size),
                 'plan_payload_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                 'precision': precision, 'retries': retries, 'regions': records,
                 'box_policy': args.box_policy,
                 'checkpoint_sha256': manifest['sha256'], 'human_acceptance': False}
        write(target/'result.json', value); run['pages'][source.stem] = value
        print(source.stem, precision, 'objects', len(records), flush=True)
    torch.cuda.synchronize()
    run.update(seconds=round(time.monotonic()-started, 3), peak_memory_bytes=torch.cuda.max_memory_allocated())
    write(out/'run.json', run)
    print(json.dumps({'seconds': run['seconds'], 'peak_memory_bytes': run['peak_memory_bytes']}), flush=True)


if __name__ == '__main__':
    main()
