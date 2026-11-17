"""Offline RMBG-2.0 (BiRefNet architecture) matting worker.

Usage: python worker.py <model_dir_or_onnx> <crop.png> <alpha.png> [--size 1024]

Writes an 8-bit alpha matte with the crop's dimensions. The matte is a
foreground hypothesis only; product gates decide whether it can be used.
RMBG-2.0 weights are CC BY-NC 4.0 (non-commercial use only).
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import warnings

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _preprocess(crop_img: Image.Image, size: int) -> np.ndarray:
    x = np.asarray(crop_img.resize((size, size), Image.BILINEAR), np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]


def load_pytorch_birefnet(model_dir: Path):
    import sys
    import types
    import safetensors.torch as st
    import torch

    if 'kornia.filters' not in sys.modules:
        try:
            __import__('kornia.filters')
        except ModuleNotFoundError:
            kornia_mod = sys.modules.get('kornia') or types.ModuleType('kornia')
            filters_mod = types.ModuleType('kornia.filters')
            filters_mod.laplacian = lambda *args, **kwargs: None
            kornia_mod.filters = filters_mod
            sys.modules['kornia'] = kornia_mod
            sys.modules['kornia.filters'] = filters_mod

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pkg_name = '_rmbg2_local'
        if pkg_name not in sys.modules:
            pkg = type(sys)(pkg_name)
            pkg.__path__ = [str(model_dir)]
            sys.modules[pkg_name] = pkg
        if f'{pkg_name}.BiRefNet_config' not in sys.modules:
            sc = importlib.util.spec_from_file_location(f'{pkg_name}.BiRefNet_config', model_dir / 'BiRefNet_config.py')
            mc = importlib.util.module_from_spec(sc)
            sys.modules[f'{pkg_name}.BiRefNet_config'] = mc
            sc.loader.exec_module(mc)
        else:
            mc = sys.modules[f'{pkg_name}.BiRefNet_config']
        if f'{pkg_name}.birefnet' not in sys.modules:
            sb = importlib.util.spec_from_file_location(f'{pkg_name}.birefnet', model_dir / 'birefnet.py')
            mb = importlib.util.module_from_spec(sb)
            sys.modules[f'{pkg_name}.birefnet'] = mb
            sb.loader.exec_module(mb)
        else:
            mb = sys.modules[f'{pkg_name}.birefnet']

    net = mb.BiRefNet(config=mc.BiRefNetConfig(bb_pretrained=False))
    sd = st.load_file(str(model_dir / 'model.safetensors'))
    net.load_state_dict(sd, strict=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device).eval()
    if device == 'cuda':
        net = net.half()
    return net, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('crop')
    parser.add_argument('alpha')
    parser.add_argument('--size', type=int, default=1024)
    args = parser.parse_args()
    model_path = Path(args.model)
    image = Image.open(args.crop).convert('RGB')
    w, h = image.size
    x = _preprocess(image, args.size)
    if model_path.is_dir():
        import torch
        net, device = load_pytorch_birefnet(model_path)
        tensor = torch.from_numpy(x).to(device=device, dtype=torch.float16 if device == 'cuda' else torch.float32)
        with torch.inference_mode():
            pred = net(tensor)[-1].sigmoid()[0, 0].float().cpu().numpy()
    else:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        avail = ort.get_available_providers()
        providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in avail] or ['CPUExecutionProvider']
        session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        meta = session.get_inputs()[0]
        if 'float16' in meta.type:
            x = x.astype(np.float16)
        outputs = session.run(None, {meta.name: x})
        pred = np.asarray(outputs[-1], np.float32).reshape(args.size, args.size)
        if pred.min() < 0 or pred.max() > 1:
            pred = 1.0 / (1.0 + np.exp(-pred))
    alpha = Image.fromarray(np.clip(pred * 255, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    alpha.save(args.alpha)


if __name__ == '__main__':
    main()
