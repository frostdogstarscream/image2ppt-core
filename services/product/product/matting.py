"""RMBG-2.0 matte candidates for whole graphic objects.

The matting model proposes a soft foreground matte for one object crop. The
foreground colours are then unmixed against an independently fitted backing,
and the candidate enters the same acceptance path as LayerD/SAM candidates
(protected-pixel exclusion, backing error, source recomposition and owner
proof in ``parallel_rebuild.build``). A matte is never ownership evidence.

Configuration: ``matting`` in ``.runtime/parallel-services.json``
(``model`` PyTorch directory or ``.onnx`` path, optional ``sha256``,
``python``, ``python_path``, ``empty_cache``, ``local_gpu_lease``) or
``IMAGE2PPT_MATTING_MODEL`` / ``IMAGE2PPT_MATTING_PYTHON``.
RMBG-2.0 weights are CC BY-NC 4.0 (non-commercial use only).
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

from .materialize import digest, write
from .object_extract import rectangle
from .parallel_config import ROOT
from .resource_lease import lease

WORKER = ROOT / 'services/matting/worker.py'
_JOB_TIMEOUT_RECORD = ContextVar('rmbg_job_timeout_record', default=None)
PAD = 8
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class MattingTimeout(ValueError):
    """Optional inference did not finish. Other asset proposals are still valid."""


@contextmanager
def job_matting(runtime_directory):
    token = _JOB_TIMEOUT_RECORD.set(Path(runtime_directory) / 'rmbg-timeouts.json')
    try:
        yield
    finally:
        _JOB_TIMEOUT_RECORD.reset(token)


def _config() -> dict:
    cfg = {}
    file = ROOT / '.runtime/parallel-services.json'
    if file.is_file():
        cfg = json.loads(file.read_text(encoding='utf-8')).get('matting', {}) or {}
    return {
        'python': os.environ.get('IMAGE2PPT_MATTING_PYTHON') or cfg.get('python'),
        'python_path': os.environ.get('IMAGE2PPT_MATTING_PYTHONPATH') or cfg.get('python_path'),
        'model': os.environ.get('IMAGE2PPT_MATTING_MODEL') or cfg.get('model'),
        'sha256': cfg.get('sha256'),
        'empty_cache': bool(cfg.get('empty_cache', False)),
        'local_gpu_lease': cfg.get('local_gpu_lease'),
        'timeout': int(cfg.get('timeout_seconds', 120)),
    }


def _model_weights_file(model_path: Path) -> Path | None:
    if model_path.is_dir():
        weights = model_path / 'model.safetensors'
        required = (weights, model_path / 'birefnet.py', model_path / 'BiRefNet_config.py')
        return weights if all(p.is_file() for p in required) else None
    return model_path if model_path.is_file() else None


def _can_run_in_process(is_pt_dir: bool, python_path: str | None) -> bool:
    if is_pt_dir:
        return importlib.util.find_spec('torch') is not None
    if python_path and (Path(python_path) / 'onnxruntime').is_dir():
        return True
    return importlib.util.find_spec('onnxruntime') is not None


def available() -> bool:
    cfg = _config()
    if not cfg['model'] or not WORKER.is_file():
        return False
    model_path = Path(cfg['model'])
    if _model_weights_file(model_path) is None:
        return False
    return _can_run_in_process(model_path.is_dir(), cfg['python_path']) or (
        bool(cfg['python']) and Path(cfg['python']).is_file()
    )


@lru_cache(maxsize=4)
def _file_sha(path: str, mtime: float, size: int) -> str:
    return digest(path)


def binding() -> dict | None:
    if not available():
        return None
    cfg = _config()
    weights = _model_weights_file(Path(cfg['model']))
    assert weights is not None
    stat = weights.stat()
    sha = _file_sha(str(weights), stat.st_mtime, stat.st_size)
    if cfg['sha256'] and sha != cfg['sha256']:
        raise ValueError('matting model hash mismatch')
    return {'method': 'rmbg2-matte/1', 'model_sha256': sha, 'worker_sha256': digest(WORKER)}


@lru_cache(maxsize=2)
def _ort_session(model_path: str, python_path: str):
    if python_path and Path(python_path).is_dir():
        if python_path in sys.path:
            sys.path.remove(python_path)
        sys.path.insert(0, python_path)
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    avail = ort.get_available_providers()
    providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in avail] or ['CPUExecutionProvider']
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


@lru_cache(maxsize=1)
def _torch_model(model_dir: str):
    spec = importlib.util.spec_from_file_location('_matting_worker', str(WORKER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_pytorch_birefnet(Path(model_dir))


def _gpu_lease(cfg: dict, *, timeout=3600):
    lg = cfg.get('local_gpu_lease')
    if isinstance(lg, dict) and lg.get('root') and lg.get('name'):
        return lease(lg['name'], root=lg['root'], capacity=1, timeout=timeout)
    return nullcontext()


def predict_alpha(crop: np.ndarray, *, size: int = 1024) -> np.ndarray:
    cfg = _config()
    model_str = cfg['model']
    if not model_str:
        raise ValueError('matting model missing')
    model_path = Path(model_str)
    if _model_weights_file(model_path) is None:
        raise ValueError('matting model missing')
    record = _JOB_TIMEOUT_RECORD.get()
    if record is not None:
        model = binding()
        history = json.loads(record.read_text('utf8')) if record.exists() else {}
        timeouts = history.get('timeouts', {}) if history.get('model') == model else {}
        crop_id = hashlib.sha256(str((crop.shape, size)).encode() + crop.tobytes()).hexdigest()
        if crop_id in timeouts or len(timeouts) >= 2:
            raise MattingTimeout('RMBG_TIMEOUT: job already recorded a timeout for this crop or two distinct crops')
        try:
            alpha = _run_subprocess(cfg, model_str, crop, size=size, bounded=True)
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            timeouts[crop_id] = {'crop_size': list(crop.shape[:2][::-1]), 'input_size': size,
                                 'reason': 'inference' if isinstance(exc, subprocess.TimeoutExpired) else 'gpu_queue'}
            write(record, {'model': model, 'timeout_seconds': cfg['timeout'], 'timeouts': timeouts})
            raise MattingTimeout('RMBG_TIMEOUT: optional matte unavailable for this crop') from exc
        if alpha.shape != crop.shape[:2]:
            raise ValueError('matte size mismatch')
        return alpha
    image = Image.fromarray(crop).convert('RGB')
    w, h = image.size
    x = np.asarray(image.resize((size, size), Image.BILINEAR), np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]

    if model_path.is_dir():
        try:
            import torch
            with _gpu_lease(cfg):
                net, device = _torch_model(str(model_path))
                tensor = torch.from_numpy(x).to(
                    device=device, dtype=torch.float16 if device == 'cuda' else torch.float32
                )
                with torch.inference_mode():
                    pred = net(tensor)[-1].sigmoid()[0, 0].float().cpu().numpy()
                if device == 'cuda' and cfg['empty_cache']:
                    torch.cuda.empty_cache()
            alpha = np.asarray(
                Image.fromarray(np.clip(pred * 255, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                np.float32,
            ) / 255.0
        except ImportError:
            alpha = _run_subprocess(cfg, model_str, crop, size=size)
    else:
        try:
            session = _ort_session(model_str, str(cfg['python_path'] or ''))
            meta = session.get_inputs()[0]
            feed_x = x.astype(np.float16) if 'float16' in meta.type else x
            outputs = session.run(None, {meta.name: feed_x})
            pred = np.asarray(outputs[-1], np.float32).reshape(size, size)
            if pred.min() < 0 or pred.max() > 1:
                pred = 1.0 / (1.0 + np.exp(-pred))
            alpha = np.asarray(
                Image.fromarray(np.clip(pred * 255, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                np.float32,
            ) / 255.0
        except ImportError:
            alpha = _run_subprocess(cfg, model_str, crop, size=size)
    if alpha.shape != crop.shape[:2]:
        raise ValueError('matte size mismatch')
    return alpha


def _run_subprocess(cfg: dict, model_str: str, crop: np.ndarray, *, size=1024, bounded=False) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / 'crop.png', Path(tmp) / 'alpha.png'
        Image.fromarray(crop).save(src)
        env = os.environ.copy()
        if cfg['python_path']:
            env['PYTHONPATH'] = os.pathsep.join(filter(None, [cfg['python_path'], env.get('PYTHONPATH', '')]))
        started = time.monotonic()
        with _gpu_lease(cfg, timeout=cfg['timeout'] if bounded else 3600):
            run_kwargs = {'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0)} if os.name == 'nt' else {}
            remaining = cfg['timeout'] - (time.monotonic() - started) if bounded else cfg['timeout']
            if remaining <= 0:
                raise TimeoutError('matting GPU queue wait expired')
            subprocess.run([cfg['python'] or sys.executable, str(WORKER), model_str, str(src), str(out), '--size', str(size)], check=True,
                           timeout=remaining, env=env, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **run_kwargs)
        return np.asarray(Image.open(out).convert('L'), np.float32) / 255.0


def unmix(crop: np.ndarray, backing: np.ndarray, alpha: np.ndarray, *, floor: float = 0.04) -> np.ndarray:
    """Foreground colours F from I = a*F + (1-a)*B; alpha below floor becomes transparent."""
    a = np.where(alpha < floor, 0.0, alpha)[..., None]
    fg = np.where(a > 0, (crop.astype(np.float32) - (1 - a) * backing.astype(np.float32)) / np.maximum(a, 1e-3), 0)
    rgba = np.zeros((*crop.shape[:2], 4), np.uint8)
    rgba[..., :3] = np.clip(np.rint(fg), 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(np.rint(a[..., 0] * 255), 0, 255).astype(np.uint8)
    return rgba


def matte_candidate(obj, source: np.ndarray, protected: np.ndarray, surface=None, *, predictor=None):
    """Return (full-canvas RGBA, metadata) or raise ValueError."""
    from .object_unmix import backing_from_evidence
    size = (source.shape[1], source.shape[0])
    region = rectangle(obj.bbox, size, pad=PAD)
    ys, xs = np.nonzero(region)
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
    excluded = protected.copy()
    for e in obj.exclusions:
        excluded |= rectangle(e.bbox, size)
    backing, error, evidence = backing_from_evidence(source, [x, y, w, h], excluded, surface)
    if not np.isfinite(error) or error > 6:
        raise ValueError('matte backing uncertain')
    crop = source[y:y + h, x:x + w]
    alpha = (predictor or predict_alpha)(crop)
    alpha[excluded[y:y + h, x:x + w]] = 0.0
    rgba = unmix(crop, backing, alpha)
    visible = rgba[..., 3] > 0
    if not visible.any():
        raise ValueError('empty matte')
    if visible[:, [0, -1]].any() or visible[[0, -1]].any():
        raise ValueError('matte touches crop boundary')
    full = np.zeros((*source.shape[:2], 4), np.uint8)
    full[y:y + h, x:x + w] = rgba
    for part in obj.parts:
        if not (rectangle(part.bbox, size) & (full[..., 3] > 0)).any():
            raise ValueError('matte missing required part')
    return full, {'method': 'rmbg2', 'scope': [x, y, w, h], 'ring_error': float(error),
                  'background_evidence': evidence, 'independent_review_required': True}
