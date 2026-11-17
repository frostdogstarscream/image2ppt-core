"""Layout region evidence (PP-DocLayout) for crop boundaries and intrinsic-text hints.

The detector runs out of process in a dedicated environment configured by
``layout.python`` in ``.runtime/parallel-services.json`` (or
``IMAGE2PPT_LAYOUT_PYTHON``). Results are cached by source and model-file
hashes. Regions are localization hints: they never authorize erasing pixels and
never change text wording.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from .materialize import digest, write
from .parallel_config import ROOT
from .parallel_store import key

WORKER = ROOT / 'services/layout/worker.py'
# Labels whose content is usually pictorial or has intrinsic text that should
# stay in one raster object when it cannot be reconstructed natively.
CROP_LABELS = ('image', 'header_image', 'footer_image', 'chart', 'seal', 'table')


def _config() -> dict:
    cfg = {}
    file = ROOT / '.runtime/parallel-services.json'
    if file.is_file():
        cfg = json.loads(file.read_text(encoding='utf-8')).get('layout', {}) or {}
    python = os.environ.get('IMAGE2PPT_LAYOUT_PYTHON') or cfg.get('python')
    return {'python': python, 'model': cfg.get('model', 'PP-DocLayoutV3'), 'device': cfg.get('device', 'cpu'),
            'cache_home': cfg.get('cache_home') or os.environ.get('PADDLE_PDX_CACHE_HOME'),
            'timeout': int(cfg.get('timeout_seconds', 300))}


def available() -> bool:
    cfg = _config()
    return bool(cfg['python']) and Path(cfg['python']).is_file() and WORKER.is_file()


def detect(snap: dict, cache: str | Path) -> dict:
    """Return LayoutRegions/1 for a source snapshot; raises when unavailable."""
    cfg = _config()
    if not available():
        raise RuntimeError('layout detector is not configured')
    identity = key({'source': snap['source_sha256'], 'model': cfg['model'], 'worker': digest(WORKER),
                    'schema': 'LayoutRegions/1'})
    file = Path(cache) / 'layout' / f'{identity}.json'
    if file.is_file():
        result = json.loads(file.read_text(encoding='utf-8'))
        if result.get('source_sha256') != snap['source_sha256']:
            raise ValueError('layout cache binding mismatch')
        return result
    file.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PADDLE_PDX_MODEL_SOURCE': os.environ.get('PADDLE_PDX_MODEL_SOURCE', 'modelscope')}
    if cfg['cache_home']:
        env['PADDLE_PDX_CACHE_HOME'] = cfg['cache_home']
    temporary = file.with_suffix('.tmp.json')
    run_kwargs = {'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0)} if os.name == 'nt' else {}
    subprocess.run([cfg['python'], str(WORKER), snap['path'], str(temporary), '--model', cfg['model'],
                    '--device', cfg['device']], check=True, timeout=cfg['timeout'], env=env,
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **run_kwargs)
    result = json.loads(temporary.read_text(encoding='utf-8'))
    if result.get('source_sha256') != snap['source_sha256']:
        raise ValueError('layout result source mismatch')
    result['identity'] = identity
    write(file, result)
    temporary.unlink(missing_ok=True)
    return result


def crop_seeds(layout: dict, graph, *, min_score: float = 0.5, max_committed_overlap: float = 0.3) -> list[dict]:
    """Pictorial layout regions still carried mostly by the page raster."""
    committed = []
    for record in graph.selected.values():
        box = record.get('restore_bbox') or record.get('bbox')
        if box and len(box) == 4 and record.get('kind') != 'surface' and not record.get('container_layer'):
            committed.append([float(v) for v in box])
    rows = []
    for region in layout.get('regions', []):
        if region['label'] not in CROP_LABELS or region['score'] < min_score:
            continue
        x, y, w, h = region['bbox']
        area = max(1.0, w * h)
        covered = 0.0
        for b in committed:
            ix = max(0.0, min(x + w, b[0] + b[2]) - max(x, b[0]))
            iy = max(0.0, min(y + h, b[1] + b[3]) - max(y, b[1]))
            covered += ix * iy
        if covered / area > max_committed_overlap:
            continue
        rows.append({'id': region['id'], 'kind': 'layout:' + region['label'], 'bbox': [x, y, w, h], 'text': ''})
    return rows
