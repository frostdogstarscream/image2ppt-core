"""Offline PP-DocLayout worker (runs in the dedicated layout environment).

Usage: python worker.py <image.png> <output.json> [--model PP-DocLayoutV3] [--device cpu]

Writes LayoutRegions/1 with xywh source-pixel boxes. Model weights come from the
PaddleX cache (PADDLE_PDX_CACHE_HOME); the default model source is ModelScope.
Region labels are localization hints only and never grant erase authority.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('PADDLE_PDX_MODEL_SOURCE', 'modelscope')
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('output')
    parser.add_argument('--model', default='PP-DocLayoutV3')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    from paddleocr import LayoutDetection
    model = LayoutDetection(model_name=args.model, device=args.device)
    result = list(model.predict(args.image, batch_size=1, layout_nms=True))[0]
    data = result.json.get('res', result.json)
    regions = []
    for index, box in enumerate(data.get('boxes', []), 1):
        x0, y0, x1, y1 = [float(v) for v in box['coordinate']]
        row = {'id': f'l-{index:03d}', 'label': str(box['label']), 'score': round(float(box['score']), 4),
               'bbox': [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]}
        if box.get('polygon_points') is not None:
            row['polygon'] = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in box['polygon_points']]
        regions.append(row)
    cache = Path(os.environ.get('PADDLE_PDX_CACHE_HOME', Path.home() / '.paddlex'))
    model_dir = cache / 'official_models' / args.model
    files = sorted(p for p in model_dir.glob('*') if p.is_file()) if model_dir.is_dir() else []
    payload = {'schema': 'LayoutRegions/1', 'model': args.model, 'source_sha256': sha(args.image),
               'model_files': {p.name: sha(p) for p in files}, 'regions': regions}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
