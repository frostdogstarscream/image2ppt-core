"""Offline PP-DocLayout worker (runs in the dedicated layout environment).

Usage:
  CLI:  python worker.py <image.png> <output.json> [--model PP-DocLayoutV3] [--device cpu]
  HTTP: python worker.py --serve [--host 127.0.0.1] [--port 18849] [--model PP-DocLayoutV3] [--device cpu]

Writes LayoutRegions/1 with xywh source-pixel boxes. Model weights come from the
PaddleX cache (PADDLE_PDX_CACHE_HOME); the default model source is ModelScope.
Region labels are localization hints only and never grant erase authority.
"""
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading

os.environ.setdefault('PADDLE_PDX_MODEL_SOURCE', 'modelscope')
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def model_file_hashes(model_name):
    cache = Path(os.environ.get('PADDLE_PDX_CACHE_HOME', Path.home() / '.paddlex'))
    model_dir = cache / 'official_models' / model_name
    files = sorted(p for p in model_dir.glob('*') if p.is_file()) if model_dir.is_dir() else []
    return {p.name: sha(p) for p in files}


def predict_regions(model, image_path, model_name, model_files=None):
    result = list(model.predict(str(image_path), batch_size=1, layout_nms=True))[0]
    data = result.json.get('res', result.json)
    regions = []
    for index, box in enumerate(data.get('boxes', []), 1):
        x0, y0, x1, y1 = [float(v) for v in box['coordinate']]
        row = {'id': f'l-{index:03d}', 'label': str(box['label']), 'score': round(float(box['score']), 4),
               'bbox': [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]}
        if box.get('polygon_points') is not None:
            row['polygon'] = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in box['polygon_points']]
        regions.append(row)
    return {
        'schema': 'LayoutRegions/1',
        'model': model_name,
        'source_sha256': sha(image_path),
        'model_files': dict(model_files) if model_files is not None else model_file_hashes(model_name),
        'regions': regions,
    }


def serve(model_name, device, host, port):
    from paddleocr import LayoutDetection
    model = LayoutDetection(model_name=model_name, device=device)
    model_files = model_file_hashes(model_name)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status, obj):
            body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip('/') == '/health':
                self._send_json(200, {'status': 'ok', 'model': model_name, 'device': device, 'schema': 'LayoutRegions/1'})
                return
            self._send_json(404, {'error': 'not found'})

        def do_POST(self):
            if self.path.rstrip('/') != '/detect':
                self._send_json(404, {'error': 'not found'})
                return
            length = int(self.headers.get('Content-Length', '0') or '0')
            raw = self.rfile.read(length) if length > 0 else b''
            temp_path = None
            try:
                req = json.loads(raw.decode('utf-8')) if raw else {}
                image_path = req.get('image_path')
                if not image_path and req.get('image_b64'):
                    suffix = req.get('suffix') or '.png'
                    fd, temp_name = tempfile.mkstemp(suffix=suffix)
                    with os.fdopen(fd, 'wb') as tf:
                        tf.write(base64.b64decode(req['image_b64']))
                    temp_path = Path(temp_name)
                    image_path = str(temp_path)
                if not image_path or not Path(image_path).is_file():
                    self._send_json(400, {'error': f'missing or nonexistent image_path: {image_path}'})
                    return
                with lock:
                    payload = predict_regions(model, image_path, model_name, model_files)
                self._send_json(200, payload)
            except Exception as exc:
                self._send_json(500, {'error': f'{type(exc).__name__}: {exc}'})
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?')
    parser.add_argument('output', nargs='?')
    parser.add_argument('--model', default='PP-DocLayoutV3')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18849)
    args = parser.parse_args()
    if args.serve:
        serve(args.model, args.device, args.host, args.port)
        return
    if not args.image or not args.output:
        parser.error('image and output are required unless --serve is set')
    from paddleocr import LayoutDetection
    model = LayoutDetection(model_name=args.model, device=args.device)
    payload = predict_regions(model, args.image, args.model)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
