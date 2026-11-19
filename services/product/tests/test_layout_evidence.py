from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import httpx
import numpy as np
from PIL import Image

from product import layout_evidence
from product.parallel_branches import snapshot


def test_layout_crop_seeds_keep_pictorial_regions_still_in_plate():
    layout = {'regions': [
        {'id': 'l-001', 'label': 'image', 'score': 0.9, 'bbox': [0, 0, 100, 100]},
        {'id': 'l-002', 'label': 'text', 'score': 0.9, 'bbox': [200, 0, 50, 20]},
        {'id': 'l-003', 'label': 'chart', 'score': 0.3, 'bbox': [300, 0, 50, 50]},
        {'id': 'l-004', 'label': 'header_image', 'score': 0.9, 'bbox': [400, 0, 100, 40]},
    ]}
    graph = SimpleNamespace(selected={
        'a-1': {'kind': 'asset', 'bbox': [400, 0, 90, 40]},          # logo already a movable asset
        's-1': {'kind': 'surface', 'bbox': [0, 0, 500, 200]},        # backing surfaces never block
        't-1': {'kind': 'text', 'bbox': [10, 10, 30, 10]},           # small native label over photo
    })
    seeds = layout_evidence.crop_seeds(layout, graph)
    assert [s['id'] for s in seeds] == ['layout:l-001']
    assert seeds[0]['kind'] == 'layout:image'


def test_layout_detect_uses_http_worker_without_subprocess(tmp_path, monkeypatch):
    src = tmp_path / 'slide.png'
    Image.fromarray(np.full((60, 80, 3), 240, np.uint8)).save(src)
    snap = snapshot(src, tmp_path / 'snap')
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/health':
                body = json.dumps({'status': 'ok'}).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            calls.append(self.path)
            body = json.dumps({
                'schema': 'LayoutRegions/1',
                'model': 'PP-DocLayoutV3',
                'source_sha256': snap['source_sha256'],
                'model_files': {},
                'regions': [{'id': 'l-001', 'label': 'chart', 'score': 0.92, 'bbox': [4.0, 4.0, 40.0, 30.0]}],
            }).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        monkeypatch.setenv('IMAGE2PPT_LAYOUT_URL', f'http://{host}:{port}')
        monkeypatch.setenv('IMAGE2PPT_LAYOUT_PYTHON', '')
        monkeypatch.setattr(layout_evidence, '_detect_cli', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('CLI called')))
        assert layout_evidence.available() is True
        res = layout_evidence.detect(snap, tmp_path / 'cache')
        assert len(calls) == 1
        assert res['regions'][0]['id'] == 'l-001'
    finally:
        server.shutdown()
        server.server_close()


def test_layout_detect_falls_back_to_cli_on_transport_error(tmp_path, monkeypatch):
    src = tmp_path / 'slide.png'
    Image.fromarray(np.full((60, 80, 3), 240, np.uint8)).save(src)
    snap = snapshot(src, tmp_path / 'snap')
    fake_py = tmp_path / 'fake_python.exe'
    fake_py.write_text('', encoding='utf-8')

    monkeypatch.setenv('IMAGE2PPT_LAYOUT_URL', 'http://127.0.0.1:9')
    monkeypatch.setenv('IMAGE2PPT_LAYOUT_PYTHON', str(fake_py))

    def _boom(*args, **kwargs):
        raise httpx.ConnectError('connection refused')

    monkeypatch.setattr(layout_evidence, '_detect_http', _boom)
    monkeypatch.setattr(
        layout_evidence, '_detect_cli',
        lambda s, temp, cfg: {
            'schema': 'LayoutRegions/1',
            'model': cfg['model'],
            'source_sha256': s['source_sha256'],
            'model_files': {},
            'regions': [{'id': 'l-009', 'label': 'image', 'score': 0.88, 'bbox': [1.0, 2.0, 20.0, 20.0]}],
        },
    )
    res = layout_evidence.detect(snap, tmp_path / 'cache')
    assert res['regions'][0]['id'] == 'l-009'
