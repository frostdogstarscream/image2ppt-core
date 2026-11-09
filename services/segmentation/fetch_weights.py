"""Fetch the two frozen ModelScope checkpoints without touching other models."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import shutil
import time
import threading
try:
    import requests
except ImportError:
    from pip._vendor import requests

MODELS = [
    ('facebook/sam3', 'sam3.pt', 3450062241, '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'),
    ('facebook/sam3.1', 'sam3.1_multiplex.pt', 3502755717, '0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6'),
]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def remaining(url, partial, size):
    """Resume a prefix and fetch disjoint bounded ranges with eight workers."""
    start = partial.stat().st_size if partial.exists() else 0
    chunk_size = 32 * 1024 * 1024
    ranges = [(a, min(a + chunk_size, size)) for a in range(start, size, chunk_size)]
    clients = threading.local()
    def part(bounds):
        a, b = bounds
        path = partial.with_name(partial.name + f'.{a}-{b}')
        if path.exists() and path.stat().st_size == b - a:
            return path
        for attempt in range(8):
            try:
                done = path.stat().st_size if path.exists() else 0
                if done > b - a:
                    raise RuntimeError('Oversized saved range')
                if done == b - a:
                    return path
                offset = a + done
                ranged_url = url + ('&' if '?' in url else '?') + f'range_start={offset}&range_end={b-1}'
                if not hasattr(clients, 'session'):
                    clients.session = requests.Session()
                with clients.session.get(ranged_url, headers={'Range': f'bytes={offset}-{b-1}'}, stream=True, timeout=(10,30)) as response:
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{b-1}/{size}':
                        raise RuntimeError('Server did not honor bounded checkpoint range')
                    with path.open('ab') as out:
                        for chunk in response.iter_content(64 * 1024):
                            out.write(chunk)
                if path.stat().st_size != b - a:
                    raise RuntimeError('Incomplete checkpoint range')
                print(json.dumps({'range_complete': [a, b], 'file': partial.name}), flush=True)
                return path
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(2)
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        pieces = list(pool.map(part, ranges))
    except Exception:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    with partial.open('ab') as out:
        for path in pieces:
            with path.open('rb') as stream:
                shutil.copyfileobj(stream, out, 1024 * 1024)


def fetch(root, only=None):
    root.mkdir(parents=True, exist_ok=True)
    receipts = []
    for repo, name, size, expected in MODELS:
        if only and repo != only:
            continue
        target = root / name
        url = f'https://modelscope.cn/models/{repo}/resolve/master/{name}'
        if not target.exists():
            partial = root / (name + '.partial')
            offset = partial.stat().st_size if partial.exists() else 0
            if offset < size:
                remaining(url, partial, size)
            if partial.stat().st_size != size or digest(partial) != expected:
                raise RuntimeError('Downloaded checkpoint integrity mismatch')
            partial.rename(target)
        if target.stat().st_size != size or digest(target) != expected:
            raise RuntimeError('Existing checkpoint integrity mismatch')
        receipts.append({'repository': repo, 'url': url, 'file': name, 'size': size, 'sha256': expected, 'verified': True})
        (root / 'download-receipt.json').write_text(json.dumps(receipts, indent=2), encoding='utf-8')
        print(json.dumps({'file': name, 'status': 'VERIFIED'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--only', choices=[m[0] for m in MODELS])
    args = parser.parse_args()
    fetch(args.directory.resolve(), args.only)
