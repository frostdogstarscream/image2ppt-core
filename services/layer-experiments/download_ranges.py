"""Resume public weight downloads using verified HTTP byte ranges."""
import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib import request


def fetch_chunk(task):
    record, url, start, end, chunk = task
    expected = end - start + 1
    if chunk.exists() and chunk.stat().st_size == expected:
        return
    temporary = chunk.with_suffix('.download')
    for attempt in range(3):
        try:
            req = request.Request(url, headers={'Range': f'bytes={start}-{end}'})
            with request.urlopen(req, timeout=45) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{record["size"]}':
                    raise ValueError('Server did not honor the requested range')
                with temporary.open('wb') as destination:
                    remaining = expected
                    while remaining:
                        block = response.read(min(1024 * 1024, remaining))
                        if not block:
                            raise EOFError('Incomplete range')
                        destination.write(block)
                        remaining -= len(block)
            temporary.replace(chunk)
            print(json.dumps({'chunk_complete': record['target'], 'start': start, 'bytes': expected}), flush=True)
            return
        except Exception as exc:
            print(json.dumps({'chunk_retry': record['target'], 'start': start,
                              'attempt': attempt + 1, 'error_type': type(exc).__name__}), flush=True)
            time.sleep(2)
    raise RuntimeError(f"Range download failed for {record['target']} at {start}")


def main():
    root = Path(sys.argv[1]).resolve()
    manifest = json.loads((root / 'asset-manifest.json').read_text())
    links = json.loads((root / 'download-url-overrides.json').read_text())
    tasks = []
    files = []
    chunk_size = 32 * 1024 * 1024
    for record in manifest['files']:
        if record['target'] not in links:
            continue
        target = root / record['target']
        if target.exists():
            continue
        partial = target.with_suffix(target.suffix + '.part')
        partial.parent.mkdir(parents=True, exist_ok=True)
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > record['size']:
            raise ValueError('Partial download exceeds publisher size')
        folder = target.with_suffix(target.suffix + '.chunks')
        folder.mkdir(exist_ok=True)
        file_tasks = []
        for start in range(offset, record['size'], chunk_size):
            end = min(start + chunk_size, record['size']) - 1
            chunk = folder / f'{start:012d}-{end:012d}.chunk'
            file_tasks.append((record, links[record['target']], start, end, chunk))
        files.append((record, target, partial, file_tasks))
        tasks.extend(file_tasks)
    # Interleave files to avoid starving the diffusion checkpoint behind encoders.
    tasks.sort(key=lambda task: (task[2] // chunk_size, task[0]['target']))
    print(json.dumps({'range_tasks': len(tasks), 'parallel_connections': 16}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for result in executor.map(fetch_chunk, tasks):
            pass
    for record, target, partial, file_tasks in files:
        with partial.open('ab') as destination:
            for task in file_tasks:
                with task[4].open('rb') as chunk:
                    for block in iter(lambda: chunk.read(8 * 1024 * 1024), b''):
                        destination.write(block)
        digest = hashlib.sha256()
        with partial.open('rb') as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if partial.stat().st_size != record['size'] or digest.hexdigest() != record['sha256']:
            raise ValueError(f"Publisher hash mismatch for {record['target']}")
        partial.replace(target)
        print(json.dumps({'verified': record['target'], 'bytes': record['size']}), flush=True)
    print('Range downloads completed and verified', flush=True)


if __name__ == '__main__':
    main()
