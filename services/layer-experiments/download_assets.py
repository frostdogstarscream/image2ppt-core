"""Download public pinned assets, checking publisher hashes before activation."""
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def download(record, root, overrides):
    target = root / record['target']
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    if not target.exists():
        command = ['curl', '-fL', '--connect-timeout', '15', '--max-time', '3600',
                   '--retry', '3', '--retry-delay', '3', '--speed-time', '120',
                   '--speed-limit', '1024', '--continue-at', '-', '--output', str(partial),
                   '--silent', '--show-error', overrides.get(record['target'], record['url'])]
        result = subprocess.run(command)
        if result.returncode:
            raise RuntimeError(f"Download failed for {record['target']}, exit {result.returncode}")
        candidate = partial
    else:
        candidate = target
    size = candidate.stat().st_size
    if size != record['size']:
        raise ValueError(f"Incorrect size for {target.name}: {size}")
    if record.get('sha256'):
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1()
        digest.update(f'blob {size}\0'.encode())
    with candidate.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    expected = record.get('sha256') or record['git_blob']
    if digest.hexdigest() != expected:
        raise ValueError(f"Hash mismatch for {target.name}")
    if candidate == partial:
        partial.replace(target)
    print(json.dumps({'verified': record['target'], 'bytes': size}), flush=True)


def main():
    root = Path(sys.argv[1]).resolve()
    records = json.loads((root / 'asset-manifest.json').read_text())['files']
    override_path = root / 'download-url-overrides.json'
    overrides = json.loads(override_path.read_text()) if override_path.exists() else {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download, record, root, overrides) for record in records]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    (root / 'evidence/assets-verified.json').write_text(json.dumps({'verified': True, 'files': len(records)}))
    print('All public assets verified', flush=True)


if __name__ == '__main__':
    main()
