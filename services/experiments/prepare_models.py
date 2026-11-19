"""Download the pinned public artifacts into an isolated experiment directory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import uuid

MODELS = {
    'omni': ('microsoft/OmniParser-v2.0', 'f55d0750e5b94db2125ef0b45b0fa4a85ddc59b4', ['icon_detect/model.pt']),
    'doclayout': ('juliozhao/DocLayout-YOLO-DocStructBench', '8c3299a30b8ff29a1503c4431b035b93220f7b11', ['doclayout_yolo_docstructbench_imgsz1024.pt']),
    'vitmatte': ('hustvl/vitmatte-small-composition-1k', '6a58ad7646403c1df626fbd746900aec7361ea1d', ['model.safetensors', 'config.json', 'preprocessor_config.json']),
    'pp-l': ('PaddlePaddle/PP-DocLayout-L', 'ca760c3336d922ee63cdbff85003181222e3911c', ['inference.json', 'inference.pdiparams', 'inference.yml', 'config.json']),
}


def prepare(item, root, hub):
    name, (repo, revision, files) = item
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    metadata = json.load(urllib.request.urlopen(f'{hub}/api/models/{repo}?blobs=true', timeout=45))
    if metadata['sha'] != revision:
        raise ValueError(f'expected pinned revision for {repo}')
    entries = {entry['rfilename']: entry for entry in metadata['siblings']}
    hashes = {}
    for filename in files:
        target = folder / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        meta = entries[filename]
        def valid(path):
            if not path.exists() or path.stat().st_size != meta['size']:
                return False
            with path.open('rb') as f:
                sha = hashlib.file_digest(f, 'sha256').hexdigest()
            expected = meta.get('lfs', {}).get('sha256')
            if expected and expected != sha:
                return False
            hashes[filename] = sha
            return True
        if not valid(target):
            temporary = target.with_suffix('.' + uuid.uuid4().hex + '.part')
            try:
                with urllib.request.urlopen(f'{hub}/{repo}/resolve/{revision}/{filename}', timeout=120) as response, temporary.open('wb') as f:
                    shutil.copyfileobj(response, f)
                if not valid(temporary):
                    raise ValueError(f'weight verification failed: {name}/{filename}')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    (folder / 'weights.json').write_text(json.dumps({'repo': repo, 'revision': revision, 'files': hashes}, indent=2))
    print(name, revision, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--hub', default='https://huggingface.co')
    args = p.parse_args()
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda item: prepare(item, args.root, args.hub.rstrip('/')), MODELS.items()))

