"""Wait for verified downloads, then run the CPU encoder and T4 smoke job."""
import json
import subprocess
import time
from pathlib import Path

from download_assets import download


ROOT = Path('/data/image2ppt-layer-experiments/20261105-r1')
IMAGE = 'image2ppt-qwen-layered:20261105-r1'
GPU = 'GPU-44e3bad2-be08-97b6-5769-f50c01fb40e4'


def run_container(name, arguments, gpu=False):
    command = ['docker', 'run', '--name', name, '--network', 'none',
               '--cpus', '8', '--memory', '72g', '--memory-swap', '72g',
               '--pids-limit', '256', '--cap-drop', 'ALL', '--read-only',
               '--security-opt', 'no-new-privileges',
               '--tmpfs', '/tmp:rw,nosuid,nodev,size=1g,mode=1777',
               '-e', 'HF_HOME=/tmp/hf', '-v', f'{ROOT}:/workspace:ro',
               '-v', f'{ROOT}/outputs:/workspace/outputs:rw',
               '-v', '/data/image2ppt-vision/data/pilot/deck-src:/input:ro']
    if gpu:
        command.extend(['--gpus', f'device={GPU}'])
    command.extend([IMAGE, '/workspace/scripts/run_qwen.py', *arguments])
    with (ROOT / 'evidence' / f'{name}.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps({'container': name, 'exit_code': result.returncode}), flush=True)
    if result.returncode:
        raise RuntimeError(f'Container {name} failed, see its evidence log')


def main():
    records = json.loads((ROOT / 'asset-manifest.json').read_text())['files']
    deadline = time.monotonic() + 5400
    while not all((ROOT / r['target']).is_file() for r in records):
        if time.monotonic() > deadline:
            raise TimeoutError('Public model downloads have not completed')
        time.sleep(10)
    for record in records:
        if record['size'] < 20_000_000:
            download(record, ROOT, {})
    (ROOT / 'evidence/assets-ready.json').write_text(json.dumps({'files': len(records),
        'large_asset_verification': 'download_ranges.py verifies size and SHA256 before rename',
        'small_asset_verification': 'Publisher Git blob SHA1 or LFS SHA256 checked'}))
    run_container('i2p-qwen-encode-r1', ['encode', '--output', '/workspace/outputs/embeddings-page03-r1',
        '--caption-file', '/workspace/scripts/caption-page03.txt'])
    used = int(subprocess.check_output(['nvidia-smi', '-i', GPU,
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if used > 256:
        raise RuntimeError('Selected GPU has acquired another workload')
    source = next(Path('/data/image2ppt-vision/data/pilot/deck-src').glob('03_*.png'))
    run_container('i2p-qwen-smoke-r1', ['infer', '--output', '/workspace/outputs/page03-smoke-r1',
        '--embeddings', '/workspace/outputs/embeddings-page03-r1',
        '--input', '/input/' + source.name, '--steps', '2', '--layers', '3'], gpu=True)
    print('Qwen T4 smoke job completed', flush=True)


if __name__ == '__main__':
    main()
