"""Optional OS-isolated scripts in an explicitly provisioned local Docker image."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from ..materialize import digest, write


def execute(state, directory, script, inputs):
    image = os.environ.get('IMAGE2PPT_SANDBOX_IMAGE', '')
    docker = shutil.which('docker')
    if not docker or not re.fullmatch(r'[\w./:-]+@sha256:[a-f0-9]{64}', image):
        return {'status': 'UNKNOWN', 'failure': 'SANDBOX_UNAVAILABLE',
                'reason': 'a local digest-pinned Docker image with Pillow/numpy/OpenCV is required'}
    if len(script.encode('utf8')) > 64000 or len(inputs) > 20:
        raise ValueError('script or input count exceeds sandbox limit')
    incoming, outgoing = directory/'input', directory/'output'
    incoming.mkdir(); outgoing.mkdir()
    references = []
    for index, identifier in enumerate(inputs):
        file, ref = state.evidence(identifier)
        target = incoming/(str(index)+file.suffix)
        shutil.copyfile(file, target)
        references.append({**ref, 'sandbox_path': '/input/'+target.name})
    (incoming/'script.py').write_text(script, encoding='utf8')
    shutil.copyfile(Path(__file__).with_name('sandbox_entry.py'),incoming/'sandbox_entry.py')
    write(incoming/'manifest.json', references)
    cidfile = directory/'container-id'
    command = [docker, 'run', '--rm', '--pull', 'never', '--cidfile', str(cidfile),
               '--network', 'none', '--read-only', '--cap-drop', 'ALL',
               '--security-opt', 'no-new-privileges', '--pids-limit', '32', '--memory', '512m',
               '--cpus', '1', '--user', '65534:65534', '--tmpfs', '/tmp:rw,noexec,nosuid,size=32m',
               '--env','OPENBLAS_NUM_THREADS=1','--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
               '--mount', f'type=bind,src={incoming.resolve()},dst=/input,readonly',
               '--mount', f'type=bind,src={outgoing.resolve()},dst=/output',
               '--workdir', '/output', image, 'python', '-I', '/input/sandbox_entry.py']
    started = time.monotonic()
    with (directory/'execution.log').open('wb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            while process.poll() is None:
                size = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
                if time.monotonic()-started > 60 or size > 32*1024*1024:
                    raise TimeoutError('sandbox resource budget exceeded')
                time.sleep(.2)
        finally:
            if process.poll() is None:
                try:
                    if cidfile.exists():
                        identifier = cidfile.read_text().strip()
                        if re.fullmatch(r'[a-f0-9]{64}', identifier):
                            subprocess.run([docker, 'rm', '-f', identifier], capture_output=True, timeout=15)
                finally:
                    process.kill(); process.wait(timeout=15)
    files = []
    for path in outgoing.rglob('*'):
        if path.is_symlink() or not path.resolve().is_relative_to(outgoing.resolve()):
            raise ValueError('sandbox output escaped task')
        if path.is_file():
            files.append(state.register(path, 'unverified_script_output'))
    receipt = {'status': 'COMPLETED' if process.returncode == 0 else 'UNKNOWN' if process.returncode==78 else 'FAILED',
               'exit_code': process.returncode, 'seconds': time.monotonic()-started,
               'image': image, 'script_sha256': digest(incoming/'script.py'),
               'isolation_entry_sha256':digest(incoming/'sandbox_entry.py'),
               'inputs': references, 'outputs': files, 'acceptance': 'UNVERIFIED'}
    write(directory/'sandbox-receipt.json', receipt)
    return receipt
