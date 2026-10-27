"""SSH/compose driver for the development perception worker (local side)."""
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import tarfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
COMPOSE = Path(__file__).resolve().parents[5]/'deploy'/'layered-worker'/'compose.yaml'
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
SSH = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes']


def load_config(file):
    value = json.loads(Path(file).read_text(encoding='utf-8-sig'))
    if set(value) != {'ssh_host', 'remote_root'}:
        raise ValueError('layered worker config requires exactly ssh_host and remote_root')
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', value['ssh_host']) or value['ssh_host'].startswith('-'):
        raise ValueError('invalid SSH host')
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', value['remote_root']) or '..' in value['remote_root']:
        raise ValueError('absolute remote root required')
    return value


def source_archive():
    """Deterministic tarball of the vision package; its hash names the remote copy."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w', format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(PACKAGE.rglob('*')):
            if path.is_dir() or '__pycache__' in path.parts:
                continue
            info = tarfile.TarInfo('image2ppt_vision/'+path.relative_to(PACKAGE).as_posix())
            data = path.read_bytes()
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    data = buffer.getvalue()
    return data, hashlib.sha256(data).hexdigest()


class Worker:
    def __init__(self, config):
        self.host = config['ssh_host']
        self.root = config['remote_root']

    def ssh(self, command, *, data=None, timeout=1800):
        process = subprocess.run(['ssh', *SSH, self.host, command], input=data, capture_output=True,
                                 timeout=timeout, creationflags=FLAGS)
        if process.returncode:
            raise RuntimeError('remote command failed: '+process.stderr.decode('utf-8', 'replace')[-2000:])
        return process.stdout.decode('utf-8', 'replace')

    def scp(self, *args, timeout=1800):
        process = subprocess.run(['scp', *SSH, '-q', '-r', *map(str, args)], capture_output=True,
                                 timeout=timeout, creationflags=FLAGS)
        if process.returncode:
            raise RuntimeError('scp failed: '+process.stderr.decode('utf-8', 'replace')[-2000:])

    def sync_source(self):
        data, digest = source_archive()
        target = f'{self.root}/src/{digest[:16]}'
        self.ssh(f'test -d {shlex.quote(target)} || (mkdir -p {shlex.quote(target)}.tmp && '
                 f'tar -x -C {shlex.quote(target)}.tmp && mv {shlex.quote(target)}.tmp {shlex.quote(target)})',
                 data=data)
        compose = COMPOSE.read_bytes()
        self.ssh(f'cat > {shlex.quote(self.root)}/deploy/compose.yaml', data=compose)
        return digest, target

    def run(self, job, mode_args, *, source):
        env = (f'V3_ROOT={shlex.quote(self.root)} V3_SRC={shlex.quote(source)} '
               f'V3_JOB={shlex.quote(self.root)}/jobs/{shlex.quote(job)}')
        args = ' '.join(shlex.quote(a) for a in mode_args)
        return self.ssh(f'cd {shlex.quote(self.root)}/deploy && {env} docker compose -f compose.yaml '
                        f'run --rm -T worker {args}', timeout=3600)

    def upload(self, job, files, subdir='in'):
        remote = f'{self.root}/jobs/{job}/{subdir}'
        self.ssh(f'mkdir -p {shlex.quote(remote)}')
        self.scp(*files, f'{self.host}:{remote}/')
        return remote

    def download(self, job, subdir, local):
        Path(local).mkdir(parents=True, exist_ok=True)
        self.scp(f'{self.host}:{self.root}/jobs/{job}/{subdir}', local)
        return Path(local)/Path(subdir).name
