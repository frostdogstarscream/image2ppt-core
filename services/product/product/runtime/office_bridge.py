"""Authenticated outbound Office worker with hash-verified portable packets."""
import argparse
import io
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import time
import zipfile

import httpx

from ..materialize import digest, write
from ..resource_lease import lease


def pack(folder, target):
    folder, target = Path(folder), Path(target)
    files = {str(p.relative_to(folder)).replace('\\', '/'): digest(p) for p in folder.rglob('*') if p.is_file()}
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('packet-manifest.json', json.dumps(files))
        for name in files:
            archive.write(folder/name, name)


def unpack(data, target):
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sum(p.file_size for p in archive.infolist()) > 512*1024*1024:
            raise ValueError('Office packet exceeds limit')
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('duplicate packet entries')
        manifest = json.loads(archive.read('packet-manifest.json'))
        if set(names) != set(manifest)|{'packet-manifest.json'}:
            raise ValueError('Office packet manifest mismatch')
        for name, sha in manifest.items():
            path = (target/name).resolve()
            if not path.is_relative_to(target) or ':' in name or '\\' in name:
                raise ValueError('unsafe Office packet path')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
            if digest(path) != sha:
                raise ValueError('Office packet hash mismatch')


def submit(job, pptx, irs, directory, timeout=1800):
    directory = Path(directory)
    request = directory/'worker-request'
    request.mkdir(parents=True)
    shutil.copyfile(pptx, request/'input.pptx')
    for i, ir in enumerate(irs):
        shutil.copytree(Path(ir).parent, request/f'page-{i}')
    write(request/'request.json', {'irs': [f'page-{i}/slideir.json' for i in range(len(irs))]})
    pack(request, directory/'input.zip')
    write(directory/'worker-state.json', {'status': 'QUEUED', 'created': time.time(),
        'input_sha256': digest(directory/'input.zip'), 'output_directory': str(directory)})
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if (Path(job)/'cancel.requested').exists():
            return {'status': 'UNKNOWN', 'failure': 'CANCELLED'}
        if (directory/'output.zip').is_file():
            unpack((directory/'output.zip').read_bytes(), directory/'worker-result')
            result = json.loads((directory/'worker-result/result.json').read_text('utf8'))
            if result.get('input_sha256') != digest(directory/'input.zip'):
                raise ValueError('Office result belongs to another input')
            result['office_directory'] = str(directory/'worker-result/check/office')
            result['files'] = {str(p): digest(p) for p in (directory/'worker-result').rglob('*') if p.is_file()}
            write(directory/'checks.json', result)
            return result
        time.sleep(.5)
    return {'status': 'UNKNOWN', 'failure': 'OFFICE_WORKER_TIMEOUT'}


class OfficeQueue:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def claim(self):
        with lease('runtime-office-queue', capacity=1):
            for file in sorted(self.root.glob('*/runtime/**/worker-state.json')):
                row = json.loads(file.read_text('utf8'))
                if row['status'] != 'QUEUED':
                    continue
                row.update(status='LEASED', token=secrets.token_urlsafe(32))
                write(file, row)
                return {'ticket': str(file.parent.relative_to(self.root)).replace('\\', '/'), 'token': row['token']}
        return None

    def directory(self, ticket, token):
        path = (self.root/ticket).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('invalid Office ticket')
        row = json.loads((path/'worker-state.json').read_text('utf8'))
        if row['status'] != 'LEASED' or not secrets.compare_digest(row['token'], token):
            raise ValueError('invalid Office lease')
        return path


def execute_ticket(client,ticket,run,binding):
    from .office import check_local
    from hashlib import sha256
    data=client.post('/api/v1/office/input',json=ticket)
    data.raise_for_status()
    unpack(data.content,run/'input')
    request=json.loads((run/'input/request.json').read_text('utf8'))
    irs=[(run/'input'/name).resolve() for name in request['irs']]
    if any(not p.is_relative_to(run/'input') for p in irs):
        raise ValueError('Office IR escaped packet')
    result=check_local(run/'input/input.pptx',irs,run/'output/check')
    result['worker_binding']=binding
    result['input_sha256']=sha256(data.content).hexdigest()
    write(run/'output/result.json',result)
    pack(run/'output',run/'output.zip')
    response=client.post('/api/v1/office/result',params=ticket,content=(run/'output.zip').read_bytes())
    response.raise_for_status()


def worker(url, token, root, once=False):
    from ..fonts import font_manifest
    from ..parallel_store import key
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    binding=key({'fonts':font_manifest([]),'worker_code':digest(__file__),
                 'office_code':digest(Path(__file__).with_name('office.py')),'process':secrets.token_hex(16)})
    with httpx.Client(base_url=url, headers={'Authorization': 'Bearer '+token,
            'X-Image2PPT-Worker-Binding':binding}, timeout=1800,trust_env=False) as client:
        while True:
            try:
                response = client.post('/api/v1/office/claim')
            except httpx.TransportError:
                if once:raise
                time.sleep(2)
                continue
            response.raise_for_status()
            ticket = response.json()
            if not ticket:
                if once:
                    return
                time.sleep(2)
                continue
            run = root/secrets.token_hex(16)
            try:
                execute_ticket(client,ticket,run,binding)
            except Exception as exc:
                # Retain the leased packet without redispatching an uncertain edit.
                write(run/'unknown.json',{'status':'UNKNOWN','ticket':ticket['ticket'],
                                          'error_type':type(exc).__name__,'automatic_retry':False})
                if once:raise
            if once:
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--once', action='store_true')
    options = parser.parse_args()
    worker(options.url, os.environ['IMAGE2PPT_OFFICE_WORKER_TOKEN'], options.root, options.once)
