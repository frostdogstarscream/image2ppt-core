"""Explicit SSH transport for the opt-in automatic validation workflow."""
import json
from pathlib import Path
import re
import shlex
import subprocess


class SSHRemote:
    def __init__(self, config):
        self.config = dict(config)
        if not re.fullmatch(r'[A-Za-z0-9_.@-]+', config['ssh_host']) or config['ssh_host'].startswith('-'):
            raise ValueError('invalid SSH host')
        for name in ('remote_worker', 'remote_config'):
            value = config[name]
            if not isinstance(value, str) or not value.startswith('/') or '\n' in value:
                raise ValueError('absolute remote path required')

    def request(self, payload, timeout=420):
        command = 'python3 '+shlex.quote(self.config['remote_worker'])+' '+shlex.quote(self.config['remote_config'])
        args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                '-o', 'StrictHostKeyChecking=yes', self.config['ssh_host'], command]
        flags = subprocess.CREATE_NO_WINDOW if __import__('os').name == 'nt' else 0
        try:
            process = subprocess.run(args, input=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                                     capture_output=True, timeout=timeout, creationflags=flags)
        except subprocess.TimeoutExpired:
            return {'ok': False, 'error': 'ssh_timeout', 'request_id': payload['request_id']}
        if process.returncode:
            return {'ok': False, 'error': 'ssh_failed', 'request_id': payload['request_id']}
        if len(process.stdout) > 8_000_000:
            return {'ok': False, 'error': 'remote_response_too_large', 'request_id': payload['request_id']}
        try:
            result = json.loads(process.stdout)
        except ValueError:
            return {'ok': False, 'error': 'remote_response_invalid', 'request_id': payload['request_id']}
        if not isinstance(result, dict) or result.get('request_id') != payload['request_id']:
            return {'ok': False, 'error': 'remote_response_unbound', 'request_id': payload['request_id']}
        return result


def load_config(file):
    value = json.loads(Path(file).read_text(encoding='utf-8-sig'))
    allowed = {'ssh_host', 'remote_worker', 'remote_config', 'models'}
    if set(value) != allowed or not isinstance(value['models'], list) or not 1 <= len(value['models']) <= 2:
        raise ValueError('invalid automatic workflow configuration')
    if any(not isinstance(m, str) or not re.fullmatch(r'[A-Za-z0-9_./-]{1,200}', m) for m in value['models']):
        raise ValueError('invalid configured model')
    SSHRemote(value)
    return value
