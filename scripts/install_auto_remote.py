"""Install the opt-in SSH worker into a NEW private server directory."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ssh-host', required=True); p.add_argument('--remote-root', required=True)
    p.add_argument('--server-config', required=True); p.add_argument('--client-output', required=True)
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', args.ssh_host) or args.ssh_host.startswith('-'):
        p.error('invalid SSH host')
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', args.remote_root) or '..' in args.remote_root.split('/'):
        p.error('absolute, simple remote path required')
    target = Path(args.client_output)
    if target.exists():
        p.error('client output already exists; choose a new file')
    config = json.loads(Path(args.server_config).read_text(encoding='utf-8-sig'))
    if any(key.lower() in ('api_key', 'token', 'secret', 'password') for key in config):
        p.error('server config must reference existing credentials, never contain secrets')
    config['data_root'] = args.remote_root+'/requests'
    root = Path(__file__).resolve().parents[1]
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', args.ssh_host]
    subprocess.run([*ssh, 'mkdir -m 700 '+shlex.quote(args.remote_root)], check=True)
    flags = ['-q', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes']
    subprocess.run(['scp', *flags, str(root/'services/vision/tools/auto_remote.py'),
                    args.ssh_host+':'+args.remote_root+'/auto_remote.py'], check=True)
    with tempfile.TemporaryDirectory(prefix='image2ppt-remote-') as temporary:
        file = Path(temporary)/'config.json'; file.write_text(json.dumps(config, indent=2), encoding='utf-8')
        subprocess.run(['scp', *flags, str(file), args.ssh_host+':'+args.remote_root+'/config.json'], check=True)
    subprocess.run([*ssh, 'chmod 600 '+shlex.quote(args.remote_root+'/config.json')+' '+shlex.quote(args.remote_root+'/auto_remote.py')], check=True)
    client = {'ssh_host': args.ssh_host, 'remote_worker': args.remote_root+'/auto_remote.py',
              'remote_config': args.remote_root+'/config.json', 'models': config['allowed_models']}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as f:
        json.dump(client, f, ensure_ascii=False, indent=2)
    print(json.dumps({'client_config': str(target.resolve()), 'remote_root': args.remote_root}))


if __name__ == '__main__': main()
