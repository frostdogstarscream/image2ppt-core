"""Launch an isolated, version-pinned OpenCode service and product API."""
import argparse
import getpass
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
import threading

from ..parallel_config import configure
from .runner import OpenCodeClient, PINNED_VERSION


def executable():
    candidate = os.environ.get('IMAGE2PPT_OPENCODE_BIN') or shutil.which('opencode.exe')
    if candidate:
        return candidate
    wrapper = shutil.which('opencode')
    if wrapper:
        binary = Path(wrapper).parent/'node_modules/@opencode/cli/bin/opencode.exe'
        if binary.is_file():
            return str(binary)
        if Path(wrapper).suffix not in ('.cmd', '.ps1'):
            return wrapper
    raise RuntimeError('OpenCode '+PINNED_VERSION+' executable unavailable')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8771)
    parser.add_argument('--opencode-port', type=int, default=4096)
    parser.add_argument('--hostname', default='127.0.0.1')
    parser.add_argument('--root', required=True)
    parser.add_argument('--prompt-key', action='store_true')
    options = parser.parse_args()
    configure()
    if options.prompt_key:
        go=os.environ.get('IMAGE2PPT_AGENT_PROVIDER')=='opencode_go'
        os.environ['IMAGE2PPT_AGENT_API_KEY' if go else 'MOMA_API_KEY'] = getpass.getpass('Agent model key (hidden): ')
    from .provider import configure_model
    selected_provider,cfg=configure_model()
    if selected_provider=='moma':os.environ['MOMA_API_KEY']=cfg.api_key
    if len(os.environ.get('IMAGE2PPT_PRODUCT_TOKEN', '')) < 24:
        raise RuntimeError('set IMAGE2PPT_PRODUCT_TOKEN before launching the service')
    password = os.environ.get('IMAGE2PPT_OPENCODE_PASSWORD') or secrets.token_urlsafe(32)
    os.environ['IMAGE2PPT_OPENCODE_PASSWORD'] = password
    os.environ['OPENCODE_PASSWORD'] = password
    os.environ['OPENCODE_SERVER_PASSWORD'] = password
    os.environ['IMAGE2PPT_OPENCODE_URL'] = f'http://127.0.0.1:{options.opencode_port}'
    os.environ['IMAGE2PPT_DEFAULT_ORCHESTRATOR']='opencode'
    os.environ.setdefault('IMAGE2PPT_JOB_WORKERS','2')
    root = Path(options.root).resolve()
    os.environ['IMAGE2PPT_RUNTIME_JOBS_ROOT']=str(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'opencode.log').open('a', encoding='utf8') as log:
        process = subprocess.Popen([executable(), 'serve', '--hostname', '127.0.0.1', '--port', str(options.opencode_port)],
            cwd=root,stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf8', errors='replace',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        ownership=None
        if os.name=='nt':
            from .windows_host import contain_child
            ownership=contain_child(process)
        def drain():
            for line in process.stdout:
                for secret in (password,cfg.api_key,os.environ.get('IMAGE2PPT_PRODUCT_TOKEN','')):
                    if secret:line=line.replace(secret,'[REDACTED]')
                log.write(line);log.flush()
        reader=threading.Thread(target=drain,daemon=True)
        reader.start()
        try:
            client = OpenCodeClient()
            for _ in range(40):
                try:
                    client.verify()
                    break
                except Exception:
                    if process.poll() is not None:
                        raise RuntimeError('OpenCode exited before startup')
                    time.sleep(.25)
            else:
                raise RuntimeError('OpenCode did not become ready')
            import uvicorn
            from ..app import create_app
            uvicorn.run(create_app(root), host=options.hostname, port=options.port)
        finally:
            if ownership is not None:
                ownership.Close()
            else:
                process.terminate()
            process.wait(timeout=15)
            reader.join(2)


if __name__ == '__main__':
    main()
