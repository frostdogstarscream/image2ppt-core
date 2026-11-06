"""Project-local Windows entry point; no activation or system policy changes."""
from pathlib import Path
import json
import os
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def main():
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help'):
        print('image2ppt-local.cmd status|doctor|demo|test|compile|validate|inspect|compose|rebuild|layerd|refine-typography [arguments]')
        print('rebuild passes arguments to ppt-rebuild/rebuild.py; layerd passes arguments to services/layerd/client.py.')
        return 0
    command, *rest = args
    node = ROOT / '.runtime/bundle/node.exe'
    core = ROOT / '.runtime/core-venv/Scripts/python.exe'
    helper = ROOT / '.runtime/bundle/python/python.exe'
    env = os.environ.copy()
    env.update(IMAGE2PPT_NODE=str(node), IMAGE2PPT_PYTHON=str(helper), PYTHONUTF8='1')
    if command == 'status':
        results = {}
        for service, port in [('layerd', 18840), ('lama', 18841)]:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=10) as response:
                    results[service] = json.load(response)
            except Exception as error:
                results[service] = {'error': str(error)}
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0 if results.get('layerd', {}).get('ready') and results.get('lama', {}).get('status') == 'alive' else 1
    if command == 'rebuild':
        if 'repair-text' in rest and not any(x == '--inpaint-url' or x.startswith('--inpaint-url=') for x in rest):
            rest += ['--inpaint-url', 'http://127.0.0.1:18841/inpaint']
        argv = [core, '-X', 'utf8', ROOT / 'ppt-rebuild/rebuild.py', *rest]
    elif command == 'refine-typography':
        argv = [core, '-X', 'utf8', ROOT / 'ppt-rebuild/src/typography_pipeline.py', *rest]
    elif command == 'layerd':
        argv = [core, '-X', 'utf8', ROOT / 'services/layerd/client.py',
                '--token-file', ROOT / '.runtime/layerd/client-token', *rest]
    elif command == 'test':
        argv = [node, '--test', ROOT / 'tests/ir.test.mjs', *rest]
    elif command == 'compose':
        argv = [node, ROOT / 'scripts/compose.mjs', *rest]
    elif command in ('doctor', 'demo', 'compile', 'validate', 'inspect', 'perturb'):
        argv = [node, ROOT / 'scripts/main.mjs', command, *rest]
    else:
        raise SystemExit('Unknown command: ' + command)
    return subprocess.run([str(a) for a in argv], cwd=ROOT, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
