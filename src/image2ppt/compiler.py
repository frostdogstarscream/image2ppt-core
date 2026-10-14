"""Process boundary around the existing fixed Node compiler."""
import importlib.metadata
import os
import signal
import subprocess
import sys
from pathlib import Path
from .storage import digest, key, read_json


def execute(args, cwd, timeout=300):
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    process = subprocess.Popen([str(a) for a in args], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding='utf-8', errors='replace', shell=False,
                               creationflags=flags, start_new_session=os.name != 'nt')
    job = None
    try:
        if os.name == 'nt':
            from .processes import WindowsJob
            job = WindowsJob(process)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if job:
                job.close()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=15)
            raise TimeoutError(f"owned subprocess tree timed out after {timeout}s")
    finally:
        if job:
            job.close()
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=15)
    if process.returncode:
        raise RuntimeError(f"subprocess exit {process.returncode}: {(stderr or stdout)[-3000:]}")
    import json
    try:
        result = json.loads(stdout)
    except ValueError as error:
        raise ValueError("subprocess returned malformed JSON") from error
    if not isinstance(result, dict):
        raise ValueError("subprocess returned non-object JSON")
    return result


class ArtifactBackend:
    def __init__(self, root=None, timeout=300):
        self.root = Path(root or os.environ.get('IMAGE2PPT_PROJECT_ROOT') or Path(__file__).resolve().parents[2]).resolve()
        if not (self.root / 'scripts/main.mjs').is_file():
            raise ValueError('set IMAGE2PPT_PROJECT_ROOT to the complete source/toolchain directory')
        primary = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies'
        self.node = Path(os.environ.get('IMAGE2PPT_NODE', str(primary / 'node/bin/node.exe')))
        self.timeout = timeout

    def invoke(self, script, *args):
        return execute([self.node, self.root / 'scripts' / script, *args], self.root, self.timeout)

    def validate(self, ir):
        result = self.invoke('main.mjs', 'validate', '--input', str(Path(ir).resolve()))
        if result.get('valid') is not True:
            raise ValueError('validator did not confirm validity')
        return result

    def compile(self, ir):
        result = self.invoke('core_bridge.mjs', 'compile', str(Path(ir).resolve()))
        for field in ('output_dir', 'pptx', 'preview', 'quality'):
            if field not in result or not Path(result[field]).exists():
                raise ValueError('compiler missing output: ' + field)
        return result

    def render(self, pptx, output):
        result = self.invoke('core_bridge.mjs', 'render', str(Path(pptx).resolve()), str(Path(output).resolve()))
        if not result.get('pages') or result.get('pptx_sha256') != digest(pptx):
            raise ValueError('renderer did not bind pages to the requested PPTX')
        for page in result['pages']:
            from PIL import Image
            with Image.open(page['path']) as image:
                image.verify()
        return result

    def doctor(self):
        result = self.invoke('core_bridge.mjs', 'doctor')
        result.update(python=sys.version, python_executable=sys.executable, node_executable=str(self.node),
                      python_packages={name: importlib.metadata.version(name) for name in ('pydantic', 'Pillow')})
        from . import __version__
        package = Path(__file__).resolve().parent
        python_files = {p.name: {'path':str(p), 'sha256':digest(p)} for p in sorted(package.glob('*.py'))}
        for name, module in sorted(sys.modules.copy().items()):
            if name == 'image2ppt' or name.startswith('image2ppt.'):
                file = getattr(module, '__file__', None)
                if file and Path(file).suffix == '.py':
                    p = Path(file).resolve()
                    python_files['loaded:'+name] = {'path':str(p), 'sha256':digest(p)}
        result['python_core'] = {'package_directory':str(package), 'version':__version__, 'files':python_files}
        result['toolchain_root'] = str(self.root)
        result['python_core_fingerprint'] = key(result['python_core'])
        # Include local helpers imported by compiler extensions, not only a fixed list.
        scripts = [p for p in (self.root/'scripts').rglob('*') if p.suffix in ('.py','.mjs') and '__pycache__' not in p.parts]
        result['js_compiler_fingerprint'] = key({p.relative_to(self.root).as_posix():digest(p) for p in scripts})
        result['renderer_fingerprint'] = key({k:result.get(k) for k in ('engine','artifact_version','node_modules')})
        result['external_helpers_fingerprint'] = key(result.get('helpers', {}))
        result['implementation_sha256'] = key({k:result[k] for k in (
            'python_core_fingerprint','js_compiler_fingerprint','renderer_fingerprint','external_helpers_fingerprint')})
        fonts = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
        result['font_files'] = {p.name: digest(p) for pattern in ('msyh*.ttc','arial*.ttf') for p in fonts.glob(pattern)} if fonts.exists() else {}
        result['effective_font_resolution'] = 'NOT_VERIFIED; inventory does not prove renderer font selection'
        return result
