"""Freeze the actual selected renderer and local QA dependencies, read-only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PYTHON_MODULES = ('numpy', 'PIL', 'cv2', 'rapidocr', 'onnxruntime', 'pydantic',
                  'pydantic_core', 'yaml', 'fontTools', 'shapely', 'pyclipper',
                  'omegaconf', 'loguru', 'flatbuffers')
MAIN_REQUIRED = {'numpy', 'PIL', 'cv2', 'rapidocr', 'onnxruntime', 'pydantic'}
SKIP_DIRS = {'__pycache__', '.git', '.cache', '.pytest_cache'}


def _hash(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def _file(path, files):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('Selected runtime file missing: ' + str(path))
    if str(path) not in files:
        files[str(path)] = _hash(path)
    return str(path)


def _tree(root, files, *, suffixes=None, exclude=()):
    """Follow real package links once, including native files, without cycles."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Selected runtime directory missing: ' + str(root))
    pending = [root]; visited = set(); count = 0
    while pending:
        directory = pending.pop().resolve()
        if directory in visited:
            continue
        visited.add(directory)
        for entry in sorted(directory.iterdir(), key=lambda p: p.name):
            if entry.is_dir():
                if entry.name not in SKIP_DIRS and entry.name not in exclude:
                    pending.append(entry)
            elif entry.is_file() and entry.suffix.lower() not in {'.pyc', '.pyo'}:
                if suffixes is None or entry.suffix.lower() in suffixes:
                    _file(entry, files); count += 1
    return {'path': str(root), 'file_count': count}


def _package(name, start):
    """Locate installed package metadata using Node's ancestor node_modules rule."""
    if not name or '..' in name.split('/') or name.startswith(('/', '\\')):
        raise ValueError('Invalid runtime package name')
    start = Path(start).resolve()
    for folder in (start, *start.parents):
        candidate = folder / 'node_modules' / name
        if (candidate / 'package.json').is_file():
            return candidate.resolve()
    return None


def _node_packages(node_modules, files):
    modules = Path(node_modules).resolve()
    if not modules.is_dir():
        raise ValueError('Selected node_modules missing')
    roots = {}; queue = []
    for name in ('@oai/artifact-tool', 'skia-canvas', '@napi-rs/canvas'):
        folder = _package(name, modules.parent)
        roots[name] = str(folder) if folder else None
        if folder:
            queue.append(folder)
    if not roots['@oai/artifact-tool']:
        raise ValueError('Selected artifact-tool package missing')
    # The helpers load these font libraries optionally. Artifact Tool may use
    # its bundled renderer without either top-level package; its full tree and
    # declared dependency closure below still freeze the actual renderer bytes.
    packages = {}; missing_optional = {}
    while queue:
        folder = queue.pop().resolve()
        if str(folder) in packages:
            continue
        metadata = json.loads((folder / 'package.json').read_text(encoding='utf-8-sig'))
        packages[str(folder)] = {'name': metadata.get('name'), 'version': metadata.get('version'),
                                 **_tree(folder, files)}
        optional = metadata.get('optionalDependencies', {})
        for name in sorted(set(metadata.get('dependencies', {})) | set(optional)):
            resolved = _package(name, folder)
            if resolved:
                queue.append(resolved)
            elif name in optional:
                # Native platform packages for other OS/architectures are
                # legitimately absent; existing ones are still fully hashed.
                missing_optional.setdefault(str(folder), []).append(name)
            else:
                raise ValueError('Installed runtime dependency missing: ' + name + ' from ' + str(folder))
    return {'node_modules': str(modules), 'roots': roots,
            'packages': dict(sorted(packages.items())), 'missing_optional': dict(sorted(missing_optional.items()))}


def _python_probe(executable):
    # Probe resolution under the real interpreter without loading OCR or models.
    script = r'''
import importlib.util,json,sys,sysconfig
from pathlib import Path
names=json.loads(sys.argv[1]);modules={}
for name in names:
    spec=importlib.util.find_spec(name)
    if spec is None: modules[name]=None;continue
    modules[name]={'origin':spec.origin,'roots':list(spec.submodule_search_locations or [])}
print(json.dumps({'executable':sys.executable,'version':sys.version,'prefix':sys.prefix,
 'base_prefix':sys.base_prefix,'stdlib':sysconfig.get_path('stdlib'),'modules':modules}))
'''
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    result = subprocess.run([str(executable), '-B', '-c', script, json.dumps(PYTHON_MODULES)],
                            capture_output=True, text=True, encoding='utf-8', errors='strict',
                            timeout=45, creationflags=flags, check=True)
    return json.loads(result.stdout)


def _python_runtime(executable, files, *, required=()):
    executable = _file(executable, files)
    probe = _python_probe(executable)
    _file(probe['executable'], files)
    base = Path(probe['base_prefix']).resolve()
    binary_roots = sorted({base, Path(executable).parent})
    for root in binary_roots:
        for file in sorted(root.iterdir()):
            if file.is_file() and file.suffix.lower() in {'.dll', '.pyd', '.so', '.dylib', '.zip'}:
                _file(file, files)
        if (root / 'DLLs').is_dir():
            _tree(root / 'DLLs', files)
    stdlib = Path(probe['stdlib']).resolve()
    if stdlib.is_dir():
        _tree(stdlib, files, suffixes={'.py', '.pyd', '.dll', '.so', '.zip'},
              exclude={'site-packages', 'dist-packages', 'test', 'tests', 'idlelib', 'tkinter', 'turtledemo'})
    packages = {}
    for name, module in sorted(probe['modules'].items()):
        if module is None:
            if name in required:
                raise ValueError('Required QA Python module missing: ' + name)
            packages[name] = None
            continue
        roots = []
        for root in module.get('roots', []):
            package = Path(root).resolve()
            roots.append(_tree(package, files))
            # NumPy/Shapely wheels can put native dependencies beside the
            # Python package rather than underneath it.
            for sibling in sorted(package.parent.glob(package.name.lower() + '*.libs')):
                if sibling.is_dir():
                    roots.append(_tree(sibling, files))
        if module.get('origin') and module['origin'] not in {'built-in', 'frozen'}:
            _file(module['origin'], files)
        packages[name] = {'origin': module.get('origin'), 'roots': roots}
    return {**{key: probe[key] for key in ('executable', 'version', 'prefix', 'base_prefix', 'stdlib')},
            'packages': packages}


def freeze_runtime(backend):
    """Deterministic actual runtime identity; no fallback to guessed local files."""
    root = Path(backend.root).resolve()
    doctor = backend.invoke('core_bridge.mjs', 'doctor')
    for key in ('node_modules', 'compiler_python', 'presentations_skill'):
        if not doctor.get(key):
            raise ValueError('Runtime doctor omitted selected ' + key)
    files = {}
    node = _file(backend.node, files)
    _file(root / 'ppt-rebuild/compiler_bridge.mjs', files)
    config = root / 'runtime.local.json'
    config_ref = {'path': str(config), 'present': config.is_file(),
                  'sha256': _hash(config) if config.is_file() else None}
    if config.is_file():
        _file(config, files)
    modules = _node_packages(doctor['node_modules'], files)
    helpers = Path(doctor['presentations_skill']).resolve() / 'container_tools'
    for name in ('artifact_tool_utils.mjs', 'runtime_helpers.mjs',
                 'inspect_presentation_package_integrity.py', 'inspect_presentation_layout_geometry.py'):
        _file(helpers / name, files)
    helper_tree = _tree(helpers, files, suffixes={'.py', '.mjs'})
    main = _python_runtime(sys.executable, files, required=MAIN_REQUIRED)
    compiler_python = Path(doctor['compiler_python']).resolve()
    helper_python = main if compiler_python == Path(sys.executable).resolve() else _python_runtime(compiler_python, files)
    from ocr_engine import MODELS
    models = {name: _file(Path(MODELS) / name, files) for name in ('det.onnx', 'cls.onnx', 'rec.onnx')}
    result = {'schema': 'R7RuntimeFreeze/1', 'node': node,
              'engine': doctor.get('engine'), 'artifact_version': doctor.get('artifact_version'),
              'node_version': doctor.get('node_version'), 'runtime_config': config_ref,
              'node_packages': modules, 'helpers': helper_tree, 'python_main': main,
              'python_compiler': helper_python, 'ocr_models': models, 'files': dict(sorted(files.items()))}
    result['sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False,
                                               separators=(',', ':')).encode('utf-8')).hexdigest()
    return result
