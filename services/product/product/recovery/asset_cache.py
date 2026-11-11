"""Content fingerprints shared by single-object and batch asset checks."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from ..materialize import digest
from ..parallel_store import key


ROOT = Path(__file__).resolve().parents[4]


def file_hashes(directory):
    directory = Path(directory).resolve()
    return {str(p.resolve()): digest(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def code_fingerprint(root=ROOT, *, compiler=False):
    root = Path(root)
    # Include the whole local Python closure, including imported reconstruction
    # helpers. Added or removed files change the mapping as well as edits.
    directories = [root/'services/product/product', root/'ppt-rebuild']
    suffixes = {'.py', '.json', '.lock', '.txt'}
    if compiler:
        directories.extend([root/'scripts', root/'src'])
        suffixes.update({'.mjs', '.js'})
    files = {p for directory in directories for p in directory.rglob('*')
             if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts
             and 'node_modules' not in p.parts}
    files.update((root/'services/product').glob('*.py'))
    shared_lease = root/'services/resource_lease.py'
    if shared_lease.is_file():
        files.add(shared_lease)
    if compiler:
        files.update((root/'services/product').glob('*.mjs'))
        files.update(p for p in [root/'runtime.local.json', root/'requirements-core.lock',
                                root/'pyproject.toml'] if p.is_file())
    return {str(p.resolve()): digest(p) for p in sorted(files)}


def python_runtime():
    packages = {}
    for name in ('numpy', 'opencv-python', 'Pillow', 'pydantic', 'scipy', 'pywin32'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'executable': digest(sys.executable), 'packages': packages}


def l1_binding(graph, obj, source, layers, baseline, strategy, model):
    return {'source_sha256': digest(source), 'scene_sha256': key(graph.model_dump()),
            'object': obj.model_dump(), 'baseline': file_hashes(baseline),
            'layers': file_hashes(layers), 'code': code_fingerprint(),
            'runtime': python_runtime(), 'strategy': strategy, 'sam_model': model}


def compiler_runtime_fingerprint():
    from ..pipeline import compiler_runtime
    _, node = compiler_runtime()
    script = """
        import {runtime} from './scripts/runtime.mjs';
        import {createRequire} from 'node:module';
        import fs from 'node:fs';
        import path from 'node:path';
        const r = await runtime();
        const packages = new Set();
        const missing = new Set();
        function visit(name, require) {
          let file;
          try { file = require.resolve(name + '/package.json'); }
          catch {
            try {
              let folder = path.dirname(require.resolve(name));
              while (!fs.existsSync(path.join(folder, 'package.json'))) {
                const parent = path.dirname(folder);
                if (parent === folder) throw new Error('package root missing');
                folder = parent;
              }
              file = path.join(folder, 'package.json');
            } catch { missing.add(name); return; }
          }
          const folder = fs.realpathSync(path.dirname(file));
          if (packages.has(folder)) return;
          packages.add(folder);
          const metadata = JSON.parse(fs.readFileSync(file, 'utf8'));
          const childRequire = createRequire(file);
          for (const child of Object.keys({...metadata.dependencies, ...metadata.optionalDependencies})) {
            visit(child, childRequire);
          }
        }
        visit('@oai/artifact-tool', r.require);
        console.log(JSON.stringify({modules:r.node_modules, skill:r.skill,
          python:r.python, version:r.artifact_version,
          packages:[...packages].sort(), missing:[...missing].sort()}));
    """
    reply = subprocess.run([str(node), '--input-type=module', '-e', script], cwd=ROOT,
                           check=True, capture_output=True, text=True, timeout=60,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    runtime = json.loads(reply.stdout.strip().splitlines()[-1])
    # Bind the actual loaded engine and finalization tools, not just their version.
    runtime['engine_files'] = file_hashes(Path(runtime['modules'])/'@oai/artifact-tool')
    runtime['dependency_files'] = {name: sha for package in runtime['packages']
                                   for name, sha in file_hashes(package).items()}
    runtime['finalizer_files'] = file_hashes(Path(runtime['skill'])/'container_tools')
    runtime['node_sha256'] = digest(node)
    runtime['python_sha256'] = digest(runtime['python'])
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r'SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\POWERPNT.EXE') as handle:
            executable = winreg.QueryValue(handle, None).strip('"')
        runtime['office'] = {'path': executable, 'sha256': digest(executable)}
    return runtime


def l2_binding(result, graph, source, layers, baseline, fonts):
    candidate = file_hashes(result['candidate_directory'])
    if {Path(p).name: sha for p, sha in candidate.items()} != result['candidate_hashes']:
        raise ValueError('candidate changed before Office cache lookup')
    return {'candidate': candidate, 'source_sha256': digest(source),
            'scene_sha256': key(graph.model_dump()), 'baseline': file_hashes(baseline),
            'layers': file_hashes(layers), 'font_configuration': fonts,
            'font_files': {str(Path(r['file']).resolve()): digest(r['file']) for r in fonts['fonts']},
            'code': code_fingerprint(compiler=True), 'python': python_runtime(),
            'compiler_runtime': compiler_runtime_fingerprint()}
