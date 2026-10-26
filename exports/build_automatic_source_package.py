"""Reproducible project-owned automatic-validation source bundle (no runtime)."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def main():
    from build_source_package import OWN_SOURCES
    # The automatic service/client entry is the deliverable, not an agent skill.
    sources = [s for s in OWN_SOURCES if not s.startswith(('skills/', '.agents/', '.claude/'))]
    # Protocol tests reference skill sources, so retain those as test fixtures;
    # there are no installed skill entries and no skill packaging step here.
    sources += [s for s in OWN_SOURCES if s.startswith('skills/')]
    sources += ['exports/build_automatic_source_package.py', 'exports/build_source_package.py']
    if (ROOT/'docs/automatic-workflow-verification.md').is_file():
        sources.append('docs/automatic-workflow-verification.md')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:6]
    destination = ROOT/'exports'/('image2ppt-automatic-validation-'+stamp+'.zip')
    entries = {}
    for name in sorted(set(sources)):
        path = (ROOT/name).resolve(strict=True)
        path.relative_to(ROOT)
        entries[name] = path.read_bytes()
    # Mirror fixtures are needed by the existing protocol test; these are source
    # bytes only, not an installed personal skill or an invitation to use one.
    for name, data in list(entries.items()):
        if name.startswith('skills/image2ppt/'):
            entries['.agents/skills/image2ppt/'+name[len('skills/image2ppt/'):]] = data
    entries['README.md'] = (ROOT/'docs/automatic-workflow.md').read_bytes()
    entries['runtime.local.example.json'] = json.dumps({
        'node_modules': 'C:/path/to/existing/node_modules',
        'python': 'C:/path/to/python.exe',
        'presentations_skill': 'C:/path/to/existing/presentations/skills/presentations'}, indent=2).encode()
    manifest = {'package_kind': 'automatic-validation-source',
                'execution': 'server_models_and_analysis_plus_local_compiler',
                'docker_included': False, 'engine_included': False, 'credentials_included': False,
                'business_images_included': False,
                'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                          for name, data in entries.items()}}
    entries['manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    with ZipFile(destination, 'x', ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr('image2ppt/'+name, data)
    with ZipFile(destination) as archive:
        if archive.testzip(): raise ValueError('package CRC failure')
        for name, record in manifest['files'].items():
            if hashlib.sha256(archive.read('image2ppt/'+name)).hexdigest() != record['sha256']:
                raise ValueError('package manifest mismatch')
    report = {'zip': str(destination), 'files': len(entries), 'bytes': destination.stat().st_size,
              'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'manifest_verified': True}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
