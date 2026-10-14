"""Allowlisted source export; excludes business fixtures and proprietary runtime."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
target = Path(args.output).resolve()
target.parent.mkdir(parents=True, exist_ok=True)
paths = [ROOT / name for name in ('README.md','AGENTS.md','.gitignore','image2ppt.ps1','image2ppt.sh','pyproject.toml','requirements-core.lock')]
for directory, extensions in {
    'src/image2ppt': {'.py'}, 'scripts': {'.py','.mjs'}, 'tests': {'.py','.mjs'},
    'skills/image2ppt': {'.md','.yaml'}, 'skills/image2ppt-claude': {'.md'},
    '.agents/skills/image2ppt': {'.md','.yaml'},
    'configs': {'.json'}, 'docs': {'.md'},
}.items():
    paths.extend(p for p in (ROOT / directory).rglob('*') if p.is_file() and p.suffix in extensions and '__pycache__' not in p.parts)
paths.extend(ROOT / 'examples/synthetic' / name for name in ('slideir.json','source/page.png','assets/visual.png'))
files = sorted(set(paths))
manifest = {'format_version': '1', 'scope': 'project source and labelled synthetic fixture only',
            'excluded': ['proprietary runtime','business source images and saved business outputs','local runtime configuration','run history'],
            'files': {p.relative_to(ROOT).as_posix(): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size} for p in files}}
with ZipFile(target, 'x', ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, 'image2ppt-core/' + p.relative_to(ROOT).as_posix())
    z.writestr('image2ppt-core/package-manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
with ZipFile(target) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(files) + 1
    for name, record in manifest['files'].items():
        assert hashlib.sha256(z.read('image2ppt-core/' + name)).hexdigest() == record['sha256']
summary = {'file': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size,
           'source_files': len(files), 'crc_and_manifest_verified': True}
target.with_suffix('.manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
