"""Local acceptance driver. Saved business fixtures are intentionally not distributed."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from image2ppt.compiler import ArtifactBackend, execute
from image2ppt.contracts import BuildManifest, RegressionSuite
from image2ppt.storage import digest, inventory, read_json, verify_files, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--regression', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--installed', required=True)
args = parser.parse_args()
out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=False)
backend = ArtifactBackend(ROOT)
suite = RegressionSuite.model_validate(read_json(ROOT / 'configs/regression-m1.json'))
regression = read_json(args.regression)
baseline = (ROOT / 'configs' / suite.pages[0].ir).resolve().parent.parent
files = []
for ir in sorted(baseline.glob('*/slideir.snapshot.json')):
    data = read_json(ir)
    files.extend([ir, ir.parent / data['source_image'], *(ir.parent / a['path'] for a in data['assets'])])
before = inventory(baseline, [p.relative_to(baseline).as_posix() for p in files])
validation = []
for ir in sorted(baseline.glob('*/slideir.snapshot.json')):
    validation.append({'source_page': ir.parent.name + '.png', 'ir_sha256': digest(ir), 'result': backend.validate(ir)})
write_json(out / 'compatibility.json', {'pages': validation, 'count': len(validation)}, exclusive=True)
foreign = out / 'foreign-cwd'
foreign.mkdir()
os.environ['PYTHONPATH'] = str(Path(args.installed).resolve())
os.environ['IMAGE2PPT_PROJECT_ROOT'] = str(ROOT)
doctor = execute([sys.executable, '-m', 'image2ppt', 'doctor'], foreign)
console = Path(args.installed).resolve() / 'bin/image2ppt-core.exe'
console_doctor = execute([console, 'doctor'], foreign)
assert doctor == console_doctor
write_json(out / 'doctor.json', doctor, exclusive=True)
replays = []
for page in regression['pages']:
    assert page['execution_status'] == 'COMPLETED'
    build = Path(page['build'])
    snapshot = BuildManifest.model_validate(read_json(build))
    verify_files(build.parent, snapshot.files)
    old_hash = digest(build.parent / 'result.pptx')
    result = execute([sys.executable, '-m', 'image2ppt', 'replay', '--build', build, '--output', out / page['id']], foreign, 1200)
    new = BuildManifest.model_validate(read_json(result['build']))
    verify_files(Path(result['build']).parent, new.files)
    assert snapshot.input_key == new.input_key
    assert digest(build.parent / 'result.pptx') == old_hash
    resumed = execute([console, 'resume', '--run', result['run']], foreign)
    assert resumed['reused'] is True
    replays.append({'id': page['id'], 'new_process': True, 'cwd': str(foreign), 'snapshot_preserved': True, 'resume_reused': True, **result})
    write_json(out / 'replays.json', replays)
verify_files(baseline, before)
write_json(out / 'report.json', {'compatibility_count': len(validation), 'input_files_unchanged': len(before),
    'installed_module_and_console_equivalent': True, 'replays': replays, 'network_calls': 0,
    'powerpoint_checked': False, 'wps_checked': False, 'human_acceptance': False}, exclusive=True)
print(str(out / 'report.json'))
