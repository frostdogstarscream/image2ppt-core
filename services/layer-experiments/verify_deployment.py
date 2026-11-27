"""Record actual trial results and compare the original container baseline."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


ROOT = Path('/data/image2ppt-layer-experiments/20261105-r1')


def main():
    evidence = ROOT / 'evidence'
    baseline = json.loads((evidence / 'before-containers.json').read_text())
    ids = subprocess.check_output(['docker', 'ps', '-aq'], text=True).split()
    containers = json.loads(subprocess.check_output(['docker', 'inspect', *ids], text=True))
    current = {c['Name'].lstrip('/'): c for c in containers}
    missing = []
    changes = []
    for before in baseline:
        container = current.get(before['name'])
        if container is None:
            missing.append(before['name'])
            continue
        after = {'name': before['name'], 'id': container['Id'],
                 'status': container['State']['Status'],
                 'started_at': container['State']['StartedAt'],
                 'restart_count': container['RestartCount']}
        differences = {key: {'before': before[key], 'after': value}
                       for key, value in after.items() if before[key] != value}
        if differences:
            changes.append({'name': before['name'], 'changes': differences})
    qwen = json.loads((ROOT / 'outputs/page03-10step-r1/manifest.json').read_text())
    encoder = json.loads((ROOT / 'outputs/embeddings-page03-r1/caption.json').read_text())
    trials = [{'name': name, 'exit_code': c['State']['ExitCode'],
               'status': c['State']['Status'], 'oom_killed': c['State']['OOMKilled'],
               'restart_count': c['RestartCount'], 'image': c['Config']['Image'],
               'image_id': c['Image']}
              for name, c in current.items() if name.startswith('i2p-qwen-')]
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT / 'scripts').glob('*.py')}
    with urlopen('http://127.0.0.1:18840/ready', timeout=5) as response:
        layerd = json.load(response)
    report = {'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'runtime_kind': 'Docker CLI trial', 'qwen': qwen,
              'encoder': {key: encoder[key] for key in
                          ['elapsed_s', 'encoder_dtype', 'encoder_device', 'process_peak_rss_mib']},
              'other_models': json.loads((evidence / 'asset-preflight.json').read_text()),
              'trial_containers': trials, 'original_container_count': len(baseline),
              'missing_original_containers': missing, 'original_container_changes': changes,
              'layerd_readiness': layerd, 'runner_sha256': source_hashes,
              'human_acceptance': False}
    (evidence / 'deployment-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    subprocess.run(['docker', 'cp',
        'i2p-qwen-encode-r1:/opt/layer-experiments-environment.lock',
        str(evidence / 'environment.lock')], check=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
