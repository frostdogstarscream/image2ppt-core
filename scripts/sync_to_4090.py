"""Safe hash-verified sync of product/runtime/config/skill files to 4090 `F:/i2oc/app`.

Preserves live 8772/14100/GPU services, `.runtime` configs, credentials, and `jobs/`.
Backs up existing remote files under `F:/i2oc/backup-native-first/` before writing.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

SYNC_FILES = [
    'configs/fidelity_tolerance.json',
    'ppt-rebuild/src/reconstruction.py',
    'scripts/capabilities.mjs',
    'scripts/compile.mjs',
    'scripts/compose.mjs',
    'scripts/ir.mjs',
    'scripts/pptx_xml.py',
    'scripts/scorecard.py',
    'scripts/slide.mjs',
    'services/experiments/worker.py',
    'services/product/compose_objects.mjs',
    'services/product/OPENCODE_RUNTIME.md',
    'services/product/PARALLEL_R10.md',
    'services/product/product/app.py',
    'services/product/product/contracts.py',
    'services/product/product/experiment.py',
    'services/product/product/experiment_assets.py',
    'services/product/product/experiment_svg.py',
    'services/product/product/go_provider.py',
    'services/product/product/object_unmix.py',
    'services/product/product/parallel_contracts.py',
    'services/product/product/parallel_edit_probe.py',
    'services/product/product/parallel_fusion.py',
    'services/product/product/parallel_office.py',
    'services/product/product/parallel_rebuild.py',
    'services/product/product/parallel_residual_gate.py',
    'services/product/product/parallel_typography.py',
    'services/product/product/peer_evidence.py',
    'services/product/product/pipeline.py',
    'services/product/product/scorecard.py',
    'services/product/product/tolerance_config.py',
    'services/product/product/recovery/adaptive_policy.py',
    'services/product/product/recovery/agent_asset_stage.py',
    'services/product/product/recovery/asset_retry.py',
    'services/product/product/recovery/candidate_stage.py',
    'services/product/product/recovery/chart_stage.py',
    'services/product/product/recovery/container_stage.py',
    'services/product/product/recovery/gradient_backing.py',
    'services/product/product/recovery/marked_fields.py',
    'services/product/product/recovery/marked_ownership.py',
    'services/product/product/recovery/marked_surfaces.py',
    'services/product/product/recovery/marked_tables.py',
    'services/product/product/recovery/native_shapes.py',
    'services/product/product/recovery/region_candidates.py',
    'services/product/product/recovery/region_crop.py',
    'services/product/product/recovery/vector_icons.py',
    'services/product/product/runtime/batch_backbone.py',
    'services/product/product/runtime/client.py',
    'services/product/product/runtime/host_baseline.py',
    'services/product/product/runtime/host_evidence.py',
    'services/product/product/runtime/independent_review.py',
    'services/product/product/runtime/launch.py',
    'services/product/product/runtime/marked_routes.py',
    'services/product/product/runtime/mcp_server.py',
    'services/product/product/runtime/office.py',
    'services/product/product/runtime/office_bridge.py',
    'services/product/product/runtime/page_structure.py',
    'services/product/product/runtime/provider.py',
    'services/product/product/runtime/review_control.py',
    'services/product/product/runtime/runner.py',
    'services/product/product/runtime/service.py',
    'services/product/product/runtime/state.py',
    'services/product/product/runtime/text_reading.py',
    'services/product/product/runtime/windows_host.py',
    'skills/image2ppt-runtime/SKILL.md',
    'AGENTS.md',
    'skills/image2ppt/SKILL.md',
    'skills/image2ppt/references/parallel-product.md',
    'skills/image2ppt/references/slideir.md',
    '.agents/skills/image2ppt/SKILL.md',
    '.agents/skills/image2ppt/references/parallel-product.md',
    '.agents/skills/image2ppt/references/slideir.md',
    'skills/image2ppt-claude/references/slideir.md',
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ssh_key = Path.home() / '.ssh/aitoy_tts_4090_ed25519'
    bundle = []
    for rel in SYNC_FILES:
        local_p = ROOT / rel
        if not local_p.is_file():
            raise FileNotFoundError(f'Missing local file: {rel}')
        raw = local_p.read_bytes()
        bundle.append({
            'rel': rel,
            'sha256': sha256_bytes(raw),
            'b64': base64.b64encode(raw).decode('ascii'),
        })

    remote_script = """
import base64, hashlib, json, shutil, sys, time
from pathlib import Path

payload = json.loads(sys.stdin.read())
app_root = Path(r'F:\\i2oc\\app')
backup_root = Path(r'F:\\i2oc\\backup-native-first')
stamp = time.strftime('%Y%m%d-%H%M%S')
results = []

for item in payload['files']:
    rel = item['rel']
    raw = base64.b64decode(item['b64'])
    if hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise RuntimeError(f'Transport hash mismatch for {rel}')
    dst = (app_root / rel).resolve()
    if not dst.is_relative_to(app_root.resolve()):
        raise RuntimeError(f'Escaped app_root: {rel}')
    if dst.is_file():
        old_sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        if old_sha == item['sha256']:
            results.append({'rel': rel, 'action': 'UNCHANGED', 'sha256': old_sha[:12]})
            continue
        bkp = backup_root / stamp / rel
        bkp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, bkp)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + '.tmp_sync')
    tmp.write_bytes(raw)
    tmp.replace(dst)
    got_sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    if got_sha != item['sha256']:
        raise RuntimeError(f'Post-write hash mismatch for {rel}')
    results.append({'rel': rel, 'action': 'UPDATED', 'sha256': got_sha[:12]})

print(json.dumps({'status': 'OK', 'results': results}, indent=2))
"""
    script_b64 = base64.b64encode(remote_script.encode('utf-8')).decode('ascii')
    cmd = [
        'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        '-b', '10.36.154.44', '-i', str(ssh_key),
        'cmcc@10.36.154.54',
        f'F:\\i2oc\\venv\\Scripts\\python.exe -c "import base64;exec(base64.b64decode(\'{script_b64}\'))"',
    ]
    proc = subprocess.run(
        cmd,
        input=json.dumps({'files': bundle}),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode
    print(proc.stdout)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
