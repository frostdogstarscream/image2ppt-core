"""Automated 4090 OpenCode mainline benchmark & before/after comparison runner (W8).

Usage:
  python scripts/run_4090_benchmark.py --slides 08,02 --output outputs/benchmark-4090-run
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'services/product'))

import httpx
from product import scorecard
from product.runtime.client import load_profile

INPUT_DIR = ROOT / 'outputs/benchmark-ai-workbench-8p/inputs'
BASELINE_08 = ROOT / 'outputs/e2e-hy4-flash-free-08-a441b7fb/scorecard.json'
BASELINE_DIR = ROOT / 'outputs/scorecard-baseline'


def _find_slide(prefix: str) -> Path:
    matches = sorted(INPUT_DIR.glob(f'{prefix}_*.png'))
    if not matches:
        matches = sorted(INPUT_DIR.glob(f'*{prefix}*.png'))
    if not matches:
        raise FileNotFoundError(f'No benchmark slide matching {prefix} in {INPUT_DIR}')
    return matches[0]


def _load_baseline_metrics(prefix: str) -> dict | None:
    if prefix == '08' and BASELINE_08.is_file():
        sc = json.loads(BASELINE_08.read_text('utf-8'))
        return {
            'run': 'baseline-hy4-08 (a441b7fb)',
            'native_editable': sc.get('native_editable_content_fraction'),
            'movable_raster': sc.get('movable_raster_content_fraction'),
            'plate_content': sc.get('raster_plate_content_fraction'),
            'plate_area': sc.get('visible_area_fraction', {}).get('plate'),
            'shapes': sc.get('object_counts', {}).get('shape', 0),
            'texts': sc.get('object_counts', {}).get('text', 0),
            'ssim': sc.get('fidelity', {}).get('ssim'),
            'de_p95': sc.get('fidelity', {}).get('delta_e2000_p95'),
            'ocr_recall': sc.get('text', {}).get('non_logo_line_recall') or sc.get('text', {}).get('line_recall'),
        }
    if BASELINE_DIR.is_dir():
        cands = sorted(BASELINE_DIR.glob(f'*p{prefix}*.json'))
        if cands:
            sc = json.loads(cands[-1].read_text('utf-8'))
            return {
                'run': cands[-1].stem,
                'native_editable': sc.get('editable_content_fraction'),
                'movable_raster': sc.get('movable_raster_content_fraction', 0.0),
                'plate_content': sc.get('raster_plate_content_fraction'),
                'plate_area': sc.get('visible_area_fraction', {}).get('plate'),
                'shapes': sc.get('object_counts', {}).get('shape', 0),
                'texts': sc.get('object_counts', {}).get('text', 0),
                'ssim': sc.get('fidelity', {}).get('ssim'),
                'de_p95': sc.get('fidelity', {}).get('delta_e2000_p95'),
                'ocr_recall': sc.get('text', {}).get('line_recall'),
            }
    return None


def run_one_slide(profile: dict, slide_path: Path, out_dir: Path, *, agent_model: str, agent_provider: str | None = None, steps: int, seconds: int, baseline_only: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = profile['url'].rstrip('/')
    token = profile['business_token']
    headers = {'Authorization': f'Bearer {token}'}

    goal = (
        '将原图恢复为尽可能全原生可编辑的 PPTX（优先还原原生文字、原生圆角/五边形/椭圆/渐变形状与独立透明图标/照片模块，'
        '允许轻微字体或阴影视觉偏差），完成实际 PowerPoint 验证与视觉审计后交付。'
    )
    t0 = time.time()
    with ExitStack() as stack, httpx.Client(base_url=url, headers=headers, timeout=60, trust_env=False) as client:
        files = [('files', (slide_path.name, stack.enter_context(open(slide_path, 'rb')), 'image/png'))]
        params = {
            'goal': goal,
            'tool_profile': 'all',
            'agent_steps': steps,
            'runtime_seconds': seconds,
            'candidate_limit': 4,
            'model_calls': 16,
            'office_validations': 60,
            'agent_thinking': False,
            'agent_model': agent_model,
        }
        if baseline_only:
            params['baseline_only'] = True
        if agent_provider:
            params['agent_provider'] = agent_provider
        resp = client.post('/api/v1/jobs', params=params, files=files)
        resp.raise_for_status()
        sub = resp.json()
    job_id = sub['id']
    print(f'[{slide_path.name}] Submitted job_id={job_id}', flush=True)

    status_data = {}
    terminal = ('SUCCEEDED', 'REVIEW_REQUIRED', 'QUALITY_FAILED', 'FAILED', 'INTERRUPTED', 'CANCELLED')
    final_status = None
    res_terminal_since: float | None = None
    consecutive_poll_errors = 0
    poll_timeout = seconds + (min(seconds, max(300, int(seconds * 0.5))) if seconds > 0 else 0) + 360
    while True:
        time.sleep(2 if res_terminal_since is not None else 5)
        try:
            with httpx.Client(timeout=45, trust_env=False) as client:
                r = client.get(f'{url}/api/v1/jobs/{job_id}', headers=headers)
                r.raise_for_status()
                status_data = r.json()
            consecutive_poll_errors = 0
        except (httpx.HTTPError, OSError) as exc:
            consecutive_poll_errors += 1
            elapsed = round(time.time() - t0, 1)
            print(f'  [{job_id[:8]}] +{elapsed}s transient poll error ({consecutive_poll_errors}/15): {type(exc).__name__}', flush=True)
            if elapsed > poll_timeout:
                print(f'  [{job_id[:8]}] client poll timeout after +{elapsed}s', flush=True)
                final_status = 'CLIENT_POLL_TIMEOUT'
                break
            if consecutive_poll_errors >= 15:
                raise
            continue
        st = status_data.get('status')
        res_st = (status_data.get('result') or {}).get('status')
        elapsed = round(time.time() - t0, 1)
        if st in terminal:
            print(f'  [{job_id[:8]}] +{elapsed}s status={st}' + (f' result.status={res_st}' if res_st and res_st != st else ''), flush=True)
            final_status = res_st if res_st in terminal else st
            break
        if res_st in terminal:
            if res_terminal_since is None:
                res_terminal_since = time.time()
                print(f'  [{job_id[:8]}] +{elapsed}s status={st} result.status={res_st} (waiting for queue settle)', flush=True)
            if time.time() - res_terminal_since >= 20.0:
                final_status = res_st
                break
        elif int(elapsed) % 20 < 5:
            print(f'  [{job_id[:8]}] +{elapsed}s status={st}', flush=True)
        if elapsed > poll_timeout:
            print(f'  [{job_id[:8]}] client poll timeout after +{elapsed}s (last status={st})', flush=True)
            final_status = 'CLIENT_POLL_TIMEOUT'
            break

    wall_s = round(time.time() - t0, 1)
    (out_dir / 'job-status.json').write_text(json.dumps(status_data, ensure_ascii=False, indent=2), encoding='utf-8')

    dl_dir = out_dir / 'delivery'
    dl_dir.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(base_url=url, headers=headers, timeout=60, trust_env=False) as client:
            def _get_with_409_retry(req_url: str):
                for attempt in range(5):
                    try:
                        resp = client.get(req_url)
                    except (httpx.TransportError, httpx.TimeoutException):
                        if attempt < 4:
                            time.sleep(2)
                            continue
                        raise
                    if resp.status_code == 409 and attempt < 4:
                        time.sleep(2)
                        continue
                    resp.raise_for_status()
                    return resp
                raise RuntimeError(f'unreachable retry exit for {req_url}')

            art_resp = _get_with_409_retry(f'/api/v1/jobs/{job_id}/artifacts')
            for art in art_resp.json().get('artifacts', []):
                name = art['name']
                dl = _get_with_409_retry(art['url'])
                if hashlib.sha256(dl.content).hexdigest() != art['sha256']:
                    raise ValueError(f'Artifact hash mismatch: {name}')
                (dl_dir / name).write_bytes(dl.content)
    except Exception as exc:
        print(f'  [{job_id[:8]}] artifact download note: {exc}', flush=True)

    pptx = dl_dir / 'result.pptx'
    render_png = dl_dir / 'page-01.png'
    sc_file = dl_dir / 'scorecard.json'
    sc = None
    if sc_file.is_file():
        sc = json.loads(sc_file.read_text('utf-8'))
    elif pptx.is_file():
        sc = scorecard.score_page(
            slide_path, pptx, render_path=render_png if render_png.is_file() else None
        )
        scorecard.write_json(out_dir / 'scorecard.json', sc)
        scorecard.save_owner_preview(slide_path, pptx, out_dir / 'owner-preview.png')

    ooxml_summary = None
    if pptx.is_file():
        import zipfile
        from collections import Counter
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(pptx) as zf:
            sxml = zf.read('ppt/slides/slide1.xml').decode('utf-8', errors='replace')
            root_el = ET.fromstring(sxml)
            ns_a = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
            prst_counts = Counter(el.attrib.get('prst', '') for el in root_el.iter(f'{ns_a}prstGeom'))
            ooxml_summary = {
                'custGeom': sum(1 for _ in root_el.iter(f'{ns_a}custGeom')),
                'gradFill': sum(1 for _ in root_el.iter(f'{ns_a}gradFill')),
                'prstGeom': dict(prst_counts),
            }

    res_obj = status_data.get('result', {}) or {}
    return {
        'slide': slide_path.name,
        'job_id': job_id,
        'status': final_status or status_data.get('status'),
        'queue_status': status_data.get('status'),
        'wall_seconds': wall_s,
        'ooxml': ooxml_summary,
        'baseline_timings': res_obj.get('baseline_timings', {}),
        'baseline_candidate_timings': res_obj.get('baseline_candidate_timings', {}),
        'baseline_isolated_dropped_ids': res_obj.get('baseline_isolated_dropped_ids', {}),
        'scorecard': sc,
        'result': res_obj,
    }


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _office_bridge_running(url: str) -> bool:
    if os.name != 'nt':
        return False
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host_port = parsed.netloc or url
    ps_cmd = (
        "Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" -ErrorAction SilentlyContinue | "
        f"Where-Object {{ $_.CommandLine -match 'product\\.runtime\\.office_bridge' -and $_.CommandLine -match '{host_port}' }} | "
        "Select-Object -First 1 -ExpandProperty ProcessId"
    )
    try:
        proc = subprocess.run(['powershell', '-NoProfile', '-Command', ps_cmd], capture_output=True, text=True, timeout=8)
        return bool(proc.stdout.strip())
    except Exception:
        return False


@contextmanager
def _ensure_bridge(profile: dict):
    procs: list[subprocess.Popen] = []
    try:
        if not _port_open('127.0.0.1', 18772):
            ssh_key = Path.home() / '.ssh/aitoy_tts_4090_ed25519'
            ssh_p = subprocess.Popen(
                [
                    'ssh', '-N', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30',
                    '-b', '10.36.154.44', '-i', str(ssh_key),
                    '-L', '18772:127.0.0.1:8772', 'cmcc@10.36.154.54',
                ],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            procs.append(ssh_p)
            for _ in range(20):
                if _port_open('127.0.0.1', 18772):
                    break
                time.sleep(0.3)

        # Ensure a PowerPoint office_bridge worker is connected to tunnel without duplicating existing ones
        if not _office_bridge_running(profile['url']):
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{ROOT / 'services/product'};{ROOT / 'ppt-rebuild/src'}"
            env['IMAGE2PPT_OFFICE_WORKER_TOKEN'] = profile['office_token']
            office_root = ROOT / '.build/office-worker'
            office_root.mkdir(parents=True, exist_ok=True)
            worker_p = subprocess.Popen(
                [sys.executable, '-m', 'product.runtime.office_bridge', '--url', profile['url'], '--root', str(office_root)],
                env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            procs.append(worker_p)
        time.sleep(1.5)
        yield
    finally:
        for p in reversed(procs):
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', default=str(ROOT / '.runtime/opencode-client.dpapi'))
    parser.add_argument('--slides', default='08,02')
    parser.add_argument('--agent-provider', default=None)
    parser.add_argument('--agent-model', default='deepseek-flash')
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--seconds', type=int, default=2400)
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    prof = load_profile(args.profile)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    prefixes = [
        f'{int(s.strip()):02d}' if s.strip().isdigit() else s.strip()
        for s in args.slides.split(',') if s.strip()
    ]
    records = []
    with _ensure_bridge(prof):
        for pfx in prefixes:
            slide_p = _find_slide(pfx)
            base_m = _load_baseline_metrics(pfx)
            run_res = run_one_slide(
                prof, slide_p, out_root / f'slide-{pfx}',
                agent_model=args.agent_model, agent_provider=args.agent_provider,
                steps=args.steps, seconds=args.seconds,
                baseline_only=args.baseline_only,
            )
            run_res['prefix'] = pfx
            run_res['baseline'] = base_m
            records.append(run_res)

    summary_path = out_root / 'benchmark-summary.json'
    summary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved benchmark summary to {summary_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
