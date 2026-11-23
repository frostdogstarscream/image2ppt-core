"""Fresh /api/v1/jobs comparisons with a source-hash-bound, evaluation-only manifest."""
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

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'services/product'))
from product.materialize import digest, write
from product.runtime.client import load_profile


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


def run_group(client, group, output, *, baseline_only, steps, seconds, probe, variant, agent_model='deepseek-flash', agent_provider=None):
    output.mkdir(parents=True, exist_ok=True)
    sources = [row['sha256'] for row, _ in group]
    for row, path in group:
        if digest(path) != row['sha256']:
            raise ValueError(f'Frozen input changed: {path}')
    params = dict(agent_model=agent_model, baseline_only=baseline_only, native_variant=variant, agent_steps=steps,
                  runtime_seconds=seconds, model_calls=32, office_validations=60, tool_profile='all',
                  agent_thinking=False)
    if agent_provider:
        params['agent_provider'] = agent_provider
    receipt = output/'admission.json'
    if receipt.exists():
        admission = json.loads(receipt.read_text(encoding='utf8'))
        if admission['sources'] != sources or admission['params'] != params:
            raise ValueError('Output directory already belongs to a different comparison')
        if not admission.get('job_id'):
            raise ValueError('Prior upload response unknown; reconcile server job before resuming')
    else:
        admission = {'sources':sources,'params':params,'started':time.time(),'job_id':None}
        write(receipt, admission)
        with ExitStack() as stack:
            files = [('files',(p.name,stack.enter_context(p.open('rb')),'image/png')) for _,p in group]
            reply = client.post('/api/v1/jobs',params=params,files=files)
            reply.raise_for_status()
            admission['job_id'] = reply.json()['id']
            write(receipt, admission)
    job = admission['job_id']
    print(f'{output.name} {job}', flush=True)
    route = '/api/v1/jobs/'+job
    terminal = ('SUCCEEDED', 'REVIEW_REQUIRED', 'QUALITY_FAILED', 'FAILED', 'INTERRUPTED', 'CANCELLED')
    consecutive_errors = 0
    res_terminal_since = None
    last_log_t = 0.0
    while True:
        try:
            response = client.get(route)
            response.raise_for_status()
            consecutive_errors = 0
        except (httpx.TransportError, httpx.HTTPError) as exc:
            consecutive_errors += 1
            elapsed = round(time.time() - admission['started'], 1)
            print(f'  [{output.name} {job[:8]}] +{elapsed}s poll error ({consecutive_errors}/8): {type(exc).__name__}', flush=True)
            if consecutive_errors >= 8:
                raise
            time.sleep(5)
            continue
        state = response.json()
        now = time.time()
        elapsed = round(now - admission['started'], 1)
        if now - admission['started'] >= 2700 and not (output/'checkpoint-45min.json').exists():
            write(output/'checkpoint-45min.json',state)
        st = state['status']
        res_st = (state.get('result') or {}).get('status')
        if st in terminal:
            print(f'  [{output.name} {job[:8]}] +{elapsed}s status={st}' + (f' result.status={res_st}' if res_st and res_st != st else ''), flush=True)
            break
        if res_st in terminal:
            if res_terminal_since is None:
                res_terminal_since = now
                print(f'  [{output.name} {job[:8]}] +{elapsed}s status={st} result.status={res_st} (waiting queue settle)', flush=True)
            if now - res_terminal_since >= 20.0:
                break
        elif now - last_log_t >= 30.0:
            last_log_t = now
            print(f'  [{output.name} {job[:8]}] +{elapsed}s status={st}', flush=True)
        time.sleep(2 if res_terminal_since is not None else 5)
    write(output/'terminal.json',state)
    def _get_retry(req_url: str):
        for attempt in range(5):
            try:
                resp = client.get(req_url)
            except (httpx.TransportError, httpx.HTTPError):
                if attempt < 4:
                    time.sleep(2)
                    continue
                raise
            if resp.status_code == 409 and attempt < 4:
                time.sleep(2)
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f'Unreachable retry exit for {req_url}')

    artifacts = _get_retry(route+'/artifacts')
    delivery = output/'delivery'
    delivery.mkdir(exist_ok=True)
    for artifact in artifacts.json()['artifacts']:
        name = artifact['name']
        if Path(name).name != name:
            raise ValueError('Invalid artifact filename')
        response = _get_retry(artifact['url'])
        if hashlib.sha256(response.content).hexdigest() != artifact['sha256']:
            raise ValueError('Downloaded artifact differs from server receipt')
        (delivery/name).write_bytes(response.content)
    results = {}
    probed_by_sha = {}
    cases = [[i,*case[1:]] for i,(row,_) in enumerate(group,1) for case in row.get('cases',[])]
    for version,filename in [('host','deterministic-baseline.pptx'),('agent_candidate','agent-candidate.pptx'),('selected','result.pptx')]:
        deck = delivery/filename
        if not deck.exists():
            results[version] = {'status':'UNKNOWN'}
            continue
        sha = digest(deck)
        if sha in probed_by_sha:
            results[version] = dict(probed_by_sha[sha])
            continue
        rec = {'sha256':sha,'actions':'UNKNOWN'}
        if cases and probe:
            from probe_fields import run
            folder = output/('office-'+version)
            run(deck,folder,cases)
            rec['actions'] = json.loads((folder/'report.json').read_text(encoding='utf8'))
            from native_region_metrics import measure
            rec['region_fidelity'] = {}
            for page, (_, source) in enumerate(group, 1):
                page_cases = [case for case in cases if case[0]==page]
                preview = next((folder/f'{i}-before.png' for i, case in enumerate(cases)
                                if case[0]==page and (folder/f'{i}-before.png').is_file()), None)
                if page_cases and preview:
                    rec['region_fidelity'][str(page)] = measure(source, preview, page_cases)
        probed_by_sha[sha] = rec
        results[version] = rec
    result = {'job_id':job,'sources':sources,'state':state['status'],'versions':results,
              'origin':state.get('result',{}).get('origin'),'agent':state.get('result',{}).get('agent'),
              'sample_catalog':[{k:r.get(k) for k in ('id','family','split','sha256')} for r,_ in group],
              'human_wps':'UNKNOWN'}
    write(output/'comparison.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'services/experiments/native_edits/workbench.json')
    parser.add_argument('--input-root',required=True,type=Path)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--profile',required=True,type=Path)
    parser.add_argument('--url')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--pages',nargs='+')
    parser.add_argument('--agent-provider',default=None)
    parser.add_argument('--agent-model',default='deepseek-flash')
    parser.add_argument('--variant',choices=['combined','shape-only','joint-only'],default='combined')
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--probe-office',action='store_true')
    parser.add_argument('--steps',type=int,default=192)
    parser.add_argument('--seconds',type=int,default=10800)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    rows=manifest['pages']
    if 'cases_sha256_lf' in manifest:
        from holdout.check_inputs import check
        check(args.project_root,args.input_root)
        # Refuse to evaluate altered case definitions.
        casefile=args.manifest.parent/manifest['cases']
        if hashlib.sha256(casefile.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=manifest['cases_sha256_lf']:
            raise ValueError('Frozen case list changed')
        cases=json.loads(casefile.read_text(encoding='utf8'))
        rows=[{**r,'id':str(r['page']),'file':r['path'],'cases':[c for c in cases if c[0]==r['page']]} for r in rows]
    development = json.loads((ROOT/'services/experiments/native_edits/workbench.json').read_text(encoding='utf8'))['pages']
    known_development = {row['sha256'] for row in development}
    for row in rows:
        row['split'] = 'development' if row['sha256'] in known_development else row.get('split','frozen')
    if args.pages:
        chosen=set(args.pages)
        rows=[r for r in rows if r['id'] in chosen]
        if chosen!={r['id'] for r in rows}:raise ValueError('Unknown page ID')
    group=[(r,(args.project_root if r.get('source_root')=='project' else args.input_root)/r['file']) for r in rows]
    profile=load_profile(args.profile)
    url=args.url or profile['url']
    from urllib.parse import urlsplit
    if urlsplit(url).hostname not in ('127.0.0.1','localhost','::1'):
        raise ValueError('Use the project SSH loopback tunnel')
    headers={'Authorization':'Bearer '+profile['business_token']}
    records=[]
    with _ensure_bridge(profile):
        with httpx.Client(base_url=url,headers=headers,timeout=90,trust_env=False) as client:
            for i in range(0,len(group),2):
                part=group[i:i+2]
                name='-'.join(r['id'] for r,_ in part)
                records.append(run_group(client,part,args.output/name,baseline_only=args.baseline_only,
                                         steps=args.steps,seconds=args.seconds,probe=args.probe_office,variant=args.variant,
                                         agent_model=args.agent_model,agent_provider=args.agent_provider))
    write(args.output/'benchmark-summary.json',records)
    from native_results import write_table
    write_table(args.output.glob('*/comparison.json'), args.output/'page-comparison.csv')


if __name__=='__main__':
    main()

