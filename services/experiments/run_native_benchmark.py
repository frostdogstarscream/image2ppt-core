"""Fresh /api/v1/jobs comparisons with a source-hash-bound, evaluation-only manifest."""
import argparse
import hashlib
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'services/product'))
from product.materialize import digest, write
from product.runtime.client import load_profile


def run_group(client, group, output, *, baseline_only, steps, seconds, probe, variant, probe_selected=False):
    output.mkdir(parents=True, exist_ok=True)
    sources = [row['sha256'] for row, _ in group]
    for row, path in group:
        if digest(path) != row['sha256']:
            raise ValueError(f'Frozen input changed: {path}')
    params = dict(agent_model='hy4-flash-free', baseline_only=baseline_only, native_variant=variant, agent_steps=steps,
                  runtime_seconds=seconds, model_calls=32, office_validations=60, tool_profile='all')
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
    while True:
        try:
            response = client.get(route)
            response.raise_for_status()
        except httpx.TransportError:
            time.sleep(5)
            continue
        state = response.json()
        if time.time()-admission['started'] >= 2700 and not (output/'checkpoint-45min.json').exists():
            write(output/'checkpoint-45min.json',state)
        if state['status'] in ('SUCCEEDED','REVIEW_REQUIRED','FAILED','INTERRUPTED','CANCELLED'):
            break
        time.sleep(5)
    write(output/'terminal.json',state)
    artifacts = client.get(route+'/artifacts')
    artifacts.raise_for_status()
    delivery = output/'delivery'
    delivery.mkdir(exist_ok=True)
    for artifact in artifacts.json()['artifacts']:
        name = artifact['name']
        if Path(name).name != name:
            raise ValueError('Invalid artifact filename')
        response = client.get(artifact['url'])
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != artifact['sha256']:
            raise ValueError('Downloaded artifact differs from server receipt')
        (delivery/name).write_bytes(response.content)
    results = {}
    cases = [[i,*case[1:]] for i,(row,_) in enumerate(group,1) for case in row.get('cases',[])]
    for version,filename in [('host','deterministic-baseline.pptx'),('agent_candidate','agent-candidate.pptx'),('selected','result.pptx')]:
        deck = delivery/filename
        if not deck.exists():
            results[version] = {'status':'UNKNOWN'}
            continue
        rec = {'sha256':digest(deck),'actions':'UNKNOWN'}
        if cases and (probe or (probe_selected and version == 'selected')):
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
    parser.add_argument('--variant',choices=['combined','shape-only','joint-only'],default='combined')
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--probe-office',action='store_true')
    parser.add_argument('--probe-selected',action='store_true',
                        help='Run frozen Office edits on the final downloaded deck only')
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
    with httpx.Client(base_url=url,headers=headers,timeout=60,trust_env=False) as client:
        for i in range(0,len(group),2):
            part=group[i:i+2]
            name='-'.join(r['id'] for r,_ in part)
            records.append(run_group(client,part,args.output/name,baseline_only=args.baseline_only,
                                     steps=args.steps,seconds=args.seconds,probe=args.probe_office,variant=args.variant,
                                     probe_selected=args.probe_selected))
    write(args.output/'benchmark-summary.json',records)
    from native_results import write_table
    write_table(args.output.glob('*/comparison.json'), args.output/'page-comparison.csv')


if __name__=='__main__':
    main()

