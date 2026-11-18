"""Run matched OpenCode uploads against one isolated product service; resume by job ID."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time
import httpx

PROFILES=('baseline','marks','pp-l','doclayout','omni','vitmatte','local-layerd','vtracer','combined')


def save(path, data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    tmp.replace(path)


def record(client, args, profile, images, label):
    output=args.out/label
    output.mkdir(parents=True,exist_ok=True)
    receipt=output/'admission.json'
    inputs=[hashlib.sha256(p.read_bytes()).hexdigest() for p in images]
    if receipt.exists():
        job=json.loads(receipt.read_text(encoding='utf8'))
        if job['inputs']!=inputs or job['profile']!=profile:raise ValueError('resume input mismatch')
        if 'id' not in job:raise RuntimeError('admission unknown; reconcile API queue before retry')
    else:
        job={'inputs':inputs,'profile':profile,'started':time.time()}
        save(receipt,job)
        with contextlib.ExitStack() as stack:
            files=[('files',(p.name,stack.enter_context(p.open('rb')),'image/png')) for p in images]
            r=client.post('/api/v1/jobs',params={'experiment_profile':profile,'agent_steps':192,
                'runtime_seconds':10800,'goal':'Reconstruct all pages as faithful editable presentations. Preserve independently editable business fields, cards, tables and images.'},files=files)
        r.raise_for_status()
        job['id']=r.json()['id']
        save(receipt,job)
    jid=job['id']
    print(label,jid,flush=True)
    while True:
        r=client.get('/api/v1/jobs/'+jid)
        r.raise_for_status()
        status=r.json()
        now=time.time()
        if now-job['started']>=2700 and not (output/'45-minutes.json').exists():
            snapshot={'observed_at':now,'elapsed_seconds':now-job['started'],'status':status.get('state',status.get('status'))}
            statefile=args.jobs/jid/'runtime/state.json'
            if statefile.exists():
                state=json.loads(statefile.read_text(encoding='utf8'))
                snapshot['committed_by_page']=[len(json.loads(Path(p['scene']).read_text(encoding='utf8'))['selected']) if p else 0 for p in state.get('pages',[])]
                snapshot['composed']=bool(state.get('composition'))
            save(output/'45-minutes.json',snapshot)
        if status.get('state',status.get('status')) not in ('QUEUED','RUNNING'):
            save(output/'terminal.json',status)
            break
        time.sleep(10)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',required=True)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--jobs',required=True,type=Path)
    p.add_argument('--images',required=True,nargs='+',type=Path)
    p.add_argument('--holdout',action='append',type=Path,default=[])
    p.add_argument('--profiles',nargs='+',choices=PROFILES,default=PROFILES)
    args=p.parse_args()
    with httpx.Client(base_url=args.url,headers={'Authorization':'Bearer '+os.environ['IMAGE2PPT_PRODUCT_TOKEN']},timeout=120,trust_env=False) as client:
        for profile in args.profiles:record(client,args,profile,args.images,profile)
        for i,image in enumerate(args.holdout,1):record(client,args,'combined',[image],f'combined-holdout-{i}')
if __name__=='__main__':main()

