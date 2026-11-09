"""Run the same backend object stage against frozen page/text evidence.

Keys are accepted through getpass or the product's normal environment only.
This tool never reads another application's credential store.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import shutil
import time
from product.contracts import Page
from product.materialize import digest,write
from product.object_contracts import ClosedPlan,CLOSED_OBJECT_PROMPT
from product.object_stage import process,effective_page
from product.pipeline import Provider,settings


def run(args):
    if args.keys_stdin:
        os.environ['MOMA_API_KEY']=getpass.getpass('MoMA key (process memory): ')
        os.environ['PLANNER_API_KEY']=os.environ['MOMA_API_KEY']
        os.environ['OPENCODE_GO_API_KEY']=getpass.getpass('Go key (process memory): ')
    root=args.input.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    provider=Provider(settings());ledger=json.loads((output/'calls.json').read_text(encoding='utf-8')) if (output/'calls.json').exists() else []
    records=json.loads((root/'inputs.json').read_text(encoding='utf-8'))
    for record in sorted(records,key=lambda r:args.pages.index(r['page']) if r['page'] in args.pages else 999):
        number=record['page']
        if number not in args.pages:continue
        directory=output/f'page-{number:02d}'
        if (directory/'result.json').exists():print(f'page {number}: cached',flush=True);continue
        page=Page.model_validate(json.loads(Path(record['baseline_page']).read_text(encoding='utf-8')))
        attempt=0
        while (directory/f'attempt-{attempt:02d}').exists():attempt+=1
        def ask(context,images,contract):
            call={'page':number,'model':provider.model,'state':'RESERVED','started_at':time.time()};ledger.append(call);write(output/'calls.json',ledger)
            try:raw=provider.ask(CLOSED_OBJECT_PROMPT,context,images,contract)
            except Exception:
                call['state']='FAILED';write(output/'calls.json',ledger);raise
            call.update(state='RECEIVED',finished_at=time.time());write(output/'calls.json',ledger)
            return raw
        try:
            ir,proposal,report=process(record['source'],page,record['baseline_ir'],directory/f'attempt-{attempt:02d}','auto',ask,
                layerdir=Path(record['directory'])/'layerd',state_directory=directory/'state')
        except (ValueError,RuntimeError,OSError) as exc:
            write(directory/f'failure-{attempt:02d}.json',{'type':type(exc).__name__,'error':str(exc)[:1000],'parent_preserved':True})
            print(json.dumps({'page':number,'state':'FALLBACK','reason':type(exc).__name__}),flush=True)
            continue
        write(directory/'effective.json',proposal.model_dump())
        write(directory/'result.json',{'page':number,'ir':str(ir),'parent_ir':record['baseline_ir'],'report':report,
            'source':record['source'],'effective':str(directory/'effective.json'),'human_acceptance':False})
        print(json.dumps({'page':number,'staged':report['staged'],'fallback':report['fallback']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pages',type=int,nargs='+',default=list(range(1,13)));p.add_argument('--keys-stdin',action='store_true')
    run(p.parse_args())
