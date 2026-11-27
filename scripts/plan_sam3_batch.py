"""Explicit bounded SAM3 planner: one request plus one schema-only retry per PNG."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from plan_segmentation import run, read, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model-config',required=True);p.add_argument('--model',required=True)
    p.add_argument('--max-calls',type=int,default=20)
    a=p.parse_args();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    if not 1<=len(a.inputs)<=10 or not 1<=a.max_calls<=20:raise ValueError('maximum ten pages and twenty calls')
    result={'schema':'SAM3PlanBatch/1','max_calls':a.max_calls,'requests':[],'pages':{},'human_acceptance':False}
    def save(): (root/'batch.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    for name in a.inputs:
        source=Path(name).resolve(strict=True);success=False
        for attempt in (1,2):
            if len(result['requests'])>=a.max_calls:save();return 2
            target=root/source.stem/f'attempt-{attempt}'
            result['requests'].append({'page':source.stem,'source_sha256':digest(source),'attempt':attempt,'directory':str(target),'reserved':True});save()
            run(SimpleNamespace(input=str(source),output=str(target),model_config=a.model_config,model=a.model,target='sam3'))
            job=read(target/'job.json');response=read(target/'response.json')
            result['pages'][source.stem]={'status':job['status'],'directory':str(target)};save()
            if job['status']=='REVIEW_REQUIRED': success=True;break
            # Provider, network and quota failures are never retried or hidden by a model switch.
            if not response.get('ok'):result['stopped']='provider_or_transport_failure';save();return 2
        if not success:result['stopped']='schema_retry_exhausted';save();return 2
    manifest={'schema':'LayeredPlanInputs/1','plans':[v['directory'] for v in result['pages'].values()]}
    (root/'plans.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');save();return 0


if __name__=='__main__':raise SystemExit(main())
