"""Run only the live regional VLM loop over retained L1/L2 failure evidence."""
import argparse
import getpass
import json
import os
from pathlib import Path

from product.materialize import write
from product.parallel_contracts import SceneGraph
from product.parallel_provider import Planner
from product.parallel_store import Store
from product.pipeline import settings
from product.recovery.adaptive_policy import AdaptiveBudget,BudgetedActor
from product.recovery.agent_recovery import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-job',required=True)
    parser.add_argument('--prepared-job',required=True)
    parser.add_argument('--layers',required=True)
    parser.add_argument('--skill',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--prompt-key',action='store_true')
    parser.add_argument('--model-cache')
    args=parser.parse_args()
    raw=Path(args.raw_job).resolve();prepared=Path(args.prepared_job).resolve()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    before=SceneGraph.model_validate_json((raw/'page-01/barrier-1.json').read_text('utf8'))
    graph=SceneGraph.model_validate_json((prepared/'page-01/barrier-1.json').read_text('utf8'))
    before.text=graph.text.model_copy(deep=True)
    ocr=None
    for ref in graph.text.references:
        file=Path(ref)
        if not file.is_file():continue
        try:value=json.loads(file.read_text('utf8'))
        except (ValueError,UnicodeError):continue
        if value.get('schema')=='OCRLines/1' and value.get('source_sha256')==graph.source_sha256:
            ocr=value;break
    if ocr is None:raise ValueError('independent OCR required')
    if args.prompt_key:os.environ['MOMA_API_KEY']=getpass.getpass('MoMA credential (hidden): ')
    store=Store(output/'state');budget=AdaptiveBudget(store,{'model_calls':3})
    actor=BudgetedActor(Planner(settings()),budget)
    source=prepared/'inputs/01.png'
    result=run(before,graph,{'path':str(source),'source_sha256':graph.source_sha256,'image_size':graph.image_size,'transform':graph.transform},
        ocr,prepared/'page-01/staged',json.loads((prepared/'fonts.json').read_text('utf8')),
        Path(args.layers).resolve(),json.loads((prepared/'page-01/asset-coordinate-report.json').read_text('utf8')),
        actor,Path(args.model_cache).resolve() if args.model_cache else output/'cache',store,output/'a',args.skill)
    write(output/'result.json',{'live_vlm_called':True,'result':result,'budget':budget.report(),
        'office':'NOT_RUN','go':'NOT_RUN','automatic_publication':False})
    print(json.dumps({'status':result['status'],'model_calls':budget.report()['counts']['model_calls']}),flush=True)
    if result['status']!='CANDIDATE_READY':raise SystemExit(1)


if __name__=='__main__':main()
