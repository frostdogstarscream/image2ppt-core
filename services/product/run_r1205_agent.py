"""Run one evidence-bound text or icon repair through the live regional VLM."""
import argparse
import getpass
import json
import os
from pathlib import Path

from product.agent_service.model_gateway import ModelGateway
from product.agent_service.skill_loader import load_skill
from product.materialize import digest,write
from product.parallel_contracts import SceneGraph
from product.parallel_provider import Planner
from product.parallel_store import Store,key
from product.pipeline import settings
from product.recovery.adaptive_policy import AdaptiveBudget,BudgetedActor
from product.recovery.agent_decision import RegionalRepairAgent,ToolCandidate
from product.recovery.agent_tool_catalog import register_repair_tools
from product.recovery.background_adapter import RecoveryBlocked


def read(file):
    return json.loads(Path(file).read_text('utf8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('job','layers','skill','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--failure',help='actual failure JSON; asset mode can discover an unowned complete-icon failure')
    parser.add_argument('--kind',choices=['text','asset'],required=True)
    parser.add_argument('--prompt-key',action='store_true')
    parser.add_argument('--model-cache')
    parser.add_argument('--rejected-candidate')
    parser.add_argument('--rejection-report')
    args=parser.parse_args()
    if args.kind=='text' and not args.failure:
        parser.error('--failure is required for a text repair')
    if bool(args.rejected_candidate)!=bool(args.rejection_report):
        parser.error('--rejected-candidate and --rejection-report must be provided together')
    if args.kind=='asset' and args.rejected_candidate:
        parser.error('background rejection arguments apply to text repairs')
    job=Path(args.job).resolve();layers=Path(args.layers).resolve()
    failure_file=Path(args.failure).resolve() if args.failure else None
    output=Path(args.output).resolve()
    result=read(job/'result.json')
    if len(result.get('slide_irs',[]))!=1:
        raise ValueError('regional CLI requires one source page')
    baseline=Path(result['slide_irs'][0]).resolve().parent
    scene_file=baseline/'scene.json' if args.kind=='asset' else job/'page-01/barrier-1.json'
    graph=SceneGraph.model_validate_json(scene_file.read_text('utf8'))
    source=baseline/'source.png'
    if failure_file is not None:
        failure=read(failure_file)
    else:
        definitions={obj.id:obj for obj in graph.asset.objects}
        transactions=baseline/'transactions.json'
        failed=next((row for row in read(transactions)
            if row['id'] in definitions and definitions[row['id']].kind=='icon'
            and row['id'] not in graph.selected and any(
                trial.get('reason')=='source ink has no complete candidate owner'
                for trial in row.get('candidate_failures',[]))),None)
        if failed is None:raise ValueError('no source-owner closure failure for an unselected complete icon')
        failure={'schema':'AssetRepairFailure/1','object_id':failed['id'],'source_sha256':graph.source_sha256,
            'scene_sha256':key(graph.model_dump()),'failure':'ASSET_EXTRACTION_UNCERTAIN',
            'original_transaction':failed,'transaction_file':str(transactions),
            'transaction_sha256':digest(transactions)}
    identifier=failure.get('object_id',failure.get('id'))
    source_binding=failure.get('source_sha256',failure.get('binding',{}).get('source_sha256'))
    if not isinstance(identifier,str) or source_binding!=graph.source_sha256 or digest(source)!=graph.source_sha256:
        raise ValueError('failure must name an object and bind the immutable job source')
    inventory=graph.text.texts if args.kind=='text' else graph.asset.objects
    if identifier not in {obj.id for obj in inventory}:
        raise ValueError('failure target does not match the requested object kind')
    package=Path(__file__).resolve().parent/'product'
    callbacks={};bindings={}
    if args.kind=='text':
        from product.recovery.text_actions import retry_text_mask_completion,switch_background_candidate
        fonts=read(job/'fonts.json')
        def mask_retry(directory):
            receipt=retry_text_mask_completion(graph,identifier,source,layers,baseline,fonts,
                                               directory,previous_receipt=failure_file)
            write(directory/'action-result.json',receipt)
            if receipt['state']!='STAGED_LOCAL':
                raise RecoveryBlocked(receipt.get('failure','MASK_INCOMPLETE'),receipt.get('reason','mask repair remains blocked'))
            return ToolCandidate(graph,Path(receipt['candidate_directory']))
        callbacks['retry_text_mask_completion']=mask_retry
        dependencies=['text_actions','text_probe','text_mask_completion','region_candidates','candidate_stage']
        bindings['retry_text_mask_completion']=key({name:digest(package/'recovery'/(name+'.py')) for name in dependencies})
        if args.rejected_candidate:
            def background_retry(directory):
                receipt=switch_background_candidate(graph,identifier,source,layers,baseline,fonts,directory,
                    rejected_candidate_directory=Path(args.rejected_candidate).resolve(),
                    rejection_report=Path(args.rejection_report).resolve())
                write(directory/'action-result.json',receipt)
                if receipt['state']!='STAGED_LOCAL':
                    raise RecoveryBlocked(receipt.get('failure','BACKGROUND_REPAIR_FAILED'),receipt.get('reason','background repair remains blocked'))
                return ToolCandidate(graph,Path(receipt['candidate_directory']))
            callbacks['switch_background_candidate']=background_retry
            bindings['switch_background_candidate']=bindings['retry_text_mask_completion']
    else:
        from product.recovery.asset_retry import retry_asset_extraction
        def asset_retry(directory):
            receipt=retry_asset_extraction(graph,identifier,source,layers,baseline,directory/'retry')
            write(directory/'action-result.json',receipt)
            if receipt['status']!='CANDIDATE_READY':
                raise RecoveryBlocked('ASSET_EXTRACTION_UNCERTAIN',receipt.get('reason','asset extraction remains blocked'))
            candidate_graph=SceneGraph.model_validate_json(Path(receipt['scene_file']).read_text('utf8'))
            return ToolCandidate(candidate_graph,Path(receipt['candidate_directory']),
                                 'TRANSPARENT_PNG',baseline,layers)
        callbacks['retry_asset_extraction']=asset_retry
        bindings['retry_asset_extraction']=key({name:digest(package/name) for name in (
            'recovery/asset_retry.py','parallel_rebuild.py','object_unmix.py')})
    tools=register_repair_tools(args.kind,callbacks,bindings)
    policy=load_skill(Path(args.skill).resolve())
    output.mkdir(parents=True,exist_ok=False)
    if failure_file is None:
        failure_file=output/'failure.json';write(failure_file,failure)
    write(output/'request.json',{'job':str(job),'layers':str(layers),'kind':args.kind,'object_id':identifier,
        'failure_file':str(failure_file),'failure_sha256':digest(failure_file),'scene_file':str(scene_file),
        'scene_sha256':key(graph.model_dump()),'source_sha256':graph.source_sha256,
        'layer_manifest_sha256':digest(layers/'manifest.json'),'skill_sha256':policy.version,
        'max_tools':1,'max_decisions':2,'tool_implementations':bindings,
        'auto_failure_selected':args.failure is None,
        'rejected_candidate':args.rejected_candidate,'rejection_report':args.rejection_report})
    if args.prompt_key:
        credential=getpass.getpass('MoMA credential (hidden): ')
        os.environ['MOMA_API_KEY']=credential;os.environ['PLANNER_API_KEY']=credential
    store=Store(output/'state');budget=AdaptiveBudget(store,{'model_calls':2})
    actor=BudgetedActor(Planner(settings()),budget)
    cache=Path(args.model_cache).resolve() if args.model_cache else output/'model-cache'
    gateway=ModelGateway(actor,cache)
    runtime=RegionalRepairAgent(graph,identifier,source,[failure_file],tools=tools,gateway=gateway,
        policy=policy,store=store,output=output/'agent',max_decisions=2,max_tools=1)
    selected=runtime.run()
    report={'schema':'R1205AgentRun/1','object_id':identifier,'kind':args.kind,'result':selected,
        'live_vlm_called':budget.report()['counts']['model_calls']>0,'budget':budget.report(),
        'office':'NOT_RUN','go':'NOT_RUN','automatic_publication':False}
    write(output/'result.json',report)
    print(json.dumps({'status':selected['status'],'object_id':identifier,
        'model_calls':report['budget']['counts']['model_calls'],'result':str(output/'result.json')}),flush=True)
    if selected['status']!='CANDIDATE_READY':raise SystemExit(1)


if __name__=='__main__':main()
