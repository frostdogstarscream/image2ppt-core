"""One source-bound icon, one changed strategy, cached L1 before any Office work."""
import json,time
from pathlib import Path
from ..materialize import digest,write
from ..parallel_store import key
from ..parallel_contracts import SceneGraph
from ..object_registry import SAM_BINDING
from .asset_retry import retry_asset_extraction
from .asset_cache import l1_binding, l2_binding


def probe(graph,identifier,source,layers,baseline,output,*,strategy='sam'):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    baseline=Path(baseline).resolve();layers=Path(layers).resolve()
    obj=next(a for a in graph.asset.objects if a.id==identifier)
    binding=l1_binding(graph,obj,source,layers,baseline,strategy,SAM_BINDING if strategy=='sam' else None)
    if binding['source_sha256']!=graph.source_sha256:raise ValueError('single-object source changed')
    root=output/'runs'/key(binding)[:16];receipt=root/'receipt.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text('utf8'))
        if saved['binding']!=binding or any(digest(p)!=sha for p,sha in saved['files'].items()):raise ValueError('single-object cache evidence changed')
        return {**saved['result'],'cache_hit':True,'model_calls':0,'receipt':str(receipt)}
    root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result={'object_id':identifier,'strategy':strategy,'status':'REVIEW_REQUIRED','cache_hit':False,
        'model_calls':0,'office':'NOT_RUN','go':'NOT_RUN','automatic_publication':False}
    try:
        samdir=plan=None
        from ..parallel_regions import regions
        if obj.kind!='icon' or graph.ownership.get(identifier)!='RESOLVED' or regions(graph)[1][identifier]['member_ids']!=[identifier]:
            raise ValueError('complete independent icon ownership required before tool dispatch')
        if strategy=='sam':
            from ..object_stage import infer_sam
            from ..object_extract import load_sam
            plan=root/'sam-plan.json'
            write(plan,{'schema':'LogicalObjects/2','source_sha256':graph.source_sha256,'image_size':graph.image_size,
                'objects':[obj.model_dump()],'limitations':[]})
            samdir=output/'sam'/key({'plan':digest(plan),'model':SAM_BINDING})[:20]
            masks,error=load_sam(samdir,obj,graph.source_sha256,digest(plan),tuple(graph.image_size),'sam3',individual=True)
            if error:
                samdir=infer_sam(source,plan,samdir,'sam3',[identifier]);result['model_calls']=1
            if samdir is None:raise RuntimeError('SAM_UNCONFIGURED')
        candidate=retry_asset_extraction(graph,identifier,source,layers,baseline,root/'c',strategy=strategy,sam_directory=samdir,sam_plan=plan)
        result.update(candidate)
        if result['status']=='CANDIDATE_READY':
            from .agent_decision import ToolCandidate,verify_candidate
            checked=verify_candidate(ToolCandidate(graph,Path(result['candidate_directory']),'TRANSPARENT_PNG',baseline,layers),graph,identifier,source,root)
            result['independent_checks']=checked
        else:result['failure']='OWNER_OR_RGBA_OR_BACKGROUND_UNPROVEN'
    except Exception as exc:
        result.update(status='UNKNOWN' if isinstance(exc,(OSError,TimeoutError,RuntimeError)) else 'BLOCKED',
            failure=type(exc).__name__,reason=str(exc)[:500])
    result.update(elapsed_seconds=round(time.monotonic()-start,3),candidate_state='GENERATED' if result.get('candidate_directory') else 'NOT_GENERATED')
    result['candidate_hashes']={p.name:digest(p) for p in Path(result['candidate_directory']).iterdir() if p.is_file()} if result.get('candidate_directory') else {}
    # Bind external model receipts too. A raw mask remains segmentation evidence.
    files={str(p.resolve()):digest(p) for p in root.rglob('*') if p.is_file()}
    if strategy=='sam' and samdir:
        files.update({str(p.resolve()):digest(p) for p in Path(samdir).rglob('*') if p.is_file()})
    write(receipt,{'binding':binding,'result':result,'files':files})
    previous=output/'latest.json';old=json.loads(previous.read_text('utf8')) if previous.exists() else None
    write(root/'difference.json',{'previous':old,'current':result,'input_sha256':key(binding),
        'code_sha256':key(binding['code']),'changed_candidate_files':None if old is None else
        [name for name,sha in result['candidate_hashes'].items() if old.get('candidate_hashes',{}).get(name)!=sha]})
    write(previous,result)
    return {**result,'receipt':str(receipt)}


def verify_office(result,graph,source,layers,baseline,fonts,output):
    from .agent_asset_stage import stage_asset_candidate
    from ..pipeline import compile_pages
    from ..parallel_pipeline import effective_texts
    from ..parallel_office import inspect
    from ..parallel_residual_gate import inspect_residuals
    from ..resource_lease import lease
    output=Path(output).resolve();binding=l2_binding(result,graph,source,layers,baseline,fonts)
    root=output/'office'/key(binding)[:16];receipt=root/'l2.json'
    if receipt.exists():
        record=json.loads(receipt.read_text('utf8'))
        if record['binding']!=binding or any(digest(p)!=sha for p,sha in record['files'].items()):raise ValueError('Office receipt evidence changed')
        return {**record['result'],'cache_hit':True}
    root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    ir=stage_asset_candidate(result['candidate_directory'],graph,baseline,root/'s',layer_directory=layers)
    composed=SceneGraph.model_validate_json((ir.parent/'scene.json').read_text('utf8'))
    old={t.id:t.model_dump() for t in graph.text.texts}
    if old!={t.id:t.model_dump() for t in composed.text.texts}:raise ValueError('staging changed prior text definitions')
    with lease('compiler'):artifacts=compile_pages([ir],root/'compiled',[effective_texts(composed,ir)])
    with lease('office'):office=inspect(artifacts['pptx'],[ir],root/'o',target_ids={result['object_id']})
    residuals=inspect_residuals(ir,root/'o',1,root/'residuals.json')
    target=next(r for r in residuals if r['id']==result['object_id'])
    status='PASS' if office['status']=='PASS_APP_AUTOMATED' and target['residual_check_status']=='PASS' else 'REVIEW_REQUIRED'
    record={'status':status,'object_id':result['object_id'],'ir':str(ir),'artifacts':artifacts,'office':office,
        'target_residual':target,'old_text_definitions_preserved':True,'elapsed_seconds':round(time.monotonic()-start,3),'go':'NOT_RUN','automatic_publication':False}
    write(receipt,{'binding':binding,'result':record,'files':{str(p):digest(p) for p in root.rglob('*') if p.is_file()}})
    return record
