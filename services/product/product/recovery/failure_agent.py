"""Failure-driven host routing into the existing bounded regional Agent."""
import json
from pathlib import Path
from ..materialize import digest,write
from ..parallel_store import key
from ..parallel_contracts import SceneGraph
from ..parallel_fusion import fuse
from ..agent_service.model_gateway import ModelGateway
from ..agent_service.skill_loader import load_skill
from .agent_decision import RegionalRepairAgent,ToolCandidate
from .agent_tool_catalog import register_repair_tools
from .background_adapter import RecoveryBlocked
from .text_actions import retry_text_mask_completion,switch_background_candidate
from .text_probe import probe_one


def route(failure,kind,files,*,already_attempted=()):
    """A retry needs a real bound failure and a strategy not already exhausted."""
    if failure in ('QA_INFRASTRUCTURE_UNAVAILABLE','TOOL_UNAVAILABLE'):
        return {'status':'UNKNOWN','reason':'external response unavailable; automatic retry disabled','tools':[]}
    reports=[]
    for path in files:
        if Path(path).name!='residual_report.json':continue
        report=json.loads(Path(path).read_text('utf8'))
        reports.append((Path(path),report))
    if kind=='asset':
        return {'status':'ELIGIBLE','tools':['retry_asset_extraction']} if not already_attempted else {
            'status':'NO_CHANGED_EVIDENCE','reason':'source-halo adapter already tried against these inputs','tools':[]}
    if failure in ('OWNER_CONFLICT','SEMANTIC_ROLE_CONFLICT'):
        return {'status':'ELIGIBLE','tools':['recheck_text_ownership']}
    for file,report in reports:
        if failure=='MASK_INCOMPLETE' and report['status']=='REVIEW_REQUIRED' and report['mask_variant'] in ('original','bounded_expansion') and not report['checks']['complete_visible_scope']:
            return {'status':'ELIGIBLE','tools':['retry_text_mask_completion'],'receipt':str(file)}
        if failure=='BACKGROUND_REPAIR_FAILED' and report['status']=='REVIEW_REQUIRED' and report['checks']['complete_visible_scope']:
            return {'status':'ELIGIBLE','tools':['switch_background_candidate'],'receipt':str(file)}
    return {'status':'NO_SAFE_TOOL','reason':'no partial-mask or background rejection supports a changed strategy','tools':[]}


def run(graph,identifier,kind,source,layers,baseline,fonts,files,output,*,failure,provider,store,cache,skill,
        attempted=(),inpaint=None,max_decisions=2):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plan=route(failure,kind,files,already_attempted=attempted)
    write(output/'route.json',{'failure':failure,'object_id':identifier,'source_sha256':graph.source_sha256,**plan})
    if plan['status']!='ELIGIBLE' or not skill:return {**plan,'status':plan['status'] if skill else 'UNKNOWN'}
    callbacks={};implementation={};package=Path(__file__).parent
    def candidate(receipt,current,directory):
        write(directory/'execution.json',receipt)
        if receipt.get('state')!='STAGED_LOCAL':raise RecoveryBlocked(receipt.get('failure','OWNER_CONFLICT'),receipt.get('reason','source evidence remains insufficient'))
        return ToolCandidate(current,Path(receipt['candidate_directory']),baseline_directory=Path(baseline))
    for tool in plan['tools']:
        if tool=='retry_text_mask_completion':
            def execute(directory):
                receipt=retry_text_mask_completion(graph,identifier,source,layers,baseline,fonts,directory/'p',previous_receipt=plan['receipt'],inpaint=inpaint)
                return candidate(receipt,graph,directory)
        elif tool=='switch_background_candidate':
            def execute(directory):
                receipt=switch_background_candidate(graph,identifier,source,layers,baseline,fonts,directory/'p',
                    rejected_candidate_directory=Path(plan['receipt']).parent,rejection_report=plan['receipt'],excluded_candidates=attempted,inpaint=inpaint)
                return candidate(receipt,graph,directory)
        elif tool=='recheck_text_ownership':
            def execute(directory):
                ocr=None
                for ref in graph.text.references:
                    path=Path(ref)
                    if not path.is_file():continue
                    value=json.loads(path.read_text('utf8'))
                    if value.get('schema')=='OCRLines/1' and value.get('source_sha256')==graph.source_sha256:ocr=value;break
                if ocr is None:raise RecoveryBlocked('OWNER_CONFLICT','independent OCR evidence unavailable')
                snap={'path':str(source),'source_sha256':graph.source_sha256,'image_size':graph.image_size,'transform':graph.transform}
                current=fuse(snap,{'text':graph.text,'asset':graph.asset,'surface':graph.surface},ocr)
                write(directory/'ownership-recheck.json',{'before':graph.ownership,'after':current.ownership,
                    'relationships_before':[r.model_dump() for r in graph.relations],
                    'relationships_after':[r.model_dump() for r in current.relations]})
                # Unchanged semantic Mask failures need new evidence, not an
                # expensive repeat of the same background or model strategy.
                if key(current.model_dump())==key(graph.model_dump()):raise RecoveryBlocked('OWNER_CONFLICT','ownership evidence unchanged')
                return candidate(probe_one(current,identifier,source,layers,baseline,fonts,directory/'p',inpaint=inpaint),current,directory)
        else:
            def execute(directory):
                from .asset_retry import retry_asset_extraction
                result=retry_asset_extraction(graph,identifier,source,layers,baseline,directory/'p')
                write(directory/'execution.json',result)
                if result['status']!='CANDIDATE_READY':raise RecoveryBlocked('OWNER_CONFLICT',result.get('reason','asset owner remains unresolved'))
                current=SceneGraph.model_validate_json(Path(result['scene_file']).read_text('utf8'))
                return ToolCandidate(current,Path(result['candidate_directory']),'TRANSPARENT_PNG',Path(baseline),Path(layers))
        callbacks[tool]=execute
        implementation[tool]=key({p.name:digest(p) for p in [Path(__file__),package/'text_actions.py',package/'text_probe.py',package/'asset_retry.py',package.parent/'peer_evidence.py',package.parent/'parallel_fusion.py']})
    tools=register_repair_tools(kind,callbacks,implementation)
    from PIL import Image
    obj=next(o for o in [*graph.text.texts,*graph.asset.objects] if o.id==identifier)
    x,y,w,h=map(round,obj.bbox)
    with Image.open(source) as image:
        image.crop((max(0,x-16),max(0,y-16),min(image.width,x+w+16),min(image.height,y+h+16))).save(output/'source-crop.png')
    diagnostics=[output/'source-crop.png']
    diagnostics.extend(Path(p).with_suffix('.png') for p in files if Path(p).name=='peer-protection.json' and Path(p).with_suffix('.png').is_file())
    session=RegionalRepairAgent(graph,identifier,source,files,tools=tools,
        gateway=ModelGateway(provider,Path(cache)/'model-calls'),policy=load_skill(skill),store=store,
        output=output/'a',max_decisions=max_decisions,max_tools=1,diagnostic_images=diagnostics)
    return session.run()
