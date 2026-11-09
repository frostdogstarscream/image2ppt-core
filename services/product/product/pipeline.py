"""Background pipeline: images -> model plan -> fixed compiler -> visual review."""
from __future__ import annotations
import asyncio
import base64
import hashlib
from dataclasses import replace
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
for directory in ('services/planner/src','services/vision/src','services/windows'):
    sys.path.insert(0,str(ROOT/directory))
from image2ppt_planner.config import Settings
from image2ppt_planner.client import MomaClient, strip_code_fence
from image2ppt_vision.conversion import _compiler, _final_json, _compiled_artifacts
from .contracts import Page, Review
from .fonts import font_manifest
from .materialize import digest, write, materialize, verify_package
from . import prompts
from .visual_checks import compare_text,compare_graphics
from .crop_fit import complete_crops


def effective_findings(review):
    """Only a pure font-slant difference has the user's explicit exemption."""
    return [f.model_dump() for f in review.findings
            if not (f.category=='font' and f.aspect=='italic_only')]


class PipelineError(RuntimeError):
    pass


def settings():
    cfg=Settings.from_env()
    if not cfg.api_key and os.name=='nt':
        try:
            from credentials import load
            cfg=replace(cfg,api_key=load('moma'))
        except (OSError,RuntimeError,ValueError):
            pass
    cfg.require_key()
    if not cfg.base_url.startswith('https://'):
        raise PipelineError('model_endpoint_requires_https')
    return replace(cfg,transport_attempts=1,max_attempts=1)


def image_part(path):
    return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+
                                           base64.b64encode(Path(path).read_bytes()).decode('ascii')}}


class Provider:
    def __init__(self, cfg):
        self.cfg=cfg
        self.model=cfg.model

    def ask(self, system, context, images, contract):
        messages=[{'role':'system','content':system},
                  {'role':'user','content':[{'type':'text','text':json.dumps(
                      {**context,'output_schema':contract.model_json_schema()},ensure_ascii=False)},
                      *[image_part(p) for p in images]]}]
        async def request():
            return await asyncio.wait_for(MomaClient(self.cfg).chat(messages),self.cfg.timeout_s+10)
        try:
            reply=asyncio.run(request())
        except Exception as exc:
            # Upstream errors can echo request bodies/credentials; never propagate them.
            message=str(exc)
            status=re.search(r'HTTP\s+(\d{3})',message)
            if status:code='model_http_'+status.group(1)
            elif isinstance(exc,(TimeoutError,asyncio.TimeoutError)) or 'Timeout' in message:code='model_timeout'
            elif 'ConnectError' in message:code='model_connect_error'
            else:code='model_request_failed'
            raise PipelineError(code) from None
        if self.cfg.api_key in reply.text:
            raise PipelineError('credential_echo_rejected')
        if reply.finish_reason not in ('stop',None):
            raise ValueError('model response truncated')
        if len(reply.text)>2_000_000:
            raise ValueError('model response too large')
        try:
            return json.loads(strip_code_fence(reply.text))
        except (ValueError,RecursionError):
            raise ValueError('model response is not JSON') from None


def compile_pages(irs, output, pages):
    compiler,node=_compiler(ROOT)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'input-irs.json',[str(Path(p).resolve()) for p in irs])
    command=[str(node),str(ROOT/'services/product/compose_objects.mjs'),str((output/'input-irs.json').resolve()),str(output.resolve())]
    with (output/'compiler.stdout').open('w',encoding='utf-8') as stdout, (output/'compiler.stderr').open('w',encoding='utf-8') as stderr:
        try:
            result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=1800,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except subprocess.TimeoutExpired:
            raise PipelineError('compiler_timeout') from None
    if result.returncode:
        raise PipelineError('fixed_compiler_failed')
    artifacts=_compiled_artifacts(_final_json((output/'compiler.stdout').read_bytes()),output,len(pages))
    verify_package(artifacts['pptx'],pages)
    return artifacts


def run(job, *, provider=None, review_provider=None, compiler=compile_pages, manifest=None, progress=lambda **kw:None):
    """No paid-call quota. Stop when accepted, cancelled, or repeatedly stagnant."""
    job=Path(job)
    request=json.loads((job/'request.json').read_text(encoding='utf-8'))
    from .object_stage import route_setting
    object_route=route_setting(request.get('object_extraction','default'))
    human_review=request.get('delivery_for_human_review') is True
    sources=[job/'inputs'/name for name in request['sources']]
    provider=provider or Provider(settings())
    if review_provider is None:
        from .go_provider import GoProvider
        review_provider=GoProvider(job)
    write(job/'provenance.json',{'model':provider.model,
        'endpoint':getattr(getattr(provider,'cfg',None),'base_url',None),
        'planning_prompt_sha256':hashlib.sha256(prompts.PLAN.encode()).hexdigest(),
        'review_prompt_sha256':hashlib.sha256(prompts.REVIEW.encode()).hexdigest(),
        'sources':[{'file':p.name,'sha256':digest(p)} for p in sources],
        'review_model':review_provider.model,'planning_provider':getattr(provider,'provider_name','moma'),'review_provider':'opencode_go',
        'italic_difference_ignored':True,'model_calls_per_page_limit':None,
        'codex_required_at_runtime':getattr(provider,'codex_required_at_runtime',False)})
    manifest=manifest or font_manifest([])
    write(job/'fonts.json',manifest)
    font_choices=[{k:f[k] for k in ('family','bold','cjk')} for f in manifest['fonts']]
    plans=[None]*len(sources);drafts=[None]*len(sources);irs=[None]*len(sources);feedback=[None]*len(sources)
    if human_review and len(sources)==1 and request.get('initial_plan') is not None:
        drafts[0]=request['initial_plan'];feedback[0]=request.get('initial_feedback')
    last_validation=[None]*len(sources)
    repeated_validation=[0]*len(sources)
    last_previews=[None]*len(sources)
    last_details=[[] for _ in sources]
    pending=set(range(len(sources)))
    calls=[0]*len(sources)
    best=[float('inf')]*len(sources);stale=[0]*len(sources)
    ledger=[]
    object_reports=[None]*len(sources);parent_irs=[None]*len(sources);asset_references=[{} for _ in sources]
    def ask(i,kind,context,images,contract,actor=None):
        actor=actor or provider
        calls[i]+=1
        ledger.append({'page':i+1,'kind':kind,'model':actor.model,'attempt':calls[i],'state':'RESERVED','started_at':time.time()})
        # Reserve before network use; a crash never silently refunds the attempt.
        write(job/'calls.json',ledger)
        progress(stage=kind,page=i+1,model_calls=sum(calls),max_model_calls=None)
        try:
            from .object_contracts import OBJECT_PROMPT,CLOSED_OBJECT_PROMPT,ClosedPlan,ObjectRepairs,OBJECT_REVIEW_PROMPT
            system={'planning':prompts.PLAN,'reviewing':prompts.REVIEW,'object_planning':CLOSED_OBJECT_PROMPT if contract in (ClosedPlan,ObjectRepairs) else OBJECT_PROMPT,
                    'object_reviewing':OBJECT_REVIEW_PROMPT}[kind]
            result=actor.ask(system,context,images,contract)
        except Exception:
            ledger[-1].update(state='FAILED',finished_at=time.time());write(job/'calls.json',ledger)
            raise
        ledger[-1].update(state='RECEIVED',finished_at=time.time());write(job/'calls.json',ledger)
        return result
    round_no=0
    while True:
        if (job/'cancel.requested').exists():raise PipelineError('cancelled')
        revision=job/f'round-{round_no+1:02d}'
        revision.mkdir()
        for i in sorted(pending):
            source=sources[i]
            with Image.open(source) as image: size=list(image.size)
            context={'source_sha256':digest(source),'image_size':size,'fonts':font_choices,
                     'previous_plan':drafts[i],'feedback':feedback[i]}
            try:
                raw=ask(i,'planning',context,[source,*([last_previews[i]] if last_previews[i] else []),*last_details[i]],Page)
                drafts[i]=raw
                write(revision/f'page-{i+1:02d}-proposal.json',raw)
                plan=Page.model_validate(raw)
                plan,crop_changes=complete_crops(plan,source)
                if asset_references[i] and object_reports[i] and object_reports[i].get('publication')=='STABLE':
                    from .object_registry import pin_graphics
                    plan=pin_graphics(plan,plans[i],object_reports[i]['objects'])
                write(revision/f'page-{i+1:02d}-crop-fit.json',crop_changes)
                drafts[i]=plan.model_dump()
                ir=materialize(plan,source,revision/f'page-{i+1:02d}',manifest,provider.model,
                    **({'asset_references':asset_references[i]} if asset_references[i] else {}))
                parent_irs[i]=ir
                if object_route!='off':
                    from .object_stage import process,obtain_layers
                    try:
                        layerdir=obtain_layers(source,job/f'layerd-page-{i+1:02d}')
                        ir,plan,object_report=process(source,plan,ir,revision/f'objects-{i+1:02d}',object_route,
                            lambda context,images,contract:ask(i,'object_planning',context,images,contract),layerdir=layerdir,
                            state_directory=job/f'object-state-{i+1:02d}')
                    except (ValueError,RuntimeError,OSError,KeyError,subprocess.SubprocessError) as exc:
                        if object_route!='auto':raise
                        ir=parent_irs[i]
                        object_report={'schema':'ObjectClosure/2','objects':[],'staged':0,'fallback':None,
                            'publication':'RETAINED_PARENT','automatic_quality':'REVIEW_REQUIRED','error_type':type(exc).__name__}
                        plan=plan.model_copy(update={'limitations':[*plan.limitations,'Object service unavailable or evidence invalid; reliable parent graphics retained.']})
                    object_reports[i]=object_report
                    write(revision/f'page-{i+1:02d}-object-report.json',object_report)
                plans[i]=plan;irs[i]=ir;feedback[i]=None
                last_validation[i]=None;repeated_validation[i]=0
            except ValueError as exc:
                # Pydantic's full exception may include source text. Keep bounded feedback private.
                feedback[i]=str(exc)[:6000]
                write(revision/f'page-{i+1:02d}-validation.json',{'error':feedback[i]})
                signature=feedback[i].split('[type=')[0]
                repeated_validation[i]=repeated_validation[i]+1 if signature==last_validation[i] else 1
                last_validation[i]=signature
                # Fixing one violation and exposing another is progress, not stagnation.
                stale[i]=repeated_validation[i]
        pending={i for i in range(len(sources)) if feedback[i] is not None or plans[i] is None}
        round_no+=1
        if pending:
            if any(stale[i]>=3 for i in pending):break
            continue
        progress(stage='compiling',round=round_no,model_calls=sum(calls),max_model_calls=None)
        if object_route=='auto':
            from .object_finalize import finalize,review_objects
            def repair_objects(i,report,verdicts,path):
                from .object_stage import process,effective_page
                feedback={'source_sha256':digest(sources[i]),'scene_file':str(irs[i]),'scene_sha256':digest(irs[i]),'objects':verdicts}
                return process(sources[i],effective_page(plans[i],parent_irs[i]),parent_irs[i],path,'auto',
                    lambda context,images,contract:ask(i,'object_planning',context,images,contract),
                    layerdir=report['layer_directory'],state_directory=report['state_directory'],review_feedback=feedback)
            artifacts,irs,plans=finalize(sources,irs,parent_irs,object_reports,plans,revision/'object-gates',compiler,
                lambda i,source,preview,ir,report,path:review_objects(source,preview,ir,report,path,
                    lambda context,images,contract:ask(i,'object_reviewing',context,images,contract,actor=review_provider),review_model=review_provider.model),repair=repair_objects)
            from .object_registry import asset_refs
            for i,r in enumerate(object_reports):
                if r and r.get('publication')=='STABLE':asset_references[i]=asset_refs(irs[i])
        else:artifacts=compiler(irs,revision/'build',plans)
        previews=sorted(Path(artifacts['preview']).glob('*.png'))
        if len(previews)!=len(sources): raise PipelineError('preview_count_mismatch')
        review_complete=set()
        for i,(source,preview) in enumerate(zip(sources,previews)):
            last_previews[i]=preview
            planfile=revision/f'page-{i+1:02d}-effective.json'
            write(planfile,plans[i].model_dump())
            binding={'source_sha256':digest(source),'plan_sha256':digest(planfile),'preview_sha256':digest(preview)}
            try:
                metrics,detail=compare_text(source,preview,plans[i],revision/f'page-{i+1:02d}-detail.png')
                graphics=compare_graphics(source,preview,plans[i],revision/f'page-{i+1:02d}-graphics.png')
                last_details[i]=[p for p in (detail,graphics) if p]
                write(revision/f'page-{i+1:02d}-ink.json',metrics)
                raw=ask(i,'reviewing',{**binding,'plan':plans[i].model_dump(),'ink_metrics':metrics},
                        [source,preview,*([detail] if detail else []),*([graphics] if graphics else [])],Review,actor=review_provider)
                review=Review.model_validate(raw)
                if any(getattr(review,k)!=v for k,v in binding.items()):
                    raise ValueError('visual review binding mismatch')
                node_ids={n.id for n in [*plans[i].texts,*plans[i].shapes,*plans[i].crops]}
                if any(not set(f.node_ids)<=node_ids for f in review.findings):
                    raise ValueError('visual review referenced unknown object IDs')
                write(revision/f'page-{i+1:02d}-review.json',review.model_dump())
                findings=effective_findings(review)
                small={}
                for m in metrics:
                    if m['width_ratio']<.84 and m['height_ratio']<.84:
                        small.setdefault(m['style'],[]).append(m['id'])
                for style,members in small.items():
                    if len(members)>=2:findings.append({'category':'font','aspect':'other','node_ids':members,
                        'description':'同组多个文字的实际墨迹宽度和高度均小于原图84%，请按组恢复字号；style='+style})
                write(revision/f'page-{i+1:02d}-decision.json',{'accepted':not findings and not plans[i].limitations,
                    'italic_difference_ignored':True,'active_findings':findings})
                review_complete.add(i)
                if findings or plans[i].limitations:
                    feedback[i]={'findings':findings,'ink_metrics':metrics,'limitations':plans[i].limitations}
                    pending.add(i)
                    score=len(findings)+len(plans[i].limitations)
                    stale[i]=1 if score<best[i] else stale[i]+1
                    best[i]=min(best[i],score)
                else:stale[i]=0
            except (ValueError,RuntimeError,OSError) as exc:
                if not isinstance(exc,ValueError) and object_route!='auto':raise
                feedback[i]=str(exc)[:6000];pending.add(i);stale[i]+=1
        if not pending or object_route=='auto' or (human_review and len(review_complete)==len(sources)):
            delivery=job/'delivery';delivery.mkdir()
            shutil.copyfile(artifacts['pptx'],delivery/'result.pptx')
            for i,p in enumerate(previews,1):shutil.copyfile(p,delivery/f'page-{i:02d}.png')
            object_unresolved={i for i,r in enumerate(object_reports) if object_route=='auto' and
                (not r or r.get('publication')!='STABLE' or r.get('inventory_missing_owner_ids') or
                 any(o.get('status')=='FALLBACK' for o in r.get('objects',[])))}
            needs_review=human_review or bool(pending|object_unresolved)
            result={'status':'REVIEW_REQUIRED' if needs_review else 'SUCCEEDED',
                    'assessment':'HUMAN_REVIEW_DRAFT' if needs_review else 'AUTOMATED_VISUAL_PASS','model':provider.model,
                    'review_model':review_provider.model,'italic_difference_ignored':True,
                    'model_calls':sum(calls),'pages':len(plans),'human_acceptance':False,
                    'powerpoint_checked':bool(artifacts.get('powerpoint_checked')),'wps_checked':False,
                    'object_extraction':object_route,
                    'unresolved_pages':sorted(i+1 for i in pending|object_unresolved),
                    'effective_plans':[str(revision/f'page-{i+1:02d}-effective.json') for i in range(len(plans))],
                    'slide_irs':[str(p) for p in irs],
                    'artifacts':{p.name:digest(p) for p in delivery.iterdir()}}
            write(job/'result.json',result)
            return result
        if any(stale[i]>=3 for i in pending):break
    result={'status':'QUALITY_FAILED','model_calls':sum(calls),'pages':len(sources),
            'unresolved_pages':sorted(i+1 for i in pending),'stop_reason':'no_progress',
            'italic_difference_ignored':True,'artifacts':{},'human_acceptance':False}
    write(job/'result.json',result)
    return result


if __name__=='__main__':
    job=Path(sys.argv[1]).resolve()
    try:
        run(job,progress=lambda **kw:write(job/'progress.json',kw))
    except Exception as exc:
        code=str(exc) if isinstance(exc,PipelineError) else type(exc).__name__
        write(job/'result.json',{'status':'FAILED','error':code,'artifacts':{}})
        sys.exit(1)
