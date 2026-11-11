"""R12 regional orchestration used directly by the three-branch product worker."""
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from ..contracts import Review
from ..materialize import digest,write
from ..parallel_contracts import SceneGraph
from ..parallel_pipeline import effective_texts
from ..parallel_regions import regions
from ..parallel_review import PROMPT,panels
from ..parallel_store import key
from ..parallel_office import inspect
from ..parallel_edit_probe import inspect_text
from ..parallel_residual_gate import inspect_residuals
from ..resource_lease import lease
from .background_adapter import RecoveryBlocked
from .region_candidates import generate, adaptive_candidates
from .adaptive_policy import AdaptiveBudget, QualityReview, QUALITY_PROMPT, quality_status
from .font_candidates import compare
from .region_state import rollback
from .validation import stage_and_verify
from .lama_adapter import LamaClient
from .remote_gpu_lease import remote_gpu_lease


def failure(exc,stage):
    code=getattr(exc,'code','')
    if 'native font frame allowance' in str(exc).lower():return 'FONT_LAYOUT_FAILED'
    if isinstance(exc,(TimeoutError,OSError)) or type(exc).__name__ in ('BudgetExceeded','UnknownCall'):
        return 'QA_INFRASTRUCTURE_UNAVAILABLE'
    if code in ('QA_INFRASTRUCTURE_UNAVAILABLE','INPAINT_RESPONSE_UNKNOWN'):return 'QA_INFRASTRUCTURE_UNAVAILABLE'
    if code in ('MASK_INCOMPLETE','INVALID_MASK','EMPTY_WRITE_MASK'):return 'MASK_INCOMPLETE'
    if code in ('OWNER_CONFLICT','PROTECTED_NEIGHBOR','MASK_OUTSIDE_SCOPE') or any(
        word in str(exc).lower() for word in ('owner','indivisible','source-authorized','protected peer')):return 'OWNER_CONFLICT'
    if code=='OFFICE_VALIDATION_FAILED':return code
    return {'generation':'BACKGROUND_REPAIR_FAILED','font':'FONT_LAYOUT_FAILED',
            'office':'OFFICE_VALIDATION_FAILED','qa':'QA_INFRASTRUCTURE_UNAVAILABLE'}.get(stage,'REGION_FAILED')


def office_frame_feedback(output,identifier):
    file=output/'replacement/report.json'
    if not file.exists():return None
    report=json.loads(file.read_text('utf8'))
    rows=[r for r in report['records'] if r['action'] in ('replace-short','replace-equal')]
    if not rows or any(not r['content_matches'] or not r['siblings_unchanged'] or not r['frame_unchanged'] or r['outside_pixels']
                       or r['layout']['unexpected_wrap'] for r in rows):return None
    scene=SceneGraph.model_validate_json((output/'region/scene.json').read_text('utf8'))
    text=next(t for t in scene.text.texts if t.id==identifier)
    sizes=[]
    for row in rows:
        left,top,width,height=row['layout']['frame'];bl,bt,bw,bh=row['layout']['text_bounds']
        if bl<left-1 or bt<top-1:return None
        sizes.append([math.ceil((bl+bw-left)*text.bbox[2]/width)+1,
                      math.ceil((bt+bh-top)*text.bbox[3]/height)+1])
    return {'size':[max(s[0] for s in sizes),max(s[1] for s in sizes)],
            'office_report_sha256':digest(file),'method':'actual-short-and-equal-replacement-bounds/1'}


class InpaintService:
    def __init__(self,config,cache,budget=None):
        def dispatch(identity):
            if budget and not budget.reserve('model_calls',{'tool':'big-lama','request_id':identity}):
                from ..parallel_store import BudgetExceeded
                raise BudgetExceeded('identical LaMa dispatch already reserved')
        self.config=config;self.client=LamaClient(config['url'],config['model_binding'],cache,on_dispatch=dispatch)
        self.receipt=None
    def __call__(self,source,mask):
        with remote_gpu_lease(self.config['gpu_lease']) as receipt:
            if receipt['status']!='ACQUIRED':raise RecoveryBlocked('GPU_LEASE_UNCONFIGURED','GPU lease unavailable')
            output=self.client(source,mask)
        self.receipt={'request':self.client.receipt,'gpu_lease':receipt}
        return output


class UnavailableReviewer:
    model='opencode-go-unconfigured'
    def ask(self,*args):
        raise RuntimeError('Go review infrastructure is unconfigured')


def review(store,reviewer,snap,ir,preview,office_dir,page,output,*,adaptive=False):
    details,refs=panels(snap['path'],preview,ir,office_dir,page,output)
    context={'source_sha256':snap['source_sha256'],'plan_sha256':digest(ir),'preview_sha256':digest(preview),
        'scene':json.loads(ir.read_text('utf8')),'transactions':json.loads((ir.parent/'transactions.json').read_text('utf8')),
        'actual_edit_evidence':refs,'office_pass':True,'raster_remainder_is_explicit_fallback':True}
    result=store.call('r12-final-review',reviewer,PROMPT+('\n'+QUALITY_PROMPT if adaptive else ''),context,
        [snap['path'],str(preview),*details],QualityReview if adaptive else Review,lease('go'))
    if any(getattr(result,k)!=context[k] for k in ('source_sha256','plan_sha256','preview_sha256')):
        raise ValueError('independent review binding mismatch')
    return result.model_dump()


def select_adaptive(graph,identifier,snap,source,saved,folder,current,fonts,font_choices,record,budget,inpaint,compiler,office):
    """Advance only on a classified defect, and stop at the first local pass."""
    text=next(t for t in graph.text.texts if t.id==identifier)
    # The upstream page-level calibration has already matched installed fonts.
    # Reuse that decision and its emphasis spans instead of searching again.
    proposals=[font_choices.get(text.style)]
    stream=adaptive_candidates(graph,identifier,snap['path'],saved,folder/identifier,inpaint=inpaint,budget=budget,baseline_directory=current.parent)
    try:
        for number,row in enumerate(stream):
            if row['status']!='PRECHECK_PASS':
                record['attempts'].append({'background':row['candidate'],'failure':'BACKGROUND_REPAIR_FAILED',
                                           'checks':row['checks']})
                continue
            trial=folder/identifier/row['candidate']
            proposal=proposals[0]
            output=folder/(identifier+'-candidate-'+str(number))
            for repair_round in range(int(budget.limits['repair_rounds'])+1):
                try:
                    options={'typography':proposal,'compiler':compiler,'office_probe':office,'include_long':False}
                    if repair_round:options['frame_minimum']=feedback
                    acceptance=stage_and_verify(trial,graph,current.parent,fonts,output,office=True,**options)
                    if acceptance.get('regional_automatic_checks')=='PASS':
                        actual=np.array(Image.open(output/'office/page-01-before.png').convert('RGB'))
                        bx,by,bw,bh=row['reference_context']['bbox']
                        score=float(np.abs(actual[by:by+bh,bx:bx+bw].astype(float)-source[by:by+bh,bx:bx+bw]).mean())
                        record['attempts'].append({'font':proposal,'background':row['candidate'],
                            'status':'VERIFIED_LOCAL','source_ppt_mae':score,'acceptance':str(output/'acceptance.json')})
                        record['early_stop']={'reason':'hard gates and real editing passed',
                            'background_candidates_consumed':number+1,'further_candidates_requested':False}
                        return score,output/'region/slideir.json',output/'acceptance.json',proposal,row['candidate']
                    cause=acceptance.get('failure','FONT_LAYOUT_FAILED')
                    record['attempts'].append({'background':row['candidate'],'failure':cause,'acceptance':str(output/'acceptance.json')})
                    # New frame parameters must come from actual failed Office
                    # bounds. No blind repeats of the same strategy are allowed.
                    feedback=office_frame_feedback(output,identifier) if cause=='FONT_LAYOUT_FAILED' else None
                    if not feedback:break
                    record['attempts'][-1]['office_frame_feedback']=feedback
                    output=folder/(identifier+'-candidate-'+str(number)+'-repair-'+str(repair_round+1))
                except Exception as exc:
                    code=failure(exc,'font')
                    record['attempts'].append({'failure':code,'error_type':type(exc).__name__,'reason':str(exc)[:400]})
                    if code in ('OWNER_CONFLICT','MASK_INCOMPLETE','QA_INFRASTRUCTURE_UNAVAILABLE'):
                        record['failure']=code
                        return None
                    break
            if record['attempts'][-1].get('failure') not in ('BACKGROUND_REPAIR_FAILED',):
                # Another background cannot repair content, font layout or an
                # unavailable Office installation. Keep that failure specific.
                break
        index=folder/identifier/'index.json'
        background= json.loads(index.read_text('utf8')) if index.is_file() else {}
        record['background_failures']=background.get('failures',[])
        unavailable=any(r.get('status')=='UNKNOWN' for r in record['background_failures'])
        record['failure']='QA_INFRASTRUCTURE_UNAVAILABLE' if unavailable else next(
            (a['failure'] for a in reversed(record['attempts']) if 'failure' in a),'BACKGROUND_REPAIR_FAILED')
        return None
    finally:
        stream.close()


def expand_assets(graph, current, source, layers, folder, fonts, compiler, office, failure_router=None):
    """Stage independent complete icons sequentially, then batch changed-object QA."""
    from .asset_retry import retry_asset_extraction
    from .agent_asset_stage import stage_asset_candidate
    rows=[];added=[]
    for number,obj in enumerate(graph.asset.objects):
        latest=SceneGraph.model_validate_json((current.parent/'scene.json').read_text('utf8'))
        if obj.id in latest.selected or obj.kind!='icon':continue
        root=folder/'ax'/str(number)
        row={'object_id':obj.id,'state':'LOCAL_FALLBACK','office':'NOT_RUN'}
        rows.append(row)
        try:
            result=retry_asset_extraction(latest,obj.id,source,layers,current.parent,root/'c')
            row.update(result)
            if result['status']!='CANDIDATE_READY':
                if failure_router:row['agent_route']=failure_router(latest,obj.id,result,root)
                continue
            current=stage_asset_candidate(result['candidate_directory'],latest,current.parent,root/'s',layer_directory=layers)
            row.update(state='STAGED_LOCAL',ir=str(current))
            added.append(obj.id)
        except Exception as exc:
            row.update(failure=failure(exc,'generation'),reason=str(exc)[:400])
    if added:
        latest=SceneGraph.model_validate_json((current.parent/'scene.json').read_text('utf8'))
        try:
            with lease('compiler'):
                artifacts=compiler([current],folder/'ax-package',[effective_texts(latest,current)])
            with lease('office'):
                receipt=(office or inspect)(artifacts['pptx'],[current],folder/'ax-office',target_ids=added)
            residuals=inspect_residuals(current,folder/'ax-office',1,folder/'ax-residuals.json')
            bad={r['id'] for r in residuals if r['id'] in added and r['residual_check_status']=='FAIL'}
            unknown={r['id'] for r in residuals if r['id'] in added and r['residual_check_status']=='UNKNOWN'}
            unknown |= set(added)-{r['id'] for r in residuals}
            if receipt.get('status')=='UNKNOWN':
                raise RecoveryBlocked('QA_INFRASTRUCTURE_UNAVAILABLE','new asset Office batch unavailable')
            if receipt.get('status')!='PASS_APP_AUTOMATED':raise ValueError('new asset Office batch failed')
            for row in rows:
                if row['object_id'] in added:
                    if row['object_id'] in unknown:
                        row.update(state='STAGED_LOCAL',office='UNKNOWN',failure='QA_INFRASTRUCTURE_UNAVAILABLE')
                    else:
                        row.update(state='VERIFIED_LOCAL' if row['object_id'] not in bad else 'LOCAL_FALLBACK',office='PASS' if row['object_id'] not in bad else 'FAIL')
            if bad:current=rollback(graph,current.parent,bad,folder/'ax-rollback','OFFICE_VALIDATION_FAILED')
        except Exception as exc:
            affected={getattr(exc,'object_id',None)} & set(added)
            if affected:
                current=rollback(graph,current.parent,affected,folder/'ax-rollback','OFFICE_VALIDATION_FAILED')
            for row in rows:
                if row['object_id'] in added:
                    row.update(office='UNKNOWN',failure=failure(exc,'office'),reason=str(exc)[:400])
                    if row['object_id'] in affected:row['state']='LOCAL_FALLBACK'
    write(folder/'asset-expansion.json',rows)
    return current,rows


def run(job,request,snaps,graphs,irs,baseline,fonts,store,expected,*,compiler,reviewer,repair_provider=None,office=None,progress=None,shadow=False):
    from ..pipeline import effective_findings
    adaptive=request.get('regional_strategy','adaptive')=='adaptive'
    if request.get('regional_strategy','adaptive') not in ('adaptive','exhaustive'):
        raise ValueError('unsupported regional strategy')
    budget=AdaptiveBudget(store,request.get('adaptive_budget')) if adaptive else None
    goal=request.get('text_region_goal')
    if goal is not None and (type(goal) is not int or goal<1):raise ValueError('text region goal must be a positive integer')
    root=job/'r12';root.mkdir(exist_ok=True)
    attempt=root/('attempt-'+key({'request':request,'code':digest(__file__)})[:16])
    if attempt.exists():
        from uuid import uuid4
        attempt=attempt.with_name(attempt.name+'-'+uuid4().hex[:8])
    attempt.mkdir()
    package=Path(__file__).resolve().parents[1]
    code_files=[*package.glob('*.py'),*Path(__file__).parent.glob('*.py'),*list((package/'agent_service').glob('*.py'))]
    write(attempt/'code-manifest.json',{'entrypoint':'product.parallel_pipeline.run',
        'files':{str(p.relative_to(package.parent)):digest(p) for p in code_files},
        'candidate_engine':'r12-shadow' if shadow else 'r12-regional'})
    config_file=Path(__file__).resolve().parents[4]/'.runtime/parallel-services.json'
    config=json.loads(config_file.read_text('utf8')).get('lama') if config_file.exists() else None
    shared_cache=Path(os.environ.get('IMAGE2PPT_PARALLEL_CACHE',str(Path(__file__).resolve().parents[4]/'.runtime/parallel-cache')))
    inpaint=InpaintService(config,shared_cache/'r12-lama',budget) if config and not shadow else None
    states=[];original_irs=list(irs);all_regions=[]
    for index,(graph,snap,ir) in enumerate(zip(graphs,snaps,irs)):
        graph=SceneGraph.model_validate(graph.model_dump());graphs[index]=graph
        folder=attempt/f'page-{index+1:02d}';folder.mkdir()
        write(folder/'source-scene.json',graph.model_dump())
        requested=request.get('regional_object_ids')
        targets=[t.id for t in graph.text.texts if (requested is not None and t.id in requested)
                 or (requested is None and graph.ownership[t.id]=='RESOLVED')]
        source=np.array(Image.open(snap['path']).convert('RGB'))
        with lease('cpu'):typography={} if adaptive else compare(graph,set(targets),source,fonts,folder)
        saved=next((p['layers'] for p in baseline['pages'] if p['source']['sha256']==snap['source_sha256']),None)
        if saved is None:
            try:
                from ..parallel_branches import layer_evidence
                saved,_=layer_evidence(snap,root/'layer-cache')
            except Exception:saved=None
        agent_count=0
        current=ir;state=[];font_choices={};region_records,by_id=regions(graph)
        seed_file=job/f'page-{index+1:02d}'/'verified-seed.json'
        if seed_file.is_file():
            seed=json.loads(seed_file.read_text('utf8'))
            for prior in seed['regions']:
                reused=dict(prior)
                reused.update(ir=str(ir),reused_from_verified_delivery=seed['source_job'],
                              quality='PENDING_FINAL_REVIEW',attempts=[])
                state.append(reused)
                font_choices[next(t.style for t in graph.text.texts if t.id==prior['object_id'])]=prior.get('font')
            targets=[identifier for identifier in targets if identifier not in {r['object_id'] for r in state}]
        if request.get('retry_assets') and saved and not shadow:
            def route_asset_failure(latest,identifier,result,root):
                from .failure_agent import run as run_failure_agent
                return run_failure_agent(latest,identifier,'asset',snap['path'],saved,
                    Path(result['baseline_directory']),fonts,[root/'c/result.json'],root/'a',
                    failure='ASSET_EXTRACTION_UNCERTAIN',provider=repair_provider,store=store,cache=shared_cache,
                    skill=request.get('agent_skill') or os.environ.get('IMAGE2PPT_AGENT_SKILL'),attempted=('retry_asset_extraction',))
            router=route_asset_failure if request.get('regional_agent') and repair_provider else None
            current,asset_rows=expand_assets(graph,current,snap['path'],saved,folder,fonts,compiler,office,failure_router=router)
        write(folder/'dependencies.json',region_records)
        for identifier in targets:
            if goal is not None and sum(r['state']=='VERIFIED_LOCAL' for r in state)>=goal:
                write(folder/'coverage-stop.json',{'goal':goal,'reason':'requested verified regional coverage reached','remaining_regions_retained_in_source':True})
                break
            record={'object_id':identifier,'region_id':by_id[identifier]['region_id'],
                    'dependencies':by_id[identifier]['dependencies'],'state':'LOCAL_FALLBACK','attempts':[],
                    'candidate_state':'NOT_GENERATED','regional_commit':'NOT_COMMITTED'}
            state.append(record)
            binding=None;parent_ir_sha=digest(current)
            try:
                binding=key({'scene':graph.model_dump(),'object':identifier,'fonts':fonts,
                    'engine':digest(__file__),'generation':digest(Path(__file__).with_name('region_candidates.py')),
                    'stage':digest(Path(__file__).with_name('candidate_stage.py')),
                    'validation':digest(Path(__file__).with_name('validation.py')),
                    'adaptive_policy':digest(Path(__file__).with_name('adaptive_policy.py')),
                    'code_binding':key({str(p):digest(p) for p in code_files}),
                    'policy':request.get('regional_strategy','adaptive'),'budget':request.get('adaptive_budget'),
                    'retry_revision':request.get('retry_revision'),'shadow':shadow})
                with store.db() as db:
                    prior=db.execute('SELECT payload FROM tasks WHERE k=?',(record['region_id'],)).fetchone()
                if prior and not shadow:
                    prior=json.loads(prior[0])
                    references=prior.get('verified_files',{})
                    if (prior.get('reconstruction_binding')==binding and prior.get('parent_ir_sha256')==digest(current)
                            and references and all(Path(p).is_file() and digest(p)==sha for p,sha in references.items())):
                        if prior['state']=='LOCAL_FALLBACK':
                            record.update(prior,resumed_from_failure_receipt=True)
                            continue
                        record.update(prior,resumed_from_verified_receipt=True)
                        text=next(t for t in graph.text.texts if t.id==identifier)
                        font_choices[text.style]=record.get('font')
                        current=Path(prior['ir']);continue
                parent_ir_sha=digest(current)
                if adaptive and not shadow:
                    selected=select_adaptive(graph,identifier,snap,source,saved,folder,current,fonts,
                        font_choices,record,budget,inpaint,compiler,office)
                    if selected is None:continue
                    score,current,acceptance,proposal,background=selected
                    record.update(selected_background=background,
                        selection_reason='first candidate passing hard regional gates; final independent QA pending')
                else:
                    with lease('cpu'):
                        passed,failures=generate(graph,identifier,snap['path'],saved,folder/identifier,inpaint=inpaint)
                    record['background_failures']=failures
                    if not passed:
                        report=json.loads((folder/identifier/'mask-completion.json').read_text('utf8'))
                        record['failure']='MASK_INCOMPLETE' if report['unassigned_contrast_pixels'] else 'BACKGROUND_REPAIR_FAILED'
                        continue
                    if shadow:
                        record.update(state='SHADOW_PRECHECK_PASS',candidates=[r['candidate'] for r in passed]);continue
                    text=next(t for t in graph.text.texts if t.id==identifier)
                    proposals=([font_choices[text.style]] if text.style in font_choices else
                               typography.get(text.style,{}).get('proposals',[])[:3]+[None])
                    alternatives=[]
                    for number,proposal in enumerate(proposals):
                        trial=folder/identifier/passed[0]['candidate'];output=folder/(identifier+'-font-'+str(number))
                        try:
                            acceptance=stage_and_verify(trial,graph,current.parent,fonts,output,office=True,
                                typography=proposal,compiler=compiler,office_probe=office)
                            if acceptance.get('regional_automatic_checks')!='PASS':
                                feedback=office_frame_feedback(output,identifier)
                                if feedback:
                                    previous=output;output=folder/(identifier+'-font-'+str(number)+'-bounds-fit')
                                    acceptance=stage_and_verify(trial,graph,current.parent,fonts,output,office=True,
                                        typography=proposal,compiler=compiler,office_probe=office,frame_minimum=feedback)
                                    record['attempts'].append({'font':proposal,'initial_bounds_report':str(previous/'acceptance.json'),
                                                              'office_frame_feedback':feedback})
                            if acceptance.get('regional_automatic_checks')!='PASS':
                                record['attempts'].append({'font':proposal,'failure':acceptance.get('failure','FONT_LAYOUT_FAILED'),'acceptance':str(output/'acceptance.json')});continue
                            actual=np.array(Image.open(output/'office/page-01-before.png').convert('RGB'))
                            bx,by,bw,bh=json.loads((trial/'residual_report.json').read_text('utf8'))['reference_context']['bbox']
                            score=float(np.abs(actual[by:by+bh,bx:bx+bw].astype(float)-source[by:by+bh,bx:bx+bw]).mean())
                            alternatives.append((score,output/'region/slideir.json',output/'acceptance.json',proposal))
                            record['attempts'].append({'font':proposal,'status':'VERIFIED_LOCAL','source_ppt_mae':score,'acceptance':str(output/'acceptance.json')})
                        except Exception as exc:
                            record['attempts'].append({'font':proposal,'failure':failure(exc,'font'),'error_type':type(exc).__name__,'reason':str(exc)[:400]})
                    if not alternatives:
                        causes=[a['failure'] for a in record['attempts'] if 'failure' in a]
                        specific=[cause for cause in causes if cause!='FONT_LAYOUT_FAILED']
                        record['failure']=specific[0] if specific else 'FONT_LAYOUT_FAILED'
                        continue
                    score,current,acceptance,proposal=min(alternatives,key=lambda row:row[0])
                    font_choices[text.style]=proposal
                    record.update(state='VERIFIED_LOCAL',ir=str(current),acceptance=str(acceptance),font=proposal,
                        selected_background=passed[0]['candidate'],source_ppt_mae=score,
                        reconstruction_binding=binding,parent_ir_sha256=parent_ir_sha,
                        selection_reason='hard gates and actual Office edits passed; minimum source-to-PowerPoint pixel error')
                font_choices[next(t.style for t in graph.text.texts if t.id==identifier)]=proposal
                record.update(state='VERIFIED_LOCAL',ir=str(current),acceptance=str(acceptance),font=proposal,
                    source_ppt_mae=score,reconstruction_binding=binding,parent_ir_sha256=parent_ir_sha,
                    quality='PENDING_FINAL_REVIEW',candidate_state='GENERATED',regional_commit='VERIFIED_LOCAL')
                proof_root=acceptance.parent
                record['verified_files']={str(p.resolve()):digest(p) for p in proof_root.rglob('*') if p.is_file()}
            except Exception as exc:
                record.update(failure=failure(exc,'generation'),error_type=type(exc).__name__,reason=str(exc)[:400])
            finally:
                if (record['state']=='LOCAL_FALLBACK' and request.get('regional_agent') and repair_provider
                        and agent_count<int(request.get('agent_region_budget',1))):
                    from .failure_agent import route,run as run_failure_agent
                    evidence=[p for p in (folder/identifier).rglob('*.json') if p.name in ('residual_report.json','peer-protection.json','mask-completion.json')]
                    attempted=[a['background'] for a in record['attempts'] if 'background' in a]
                    plan=route(record.get('failure'),'text',evidence,already_attempted=attempted)
                    record['agent_route']=plan
                    if evidence and plan['status']=='ELIGIBLE':
                        agent_count+=1
                        try:
                            agent_folder=folder/'fa'/str(agent_count)
                            repaired=run_failure_agent(graph,identifier,'text',snap['path'],saved,current.parent,fonts,evidence,
                                agent_folder,failure=record.get('failure'),provider=repair_provider,store=store,
                                cache=shared_cache,skill=request.get('agent_skill') or os.environ.get('IMAGE2PPT_AGENT_SKILL'),attempted=attempted,inpaint=inpaint)
                            record['agent_result']={'status':repaired['status'],'directory':str(agent_folder)}
                            if repaired['status']=='CANDIDATE_READY':
                                proposed=SceneGraph.model_validate_json(Path(repaired['selected']['candidate_scene']).read_text('utf8'))
                                trial=Path(repaired['selected']['candidate_directory']);output=agent_folder/'v'
                                proposal=font_choices.get(next(t.style for t in proposed.text.texts if t.id==identifier))
                                checked=stage_and_verify(trial,proposed,current.parent,fonts,output,office=True,
                                    typography=proposal,compiler=compiler,office_probe=office,include_long=False)
                                if checked.get('regional_automatic_checks')=='PASS':
                                    actual=np.array(Image.open(output/'office/page-01-before.png').convert('RGB'))
                                    bx,by,bw,bh=json.loads((trial/'residual_report.json').read_text('utf8'))['reference_context']['bbox']
                                    score=float(np.abs(actual[by:by+bh,bx:bx+bw].astype(float)-source[by:by+bh,bx:bx+bw]).mean())
                                    current=output/'region/slideir.json';proof=output/'acceptance.json'
                                    record.update(state='VERIFIED_LOCAL',ir=str(current),acceptance=str(proof),font=proposal,
                                        source_ppt_mae=score,selected_background=trial.name,candidate_state='GENERATED',
                                        regional_commit='VERIFIED_LOCAL',quality='PENDING_FINAL_REVIEW',
                                        selection_reason='failure-driven Agent tool passed original stage and actual Office gates')
                                    record.pop('failure',None)
                                    record['verified_files']={str(p.resolve()):digest(p) for p in output.rglob('*') if p.is_file()}
                        except Exception as exc:
                            record['agent_result']={'status':'UNKNOWN' if failure(exc,'qa')=='QA_INFRASTRUCTURE_UNAVAILABLE' else 'REVIEW_REQUIRED','reason':str(exc)[:400]}
                if binding and record['state']=='LOCAL_FALLBACK' and not record.get('resumed_from_failure_receipt'):
                    record.update(reconstruction_binding=binding,parent_ir_sha256=parent_ir_sha)
                    proof=folder/(identifier+'-failure.json')
                    write(proof,{'source_sha256':graph.source_sha256,'binding':binding,
                        'parent_ir_sha256':parent_ir_sha,'failure':record.get('failure'),'attempts':record['attempts'],
                        'retry_requires':'changed input, code, strategy or explicit retry_revision'})
                    paths=[proof,*[p for p in (folder/identifier).rglob('*') if p.is_file()]]
                    record['verified_files']={str(p.resolve()):digest(p) for p in paths}
                    if (folder/identifier/'index.json').is_file():record['candidate_state']='GENERATED'
                write(folder/'regions.json',state)
                store.task(record['region_id'],'region',record['state'],record,record['dependencies'])
        irs[index]=current;states.append(state);all_regions.extend(state)
    # All final objects are rechecked in the actual combined package. Earlier
    # per-region receipts never stand in for final composition validation.
    def compile_set(paths,name):
        composed=[]
        for original,path in zip(graphs,paths):
            graph=SceneGraph.model_validate_json((path.parent/'scene.json').read_text('utf8'))
            if (graph.source_sha256!=original.source_sha256 or
                {t.id:t.text for t in graph.text.texts}!={t.id:t.text for t in original.text.texts}):
                raise ValueError('composed scene changed confirmed source content')
            composed.append(effective_texts(graph,path))
        with lease('compiler'):
            return compiler(paths,attempt/name,composed)
    artifacts=compile_set(irs,'combined')
    office_dir=attempt/'office-combined';office_ok=False;reviews=[]
    try:
        with lease('office'):
            office_receipt=(office or inspect)(artifacts['pptx'],irs,office_dir)
            replacements=inspect_text(artifacts['pptx'],irs,attempt/'replacements',**({'include_long':False} if adaptive else {}))
        if office_receipt.get('status')!='PASS_APP_AUTOMATED':raise ValueError('combined Office failed')
        residuals=[inspect_residuals(ir,office_dir,i+1,attempt/f'residuals-{i+1:02d}.json') for i,ir in enumerate(irs)]
        owner_ok=all(r['residual_check_status']=='PASS' for rows in residuals for r in rows)
        if not owner_ok or replacements['status']!='PASS':raise ValueError('combined editable consistency failed')
        office_ok=True
    except Exception as exc:
        office_receipt={'status':'UNKNOWN','failure':failure(exc,'office'),'error_type':type(exc).__name__}
        write(attempt/'combined-office-failure.json',office_receipt)
        identifier=getattr(exc,'object_id',None)
        if identifier:
            for i,ir in enumerate(irs):
                selected=SceneGraph.model_validate_json((ir.parent/'scene.json').read_text('utf8')).selected
                if identifier not in selected:continue
                irs[i]=rollback(graphs[i],ir.parent,{identifier},attempt/f'office-rollback-{i+1:02d}','OFFICE_VALIDATION_FAILED')
                retained=SceneGraph.model_validate_json((irs[i].parent/'scene.json').read_text('utf8')).selected
                for record in states[i]:
                    if record['object_id'] not in retained and record['state']=='VERIFIED_LOCAL':
                        record.update(state='LOCAL_FALLBACK',failure='OFFICE_VALIDATION_FAILED')
    previews=[office_dir/f'page-{i+1:02d}-before.png' for i in range(len(snaps))] if office_ok else sorted(Path(artifacts['preview']).glob('*.png'))
    for i,(snap,ir,preview) in enumerate(zip(snaps,irs,previews)):
        try:
            if not office_ok:raise RuntimeError('Office evidence unavailable')
            visual=review(store,reviewer,snap,ir,preview,office_dir,i+1,attempt/f'review-panels-{i+1:02d}',adaptive=adaptive)
            findings=effective_findings((QualityReview if adaptive else Review).model_validate(visual))
            affected={n for finding in findings for n in finding.get('node_ids',[])}
            if affected:
                irs[i]=rollback(graphs[i],ir.parent,affected,attempt/f'visual-rollback-{i+1:02d}','VISUAL_REVIEW_FAILED')
                for record in states[i]:
                    if record['object_id'] not in SceneGraph.model_validate_json((irs[i].parent/'scene.json').read_text('utf8')).selected:
                        if record['state']=='VERIFIED_LOCAL':record.update(state='LOCAL_FALLBACK',failure='VISUAL_REVIEW_FAILED')
            visual['status']='PASS' if not findings else 'REVIEW_REQUIRED'
            reviews.append(visual)
        except Exception as exc:
            reviews.append({'status':'UNKNOWN','failure':failure(exc,'qa'),'error_type':type(exc).__name__,
                            'verified_regions_retained':True,'automatic_retry':False})
    # A visual rollback creates a new bound package and invalidates the previous
    # final-file review. Keep the candidate as review-required until re-reviewed.
    if any((attempt/f'{kind}-rollback-{i+1:02d}').exists() for i in range(len(snaps)) for kind in ('visual','office')):
        artifacts=compile_set(irs,'combined-local-rollback')
        previews=sorted(Path(artifacts['preview']).glob('*.png'))
        office_dir=attempt/'office-local-rollback'
        try:
            with lease('office'):
                office_receipt=(office or inspect)(artifacts['pptx'],irs,office_dir)
                replacement=inspect_text(artifacts['pptx'],irs,attempt/'replacement-local-rollback',**({'include_long':False} if adaptive else {}))
            office_ok=office_receipt.get('status')=='PASS_APP_AUTOMATED' and replacement['status']=='PASS'
            if not office_ok:raise ValueError('final rollback package Office checks failed')
            previews=[office_dir/f'page-{i+1:02d}-before.png' for i in range(len(snaps))]
            residuals=[]
            for i,(snap,ir,preview) in enumerate(zip(snaps,irs,previews)):
                residuals.append(inspect_residuals(ir,office_dir,i+1,attempt/f'rollback-residuals-{i+1:02d}.json'))
                if any(r['residual_check_status']!='PASS' for r in residuals[-1]):
                    raise ValueError('rollback source-owner residual check failed')
                try:reviews[i]=review(store,reviewer,snap,ir,preview,office_dir,i+1,attempt/f'rollback-review-panels-{i+1:02d}',adaptive=adaptive)
                except Exception as exc:reviews[i]={'status':'UNKNOWN','failure':failure(exc,'qa'),'verified_regions_retained':True}
        except Exception as exc:
            office_ok=False
            write(attempt/'rollback-office-failure.json',{'failure':failure(exc,'office'),'error_type':type(exc).__name__})
    delivery=job/'delivery';delivery.mkdir(exist_ok=True)
    shutil.copyfile(artifacts['pptx'],delivery/'result.pptx')
    for index,preview in enumerate(previews,1):shutil.copyfile(preview,delivery/f'page-{index:02d}.png')
    from ..parallel_pipeline import safe_graph
    from ..parallel_rebuild import build
    stable_irs=[]
    for index,(graph,snap) in enumerate(zip(graphs,snaps),1):
        with lease('cpu'):
            stable_ir,_=build(safe_graph(graph),snap,fonts,attempt/f'stable-source-{index:02d}',
                strategy='off',text_candidates=False)
        stable_irs.append(stable_ir)
    stable=compile_set(stable_irs,'stable-fallback')
    shutil.copyfile(stable['pptx'],delivery/'stable-fallback.pptx')
    for index,page in enumerate(states):
        visual=reviews[index]
        visual_pass=visual.get('status')=='PASS' or (visual.get('passed') is True and not visual.get('findings'))
        for record in page:
            record['final_ir_sha256']=digest(irs[index])
            regional_visual={**visual,'minor_deviations':[r for r in visual.get('minor_deviations',[]) if record['object_id'] in r.get('node_ids',[])]}
            record['quality']='UNKNOWN' if record.get('failure')=='QA_INFRASTRUCTURE_UNAVAILABLE' else quality_status(office_ok and record['state']=='VERIFIED_LOCAL',regional_visual)
            record['minor_deviations']=[r for r in visual.get('minor_deviations',[]) if record['object_id'] in r.get('node_ids',[])]
            record['visual_review']='PASS' if visual_pass else 'UNKNOWN' if visual.get('status')=='UNKNOWN' else 'REVIEW_REQUIRED'
            store.task(record['region_id'],'region',record['state'],record,record['dependencies'])
    write(attempt/'regions-final.json',states)
    write(job/'qa.json',{'status':'REVIEW_REQUIRED','architecture':'r12-regional','reviews':reviews,
        'office':office_receipt,'regions':states,'automatic_publication':False,'human_acceptance':False})
    stable_assets=[]
    if office_ok and all(r.get('passed') is True and not r.get('findings') for r in reviews):
        for i,ir in enumerate(irs):
            scene_file=ir.parent/'scene.json'
            composed=SceneGraph.model_validate_json(scene_file.read_text('utf8'))
            acceptance=attempt/f'stable-acceptance-{i+1:02d}.json'
            write(acceptance,{'source_sha256':composed.source_sha256,'artifacts':artifacts,
                'office':office_receipt,'owner_residual':residuals[i],'page':i+1,'visual':reviews[i]})
            definitions={a.id:a for a in composed.asset.objects}
            for identifier,record in composed.selected.items():
                if record['kind']!='asset' or identifier not in definitions:continue
                asset=record['asset_reference'];file=(ir.parent/asset['path']).resolve()
                owner=ir.parent/record['owner_evidence']['file']
                stable_assets.append({'source_sha256':composed.source_sha256,'id':identifier,'bbox':record['bbox'],
                    'path':str(file),'sha256':digest(file),'definition':definitions[identifier].model_dump(),
                    'automatic_quality':'PASS','human_acceptance':False,
                    'verified_replacement':{field:{'path':str(path.resolve()),'sha256':digest(path)} for field,path in
                        [('scene',scene_file),('owner',owner),('acceptance',acceptance)]}})
    count=sum(r['state']=='VERIFIED_LOCAL' for r in all_regions)
    result={'status':'REVIEW_REQUIRED','pipeline_mode':request.get('pipeline_mode','parallel_review'),
        'pages':len(snaps),'generation':expected+1,'candidate_engine':'r12-shadow' if shadow else 'r12-regional',
        'locally_verified_text_regions':count,'regions':states,'independent_review':reviews,'office_pass':office_ok,
        'reconstructed_pages':sum(bool(SceneGraph.model_validate_json((p.parent/'scene.json').read_text('utf8')).selected) for p in irs),
        'slide_irs':[str(p) for p in irs],'human_acceptance':False,'automatic_publication':False,'fallback_basis':'immutable_source_raster',
        'artifacts':{p.name:digest(p) for p in delivery.iterdir() if p.is_file()}}
    result['workflow_states']={'candidate_generation':'COMPLETED','regional_commit':'COMPLETED',
        'final_qa':'PASS' if office_ok and all(r.get('passed') is True and not r.get('findings') for r in reviews) else 'UNKNOWN_OR_REVIEW_REQUIRED',
        'automatic_publication':'GATED'}
    if budget:
        write(attempt/'cost-report.json',budget.report())
        result['execution_cost']=budget.report()
    write(job/'result.json',result)
    files={str(p.resolve()):digest(p) for p in [*attempt.rglob('*'),*delivery.iterdir(),job/'qa.json',job/'request.json',
           job/'provenance.json',job/'fonts.json'] if p.is_file()}
    for ir in irs:
        for p in ir.parent.rglob('*'):
            if p.is_file():files[str(p.resolve())]=digest(p)
    for p in job.glob('page-*/perception-replay.json'):files[str(p.resolve())]=digest(p)
    for p in job.glob('page-*/barrier-1.json'):files[str(p.resolve())]=digest(p)
    for pattern in ('page-*/asset-coordinate-report.json','page-*/agent/**/*','page-*/verified-seed.json'):
        for p in job.glob(pattern):
            if p.is_file():files[str(p.resolve())]=digest(p)
    if request.get('perception_replay'):files[str(Path(request['perception_replay']).resolve())]=digest(request['perception_replay'])
    for original in [*original_irs,*stable_irs]:
        for p in original.parent.rglob('*'):
            if p.is_file():files[str(p.resolve())]=digest(p)
    for record in all_regions:
        files.update(record.get('verified_files',{}))
    if request.get('perception_replay'):
        replay=json.loads(Path(request['perception_replay']).read_text('utf8'))
        for row in replay['pages']:
            for field in ('scene','ocr'):files[str(Path(row[field]['path']).resolve())]=row[field]['sha256']
    for asset in baseline['stable_assets']:
        for proof in asset.get('verified_replacement',{}).values():
            files[str(Path(proof['path']).resolve())]=proof['sha256']
    store.publish({'schema':'SceneCommit/1','files':files,'result':result,'stable_assets':stable_assets},expected)
    try:
        from ..parallel_catalog import Catalog
        Catalog(shared_cache).register(job)
    except Exception as exc:write(job/'catalog-recovery-required.json',{'error_type':type(exc).__name__})
    if progress:progress(stage='complete',status='REVIEW_REQUIRED',verified_regions=count)
    return result
