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
from .region_candidates import generate
from .font_candidates import compare
from .region_state import rollback
from .validation import stage_and_verify
from .lama_adapter import LamaClient
from .remote_gpu_lease import remote_gpu_lease


def failure(exc,stage):
    code=getattr(exc,'code','')
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
    def __init__(self,config,cache):
        self.config=config;self.client=LamaClient(config['url'],config['model_binding'],cache)
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


def review(store,reviewer,snap,ir,preview,office_dir,page,output):
    details,refs=panels(snap['path'],preview,ir,office_dir,page,output)
    context={'source_sha256':snap['source_sha256'],'plan_sha256':digest(ir),'preview_sha256':digest(preview),
        'scene':json.loads(ir.read_text('utf8')),'transactions':json.loads((ir.parent/'transactions.json').read_text('utf8')),
        'actual_edit_evidence':refs,'office_pass':True,'raster_remainder_is_explicit_fallback':True}
    result=store.call('r12-final-review',reviewer,PROMPT,context,[snap['path'],str(preview),*details],Review,lease('go'))
    if any(getattr(result,k)!=context[k] for k in ('source_sha256','plan_sha256','preview_sha256')):
        raise ValueError('independent review binding mismatch')
    return result.model_dump()


def run(job,request,snaps,graphs,irs,baseline,fonts,store,expected,*,compiler,reviewer,office=None,progress=None,shadow=False):
    from ..pipeline import effective_findings
    root=job/'r12';root.mkdir(exist_ok=True)
    attempt=root/('attempt-'+key({'request':request,'code':digest(__file__)})[:16])
    if attempt.exists():
        from uuid import uuid4
        attempt=attempt.with_name(attempt.name+'-'+uuid4().hex[:8])
    attempt.mkdir()
    package=Path(__file__).resolve().parents[1]
    code_files=[*package.glob('*.py'),*Path(__file__).parent.glob('*.py')]
    write(attempt/'code-manifest.json',{'entrypoint':'product.parallel_pipeline.run',
        'files':{str(p.relative_to(package.parent)):digest(p) for p in code_files},
        'candidate_engine':'r12-shadow' if shadow else 'r12-regional'})
    config_file=Path(__file__).resolve().parents[4]/'.runtime/parallel-services.json'
    config=json.loads(config_file.read_text('utf8')).get('lama') if config_file.exists() else None
    shared_cache=Path(os.environ.get('IMAGE2PPT_PARALLEL_CACHE',str(Path(__file__).resolve().parents[4]/'.runtime/parallel-cache')))
    inpaint=InpaintService(config,shared_cache/'r12-lama') if config and not shadow else None
    states=[];original_irs=list(irs);all_regions=[]
    for index,(graph,snap,ir) in enumerate(zip(graphs,snaps,irs)):
        graph=SceneGraph.model_validate(graph.model_dump());graphs[index]=graph
        folder=attempt/f'page-{index+1:02d}';folder.mkdir()
        write(folder/'source-scene.json',graph.model_dump())
        requested=request.get('regional_object_ids')
        targets=[t.id for t in graph.text.texts if (requested is not None and t.id in requested)
                 or (requested is None and graph.ownership[t.id]=='RESOLVED')]
        source=np.array(Image.open(snap['path']).convert('RGB'))
        with lease('cpu'):typography=compare(graph,set(targets),source,fonts,folder)
        saved=next((p['layers'] for p in baseline['pages'] if p['source']['sha256']==snap['source_sha256']),None)
        if saved is None:
            try:
                from ..parallel_branches import layer_evidence
                saved,_=layer_evidence(snap,root/'layer-cache')
            except Exception:saved=None
        current=ir;state=[];font_choices={};region_records,by_id=regions(graph)
        write(folder/'dependencies.json',region_records)
        for identifier in targets:
            record={'object_id':identifier,'region_id':by_id[identifier]['region_id'],
                    'dependencies':by_id[identifier]['dependencies'],'state':'LOCAL_FALLBACK','attempts':[]}
            state.append(record)
            try:
                binding=key({'scene':graph.model_dump(),'object':identifier,'fonts':fonts,
                    'engine':digest(__file__),'generation':digest(Path(__file__).with_name('region_candidates.py')),
                    'stage':digest(Path(__file__).with_name('candidate_stage.py')),
                    'validation':digest(Path(__file__).with_name('validation.py')),'shadow':shadow})
                with store.db() as db:
                    prior=db.execute('SELECT payload FROM tasks WHERE k=? AND state=?',
                                     (record['region_id'],'VERIFIED_LOCAL')).fetchone()
                if prior and not shadow:
                    prior=json.loads(prior[0])
                    references=prior.get('verified_files',{})
                    if (prior.get('reconstruction_binding')==binding and prior.get('parent_ir_sha256')==digest(current)
                            and references and all(Path(p).is_file() and digest(p)==sha for p,sha in references.items())):
                        record.update(prior,resumed_from_verified_receipt=True)
                        text=next(t for t in graph.text.texts if t.id==identifier)
                        font_choices[text.style]=record.get('font')
                        current=Path(prior['ir']);continue
                parent_ir_sha=digest(current)
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
                proof_root=acceptance.parent
                record['verified_files']={str(p.resolve()):digest(p) for p in proof_root.rglob('*') if p.is_file()}
            except Exception as exc:
                record.update(failure=failure(exc,'generation'),error_type=type(exc).__name__)
            finally:
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
            replacements=inspect_text(artifacts['pptx'],irs,attempt/'replacements')
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
            visual=review(store,reviewer,snap,ir,preview,office_dir,i+1,attempt/f'review-panels-{i+1:02d}')
            findings=effective_findings(Review.model_validate(visual))
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
                replacement=inspect_text(artifacts['pptx'],irs,attempt/'replacement-local-rollback')
            office_ok=office_receipt.get('status')=='PASS_APP_AUTOMATED' and replacement['status']=='PASS'
            if not office_ok:raise ValueError('final rollback package Office checks failed')
            previews=[office_dir/f'page-{i+1:02d}-before.png' for i in range(len(snaps))]
            residuals=[]
            for i,(snap,ir,preview) in enumerate(zip(snaps,irs,previews)):
                residuals.append(inspect_residuals(ir,office_dir,i+1,attempt/f'rollback-residuals-{i+1:02d}.json'))
                try:reviews[i]=review(store,reviewer,snap,ir,preview,office_dir,i+1,attempt/f'rollback-review-panels-{i+1:02d}')
                except Exception as exc:reviews[i]={'status':'UNKNOWN','failure':failure(exc,'qa'),'verified_regions_retained':True}
        except Exception as exc:
            office_ok=False
            write(attempt/'rollback-office-failure.json',{'failure':failure(exc,'office'),'error_type':type(exc).__name__})
    delivery=job/'delivery';delivery.mkdir(exist_ok=True)
    shutil.copyfile(artifacts['pptx'],delivery/'result.pptx')
    for index,preview in enumerate(previews,1):shutil.copyfile(preview,delivery/f'page-{index:02d}.png')
    stable=compile_set(original_irs,'stable-fallback')
    shutil.copyfile(stable['pptx'],delivery/'stable-fallback.pptx')
    for index,page in enumerate(states):
        visual=reviews[index]
        visual_pass=visual.get('status')=='PASS' or (visual.get('passed') is True and not visual.get('findings'))
        for record in page:
            record['final_ir_sha256']=digest(irs[index])
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
        'slide_irs':[str(p) for p in irs],'human_acceptance':False,'automatic_publication':False,
        'artifacts':{p.name:digest(p) for p in delivery.iterdir() if p.is_file()}}
    write(job/'result.json',result)
    files={str(p.resolve()):digest(p) for p in [*attempt.rglob('*'),*delivery.iterdir(),job/'qa.json',job/'request.json',
           job/'provenance.json',job/'fonts.json'] if p.is_file()}
    for ir in irs:
        for p in ir.parent.rglob('*'):
            if p.is_file():files[str(p.resolve())]=digest(p)
    for p in job.glob('page-*/perception-replay.json'):files[str(p.resolve())]=digest(p)
    for p in job.glob('page-*/barrier-1.json'):files[str(p.resolve())]=digest(p)
    if request.get('perception_replay'):files[str(Path(request['perception_replay']).resolve())]=digest(request['perception_replay'])
    for original in original_irs:
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
