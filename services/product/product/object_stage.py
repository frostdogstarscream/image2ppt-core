"""Opt-in backend stage. Production default remains off until real-image acceptance."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from .contracts import Page,Crop,Shape
from .materialize import digest,write
from .object_contracts import ObjectPlan,OBJECT_PROMPT
from .object_extract import extract

ROOT=Path(__file__).resolve().parents[3]


def route_setting(value='default'):
    route=os.environ.get('IMAGE2PPT_OBJECT_EXTRACTION','off') if value=='default' else value
    if route not in ('off','alpha','sam3','sam3.1','auto'):raise ValueError('unsupported object extraction route')
    return route


def obtain_layers(source,directory,*,resume=False):
    """Fresh uploads use LayerD; experiment callers inject an immutable saved pack."""
    directory=Path(directory)
    if (directory/'manifest.json').is_file():
        meta=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if all((directory/r['file']).is_file() and digest(directory/r['file'])==r['sha256'] for r in meta['layers']):return directory
        if not resume:raise ValueError('incomplete LayerD pack; resume explicitly')
    token=os.environ.get('LAYERD_TOKEN_FILE')
    if not token:raise RuntimeError('layerd_token_file_missing')
    directory.parent.mkdir(parents=True,exist_ok=True)
    log=directory.with_suffix('.log')
    args=[sys.executable,str(ROOT/'services/layerd/client.py'),'--input',str(source),'--output',str(directory),
          '--url',os.environ.get('LAYERD_URL','http://127.0.0.1:18840'),'--token-file',token]
    timeout=int(os.environ.get('IMAGE2PPT_LAYERD_TIMEOUT','1200'))
    if not 1<=timeout<=1200:raise ValueError('invalid LayerD tool timeout')
    args.extend(['--timeout',str(timeout)])
    if resume:args.append('--resume')
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run(args,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,timeout=timeout+100,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode:raise RuntimeError('layerd_decomposition_failed')
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['source_sha256']!=digest(source):raise ValueError('layerd_upload_binding_mismatch')
    if not np.array_equal(np.array(Image.open(source).convert('RGB')),np.array(Image.open(directory/'source.png').convert('RGB'))):raise ValueError('layerd_normalized_pixels_changed')
    binding={'source_sha256':digest(source),'normalized_sha256':digest(directory/'source.png')}
    receipt=directory/'request.json'
    if receipt.exists():
        original=json.loads(receipt.read_text(encoding='utf-8'))
        if any(original.get(k)!=v for k,v in binding.items()):raise ValueError('layerd_request_receipt_binding_mismatch')
    else:write(receipt,binding)
    write(directory/'upload-binding.json',binding)
    return directory


def object_request(source,layerdir,page,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    meta=json.loads((Path(layerdir)/'manifest.json').read_text(encoding='utf-8'))
    w,h=page.image_size;yy,xx=np.indices((h,w));checker=np.repeat(np.where(((xx//20+yy//20)%2)[...,None],215,245).astype('uint8'),3,2)
    images=[Path(source)];layers=[]
    for row in meta['layers']:
        rgba=Image.open(Path(layerdir)/row['file']).convert('RGBA')
        preview=Image.fromarray(checker).convert('RGBA');preview.alpha_composite(rgba)
        path=directory/(row['file'].replace('.png','-checker.png'));preview.convert('RGB').save(path);images.append(path)
        layers.append({k:row[k] for k in ('file','z','sha256')})
    return {'source_sha256':digest(source),'image_size':page.image_size,'layers':layers,
            'image_order':['source']+[r['file'] for r in layers],
            'protected_texts':[{'id':t.id,'bbox':t.bbox,'text':t.text} for t in page.texts],
            'baseline_graphics':[n.model_dump() for n in [*page.crops,*page.shapes]]},images


def infer_sam(source,plan,directory,route,object_ids):
    client_file=ROOT/'.runtime/sam-client.json'
    client=json.loads(client_file.read_text(encoding='utf-8')) if client_file.exists() else {}
    service_url=os.environ.get('IMAGE2PPT_SAM_URL') or client.get('url')
    if service_url and route in ('sam3','auto'):
        from .sam_jobs import infer_remote
        token_file=os.environ.get('IMAGE2PPT_SAM_TOKEN_FILE') or client.get('token_file')
        if not token_file:raise RuntimeError('private_sam_token_file_missing')
        return infer_remote(source,plan,directory,object_ids,url=service_url,
            token_file=token_file)
    config=os.environ.get('IMAGE2PPT_SAM_CONFIG')
    if not config:return None
    cfg=json.loads(Path(config).read_text(encoding='utf-8'));route='sam3' if route=='auto' else route
    model=cfg.get('models',{}).get(route)
    if not model:raise RuntimeError('sam_model_not_configured')
    args=[cfg['python'],str(ROOT/'services/segmentation/single_frame.py'),'--version',route,
        '--source',str(source),'--plan',str(plan),'--output',str(directory),'--sam-source',cfg['source'],
        '--code-revision',cfg['code_revision'],'--checkpoint',model['checkpoint'],'--checkpoint-sha256',model['sha256'],
        '--objects',*object_ids]
    with Path(directory).with_suffix('.log').open('w',encoding='utf-8') as stream:
        try:subprocess.run(args,stdout=stream,stderr=stream,timeout=1800,check=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except (subprocess.SubprocessError,OSError):return None
    return directory


def effective_page(page,irfile):
    ir=json.loads(Path(irfile).read_text(encoding='utf-8'));payload=page.model_dump();ids={n['id'] for n in ir['nodes']}
    payload['shapes']=[s for s in payload['shapes'] if s['id'] in ids]
    descriptions={a['id']:a.get('description','source graphic') for a in ir['assets']}
    payload['crops']=[{'id':n['id'],'bbox':n['bbox'],'z':n['z'],'description':descriptions.get(n['asset_id'],'source graphic')[:300]} for n in ir['nodes'] if n['type']=='image']
    # VisualPage's public crop validator correctly assumes an opaque source
    # rectangle. Extracted PNGs are different: their transparent bounding-box
    # corners may intersect text, but extract() has already rejected ANY actual
    # alpha/text intersection. Do not weaken the model-facing crop validator or
    # allow models to self-declare transparency to bypass it.
    return page.model_copy(update={
        'texts':[t.model_copy(update={'z':next(n['z'] for n in ir['nodes'] if n['id']==t.id)}) for t in page.texts],
        'shapes':[Shape.model_validate({'id':n['id'],'bbox':n['bbox'],'z':n['z'],'geometry':n['geometry'],'fill':n['fill'],
            'line_color':n.get('line',{}).get('color','FFFFFF'),'line_width_pt':n.get('line',{}).get('width_pt',0),
            **{k:n[k] for k in ('fill_gradient','corner_radius_px') if k in n}}) for n in ir['nodes'] if n['type']=='shape'],
        'crops':[Crop.model_validate(c) for c in payload['crops']]})


def process(source,page,baseline_ir,directory,route,ask,*,layerdir=None,state_directory=None,review_feedback=None):
    if route=='auto':return process_closed(source,page,baseline_ir,directory,ask,layerdir=layerdir,state_directory=state_directory,review_feedback=review_feedback)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    layerdir=Path(layerdir) if layerdir else obtain_layers(source,directory/'layers')
    # Source may be reencoded by LayerD; normalized RGB equality is already gated.
    normalized=layerdir/'source.png'
    context,images=object_request(normalized,layerdir,page,directory/'previews')
    raw=ask(context,images,ObjectPlan)
    write(directory/'object-plan-raw.json',raw)
    plan=ObjectPlan.model_validate(raw)
    if plan.source_sha256!=digest(normalized) or plan.image_size!=page.image_size:raise ValueError('object plan source binding mismatch')
    planfile=directory/'object-plan.json';write(planfile,plan.model_dump())
    ir,report=extract(planfile,layerdir,baseline_ir,directory/'alpha',route='alpha')
    difficult=[r['id'] for r in report['objects'] if 'alpha_ambiguous_or_empty' in r['reasons']]
    if route!='alpha' and difficult:
        samdir=infer_sam(normalized,planfile,directory/'sam',route,difficult)
        ir,report=extract(planfile,layerdir,baseline_ir,directory/'applied',route=route,sam_dir=samdir)
    report['requested_route']=route
    proposal=effective_page(page,ir)
    # Any object fallback is an unresolved issue, never an automatic visual pass.
    if report['fallback']:
        proposal=proposal.model_copy(update={'limitations':[ *proposal.limitations, f"Object extraction retained {report['fallback']} original regions; incomplete independent extraction."]})
    return ir,proposal,report


def process_closed(source,page,baseline_ir,directory,ask,*,layerdir=None,state_directory=None,review_feedback=None):
    from .object_contracts import ClosedPlan,read_plan,ObjectRepairs,apply_repairs
    from .object_closure import extract_closed
    from .object_scheduler import ObjectSchedule,repair_route
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    state=Path(state_directory) if state_directory else directory/'state';state.mkdir(parents=True,exist_ok=True)
    layerdir=Path(layerdir) if layerdir else obtain_layers(source,state/'layers')
    normalized=layerdir/'source.png'
    context,images=object_request(normalized,layerdir,page,directory/'previews')
    confirmation=state/'confirmed-boundaries.json'
    confirmed=[]
    if confirmation.exists():
        value=json.loads(confirmation.read_text(encoding='utf-8'))
        if value['source_sha256']!=digest(normalized):raise ValueError('confirmed object boundary source mismatch')
        confirmed=value['boundaries'];context['confirmed_boundaries']=confirmed
    planfile=state/'object-plan.json'
    active=state/'active-plan.json'
    if active.exists():
        binding=json.loads(active.read_text(encoding='utf-8'));file=(state/binding['file']).resolve()
        if not file.is_relative_to(state.resolve()) or digest(file)!=binding['sha256']:raise ValueError('object plan cache changed')
        planfile=file
    if not planfile.exists():
        previous=sorted(state.glob('object-plan-raw-*.json'))
        if previous:
            context.update(previous_plan=json.loads(previous[-1].read_text(encoding='utf-8')),
                feedback='Previous output failed contract validation. Fix only the contract error; do not silently alter original pixels.')
        for index in range(len(previous),3):
            raw=ask(context,images,ClosedPlan);write(state/f'object-plan-raw-{index:02d}.json',raw)
            try:
                plan,legacy=read_plan(raw)
                if plan.source_sha256!=digest(normalized) or plan.image_size!=page.image_size:raise ValueError('object plan binding mismatch')
            except ValueError as exc:
                write(state/f'object-plan-invalid-{index:02d}.json',{'error':str(exc)[:4000]})
                context.update(previous_plan=raw,feedback=str(exc)[:4000])
                continue
            write(planfile,plan.model_dump());break
        else:raise ValueError('object inventory contract repair exhausted; parent preserved')
    else:plan,_=read_plan(json.loads(planfile.read_text(encoding='utf-8')))
    ir,report=extract_closed(planfile,layerdir,baseline_ir,directory/'alpha',registry=state/'registry')
    review_rows={}
    if review_feedback:
        from .object_contracts import ObjectVerdict
        scene=Path(review_feedback['scene_file']).resolve()
        if not scene.is_relative_to(state.parent.resolve()) or digest(scene)!=review_feedback['scene_sha256'] or digest(source)!=review_feedback['source_sha256']:
            raise ValueError('independent object feedback binding mismatch')
        previous=json.loads(scene.read_text(encoding='utf-8'));assets={a['id']:a for a in previous['assets']}
        for raw in review_feedback['objects']:
            verdict=ObjectVerdict.model_validate(raw);node=next(n for n in previous['nodes'] if n['id']=='extracted-'+verdict.id)
            asset=assets[node['asset_id']]
            if digest(scene.parent/asset['path'])!=verdict.candidate_sha256:raise ValueError('independent object feedback PNG mismatch')
            review_rows[verdict.id]=verdict.model_dump()
        context['independent_object_review']=review_rows
    schedule=ObjectSchedule(state/'schedule.json')
    from .object_registry import fingerprint
    def definition(obj):
        return {k:v for k,v in obj.model_dump().items() if k not in ('id','description','components','kind')}
    for repair in range(2):
        targets=[];calls=[]
        for row in report['objects']:
            obj=next(o for o in plan.objects if o.id==row['id'])
            if row['id'] in review_rows:
                row['independent_review']=review_rows[row['id']];row['reasons'].append('independent_visual_review')
            for boundary in confirmed:
                if set(boundary['owners'])<=set(obj.replaces) and any(term.lower() in p.name.lower()
                    for term in boundary.get('forbidden_component_terms',[]) for p in obj.parts):
                    row['reasons'].append('confirmed_boundary_needs_replan');row['confirmed_boundary']=boundary
            route=repair_route(row['reasons'])
            if (row['status']!='FALLBACK' and not row.get('confirmed_boundary') and not row.get('independent_review')) or route not in ('replan_layers','replan_owners','joint_arbitration','local_text_evidence'):continue
            call,fresh=schedule.reserve(plan.source_sha256+':'+obj.id,'planning_revision',{'object':definition(obj),'failure':row['reasons'],
                'protocol':'ObjectRepairs/1','confirmed_boundaries':confirmed,'independent_review':review_rows.get(row['id'])})
            if fresh:targets.append(row);calls.append(call)
        if not targets:break
        repair_context={**context,'previous_plan':plan.model_dump(),'target_object_ids':[r['id'] for r in targets],
            'feedback':targets,'instruction':'Return ObjectRepairs/1 with ONLY substantively changed listed objects. Omit unchanged objects. The server retains all other definitions. Do not change fonts, cards or wording. No change is better than inventing background ownership.'}
        try:
            raw=ask(repair_context,images,ObjectRepairs)
            index=len(list(state.glob('repair-raw-*.json')));write(state/f'repair-raw-{index:02d}.json',raw)
            ids={r['id'] for r in targets};revised=apply_repairs(plan,raw,ids);old={o.id:o for o in plan.objects};new={o.id:o for o in revised.objects}
            if revised.source_sha256!=plan.source_sha256 or revised.image_size!=plan.image_size or set(new)!=set(old):raise ValueError('repair changed inventory binding')
            if any(new[k].model_dump()!=old[k].model_dump() for k in old.keys()-ids):raise ValueError('repair changed a locked non-target definition')
            changed={k for k in ids if fingerprint(definition(new[k]))!=fingerprint(definition(old[k]))}
            for row,call in zip(targets,calls):schedule.finish(call,'SUCCEEDED' if row['id'] in changed else 'FAILED',{'substantive_change':row['id'] in changed})
            if not changed:break
            plan=revised;planfile=state/f'repair-plan-{index:02d}.json';write(planfile,plan.model_dump())
            for identifier in changed:review_rows.pop(identifier,None)
            write(active,{'file':planfile.name,'sha256':digest(planfile)})
            ir,report=extract_closed(planfile,layerdir,baseline_ir,directory/f'repair-{repair:02d}',registry=state/'registry')
        except (ValueError,RuntimeError,OSError) as exc:
            write(state/f'repair-invalid-{len(list(state.glob("repair-invalid-*.json"))):02d}.json',{'error':str(exc)[:2000]})
            for call in calls:
                if call['state']=='RESERVED':schedule.finish(call,'FAILED')
            break
    violations={o.id:['confirmed source boundary contradicts object parts'] for o in plan.objects for b in confirmed
        if set(b['owners'])<=set(o.replaces) and any(term.lower() in p.name.lower() for term in b.get('forbidden_component_terms',[]) for p in o.parts)}
    if violations:ir,report=extract_closed(planfile,layerdir,baseline_ir,directory/'boundary-fallback',registry=state/'registry',rejected=violations)
    objects={o.id:o for o in plan.objects};difficult=[];reservations=[]
    for row in report['objects']:
        row['repair_route']=repair_route(row['reasons'])
        if row['repair_route']=='sam_prompt':
            obj=objects[row['id']]
            call,fresh=schedule.reserve(plan.source_sha256+':'+obj.id,'sam_prompt',{'bbox':obj.bbox,'prompt':obj.prompt_en,'model':'sam3'})
            if fresh:difficult.append(obj.id);reservations.append(call)
            elif call:
                # Private service submission is idempotent. Resume a transport
                # interruption with the SAME reservation, never a new prompt.
                difficult.append(obj.id);reservations.append(call)
    if difficult:
        try:
            samroot=state/'sam'/fingerprint({'plan_sha256':digest(planfile),'objects':sorted(difficult),'model':'sam3'})
            samroot.parent.mkdir(exist_ok=True)
            samdir=infer_sam(normalized,planfile,samroot,'auto',difficult)
            for call in reservations:schedule.finish(call,'SUCCEEDED' if samdir else 'FAILED',str(samdir) if samdir else None)
            if samdir:ir,report=extract_closed(planfile,layerdir,baseline_ir,directory/'sam-applied',sam_dir=samdir,registry=state/'registry',rejected=violations)
        except (OSError,RuntimeError,ValueError,KeyError,TimeoutError):
            for call in reservations:schedule.finish(call,'FAILED')
            report['sam_service']='UNAVAILABLE_RETAIN_PARENT'
    report.update(requested_route='auto',state_directory=str(state),publication='STAGED_ONLY')
    baseline=json.loads(Path(baseline_ir).read_text(encoding='utf-8'))
    required={n['id'] for n in baseline['nodes'] if n['type']=='image' and not n['id'].startswith('restored-bg-')}
    claimed={owner for r in report['objects'] for owner in r['replaces']}
    report['inventory_missing_owner_ids']=sorted(required-claimed)
    write(directory/'closure.json',report)
    proposal=effective_page(page,ir)
    if report['fallback'] or report['inventory_missing_owner_ids']:proposal=proposal.model_copy(update={'limitations':[*proposal.limitations,'Visual objects retained reliable parent representations; independent extraction remains incomplete.']})
    return ir,proposal,report
