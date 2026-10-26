"""Opt-in region-routed automatic workflow, isolated from historical v1 jobs."""
from copy import deepcopy
from pathlib import Path
import shutil
import time
from uuid import uuid4

from PIL import Image

from image2ppt.storage import digest, read_json, write_json, run_lock
from . import auto_workflow as legacy
from .auto_artifacts import DeadlineBackend, build_revision, verify_hashes, file_hashes
from .auto_provider import save_new, fingerprint, preflight
from .auto_protocol import transform_model_boxes
from .auto_transport import SSHRemote
from .auto_typography import draft_feedback
from .auto_region_protocol import (POLICY, LAYOUT_PROMPT, AUTHOR_PROMPT, PATCH_PROMPT,
    VISUAL_PROMPT, PROBE_PROMPT, layout_from_draft, layout_diagnostics, decision_record, text_candidates,
    expand_region, element_diagnostics, retained_region, assemble, validate_region, validate_correction, require)
from .auto_region_evidence import (crop_source, region_packets, verify_deletion,
    typography_checks, validate_visual, validate_probe, issue, check_transitions)
from .region_features import features
from .region_plan import PROMPT as DECISION_PROMPT, CAPABILITIES, CLASSIFICATIONS
from .supervised_elements import GEOMETRIES, _intersection
from .typography import font_status
from .font_contract import resolve_font


def create_job(inputs, output, config, *, gate_pages=None, preference='faithful_editable', gate_only=True):
    root=legacy.create_job(inputs,output,config,gate_pages=gate_pages,preference=preference,
                           pipeline_version=2,policy=POLICY,gate_only=gate_only)
    state=read_json(root/'state.json'); state.update(pipeline_version=2,gate_only=gate_only)
    write_json(root/'state.json',state)
    return root


def step(path, inputs, produce, *, session=None, allow_prompt_history=False):
    """One immutable result bound to exact source/prompt/config/parent inputs."""
    path=Path(path); meta=path.with_suffix('.binding.json'); expected=fingerprint(inputs)
    if path.exists():
        binding=read_json(meta)
        require(binding['sha256']==digest(path),'saved stage evidence changed: '+path.name)
        matches=binding['input_binding']==expected
        if not matches and allow_prompt_history and session and 'prompt' in inputs:
            # A successfully authored stage remains valid after improving retry
            # wording. Only the prompt may differ; all parent/source/model/crop
            # inputs must hash identically. Keep the old proof and log reuse.
            for call in sorted((session.root/'calls').glob('c[0-9][0-9]/request.json')):
                past=read_json(call)['request']['body']['messages'][0]['content']
                if fingerprint({**inputs,'prompt':past})==binding['input_binding']:
                    receipt=path.with_name(path.stem+'-prompt-reuse-'+expected[:12]+'.json')
                    if not receipt.exists():
                        save_new(receipt,{'original_input_binding':binding['input_binding'],
                            'current_input_binding':expected,'result_sha256':binding['sha256'],
                            'old_prompt_sha256':fingerprint(past),'new_prompt_sha256':fingerprint(inputs['prompt']),
                            'reason':'Only prompt wording changed; exact semantic inputs and validated result preserved.'})
                    matches=True; break
        require(matches,'saved stage evidence changed: '+path.name)
        return read_json(path)
    value=produce(); save_new(path,value)
    save_new(meta,{'input_binding':expected,'sha256':digest(path),
                   'response_sha256':getattr(session,'last_response_sha256',None)})
    return value


def checked(session, phase, prompt, evidence, images, validator, *, tokens=16000, feedback=None, attempts=3):
    return legacy.checked_model_json(session,phase,prompt,evidence,images,validator,tokens,feedback_builder=feedback,max_attempts=attempts)


def capabilities(project_root, session):
    def measure():
        report=DeadlineBackend(project_root,session.deadline).doctor()
        require(report.get('capability_source',{}).get('engine_probe',{}).get('available'),
                'fixed compiler runtime unavailable')
        value=deepcopy(CAPABILITIES)
        value['schema']='automatic-region-capabilities/1'
        value['compiler']={k:report.get(k) for k in ['artifact_version','supported','not_implemented','capability_source']}
        # ElementPlan authoring currently emits editable shapes/text for tables
        # and bars. The wider compiler's chart support is not falsely exposed.
        value['ir_limits'].update(native_tables='editable_shapes_and_text',native_charts='editable_bar_shapes_only')
        value['authoring_adapter']={'node_types':['text','shape','image'],
                                    'geometries':sorted(GEOMETRIES),'gradients':True,
                                    'semantic_connectors':False,'inpainting':False}
        return value
    return step(session.root/'capabilities.json',{'protocol':2,'project_root':str(project_root)},measure)


def get_layout(session, source, inventory, caps):
    size=inventory['image_size']
    evidence=transform_model_boxes({'source_sha256':inventory['source_sha256'],'image_size':size,
              'ocr':[{'id':t['id'],'text':t['text'],'bbox':t['bbox']} for t in inventory['texts']],
              'candidates':[{k:c[k] for k in ('id','bbox','geometry_hint')} for c in inventory.get('candidates',[])],
              'classifications':list(CLASSIFICATIONS)},size,to_model=True)
    return step(session.root/'layout.json',{'prompt':LAYOUT_PROMPT,'evidence':evidence,'model':session.model},
                lambda:checked(session,'initial/layout',LAYOUT_PROMPT,evidence,[source],
                               lambda v:layout_from_draft(v,inventory),
                               feedback=lambda v:layout_diagnostics(v,inventory),attempts=6),session=session)


def decide(session, source, layout, region, caps, directory, *, phase, previous=None, review=None):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    crop=crop_source(source,region['bbox'],directory/'source.png')
    evidence={'classification':region['classification'],'features':features(Image.open(crop)),
              'capabilities':caps}
    if previous is not None:
        evidence.update(previous_decision=previous['decision'],review=review)
    prompt=DECISION_PROMPT
    if previous is not None:
        prompt+='\nReassess the previous region scope using these bound review or author-validation findings. Use the exact capabilities.authoring_adapter.geometries list; do not invent native icon geometries. Preserve source fidelity; do not claim human approval. A syntax error alone is not a reason to retain a whole module.'
    return step(directory/'decision.json',{'prompt':prompt,'evidence':evidence,'crop_sha256':digest(crop),
                'layout':fingerprint(layout),'model':session.model},
                lambda:checked(session,phase,prompt,evidence,[crop],
                    lambda raw:decision_record(raw,region=region,layout=layout,crop_sha256=digest(crop),
                        capabilities=caps,response_sha256=session.last_response_sha256),tokens=5000),session=session)


def author(session, source, inventory, defaults, layout, region, decision, directory, *, phase,
           previous=None, review=None):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    crop=crop_source(source,region['bbox'],directory/'source.png')
    evidence={'source_sha256':inventory['source_sha256'],'image_size':inventory['image_size'],
              'region':transform_model_boxes(region,inventory['image_size'],to_model=True),
              'decision':decision['decision'],'decision_binding':decision['binding'],
              'text_defaults':text_candidates(defaults,inventory,region),
              'available_fonts':font_status()['families'],'allowed_geometries':sorted(GEOMETRIES)}
    prompt=AUTHOR_PROMPT
    if previous is not None:
        evidence['previous_elements']=transform_model_boxes(previous,inventory['image_size'],to_model=True)
        evidence['review']=transform_model_boxes(review,inventory['image_size'],to_model=True)
        prompt=PATCH_PROMPT
    def produce():
        if decision['decision']['decision']=='retain_module':
            value=retained_region(defaults,inventory,region,decision,layout)
            value['model_response_sha256']=decision['response_sha256']
            return value
        def accept(raw):
            value=expand_region(raw,defaults,inventory,region,decision,layout)
            for text in value['text_decisions']:
                if text['action']=='native': resolve_font(text,available=evidence['available_fonts'])
            if previous is not None: validate_correction(previous,value,review,region)
            return value
        def feedback(raw):
            # Reuse source OCR crop conflict diagnostics with explicit draft
            # coordinate convention; never rewrite the model's proposal.
            report={'region_diagnostics':element_diagnostics(raw,defaults,inventory,region,decision)}
            try:
                report['crop_diagnostics']=draft_feedback({'schema':'ElementPlanDraft/2','coordinate_space':'normalized_1000_xyxy',
                                   'objects':raw.get('objects',[])},inventory)
            except (ValueError,KeyError,TypeError):
                report['crop_diagnostics']='Malformed crop fields; correct schema first.'
            return report
        value=checked(session,phase,prompt,evidence,[source,crop],accept,feedback=feedback)
        value['model_response_sha256']=session.last_response_sha256
        return value
    value=step(directory/'elements.json',{'prompt':prompt,'evidence':evidence,'crop_sha256':digest(crop),
                'model':session.model},produce,session=session,allow_prompt_history=True)
    validate_region(value,inventory,region,decision,layout)
    if previous is not None: validate_correction(previous,value,review,region)
    return value


def routing_scope(decision):
    value=decision['decision']
    return (value['decision'],tuple(sorted(value['native_parts'])),tuple(sorted(value['retained_parts'])))


def initial_region(session,source,inventory,defaults,layout,region,caps,directory):
    """One bounded rerouting opportunity when the chosen scope cannot be authored."""
    rid=region['id']; directory=Path(directory)
    decision=decide(session,source,layout,region,caps,directory/'routing',phase='initial/decision/'+rid)
    try:
        elements=author(session,source,inventory,defaults,layout,region,decision,directory/'author',phase='initial/author/'+rid)
    except ValueError as exc:
        message=legacy.safe_error(exc)
        blocked=message.startswith('model_output_validation_exhausted:') and any(part in message for part in (
            'geometry must','native part mapped to raster','retained part must','empty or duplicated part mapping','element parts differ'))
        if not blocked: raise
        retry=directory/'redecision-1'; retry.mkdir(parents=True,exist_ok=True)
        failure={'schema':'InitialRegionScopeFailure/1','region_id':rid,'source_sha256':inventory['source_sha256'],
                 'decision_binding':decision['binding'],'response_sha256':session.last_response_sha256,'error':message}
        step(retry/'author-failure.json',failure,lambda:failure)
        revised=decide(session,source,layout,region,caps,retry/'routing',phase='initial/redecision/'+rid,
                       previous=decision,review=failure)
        require(routing_scope(revised)!=routing_scope(decision),'initial redecision did not change region scope; authoring remains blocked')
        elements=author(session,source,inventory,defaults,layout,region,revised,retry/'author',phase='initial/reauthor/'+rid)
        decision=revised
    return decision,elements


def build_page(session, source, inventory, layout, decisions, elements, revision, project_root, builder):
    revision=Path(revision); revision.mkdir(parents=True,exist_ok=True)
    plan,scopes,retained=assemble(layout,inventory,decisions,elements)
    parents={'layout':fingerprint(layout),'decisions':fingerprint(decisions),'elements':fingerprint(elements)}
    step(revision/'plan.json',parents,lambda:plan)
    step(revision/'regions.json',parents,lambda:{'decisions':decisions,'elements':elements})
    pointer=revision/'build-pointer.json'
    if pointer.exists():
        saved=read_json(pointer); directory=revision/saved['directory']; built=read_json(directory/'result.json')
        require(saved['sha256']==digest(directory/'result.json') and built['plan_sha256']==fingerprint(plan),'build pointer changed')
        verify_hashes(directory,built['files'])
    else:
        session.remaining()
        write_json(session.root/'progress.json',{'stage':'compile_and_actual_file_checks','revision':revision.name,**session.summary()})
        directory=revision/('b-'+uuid4().hex[:6])
        built=builder(source,inventory,plan,directory,project_root=project_root,deadline=session.deadline,
                      model_id=session.model,policy_binding=fingerprint({'policy':POLICY,**parents}),
                      response_sha256=fingerprint({rid:{'decision':d['response_sha256'],
                                      'author':elements[rid].get('model_response_sha256')} for rid,d in decisions.items()}),
                      raster_scopes=scopes,allowed_retained_ids=retained)
        save_new(pointer,{'directory':directory.name,'sha256':digest(directory/'result.json')})
    packets=region_packets(directory,layout)
    proof=step(directory/'actual-deletion-check.json',{'build':built['artifact_binding']},
               lambda:verify_deletion(directory/'evidence/result.pptx',directory/'evidence/delete-probe.pptx',built['probe_targets']))
    return {'revision':revision.name,'index':int(revision.name[1:]),'plan':plan,'decisions':deepcopy(decisions),
            'elements':deepcopy(elements),'built':built,'build_directory':directory.relative_to(session.root).as_posix(),
            'packets':packets,'deletion_check':proof,'reviews':{},'probe_reviews':{}}


def append_checks(review, checks):
    value=deepcopy(review)
    existing={i['id'] for i in value['issues']}
    value['issues'].extend(i for i in checks if i['id'] not in existing)
    if any(i['severity']=='blocking' for i in value['issues']): value['quality_passed']=False
    return value


def visual_review(session, candidate, region, inventory, previous=None):
    rid=region['id']; packet=candidate['packets'][rid]; element=candidate['elements'][rid]
    decision=candidate['decisions'][rid]; binding=candidate['built']['artifact_binding']
    directory=session.root/candidate['build_directory']; actual=read_json(directory/'evidence/objects.json')
    checks=typography_checks(candidate['plan'],region,element,actual)
    ids={o['id'] for o in element['objects']+element['text_decisions']}
    checks.extend(i for i in candidate['built']['structural_issues'] if set(i['object_ids'])&ids)
    evidence=transform_model_boxes({'region':region,'decision':decision,'artifact_binding':binding,
              'decision_binding':decision['binding'],'elements':element,'structural_checks':checks,
              'actual_objects':[o for o in actual['slides'][0]['objects'] if o['name'] in ids]},inventory['image_size'],to_model=True)
    path=directory/'reviews'/f'{rid}-visual.json'
    def produce():
        if previous and rid in previous['reviews'] and previous['elements'][rid]==element and previous['decisions'][rid]==decision:
            old=previous['packets'][rid]
            require(old['files']['preview.png']==packet['files']['preview.png'],'frozen region rendered differently: '+rid)
            inherited=deepcopy(previous['reviews'][rid]); inherited['artifact_binding']=binding
            inherited['inherited_from']={'artifact_binding':previous['built']['artifact_binding'],
                                         'review_sha256':fingerprint(previous['reviews'][rid]),
                                         'reason':'identical region elements, decision and rendered pixels'}
            return inherited
        raw=checked(session,'review/'+candidate['revision']+'/'+rid,VISUAL_PROMPT,evidence,
                    [packet['directory']/'source.png',packet['directory']/'preview.png'],
                    lambda v:validate_visual(v,region=region,elements=element,binding=binding,
                              decision_binding=decision['binding'],size=inventory['image_size']),tokens=7000)
        return append_checks(raw,checks)
    return step(path,{'prompt':VISUAL_PROMPT,'evidence':evidence,'images':packet['files'],
                     'previous_review':fingerprint(previous['reviews'].get(rid)) if previous else None},produce,session=session)


def overview_review(session,candidate,layout,inventory):
    directory=session.root/candidate['build_directory']; size=inventory['image_size']
    region={'id':'page','bbox':[0,0,*size]}; binding=candidate['built']['artifact_binding']
    decision_binding=fingerprint(candidate['decisions'])
    evidence=transform_model_boxes({'region':region,'artifact_binding':binding,'decision_binding':decision_binding,
              'region_layout':layout,'decisions':candidate['decisions'],'elements':candidate['plan'],
              'structural_checks':candidate['built']['structural_issues']},size,to_model=True)
    prompt=VISUAL_PROMPT+'\nThis is the full-page overview. Check missing areas between regions, alignment, source text, and overall reading order. Use existing object IDs where possible. Do not request whole-page raster retention.'
    return step(directory/'reviews/page-visual.json',{'prompt':prompt,'evidence':evidence},
                lambda:checked(session,'review/'+candidate['revision']+'/page',prompt,evidence,
                    [directory/'evidence/source.png',directory/'evidence/preview.png'],
                    lambda raw:validate_visual(raw,region=region,elements=candidate['plan'],binding=binding,
                         decision_binding=decision_binding,size=size),tokens=7000),session=session)


def probe_review(session,candidate,region,previous=None):
    rid=region['id']; directory=session.root/candidate['build_directory']; packet=candidate['packets'][rid]
    targets=[tid for tid in candidate['built']['probe_targets'] if tid in region['text_ids']]
    if not targets: return None
    binding=candidate['built']['artifact_binding']
    evidence={'artifact_binding':binding,'region_id':rid,'targets':targets,
              'actual_deletion_check':candidate['deletion_check'],
              'target_objects':[t for t in candidate['elements'][rid]['text_decisions'] if t['id'] in targets]}
    def produce():
        if previous and rid in previous['probe_reviews'] and previous['packets'][rid]['files']==packet['files']:
            old=previous['probe_reviews'][rid]
            result={**deepcopy(old),'artifact_binding':binding,'inherited_from':fingerprint(old)}
            return result
        return checked(session,'probe/'+candidate['revision']+'/'+rid,PROBE_PROMPT,evidence,
                       [packet['directory']/'preview.png',packet['directory']/'delete-probe.png'],
                       lambda raw:validate_probe(raw,region_id=rid,binding=binding,targets=targets),tokens=3500)
    return step(directory/'reviews'/f'{rid}-probe.json',{'prompt':PROBE_PROMPT,'evidence':evidence,
                'images':packet['files'],'previous':fingerprint(previous['probe_reviews'].get(rid)) if previous else None},produce,session=session)


def correction_review(candidate,region):
    """Derive a region task from immutable reviews without changing raw responses."""
    rid=region['id']; result=deepcopy(candidate['reviews'][rid]); ids={o['id'] for o in candidate['elements'][rid]['objects']+candidate['elements'][rid]['text_decisions']}
    for item in candidate['overview']['issues']:
        refs=set(item['object_ids'])&ids
        if refs or (not item['object_ids'] and _intersection(item['bbox'],region['bbox'])>0):
            derived=deepcopy(item); derived['object_ids']=sorted(refs)
            a,b=item['bbox'],region['bbox']; x,y=max(a[0],b[0]),max(a[1],b[1]); w=min(a[0]+a[2],b[0]+b[2])-x; h=min(a[1]+a[3],b[1]+b[3])-y
            if w<=0 or h<=0: derived['bbox']=region['bbox']
            else: derived['bbox']=[x,y,w,h]
            derived['derived_from_overview']=fingerprint(candidate['overview'])
            result['issues'].append(derived)
    probe=candidate['probe_reviews'].get(rid)
    if probe and not probe['ghost_free']:
        raster_ids=[o['id'] for o in candidate['elements'][rid]['objects'] if o['kind']=='raster_asset']
        for finding in probe['findings']:
            result['issues'].append({'id':'ghost-'+finding['target_id'],'kind':'crop','severity':'blocking',
                'object_ids':[finding['target_id']]+raster_ids,'bbox':region['bbox'],'description':finding['description'],
                'derived_from_probe':fingerprint(probe)})
    result['quality_passed']=result['quality_passed'] and not any(i['severity']=='blocking' for i in result['issues'])
    return result


def review_page(session,candidate,layout,inventory,previous=None):
    for region in layout['regions']:
        rid=region['id']; candidate['reviews'][rid]=visual_review(session,candidate,region,inventory,previous)
        probe=probe_review(session,candidate,region,previous)
        if probe: candidate['probe_reviews'][rid]=probe
    candidate['overview']=overview_review(session,candidate,layout,inventory)
    issues=[i for r in candidate['reviews'].values() for i in r['issues']]+candidate['overview']['issues']
    for rid,probe in candidate['probe_reviews'].items():
        if not probe['ghost_free']:
            region=next(r for r in layout['regions'] if r['id']==rid)
            issues.extend(i for i in correction_review(candidate,region)['issues'] if i.get('derived_from_probe'))
    if not candidate['deletion_check']['passed']:
        issues.append(issue(None,'probe',[0,0,*inventory['image_size']],'Actual-file deletion or frozen-object verification failed.'))
    passed=(all(r['quality_passed'] for r in candidate['reviews'].values()) and candidate['overview']['quality_passed']
            and candidate['deletion_check']['passed'] and all(p['ghost_free'] for p in candidate['probe_reviews'].values()))
    candidate['review']={'schema':'AutomaticReview/2','artifact_binding':candidate['built']['artifact_binding'],
                         'quality_passed':passed,'edit_probe_clear':candidate['deletion_check']['passed'] and all(p['ghost_free'] for p in candidate['probe_reviews'].values()),
                         'issues':issues,'summary':'Decision-scoped regional and full-page visual review with separate actual deletion checks.'}
    directory=session.root/candidate['build_directory']
    if previous:
        transitions={}
        for region in layout['regions']:
            rid=region['id']; old=previous['elements'][rid]; new=candidate['elements'][rid]
            a={o['id']:o for o in old['objects']+old['text_decisions']}; b={o['id']:o for o in new['objects']+new['text_decisions']}
            changed=sorted(oid for oid in set(a)|set(b) if a.get(oid)!=b.get(oid))
            transitions[rid]=check_transitions(previous['reviews'][rid],candidate['reviews'][rid],changed)
            transitions[rid]['render_changed']=previous['packets'][rid]['files']['preview.png']!=candidate['packets'][rid]['files']['preview.png']
            if changed and not transitions[rid]['render_changed']:
                candidate['review']['quality_passed']=False
                candidate['review']['issues'].append(issue(None,'layout',region['bbox'],'Element repair produced no visible pixel change; improvement is unverified.'))
        step(directory/'issue-transitions.json',{'before':previous['built']['artifact_binding'],'after':candidate['built']['artifact_binding']},lambda:transitions)
    step(directory/'reviews/aggregate.json',{'visual':candidate['reviews'],'overview':candidate['overview'],
         'probe':candidate['probe_reviews'],'deletion':candidate['deletion_check'],
         'previous_build':previous['built']['artifact_binding'] if previous else None},lambda:candidate['review'])
    return candidate


def finish_page(session,candidates,errors,layout=None):
    reviewed=[c for c in candidates if c.get('review')]
    pool=reviewed or candidates
    def rank(c):
        review=c.get('review',{}); issues=review.get('issues',c['built']['structural_issues'])
        return (not review.get('quality_passed',False),sum(i['severity']=='blocking' for i in issues),len(issues),
                c['built']['metrics']['mean_absolute_pixel_difference'],-c['index'])
    chosen=min(pool,key=rank) if pool else None
    review=chosen.get('review') if chosen else None
    result={'schema':'AutomaticPageResult/2','pipeline_version':2,
            'execution_status':'COMPLETED' if review else 'PARTIAL' if chosen else 'FAILED',
            'pipeline_completed':bool(review),'review_status':'VALIDATED' if review else 'INCOMPLETE',
            'machine_quality_status':'PASSED' if review and review['quality_passed'] else 'REVIEW_REQUIRED' if chosen else 'FAILED',
            'errors':errors,'model':session.model,'human_acceptance':False,'office_acceptance':False,
            'revision_count':len(candidates),'regions':[],**session.summary()}
    if chosen:
        result.update(build_directory=chosen['build_directory'],artifact_binding=chosen['built']['artifact_binding'],
                      selected_revision=chosen['revision'],counts=chosen['built']['counts'],metrics=chosen['built']['metrics'],
                      issues=review['issues'] if review else chosen['built']['structural_issues'],
                      selection_reason='Verified regional/actual-file findings first; pixel difference is only a tie-breaker.')
        for region in layout['regions']:
            rid=region['id']; decision=chosen['decisions'][rid]
            result['regions'].append({'id':rid,'bbox':region['bbox'],'decision':decision['decision'],
                'execution_status':'BUILT',
                'decision_binding':decision['binding'],'mapping':chosen['elements'][rid]['mapping'],
                'quality_status':'PASSED' if chosen['reviews'].get(rid,{}).get('quality_passed') else 'REVIEW_REQUIRED',
                'issues':chosen['reviews'].get(rid,{}).get('issues',[]),
                'native_text_ids':[t['id'] for t in chosen['elements'][rid]['text_decisions'] if t['action']=='native'],
                'retained_text_ids':[t['id'] for t in chosen['elements'][rid]['text_decisions'] if t['action']=='retained_in_asset']})
    elif layout:
        for region in layout['regions']:
            rid=region['id']; path=session.root/'initial'/rid
            revised=path/'redecision-1'
            if not (path/'author/elements.json').exists() and (revised/'routing/decision.json').exists():
                original=read_json(path/'routing/decision.json'); replacement=read_json(revised/'routing/decision.json')
                if routing_scope(original)!=routing_scope(replacement): path=revised
            decision=read_json(path/'routing/decision.json') if (path/'routing/decision.json').exists() else None
            element=read_json(path/'author/elements.json') if (path/'author/elements.json').exists() else None
            result['regions'].append({'id':rid,'bbox':region['bbox'],
                'decision':decision['decision'] if decision else {'decision':'PENDING'},
                'decision_binding':decision['binding'] if decision else None,
                'execution_status':'AUTHORED_NOT_BUILT' if element else 'DECIDED' if decision else 'PENDING',
                'mapping':element['mapping'] if element else {},'quality_status':'INCOMPLETE','issues':[],
                'native_text_ids':[],'retained_text_ids':[]})
    save_new(session.root/'result.json',result)
    return result


def run_page(directory,page,manifest,config,model,*,project_root,remote=None,builder=build_revision):
    require(manifest.get('schema')=='AutomaticJob/2' and manifest.get('pipeline_version')==2,'v2 rejects legacy jobs')
    directory=Path(directory); source=directory/'source.png'
    require(digest(source)==page['source_sha256'] and digest(directory/'original.image')==page['original_sha256'],'input source changed')
    if (directory/'result.json').exists():
        result=read_json(directory/'result.json'); require(result.get('pipeline_version')==2,'legacy page cannot resume in v2')
        if result.get('build_directory'):
            build=directory/result['build_directory']; built=read_json(build/'result.json')
            verify_hashes(build,built['files']); require(built['artifact_binding']==result['artifact_binding'],'selected build changed')
        return result
    session=legacy.PageSession(directory,remote or SSHRemote(config),model,policy=manifest['policy'])
    candidates=[]; errors=[]; layout=None
    try:
        analysis=legacy.get_analysis(session,source); inventory=legacy.get_inventory(session,source,analysis)
        caps=capabilities(project_root,session)
        defaults=legacy.get_text_defaults(session,source,inventory,project_root)
        layout=get_layout(session,source,inventory,caps)
        decisions={}; elements={}
        for region in layout['regions']:
            rid=region['id']; path=directory/'initial'/rid
            decisions[rid],elements[rid]=initial_region(session,source,inventory,defaults,layout,region,caps,path)
        previous=None
        for index in range(1+manifest['policy']['max_corrections']):
            revision=directory/f'r{index:03d}'
            if index:
                changed=False
                for region in layout['regions']:
                    rid=region['id']; review=correction_review(previous,region)
                    if review['quality_passed']: continue
                    path=directory/'corrections'/f'r{index:03d}'/rid
                    try:
                        if review['redecision_required']:
                            new_decision=decide(session,source,layout,region,caps,path/'routing',phase=f'redecision/{index}/{rid}',previous=decisions[rid],review=review)
                            require(routing_scope(new_decision)!=routing_scope(decisions[rid]),'redecision did not change region scope')
                            new_elements=author(session,source,inventory,defaults,layout,region,new_decision,path/'author',phase=f'reauthor/{index}/{rid}')
                            decisions[rid]=new_decision; elements[rid]=new_elements
                        else:
                            new_elements=author(session,source,inventory,defaults,layout,region,decisions[rid],path/'author',phase=f'patch/{index}/{rid}',previous=elements[rid],review=review)
                            validate_correction(elements[rid],new_elements,review,region)
                            elements[rid]=new_elements
                        changed=True
                    except (ValueError,KeyError,TypeError) as exc:
                        record={'revision':index,'region_id':rid,'code':legacy.safe_error(exc),'type':type(exc).__name__}
                        errors.append(record)
                        path.mkdir(parents=True,exist_ok=True)
                        if not (path/'error.json').exists(): save_new(path/'error.json',record)
                if not changed: break
            current=build_page(session,source,inventory,layout,decisions,elements,revision,project_root,builder)
            candidates.append(current)
            review_page(session,current,layout,inventory,previous)
            previous=current
            write_json(directory/'progress.json',{'stage':'reviewed','revision':current['revision'],
                       'pipeline_version':2,'region_count':len(layout['regions']),
                       'machine_quality_status':'PASSED' if current['review']['quality_passed'] else 'REVIEW_REQUIRED',**session.summary()})
            if current['review']['quality_passed']: break
    except Exception as exc:
        errors.append({'code':legacy.safe_error(exc),'type':type(exc).__name__})
    return finish_page(session,candidates,errors,layout)


def deliver(root,manifest,results):
    report=legacy.deliver(root,manifest,results)
    report.update(schema='AutomaticDelivery/2',pipeline_version=2,gate_only=manifest['gate_only'])
    write_json(Path(root)/'delivery/report.json',report)
    for page,result in results.items():
        write_json(Path(root)/'delivery'/(page+'-regions.json'),{'pipeline_version':2,'page':page,
                   'regions':result.get('regions',[]),'errors':result.get('errors',[]),'human_acceptance':False})
    return report


def run_job(root,*,project_root,preflight_directory=None,remote=None,builder=build_revision,retry_failed=False):
    root=Path(root).resolve(strict=True)
    with run_lock(root):
        manifest=read_json(root/'job.json'); config=read_json(root/'connection.json'); state=read_json(root/'state.json')
        require(manifest.get('schema')=='AutomaticJob/2' and manifest.get('pipeline_version')==2,'v2 refuses to resume a legacy job; create an isolated new job')
        require(manifest['policy']==POLICY,'v2 policy differs from the bounded supported contract')
        require(state['job_sha256']==digest(root/'job.json') and manifest['config_sha256']==fingerprint(config),'job binding changed')
        remote=remote or SSHRemote(config); results={}
        state.update(status='RUNNING',stage='preflight',pipeline_version=2); write_json(root/'state.json',state)
        try:
            local=root/'preflight'
            if not local.exists():
                if preflight_directory:
                    legacy.verified_preflight(preflight_directory,config); shutil.copytree(preflight_directory,local)
                else: preflight(remote,config,local)
            selected=legacy.verified_preflight(local,config)['selected_model']
            order=manifest['gate_pages'] if manifest['gate_only'] else manifest['gate_pages']+[p['page'] for p in manifest['pages'] if p['page'] not in manifest['gate_pages']]
            for i,number in enumerate(order):
                if i==len(manifest['gate_pages']) and any(results[f'P{n}']['machine_quality_status']!='PASSED' for n in manifest['gate_pages']):
                    state.update(status='GATE_REVIEW_REQUIRED',stage='finished',skipped_pages=[f'P{n}' for n in order[i:]])
                    break
                page=manifest['pages'][number-1]; directory=root/page['id']; result_path=directory/'result.json'
                if retry_failed and result_path.exists():
                    old=read_json(result_path); budget=read_json(directory/'budget.json')
                    if old['execution_status']=='FAILED' and old['model_requests']<budget['max_calls'] and time.time()<budget['started_at']+budget['timeout_seconds']:
                        history=directory/'previous-results'; history.mkdir(exist_ok=True)
                        saved=history/('failed-'+uuid4().hex[:8]+'.json'); shutil.copyfile(result_path,saved)
                        require(digest(saved)==digest(result_path),'failed result preservation failed'); result_path.unlink()
                state.update(stage='page',current_page=page['id']); write_json(root/'state.json',state)
                result=run_page(directory,page,manifest,config,selected,project_root=project_root,remote=remote,builder=builder)
                results[page['id']]=result; state['pages'][page['id']]=result
                write_json(root/'state.json',state); deliver(root,manifest,results)
            else:
                passed=all(r['machine_quality_status']=='PASSED' for r in results.values())
                state.update(status='GATE_PASSED' if passed and manifest['gate_only'] else 'COMPLETED' if passed else 'GATE_REVIEW_REQUIRED',stage='finished')
            skipped=[p['id'] for p in manifest['pages'] if p['id'] not in results]
            deliver(root,manifest,results)
            state.update(delivery='delivery/README.md',selected_model=selected,skipped_pages=skipped,
                         machine_quality_status='PASSED' if results and all(r['machine_quality_status']=='PASSED' for r in results.values()) else 'REVIEW_REQUIRED',
                         human_acceptance=False,office_acceptance=False)
        except Exception as exc:
            state.update(status='FAILED',stage='finished',error=legacy.safe_error(exc))
            write_json(root/'state.json',state); raise
        write_json(root/'state.json',state)
        return {'job':str(root),**state}
