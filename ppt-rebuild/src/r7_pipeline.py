"""Source-bound r7 orchestration. No model calls, no mutation of historical scenes."""
from __future__ import annotations
import argparse
import copy
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from rebuild_pipeline import ArtifactBackend, REPO, ROOT, compile_ir, pixels, stamp
from slideir_adapter import sha256, write_json
from reconstruction_transaction import RunLock, ReconstructionTransaction, scene_fingerprint, _atomic_write
from object_resolver import from_ir, registry_from_ir, resolve, validate_ir_owners
from relation_validator import prepare_spec, check_relations, check_authorized_changes
from r7_candidates import build_candidate
from background_models import inspect_residue
from text_repair_acceptance import delete_deck_text, verify_styles
from surface_acceptance import verify_native_shape_properties
from visual_eval import metrics
from ocr_engine import detect
from reconstruction import intersect
from r7_baseline import load_baseline
from qa_status import aggregate_checks, summarize_edit_report
from r7_runtime import freeze_runtime


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def clone_scene(irfile, output):
    irfile, output = Path(irfile), Path(output)
    record=scene_fingerprint(irfile)
    output.mkdir(parents=True,exist_ok=False)
    for relative in record['files']:
        target=output/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(irfile.parent/relative,target)
    return output/irfile.name


def owners(ir):
    return validate_ir_owners(ir)


def frozen_scene(before, after, removed=(), added=(), *, allowed_fields=None):
    allowed_fields = allowed_fields or {}
    for field in ('schema_version','background','canvas','source_image','source_sha256'):
        if before.get(field)!=after.get(field): raise ValueError('Frozen scene metadata changed: '+field)
    evidence={item['id']:item for item in after['evidence']}
    if len(evidence)!=len(after['evidence']): raise ValueError('Duplicate source evidence')
    for item in before['evidence']:
        if evidence.get(item['id'])!=item: raise ValueError('Frozen source evidence changed: '+item['id'])
    a={n['id']:n for n in before['nodes']}; b={n['id']:n for n in after['nodes']}
    if set(a)-set(b)!=set(removed) or set(b)-set(a)!=set(added):
        raise ValueError('Undeclared scene object change')
    for identifier in set(a)&set(b):
        left,right=copy.deepcopy(a[identifier]),copy.deepcopy(b[identifier])
        left.pop('z');right.pop('z')
        permission = allowed_fields.get(identifier, {})
        if permission:
            if set(permission) != {'bbox'}:
                raise ValueError('Unsupported authorized field change')
            rule = permission['bbox']
            if left.get('bbox') != rule.get('before') or right.get('bbox') != rule.get('after'):
                raise ValueError('Authorized bbox change differs from receipt')
            if any(left['bbox'][i] != right['bbox'][i] for i in (0, 2)):
                raise ValueError('Only vertical container bounds may change')
            right['bbox'] = left['bbox']
        if left!=right: raise ValueError('Frozen existing object changed: '+identifier)
    if set(allowed_fields) - (set(a) & set(b)):
        raise ValueError('Authorized object missing')
    order=lambda ir:[n['id'] for n in sorted(ir['nodes'],key=lambda x:x['z']) if n['id'] in set(a)&set(b)]
    if order(before)!=order(after): raise ValueError('Existing relative z order changed')
    if before.get('groups',[])!=after.get('groups',[]): raise ValueError('Existing edit groups changed')
    assets={x['id']:x for x in after['assets']}
    for item in before['assets']:
        if assets.get(item['id'])!=item: raise ValueError('Frozen original asset changed')
    return {'status':'PASS','existing_text_and_styles':'UNCHANGED','existing_assets':'UNCHANGED'}


def environment(input_path, batch, backend):
    if '_baseline_context' not in batch:
        batch = load_baseline(input_path)
    context = batch['_baseline_context']
    for path, expected in context['evidence_files'].items():
        if sha256(path) != expected:
            raise ValueError('Frozen baseline dependency changed: '+path)
    catalog=batch.get('_relations')
    if catalog and sha256(catalog['path']) != catalog['sha256']:
        raise ValueError('Frozen relation catalog changed')
    fonts = context['font_manifest']['fonts']
    if not fonts: raise ValueError('Baseline verified font manifest required')
    for font in fonts['fonts']:
        if sha256(font['file'])!=font['sha256']: raise ValueError('Baseline font file changed')
    code={str(p.relative_to(REPO)):sha256(p) for folder in [ROOT/'src',REPO/'src/image2ppt',REPO/'scripts'] for p in folder.glob('*') if p.suffix in ('.py','.mjs')}
    code['ppt-rebuild/compiler_bridge.mjs']=sha256(ROOT/'compiler_bridge.mjs')
    page_refs=[]
    for page in batch['pages']:
        if sha256(page['ir'])!=page['ir_sha256'] or sha256(page['pptx'])!=page['pptx_sha256']:
            raise ValueError('Baseline IR/PPTX changed')
        ir=read(page['ir'])
        if sha256(Path(page['ir']).parent/ir['source_image'])!=page['source_sha256']: raise ValueError('Source changed')
        masks=Path(page['evidence_baseline']['directory'])/'text-masks'
        evidence={str(p.resolve()):sha256(p) for p in masks.glob('*.png')}
        if not evidence: raise ValueError('Frozen source text masks missing: '+page['page'])
        page_refs.append({'page':page['page'],'scene':scene_fingerprint(page['ir']),
                          'pptx_sha256':sha256(page['pptx']),'source_masks':evidence,
                          'expected_probes':page['expected_probes'],'baseline_refs':page['baseline_refs']})
    return {'schema':'R7RunInputs/1','baseline':str(Path(input_path).resolve()),'baseline_sha256':sha256(input_path),
            'pages':page_refs,'fonts':fonts,'code':code,'runtime':freeze_runtime(backend),
            'machine':platform.node(),'python':platform.python_version(),'model_calls':0,
            'baseline_code':'be31355','source_layerd_rerun':False,
            'baseline_context':context,'relation_catalog':batch.get('_relations')}


def deletion(delivery, output, backend):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/'deleted.pptx'
    delete_deck_text(delivery['pptx'],target,[delivery])
    rendered=backend.render(target,output/'render')
    return str(target),rendered['pages'][0]['path']


def clean_check(baseline, source_path, preview, output):
    """Independent frozen source glyph masks + OCR, not only plate equality."""
    baseline=baseline.get('evidence_baseline',baseline)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    ir=read(baseline['ir']);size=ir['canvas']['source_px']
    source=pixels(source_path,size);actual=pixels(preview,size)
    rows=[];maskroot=Path(baseline['directory'])/'text-masks'
    for glyph in sorted(maskroot.glob('*-glyph.png')):
        identifier=glyph.name[:-len('-glyph.png')]
        inspection=maskroot/(identifier+'-inspection.png')
        if not inspection.exists(): raise ValueError('Frozen inspection missing')
        core=np.asarray(Image.open(glyph).convert('L'))>0
        inspect=np.asarray(Image.open(inspection).convert('L'))>0
        measured=inspect_residue(source,actual,core,inspect)
        fraction=measured.get('residual_fraction')
        if (measured.get('status')=='REVIEW_REQUIRED' and type(measured.get('checked_pixels')) is int
                and measured['checked_pixels']>0 and isinstance(fraction,(int,float))
                and math.isfinite(fraction) and fraction>.01):
            measured.update(prior_status='REVIEW_REQUIRED',status='FAIL',
                            reason='SOURCE_STROKE_RESIDUE_LIMIT_EXCEEDED',limit=.01,observed=fraction)
        rows.append({'id':identifier,'glyph_sha256':sha256(glyph),'inspection_sha256':sha256(inspection),**measured})
    enhanced=output/'contrast.png';ImageEnhance.Contrast(Image.open(preview).convert('RGB')).enhance(3).save(enhanced)
    boxes=[e['bbox'] for e in ir['evidence'] if e.get('kind')=='text']
    hits=[]
    for file in (preview,enhanced):
        for hit in detect(file,min_score=.45):
            if any(intersect(hit.bbox,b)>.5*hit.bbox[2]*hit.bbox[3] for b in boxes):
                hits.append({'text':hit.text,'bbox':hit.bbox})
    expected=[item['id'] for item in baseline.get('text_checks',[])]
    found=[item['id'] for item in rows]
    complete=bool(expected) and len(set(expected))==len(expected) and set(found)==set(expected)
    checks=aggregate_checks({**{r['id']:r for r in rows},
                            'coverage':{'status':'PASS' if complete else 'REVIEW_REQUIRED','reason':'source_stroke_inventory'},
                            'ocr':{'status':'PASS' if not hits else 'REVIEW_REQUIRED','reason':'ocr_hits_require_source_review'}})
    passed=checks['status']=='PASS'
    result={'schema':'R7CleanPlateQA/1','source_sha256':sha256(source_path),'method':'frozen_source_strokes_and_independent_ocr',
            'status':checks['status'],'checks':checks['checks'],
            'paragraphs':rows,'ocr_hits':hits,'scope':'native_text_removal_only',
            'mask_coverage':{'status':'PASS' if complete else 'REVIEW_REQUIRED','expected_line_ids':expected,
                             'missing':sorted(set(expected)-set(found)),'unexpected':sorted(set(found)-set(expected))},
            'verified_target_ids':[n['id'] for n in ir['nodes'] if n['type']=='text'] if passed else []}
    write_json(output/'report.json',result)
    return result


def checkpoint_files(result):
    paths={Path(result[key]).resolve() for key in ('ir','pptx','preview','deleted_text_pptx',
                                                  'deleted_text_preview','source','r6_preview')}
    root=Path(result['execution_directory'])
    for folder in ('relations','registry','final-clean','transactions','fault-injection','cover-container'):
        paths.update(p.resolve() for p in (root/folder).rglob('*') if p.is_file())
    paths.update(p.resolve() for p in root.glob('candidate-*-gates.json'))
    return {str(path):sha256(path) for path in sorted(paths)}


def validate_qa_environment(parent, current, *, allow_code_change=False):
    """Only explicit versioned QA may use new validators; data/runtime stay frozen."""
    before=copy.deepcopy(parent);after=copy.deepcopy(current)
    old_code=before.pop('code',None);new_code=after.pop('code',None)
    if before!=after:
        changed=sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))
        raise ValueError('QA frozen input/font/runtime/environment changed: '+', '.join(changed))
    if not old_code or not new_code: raise ValueError('QA code fingerprints missing')
    changes=sorted(k for k in set(old_code)|set(new_code) if old_code.get(k)!=new_code.get(k))
    if changes and not allow_code_change:
        raise ValueError('QA code changed; use --output with a new independent QA directory')
    validators={'ppt-rebuild/src/qa_status.py','ppt-rebuild/src/r7_edit_checks.py'}
    if allow_code_change and any(k.replace('\\','/') not in validators for k in changes):
        raise ValueError('QA generation dependency changed; create a new reconstruction run')
    return {'allowed_change':'validation_code_only' if allow_code_change else 'none','changed_code_files':changes}


def validate_completed_batch(batch):
    """Verify immutable page checkpoints and the exact composed 12-page deck."""
    baseline=read(batch['input'])
    pages=batch['pages']
    if (len(pages)!=12 or len(baseline['pages'])!=12
            or [p['page'] for p in pages]!=[p['page'] for p in baseline['pages']]
            or len({p['page'] for p in pages})!=12):
        raise ValueError('QA requires all twelve original pages in their original order')
    for page in pages:
        if (scene_fingerprint(page['ir'])['sha256']!=page['scene_sha256']
                or sha256(page['ir'])!=page['ir_sha256'] or sha256(page['pptx'])!=page['pptx_sha256']):
            raise ValueError('QA completed scene or PPTX changed: '+page['page'])
        if not page.get('cache_dependency_hashes') or checkpoint_files(page)!=page['cache_dependency_hashes']:
            raise ValueError('QA completed render/spec/mask/receipt changed: '+page['page'])
    quality_path=Path(batch['deck']['quality']);quality=read(quality_path)
    deck_hash=sha256(batch['deck']['pptx'])
    if quality.get('slide_count')!=12 or quality.get('pptx_sha256')!=deck_hash:
        raise ValueError('QA composed deck does not match its frozen compiler receipt')
    if [p.get('ir_sha256') for p in quality.get('pages',[])]!=[p['ir_sha256'] for p in pages]:
        raise ValueError('QA composed deck page order/IR differs from completed scenes')
    return {'deck_sha256':deck_hash,'compiler_receipt_sha256':sha256(quality_path),
            'scene_sha256':[p['scene_sha256'] for p in pages]}


def qa_run(parent_run, output=None):
    """Re-audit immutable deliverables; an explicit new directory versions QA code."""
    parent=Path(parent_run).resolve()
    if not (parent/'batch.json').is_file() or not (parent/'run-manifest.json').is_file():
        raise ValueError('Completed r7 batch and manifest required')
    destination=Path(output).resolve() if output else parent
    versioned=output is not None
    if versioned and (destination.exists() or destination.is_relative_to(parent) or parent.is_relative_to(destination)):
        raise ValueError('Versioned QA output must be a new independent directory outside the parent run')
    backend=ArtifactBackend(REPO,600)
    with RunLock(parent):
        batch=read(parent/'batch.json');old_manifest=read(parent/'run-manifest.json')
        normalized=load_baseline(batch['input']);normalized['_relations']=old_manifest.get('relation_catalog')
        current=environment(batch['input'],normalized,backend)
        policy=validate_qa_environment(old_manifest,current,allow_code_change=versioned)
        completed=validate_completed_batch(batch)
        parent_hashes={'manifest_sha256':sha256(parent/'run-manifest.json'),
                       'batch_sha256':sha256(parent/'batch.json'),**completed}
        if versioned:
            destination.mkdir(parents=True,exist_ok=False)
            with RunLock(destination):
                shutil.copy2(parent/'run-manifest.json',destination/'parent-run-manifest.json')
                shutil.copy2(parent/'batch.json',destination/'parent-batch.json')
                link={'schema':'R7QAParent/1','parent_run':str(parent),'hashes':parent_hashes,
                      'original_reconstruction_code':old_manifest['code'],
                      'current_validation_code':current['code'],'policy':policy,
                      'reconstruction_rerun':False,'parent_artifacts_modified':False}
                write_json(destination/'parent-link.json',link)
                write_json(destination/'run-manifest.json',current)
                copied=copy.deepcopy(batch);copied['directory']=str(destination)
                write_json(destination/'batch.json',copied)
                checked=audit(copied,destination,backend)
                # Catch unexpected writes during probes before issuing a new delivery.
                if (sha256(parent/'run-manifest.json')!=parent_hashes['manifest_sha256']
                        or sha256(parent/'batch.json')!=parent_hashes['batch_sha256']
                        or validate_completed_batch(batch)!=completed):
                    raise ValueError('Parent reconstruction artifacts changed during QA')
                validate_qa_environment(current,environment(batch['input'],normalized,backend))
                return package(copied,checked,destination)
        checked=audit(batch,parent,backend)
        validate_qa_environment(current,environment(batch['input'],normalized,backend))
        if validate_completed_batch(batch)!=completed:
            raise ValueError('Parent reconstruction artifacts changed during QA')
        return package(batch,checked,parent)


def process_page(page, run, backend, *, relation_config=None):
    prefix=page['page'][:2];directory=Path(run)/'pages'/prefix;done=directory/'result.json'
    if done.exists():
        result=read(done)
        if scene_fingerprint(result['ir'])['sha256']!=result['scene_sha256'] or sha256(result['pptx'])!=result['pptx_sha256']:
            raise ValueError('Completed page changed')
        if not result.get('cache_dependency_hashes') or checkpoint_files(result)!=result['cache_dependency_hashes']:
            raise ValueError('Completed page render/spec/mask/receipt changed')
        return result
    directory.mkdir(parents=True,exist_ok=True)
    attemptdir=directory/('execution-'+stamp());attemptdir.mkdir()
    original=read(page['ir']);source=Path(page['ir']).parent/original['source_image'];size=original['canvas']['source_px']
    from_ir(page['ir'],attemptdir/'registry')
    spec=prepare_spec(prefix,source,attemptdir/'relations',config=relation_config,ir=page['ir'])
    base_ir=clone_scene(page['ir'],attemptdir/'baseline')
    base=compile_ir(backend,base_ir)
    baseline_check=check_relations(source,base['preview'],spec)
    write_json(attemptdir/'baseline-relation-qa.json',baseline_check)
    result={**page,**base,'ir':str(base_ir),'action':'NO_OP','attempts':[]}
    decisions=[];repair=[]
    measured={row['id']:row for row in baseline_check.get('objects',[])}
    for element in spec['elements']:
        row=measured.get(element['id'],{'status':'REVIEW_REQUIRED','reason':'relation_not_executed'})
        state=row.get('status')
        if state not in {'PASS','FAIL','REVIEW_REQUIRED','UNKNOWN','NOT_APPLICABLE'}:
            raise ValueError('Invalid relation status: '+str(state))
        action=('NO_OP' if state=='PASS' else 'REPAIR' if state=='FAIL' and element.get('repairable',False)
                else 'REVIEW_REQUIRED')
        decisions.append({'id':element['id'],'action':action,'check':row})
        if action=='REPAIR':repair.append(element['id'])
    result['object_decisions']=decisions
    if any(d['action']=='REVIEW_REQUIRED' for d in decisions):result['action']='BASELINE_RETAINED'
    if prefix=='01':
        from r71_cover import attempt_cover
        result=attempt_cover(page,base_ir,result,attemptdir,backend,frozen_scene)
    elif repair:
        result['action']='BASELINE_RETAINED'
        raster_ids=[];parent_fingerprint=scene_fingerprint(base_ir)
        immutable_inputs={str(source):sha256(source),str(attemptdir/'relations/relations.json'):sha256(attemptdir/'relations/relations.json')}
        for element in spec['elements']:
            immutable_inputs.update({ref['path']:ref['sha256'] for ref in element.get('masks',{}).values() if isinstance(ref,dict) and 'path' in ref})
        for number in (1,2):
            tx=None;tx_id=f'correction-{prefix}-{number}'
            try:
                candidate=build_candidate(base_ir,spec,attemptdir/f'candidate-{number}',raster_ids=raster_ids,
                                          object_ids=repair,tx_id=tx_id,parent_scene_sha256=parent_fingerprint['sha256'])
                required=['relations','frozen','write_boundary','text_clean','render_integrity']
                if candidate['raster_records']:required.append('raster_fit')
                tx=ReconstructionTransaction(attemptdir/'transactions',tx_id,base_ir,
                        {'source':sha256(source),'spec':fingerprint(spec)},required,candidate['added_ids'],
                        authorization=candidate['authorization'])
                frozen=tx.prepare(candidate['ir']);trial=compile_ir(backend,frozen)
                relations=check_relations(source,trial['preview'],spec)
                measured_trial={row['id']:row for row in relations['objects']}
                selected={identifier:measured_trial.get(identifier,{'status':'FAIL','reason':'required_relation_missing'}) for identifier in repair}
                relation_gate=aggregate_checks(selected)
                struct=frozen_scene(original,read(frozen),candidate['removed_ids'],candidate['added_ids'])
                allowed=np.asarray(Image.open(candidate['authorized_mask']).convert('L'))>0
                rendered_allowed=cv2.dilate(allowed.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
                boundary=check_authorized_changes(pixels(base['preview'],size),pixels(trial['preview'],size),rendered_allowed)
                deleted,naked=deletion({**result,**trial,'ir':str(frozen)},attemptdir/f'candidate-{number}-deleted',backend)
                cleanliness=clean_check(page,source,naked,attemptdir/f'candidate-{number}-clean')
                fits=[r['fit'] for item in candidate['raster_records'] for r in item['records']]
                fit_gate=({'status':'PASS' if all(r['mae']<=2 and r['p95']<=5 for r in fits) else 'FAIL','records':fits}
                          if fits else {'status':'NOT_APPLICABLE','reason':'no_raster_candidate'})
                gates={'relations':relation_gate,'frozen':struct,'write_boundary':boundary,'text_clean':cleanliness,
                       'raster_fit':fit_gate,'render_integrity':{'status':'PASS','preview_sha256':sha256(trial['preview'])}}
                gate_path=attemptdir/f'candidate-{number}-gates.json';write_json(gate_path,gates)
                verdict=tx.verify(gates,owners,evidence_files={'render':trial['preview'],'qa':str(gate_path)})
                record={'number':number,'raster_ids':list(raster_ids),'gates':gates,'added_ids':candidate['added_ids']}
                if verdict['state']=='VERIFIED':
                    receipt=tx.commit();record['state']='COMMITTED'
                    result.update(trial,ir=str(receipt['ir']),action='UPDATED',deleted_text_pptx=deleted,deleted_text_preview=naked)
                    result['attempts'].append(record);break
                tx.rollback('required gates not PASS');record['state']='ROLLED_BACK';result['attempts'].append(record)
                raster_ids=[row['id'] for row in relations['objects'] if row['id'] in repair and row['status']=='FAIL']
                if not raster_ids:break
            except Exception as error:
                if scene_fingerprint(base_ir)!=parent_fingerprint or any(sha256(p)!=expected for p,expected in immutable_inputs.items()):
                    raise ValueError('Immutable candidate inputs changed') from error
                if tx is not None and tx.prepare_file.exists():tx.rollback('candidate exception: '+str(error))
                failure={'number':number,'state':'ROLLED_BACK','status':getattr(error,'status','FAIL'),
                         'error_type':type(error).__name__,'reason':str(error)}
                result['attempts'].append(failure);write_json(attemptdir/f'candidate-{number}-error.json',failure);break
    # Parent inherited deletion paths must never stand in for the actual accepted scene.
    result['deleted_text_pptx'],result['deleted_text_preview']=deletion(result,attemptdir/'accepted-deleted',backend)
    final_relations=check_relations(source,result['preview'],spec)
    clean=clean_check(page,source,result['deleted_text_preview'],attemptdir/'final-clean')
    refs=page['baseline_refs']
    r6_preview=refs['r6'].get('preview')
    r7_preview=(refs.get('r7') or refs['current']).get('preview')
    result.update(schema='R7Page/1',r6_ir=refs['r6']['ir']['path'],r6_pptx=refs['r6']['pptx']['path'],
                  r6_preview=r6_preview['path'] if r6_preview else page['preview'],
                  r7_preview=r7_preview['path'] if r7_preview else page['preview'],
                  source=str(source),execution_directory=str(attemptdir),relations=final_relations,clean_plate=clean,
                  visual={'parent':metrics(pixels(source,size),pixels(base['preview'],size)),
                          'r6':metrics(pixels(source,size),pixels(r6_preview['path'] if r6_preview else page['preview'],size)),
                          'r7':metrics(pixels(source,size),pixels(r7_preview['path'] if r7_preview else page['preview'],size)),
                          'r71':metrics(pixels(source,size),pixels(result['preview'],size))},
                  scene_sha256=scene_fingerprint(result['ir'])['sha256'],ir_sha256=sha256(result['ir']),
                  pptx_sha256=sha256(result['pptx']),status='REVIEW_REQUIRED')
    result.pop('verified_clean_plate',None)
    if clean['status']=='PASS':
        report=attemptdir/'final-clean/report.json'
        result['verified_clean_plate']={'report_path':str(report),'report_sha256':sha256(report)}
    result['cache_dependency_hashes']=checkpoint_files(result)
    write_json(done,result)
    print('[r71]',prefix,result['action'],'relations',final_relations['status'],'clean',clean['status'],flush=True)
    return result


def edit_hard_failures(report, expected):
    return summarize_edit_report(report, expected)['failures']


def audit(batch, run, backend):
    from r7_edit_checks import audit_page, probe_specs
    output=Path(run)/('acceptance-'+stamp());output.mkdir()
    target=output/'all-native-text-deleted.pptx'
    delete_deck_text(batch['deck']['pptx'],target,batch['pages'])
    rendering=backend.render(target,output/'deleted-render')
    normal=backend.render(batch['deck']['pptx'],output/'final-render')
    if len(rendering['pages'])!=12 or len(normal['pages'])!=12: raise ValueError('Final deck must have 12 pages')
    rows=[]
    for index,page in enumerate(batch['pages'],1):
        ir=read(page['ir']);size=ir['canvas']['source_px'];prefix=page['page'][:2]
        verify_styles(batch['deck']['pptx'],ir,index);verify_native_shape_properties(batch['deck']['pptx'],ir,index)
        naked=rendering['pages'][index-1]['path'];render=normal['pages'][index-1]['path']
        same=float(np.abs(pixels(naked,size).astype(float)-pixels(page['deleted_text_preview'],size)).mean())
        intact=float(np.abs(pixels(render,size).astype(float)-pixels(page['preview'],size)).mean())
        if max(same,intact)>.1: raise ValueError('Composed scene differs from accepted page')
        baseline=page
        clean=clean_check(baseline,page['source'],naked,output/(prefix+'-clean'))
        spec=read(Path(page['execution_directory'])/'relations/relations.json')
        relations=check_relations(page['source'],render,spec)
        expected=page['expected_probes'] if 'expected_probes' in page else probe_specs(ir,prefix)
        probes=audit_page(page,output/(prefix+'-edits'),backend,expected_probes=expected,
                          deck_pptx=batch['deck']['pptx'],slide_number=index,baseline_deck_render=normal)
        edit_result=summarize_edit_report(probes,expected)
        checks=aggregate_checks({'relations':relations,'text_clean':clean,'edit_probes':edit_result})
        row={'page':page['page'],'action':page['action'],'final_preview':render,'deleted_preview':naked,
             'export_consistency':{'status':'PASS','normal_mae':intact,'deleted_mae':same},
             'relations':relations,'clean_plate':clean,'edit_probes':probes,'visual':page['visual'],
             'hard_failures':checks['failures'],'review_required':checks['review_required'],
             'not_applicable':checks['not_applicable'],'status':checks['status'],
             'cover_bounds':page.get('cover_bounds'),'attempts':page.get('attempts',[]),
             'object_decisions':page.get('object_decisions',[]),'expected_probes':expected,
             'editability_structure':{
                 'native_text_nodes':sum(n['type']=='text' for n in ir['nodes']),
                 'native_shape_nodes':sum(n['type']=='shape' for n in ir['nodes']),
                 'raster_image_nodes':sum(n['type']=='image' for n in ir['nodes']),
                 'declared_edit_groups':len(ir.get('groups',[])),
                 'scope':'frozen_IR_checked_against_final_PPTX; raster_is_not_editable_path'}}
        rows.append(row);write_json(output/f'progress-{index:02d}.json',rows)
        print('[r7-qa]',prefix,flush=True)
    hard=[r['page'] for r in rows if r['hard_failures']]
    overall=aggregate_checks({r['page']:r for r in rows})
    result={'schema':'ReconstructionQA/1','status':overall['status'],'hard_failure_pages':hard,
            'review_required_pages':[r['page'] for r in rows if r['review_required']],
            'pages':rows,'directory':str(output),'deleted_pptx':str(target),'powerpoint':'NOT_RUN','wps':'NOT_RUN','human':'REVIEW_REQUIRED'}
    write_json(output/'report.json',result);_atomic_write(Path(run)/'qa.json',result)
    return result


def package(batch,qa,run):
    out=Path(run)/('delivery-'+stamp());out.mkdir()
    deck='徐州高新区AI交流_r7.1_12页_待人工审阅.pptx'
    shutil.copy2(batch['deck']['pptx'],out/deck);shutil.copy2(qa['deleted_pptx'],out/'删除全部原生文字_r7.1.pptx')
    gallery=[]
    for page,row in zip(batch['pages'],qa['pages']):
        prefix=page['page'][:2];d=out/'pages'/prefix;d.mkdir(parents=True)
        for name,path in [('source.png',page['source']),('r6.png',page['r6_preview']),('r7.png',page.get('r7_preview',page['r6_preview'])),('r71.png',row['final_preview']),('deleted.png',row['deleted_preview'])]: shutil.copy2(path,d/name)
        overlay=Image.open(page['source']).convert('RGB');draw=ImageDraw.Draw(overlay)
        spec=read(Path(page['execution_directory'])/'relations/relations.json')
        states={item['id']:item['status'] for item in row['relations'].get('objects',[])}
        for element in spec['elements']:
            box=element.get('bbox_xyxy')
            if not box:continue
            state=states.get(element['id'],'REVIEW_REQUIRED')
            color={'PASS':'#16A34A','FAIL':'#DC2626'}.get(state,'#D97706')
            draw.rectangle(box,outline=color,width=3)
            draw.text((box[0],max(0,box[1]-12)),element['id']+' '+state,fill=color)
        overlay.save(d/'relations-annotated.png')
        clone_scene(page['ir'],d/'scene')
        shutil.copytree(Path(page['execution_directory'])/'registry',d/'registry')
        shutil.copytree(Path(page['execution_directory'])/'relations',d/'relations')
        for sub in ('transactions','fault-injection','cover-container'):
            if (Path(page['execution_directory'])/sub).exists(): shutil.copytree(Path(page['execution_directory'])/sub,d/sub)
        for directory in Path(page['execution_directory']).glob('candidate-*'):
            if directory.is_dir() and (directory/'registry.json').exists(): shutil.copytree(directory,d/directory.name)
        editdir=Path(qa['directory'])/(prefix+'-edits')
        if editdir.exists(): shutil.copytree(editdir,d/'edit-probes')
        write_json(d/'report.json',row)
        write_json(d/'object-decisions.json',row['object_decisions'])
        gallery.append('<h2>'+html.escape(page['page'])+'</h2><p>'+page['action']+'</p><div class="row">'+''.join(f'<figure><figcaption>{label}</figcaption><img src="pages/{prefix}/{name}"></figure>' for name,label in [('source.png','原图'),('r6.png','r6'),('r7.png','r7'),('r71.png','r7.1'),('deleted.png','删除原生文字')])+'</div>')
    write_json(out/'qa.json',qa);shutil.copy2(Path(run)/'run-manifest.json',out/'run-manifest.json')
    for name in ('parent-link.json','parent-run-manifest.json','parent-batch.json'):
        if (Path(run)/name).is_file(): shutil.copy2(Path(run)/name,out/name)
    summary={'schema':'R7Delivery/1','deck':deck,'deck_sha256':sha256(out/deck),'pages':12,
             'updated_pages':[p['page'] for p in batch['pages'] if p['action']=='UPDATED'],
             'hard_failure_pages':qa['hard_failure_pages'],'status':'REVIEW_REQUIRED','automatic_status':qa['status'],'model_calls':0,
             'powerpoint':'NOT_RUN','wps':'NOT_RUN','human':'REVIEW_REQUIRED'}
    write_json(out/'delivery.json',summary)
    write_json(out/'failed_objects.json',[{'page':r['page'],'relations':r['relations'],'clean_plate':r['clean_plate'],
                 'hard_failures':r['hard_failures'],'edit_probes':r['edit_probes']} for r in qa['pages'] if r['hard_failures'] or r['review_required']])
    write_json(out/'hard-failures.json',[{'page':r['page'],'checks':r['hard_failures']} for r in qa['pages'] if r['hard_failures']])
    write_json(out/'review-required.json',[{'page':r['page'],'checks':r['review_required']} for r in qa['pages'] if r['review_required']])
    write_json(out/'object-coverage.json',[{'page':r['page'],**r['relations'].get('coverage',{})} for r in qa['pages']])
    write_json(out/'editability-structure.json',[{'page':r['page'],**r['editability_structure']} for r in qa['pages']])
    (out/'逐页对照.html').write_text('<meta charset="utf-8"><style>body{font-family:system-ui}.row{display:grid;grid-template-columns:1fr 1fr}figure{margin:8px}img{width:100%}</style><h1>r7 原图与实际文件对照</h1><p>自动验收与 Office/WPS/人工验收分别记录；保留基线不等于问题已修复。</p>'+''.join(gallery),encoding='utf-8')
    archive=Path(run)/(out.name+'.zip')
    with ZipFile(archive,'x',compression=ZIP_DEFLATED) as z:
        for f in out.rglob('*'):
            if f.is_file(): z.write(f,str(f.relative_to(out.parent)))
    _atomic_write(Path(run)/'delivery-index.json',{'directory':str(out),'archive':str(archive),**summary})
    return summary


def run(args):
    directory=Path(args.output or args.resume).resolve()
    if args.resume and Path(args.resume).resolve()!=directory: raise ValueError('Resume must use the same run directory')
    backend=ArtifactBackend(REPO,600);batch=load_baseline(args.input)
    relation_file=getattr(args,'relations',None)
    batch['_relations']=({'path':str(Path(relation_file).resolve()),'sha256':sha256(relation_file)} if relation_file else None)
    if len(batch['pages'])!=12: raise ValueError('Frozen baseline must contain twelve pages')
    with RunLock(directory):
        env=environment(args.input,batch,backend);manifest=directory/'run-manifest.json'
        if manifest.exists() and read(manifest)!=env: raise ValueError('Source/code/font/runtime changed; use a new run')
        if not manifest.exists(): write_json(manifest,env)
        if args.mode=='shadow':
            rows=[]
            for p in batch['pages']:
                dest=directory/'shadow'/p['page'][:2]
                record=read(dest/'objects.json') if dest.exists() else from_ir(p['ir'],dest)
                rows.append({'page':p['page'],'objects':len(record['objects']),'conflicts':record['conflicts']})
            _atomic_write(directory/'shadow.json',rows);return {'shadow_pages':len(rows)}
        order=['01','02','08','03','04','10','11']+[p['page'][:2] for p in batch['pages'] if p['page'][:2] not in ('01','02','08','03','04','10','11')]
        results={}
        for prefix in order:
            if args.pages and prefix not in args.pages: continue
            p=next(p for p in batch['pages'] if p['page'].startswith(prefix))
            results[p['page']]=process_page(p,directory,backend,relation_config=relation_file)
        if args.pages:
            if environment(args.input,batch,backend)!=env:
                raise ValueError('Frozen inputs or generation environment changed during pilot')
            return {'pilot_pages':list(results),'status':'REVIEW_REQUIRED'}
        ordered=[results[p['page']] for p in batch['pages']]
        if environment(args.input,batch,backend)!=env:
            raise ValueError('Frozen inputs or generation environment changed before composition')
        from image2ppt.compiler import execute
        deck=execute([backend.node,ROOT/'compiler_bridge.mjs','compose',directory/('deck-'+stamp()),*[p['ir'] for p in ordered]],REPO,600)
        combined={'schema':'R7Batch/1','input':str(Path(args.input).resolve()),'directory':str(directory),'pages':ordered,'deck':deck,'status':'REVIEW_REQUIRED','revision':'r7.1','baseline_context':batch['_baseline_context'],'relation_catalog':batch.get('_relations')}
        _atomic_write(directory/'batch.json',combined)
        qa=audit(combined,directory,backend)
        if environment(args.input,batch,backend)!=env:
            raise ValueError('Frozen inputs or generation environment changed during final QA')
        return package(combined,qa,directory)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['shadow','gated'],default='shadow');parser.add_argument('--pages',nargs='+')
    parser.add_argument('--resume');parser.add_argument('--relations')
    args=parser.parse_args(argv)
    print(json.dumps(run(args),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
