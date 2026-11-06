"""Source-bound r7 orchestration. No model calls, no mutation of historical scenes."""
from __future__ import annotations
import argparse
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED
import cv2
import numpy as np
from PIL import Image, ImageEnhance
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


def frozen_scene(before, after, removed=(), added=()):
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
        if left!=right: raise ValueError('Frozen existing object changed: '+identifier)
    order=lambda ir:[n['id'] for n in sorted(ir['nodes'],key=lambda x:x['z']) if n['id'] in set(a)&set(b)]
    if order(before)!=order(after): raise ValueError('Existing relative z order changed')
    if before.get('groups',[])!=after.get('groups',[]): raise ValueError('Existing edit groups changed')
    assets={x['id']:x for x in after['assets']}
    for item in before['assets']:
        if assets.get(item['id'])!=item: raise ValueError('Frozen original asset changed')
    return {'status':'PASS','existing_text_and_styles':'UNCHANGED','existing_assets':'UNCHANGED'}


def environment(input_path, batch, backend):
    fontfile=Path(input_path).parent/'font-manifest.json'
    fonts=read(fontfile) if fontfile.exists() else None
    if not fonts: raise ValueError('Baseline verified font manifest required')
    for font in fonts['fonts']:
        if sha256(font['file'])!=font['sha256']: raise ValueError('Baseline font file changed')
    code={str(p.relative_to(REPO)):sha256(p) for folder in [ROOT/'src',REPO/'src/image2ppt',REPO/'scripts'] for p in folder.glob('*') if p.suffix in ('.py','.mjs')}
    page_refs=[]
    for page in batch['pages']:
        if sha256(page['ir'])!=page['ir_sha256'] or sha256(page['pptx'])!=page['pptx_sha256']:
            raise ValueError('Baseline IR/PPTX changed')
        ir=read(page['ir'])
        if sha256(Path(page['ir']).parent/ir['source_image'])!=page['source_sha256']: raise ValueError('Source changed')
        masks=Path(page['directory'])/'text-masks'
        evidence={str(p.resolve()):sha256(p) for p in masks.glob('*.png')}
        if not evidence: raise ValueError('Frozen source text masks missing: '+page['page'])
        page_refs.append({'page':page['page'],'scene':scene_fingerprint(page['ir']),
                          'pptx_sha256':sha256(page['pptx']),'source_masks':evidence})
    runtime=[backend.node,REPO/'.runtime/bundle/python/python.exe',Path(__import__('sys').executable)]
    return {'schema':'R7RunInputs/1','baseline':str(Path(input_path).resolve()),'baseline_sha256':sha256(input_path),
            'pages':page_refs,'fonts':fonts,'code':code,'runtime':{str(p):sha256(p) for p in runtime if Path(p).exists()},
            'machine':platform.node(),'python':platform.python_version(),'model_calls':0,
            'baseline_code':'e11d5b0','source_layerd_rerun':False}


def deletion(delivery, output, backend):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/'deleted.pptx'
    delete_deck_text(delivery['pptx'],target,[delivery])
    rendered=backend.render(target,output/'render')
    return str(target),rendered['pages'][0]['path']


def clean_check(baseline, source_path, preview, output):
    """Independent frozen source glyph masks + OCR, not only plate equality."""
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
        rows.append({'id':identifier,'glyph_sha256':sha256(glyph),'inspection_sha256':sha256(inspection),
                     **inspect_residue(source,actual,core,inspect)})
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
    passed=complete and rows and all(r['status']=='PASS' for r in rows) and not hits
    result={'schema':'R7CleanPlateQA/1','source_sha256':sha256(source_path),'method':'frozen_source_strokes_and_independent_ocr',
            'status':'PASS' if passed else 'REVIEW_REQUIRED',
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
    for folder in ('relations','registry','final-clean','transactions','fault-injection'):
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
        current=environment(batch['input'],read(batch['input']),backend)
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
                validate_qa_environment(current,environment(batch['input'],read(batch['input']),backend))
                return package(copied,checked,destination)
        checked=audit(batch,parent,backend)
        return package(batch,checked,parent)


def process_page(page, run, backend):
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
    spec=prepare_spec(prefix,source,attemptdir/'relations')
    base_ir=clone_scene(page['ir'],attemptdir/'baseline')
    base=compile_ir(backend,base_ir)
    negative=check_relations(source,base['preview'],spec)
    write_json(attemptdir/'baseline-relation-qa.json',negative)
    if prefix in ('02','08') and negative['status']!='FAIL': raise ValueError('Known negative fixture unexpectedly passed')
    result={**page,**base,'ir':str(base_ir),'action':'BASELINE_RETAINED','attempts':[]}
    if spec['elements']:
        raster_ids=[]
        immutable_inputs={str(source):sha256(source),str(attemptdir/'relations/relations.json'):sha256(attemptdir/'relations/relations.json')}
        immutable_inputs.update({ref['path']:ref['sha256'] for element in spec['elements'] for ref in element['masks'].values()})
        parent_fingerprint=scene_fingerprint(base_ir)
        for number in (1,2):
            tx=None
            try:
                candidate=build_candidate(base_ir,spec,attemptdir/f'candidate-{number}',raster_ids=raster_ids)
                tx=ReconstructionTransaction(attemptdir/'transactions',f'correction-{number}',base_ir,
                        {'source':sha256(source),'spec':fingerprint(spec)},
                        ['relations','frozen','write_boundary','text_clean','raster_fit'],candidate['added_ids'])
                frozen=tx.prepare(candidate['ir'])
                trial=compile_ir(backend,frozen)
                relations=check_relations(source,trial['preview'],spec)
                struct=frozen_scene(original,read(frozen),candidate['removed_ids'],candidate['added_ids'])
                allowed=np.asarray(Image.open(candidate['authorized_mask']).convert('L'))>0
                # Renderer AA has its own inspection envelope, never an expanded write permission.
                rendered_allowed=cv2.dilate(allowed.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
                boundary=check_authorized_changes(pixels(base['preview'],size),pixels(trial['preview'],size),rendered_allowed)
                trial_delivery={**result,**trial,'ir':str(frozen)}
                deleted,naked=deletion(trial_delivery,attemptdir/f'candidate-{number}-deleted',backend)
                cleanliness=clean_check(page,source,naked,attemptdir/f'candidate-{number}-clean')
                fits=[r['fit'] for item in candidate['raster_records'] for r in item['records']]
                fit_gate={'status':'PASS' if all(r['mae']<=2 and r['p95']<=5 for r in fits) else 'FAIL',
                          'applicable':bool(fits),'records':fits}
                gates={'relations':relations,'frozen':struct,'write_boundary':boundary,'text_clean':cleanliness,'raster_fit':fit_gate}
                verdict=tx.verify(gates,owners)
                write_json(attemptdir/f'candidate-{number}-gates.json',gates)
                record={'number':number,'raster_ids':list(raster_ids),'gates':gates,'added_ids':candidate['added_ids']}
                if verdict['state']=='VERIFIED':
                    receipt=tx.commit();record['state']='COMMITTED'
                    result.update(trial,ir=str(receipt['ir']),action='UPDATED',deleted_text_pptx=deleted,deleted_text_preview=naked)
                    result['attempts'].append(record);break
                tx.rollback('required gates not PASS');record['state']='ROLLED_BACK';result['attempts'].append(record)
                raster_ids=[row['id'] for row in relations['objects'] if row['status']!='PASS']
                if not raster_ids: break
            except Exception as error:
                # Input tampering invalidates the run; ordinary candidate failures retain the accepted parent.
                if scene_fingerprint(base_ir)!=parent_fingerprint or any(sha256(p)!=expected for p,expected in immutable_inputs.items()):
                    raise ValueError('Immutable candidate inputs changed') from error
                if tx is not None and tx.prepare_file.exists(): tx.rollback('candidate exception: '+str(error))
                failure={'number':number,'state':'ROLLED_BACK','error_type':type(error).__name__,'reason':str(error)}
                result['attempts'].append(failure)
                write_json(attemptdir/f'candidate-{number}-error.json',failure)
                # A programming/compiler failure is not evidence for selecting a different visual candidate.
                break
    if result['action']=='BASELINE_RETAINED':
        result['deleted_text_pptx'],result['deleted_text_preview']=deletion(result,attemptdir/'baseline-deleted',backend)
    final_relations=check_relations(source,result['preview'],spec)
    clean=clean_check(page,source,result['deleted_text_preview'],attemptdir/'final-clean')
    result.update(schema='R7Page/1',r6_ir=page['ir'],r6_pptx=page['pptx'],r6_preview=base['preview'],
                  source=str(source),execution_directory=str(attemptdir),relations=final_relations,clean_plate=clean,
                  visual={'r6':metrics(pixels(source,size),pixels(base['preview'],size)),
                          'r7':metrics(pixels(source,size),pixels(result['preview'],size))},
                  scene_sha256=scene_fingerprint(result['ir'])['sha256'],ir_sha256=sha256(result['ir']),
                  pptx_sha256=sha256(result['pptx']),status='REVIEW_REQUIRED')
    if clean['status']=='PASS':
        report=attemptdir/'final-clean/report.json'
        result['verified_clean_plate']={'report_path':str(report),'report_sha256':sha256(report)}
    # Real prepared candidate forced to fail: prove exact accepted r6 scene recovery.
    if prefix=='03':
        fault=ReconstructionTransaction(attemptdir/'fault-injection','forced-failure',base_ir,
                                        {'source':sha256(source)},['render','relations'])
        fault.prepare(base_ir);fault.verify({'render':'PASS','relations':'FAIL'},owners)
        recovered=fault.rollback('deliberate relation failure')
        if scene_fingerprint(recovered['ir'])!=scene_fingerprint(base_ir): raise ValueError('Rollback lost scene content')
        result['forced_rollback']={'status':'PASS','receipt':recovered,'editable_degradation':False}
    result['cache_dependency_hashes']=checkpoint_files(result)
    write_json(done,result)
    print('[r7]',prefix,result['action'],'relations',final_relations['status'],'clean',clean['status'],flush=True)
    return result


def edit_hard_failures(report, expected):
    """Separate failed edit behavior from missing independent clean-plate evidence."""
    from collections import Counter
    probes=report.get('probes',[])
    failures=[]
    actual=Counter((p.get('target'),p.get('action')) for p in probes)
    if actual!=Counter(expected): failures.append('missing_or_duplicate_required_edit_probe')
    for probe in probes:
        long_overflow=probe.get('action')=='replace_long' and probe.get('overflow_detected') is True
        reasons=[]
        if not long_overflow and any(key not in probe for key in (
                'export_consistency','protected_pixels','actual_changed_pixels','protected_xml','package_parts')):
            reasons.append('required_edit_measurement_missing')
        if probe.get('error_type') and not long_overflow: reasons.append('edit_execution_failed')
        if probe.get('export_consistency')=='FAIL': reasons.append('edit_export_inconsistent')
        if probe.get('outside_changed_pixels',0)>0 and not long_overflow: reasons.append('edit_changed_protected_pixels')
        if probe.get('actual_changed_pixels')==0: reasons.append('edit_had_no_effect')
        if probe.get('protected_xml')=='FAIL' or probe.get('package_parts')=='FAIL': reasons.append('edit_changed_protected_xml')
        if reasons: failures.append({'target':probe.get('target'),'action':probe.get('action'),'reasons':reasons})
    return failures


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
        baseline=next(p for p in read(batch['input'])['pages'] if p['page']==page['page'])
        clean=clean_check(baseline,page['source'],naked,output/(prefix+'-clean'))
        spec=read(Path(page['execution_directory'])/'relations/relations.json')
        relations=check_relations(page['source'],render,spec)
        probes=audit_page(page,output/(prefix+'-edits'),backend)
        failures=[]
        if spec.get('elements') and relations['status']!='PASS': failures.append('required_relations_not_passed')
        if clean['status']!='PASS': failures.append('native_text_cleanliness_not_passed')
        failures.extend(edit_hard_failures(probes,probe_specs(ir,prefix)))
        row={'page':page['page'],'action':page['action'],'final_preview':render,'deleted_preview':naked,
             'export_consistency':{'status':'PASS','normal_mae':intact,'deleted_mae':same},
             'relations':relations,'clean_plate':clean,'edit_probes':probes,'visual':page['visual'],
             'hard_failures':failures,'review_required':probes.get('status')=='REVIEW_REQUIRED'}
        rows.append(row);write_json(output/f'progress-{index:02d}.json',rows)
        print('[r7-qa]',prefix,flush=True)
    hard=[r['page'] for r in rows if r['hard_failures']]
    result={'schema':'ReconstructionQA/1','status':'REVIEW_REQUIRED','hard_failure_pages':hard,
            'pages':rows,'directory':str(output),'deleted_pptx':str(target),'powerpoint':'NOT_RUN','wps':'NOT_RUN','human':'REVIEW_REQUIRED'}
    write_json(output/'report.json',result);_atomic_write(Path(run)/'qa.json',result)
    return result


def package(batch,qa,run):
    out=Path(run)/('delivery-'+stamp());out.mkdir()
    deck='徐州高新区AI交流_r7_12页_待人工审阅.pptx'
    shutil.copy2(batch['deck']['pptx'],out/deck);shutil.copy2(qa['deleted_pptx'],out/'删除全部原生文字_r7.pptx')
    gallery=[]
    for page,row in zip(batch['pages'],qa['pages']):
        prefix=page['page'][:2];d=out/'pages'/prefix;d.mkdir(parents=True)
        for name,path in [('source.png',page['source']),('r6.png',page['r6_preview']),('r7.png',row['final_preview']),('deleted.png',row['deleted_preview'])]: shutil.copy2(path,d/name)
        clone_scene(page['ir'],d/'scene')
        shutil.copytree(Path(page['execution_directory'])/'registry',d/'registry')
        shutil.copytree(Path(page['execution_directory'])/'relations',d/'relations')
        for sub in ('transactions','fault-injection'):
            if (Path(page['execution_directory'])/sub).exists(): shutil.copytree(Path(page['execution_directory'])/sub,d/sub)
        for directory in Path(page['execution_directory']).glob('candidate-*'):
            if directory.is_dir() and (directory/'registry.json').exists(): shutil.copytree(directory,d/directory.name)
        editdir=Path(qa['directory'])/(prefix+'-edits')
        if editdir.exists(): shutil.copytree(editdir,d/'edit-probes')
        write_json(d/'report.json',row)
        gallery.append('<h2>'+html.escape(page['page'])+'</h2><p>'+page['action']+'</p><div class="row">'+''.join(f'<figure><figcaption>{label}</figcaption><img src="pages/{prefix}/{name}"></figure>' for name,label in [('source.png','原图'),('r6.png','r6'),('r7.png','r7'),('deleted.png','删除原生文字')])+'</div>')
    write_json(out/'qa.json',qa);shutil.copy2(Path(run)/'run-manifest.json',out/'run-manifest.json')
    for name in ('parent-link.json','parent-run-manifest.json','parent-batch.json'):
        if (Path(run)/name).is_file(): shutil.copy2(Path(run)/name,out/name)
    summary={'schema':'R7Delivery/1','deck':deck,'deck_sha256':sha256(out/deck),'pages':12,
             'updated_pages':[p['page'] for p in batch['pages'] if p['action']=='UPDATED'],
             'hard_failure_pages':qa['hard_failure_pages'],'status':'REVIEW_REQUIRED','model_calls':0,
             'powerpoint':'NOT_RUN','wps':'NOT_RUN','human':'REVIEW_REQUIRED'}
    write_json(out/'delivery.json',summary)
    write_json(out/'failed_objects.json',[{'page':r['page'],'relations':r['relations'],'clean_plate':r['clean_plate'],
                 'hard_failures':r['hard_failures'],'edit_probes':r['edit_probes']} for r in qa['pages'] if r['page'] in qa['hard_failure_pages']])
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
    backend=ArtifactBackend(REPO,600);batch=read(args.input)
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
        order=['02','08','03','04','10','11']+[p['page'][:2] for p in batch['pages'] if p['page'][:2] not in ('02','08','03','04','10','11')]
        results={}
        for prefix in order:
            if args.pages and prefix not in args.pages: continue
            p=next(p for p in batch['pages'] if p['page'].startswith(prefix))
            results[p['page']]=process_page(p,directory,backend)
        if args.pages: return {'pilot_pages':list(results),'status':'REVIEW_REQUIRED'}
        ordered=[results[p['page']] for p in batch['pages']]
        from image2ppt.compiler import execute
        deck=execute([backend.node,ROOT/'compiler_bridge.mjs','compose',directory/('deck-'+stamp()),*[p['ir'] for p in ordered]],REPO,600)
        combined={'schema':'R7Batch/1','input':str(Path(args.input).resolve()),'directory':str(directory),'pages':ordered,'deck':deck,'status':'REVIEW_REQUIRED'}
        _atomic_write(directory/'batch.json',combined)
        qa=audit(combined,directory,backend)
        return package(combined,qa,directory)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['shadow','gated'],default='shadow');parser.add_argument('--pages',nargs='+')
    parser.add_argument('--resume')
    args=parser.parse_args(argv)
    print(json.dumps(run(args),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
