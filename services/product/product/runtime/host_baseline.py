"""Host-owned preflight, immutable control deck and per-page non-regression gate."""
from uuid import uuid4
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from ..materialize import digest, write
from ..tolerance_config import load_tolerance_config
from ..scorecard import fidelity, score_page, _native_fields, _norm, _overlap
from .host_evidence import prepare_all, questions, read
from .state import RuntimeState, RuntimeConflict
from .service import RuntimeService

ORIGIN = 'deterministic-baseline'


def local_fidelity(source, preview, marks):
    with Image.open(source) as image:
        original = np.asarray(image.convert('RGB'))
    height, width = original.shape[:2]
    with Image.open(preview) as image:
        rendered = np.asarray(image.convert('RGB').resize((width, height), Image.Resampling.LANCZOS))
    regions = {row['id']: row['bbox'] for row in marks}
    regions['whole-page'] = (0, 0, width, height)
    metrics = {}
    for name, (x, y, w, h) in regions.items():
        region = np.s_[max(0, int(y)):min(height, int(np.ceil(y+h))),
                       max(0, int(x)):min(width, int(np.ceil(x+w)))]
        if original[region].size:
            measured = fidelity(original[region], rendered[region], max_pixels=6000)
            metrics[name] = {key: measured[key] for key in ('mae_rgb', 'ssim', 'delta_e2000_p95')}
    return metrics


def assess(job, current):
    composition = RuntimeService.composition(current)
    deck = composition['artifacts']['pptx']
    office = composition['office']
    sha = digest(deck)
    verified = (office.get('status')=='PASS' and office.get('pptx_sha256')==sha
                and office.get('office',{}).get('pptx_sha256')==sha
                and office.get('office',{}).get('status')=='PASS_APP_AUTOMATED')
    pages = []
    for page,row in enumerate(current['pages'],1):
        k = str(page)
        source = row['snapshot']['path']
        preview = Path(office.get('office_directory',''))/f'page-{page:02d}-before.png'
        measures = score_page(source, deck, slide=page, render_path=preview if preview.is_file() else None,
                              checks=office if verified else None)
        fields = _native_fields(measures.pop('objects'))
        lines = read(current['ocr'][k])['lines']
        hits = [line['id'] for line in lines if _norm(line['text']) and any(
            _norm(line['text']) == _norm(obj['text']) and _overlap(line['bbox'],obj['bbox']) >
            .5*line['bbox'][2]*line['bbox'][3] for obj in fields)]
        selected = read(row['scene'])['selected']
        actions = {}
        for action in office.get('office',{}).get('records',[]):
            if action.get('page')==page and action.get('saved_reopened') and action.get('outside_pixels')==0:
                actions.setdefault(action['id'],set()).add(action.get('action'))
        editables = sorted(i for i,r in selected.items() if r.get('move_safety')!='MOVE_UNSAFE'
                           and {'move-scale','delete'} <= actions.get(i,set()))
        marks = read(current['experiment_marks'][k])['rows']
        local = local_fidelity(source, preview, marks) if preview.is_file() else {}
        limits = load_tolerance_config()['fidelity']
        minor = [key for key, val in local.items()
                 if val['ssim'] < limits['min_ssim'] or val['delta_e2000_p95'] > limits['max_delta_e2000_p95']]
        pages.append({'page':page, 'source_sha256':current['host_evidence'][k]['source_sha256'],
                      'passed_edits':sorted(f'{oid}/{action}' for oid in editables for action in actions[oid]),
                      'editable_count':len(editables) if verified else None, 'selected_editables':editables,
                      'ocr_native_hits':sorted(hits), 'native_field_count':len(fields), 'local_fidelity':local, 'minor_deviations':minor,
                      'max_local_mae':max((v['mae_rgb'] for v in local.values()), default=None), 'scorecard':measures})
    return {'pptx_sha256':sha,'office_status':office.get('status','UNKNOWN'),'verified':verified,
            'fidelity_limits':load_tolerance_config()['fidelity'],'pages':pages}


def regressions(base, candidate):
    reasons = []
    if not base['verified'] or not candidate['verified']:
        reasons.append('OFFICE_COMPARISON_UNVERIFIED')
    if len(base['pages'])!=len(candidate['pages']):
        reasons.append('PAGE_COUNT_CHANGED')
    for a,b in zip(base['pages'], candidate['pages']):
        prefix = f"p{a['page']}:"
        if a['source_sha256']!=b['source_sha256']:
            reasons.append(prefix+'SOURCE_CHANGED')
        if (b['editable_count'] or 0)<(a['editable_count'] or 0):
            reasons.append(prefix+'EDITABLE_COUNT_DECLINED')
        if set(a['ocr_native_hits'])-set(b['ocr_native_hits']) or b['native_field_count']<a['native_field_count']:
            reasons.append(prefix+'NATIVE_FIELDS_LOST')
        if set(a.get('passed_edits', [])) - set(b.get('passed_edits', [])):
            reasons.append(prefix+'OFFICE_EDITS_LOST')
        reference = a.get('local_fidelity', {})
        compared = b.get('local_fidelity', {})
        if not reference or not compared:
            reasons.append(prefix+'LOCAL_VISUAL_EVIDENCE_MISSING')
            continue
        limits = base.get('fidelity_limits') or load_tolerance_config()['fidelity']
        lost = []
        for region, old in reference.items():
            new = compared.get(region)
            if new is None:
                lost.append(region)
                continue
            de, similarity = new['delta_e2000_p95'], new['ssim']
            if (de > max(old['delta_e2000_p95'], limits['hard_max_delta_e2000_p95'])
                    or similarity < min(old['ssim'], limits['hard_min_ssim'])):
                lost.append(region)
        if lost:
            reasons.append(prefix+'LOCAL_VISUAL_REGRESSION:'+','.join(lost[:12]))
    return reasons


def ensure_baseline(job):
    state = RuntimeState(job)
    svc = RuntimeService(job.parent)
    request = read(job/'request.json')
    sources = [digest(job/'inputs'/n) for n in request['sources']]
    originals = []
    for index,ref in enumerate(request.get('original_inputs',[])):
        if (ref['canonical_file']!=request['sources'][index] or ref['canonical_sha256']!=sources[index]
                or digest(job/ref['path'])!=ref['sha256']):
            raise RuntimeConflict('uploaded source or canonical image changed')
        originals.append(ref['sha256'])
    root = job/'host-baseline'
    root.mkdir(exist_ok=True)
    manifest = root/'manifest.json'
    if manifest.exists():
        info = read(manifest)
        if sources!=info['sources'] or any(digest(root/k)!=v for k,v in info['files'].items()):
            raise RuntimeConflict('frozen baseline inputs or artifacts changed')
        return info
    if not (root/'started.json').exists():
        write(root/'started.json',{'time':time.time()})
    def step(name, fn):
        return state.mutate(name,{'sources':sources},None,'host-stage0-'+name,fn)
    step('prepare',lambda c,d:prepare_all(svc,state,c,d,sources=sources))
    for page in range(1,len(sources)+1):
        def reconstruct(c,d):
            outcome = svc.batch_reconstruct_page(state,c,d,page=page,include_region_crops=False,verify_office=True)
            if outcome['status']=='VALIDATED':
                svc.commit_region(state,c,d/'commit',candidate_id=outcome['candidate_id'])
            failures = outcome.get('failed_regions',[])
            c.setdefault('regional_failures',{})[str(page)] = failures
            c.setdefault('questions',{})[str(page)] = questions(state,c,page,failures,d)
            write(d/'regional-failures.json',failures)
            return {'page':page,'status':outcome['status'],'failed_count':len(failures)}
        step(f'reconstruct-{page}',reconstruct)
        def finish_photos(c,d):
            # Text repairs and their exclusions are settled before cropping or upgrading enclosing cards.
            candidate = svc.crop_region(state,c,d,page=page,upgrade_containers=request.get('native_variant')!='joint-only')
            if candidate['status']=='STAGED_LOCAL':
                checked = svc.validate_candidate(state,c,d/'validate',candidate_id=candidate['candidate_id'])
                if c['candidates'][candidate['candidate_id']]['status']=='VALIDATED':
                    svc.commit_region(state,c,d/'commit',candidate_id=candidate['candidate_id'])
            return {'page':page,'status':candidate['status']}
        step(f'finish-assets-{page}',finish_photos)
    step('compose',lambda c,d:svc.compose_delivery(state,c,d))
    step('office',lambda c,d:svc.validate_office(state,c,d))
    current = state.read()
    score = assess(job,current)
    shutil.copyfile(current['composition']['artifacts']['pptx'],root/'baseline.pptx')
    write(root/'state.json',current)
    write(root/'scorecard.json',score)
    files = ['baseline.pptx','state.json','scorecard.json']
    office = current['composition']['office']
    for p in score['pages']:
        name = f"page-{p['page']:02d}-before.png"
        preview = Path(office.get('office_directory',''))/name
        if preview.is_file():
            shutil.copyfile(preview,root/name)
            files.append(name)
    info = {'origin':ORIGIN,'native_variant':request.get('native_variant','combined'),'sources':sources,'uploaded_sha256':originals,'finished':time.time(),
            'files':{n:digest(root/n) for n in files}}
    write(manifest,info)
    return info


def summary(score):
    return [{'page':p['page'],'editable_objects':p['editable_count'],'native_fields':p['native_field_count'],
             'ocr_native_hits':len(p['ocr_native_hits']), 'max_local_mae':p['max_local_mae']} for p in score['pages']]


def install_file(source,destination):
    temp = destination.with_name('.'+uuid4().hex+'.tmp')
    shutil.copyfile(source,temp)
    temp.replace(destination)


def selected_result(job):
    selection = job/'runtime/delivery-selection.json'
    if not selection.is_file():
        return None
    receipt = read(selection)
    if digest(job/'result.json')!=receipt['result_sha256'] or digest(job/'delivery/result.pptx')!=receipt['selected_sha256']:
        raise RuntimeConflict('selected delivery bytes changed')
    return read(job/'result.json')


def publish(job, agent, reasons=None, candidate=None):
    with RuntimeState(job).publication_lock():
        return selected_result(job) or _publish(job,agent,reasons,candidate)


def _publish(job, agent, reasons, candidate):
    root = job/'host-baseline'
    manifest = ensure_baseline(job)
    base = read(root/'scorecard.json')
    delivery = job/'delivery'
    delivery.mkdir(exist_ok=True)
    state = RuntimeState(job)
    current = state.read()
    usage_file = state.root/'agent-usage.json'
    usage = read(usage_file) if usage_file.exists() else {}
    if not reasons and candidate:
        staging = state.root/'agent-delivery'
        files = agent.get('artifacts',{})
        if (files.get('result.pptx')!=candidate['pptx_sha256'] or
                any(Path(name).name!=name or not (staging/name).is_file() or digest(staging/name)!=sha
                    for name,sha in files.items())):
            reasons = ['AGENT_PACKAGE_HASH_MISMATCH']
    fallback = bool(reasons)
    if fallback:
        frozen = read(root/'state.json')
        if current['version']!=frozen['version'] and current['status']!='REVIEW_REQUIRED':
            write(state.root/'agent-attempt/state.json',current)
        install_file(root/'baseline.pptx',delivery/'result.pptx')
        office_dir = Path(frozen['composition']['office'].get('office_directory',''))
        for p in base['pages']:
            n = p['page']
            preview = office_dir/f'page-{n:02d}-before.png'
            if preview.name in manifest['files']:
                preview = root/preview.name
            if preview.is_file():
                install_file(preview,delivery/f'page-{n:02d}.png')
        editable = [p['selected_editables'] if base['verified'] else [] for p in base['pages']]
        result = {'status':'REVIEW_REQUIRED','completion':'PARTIAL_EDITABLE' if any(editable) else 'RASTER_ONLY',
                  'office_pass':base['verified'],'editable_objects':editable,'pages':len(editable),
                  'human_acceptance':False,'automatic_publication':False,'final_qa':'UNKNOWN',
                  'reason':agent.get('reason') or '; '.join(reasons),'failure':agent.get('failure') or reasons[0]}
        frozen.update(status='REVIEW_REQUIRED',version=current['version']+1)
        write(state.root/'state.json',frozen)
    else:
        result = dict(agent)
        for name in files:
            if name!='summary.json':
                install_file(staging/name,delivery/name)
    changed = not fallback and candidate['pptx_sha256']!=base['pptx_sha256']
    result.update(origin='agent-incremental' if changed else ORIGIN,baseline_sha256=manifest['files']['baseline.pptx'],
                  agent={'participated':(bool(usage.get('tool_calls')) or bool(usage.get('tokens',{}).get('output')))
                        if usage else 'UNKNOWN' if (state.root/'session.json').exists() else False,
                         'delivered_changes':changed, 'fallback_reasons':reasons or [],'usage':{k:v for k,v in usage.items() if k!='tool_calls'}},
                  independent_go_review=('PASS' if not fallback and agent.get('independent_audit',{}).get('passed') is True else 'UNKNOWN'),wps_acceptance='UNKNOWN', baseline_pages=summary(base))
    shutil.copyfile(root/'baseline.pptx',delivery/'deterministic-baseline.pptx')
    shutil.copyfile(root/'scorecard.json',delivery/'baseline-scorecard.json')
    if candidate:
        write(delivery/'agent-scorecard.json',candidate)
    result.pop('artifacts',None)
    write(delivery/'summary.json',result)
    names = {'result.pptx','deterministic-baseline.pptx','baseline-scorecard.json','summary.json'}
    names.update(f"page-{p['page']:02d}.png" for p in base['pages'])
    if not fallback:
        names.update(files)
    if candidate:
        names.add('agent-scorecard.json')
        trial = state.root/'agent-delivery/result.pptx'
        if candidate.get('verified') and trial.is_file() and digest(trial) == candidate['pptx_sha256']:
            install_file(trial, delivery/'agent-candidate.pptx')
            names.add('agent-candidate.pptx')
    result['artifacts'] = {name:digest(delivery/name) for name in sorted(names) if (delivery/name).is_file()}
    write(job/'result.json',result)
    write(state.root/'delivery-selection.json',{'result_sha256':digest(job/'result.json'),
          'selected_sha256':result['artifacts']['result.pptx'],'origin':result['origin']})
    return result
