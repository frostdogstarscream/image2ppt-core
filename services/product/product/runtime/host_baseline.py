"""Host-owned preflight, immutable control deck and per-page non-regression gate."""
from uuid import uuid4
from copy import deepcopy
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from ..materialize import digest, write
from ..tolerance_config import load_tolerance_config
from ..scorecard import fidelity, score_page, _native_fields, _overlap
from .plate_audit import audit as audit_plate, bind_native_paragraphs, bind_separate_symbols, native_line_fields, visible_divider_lines
from .host_evidence import prepare_all, questions, read
from .reading_targets import revised_lines
from .state import RuntimeState, RuntimeConflict
from .service import RuntimeService

ORIGIN = 'deterministic-baseline'


def _reopen_residual_fields(service, trial, page, ids, directory):
    """Withdraw only independently owned text on the trial copy, so the batch can restage it."""
    from ..contracts import Text
    from ..recovery.region_state import closure, rollback
    row = service.page(trial, page)
    graph = service.graph(row)
    eligible = set()
    skipped = {}
    for oid in ids & graph.selected.keys():
        record = graph.selected[oid]
        if (record.get('kind') != 'text' or record.get('source_owners') != ['source-plate']
                or not record.get('source_text_definition')):
            skipped[oid] = 'SHARED_OR_UNPROVEN_SOURCE'
        elif closure(graph, {oid}) & graph.selected.keys() != {oid}:
            skipped[oid] = 'DEPENDENT_COMMITTED_OBJECT'
        else:
            eligible.add(oid)
    if eligible:
        for i, text in enumerate(graph.text.texts):
            if text.id in eligible:
                graph.text.texts[i] = Text.model_validate(graph.selected[text.id]['source_text_definition'])
        ir = rollback(graph, Path(row['ir']).parent, eligible, directory, 'RETRY_VISIBLE_SOURCE_TEXT')
        row.update(ir=str(ir), scene=str(ir.parent/'scene.json'))
        trial.pop('composition', None)
        trial.pop('visual_review', None)
    return sorted(eligible), skipped


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


def _text_in_retained_artwork(record, rows, directory, source_sha):
    """Match text visible with Office text hidden to verified glyph pixels owned by card artwork."""
    ref = record.get('asset_reference', {})
    owner = record.get('owner_evidence', {})
    if not record.get('retained_card_artwork') or ref.get('source_sha256') != source_sha:
        return []
    ax, ay, aw, ah = map(round, record['bbox'])
    candidates = [r for r in rows if r.get('status') == 'retained' and r.get('raster_reading')
                  and not r.get('plate_reading') and _overlap(record['bbox'], r['bbox']) >= .7*r['bbox'][2]*r['bbox'][3]]
    if not candidates:
        return []
    for info, field in ((ref,'path'), (owner,'file')):
        path = (directory / info.get(field, '')).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or digest(path) != info.get('sha256'):
            return []
    with Image.open(directory/ref['path']) as image:
        alpha = np.asarray(image.convert('RGBA'))[..., 3]
    if alpha.shape != (ah,aw):
        return []
    with np.load(directory/owner['file'], allow_pickle=False) as proof:
        owned = proof['mask'].astype(bool)
    source = directory/'source.png'
    if digest(source) != source_sha:
        raise RuntimeConflict('retained artwork source changed')
    hits = []
    with Image.open(source) as original:
        for row in candidates:
            x,y,w,h = map(round, row['bbox'])
            if min(w,h) <= 0 or min(x,y) < 0 or x+w > original.width or y+h > original.height:
                continue
            tile = np.asarray(original.crop((x,y,x+w,y+h)).convert('RGB')).astype(float)
            paper = np.median(tile.reshape(-1,3), axis=0)
            strokes = abs(tile-paper).max(2) > 55
            yy, xx = np.where(strokes)
            if len(xx) < 3 or np.ptp(xx) < .48*w or np.ptp(yy) < .30*h:
                continue
            xx, yy = xx+x, yy+y
            inside = (xx>=ax) & (xx<ax+aw) & (yy>=ay) & (yy<ay+ah)
            sx, sy = xx[inside], yy[inside]
            supported = owned[sy,sx] & (alpha[sy-ay,sx-ax] >= 128)
            if supported.sum() / len(xx) >= .85:
                hits.append(row['ocr_id'])
    return hits


def assess(job, current):
    composition = RuntimeService.composition(current)
    deck = composition['artifacts']['pptx']
    office = composition['office']
    sha = digest(deck)
    verified = (office.get('status')=='PASS' and office.get('pptx_sha256')==sha
                and office.get('office',{}).get('pptx_sha256')==sha
                and office.get('office',{}).get('status')=='PASS_APP_AUTOMATED')
    request_file = Path(job) / 'request.json'
    checked_icons = (verified and request_file.is_file() and read(request_file).get('object_scoped_delivery')
                     and office['office'].get('embedded_alpha_and_visible_rgb_checked'))
    pages = []
    for page,row in enumerate(current['pages'],1):
        k = str(page)
        source = row['snapshot']['path']
        preview = Path(office.get('office_directory',''))/f'page-{page:02d}-before.png'
        measures = score_page(source, deck, slide=page, render_path=preview if preview.is_file() else None,
                              checks=office if verified else None)
        objects = measures.pop('objects')
        fields = _native_fields(objects)
        lines = read(current.get('source_ocr', {}).get(k, current['ocr'][k]))['lines']
        source_fields = current.get('scene_relations', {}).get(k, {}).get('source_fields', {})
        hidden = preview.with_name(f'page-{page:02d}-text-hidden.png')
        receipt = office.get('office', {}).get('text_hidden_views', {}).get(k, {}) if verified else {}
        lines = visible_divider_lines(lines,source_fields,source,hidden,receipt)
        late_fields = current.get('late_fields',{}).get(k,{})
        known_lines = {line['id'] for line in lines}
        lines += [item['line'] for item in late_fields.values()
                  if item['status']=='ready' and item['line']['id'] not in known_lines]
        scene = read(row['scene'])
        selected = scene['selected']
        readings = {i:r['source_reading'] for i,r in selected.items() if r.get('source_reading')}
        updated = {v['ocr_id'] for v in readings.values() if 'source_span' not in v}
        # A retained OCR value may also be uncertain (for example a disputed amount).
        # Include the latest bound observation even when it did not change the text.
        observed = {}
        for path in current.get('reading_reports', {}).get(k, []):
            report = read(path)
            if report.get('source_sha256') != row['snapshot']['source_sha256']:
                raise RuntimeConflict('source of reading decisions changed')
            for obs in report.get('observations', []):
                oid = obs['object_id']
                observed[oid] = obs
        for oid, obs in observed.items():
            if oid in selected and oid not in readings and obs.get('status') == 'pending_confirmation':
                readings[oid] = {**obs, 'after':obs['before']}
        pending = {v['ocr_id'] for v in readings.values() if v['status']=='pending_confirmation'}
        adopted = revised_lines(lines, readings)
        members = scene['text']['members']
        fields = bind_native_paragraphs(lines, fields, members, adopted)
        actions = {}
        for action in office.get('office',{}).get('records',[]):
            if action.get('page')==page and action.get('saved_reopened') and action.get('outside_pixels')==0:
                actions.setdefault(action['id'],set()).add(action.get('action'))
        editables = sorted(i for i,r in selected.items() if r.get('move_safety')!='MOVE_UNSAFE'
                           and {'move-scale','delete'} <= actions.get(i,set()))
        graphics = [o for o in objects if verified and o['name'] in editables
                    and o['class'] in ('asset','shape','opaque_picture')]
        fields = bind_separate_symbols(fields, members, source_fields, graphics)
        hits = [line['id'] for line in lines
                if native_line_fields({**line, 'text': adopted.get(line['id'], line['text'])}, fields)]
        updated.update(pending.intersection(hits))
        updated.update(adopted.keys() & set(hits))
        marks = read(current['experiment_marks'][k])['rows']
        local = local_fidelity(source, preview, marks) if preview.is_file() else {}
        limits = load_tolerance_config()['fidelity']
        minor = [key for key, val in local.items()
                 if val['ssim'] < limits['min_ssim'] or val['delta_e2000_p95'] > limits['max_delta_e2000_p95']]
        plate = audit_plate(job, deck, sha, page, row['snapshot'], lines, fields, marks, adopted,
                            office_view=hidden, view_receipt=receipt, check_icon_fragments=checked_icons)
        residual_pages = office.get('residuals', [])
        if checked_icons and len(residual_pages) >= page:
            from .graphic_readings import bind_graphic_readings
            bind_graphic_readings(plate, job=Path(job), page=page, scene=scene, backing=Path(row['ir']).parent,
                                  source_path=Path(source), hidden_path=hidden, lines=lines, editable_ids=editables,
                                  residuals=residual_pages[page-1], directory=Path(job)/'plate-audit'/sha/f'p{page}')
        for item in plate['rows']:
            if item['status']=='plate_only':
                item['mark_ids']=[oid for oid,r in late_fields.items() if r['bbox']==item['bbox']]
        unscoped = []
        artwork_text = {}
        page_width, page_height = row['snapshot']['image_size']
        for oid, record in selected.items():
            if record.get('kind') != 'asset':
                continue
            box = record['bbox']
            covers_page = _overlap(box, (0, 0, page_width, page_height)) >= page_width * page_height
            seeds = record.get('seed_ids', [])
            is_crop = record.get('region_crop') or record.get('payload', {}).get('upgraded_from_region_crop')
            trapped_text = any(item['status'] == 'unresolved' and _overlap(box, item['bbox']) >
                               .8 * item['bbox'][2] * item['bbox'][3] for item in plate['rows'])
            trapped_art = _text_in_retained_artwork(record, plate['rows'], Path(row['scene']).parent,
                                                   row['snapshot']['source_sha256'])
            if trapped_art:
                artwork_text[oid] = trapped_art
            if covers_page or trapped_art or (is_crop and (len(seeds) != 1 or oid not in editables or
                                            record.get('move_safety') != 'MOVABLE' or trapped_text)):
                unscoped.append(oid)
        replaced = {aid: {'field':tid, 'asset_sha256':proof['asset_sha256']}
                    for tid in editables for aid, proof in selected[tid].get('text_cutouts', {}).items()}
        pictures = {oid:r['asset_reference']['sha256'] for oid,r in selected.items() if r.get('asset_reference')}
        pages.append({'unscoped_crops':unscoped, 'raster_text_in_artwork':artwork_text,
                      'plate_audit':plate,'page':page, 'source_sha256':current['host_evidence'][k]['source_sha256'],
                      'replaced_text_cutouts':replaced, 'picture_hashes':pictures,
                      'passed_edits':sorted(f'{oid}/{action}' for oid in editables for action in actions[oid]),
                      'editable_count':len(editables) if verified else None, 'selected_editables':editables,
                      'ocr_native_hits':sorted(set(hits)-pending), 'revised_ocr_lines':sorted(updated),
                      'pending_readings':{i:v for i,v in readings.items() if v['status']=='pending_confirmation'},
                      'model_confirmed_readings':{i:v for i,v in readings.items() if v['status']=='model_confirmed'},
                      'native_field_count':len(fields), 'local_fidelity':local, 'minor_deviations':minor,
                      'max_local_mae':max((v['mae_rgb'] for v in local.values()), default=None), 'scorecard':measures})
    return {'pptx_sha256':sha,'office_status':office.get('status','UNKNOWN'),'verified':verified,
            'fidelity_limits':load_tolerance_config()['fidelity'],'pages':pages}


def regressions(base, candidate, *, reviewed_reading_draft=False):
    reasons = []
    if not base['verified'] or not candidate['verified']:
        reasons.append('OFFICE_COMPARISON_UNVERIFIED')
    if len(base['pages'])!=len(candidate['pages']):
        reasons.append('PAGE_COUNT_CHANGED')
    for a,b in zip(base['pages'], candidate['pages']):
        prefix = f"p{a['page']}:"
        old_plate = a.get('plate_audit', {})
        new_plate = b.get('plate_audit', {})
        if old_plate.get('status') == 'MEASURED':
            if new_plate.get('status') != 'MEASURED':
                reasons.append(prefix+'PLATE_AUDIT_MISSING')
            clean = {r['ocr_id'] for r in old_plate['rows'] if r['status'] == 'native'}
            if any(r['ocr_id'] in clean and r['status'] in ('retained','residual','unresolved') for r in new_plate.get('rows', ())):
                reasons.append(prefix+'NEW_VISIBLE_TEXT_ON_PLATE')
        if a['source_sha256']!=b['source_sha256']:
            reasons.append(prefix+'SOURCE_CHANGED')
        if (b['editable_count'] or 0)<(a['editable_count'] or 0):
            reasons.append(prefix+'EDITABLE_COUNT_DECLINED')
        if set(a['ocr_native_hits'])-set(b['ocr_native_hits'])-set(b.get('revised_ocr_lines',[])) or b['native_field_count']<a['native_field_count']:
            reasons.append(prefix+'NATIVE_FIELDS_LOST')
        passed = set(b.get('passed_edits', []))
        for aid, proof in b.get('replaced_text_cutouts', {}).items():
            if a.get('picture_hashes', {}).get(aid) == proof['asset_sha256']:
                passed.update(f'{aid}/{action}' for action in ('move-scale','delete')
                              if f"{proof['field']}/{action}" in passed)
        if set(a.get('passed_edits', [])) - passed:
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
            max_de = max(old['delta_e2000_p95'], limits['hard_max_delta_e2000_p95'])
            min_ssim = min(old['ssim'], limits['hard_min_ssim'])
            reading = b.get('pending_readings', {}).get(region) or b.get('model_confirmed_readings', {}).get(region, {})
            if reviewed_reading_draft and reading.get('after') and reading['after'] != reading.get('before'):
                # A different transcription necessarily changes the ink inside its own label.
                # Bound the additional change relative to the control, even in a poor-fit label.
                reading_limits = load_tolerance_config()['fidelity']
                max_de = max(max_de, old['delta_e2000_p95'] + reading_limits['reading_delta_e_margin'])
                min_ssim = min(min_ssim, old['ssim'] - reading_limits['reading_ssim_margin'])
            if de > max_de or similarity < min_ssim:
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
        if request.get('object_scoped_delivery'):
            def preserve_text(c, d):
                graph = svc.graph(svc.page(c, page))
                targets = [line.id for line in graph.text.texts if line.id not in graph.selected]
                # Reserve the table/chart as a unit before its labels become committed text.
                # If the structured candidate is not verified, the ordinary text trials still run.
                targets.extend(field['id'] for field in c.get('marked_plans', {}).get(str(page), ())
                               if field['route'] in ('table', 'chart'))
                if not targets:
                    return {'status':'NO_TEXT_TARGETS'}
                outcome = svc.batch_reconstruct_page(state,c,d,page=page,object_ids=targets,
                    include_assets=False,include_containers=False,include_region_crops=False,verify_office=True)
                if outcome['status'] == 'VALIDATED':
                    return svc.commit_region(state,c,d/'commit',candidate_id=outcome['candidate_id'])
                return {'status':outcome['status']}
            step(f'preserve-text-{page}', preserve_text)
            if request.get('native_variant') != 'joint-only':
                def preserve_panels(c, d):
                    inventory_path = c.get('experiment_marks', {}).get(str(page))
                    if not inventory_path:
                        return {'status':'NO_PANEL_TARGETS'}
                    inventory = read(inventory_path)
                    if inventory['source_sha256'] != sources[page-1]:
                        raise RuntimeConflict('panel inventory source changed')
                    card_ids = {r['id'] for r in c.get('marked_plans',{}).get(str(page),[])
                                if r['route']=='surface' and r.get('move_with')}
                    boxes = [r['bbox'] for r in inventory['rows'] if r['label']=='card_frame' or r['id'] in card_ids]
                    if not boxes:
                        return {'status':'NO_PANEL_TARGETS'}
                    outcome = svc.batch_reconstruct_page(state,c,d,page=page,panel_regions=boxes,verify_office=True)
                    if outcome['status']=='VALIDATED':
                        return svc.commit_region(state,c,d/'commit',candidate_id=outcome['candidate_id'])
                    return {'status':outcome['status']}
                step(f'preserve-panels-{page}', preserve_panels)
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
        inventory_path = state.read().get('experiment_marks', {}).get(str(page))
        if inventory_path and request.get('native_variant') != 'joint-only':
            inventory = read(inventory_path)
            if inventory['source_sha256'] != sources[page-1]:
                raise RuntimeConflict('card inventory source changed')
            if request.get('object_scoped_delivery'):
                from ..recovery.layered_card import ordered_card_frames
                frames = ordered_card_frames(inventory['rows'])
            else:
                frames = sorted((r for r in inventory['rows'] if r['label'] == 'card_frame'),
                                key=lambda r: -r['bbox'][2] * r['bbox'][3])
            for frame in frames:
                def finish_frame(current, directory):
                    staged = svc.stage_photo_frame(state, current, directory, page, frame)
                    cid = staged.get('candidate_id')
                    if cid:
                        checked = svc.validate_candidate(state, current, directory/'validate', candidate_id=cid)
                        if checked['status'] == 'PASS':
                            return svc.commit_region(state, current, directory/'commit', candidate_id=cid)
                        return {'status': checked['status'], 'frame': frame['id']}
                    return staged
                step(f"photo-frame-{page}-{frame['id']}", finish_frame)
        if inventory_path and request.get('object_scoped_delivery') and request.get('native_variant') != 'joint-only':
            def finish_groups(c, d):
                from .marked_routes import bind_plan, plan_move_groups
                row = svc.page(c, page)
                graph = svc.graph(row)
                plans = c.get('marked_plans', {}).get(str(page), [])
                routes = bind_plan(plans, inventory_path, graph, row['snapshot']['path'], separate_cards=True)
                sources = {}
                groups = plan_move_groups(routes, graph, sources=sources)
                existing = [set(r['move_members']) for r in graph.selected.values() if r.get('move_members')]
                ids = [sources[tuple(g)]['id'] for g in groups if not any(set(g) <= old for old in existing)]
                if not ids:
                    return {'status':'NO_NEW_MOVEMENT_GROUPS'}
                staged = svc.batch_reconstruct_page(state,c,d,page=page,object_ids=ids,
                    include_assets=False,include_containers=False,include_region_crops=False,verify_office=True)
                if staged['status']=='VALIDATED':
                    svc.commit_region(state,c,d/'commit',candidate_id=staged['candidate_id'])
                old = c.get('regional_failures',{}).get(str(page), [])
                failures = [f for f in old if f.get('kind')!='group' or f['object_id'] not in ids]
                failures.extend(staged.get('failed_regions', []))
                c.setdefault('regional_failures',{})[str(page)] = failures
                c.setdefault('questions',{})[str(page)] = questions(state,c,page,failures,d)
                return {'status':staged['status'],'objects':ids}
            step(f'finish-groups-{page}', finish_groups)
    step('compose',lambda c,d:svc.compose_delivery(state,c,d))
    step('office',lambda c,d:svc.validate_office(state,c,d))
    current = state.read()
    score = assess(job,current)
    if request.get('object_scoped_delivery') and score['verified']:
        from .late_fields import register
        evidence_added=False
        for measured in score['pages']:
            findings=[r for r in measured['plate_audit']['rows'] if r['status']=='plate_only' or
                      (r['status'] in ('retained','unresolved') and not r['mark_ids'])]
            if findings and measured['plate_audit']['status']=='MEASURED':
                number=measured['page']
                def discover(c,d):
                    records=register(svc,state,c,d,page=number,findings=findings)
                    return {'ready':[r['id'] for r in records if r['status']=='ready'], 'observed':len(records),
                            'numbered':sum('mark' in r for r in records)}
                evidence_added |= bool(step(f'discover-plate-fields-{number}',discover).get('numbered'))
        if evidence_added:
            current = state.read()
            score = assess(job,current)
    # Trial against an isolated state. Only adopt a new page after the real
    # compiled candidate removes an observed finding and keeps prior edits.
    for page_score in score['pages']:
        page = page_score['page']
        statuses = ('retained','unresolved','residual') if request.get('object_scoped_delivery') else ('retained','unresolved')
        retained = [r for r in page_score['plate_audit']['rows'] if r['status'] in statuses]
        residual_ids = {oid for r in retained if r['status'] == 'residual' for oid in r.get('exact_native_fields', ())}
        ids = [*sorted(residual_ids), *sorted({oid for r in retained for oid in r['mark_ids']}-residual_ids)]
        if not ids or not score['verified']:
            continue
        def retry_plate(c, d):
            trial = deepcopy(c)
            reopened, skipped = _reopen_residual_fields(svc, trial, page, residual_ids.intersection(batch), d/'reopened') if residual_ids.intersection(batch) else ([], {})
            target_ids = [i for i in batch if i not in skipped]
            write(d/'retry-fields.json', {'requested':batch,'reopened':reopened,'skipped':skipped})
            if not target_ids:
                return {'status':'KEPT_BASELINE', 'reasons':skipped}
            (d/'stage').mkdir(parents=True,exist_ok=True)
            attempt = svc.batch_reconstruct_page(state, trial, d/'stage', page=page,
                object_ids=target_ids, include_assets=False, include_containers=False,
                include_region_crops=False, verify_office=True)
            if attempt['status'] != 'VALIDATED':
                return {'status':attempt['status']}
            svc.commit_region(state, trial, d/'commit', candidate_id=attempt['candidate_id'])
            svc.compose_delivery(state, trial, d/'compose')
            svc.validate_office(state, trial, d/'office')
            measured = assess(job, trial)
            audit = measured['pages'][page-1]['plate_audit']
            recovered = {r['ocr_id'] for r in audit['rows'] if r['status'] == 'native' and r.get('exact_native_fields')}
            before = score['pages'][page-1]['plate_audit']['rows']
            recovered.intersection_update(r['ocr_id'] for r in before if r['status'] in statuses
                and set(r['mark_ids'] + r.get('exact_native_fields', [])) & set(target_ids))
            failures = regressions(score, measured)
            if not recovered or failures:
                return {'status':'KEPT_BASELINE','recovered_fields':sorted(recovered),
                        'reasons':failures or ['NO_ADDITIONAL_NATIVE_FIELD']}
            c.update(trial)
            return {'status':'ADOPTED','recovered':sorted(recovered)}
        def safe_retry(c, d):
            try:
                return retry_plate(c, d)
            except (ValueError, RuntimeError, OSError) as exc:
                return {'status':'KEPT_BASELINE','error_type':type(exc).__name__}
        for offset in range(0,len(ids),30):
            batch = ids[offset:offset+30]
            name = f'plate-text-retry-{page}' + (f'-{offset//30+1}' if offset else '')
            outcome = step(name, safe_retry)
            current = state.read()
            if outcome['status'] == 'ADOPTED':
                score = assess(job,current)
    def expose_plate_findings(c, d):
        for page_score in score['pages']:
            page = page_score['page']
            k = str(page)
            issues = {oid:r['status'] for r in page_score['plate_audit']['rows']
                      if r['status'] in ('residual','retained','unresolved','plate_only') for oid in r['mark_ids']}
            if request.get('object_scoped_delivery'):
                issues.update((oid,'source_reading_disagreement') for r in page_score['plate_audit']['rows']
                              if r.get('source_reading_disagrees') for oid in r['mark_ids'])
            c.setdefault('plate_issues', {})[k] = issues
            c['questions'][k] = questions(state, c, page, c['regional_failures'][k], d)
        return {'pages':len(score['pages'])}
    step('background-questions', expose_plate_findings)
    current = state.read()
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
    info = {'origin':ORIGIN,'native_variant':request.get('native_variant','combined'),
            'text_profile':{name:bool(request.get(name,False)) for name in ('region_text','page_reading')},
            'sources':sources,'uploaded_sha256':originals,'finished':time.time(),
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


def prepare_reading_draft(job):
    """Start the Agent from Office-checked source readings; keep the control immutable."""
    state = RuntimeState(job)
    service = RuntimeService(job.parent)
    full_page = read(job/'request.json').get('page_reading', False)
    results = []
    for page in range(1, len(state.read()['pages']) + 1):
        def read_page(current, directory):
            graph = service.graph(current['pages'][page-1])
            ids = None if full_page else [
                oid for oid,rec in graph.selected.items() if rec['kind']=='text']
            out = service.review_numbered_text(state,current,directory,page=page,object_ids=ids)
            cid = out.get('candidate_id')
            if not cid:
                return out
            checked = service.validate_candidate(state,current,directory/'validation',candidate_id=cid)
            if full_page and checked.get('status')=='REVIEW_REQUIRED':
                from .batch_backbone import _collect_office_defects
                staged = current['candidates'][cid]
                ids = staged['object_ids']
                frame,failed = _collect_office_defects(checked,graph,ids)
                blocked = set(frame) | failed
                if blocked:
                    reasons = current.setdefault('reading_office_failures',{}).setdefault(str(page),{})
                    reasons.update({oid:'PowerPoint replacement or residual check failed' for oid in blocked})
                keep = [oid for oid in ids if oid not in blocked]
                if blocked and keep:
                    # Reuse the blind calls; only independently safe native edits are restaged.
                    trial = service.review_numbered_text(state,current,directory/'unaffected',page=page,object_ids=keep)
                    if trial.get('candidate_id'):
                        cid = trial['candidate_id']
                        checked = service.validate_candidate(state,current,directory/'validation-unaffected',candidate_id=cid)
            if checked.get('status') != 'PASS':
                return {'status':'REVIEW_REQUIRED','candidate_id':cid}
            committed = service.commit_region(state,current,directory/'commit',candidate_id=cid)
            if len(current['pages'])==1:
                bundle = current['last_verified_bundle']
                write(job/'reading-draft.json',{'pptx':bundle['pptx'],'sha256':bundle['pptx_sha256'],
                      'origin':'moma-reading-draft','candidate_id':cid,'office_status':'PASS',
                      'report':current['candidates'][cid]['reading_receipt']})
            return committed
        results.append(state.mutate('host-review-numbered-text',{'page':page},None,'host-reading-draft-'+str(page),read_page))
    if full_page and len(results)>1 and any(r.get('status')=='COMMITTED' for r in results):
        def check_full_draft(current, directory):
            # Per-page candidate receipts certify just one slide. Export a multi-slide
            # draft only after every page is present in the same Office-checked file.
            service.compose_delivery(state,current,directory/'compose')
            composition = service.composition(current)
            pptx = composition['artifacts']['pptx']
            irs = [Path(row['ir']) for row in current['pages']]
            checked = service.office_check(state,pptx,irs,directory/'office')
            if checked.get('status')=='PASS':
                write(job/'reading-draft.json',{'pptx':pptx,'sha256':digest(pptx),'pages':len(irs),
                      'origin':'moma-reading-draft','office_status':'PASS','reports':current['reading_reports']})
            return {'status':checked.get('status','UNKNOWN')}
        state.mutate('host-check-full-reading-draft',{},None,'host-reading-draft-full',check_full_draft)
    return results


def selected_result(job):
    selection = job/'runtime/delivery-selection.json'
    if not selection.is_file():
        return None
    receipt = read(selection)
    if digest(job/'result.json') != receipt['result_sha256']:
        raise RuntimeConflict('selected report changed')
    if receipt['selected_sha256'] is not None and digest(job/'delivery/result.pptx') != receipt['selected_sha256']:
        raise RuntimeConflict('selected delivery bytes changed')
    return read(job/'result.json')


def publish(job, agent, reasons=None, candidate=None):
    with RuntimeState(job).publication_lock():
        return selected_result(job) or _publish(job,agent,reasons,candidate)


def delivery_defects(score):
    defects = []
    for page in score['pages']:
        audit = page.get('plate_audit', {})
        reasons = []
        if not score.get('verified') or not page.get('selected_editables'):
            reasons.append('NO_VERIFIED_EDITABLE_OBJECT')
        if audit.get('status') != 'MEASURED':
            reasons.append('SOURCE_PLATE_UNVERIFIED')
        retained = [row for row in audit.get('rows', [])
                    if row['status'] in ('retained', 'residual', 'plate_only')]
        if retained:
            reasons.append('TEXT_STILL_ON_BACKGROUND')
        unresolved = [r for r in audit.get('rows', []) if r['status'] == 'unresolved']
        if unresolved:
            reasons.append('SOURCE_TEXT_WITHOUT_NATIVE_FIELD')
        if page.get('unscoped_crops'):
            reasons.append('IMAGE_WITHOUT_SINGLE_OBJECT_OWNER')
        if reasons:
            defects.append({'page': page['page'], 'reasons': reasons, 'retained_fields': retained + unresolved,
                            'unscoped_crops': page.get('unscoped_crops', [])})
    return defects


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
    strict = read(job/'request.json').get('object_scoped_delivery', False)
    if strict and candidate and not reasons and delivery_defects(candidate):
        reasons = ['AGENT_FAILED_OBJECT_SCOPE_GATE']
    fallback = bool(reasons)
    if strict and fallback and delivery_defects(base):
        result = {'status':'REVIEW_REQUIRED', 'completion':'RECONSTRUCTION_INCOMPLETE',
                  'reason':'No verified object-scoped presentation', 'origin': ORIGIN,
                  'failed_pages':delivery_defects(base), 'fallback_reasons':reasons,
                  'office_pass':base['verified'], 'automatic_publication':False, 'human_acceptance':False,
                  'baseline_pages':summary(base), 'independent_go_review':'UNKNOWN', 'wps_acceptance':'UNKNOWN'}
        write(delivery/'summary.json', result)
        install_file(root/'scorecard.json', delivery/'baseline-scorecard.json')
        names = ['summary.json','baseline-scorecard.json']
        for p in base['pages']:
            preview = root/('page-%02d-before.png' % p['page'])
            if preview.is_file():
                name = 'diagnostic-page-%02d.png' % p['page']
                install_file(preview, delivery/name)
                names.append(name)
        result['artifacts'] = {name:digest(delivery/name) for name in names}
        write(job/'result.json',result)
        write(state.root/'delivery-selection.json', {'result_sha256':digest(job/'result.json'),
              'selected_sha256':None,'origin':ORIGIN})
        return result
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
        pending = {str(page['page']): sorted(page['pending_readings']) for page in candidate['pages']
                   if page.get('pending_readings')}
        if pending:
            result.update(status='REVIEW_REQUIRED', reason='CONTENT_CONFIRMATION_PENDING',
                          pending_text_objects=pending, automatic_publication=False)
            result['workflow_states'] = {**result.get('workflow_states', {}), 'final_qa':'REVIEW_REQUIRED'}
        for name in files:
            if name!='summary.json':
                install_file(staging/name,delivery/name)
    changed = not fallback and candidate['pptx_sha256']!=base['pptx_sha256']
    result.update(origin='agent-incremental' if changed else ORIGIN,baseline_sha256=manifest['files']['baseline.pptx'],
                  agent={'participated':(bool(usage.get('tool_calls')) or bool(usage.get('tokens',{}).get('output')))
                        if usage else 'UNKNOWN' if (state.root/'session.json').exists() else False,
                         'delivered_changes':changed, 'fallback_reasons':reasons or [],'usage':{k:v for k,v in usage.items() if k!='tool_calls'}},
                  independent_go_review=('PASS' if not fallback and agent.get('independent_audit',{}).get('passed') is True
                                         and all(p.get('vlm_audit',{}).get('provider','opencode_go')=='opencode_go'
                                                 for p in agent['independent_audit'].get('pages',[])) else 'UNKNOWN'),
                  wps_acceptance='UNKNOWN', baseline_pages=summary(base))
    shutil.copyfile(root/'baseline.pptx',delivery/'deterministic-baseline.pptx')
    shutil.copyfile(root/'scorecard.json',delivery/'baseline-scorecard.json')
    result['text_profile'] = manifest.get('text_profile', {'region_text':False, 'page_reading':False})
    if candidate:
        write(delivery/'agent-scorecard.json',candidate)
    reading_reports = {page:[read(path) for path in paths] for page,paths in current.get('reading_reports',{}).items()}
    if reading_reports:
        for page, reports in reading_reports.items():
            dual = [report for report in reports if report.get('protocol')=='two-blind-views/1']
            if dual:
                from .page_reading import bind_delivered_text
                graph = read((frozen if fallback else current)['pages'][int(page)-1]['scene'])
                for report in dual:
                    bind_delivered_text(report,graph,fallback)
                    failures = current.get('reading_office_failures',{}).get(page,{})
                    for row in report.get('observations',[]):
                        if row['object_id'] in failures and row['written_text']!=row['proposed']:
                            row['reconstruction_reason'] = failures[row['object_id']]
        write(delivery/'reading-decisions.json', {'candidate_only':fallback, 'pages':reading_reports})
    result.pop('artifacts',None)
    write(delivery/'summary.json',result)
    names = {'result.pptx','deterministic-baseline.pptx','baseline-scorecard.json','summary.json'}
    names.update(f"page-{p['page']:02d}.png" for p in base['pages'])
    if reading_reports:
        names.add('reading-decisions.json')
    draft_path = job/'reading-draft.json'
    if draft_path.is_file():
        draft = read(draft_path)
        if draft.get('office_status')=='PASS' and digest(draft['pptx']) == draft['sha256']:
            install_file(draft['pptx'],delivery/'reading-draft.pptx')
            names.add('reading-draft.pptx')
    if not fallback:
        names.update(files)
    if candidate:
        names.add('agent-scorecard.json')
        trial = state.root/'agent-delivery/result.pptx'
        if candidate.get('verified') and trial.is_file() and digest(trial) == candidate['pptx_sha256']:
            install_file(trial, delivery/'agent-candidate.pptx')
            names.add('agent-candidate.pptx')
    if strict:
        names = {name for name in names if not name.lower().endswith('.pptx') or name == 'result.pptx'}
    result['artifacts'] = {name:digest(delivery/name) for name in sorted(names) if (delivery/name).is_file()}
    write(job/'result.json',result)
    if not fallback and result.get('pending_text_objects') and current['status'] != 'REVIEW_REQUIRED':
        current.update(status='REVIEW_REQUIRED',version=current['version']+1)
        write(state.root/'state.json',current)
    write(state.root/'delivery-selection.json',{'result_sha256':digest(job/'result.json'),
          'selected_sha256':result['artifacts']['result.pptx'],'origin':result['origin']})
    return result
