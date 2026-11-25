"""Read doubtful numbered text from the source crop after the host baseline is frozen."""
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ..materialize import digest, write
from ..parallel_rebuild import adapt
from ..parallel_store import BudgetExceeded, Store, key
from ..pipeline import PipelineError
from ..recovery.adaptive_policy import AdaptiveBudget
from .page_structure import OCRCorrection, verify_and_apply_ocr_corrections
from .provider import planning_actor


class CropReading(BaseModel):
    legible: bool
    text: str = Field(max_length=500)


class ReadingChoice(BaseModel):
    index: int = Field(ge=1, le=2)


_QUANTITIES = re.compile(
    r'[负正+-]?(?:\d+[.,]?\d*|[零〇一二两三四五六七八九十百千万亿]+)(?:\s*(?:万|亿|千|百))?'
    r'(?:\s*(?:个百分点|万元|亿元|人民币|公里|千米|公斤|小时|分钟|秒|米|吨|项|套|名|个|例|人|元|年|月|日|万|亿|%|％|‰|[A-Za-z]+))?'
    r'|[￥¥$€£%％‰]|(?<![A-Za-z])(?:kg|km|cm|mm|ms|GB|MB|kWh)(?![A-Za-z])'
)


def quantity_changed(before, after):
    from unicodedata import normalize
    def quantities(text):
        text = normalize('NFKC', text)
        declared_units = re.findall(r'(?:单位\s*[:：]?\s*|[（(])((?:万|亿)?元|千?米|公斤|吨|人次|小时|分钟|个|项|名)(?=[）)\s，,。]|$)', text)
        return _QUANTITIES.findall(text), declared_units
    return quantities(before) != quantities(after)


def reading_frame(obj, before, after, scene, fonts, canvas_width):
    """Keep the original origin and erase mask; resize a longer single-line editable label."""
    from math import ceil
    from PIL import ImageFont
    from ..contracts import intersection
    x, y, w, h = obj.bbox
    if after == before or obj.align != 'left' or '\n' in before or '\n' in after:
        return list(obj.bbox)
    style = next(s for s in scene.text.styles if s.id == obj.style)
    family = style.ea if any(ord(c)>255 for c in before+after) else style.latin
    font = next((v for v in fonts.get('fonts', []) if v['family']==family and v['bold']==style.bold), None)
    if not font:
        return list(obj.bbox)
    face = ImageFont.truetype(font['file'], max(1, round(style.size_px)), index=font.get('index', 0))
    old_advance, new_advance = face.getlength(before), face.getlength(after)
    if old_advance <= 0 or new_advance <= old_advance:
        return list(obj.bbox)
    # The current frame has already passed Office. Carry its measured clearance to
    # the new string, including Office's spacing when it differs from Pillow's.
    need = ceil(max(new_advance + 1, w * new_advance / old_advance))
    extension = [x+w, y, max(0, need-w), h]
    if x+need > canvas_width or any(t.id!=obj.id and intersection(extension, t.bbox)>0 for t in scene.text.texts):
        return None
    return [x,y,max(w,need),h]


def authorize_readings(original, revised, receipt):
    before = {t.id:t.text for t in original.text.texts}
    after = {t.id:t.text for t in revised.text.texts}
    if receipt and receipt.get('source_sha256') == original.source_sha256:
        for row in receipt['changes']:
            oid = row['object_id']
            tentative = row.get('status') == 'pending_confirmation'
            if row.get('protocol') == 'two-blind-views/1':
                from .page_reading import authorized
                valid = authorized(row, original.source_sha256)
            elif tentative:
                valid = (row.get('after') in row.get('alternatives', []) and row.get('context_sha256')
                         and row.get('choice_receipt') and not quantity_changed(row['before'],row['after']))
            else:
                valid = row.get('iou_before') is not None and row.get('iou_after') is not None
            if before.get(oid) != row['before'] or not valid or not row.get('crop_sha256'):
                raise ValueError('invalid source reading evidence')
            before[oid] = row['after']
    if before != after:
        raise ValueError('confirmed text changed')


def stage_readings(service, state, current, directory, *, page, object_ids, explicit=False):
    row = service.page(current, page)
    scene = service.graph(row)
    ocr = json.loads(Path(current['ocr'][str(page)]).read_text('utf8'))
    lines = {line['id']:line for line in ocr['lines']}
    fonts = json.loads(Path(current['fonts']).read_text('utf8'))
    request = json.loads((state.job/'request.json').read_text('utf8'))
    full_page = request.get('page_reading', False)
    budget = AdaptiveBudget(Store(state.job/'state'), request.get('adaptive_budget'),
                            model_page=(page,len(request['sources'])) if full_page else None)
    source = Path(row['snapshot']['path'])
    if digest(source) != scene.source_sha256:
        raise ValueError('source reading image changed')
    image = Image.open(source).convert('RGB')
    pixels = np.asarray(image)
    changes, observ = [], []
    reading_model = os.environ.get('IMAGE2PPT_MOMA_READING_MODEL')
    targets, items = None, None
    if explicit:
        from .reading_targets import resolve_targets
        inventory = json.loads(Path(current['experiment_marks'][str(page)]).read_text('utf8'))
        if inventory['source_sha256'] != scene.source_sha256 or ocr.get('source_sha256') != scene.source_sha256:
            raise ValueError('numbered inventory or OCR belongs to a different source')
        marks = {r['id']:r for r in inventory['rows']}
        relations = current.get('scene_relations', {}).get(str(page), {})
        logos = [r['id'] for r in relations.get('logo_lockup_candidates', [])]
        targets, items = resolve_targets(scene, lines, marks, object_ids,
                                         current.get('marked_plans', {}).get(str(page), []), logos)
        if not targets:
            return stage_changes(service, current, directory, page, row, scene, fonts, image.width,
                                 [], [], save_report=True, items=items)
    if full_page:
        if not reading_model:
            if explicit:
                for item in items:
                    if 'target_id' in item:
                        item.update(status='unavailable', reason='reading_model_not_configured')
                return stage_changes(service,current,directory,page,row,scene,fonts,image.width,[],[],
                                     save_report=True,items=items)
            return {'page':page,'status':'READING_MODEL_UNAVAILABLE','checked':0}
        from .page_reading import observe_page
        observ, changes, calls = observe_page(reading_model,state,current,directory,page,scene,lines,
                                               object_ids,source,image,budget,targets=targets)
        return stage_changes(service, current, directory, page, row, scene, fonts, image.width,
                             observ, changes, save_report=True, calls=calls, items=items)
    try:
        if reading_model:
            from .moma_review import MomaReviewer
            actor = MomaReviewer(reading_model)
        else:
            actor = planning_actor(state.job)
    except (PipelineError, ValueError, RuntimeError) as exc:
        if not explicit:
            raise
        for item in items:
            if 'target_id' in item:
                item.update(status='unavailable', reason=type(exc).__name__)
        return stage_changes(service,current,directory,page,row,scene,fonts,image.width,[],[],
                             save_report=True,items=items)
    from ..contracts import intersection
    flagged_boxes = [defect['bbox'] for audit in current.get('independent_audit', {}).get('pages', [])
                     if audit['page'] == page for defect in audit.get('vlm_audit', {}).get('critical_defects', [])
                     if defect.get('category') == 'text_corruption' and len(defect.get('bbox') or []) == 4]
    def call(identity, prompt, context, images, contract, evidence):
        try:
            if not budget.reserve('model_calls', identity):
                evidence.update(status='pending_confirmation', reason='previous_request_reserved')
                return None
            response = contract.model_validate(actor.ask(prompt, context, images, contract))
        except (BudgetExceeded, PipelineError, ValueError, RuntimeError) as exc:
            reason = str(exc)
            safe = re.fullmatch(r'moma_http_[45]\d\d|moma_response_unknown|invalid_moma_reply', reason)
            evidence.update(status='pending_confirmation', reason=reason if safe else type(exc).__name__)
            return None
        evidence.setdefault('calls', []).append({'tool': identity['tool'], 'model': getattr(actor,'reported_model',actor.model),
                                                 'usage':getattr(actor,'last_usage',None)})
        return response

    if targets is None:
        targets = {}
        for obj in scene.text.texts:
            members = scene.text.members.get(obj.id, [])
            if (obj.id not in object_ids or len(members) != 1 or obj.spans
                    or scene.selected.get(obj.id, {}).get('kind') != 'text'):
                continue
            line = lines.get(members[0], {})
            flagged = any(intersection(obj.bbox, box) >= .5 * obj.bbox[2] * obj.bbox[3] for box in flagged_boxes)
            if line.get('text') == obj.text and not line.get('reading_status') and (line.get('score', 1) < .94 or flagged):
                targets[obj.id] = (line, True)
    for oid, (line, writable) in targets.items():
        before = line['text']
        x, y, w, h = map(round, line['bbox'])
        crop = image.crop((x,y,x+w,y+h))
        stem = oid if re.fullmatch(r'[a-zA-Z0-9_-]{1,60}', oid) else key(oid)
        crop_path = directory / (stem+'.png')
        crop.resize((2*w,2*h), Image.Resampling.LANCZOS).save(crop_path)
        evidence = {'object_id':oid, 'before':before, 'bbox':line['bbox'], 'crop_sha256':digest(crop_path)}
        if 'source_span' in line:
            evidence['source_span'] = line['source_span']
        margin = max(1, round(h * .55))
        horizontal_margin = 0 if line.get('source_span') else margin
        context_path = directory / (stem+'-context.png')
        image.crop((max(0,x-horizontal_margin),max(0,y-margin),min(image.width,x+w+horizontal_margin),
                    min(image.height,y+h+margin))).save(context_path)
        evidence['context_sha256'] = digest(context_path)
        evidence['ocr_id'] = line['id']
        evidence['delivery'] = 'native_field' if writable else 'observed_without_native_field'
        observ.append(evidence)
        identity = {k:evidence[k] for k in ('object_id','before','crop_sha256','context_sha256','ocr_id')}
        identity.update(model=actor.model,source=scene.source_sha256,tool='proofread-text/1')
        evidence.update(crop_file=str(crop_path), context_file=str(context_path))
        prompt = ('Transcribe the most likely printed wording from these source crops. Where the exact strokes '
                  'are unclear, include your best reading and set legible=false. Do not invent missing quantities.'
                  if reading_model else 'Transcribe only the exact visible glyphs and mark unclear glyphs legible=false.')
        if explicit:
            prompt += ' Transcribe the first crop only. The second is context; ignore adjacent fields.'
        reading = call(identity, prompt, {}, [crop_path,context_path], CropReading, evidence)
        if reading is None:
            continue
        evidence.update(reading=reading.text,legible=reading.legible,model=actor.model)
        if not reading.text.strip() or reading.text == before:
            evidence['status'] = 'agreed' if reading.text.strip() and reading.legible else 'pending_confirmation'
            continue
        if quantity_changed(before,reading.text):
            evidence['status'] = 'pending_confirmation'
            evidence['reason'] = 'numeric_change_requires_independent_confirmation'
            continue
        if explicit:
            from .page_reading import protected_reading
            project_field = any(p.get('role') == 'project' and intersection(p['bbox'],line['bbox']) >= .95*w*h
                                for p in current.get('marked_plans',{}).get(str(page), []))
            if project_field or protected_reading(before,reading.text, []):
                evidence.update(status='pending_confirmation', reason='protected_field_needs_separate_source')
                continue
        if not writable:
            evidence.update(status='pending_confirmation', proposed=reading.text.strip(),
                            reason='observed_without_native_field')
            continue
        _, verified = verify_and_apply_ocr_corrections(pixels, {'lines':[dict(line)]}, [
            OCRCorrection(ocr_id=line['id'], corrected_text=reading.text, reason='numbered-source-crop')], fonts)
        if reading_model:
            choices = [before, reading.text.strip()]
            decision = {'tool':'choose-source-reading/1','model':actor.model,'source':scene.source_sha256,
                        'crops':[evidence['crop_sha256'],evidence['context_sha256']], 'alternatives':choices}
            picked = call(decision, 'Select the most likely reading of the printed label from the numbered '
                          'candidates using the two source crops. Return only the index.',
                          {'candidates':dict(enumerate(choices,1))}, [crop_path,context_path], ReadingChoice, evidence)
            if picked is None:
                continue
            evidence.update(alternatives=choices, choice_receipt=key([decision,picked.index]),status='pending_confirmation')
            if picked.index == 1:
                continue
        elif not reading.legible:
            evidence.update(status='pending_confirmation', reason='source_glyphs_unconfirmed')
            continue
        if verified and verified[0]['iou_before'] is not None and verified[0]['iou_after'] is not None:
            change = {**evidence, **verified[0]}
            if not reading_model and reading.legible:
                change['status'] = 'source_supported'
        elif reading_model:
            change = {**evidence,'after':reading.text.strip()}
        else:
            evidence['reason'] = 'source_glyphs_unconfirmed'
            continue
        changes.append(change)
    return stage_changes(service, current, directory, page, row, scene, fonts, image.width,
                         observ, changes, save_report=bool(reading_model) or explicit, items=items)


def reading_outcome(obs):
    """Return actual source transcriptions; preserve disagreement and unanswered views."""
    result = {'before':obs['before'], 'delivery':obs.get('delivery'),
              'reading_status':obs.get('status','pending_confirmation'),
              'reason':obs.get('reason') or obs.get('status') or 'no_reading_returned'}
    read = False
    if 'views' in obs:
        answers = {}
        missing = {}
        for view, receipt in obs['views'].items():
            answer = receipt.get('answer') or {}
            if answer.get('text','').strip():
                answers[view] = {'text':answer['text'], 'legible':answer['legible']}
            else:
                missing[view] = receipt.get('reason') or 'no_reading_returned'
        result['readings'] = answers
        if missing:
            result['unavailable_views'] = missing
        read = bool(answers)
    elif obs.get('reading','').strip():
        result.update(reading=obs['reading'], legible=obs['legible'])
        read = True
    if not read and ('reading' in obs or result.get('reason') == 'no_plausible_reading'):
        result['reason'] = 'no_reading_returned'
    result['status'] = 'observed' if read else 'unavailable'
    return result


def stage_changes(service, current, directory, page, row, scene, fonts, canvas_width,
                  observ, proposals, *, save_report, calls=None, items=None):
    """Publish only source-bound, native text changes; retain the previously verified background."""
    by_id = {t.id:t for t in scene.text.texts}
    observations = {r['object_id']:r for r in observ}
    changes = []
    for change in proposals:
        obj = by_id[change['object_id']]
        evidence = observations[obj.id]
        fitted = reading_frame(obj, obj.text, change['after'], scene, fonts, canvas_width)
        if fitted is None:
            evidence.update(delivery='insufficient_space_for_reading', proposed=change['after'],
                            status='pending_confirmation', reason='insufficient_space_for_reading', after=obj.text)
            continue
        evidence['delivery'] = 'staged_for_office'
        evidence['status'] = change.get('status','source_supported')
        changes.append(change)
        rec = scene.selected[obj.id]
        if list(obj.bbox) != fitted:
            change['frame_before'] = list(obj.bbox)
            change['frame_after'] = fitted
            obj.bbox = fitted
            rec['payload']['bbox'] = fitted
            rec.setdefault('native_layout', {})['proposed_bbox'] = fitted
        obj.text = change['after']
        rec['reading_status'] = change.get('status','source_supported')
        rec['source_reading'] = {k:change.get(k) for k in ('ocr_id','before','after','crop_sha256','context_sha256','reason','status')}
        if 'source_span' in change:
            rec['source_reading']['source_span'] = change['source_span']
        for field in ('payload', 'source_text_definition'):
            if field in rec:
                rec[field]['text'] = obj.text
    proof = {'source_sha256':scene.source_sha256, 'observations':observ, 'changes':changes}
    summary = {}
    if items is not None:
        checked = set()
        for item in items:
            obs = observations.get(item.get('target_id'))
            if not obs:
                if 'target_id' in item and not item.get('reason'):
                    item.update(status='unavailable', reason='no_field_observation')
                continue
            item.update(reading_outcome(obs))
            if item['status'] == 'observed':
                checked.add(obs['object_id'])
        summary = {'checked':len(checked), 'items':items}
        proof.update(summary)
    if calls is not None:
        proof.update(protocol='two-blind-views/1', calls=calls)
    write(directory/'readings.json', proof)
    if save_report:
        current.setdefault('reading_reports',{}).setdefault(str(page),[]).append(str(directory/'readings.json'))
    if not changes:
        unavailable = ((calls and all(call['status']!='ok' for call in calls)) or
                       (items and not summary['checked'] and any(i['status']=='unavailable' for i in items)))
        return {'page':page, 'status':'READING_MODEL_UNAVAILABLE' if unavailable else 'NO_VERIFIED_READING',
                'checked':len(observ), **summary}
    bundle = directory/'staged'
    shutil.copytree(Path(row['ir']).parent, bundle)
    write(bundle/'scene.json', scene.model_dump())
    ir = bundle/'slideir.json'
    write(ir, adapt(scene,bundle))
    ids = [change['object_id'] for change in changes]
    cid = 'c-'+key([digest(ir),digest(directory/'readings.json'),current['version']])[:24]
    evidence = [directory/'readings.json', *bundle.rglob('*'), *directory.glob('*.png')]
    if calls is not None:
        evidence.extend((directory/'source-readings').glob('*.png'))
        evidence.extend(Path(v['receipt_file']) for v in calls)
    current['candidates'][cid] = {'id':cid,'kind':'text_reading','status':'STAGED_LOCAL',
        'strategy':'numbered-source-reading','page':page,'object_id':ids[0],'object_ids':ids,
        'parent_ir_sha256':digest(row['ir']),'parent_scene_sha256':digest(row['scene']),
        'code':service.code_binding(),'fingerprint':key([ids,digest(ir)]),'ir':str(ir),
        'reading_receipt':str(directory/'readings.json'),
        'files':{str(p):digest(p) for p in evidence if p.is_file()}}
    return {'page':page,'status':'STAGED_LOCAL','candidate_id':cid,'object_ids':ids, **summary}

