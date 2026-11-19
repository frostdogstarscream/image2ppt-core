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
            if tentative:
                valid = (row.get('after') in row.get('alternatives', []) and row.get('context_sha256')
                         and row.get('choice_receipt') and not quantity_changed(row['before'],row['after']))
            else:
                valid = row.get('iou_before') is not None and row.get('iou_after') is not None
            if before.get(oid) != row['before'] or not valid or not row.get('crop_sha256'):
                raise ValueError('invalid source reading evidence')
            before[oid] = row['after']
    if before != after:
        raise ValueError('confirmed text changed')


def stage_readings(service, state, current, directory, *, page, object_ids):
    row = service.page(current, page)
    scene = service.graph(row)
    ocr = json.loads(Path(current['ocr'][str(page)]).read_text('utf8'))
    lines = {line['id']:line for line in ocr['lines']}
    fonts = json.loads(Path(current['fonts']).read_text('utf8'))
    request = json.loads((state.job/'request.json').read_text('utf8'))
    budget = AdaptiveBudget(Store(state.job/'state'), request.get('adaptive_budget'))
    source = Path(row['snapshot']['path'])
    if digest(source) != scene.source_sha256:
        raise ValueError('source reading image changed')
    image = Image.open(source).convert('RGB')
    pixels = np.asarray(image)
    changes, observ = [], []
    reading_model = os.environ.get('IMAGE2PPT_MOMA_READING_MODEL')
    if reading_model:
        from .moma_review import MomaReviewer
        actor = MomaReviewer(reading_model)
    else:
        actor = planning_actor(state.job)
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
        except (BudgetExceeded, PipelineError, ValueError) as exc:
            evidence.update(status='pending_confirmation', reason=type(exc).__name__)
            return None
        evidence.setdefault('calls', []).append({'tool': identity['tool'], 'model': getattr(actor,'reported_model',actor.model),
                                                 'usage':getattr(actor,'last_usage',None)})
        return response

    for obj in scene.text.texts:
        members = scene.text.members.get(obj.id, [])
        if (obj.id not in object_ids or obj.id not in scene.selected or len(members) != 1
                or obj.spans or scene.selected[obj.id].get('kind') != 'text'):
            continue
        line = lines.get(members[0], {})
        flagged = any(intersection(obj.bbox, box) >= .5 * obj.bbox[2] * obj.bbox[3] for box in flagged_boxes)
        if line.get('text') != obj.text or line.get('reading_status') or (line.get('score', 1) >= .94 and not flagged):
            continue
        x, y, w, h = map(round, line['bbox'])
        crop = image.crop((x,y,x+w,y+h))
        crop_path = directory / (obj.id+'.png')
        crop.resize((2*w,2*h), Image.Resampling.LANCZOS).save(crop_path)
        evidence = {'object_id':obj.id, 'before':obj.text, 'crop_sha256':digest(crop_path)}
        margin = max(1, round(h * .55))
        context_path = directory / (obj.id+'-context.png')
        image.crop((max(0,x-margin),max(0,y-margin),min(image.width,x+w+margin),
                    min(image.height,y+h+margin))).save(context_path)
        evidence['context_sha256'] = digest(context_path)
        evidence['ocr_id'] = line['id']
        observ.append(evidence)
        identity = {**evidence,'model':actor.model,'source':scene.source_sha256,'tool':'proofread-text/1'}
        prompt = ('Transcribe the most likely printed wording from these source crops. Where the exact strokes '
                  'are unclear, include your best reading and set legible=false. Do not invent missing quantities.'
                  if reading_model else 'Transcribe only the exact visible glyphs and mark unclear glyphs legible=false.')
        reading = call(identity, prompt, {}, [crop_path,context_path], CropReading, evidence)
        if reading is None:
            continue
        evidence.update(reading=reading.text,legible=reading.legible,model=actor.model)
        if not reading.text.strip() or reading.text == obj.text:
            evidence['status'] = 'agreed' if reading.text.strip() and reading.legible else 'pending_confirmation'
            continue
        if quantity_changed(obj.text,reading.text):
            evidence['status'] = 'pending_confirmation'
            evidence['reason'] = 'numeric_change_requires_independent_confirmation'
            continue
        _, verified = verify_and_apply_ocr_corrections(pixels, {'lines':[dict(line)]}, [
            OCRCorrection(ocr_id=line['id'], corrected_text=reading.text, reason='numbered-source-crop')], fonts)
        if reading_model:
            choices = [obj.text, reading.text.strip()]
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
        fitted = reading_frame(obj, obj.text, change['after'], scene, fonts, image.width)
        if fitted is None:
            evidence.update(status='pending_confirmation', reason='insufficient_space_for_reading')
            continue
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
        for field in ('payload', 'source_text_definition'):
            if field in rec:
                rec[field]['text'] = obj.text
    proof = {'source_sha256':scene.source_sha256, 'observations':observ, 'changes':changes}
    write(directory/'readings.json', proof)
    if reading_model:
        current.setdefault('reading_reports',{}).setdefault(str(page),[]).append(str(directory/'readings.json'))
    if not changes:
        return {'page':page, 'status':'NO_VERIFIED_READING','checked':len(observ)}
    bundle = directory/'staged'
    shutil.copytree(Path(row['ir']).parent, bundle)
    write(bundle/'scene.json', scene.model_dump())
    ir = bundle/'slideir.json'
    write(ir, adapt(scene,bundle))
    ids = [change['object_id'] for change in changes]
    cid = 'c-'+key([digest(ir),digest(directory/'readings.json'),current['version']])[:24]
    current['candidates'][cid] = {'id':cid,'kind':'text_reading','status':'STAGED_LOCAL',
        'strategy':'numbered-source-reading','page':page,'object_id':ids[0],'object_ids':ids,
        'parent_ir_sha256':digest(row['ir']),'parent_scene_sha256':digest(row['scene']),
        'code':service.code_binding(),'fingerprint':key([ids,digest(ir)]),'ir':str(ir),
        'reading_receipt':str(directory/'readings.json'),
        'files':{str(p):digest(p) for p in [directory/'readings.json', *bundle.rglob('*')] if p.is_file()}}
    return {'page':page,'status':'STAGED_LOCAL','candidate_id':cid,'object_ids':ids}

