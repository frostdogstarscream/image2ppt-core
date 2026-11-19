"""Read doubtful numbered text from the source crop after the host baseline is frozen."""
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ..materialize import digest, write
from ..parallel_rebuild import adapt
from ..parallel_store import Store, key
from ..recovery.adaptive_policy import AdaptiveBudget
from .page_structure import OCRCorrection, verify_and_apply_ocr_corrections
from .provider import planning_actor


class CropReading(BaseModel):
    legible: bool
    text: str = Field(max_length=500)


def authorize_readings(original, revised, receipt):
    before = {t.id:t.text for t in original.text.texts}
    after = {t.id:t.text for t in revised.text.texts}
    if receipt and receipt.get('source_sha256') == original.source_sha256:
        for row in receipt['changes']:
            oid = row['object_id']
            if (before.get(oid) != row['before'] or row.get('iou_before') is None
                    or row.get('iou_after') is None or not row.get('crop_sha256')):
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
    actor = planning_actor(state.job)
    for obj in scene.text.texts:
        members = scene.text.members.get(obj.id, [])
        if (obj.id not in object_ids or obj.id not in scene.selected or len(members) != 1
                or obj.spans or scene.selected[obj.id].get('kind') != 'text'):
            continue
        line = lines.get(members[0], {})
        if line.get('text') != obj.text or line.get('score', 1) >= .94:
            continue
        x, y, w, h = map(round, line['bbox'])
        crop = image.crop((x,y,x+w,y+h))
        crop_path = directory / (obj.id+'.png')
        crop.resize((2*w,2*h), Image.Resampling.LANCZOS).save(crop_path)
        evidence = {'object_id':obj.id, 'before':obj.text, 'crop_sha256':digest(crop_path)}
        observ.append(evidence)
        identity = {**evidence,'model':actor.model,'source':scene.source_sha256,'tool':'proofread-text/1'}
        if not budget.reserve('model_calls', identity):
            evidence['reason'] = 'reserved_or_unknown'
            continue
        reading = CropReading.model_validate(actor.ask(
            'Transcribe only the exact visible glyphs. If a character is malformed or unreadable, return legible=false. Do not infer words from context.',
            {},[crop_path],CropReading))
        evidence.update(reading=reading.text,legible=reading.legible,model=actor.model)
        if not reading.legible or reading.text == obj.text:
            continue
        if re.findall(r'[\d%‰]+',reading.text) != re.findall(r'[\d%‰]+',obj.text):
            evidence['reason'] = 'numeric_change_requires_independent_confirmation'
            continue
        _, verified = verify_and_apply_ocr_corrections(pixels, {'lines':[dict(line)]}, [
            OCRCorrection(ocr_id=line['id'], corrected_text=reading.text, reason='numbered-source-crop')], fonts)
        if not verified or verified[0]['iou_before'] is None or verified[0]['iou_after'] is None:
            evidence['reason'] = 'source_glyphs_unconfirmed'
            continue
        change = {**evidence, **verified[0]}
        changes.append(change)
        obj.text = change['after']
        rec = scene.selected[obj.id]
        for field in ('payload', 'source_text_definition'):
            if field in rec:
                rec[field]['text'] = obj.text
    proof = {'source_sha256':scene.source_sha256, 'observations':observ, 'changes':changes}
    write(directory/'readings.json', proof)
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

