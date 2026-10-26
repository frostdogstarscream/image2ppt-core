"""Source-measured text defaults for compact model authoring, never acceptance."""
from copy import deepcopy
import math
import time
import numpy as np
from PIL import Image

from .typography import estimate_text, font_status
from .renderer_calibration import origin_correction
from .auto_protocol import validate_plan


def measure_text_defaults(source, inventory, *, renderer_version, deadline, progress=None):
    with Image.open(source) as image:
        pixels = np.asarray(image.convert('RGB'))
    w, h = inventory['image_size']; scale = 1280/w
    available = font_status()['families']
    family = '微软雅黑' if '微软雅黑' in available else available[0]
    texts = []; diagnostics = []
    for index, original in enumerate(inventory['texts']):
        if time.time() >= deadline:
            raise TimeoutError('source typography budget exhausted')
        x, y, bw, bh = original['bbox']; size = max(1, min(400, bh*1.1))
        ty = max(0, y-bh*.15)
        text = {'id': original['id'], 'action': 'native', 'text': original['text'],
                'bbox': [x, ty, min(w-x, bw+4), min(h-ty, max(bh+4, size*1.5))],
                'font_size_px': size, 'font_family': family, 'font_latin': family,
                'font_confidence': .25, 'font_evidence': 'OCR-size fallback; source font not identified.',
                'bold': False, 'color': '222222', 'align': 'left', 'z': 10000+index}
        estimate = estimate_text(pixels, original, isolate_ocr_window=True)
        diagnostic = estimate.get('diagnostics', {})
        fit = diagnostic.get('font', {})
        applied = (estimate.get('native_eligible') and fit.get('dimension_error', 0) <= .12
                   and fit.get('shape_overlap', 1) >= .58 and estimate.get('font_family') in available)
        if applied:
            a, b, width, height = estimate['bbox']; size = estimate['font_size_px']
            origin = fit['render_origin']; glyph = fit.get('glyph_bbox')
            if glyph:
                origin = [a-glyph[0], b-glyph[1]]
            correction = origin_correction(estimate, [w, h], [1280, round(h*scale)], renderer_version=renderer_version)
            tx, ty = max(0, float(origin[0])), max(0, float(origin[1])+correction['dy'])
            text.update(bbox=[tx, ty, min(w-tx, max(width+3, size*len(text['text'])*.45)),
                              min(h-ty, max(size*1.5, height+4))],
                        font_size_px=size, font_family=estimate['font_family'], font_latin=estimate['font_family'],
                        bold=estimate['bold'], color=estimate['color'].lstrip('#'), font_confidence=.65,
                        font_evidence='Source pixel font/ink fit; candidate font, not identified original.')
            if estimate.get('runs'):
                text['runs'] = [{**r, 'color': r['color'].lstrip('#')} for r in estimate['runs']]
        elif (estimate.get('color') and diagnostic.get('foreground_color_count') == 1
              and diagnostic.get('background', {}).get('reliable')):
            text['color'] = estimate['color'].lstrip('#')
        texts.append(text)
        diagnostics.append({'id': text['id'], 'applied': bool(applied), 'diagnostics': diagnostic})
        if progress:
            progress(index+1, len(inventory['texts']))
    return {'schema': 'AutomaticTextDefaults/1', 'source_sha256': inventory['source_sha256'],
            'renderer_version': renderer_version, 'text_decisions': texts, 'diagnostics': diagnostics,
            'human_acceptance': False}


def expand_draft(draft, defaults, inventory):
    if draft.get('schema') == 'ElementPlan/3':
        return validate_plan(draft, inventory)
    draft = deepcopy(draft)
    if draft.get('schema') == 'ElementPlanDraft/2':
        if draft.pop('coordinate_space', None) != 'normalized_1000_xyxy':
            raise ValueError('draft /2 requires explicit normalized_1000_xyxy coordinates')
        w, h = inventory['image_size']
        for record in draft['objects']+draft['text_overrides']:
            for field in ('bbox', 'source_crop_bbox', 'dest_bbox'):
                if field not in record:
                    continue
                b = record[field]
                if (not isinstance(b, list) or len(b) != 4 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in b) or
                    not (0 <= b[0] < b[2] <= 1000 and 0 <= b[1] < b[3] <= 1000)):
                    raise ValueError(f"invalid normalized box: {record.get('id')} {field}")
                record[field] = [b[0]*w/1000, b[1]*h/1000, (b[2]-b[0])*w/1000, (b[3]-b[1])*h/1000]
        draft['schema'] = 'ElementPlanDraft/1'
    if set(draft) != {'schema', 'source_sha256', 'image_size', 'background_color', 'objects', 'text_overrides', 'limitations'}:
        raise ValueError('invalid compact draft fields')
    if draft['schema'] != 'ElementPlanDraft/1' or defaults['source_sha256'] != inventory['source_sha256']:
        raise ValueError('compact draft binding mismatch')
    texts = {t['id']: deepcopy(t) for t in defaults['text_decisions']}
    seen = set()
    for update in draft['text_overrides']:
        if update['id'] not in texts or update['id'] in seen:
            raise ValueError('invalid compact text override ID')
        seen.add(update['id'])
        texts[update['id']].update(deepcopy(update))
    # Crop ownership is an explicit model decision; normalize its redundant
    # text action so the complete downstream ElementPlan/3 remains unambiguous.
    for obj in draft['objects']:
        for tid in obj.get('contains_text_ids', []):
            if tid not in texts:
                raise ValueError('unknown retained OCR text')
            texts[tid]['action'] = 'retained_in_asset'
    plan = {k: deepcopy(draft[k]) for k in ('source_sha256', 'image_size', 'background_color', 'objects', 'limitations')}
    plan.update(schema='ElementPlan/3', text_decisions=list(texts.values()))
    return validate_plan(plan, inventory)


def compact_evidence_texts(defaults, size):
    """Describe measured candidates in the explicitly declared model coordinate space."""
    w, h = size
    result = []
    for original in defaults['text_decisions']:
        text = {k: deepcopy(v) for k, v in original.items()
                if k in ('id', 'text', 'font_size_px', 'font_family', 'bold', 'color', 'align', 'runs')}
        x, y, bw, bh = original['bbox']
        text['bbox'] = [round(v, 3) for v in (x*1000/w, y*1000/h, (x+bw)*1000/w, (y+bh)*1000/h)]
        result.append(text)
    return result


def draft_feedback(draft, inventory):
    """Source-grounded constraint diagnostics, never edits to the model proposal."""
    from .supervised_elements import GEOMETRIES, _intersection
    issues = []; w, h = inventory['image_size']
    normalized = draft.get('schema') == 'ElementPlanDraft/2' and draft.get('coordinate_space') == 'normalized_1000_xyxy'
    for obj in draft.get('objects', []):
        if obj.get('kind') == 'native_shape' and obj.get('geometry') not in GEOMETRIES:
            issues.append({'id': obj.get('id'), 'error': 'unsupported_geometry', 'allowed': sorted(GEOMETRIES),
                           'hint': 'Use a tightly bounded raster_asset for unsupported complex icons.'})
        if obj.get('kind') != 'raster_asset':
            continue
        b = obj.get('source_crop_bbox', [])
        if not isinstance(b, list) or len(b) != 4 or any(type(v) not in (int,float) for v in b):
            continue
        crop = [b[0]*w/1000,b[1]*h/1000,(b[2]-b[0])*w/1000,(b[3]-b[1])*h/1000] if normalized else b
        contained = []; crossing = []
        for text in inventory['texts']:
            box = text['bbox']; overlap = _intersection(crop, box)/(box[2]*box[3])
            if overlap >= .95: contained.append(text['id'])
            elif overlap > .12: crossing.append(text['id'])
        if set(contained) != set(obj.get('contains_text_ids', [])) or crossing:
            issues.append({'id': obj['id'], 'error': 'source_crop_text_ownership',
                           'fully_contained_text_ids': contained, 'crossing_text_ids': crossing,
                           'crossing_text_boxes_source_xywh': {t['id']: t['bbox'] for t in inventory['texts'] if t['id'] in crossing},
                           'hint': 'Declare exactly the fully contained IDs. Adjust the crop to exclude crossing text, or expand it to fully include that text and disclose retention. If no valid crop is possible, omit this raster object and disclose the missing visual in limitations for review; do not submit the unchanged invalid crop.'})
    return issues[:40]
