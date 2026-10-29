"""Pre-inpainting evidence scan and deterministic text/decoration conflicts."""
import copy
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ..storage import digest
from .loop_review import clip
from .textmask import analyze_page, split_wide_gaps


def scan(source, output, ocr):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    with Image.open(source) as image:
        rgb = np.asarray(image.convert('RGB')).copy()
    lines = split_wide_gaps(rgb, ocr(rgb))
    for i, line in enumerate(lines):
        line['id'] = f't{i:03d}'
    records, text_mask, icon_mask = analyze_page(rgb, lines)
    Image.fromarray(rgb).save(output/'source.png')
    Image.fromarray(text_mask.astype(np.uint8)*255).save(output/'text-mask.png')
    Image.fromarray(icon_mask.astype(np.uint8)*255).save(output/'icon-mask.png')
    result = {'schema': 'LayeredScan/1', 'source_sha256': digest(source), 'image_size': [rgb.shape[1], rgb.shape[0]],
              'ocr': lines, 'text': records}
    (output/'scan.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result


def candidates(scan):
    records = {r['id']: r for r in scan['text']}; out = []
    for line in scan['ocr']:
        record = records.get(line['id'], {}); text = line['text'].strip(); reasons = []; priority = 0
        if re.match(r'^[1Il|丨只口]\s*[^\W\d_]', text) and len(text) > 3:
            reasons.append('possible_leading_decoration'); priority += 90
        if record.get('icon_candidates') or record.get('icon_text_split'):
            reasons.append('icon_text_boundary'); priority += 70
        if len(text) <= 3 and text.isascii() and (record.get('status') != 'ok' or line.get('score', 1) < .9):
            reasons.append('short_glyph_conflict'); priority += 60
        if line.get('score', 1) < .8:
            reasons.append('low_ocr_confidence'); priority += 20
        if not reasons:
            continue
        box = record.get('badge', {}).get('bbox') or line['bbox']
        out.append({'id': line['id'], 'text': line['text'], 'bbox': clip(box, scan['image_size'], 2),
                    'context_bbox': clip(box, scan['image_size'], max(12, box[3]*.6)),
                    'reasons': reasons, 'priority': priority})
    return sorted(out, key=lambda r: (-r['priority'], r['bbox'][1], r['bbox'][0]))


def guarded_lines(scan, guards):
    """Apply only previously corroborated decisions, preserving original OCR."""
    if guards.get('source_sha256') != scan['source_sha256']:
        raise ValueError('pre-review source binding mismatch')
    lines = copy.deepcopy(scan['ocr']); known = {t['id'] for t in lines}
    if (set(guards.get('corrections', {})) | set(guards.get('retained', {}))) - known:
        raise ValueError('unknown pre-review OCR id')
    for line in lines:
        correction = guards.get('corrections', {}).get(line['id'])
        if correction:
            if correction.get('independent_text') != correction['text']:
                raise ValueError('uncorroborated correction')
            line['original_text'] = line['text']; line['text'] = correction['text']
    return lines


def protect_masks(text_mask, icon_mask, lines, records, guards):
    H, W = text_mask.shape
    by_id = {r['id']: r for r in records}
    protected = []
    for line in lines:
        if line['id'] not in guards.get('retained', {}):
            continue
        record = by_id.get(line['id'], {})
        box = record.get('badge', {}).get('bbox') or line['bbox']
        x, y, w, h = clip(box, [W, H], 3)
        text_mask[y:y+h, x:x+w] = False
        icon_mask[y:y+h, x:x+w] = True
        protected.append([x, y, w, h])
    return protected
