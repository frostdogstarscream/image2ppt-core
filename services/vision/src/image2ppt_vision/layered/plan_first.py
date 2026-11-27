"""Consume bound vision plans before segmentation, without new model calls.

Only keep_pixels regions guide raster ownership. OCR and native geometry remain
measured by the existing worker; speculative table/chart/shape actions fall back.
"""
import base64
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..auto_provider import parse_object
from .plan_schema import validate_plan


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def load_bundle(directory, source):
    directory = Path(directory)
    binding = read(directory/'binding.json')
    result_file = directory/('result.json' if (directory/'result.json').exists() else 'job.json')
    result = read(result_file)
    if sha(source) != binding['source_sha256'] or sha(directory/'source.png') != sha(source):
        raise ValueError('plan source hash mismatch')
    for name in ('request', 'response'):
        expected = result.get(name+'_sha256') or binding.get(name+'_sha256')
        if not expected or sha(directory/(name+'.json')) != expected:
            raise ValueError('plan '+name+' hash mismatch')
    response = read(directory/'response.json'); request = read(directory/'request.json')
    if not response.get('ok') or result.get('status') != 'REVIEW_REQUIRED':
        raise ValueError('completed successful plan required')
    if response.get('finish_reason') in ('length', 'content_filter'):
        raise ValueError('incomplete plan response')
    images = [p['image_url']['url'] for m in request['body']['messages']
              if isinstance(m.get('content'), list) for p in m['content'] if p.get('type') == 'image_url']
    if len(images) != 1 or not images[0].startswith('data:image/png;base64,'):
        raise ValueError('plan requires one bound PNG request image')
    if hashlib.sha256(base64.b64decode(images[0].split(',', 1)[1], validate=True)).hexdigest() != sha(source):
        raise ValueError('plan request image mismatch')
    plan = parse_object(response['content']); validate_plan(plan)
    if read(directory/'plan.json') != plan:
        raise ValueError('saved plan differs from model response')
    return plan, result_file.name


def prepare(manifest, sources, destination):
    """Verify every bundle before any worker invocation, then freeze local copies."""
    manifest = Path(manifest).resolve(strict=True); spec = read(manifest)
    if spec.get('schema') != 'LayeredPlanInputs/1' or not isinstance(spec.get('plans'), list):
        raise ValueError('expected LayeredPlanInputs/1 plans list')
    by_hash = {}
    for entry in spec['plans']:
        directory = (manifest.parent/entry).resolve(strict=True)
        key = read(directory/'binding.json')['source_sha256']
        if key in by_hash: raise ValueError('duplicate plan source')
        by_hash[key] = directory
    verified = [(p, by_hash.get(sha(p))) for p in sources]
    if any(d is None for p, d in verified) or len(verified) != len(by_hash):
        raise ValueError('plan and input source sets must match exactly')
    loaded = [(p, d, *load_bundle(d, p)) for p, d in verified]
    target = Path(destination); target.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest, target/'manifest.json')
    outputs = []
    for source, directory, plan, result_name in loaded:
        bundle = target/source.stem; bundle.mkdir()
        for name in ('source.png', 'request.json', 'response.json', 'binding.json', 'plan.json', result_name):
            shutil.copyfile(directory/name, bundle/name)
        with Image.open(source) as im: size = list(im.size)
        value = {'schema': 'LayeredPlan/1', 'source_sha256': sha(source), 'image_size': size,
                 'plan': plan, 'bundle_files': {p.name: sha(p) for p in bundle.iterdir()},
                 'new_model_calls': 0, 'human_acceptance': False}
        file = target/(source.stem+'.json'); write(file, value); outputs.append(file)
    return outputs


def union(a, b):
    x, y = min(a[0], b[0]), min(a[1], b[1])
    return [x, y, max(a[0]+a[2], b[0]+b[2])-x, max(a[1]+a[3], b[1]+b[3])-y]


def intersection(a, b):
    return max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))


def guide(rgb, lines, tables, payload, source_hash, *, hard_boxes=False):
    """Bounded outward correction; uncertain regions go back to normal segmentation."""
    from .review import _fit_to_ink
    from .segment import gradient
    H, W = rgb.shape[:2]
    if payload.get('schema') != 'LayeredPlan/1' or payload['source_sha256'] != source_hash or payload['image_size'] != [W, H]:
        raise ValueError('worker plan binding mismatch')
    plan = payload['plan']; validate_plan(plan)
    rows = plan['regions']; parents = {r.get('parent_id') for r in rows}
    report = {'schema': 'LayeredPlanExecution/1', 'source_sha256': source_hash,
              'plan_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
              'regions': [], 'new_model_calls': 0}
    claimed = np.zeros((H, W), bool); grad = gradient(rgb)
    for row in rows:
        a, b, c, d = row['bbox']
        x, y = math.floor(a*W/1000), math.floor(b*H/1000)
        box = [x, y, math.ceil(c*W/1000)-x, math.ceil(d*H/1000)-y]
        item = {'id': row['id'], 'kind': row['kind'], 'action': row['action'], 'raw_bbox': box,
                'bbox': list(box), 'status': 'fallback', 'reasons': []}
        report['regions'].append(item)
        if row['id'] in parents or row['action'] == 'split_children':
            item.update(status='container', reasons=['logical_container_no_duplicate_crop']); continue
        if row['action'] != 'keep_pixels' or row['kind'] == 'text':
            item['reasons'].append('existing_measured_ocr_or_geometry'); continue
        fitted = None if hard_boxes else _fit_to_ink(grad, box)
        refined = union(box, fitted) if fitted else list(box)
        if refined != box: item['reasons'].append('outward_ink_boundary')
        for line in lines:
            lb = [int(round(v)) for v in line['bbox']]
            if intersection(refined, lb) > 0:
                grown = union(refined, lb)
                if max(abs(grown[0]-box[0]), abs(grown[1]-box[1]), abs(grown[0]+grown[2]-box[0]-box[2]), abs(grown[1]+grown[3]-box[1]-box[3])) > (0 if hard_boxes else 24):
                    item['reasons'].append('text_crosses_boundary'); break
                refined = grown
        if 'text_crosses_boundary' in item['reasons']: continue
        if refined != box: item['reasons'].append('complete_intersecting_ocr_lines')
        x, y, w, h = refined; item['bbox'] = refined
        if w*h > .45*W*H or w*h < 16 or w*h > 1.6*box[2]*box[3]:
            item['reasons'].append('bounded_region_limit'); continue
        if any(intersection(refined, t['bbox']) for t in tables):
            item['reasons'].append('measured_table_priority'); continue
        if claimed[y:y+h, x:x+w].any():
            item['reasons'].append('overlapping_plan_region'); continue
        claimed[y:y+h, x:x+w] = True; item['status'] = 'guided'
        item['reasons'].append('cleaned_pixels_with_separate_ocr_text')
    work = rgb.copy()
    colour = np.median(np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1])), axis=0)
    work[claimed] = colour.astype(np.uint8)
    report['guided_fraction'] = round(float(claimed.mean()), 5)
    return work, claimed, report


def install(elements, report, H, W):
    """One pixel owner per planned crop. Existing crossing shapes stay native."""
    from .review import _hard_rect, _next_id
    for item in report['regions']:
        if item['status'] != 'guided': continue
        x, y, w, h = item['bbox']; crop = _hard_rect(item['bbox'], H, W)
        kept = []
        for old in elements:
            overlap = old['mask'] & crop['mask']
            if not overlap.any(): kept.append(old); continue
            if old['kind'] == 'native_shape' and not np.array_equal(overlap, old['mask']):
                # Keep the designed enclosing card below the crop.
                kept.append(old); continue
            old['mask'] = old['mask'] & ~crop['mask']; old['area'] = int(old['mask'].sum())
            if old['area']:
                old['alpha'] = 'mask'; kept.append(old)
        crop.update(id=_next_id(elements), source='plan_first', plan_region_id=item['id'])
        item['element_id'] = crop['id']; kept.append(crop); elements[:] = kept
    from .segment import assign_levels
    assign_levels(elements)
    erase = np.zeros((H, W), bool)
    for e in elements: erase |= e['mask']
    return elements, erase


def unclaimed_candidates(candidates, claimed):
    """A filled planning window is not evidence of a new native rectangle.

    Discard proposals touching that synthetic surface. Original-image component
    measurements and residual recovery still handle unclaimed content.
    """
    return [(mask, meta) for mask, meta in candidates if not (mask & claimed).any()]


def overlay(source, report, path):
    image = Image.fromarray(source.copy()); draw = ImageDraw.Draw(image)
    for row in report['regions']:
        if row['action'] != 'keep_pixels': continue
        for key, color in [('raw_bbox', '#e64444'), ('bbox', '#139d55' if row['status'] == 'guided' else '#ed9b22')]:
            x, y, w, h = row[key]; draw.rectangle((x, y, x+w-1, y+h-1), outline=color, width=2)
        draw.text((row['bbox'][0], row['bbox'][1]), row['id'], fill='black', stroke_width=1, stroke_fill='white')
    image.save(path)
