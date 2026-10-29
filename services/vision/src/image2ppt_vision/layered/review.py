"""One vision call per page. The model names hard rectangles on the slide.

The model's box only says which pieces are one object; its edges come from the
page. A box is clipped to the page, fitted to the ink mostly inside it, then
grown to enclose nearby raster pieces of similar size — a disc and its glyph,
not the parent card. A ``shape`` box becomes a native shape when ``measure``
reproduces it; otherwise it is the same hard rectangular crop as a picture.
Leftover ink the model missed (lines, dots, icons) is lifted the same way.
The slide fill is a solid colour; cutouts still come from the text-free page.
"""
import hashlib
import json
import re
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from .components import (CALM, CANNY, COMPONENT_IOU, DUPLICATE_IOU, MAX_AREA, MIN_AREA, MIN_SIDE,
                         _bbox, _iou, _share, claim_fringe, measure)
from .segment import assign_levels, cutout, fit_geometry, gradient, layer_erase

SCHEMA = 'LayeredReview/1'
SNAP_GROW = 24
SNAP_LIMIT = 2.5
SNAP_MAJORITY = .5
SNAP_TOUCH = 4
ENCLOSED = .9
ICON_MERGE = 120
MAX_OBJECTS = 24
LEFTOVER_GAP = 8
LEFTOVER_MIN = 8
INSTRUCTION = (
    'You see one presentation slide. Name the visual objects as hard rectangles in source pixels. '
    'Return json only: {"objects":[{"bbox":[x,y,w,h],"kind":"shape"|"picture"}]}. '
    'At most 24 objects; do not repeat a box. '
    'Include cards, title bars, photos and icon groups. Merge adjacent pictures that share one background '
    'into one box. Each box must enclose the whole object, including an icon\'s disc, glow or badge. '
    'Do not list text lines and do not restate any text. '
    'shape: the object is a designed rectangle or rounded rectangle. '
    'picture: the object is a photo, illustration, logo or texture and must stay a crop of its own pixels.')
OBJECT_RE = re.compile(
    r'\{\s*"bbox"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
    r'\s*,\s*"kind"\s*:\s*"(shape|picture)"\s*\}'
    r'|\{\s*"kind"\s*:\s*"(shape|picture)"\s*,\s*"bbox"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,'
    r'\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*\}')


def review(directory, perception, *, remote=None, model=None, record_dir=None):
    """Return (perception, report). Always rewrites a solid slide fill."""
    directory = Path(directory)
    report = {'schema': SCHEMA, 'model': model, 'model_used': False, 'issues': [], 'usage': {}, 'applied': []}
    mask_file = directory/'masks'/'elements.npz'
    if not mask_file.is_file():
        return perception, report
    notext = np.asarray(Image.open(directory/'notext.png').convert('RGB'))
    elements = [dict(e) for e in perception['elements']]
    _attach(mask_file, elements)
    H, W = notext.shape[:2]
    grad = gradient(notext)
    if remote is not None and model:
        from ..auto_provider import model_body, parse_object, save_new
        body = model_body(model, INSTRUCTION, {'width': W, 'height': H}, [directory/'source.png'], max_tokens=4000)
        request = {'request_id': uuid4().hex, 'operation': 'model', 'timeout': 180, 'body': body}
        if record_dir:
            save_new(Path(record_dir)/'review-request.json', request)
        response = remote.request(request, timeout=240)
        if record_dir:
            save_new(Path(record_dir)/'review-response.json', response)
        report['usage'] = response.get('usage', {})
        objects, parsed = _response_objects(response, report, parse_object)
        report['model_used'] = parsed
        _apply(notext, grad, elements, objects, report)
    _merge_icon_crumbs(elements, H, W, report)
    _lift_leftover_ink(grad, elements, H, W, report)
    assign_levels(elements)
    claim_fringe(notext, elements)
    _rerender(directory, notext, elements, perception)
    perception['review'] = {'applied': report['applied'], 'issues': report['issues']}
    perception['residual_ocr'] = []
    _write(directory/'perception.json', perception)
    return perception, report


def _response_objects(response, report, parse_object):
    content = response.get('content') or ''
    if response.get('ok'):
        try:
            return _dedupe(_objects(parse_object(content))), True
        except (ValueError, json.JSONDecodeError):
            report['issues'].append('model_json_invalid')
            found = _dedupe(_partial_objects(content))
            return found, bool(found)
    report['issues'].append('model_call_failed:'+str(response.get('error')))
    found = _dedupe(_partial_objects(response.get('partial_content') or content))
    return found, bool(found)


def _objects(payload):
    items = payload.get('objects')
    if not isinstance(items, list):
        raise ValueError('objects')
    out = []
    for item in items:
        if not isinstance(item, dict) or item.get('kind') not in ('shape', 'picture'):
            continue
        out.append({'bbox': item.get('bbox'), 'kind': item['kind']})
    return out


def _partial_objects(text):
    if not isinstance(text, str):
        return []
    out = []
    for match in OBJECT_RE.finditer(text):
        if match.group(5):
            bbox = [float(match.group(i)) for i in (1, 2, 3, 4)]
            kind = match.group(5)
        else:
            kind = match.group(6)
            bbox = [float(match.group(i)) for i in (7, 8, 9, 10)]
        out.append({'bbox': bbox, 'kind': kind})
    return out


def _dedupe(items):
    seen = set()
    out = []
    for item in items:
        window = item.get('bbox')
        if not isinstance(window, (list, tuple)) or len(window) != 4:
            continue
        key = (item['kind'], tuple(int(round(v)) for v in window))
        if key in seen:
            continue
        seen.add(key)
        out.append({'bbox': list(key[1]), 'kind': item['kind']})
        if len(out) >= MAX_OBJECTS:
            break
    return out


def _window(value, H, W):
    """Clip a box to the page. Reject only when nothing usable remains."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        return None
    x, y, w, h = (int(round(v)) for v in value)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x+w), min(H, y+h)
    w, h = x1-x0, y1-y0
    if w < MIN_SIDE or h < MIN_SIDE:
        return None
    if not MIN_AREA <= w*h <= MAX_AREA*H*W:
        return None
    return [x0, y0, w, h]


def _contour(rgb, window):
    """A closed rectangle inside the window, or None when the outline is not closed."""
    x, y, w, h = window
    H, W = rgb.shape[:2]
    crop = rgb[y:y+h, x:x+w]
    best = None
    for canny, close in ((CANNY, 3), ((4, 12), 7)):
        gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (3, 3), 0)
        edges = cv2.morphologyEx(cv2.Canny(gray, *canny), cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA:
                continue
            local = np.zeros(crop.shape[:2], np.uint8)
            cv2.drawContours(local, [contour], -1, 1, -1)
            local = cv2.erode(local, np.ones((3, 3), np.uint8)) > 0
            if int(local.sum()) < MIN_AREA:
                continue
            fit = fit_geometry(local)
            if fit['geometry'] not in ('rect', 'roundRect') or fit['iou'] < COMPONENT_IOU:
                continue
            if best is None or fit['iou'] > best[0]:
                mask = np.zeros((H, W), bool)
                mask[y:y+h, x:x+w] = local
                best = (fit['iou'], mask)
    return None if best is None else best[1]


def _measured_shape(rgb, grad, window):
    mask = _contour(rgb, window)
    if mask is None:
        return None
    comp = {'mask': mask, 'bbox': _bbox(mask), 'area': int(mask.sum()), 'source': 'review'}
    empty = np.zeros(rgb.shape[:2], bool)
    record = measure(rgb, comp, empty, empty, grad <= CALM)
    if not record:
        return None
    record.pop('icons', None)
    record['source'] = 'review'
    return record


def _hard_rect(window, H, W):
    x, y, w, h = window
    mask = np.zeros((H, W), bool)
    mask[y:y+h, x:x+w] = True
    return {'kind': 'raster', 'role': 'block', 'source': 'review', 'bbox': [x, y, w, h],
            'area': int(mask.sum()), 'mask': mask, 'alpha': 'rect', 'members': 1, 'absorbed': 0,
            'geometry': 'rect', 'fallback': True}


def _fit_to_ink(grad, window):
    """Fit the box to the ink strokes that mostly lie inside it.

    Strokes are connected ink within ``SNAP_GROW`` of the box. A stroke whose
    pixels are mostly inside belongs to the object and is kept whole, so a cut
    glyph or disc is restored; a line or neighbour that only clips the box is
    left out. None when no stroke belongs."""
    H, W = grad.shape
    x, y, w, h = window
    ex0, ey0 = max(0, x-SNAP_GROW), max(0, y-SNAP_GROW)
    ex1, ey1 = min(W, x+w+SNAP_GROW), min(H, y+h+SNAP_GROW)
    ink = (grad[ey0:ey1, ex0:ex1] > CALM).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    inside = np.bincount(labels[y-ey0:y+h-ey0, x-ex0:x+w-ex0].ravel(), minlength=count)
    keep = [k for k in range(1, count) if inside[k] >= SNAP_MAJORITY*stats[k, cv2.CC_STAT_AREA]]
    if not keep:
        return None
    sx, sy, sw, sh = (stats[keep, i] for i in range(4))
    x0, y0 = int(sx.min())+ex0, int(sy.min())+ey0
    x1, y1 = int((sx+sw).max())+ex0, int((sy+sh).max())+ey0
    if x1-x0 < MIN_SIDE or y1-y0 < MIN_SIDE:
        return None
    return [x0, y0, x1-x0, y1-y0]


def _similar(piece, window):
    """A disc or glyph next to the box, not the parent card."""
    ex, ey, ew, eh = piece['bbox'] if 'bbox' in piece else _bbox(piece['mask'])
    area = max(int(piece.get('area') or 0), ew*eh)
    return area <= SNAP_LIMIT*window[2]*window[3] and min(ew, eh) <= 2*min(window[2], window[3])


def _enclose(window, elements, scale):
    """Grow the box over similarly sized raster pieces it already touches.

    ``scale`` is the model's original box: each absorbed piece, and the grown
    result, must stay within ``SNAP_LIMIT`` of that box so chained fragments of
    a parent card cannot swallow it."""
    x0, y0, w, h = window
    x1, y1 = x0+w, y0+h
    boxes = [(e, _bbox(e['mask'])) for e in elements if e.get('kind') == 'raster' and e['mask'].any()]
    while True:
        grown = False
        for e, (ex, ey, ew, eh) in boxes:
            if ex >= x0 and ey >= y0 and ex+ew <= x1 and ey+eh <= y1:
                continue
            if ex >= x1 or ey >= y1 or ex+ew <= x0 or ey+eh <= y0:
                continue
            if int(e['mask'][y0:y1, x0:x1].sum()) < SNAP_TOUCH:
                continue
            if not _similar({'bbox': [ex, ey, ew, eh], 'area': e['area']}, scale):
                continue
            nx0, ny0 = min(x0, ex), min(y0, ey)
            nx1, ny1 = max(x1, ex+ew), max(y1, ey+eh)
            if not _similar({'bbox': [nx0, ny0, nx1-nx0, ny1-ny0], 'area': (nx1-nx0)*(ny1-ny0)}, scale):
                continue
            x0, y0, x1, y1 = nx0, ny0, nx1, ny1
            grown = True
        if not grown:
            return [x0, y0, x1-x0, y1-y0]


def _drop_enclosed(elements, box):
    """Remove every raster piece and every smaller native shape inside the crop."""
    rect = {'mask': box['mask'], 'bbox': box['bbox']}
    kept = []
    for old in elements:
        inside = _share(old, rect) >= ENCLOSED
        if inside and (old.get('kind') == 'raster' or old['area'] <= box['area']):
            continue
        kept.append(old)
    elements[:] = kept


def _drop_duplicates(elements, mask):
    """Remove an old shape or fuzzy picture that the new native shape repeats."""
    elements[:] = [old for old in elements
                   if not ((old.get('kind') == 'native_shape' or old.get('fallback'))
                           and _iou(old['mask'], mask) >= DUPLICATE_IOU)]


def _picture(grad, window, elements, H, W, report):
    fitted = _fit_to_ink(grad, window)
    seed = fitted or window
    box = _enclose(seed, elements, window)
    if box[2]*box[3] > MAX_AREA*H*W:
        report['issues'].append('snap_limit')
        return None
    if fitted is None and box == window and float(grad[box[1]:box[1]+box[3], box[0]:box[0]+box[2]].mean()) < CALM:
        report['issues'].append('snap_empty')
        return None
    record = _hard_rect(box, H, W)
    _drop_enclosed(elements, record)
    return record


def _punch(elements, box):
    """Take a smaller crop out of a larger picture that already contains it."""
    x, y, w, h = box['bbox']
    for old in elements:
        if old is box or old.get('kind') != 'raster' or old['area'] <= box['area']:
            continue
        if _share(box, old) < ENCLOSED:
            continue
        old['mask'][y:y+h, x:x+w] = False
        old['alpha'] = 'mask'
        old['area'] = int(old['mask'].sum())
        if 'bbox' in old and old['mask'].any():
            old['bbox'] = list(_bbox(old['mask']))


def _apply(rgb, grad, elements, objects, report):
    H, W = rgb.shape[:2]
    changed = False
    for item in objects:
        window = _window(item['bbox'], H, W)
        if window is None:
            report['issues'].append('window')
            continue
        record = None
        if item['kind'] == 'shape':
            record = _measured_shape(rgb, grad, _fit_to_ink(grad, window) or window)
            if record is not None:
                _drop_duplicates(elements, record['mask'])
            else:
                report['issues'].append('shape_unmeasured')
                record = _picture(grad, window, elements, H, W, report)
        else:
            record = _picture(grad, window, elements, H, W, report)
        if record is None:
            continue
        record['id'] = _next_id(elements)
        if record.get('kind') == 'raster':
            _punch(elements, record)
        elements.append(record)
        report['applied'].append({'id': record['id'], 'kind': record['kind'], 'bbox': list(record['bbox']),
                                   'alpha': record.get('alpha', '')})
        changed = True
    return changed


def _is_crumb(element):
    return element.get('kind') == 'raster' and (element.get('role') == 'icon' or element.get('source') == 'shape_residual')


def _touch(a, b):
    if a['bbox'][0] >= b['bbox'][0]+b['bbox'][2]+2 or b['bbox'][0] >= a['bbox'][0]+a['bbox'][2]+2:
        return False
    if a['bbox'][1] >= b['bbox'][1]+b['bbox'][3]+2 or b['bbox'][1] >= a['bbox'][1]+a['bbox'][3]+2:
        return False
    pad = np.ones((3, 3), np.uint8)
    grown = cv2.dilate(a['mask'].astype(np.uint8), pad) > 0
    return int((grown & b['mask']).sum()) >= SNAP_TOUCH


def _merge_icon_crumbs(elements, H, W, report):
    crumbs = [e for e in elements if _is_crumb(e)]
    if len(crumbs) < 2:
        return False
    parent = list(range(len(crumbs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(crumbs):
        for j, b in enumerate(crumbs[i+1:], i+1):
            if _touch(a, b):
                parent[find(j)] = find(i)
    groups = {}
    for i, e in enumerate(crumbs):
        groups.setdefault(find(i), []).append(e)
    changed = False
    for group in groups.values():
        if len(group) < 2:
            continue
        xs = [e['bbox'][0] for e in group]
        ys = [e['bbox'][1] for e in group]
        x1 = [e['bbox'][0]+e['bbox'][2] for e in group]
        y1 = [e['bbox'][1]+e['bbox'][3] for e in group]
        box = [min(xs), min(ys), max(x1)-min(xs), max(y1)-min(ys)]
        if min(box[2], box[3]) > ICON_MERGE or box[2]*box[3] > MAX_AREA*H*W:
            continue
        record = _hard_rect(box, H, W)
        drop = {id(e) for e in group}
        elements[:] = [e for e in elements if id(e) not in drop]
        _punch(elements, record)
        record['id'] = _next_id(elements)
        elements.append(record)
        report['applied'].append({'id': record['id'], 'kind': 'raster', 'bbox': list(record['bbox']),
                                   'alpha': 'rect'})
        changed = True
    return changed


def _leftover_boxes(local, H, W):
    """Tight boxes for leftover ink. A card-sized group is split back to strokes."""
    box = list(_bbox(local))
    if box[2]*box[3] > MAX_AREA*H*W:
        return []
    if min(box[2], box[3]) <= ICON_MERGE:
        return [box]
    count, labels, _, _ = cv2.connectedComponentsWithStats(local.astype(np.uint8), 8)
    out = []
    for i in range(1, count):
        piece = labels == i
        if int(piece.sum()) < LEFTOVER_MIN:
            continue
        pb = list(_bbox(piece))
        if pb[2]*pb[3] > MAX_AREA*H*W:
            continue
        out.append(pb)
    return out


def _lift_leftover_ink(grad, elements, H, W, report):
    """Turn unnamed strokes into hard crops so they do not stay on the fill."""
    claimed = np.zeros((H, W), bool)
    for e in elements:
        mask = e.get('mask')
        if mask is not None and mask.any():
            claimed |= mask
    claimed = cv2.dilate(claimed.astype(np.uint8), np.ones((LEFTOVER_GAP, LEFTOVER_GAP), np.uint8)) > 0
    ink = ((grad > CALM) & ~claimed).astype(np.uint8)
    if not ink.any():
        return False
    grouped = cv2.dilate(ink, np.ones((LEFTOVER_GAP, LEFTOVER_GAP), np.uint8))
    count, labels, _, _ = cv2.connectedComponentsWithStats(grouped, 8)
    changed = False
    for i in range(1, count):
        local = (labels == i) & (ink > 0)
        if int(local.sum()) < LEFTOVER_MIN:
            continue
        for box in _leftover_boxes(local, H, W):
            if max(box[2], box[3]) < 4:
                continue
            record = _hard_rect(box, H, W)
            _punch(elements, record)
            record['id'] = _next_id(elements)
            elements.append(record)
            report['applied'].append({'id': record['id'], 'kind': 'raster', 'bbox': list(record['bbox']),
                                       'alpha': 'rect'})
            changed = True
    return changed


def _board_color(notext, claimed, grad):
    """Page colour from calm pixels the objects do not cover, else the margin."""
    calm = (grad <= CALM) & ~claimed
    if int(calm.sum()) < 500:
        ring = np.ones(claimed.shape, bool)
        ring[6:-6, 6:-6] = False
        calm = ring
    colour = np.clip(np.round(np.median(notext[calm], 0)), 0, 255).astype(np.uint8)
    return colour, ''.join(f'{int(v):02X}' for v in colour)


def _next_id(elements):
    return 'el-{:03d}'.format(1+max((int(e['id'].split('-')[-1]) for e in elements
                                     if str(e.get('id', '')).startswith('el-')), default=-1))


def _rerender(directory, notext, elements, perception):
    """Rewrite cutouts from the text-free page and a solid slide fill."""
    H, W = notext.shape[:2]
    table = np.zeros((H, W), bool)
    for item in perception.get('tables', []):
        x, y, w, h = (int(v) for v in item['bbox'])
        table[max(0, y-3):y+h+3, max(0, x-3):x+w+3] = True
    top = max((e['level'] for e in elements), default=0)
    raster_levels = {e['level'] for e in elements if e['kind'] == 'raster'}
    layers = {}
    for k in range(top+1):
        if k in raster_levels or k+1 in raster_levels:
            layers[k] = notext
    (directory/'layers').mkdir(exist_ok=True)
    for k, layer in layers.items():
        Image.fromarray(layer).save(directory/f'layers/level-{k}.png')
    erase = layer_erase(elements, -1, H, W) | table
    colour, solid = _board_color(notext, erase, gradient(notext))
    background = np.broadcast_to(colour, notext.shape).copy()
    Image.fromarray(background).save(directory/'background.png')
    Image.fromarray((erase.astype(np.uint8)*255)).save(directory/'masks/erase.png')
    perception['background']['solid_color'] = solid
    perception['background']['element_method'] = 'solid_whiteboard'
    public = []
    for e in elements:
        if e['kind'] == 'raster':
            path = f'assets/{e["id"]}.png'
            Image.fromarray(cutout(layers[e['level']], e), 'RGBA').save(directory/path)
            e['beneath'] = f'layers/level-{e["level"]-1}.png' if e['level'] else 'background.png'
            e['asset'] = path
            e['asset_sha256'] = _sha(directory/path)
        public.append({k: v for k, v in e.items() if k != 'mask'})
    referenced = {e.get('asset') for e in elements if e.get('asset')}
    assets = directory/'assets'
    if assets.is_dir():
        for path in assets.glob('*.png'):
            if path.relative_to(directory).as_posix() not in referenced:
                path.unlink()
    ids = [e['id'] for e in elements]
    np.savez_compressed(directory/'masks/elements.npz',
                        masks=np.stack([e['mask'] for e in elements]), ids=np.array(ids))
    perception['elements'] = public
    counts = perception['counts']
    counts.update(native_shapes=sum(e['kind'] == 'native_shape' for e in public),
                  rasters=sum(e['kind'] == 'raster' for e in public),
                  blocks=sum(e.get('role') == 'block' for e in public),
                  icons=sum(e.get('role') == 'icon' for e in public),
                  levels=top+1,
                  native_share=round(sum(e['kind'] == 'native_shape' for e in public)/max(len(public), 1), 3),
                  erased_fraction=round(float(erase.mean()), 4))
    perception['background']['sha256'] = _sha(directory/'background.png')
    perception['files'] = {p.relative_to(directory).as_posix(): _sha(p)
                           for p in sorted(directory.rglob('*')) if p.is_file()}


def _attach(path, elements):
    data = np.load(path, allow_pickle=False)
    masks = {str(i): m for i, m in zip(data['ids'], data['masks'])}
    missing = [e['id'] for e in elements if e['id'] not in masks]
    if missing:
        raise ValueError('element masks missing: '+','.join(missing))
    for e in elements:
        e['mask'] = masks[e['id']].astype(bool)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')
