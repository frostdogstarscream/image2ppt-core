"""One vision call per page for components the measurement left ambiguous.

The model sees the slide and the candidate boxes. It may demote a large shape
to a picture, or name a rough window inside a candidate where a card or a photo
was missed. A window is snapped to a contour in that crop and accepted only when
``measure`` reproduces it, or when the crop is busy enough to be a picture.
Anything else is ignored and the measured page stands.
"""
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from .components import (BUSY, CALM, CANNY, COMPONENT_IOU, DUPLICATE_IOU, LARGE, MAX_AREA, MIN_AREA, MIN_SIDE,
                         _bbox, _iou, claim_fringe, measure, smooth_under)
from .segment import assign_levels, cutout, fit_geometry, gradient, layer_erase

SCHEMA = 'LayeredReview/1'
MAX_CANDIDATES = 16
INSTRUCTION = (
    'You review measured components of one presentation slide. The image is the slide with each candidate id '
    'drawn on its box. Input json: candidates (id, kind, bbox [x,y,w,h], reason). '
    'Return json only: {"decisions":[{"id":"...","action":"keep|picture|split","windows":[[x,y,w,h]]}]}. '
    'picture: this large shape is translucent or textured and must be a picture of its own pixels. '
    'split: the candidate contains a designed card or a photo the boxes missed; each window is that piece, '
    'inside the candidate, in pixels. keep: the measurement is right. Do not move text and do not restate it.')


def review(directory, perception, *, remote=None, model=None, record_dir=None):
    """Return (perception, report). Writes back only when a decision is applied."""
    directory = Path(directory)
    report = {'schema': SCHEMA, 'model': model, 'model_used': False, 'issues': [], 'usage': {}, 'applied': []}
    mask_file = directory/'masks'/'elements.npz'
    if remote is None or not model or not mask_file.is_file():
        return perception, report
    notext = np.asarray(Image.open(directory/'notext.png').convert('RGB'))
    elements = [dict(e) for e in perception['elements']]
    _attach(mask_file, elements)
    H, W = notext.shape[:2]
    found = _candidates(elements, H, W)
    if not found:
        return perception, report
    sheet = directory/'review-sheet.png'
    _draw(np.asarray(Image.open(directory/'source.png').convert('RGB')), found, sheet)
    evidence = {'candidates': [{k: c[k] for k in ('id', 'kind', 'bbox', 'reason')} for c in found]}
    from ..auto_provider import model_body, parse_object, save_new
    body = model_body(model, INSTRUCTION, evidence, [sheet], max_tokens=4000)
    request = {'request_id': uuid4().hex, 'operation': 'model', 'timeout': 180, 'body': body}
    if record_dir:
        save_new(Path(record_dir)/'review-request.json', request)
    response = remote.request(request, timeout=240)
    if record_dir:
        save_new(Path(record_dir)/'review-response.json', response)
    report['usage'] = response.get('usage', {})
    if not response.get('ok'):
        report['issues'].append('model_call_failed:'+str(response.get('error')))
        return perception, report
    content = response.get('content', '')
    try:
        decisions = _decisions(parse_object(content))
    except (ValueError, json.JSONDecodeError):
        report['issues'].append('model_json_invalid')
        return perception, report
    report['model_used'] = True
    grad = gradient(notext)
    changed = _apply(notext, grad, elements, decisions, report)
    if not changed:
        return perception, report
    _rerender(directory, notext, elements, perception)
    perception['review'] = {'applied': report['applied'], 'issues': report['issues']}
    _write(directory/'perception.json', perception)
    return perception, report


def _candidates(elements, H, W):
    page = H*W
    found = []
    for e in elements:
        if e['area'] < LARGE*page or e.get('role') == 'icon' or e.get('source') == 'embedded_texture':
            continue
        if e['kind'] == 'native_shape' or e.get('fallback'):
            reason = 'large_shape' if e['kind'] == 'native_shape' else 'fallback_picture'
            found.append({'id': e['id'], 'kind': e['kind'], 'bbox': [int(v) for v in e['bbox']], 'reason': reason})
    found.sort(key=lambda c: -c['bbox'][2]*c['bbox'][3])
    return found[:MAX_CANDIDATES]


def _draw(rgb, found, path):
    canvas = rgb.copy()
    for c in found:
        x, y, w, h = c['bbox']
        cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), (220, 30, 30), 2)
        cv2.putText(canvas, c['id'], (x+4, min(y+h-4, y+22)), cv2.FONT_HERSHEY_SIMPLEX, .6, (180, 0, 0), 2, cv2.LINE_AA)
    Image.fromarray(canvas).save(path)


def _decisions(payload):
    items = payload.get('decisions')
    if not isinstance(items, list):
        raise ValueError('decisions')
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        action = item.get('action')
        windows = item.get('windows') or []
        if action not in ('keep', 'picture', 'split') or not isinstance(windows, list):
            continue
        out.append({'id': item.get('id'), 'action': action, 'windows': windows})
    return out


def _window(value, H, W):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        return None
    x, y, w, h = (int(round(v)) for v in value)
    x, y, w, h = max(0, x), max(0, y), min(W, w), min(H, h)
    if x+w > W or y+h > H or w < MIN_SIDE or h < MIN_SIDE:
        return None
    if not MIN_AREA <= w*h <= MAX_AREA*H*W:
        return None
    return [x, y, w, h]


def _inside(window, parent):
    x, y, w, h = window
    px, py, pw, ph = parent['bbox']
    box = parent['mask'][y:y+h, x:x+w]
    if box.size == 0 or float(box.mean()) < .5:
        return False
    return w*h < .85*(pw*ph) and px-8 <= x and py-8 <= y and x+w <= px+pw+8 and y+h <= py+ph+8


def _snap(rgb, grad, window):
    """A closed rectangle inside the window, or the window itself when it is busy."""
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
    if best:
        return best[1]
    if float(grad[y:y+h, x:x+w].mean()) >= BUSY:
        mask = np.zeros((H, W), bool)
        mask[y:y+h, x:x+w] = True
        return mask
    return None


def _accept(rgb, grad, mask, elements):
    if any(_iou(mask, e['mask']) >= DUPLICATE_IOU for e in elements):
        return None
    comp = {'mask': mask, 'bbox': _bbox(mask), 'area': int(mask.sum()), 'source': 'review'}
    H, W = rgb.shape[:2]
    empty = np.zeros((H, W), bool)
    record = measure(rgb, comp, empty, empty, grad <= CALM)
    if record:
        record.pop('icons', None)
        record['source'] = 'review'
        return record
    if float(grad[mask].mean()) >= BUSY:
        return {'kind': 'raster', 'role': 'block', 'source': 'review', 'bbox': list(comp['bbox']),
                'area': comp['area'], 'mask': mask, 'alpha': 'mask', 'members': 1, 'absorbed': 0,
                'geometry': 'outline', 'fallback': True}
    return None


def _apply(rgb, grad, elements, decisions, report):
    by_id = {e['id']: e for e in elements}
    H, W = rgb.shape[:2]
    changed = False
    next_id = 1+max((int(e['id'].split('-')[-1]) for e in elements if e['id'].startswith('el-')), default=-1)
    for item in decisions:
        host = by_id.get(item['id'])
        if host is None:
            report['issues'].append('unknown:'+str(item['id']))
            continue
        if item['action'] == 'keep':
            continue
        if item['action'] == 'picture':
            if host['kind'] != 'native_shape':
                continue
            for key in ('fill', 'fill_gradient', 'line', 'geometry_fit', 'fill_fit', 'corner_radius_px'):
                host.pop(key, None)
            host.update(kind='raster', role='block', alpha='mask', geometry='outline', fallback=True,
                        members=host.get('members', 1), absorbed=host.get('absorbed', 0))
            report['applied'].append({'id': host['id'], 'action': 'picture'})
            changed = True
            continue
        for raw in item['windows']:
            window = _window(raw, H, W)
            if window is None or not _inside(window, host):
                report['issues'].append('window:'+str(host['id']))
                continue
            mask = _snap(rgb, grad, window)
            if mask is None:
                report['issues'].append('snap:'+str(host['id']))
                continue
            record = _accept(rgb, grad, mask, elements)
            if record is None:
                report['issues'].append('reject:'+str(host['id']))
                continue
            record['id'] = f'el-{next_id:03d}'
            next_id += 1
            elements.append(record)
            by_id[record['id']] = record
            report['applied'].append({'id': record['id'], 'action': 'split', 'bbox': list(record['bbox']),
                                      'kind': record['kind'], 'inside': host['id']})
            changed = True
    if changed:
        assign_levels(elements)
        claim_fringe(rgb, elements)
    return changed


def _rerender(directory, notext, elements, perception):
    """Rewrite cutouts, layers and the background from the updated masks."""
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
            above = layer_erase(elements, k, H, W)
            layers[k] = smooth_under(notext, above) if above.any() else notext
    (directory/'layers').mkdir(exist_ok=True)
    for k, layer in layers.items():
        Image.fromarray(layer).save(directory/f'layers/level-{k}.png')
    erase = layer_erase(elements, -1, H, W) | table
    background = smooth_under(notext, erase)
    Image.fromarray(background).save(directory/'background.png')
    Image.fromarray((erase.astype(np.uint8)*255)).save(directory/'masks/erase.png')
    public = []
    for e in elements:
        if e['kind'] == 'raster':
            path = f'assets/{e["id"]}.png'
            Image.fromarray(cutout(layers[e['level']], e), 'RGBA').save(directory/path)
            e['beneath'] = f'layers/level-{e["level"]-1}.png' if e['level'] else 'background.png'
            e['asset'] = path
            e['asset_sha256'] = _sha(directory/path)
        public.append({k: v for k, v in e.items() if k != 'mask'})
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
