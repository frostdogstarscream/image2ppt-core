"""Evidence-backed checks on source pixels and the *delivered* PPTX.

Geometric/image heuristics are disclosed separately from source-confirmed text
and OOXML failures. No check here promotes a file to human acceptance.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..storage import digest
from .loop_review import clip, intersect
from .loop_text import normalized, compare_text


def inventory(directory):
    root = Path(directory)
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def verify_inventory(directory, expected):
    for relative, sha in expected.items():
        p = (Path(directory)/relative).resolve()
        if not p.is_relative_to(Path(directory).resolve()) or digest(p) != sha:
            raise ValueError('immutable artifact changed: '+relative)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def issue(code, bbox=None, *, target=None, severity='review', confirmed=False, **data):
    return {'code': code, 'bbox': bbox, 'target': target, 'severity': severity,
            'confirmed': confirmed, **data}


def _edge(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))


def graphic_losses(source, rendered, text_boxes=(), *, stage='final', protected=()):
    """Find localized missing source strokes, including faint maps and lines.

    Thresholds are source-resolution normalized; text boxes are excluded so
    different glyph antialiasing cannot be mistaken for a missing icon.
    """
    H, W = source.shape[:2]
    b = np.asarray(Image.fromarray(rendered).resize((W, H), Image.Resampling.LANCZOS))
    aedge, bedge = _edge(source), _edge(b)
    excluded = np.zeros((H, W), np.uint8)
    for box in text_boxes:
        x, y, w, h = clip(box, [W, H], 3); excluded[y:y+h, x:x+w] = 1
    # Protected icon boxes override OCR exclusions.
    for box in protected:
        x, y, w, h = clip(box, [W, H]); excluded[y:y+h, x:x+w] = 0
    tolerance = max(1, round(W/1000))
    nearby = cv2.dilate((bedge > 24).astype(np.uint8), np.ones((2*tolerance+1,)*2, np.uint8)) > 0
    delta = np.max(np.abs(source.astype(np.int16)-b.astype(np.int16)), axis=2)
    missing = (aedge > 24) & ~nearby & (delta > 12) & ~excluded.astype(bool)
    gap = max(5, round(W/180))
    connected = cv2.dilate(missing.astype(np.uint8), np.ones((gap, gap), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, 8)
    regions = []
    for i in range(1, count):
        component = (labels == i) & missing
        support = int(component.sum())
        if support < max(18, W*H*.000012):
            continue
        ys, xs = np.where(component)
        box = clip([int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)], [W, H], gap//2)
        x, y, w, h = box
        expected = (aedge[y:y+h, x:x+w] > 24) & ~excluded[y:y+h, x:x+w].astype(bool)
        ratio = support/max(int(expected.sum()), 1)
        if ratio < .24 or w*h > .18*W*H:
            continue
        # Heuristic evidence remains review-only until source review confirms it.
        regions.append(issue('graphic_loss', box, target=f'graphic-{len(regions):03d}', stage=stage,
                             missing_edges=support, missing_fraction=round(ratio, 4),
                             evidence='missing_source_edges_outside_text', priority=support))
    return sorted(regions, key=lambda r: -r['priority'])


def text_geometry(ir):
    issues = []; W, H = ir['canvas']['source_px']
    texts = [n for n in ir['nodes'] if n['type'] == 'text']
    for n in texts:
        x, y, w, h = n['bbox']
        if x < 0 or y < 0 or x+w > W+.01 or y+h > H+.01:
            issues.append(issue('text_outside_canvas', n['bbox'], target=n['id'], severity='blocking', confirmed=True))
    # Prefer source-ink regions to generously sized layout textboxes.
    for i, a in enumerate(texts):
        for b in texts[i+1:]:
            overlap = intersect(a['bbox'], b['bbox'])
            smaller = min(a['bbox'][2]*a['bbox'][3], b['bbox'][2]*b['bbox'][3])
            if overlap/max(1, smaller) > .3:
                issues.append(issue('text_box_overlap', a['bbox'], target=a['id'], other=b['id']))
    return issues


def final_file(ir, pptx):
    from image2ppt.quality import inspect_pptx
    inspected = inspect_pptx(pptx)
    issues = []
    if inspected['slide_count'] != 1:
        issues.append(issue('page_count_mismatch', severity='blocking', confirmed=True))
    objects = {o['name']: o for o in inspected['slides'][0]['objects']}
    for node in ir['nodes']:
        actual = objects.get(node['id'])
        if actual is None:
            issues.append(issue('missing_output_node', node['bbox'], target=node['id'], severity='blocking', confirmed=True))
        elif node['type'] == 'text' and normalized(actual.get('text') or '') != normalized(node['text']):
            issues.append(issue('output_text_changed', node['bbox'], target=node['id'], severity='blocking', confirmed=True,
                                comparison=compare_text(node['text'], actual.get('text') or '')))
    for group in ir['groups']:
        actual = objects.get(group['id'])
        children = {o['name'] for o in objects.values() if o.get('parent_group') == group['id']}
        if actual is None or children != set(group['member_ids']):
            issues.append(issue('group_members_changed', target=group['id'], severity='blocking', confirmed=True))
    return {'pptx_sha256': digest(pptx), 'issues': issues, 'slide_count': inspected['slide_count'],
            'objects': inspected['slides'][0]['objects']}


def lineage(original, effective, ir, retention):
    nodes = ir['nodes']; groups = {nid: g['id'] for g in ir['groups'] for nid in g['member_ids']}
    origins = []
    elements = original['elements']
    for node in nodes:
        related = [{'id': e['id'], 'bbox': e['bbox'], 'intersection': round(intersect(e['bbox'], node['bbox']), 3)}
                   for e in elements if intersect(e['bbox'], node['bbox']) > .5*min(e['bbox'][2]*e['bbox'][3], node['bbox'][2]*node['bbox'][3])]
        parents = [e for e in effective['elements'] if e['bbox'][2]*e['bbox'][3] > node['bbox'][2]*node['bbox'][3]*1.05
                   and intersect(e['bbox'], node['bbox']) >= node['bbox'][2]*node['bbox'][3]-.1]
        parent = min(parents, key=lambda e: e['bbox'][2]*e['bbox'][3], default=None)
        origins.append({'output_id': node['id'], 'source_regions': related, 'bbox': node['bbox'],
                        'parent_object': parent['id'] if parent else None, 'group': groups.get(node['id']),
                        'editable': 'move_resize_only' if node['type'] == 'image' else 'native_'+node['type'],
                        'evidence_ids': node['evidence_ids']})
    return {'schema': 'LayeredLineage/1', 'source_sha256': original['source_sha256'],
            'objects': origins, 'retention': retention}


def score(candidate):
    issues = candidate['issues']
    blocking = sum(i['severity'] == 'blocking' for i in issues)
    confirmed = sum(i['confirmed'] for i in issues)
    return (blocking, confirmed, candidate['region_error'], -candidate['native_count'])


def better(candidate, previous):
    """Do not trade newly confirmed content loss for a higher global score."""
    # Removing an overlap can expose a previously hidden duplicate candidate.
    # It is still counted as blocking, but is not newly missing source content.
    before = {(i['code'], i.get('target')) for i in previous['issues'] if i['confirmed'] and i['code'] != 'actual_duplicate'}
    after = {(i['code'], i.get('target')) for i in candidate['issues'] if i['confirmed'] and i['code'] != 'actual_duplicate'}
    if after-before:
        return False
    a, b = score(candidate), score(previous)
    if a[:2] != b[:2]:
        return a[:2] < b[:2]
    if abs(a[2]-b[2]) > .05:
        return a[2] < b[2]
    return a[3] < b[3]


def measured_error(source, rendered, boxes):
    H, W = source.shape[:2]
    b = np.asarray(Image.fromarray(rendered).resize((W, H), Image.Resampling.LANCZOS))
    mask = np.zeros((H, W), bool)
    for box in boxes:
        x, y, w, h = clip(box, [W, H]); mask[y:y+h, x:x+w] = True
    if not mask.any():
        mask[:] = True
    return round(float(np.abs(source.astype(float)-b.astype(float))[mask].mean()), 6)
