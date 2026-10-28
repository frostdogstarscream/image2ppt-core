"""Rendered-file QA for the layered pipeline. Measurements, never acceptance."""
from collections import Counter
import re
import unicodedata

import cv2
import numpy as np
from PIL import Image

RESIDUAL_MIN_SCORE = .5
DUPLICATE_SSIM = .6
EDGE = 40.
MIN_DETAIL = 30
GHOST = .3
GHOST_CONTRAST = 40.


def _load_pair(source, preview):
    with Image.open(source) as image:
        a = np.asarray(image.convert('RGB'))
    with Image.open(preview) as image:
        b = np.asarray(image.convert('RGB').resize((a.shape[1], a.shape[0]), Image.Resampling.LANCZOS))
    return a, b


def ssim_map(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (.01*255)**2, (.03*255)**2
    blur = lambda x: cv2.GaussianBlur(x, (7, 7), 1.5)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a*a)-ma*ma, blur(b*b)-mb*mb, blur(a*b)-ma*mb
    return ((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))


def _chars(texts):
    joined = unicodedata.normalize('NFKC', ''.join(texts))
    return Counter(re.sub(r'\s+', '', joined))


def text_recall(source_lines, rendered_lines):
    expected, found = _chars(t['text'] for t in source_lines), _chars(t['text'] for t in rendered_lines)
    total = sum(expected.values())
    return round(sum((expected & found).values())/total, 4) if total else None


def residual_violations(perception, retained_ids):
    """Background OCR hits not explained by text deliberately left in the background."""
    owned = [t['bbox'] for t in perception['ocr'] if t['id'] in retained_ids]
    violations = []
    for hit in perception['residual_ocr']:
        if hit['score'] < RESIDUAL_MIN_SCORE:
            continue
        x, y, w, h = hit['bbox']
        cx, cy = x+w/2, y+h/2
        if any(a <= cx <= a+c and b <= cy <= b+d for a, b, c, d in owned):
            continue
        violations.append(hit)
    return violations


def _ghost(top, under_page, own_page, inner_page):
    """(score, contrast): 0 when the repair looks like the surroundings just
    outside the object, 1 when it looks like the object itself."""
    k = lambda n: np.ones((n, n), np.uint8)
    ring = (cv2.dilate(own_page.astype(np.uint8), k(17)) > 0) & ~(cv2.dilate(own_page.astype(np.uint8), k(5)) > 0)
    if inner_page.sum() < MIN_DETAIL or ring.sum() < MIN_DETAIL:
        return 0., 0.
    under = under_page.astype(np.float32)
    surround = np.median(under[ring], 0)
    obj, rep = top[inner_page].astype(np.float32), under[inner_page]
    d_obj, d_sur = np.abs(rep-obj).sum(1).mean(), np.abs(rep-surround).sum(1).mean()
    return float(d_sur/(d_sur+d_obj+1e-6)), float(np.abs(obj-surround).sum(1).mean())


def duplicate_under_objects(perception_dir, perception, *, threshold=DUPLICATE_SSIM):
    """Pictures still present in the layer beneath them: moving one in
    PowerPoint would reveal a copy instead of a repaired surface.

    Two signs: the object's own edges reappear (an exact copy), or the repaired
    area keeps the object's colours instead of its surroundings' (a faded copy).
    A flat object over a flat repair is similar by any measure and is not a copy,
    so edges are compared only where the object has them, and colours only when
    the object stands out from its surroundings.
    """
    found = []
    for e in perception['elements']:
        if e['kind'] != 'raster' or 'beneath' not in e:
            continue
        with Image.open(perception_dir/e['asset']) as image:
            rgba = np.asarray(image.convert('RGBA'))
        x, y, w, h = e['bbox']
        with Image.open(perception_dir/e['beneath']) as image:
            under_page = np.asarray(image.convert('RGB'))
        under = under_page[y:y+h, x:x+w]
        own = rgba[:, :, 3] > 0
        if under.shape[:2] != own.shape:
            continue
        top = np.ascontiguousarray(rgba[:, :, :3])
        gray = cv2.cvtColor(top, cv2.COLOR_RGB2GRAY).astype(np.float32)
        edge = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)) > EDGE
        inner = cv2.erode(own.astype(np.uint8), np.ones((7, 7), np.uint8), borderValue=0) > 0
        detail = edge & inner
        score = float(ssim_map(top, np.ascontiguousarray(under))[detail].mean()) if detail.sum() >= MIN_DETAIL else 0.
        page = lambda local: np.pad(local, ((y, under_page.shape[0]-y-h), (x, under_page.shape[1]-x-w)))
        top_page = np.zeros_like(under_page)
        top_page[y:y+h, x:x+w] = top
        ghost, contrast = _ghost(top_page, under_page, page(own), page(inner))
        if score >= threshold or (contrast >= GHOST_CONTRAST and ghost >= GHOST):
            found.append({'id': e['id'], 'bbox': e['bbox'], 'edge_ssim': round(score, 4),
                          'ghost': round(ghost, 3), 'contrast': round(contrast, 1)})
    return found


def lift_probe(perception_dir, perception, path):
    """The page with every picture lifted away: the repaired background plus the
    native shapes. Anything that still looks like an object here is a duplicate."""
    from .segment import _chevron, _rounded
    with Image.open(perception_dir/perception['background']['path']) as image:
        canvas = np.asarray(image.convert('RGB')).copy()
    for e in sorted((e for e in perception['elements'] if e['kind'] == 'native_shape'),
                    key=lambda e: (e.get('level', 0), -e['bbox'][2]*e['bbox'][3])):
        x, y, w, h = e['bbox']
        if e['geometry'] == 'roundRect':
            local = _rounded(w, h, e.get('corner_radius_px', 0))
        elif e['geometry'] == 'chevron':
            local = _chevron(w, h, e.get('corner_radius_px', 0) or h//4)
        elif e['geometry'] == 'ellipse':
            t = np.zeros((h, w), np.uint8)
            cv2.ellipse(t, (((w-1)/2, (h-1)/2), (float(w), float(h)), 0.), 1, -1)
            local = t > 0
        else:
            local = np.ones((h, w), bool)
        color = np.array([int(e['fill'][i:i+2], 16) for i in (0, 2, 4)], np.uint8)
        window = canvas[y:y+h, x:x+w]
        window[local[:window.shape[0], :window.shape[1]]] = color
    Image.fromarray(canvas).save(path)
    return path


def measure(source, preview, ir, *, source_ocr=None, rendered_ocr=None, directory=None):
    a, b = _load_pair(source, preview)
    full = ssim_map(a, b)
    regions = {}
    for node in ir['nodes']:
        x, y, w, h = (int(round(v)) for v in node['bbox'])
        window = full[max(0, y):y+h, max(0, x):x+w]
        if window.size:
            regions.setdefault(node['type'], []).append(float(window.mean()))
    result = {'purpose': 'comparison_only_not_acceptance',
              'mean_absolute_pixel_difference': round(float(np.abs(a.astype(float)-b.astype(float)).mean()), 4),
              'ssim': round(float(full.mean()), 4),
              'region_ssim': {k: {'count': len(v), 'median': round(float(np.median(v)), 4), 'min': round(float(np.min(v)), 4)}
                              for k, v in regions.items()}}
    if source_ocr is not None and rendered_ocr is not None:
        result['ocr_char_recall'] = text_recall(source_ocr, rendered_ocr)
    if directory is not None:
        side = np.concatenate([a, b], 1)
        Image.fromarray(side).save(directory/'comparison.png')
        Image.fromarray(np.abs(a.astype(np.int16)-b.astype(np.int16)).astype(np.uint8)).save(directory/'difference.png')
    return result
