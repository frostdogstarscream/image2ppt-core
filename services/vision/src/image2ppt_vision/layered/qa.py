"""Rendered-file QA for the layered pipeline. Measurements, never acceptance."""
from collections import Counter
import re
import unicodedata

import cv2
import numpy as np
from PIL import Image

RESIDUAL_MIN_SCORE = .5


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
