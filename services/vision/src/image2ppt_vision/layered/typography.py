"""Per-line typography from source pixels, calibrated per page.

Fitted lines use the installed-font raster fit (``estimate_text``); unfitted
lines borrow this page's measured font-size/ink-height ratio and origin offsets,
so nothing falls back to a fixed constant when the page itself has evidence.
"""
import statistics

import numpy as np

from ..font_contract import family_name
from ..renderer_calibration import origin_correction
from ..typography import estimate_text, font_status

DEFAULT_RATIO = 1.18
DEFAULT_OFFSET = (.02, .2)
DEFAULT_BOLD_RATIO = .125
MIN_FITS = 4


def _family(available):
    for name in ('微软雅黑', 'Microsoft YaHei'):
        if name in available:
            return name
    return available[0]


def measure(rgb, perception, *, renderer_version=None):
    W, H = perception['image_size']
    scale = 1280/W
    available = font_status()['families']
    default_family = _family(available)
    ocr = {t['id']: t for t in perception['ocr']}
    styles, fits = {}, []
    for record in perception['text']:
        if record['status'] != 'ok':
            continue
        line = ocr[record['id']]
        text = record.get('icon_text_split', {}).get('native', line['text'])
        x, y, w, h = record['ink_bbox']
        probe = {'text': text, 'bbox': [max(0, x-2), max(0, y-2), min(W-x+2, w+4), min(H-y+2, h+4)]}
        estimate = estimate_text(rgb, probe, isolate_ocr_window=True)
        fit = estimate.get('diagnostics', {}).get('font', {})
        style = {'id': record['id'], 'text': text, 'ink_bbox': record['ink_bbox'],
                 'color': record['color'], 'bold_ratio': record['bold_ratio'], 'fitted': False}
        if (estimate.get('native_eligible') and fit.get('dimension_error', 1) <= .12
                and fit.get('shape_overlap', 0) >= .58 and estimate.get('font_family') in available):
            origin = list(fit['render_origin'])
            glyph = fit.get('glyph_bbox')
            a, b = estimate['bbox'][:2]
            if glyph:
                origin = [a-glyph[0], b-glyph[1]]
            correction = origin_correction(estimate, [W, H], [1280, round(H*scale)], renderer_version=renderer_version)
            size = float(estimate['font_size_px'])
            style.update(fitted=True, size_px=size, family=estimate['font_family'], bold=bool(estimate['bold']),
                         origin=[float(origin[0]), float(origin[1])+correction['dy']],
                         color=estimate['color'].lstrip('#'), shape_overlap=fit['shape_overlap'],
                         evidence='installed-font raster fit to source ink; candidate, not identified original')
            if estimate.get('runs'):
                style['runs'] = [{'text': r['text'], 'color': r['color'].lstrip('#')} for r in estimate['runs']]
            fits.append(style)
        styles[record['id']] = style
    ratio = statistics.median(f['size_px']/f['ink_bbox'][3] for f in fits) if len(fits) >= MIN_FITS else DEFAULT_RATIO
    ox = statistics.median((f['ink_bbox'][0]-f['origin'][0])/f['size_px'] for f in fits) if len(fits) >= MIN_FITS else DEFAULT_OFFSET[0]
    oy = statistics.median((f['ink_bbox'][1]-f['origin'][1])/f['size_px'] for f in fits) if len(fits) >= MIN_FITS else DEFAULT_OFFSET[1]
    bold_threshold = _bold_threshold(fits)
    for style in styles.values():
        if style['fitted']:
            continue
        x, y, w, h = style['ink_bbox']
        size = max(6., h*ratio)
        style.update(size_px=size, family=default_family, bold=style['bold_ratio'] >= bold_threshold,
                     origin=[x-ox*size, y-oy*size],
                     evidence=f'page-calibrated fallback: size = ink height x {ratio:.3f}; bold by stroke ratio')
    width_adjusted = sum(_width_limit(style) for style in styles.values())
    snapped = snap_sizes(styles.values())
    calibration = {'fitted_lines': len(fits), 'size_over_ink_height': round(ratio, 4),
                   'origin_offset_em': [round(ox, 4), round(oy, 4)], 'bold_stroke_ratio': round(bold_threshold, 4),
                   'default_family': default_family, 'width_limited_lines': width_adjusted, 'size_clusters': snapped}
    for style in styles.values():
        style['family'] = family_name(style['family'])
        style['size_pt'] = round(style['size_px']*scale*.75, 3)
    return styles, calibration


def _font_path(family, bold):
    from ..typography import _candidates
    pool = [c for c in _candidates(None) if family_name(c['family']) == family_name(family)]
    match = [c for c in pool if c['bold'] == bold] or pool
    return (match[0]['path'], match[0].get('index', 0)) if match else (None, 0)


def _width_limit(style):
    """Shrink a line whose candidate font would overrun the source ink width."""
    from PIL import ImageFont
    path, index = _font_path(style['family'], style['bold'])
    if not path or len(style['text']) < 3:
        return 0
    font = ImageFont.truetype(path, max(1, int(round(style['size_px']*4))), index=index)
    left, _, right, _ = font.getbbox(style['text'])
    rendered = (right-left)/4
    target = style['ink_bbox'][2]
    if rendered <= target*1.02:
        return 0
    old = style['size_px']
    new = style['size_px'] = max(old*.75, old*target/rendered)
    x, y = style['origin']
    ix, iy = style['ink_bbox'][:2]
    style['origin'] = [ix-(ix-x)*new/old, iy-(iy-y)*new/old]
    style['width_limited_from_px'] = round(old, 3)
    return 1


def snap_sizes(styles, tolerance=.045):
    """Lines a few percent apart are one typographic size in the source design."""
    ordered = sorted(styles, key=lambda s: s['size_px'])
    clusters, current = [], []
    for style in ordered:
        if current and style['size_px'] > current[0]['size_px']*(1+tolerance):
            clusters.append(current); current = []
        current.append(style)
    if current:
        clusters.append(current)
    for cluster in clusters:
        fitted = [s['size_px'] for s in cluster if s['fitted']] or [s['size_px'] for s in cluster]
        size = statistics.median(fitted)
        for s in cluster:
            s['size_px'] = size
    return len(clusters)


def _bold_threshold(fits):
    bold = [f['bold_ratio'] for f in fits if f['bold']]
    regular = [f['bold_ratio'] for f in fits if not f['bold']]
    if len(bold) >= 2 and len(regular) >= 2 and statistics.median(bold) > statistics.median(regular):
        return (statistics.median(bold)+statistics.median(regular))/2
    return DEFAULT_BOLD_RATIO
