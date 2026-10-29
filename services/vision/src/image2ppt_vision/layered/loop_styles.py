"""Conservative source-derived deck styles and installed-font provenance."""
from collections import Counter, defaultdict
import re
import statistics

from ..storage import digest
from . import typography


def role(style, size, median_height):
    x, y, w, h = style['ink_bbox']; W, H = size
    if y > .93*H:
        return 'footer'
    if y < .18*H and h >= 1.35*median_height:
        return 'title'
    if re.fullmatch(r'[\d.,%‰+\-万亿余人次年月日个名对\s]+', style['text']):
        return 'number'
    if style.get('bold') and h >= median_height*.95:
        return 'heading'
    return 'body'


def learn(pages):
    pools = defaultdict(list)
    for size, styles in pages:
        median = statistics.median(s['ink_bbox'][3] for s in styles.values()) if styles else 1
        for s in styles.values():
            if s.get('fitted') and s.get('shape_overlap', 0) >= .7 and not s.get('runs'):
                pools[role(s, size, median)].append({**s, 'relative_size': s['size_px']/size[1]})
    result = {}
    for name, samples in pools.items():
        family, count = Counter(s['family'] for s in samples).most_common(1)[0]
        if count < 3 or count/len(samples) < .65:
            continue
        same = [s for s in samples if s['family'] == family]
        result[name] = {'family': family, 'support': count, 'share': count/len(samples),
                        'relative_size': statistics.median(s['relative_size'] for s in same),
                        'bold_share': sum(s['bold'] for s in same)/len(same)}
    return {'schema': 'LayeredDeckStyles/1', 'roles': result,
            'basis': 'source ink fit; no original-font identity claim'}


def apply(rgb, perception, styles, learned):
    """Re-fit uncertain lines in a supported family, never blindly restyle."""
    catalog, installed = typography.layered_font_catalog()
    if not styles:
        return []
    median = statistics.median(s['ink_bbox'][3] for s in styles.values()); changes = []
    for s in styles.values():
        kind = role(s, perception['image_size'], median)
        expected = learned['roles'].get(kind)
        if not expected or s.get('runs') or s.get('shape_overlap', 0) >= .8:
            continue
        family = expected['family']
        subset = [c for c in catalog if typography.family_name(c['family']) == family]
        if not subset:
            continue
        estimate, _ = typography._estimate_drawable(rgb, {'text': s['text'], 'bbox': s['ink_bbox']}, subset, installed)
        fit = estimate.get('diagnostics', {}).get('font', {})
        if not estimate.get('native_eligible') or fit.get('dimension_error', 1) > .12:
            continue
        if fit.get('shape_overlap', 0) < max(.7, s.get('shape_overlap', 0)+.02):
            continue
        before = {k: s.get(k) for k in ('family', 'size_px', 'bold', 'origin')}
        s.update(family=typography.family_name(estimate['font_family']), size_px=float(estimate['font_size_px']),
                 bold=bool(estimate['bold']), fitted=True, shape_overlap=fit['shape_overlap'])
        glyph = fit.get('glyph_bbox')
        if glyph:
            s['origin'] = [estimate['bbox'][0]-glyph[0], estimate['bbox'][1]-glyph[1]]
        typography._width_limit(s)
        s['size_pt'] = s['size_px']*1280/perception['image_size'][0]*.75
        changes.append({'id': s['id'], 'role': kind, 'before': before,
                        'after': {k: s[k] for k in before}, 'basis': 'improved local ink fit in deck-supported font'})
    return changes


def font_inventory(styles):
    rows = []
    for family, bold in sorted({(s['family'], s['bold']) for s in styles.values()}):
        path, face = typography._font_path(family, bold)
        rows.append({'family': family, 'bold': bold, 'face_index': face,
                     'font_file_sha256': digest(path) if path else None,
                     'available_for_local_measurement': bool(path)})
    return rows
