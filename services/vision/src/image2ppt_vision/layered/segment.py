"""Element extraction from the text-free page.

Candidates come from SAM2 automatic masks plus local-contrast connected
components (small icons SAM's point grid misses). Each candidate is accepted
top-down by area as:

* ``native_shape`` -- only when a preset geometry and a flat or linear fill
  reproduce its pixels within tolerance (the fit residual is recorded);
* ``raster`` -- an RGBA cutout (alpha from the mask) for icons, illustrations
  and photos; it absorbs every candidate nested inside it;
* nothing -- large irregular or textured areas stay in the repaired background.
"""
import math

import cv2
import numpy as np

SOLID_TOLERANCE = 7.
LINEAR_TOLERANCE = 8.
GEOMETRY_IOU = .95
RASTER_MAX_AREA = .05
RASTER_MAX_SIDE = .3


def _hex(rgb):
    return ''.join(f'{int(v):02X}' for v in np.clip(np.rint(rgb), 0, 255))


def _bbox(mask):
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]


def _iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()/union) if union else 0.


def _rounded(w, h, r):
    t = np.zeros((h, w), np.uint8)
    r = int(min(r, w//2, h//2))
    if r <= 0:
        t[:] = 1
        return t.astype(bool)
    cv2.rectangle(t, (r, 0), (w-1-r, h-1), 1, -1)
    cv2.rectangle(t, (0, r), (w-1, h-1-r), 1, -1)
    for cx, cy in ((r, r), (w-1-r, r), (r, h-1-r), (w-1-r, h-1-r)):
        cv2.circle(t, (cx, cy), r, 1, -1)
    return t.astype(bool)


def fit_geometry(local):
    """Best of rect / roundRect / ellipse for a bbox-cropped mask."""
    h, w = local.shape
    best = ('rect', 0, float(local.mean()))
    limit = min(w, h)//2
    if limit >= 2:
        for r in sorted({max(1, int(limit*f)) for f in np.linspace(.04, 1, 24)}):
            score = _iou(local, _rounded(w, h, r))
            if score > best[2]+1e-6:
                best = ('roundRect', r, score)
    e = np.zeros((h, w), np.uint8)
    cv2.ellipse(e, (((w-1)/2, (h-1)/2), (float(w), float(h)), 0.), 1, -1)
    score = _iou(local, e.astype(bool))
    if score > best[2]+.01:
        best = ('ellipse', 0, score)
    return {'geometry': best[0], 'corner_radius_px': best[1], 'iou': round(best[2], 4)}


def fit_fill(rgb, own, bbox):
    """Flat or linear-gradient fill for the visible pixels ``own`` (bool, page-sized)."""
    ys, xs = np.where(own)
    if len(xs) < 12:
        return None
    values = rgb[ys, xs].astype(np.float64)
    median = np.median(values, 0)
    flat = float(np.percentile(np.abs(values-median).max(1), 90))
    result = {'kind': 'solid', 'color': _hex(median), 'residual': round(flat, 3)}
    x, y, w, h = bbox
    if len(xs) >= 60 and max(w, h) >= 24:
        design = np.stack([(xs-x)/max(w, 1), (ys-y)/max(h, 1), np.ones(len(xs))], 1)
        coef, *_ = np.linalg.lstsq(design, values, rcond=None)
        linear = float(np.percentile(np.abs(values-design@coef).max(1), 90))
        lum = coef[:2]@np.array([.299, .587, .114])
        span = float(np.abs(coef[:2]).max())
        if linear < flat*.7 and span >= 8:
            gx, gy = lum[0]/max(w, 1), lum[1]/max(h, 1)
            if abs(gx)+abs(gy) < 1e-9:
                channel = int(np.abs(coef[:2]).max(0).argmax())
                gx, gy = coef[0, channel]/max(w, 1), coef[1, channel]/max(h, 1)
            angle = math.degrees(math.atan2(gy, gx)) % 360
            corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
            direction = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            t = corners@direction
            start, end = corners[t.argmin()], corners[t.argmax()]
            color = lambda p: np.array([p[0]/max(w, 1), p[1]/max(h, 1), 1])@coef
            result = {'kind': 'linear', 'angle_deg': round(angle, 2), 'residual': round(linear, 3),
                      'flat_residual': round(flat, 3), 'stops': [{'offset': 0, 'color': _hex(color(start))},
                                                                  {'offset': 100000, 'color': _hex(color(end))}]}
    return result


def _border(rgb, mask, fill):
    ring = mask & ~(cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0)
    if ring.sum() < 20 or fill['kind'] != 'solid':
        return None
    values = rgb[ring].astype(np.float64)
    median = np.median(values, 0)
    fill_rgb = np.array([int(fill['color'][i:i+2], 16) for i in (0, 2, 4)], float)
    spread = float(np.percentile(np.abs(values-median).max(1), 75))
    if np.abs(median-fill_rgb).max() > 22 and spread <= 14:
        return _hex(median)
    return None


def contrast_candidates(rgb, exclude, *, max_side=120, kernel=61, threshold=26.):
    """Compact high-contrast blobs relative to a heavily median-filtered page."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    smooth = cv2.cvtColor(cv2.medianBlur(rgb, kernel), cv2.COLOR_RGB2LAB).astype(np.float32)
    fg = (np.linalg.norm(lab-smooth, axis=2) > threshold) & ~exclude
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < 60 or max(w, h) > max_side or min(w, h) < 6:
            continue
        member = labels == i
        filled = np.zeros_like(member)
        contours, _ = cv2.findContours(member.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled.view(np.uint8), contours, -1, 1, -1)
        out.append((filled.astype(bool), {'source': 'contrast_component'}))
    return out


def fill_holes(mask):
    """Elements are solid regions: a mask hole is an object on top, never a window."""
    ys, xs = np.where(mask)
    if not len(xs):
        return mask
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
    local = mask[y0:y1, x0:x1].astype(np.uint8)
    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(local)
    cv2.drawContours(filled, contours, -1, 1, -1)
    out = mask.copy()
    out[y0:y1, x0:x1] |= filled > 0
    return out


def _dedupe(cands):
    kept = []
    for mask, meta in sorted(((fill_holes(m), meta) for m, meta in cands), key=lambda c: -int(c[0].sum())):
        box = _bbox(mask)
        if any(abs(box[0]-k['bbox'][0]) < 4 and abs(box[1]-k['bbox'][1]) < 4 and
               abs(box[2]-k['bbox'][2]) < 6 and abs(box[3]-k['bbox'][3]) < 6 and _iou(mask, k['mask']) > .88
               for k in kept):
            continue
        kept.append({'mask': mask, 'bbox': box, 'area': int(mask.sum()), 'meta': meta})
    return kept


def _predict(fill, xs, ys, bbox):
    if fill['kind'] == 'solid':
        return np.array([int(fill['color'][i:i+2], 16) for i in (0, 2, 4)], float)[None]
    x, y, w, h = bbox
    a = math.radians(fill['angle_deg'])
    d = np.array([math.cos(a), math.sin(a)])
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)@d
    t = (np.stack([xs-x, ys-y], 1)@d-corners.min())/max(np.ptp(corners), 1e-6)
    c0, c1 = (np.array([int(s['color'][i:i+2], 16) for i in (0, 2, 4)], float) for s in fill['stops'])
    return c0[None]+np.clip(t, 0, 1)[:, None]*(c1-c0)[None]


def residual_blobs(rgb, region, fill, bbox, *, threshold=38., min_area=25):
    """Off-fill blobs inside a native shape that no candidate described."""
    ys, xs = np.where(region)
    if not len(xs):
        return []
    deviation = np.abs(rgb[ys, xs].astype(float)-_predict(fill, xs, ys, bbox)).max(1)
    off = np.zeros(region.shape, np.uint8)
    off[ys[deviation > threshold], xs[deviation > threshold]] = 1
    off = cv2.morphologyEx(off, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(off, 8)
    return [cv2.dilate((labels == i).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            for i in range(1, n) if stats[i][4] >= min_area]


def classify(rgb, sam_masks, *, text_mask, icon_mask, extra=()):
    """Return (elements, erase_mask). ``rgb`` is the text-free page.

    A native shape is opaque above the background, so everything inside it
    must itself become an element: nested candidates are forced to raster and
    off-fill residue becomes extra cutouts.
    """
    H, W = rgb.shape[:2]
    page = H*W
    raw = [(m, meta) for m, meta in sam_masks if m.sum() >= 120 and m.sum() <= .45*page]
    raw += list(extra)
    cands = _dedupe(raw)
    taken = np.zeros((H, W), bool)
    absorbed, forced = set(), set()
    elements = []
    erase = np.zeros((H, W), bool)
    protect = cv2.dilate((icon_mask | text_mask).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    def add_raster(mask, record, children=()):
        box = _bbox(mask)
        record.update(kind='raster', bbox=box, area=int(mask.sum()), mask=mask, absorbed=len(children),
                      alpha='rect' if record.get('geometry_fit', {}).get('geometry') == 'rect'
                      and record['geometry_fit']['iou'] >= .97 else 'mask')
        record['id'] = f'el-{len(elements):03d}'
        elements.append(record)
        absorbed.update(children)
        taken[:] |= mask
        erase[:] |= cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

    for index, c in enumerate(cands):
        if index in absorbed:
            continue
        mask, (x, y, w, h) = c['mask'], c['bbox']
        if float((mask & taken).sum())/c['area'] > .15:
            continue
        children = [j for j in range(index+1, len(cands)) if j not in absorbed and
                    (cands[j]['mask'] & mask).sum() >= .9*cands[j]['area']]
        child_union = np.zeros((H, W), bool)
        for j in children:
            child_union |= cands[j]['mask']
        geometry = fit_geometry(mask[y:y+h, x:x+w])
        thin = min(w, h) <= 4 and max(w, h) >= 12
        own = mask & ~child_union & ~icon_mask & ~text_mask
        if not thin:
            own = own & (cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        fill = fit_fill(rgb, own, c['bbox'])
        record = {'bbox': [x, y, w, h], 'area': c['area'], 'source': c['meta'].get('source', 'sam2'),
                  'geometry_fit': geometry, 'fill_fit': fill}
        tol = SOLID_TOLERANCE*(2.5 if thin else 1)
        clean = fill is not None and (
            (fill['kind'] == 'solid' and fill['residual'] <= tol) or
            (fill['kind'] == 'linear' and fill['residual'] <= LINEAR_TOLERANCE))
        fits = thin or geometry['iou'] >= GEOMETRY_IOU
        rasterable = c['area'] <= RASTER_MAX_AREA*page and max(w, h) <= RASTER_MAX_SIDE*max(W, H)
        nested_ok = all(cands[j]['area'] <= RASTER_MAX_AREA*page for j in children)
        if fits and clean and nested_ok and c['meta'].get('source') != 'contrast_component':
            inner = mask & ~child_union & ~protect
            inner &= cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            blobs = [] if thin else residual_blobs(rgb, inner, fill, c['bbox'])
            record.update(kind='native_shape', geometry='rect' if thin else geometry['geometry'],
                          id=f'el-{len(elements):03d}')
            if record['geometry'] == 'roundRect':
                record['corner_radius_px'] = geometry['corner_radius_px']
            if fill['kind'] == 'solid':
                record['fill'] = fill['color']
            else:
                record['fill'] = fill['stops'][0]['color']
                record['fill_gradient'] = {'angle_deg': fill['angle_deg'], 'stops': fill['stops']}
            border = None if thin else _border(rgb, mask, fill)
            if border:
                record['line'] = {'color': border, 'width_px': 1}
            elements.append(record)
            taken |= mask & ~child_union
            erase |= cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            forced.update(children)
            for blob in blobs:
                add_raster(blob, {'source': 'shape_residual', 'parent': record['id']})
            continue
        if rasterable or index in forced:
            add_raster(mask | child_union, record, children)
    return elements, erase


def icon_fallbacks(rgb, candidates, elements):
    """Textmask icon candidates that no element covers become small RGBA cutouts."""
    H, W = rgb.shape[:2]
    covered = np.zeros((H, W), bool)
    for e in elements:
        if e['kind'] == 'raster':
            covered |= e['mask']
    out = []
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    for box in candidates:
        x, y, w, h = box
        if w < 4 or h < 4 or covered[y:y+h, x:x+w].mean() >= .5:
            continue
        p = 3
        x0, y0, x1, y1 = max(0, x-p), max(0, y-p), min(W, x+w+p), min(H, y+h+p)
        window = lab[y0:y1, x0:x1]
        ring = np.ones(window.shape[:2], bool); ring[2:-2, 2:-2] = False
        fg = np.linalg.norm(window-np.median(window[ring], 0), axis=2) > 14
        mask = np.zeros((H, W), bool)
        mask[y0:y1, x0:x1] = cv2.dilate(fg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if not mask.any():
            continue
        out.append({'id': '', 'bbox': _bbox(mask), 'area': int(mask.sum()), 'source': 'textmask_icon',
                    'kind': 'raster', 'alpha': 'mask', 'absorbed': 0, 'mask': mask})
        covered[y0:y1, x0:x1] = True
    return out


def cutout(rgb, element):
    """RGBA crop of one raster element with a lightly feathered alpha."""
    x, y, w, h = element['bbox']
    crop = rgb[y:y+h, x:x+w]
    if element['alpha'] == 'rect':
        alpha = np.full((h, w), 255, np.uint8)
    else:
        alpha = (element['mask'][y:y+h, x:x+w].astype(np.float32))
        alpha = np.clip(cv2.GaussianBlur(alpha, (3, 3), .6)*255, 0, 255).astype(np.uint8)
    return np.dstack([crop, alpha])
