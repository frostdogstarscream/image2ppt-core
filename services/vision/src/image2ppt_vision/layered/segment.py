"""Element extraction from the text-free page.

Candidates come from SAM2 automatic masks plus local-contrast connected
components (small icons SAM's point grid misses). Each candidate is accepted
top-down by area as:

* ``native_shape`` -- only when a preset geometry and a flat or linear fill
  reproduce its pixels within tolerance (the fit residual is recorded);
* ``raster`` -- an RGBA cutout (alpha from the mask) for icons, illustrations
  and photos; it absorbs every candidate nested inside it;
* nothing -- large irregular or textured areas stay in the repaired background.

``blockify`` then regroups the rasters: touching texture rasters become one
rectangular ``block`` whose alpha is a rect or roundRect, and small or
detector-found rasters stay separate ``icon`` cutouts above the blocks.
"""
import math

import cv2
import numpy as np

SOLID_TOLERANCE = 7.
LINEAR_TOLERANCE = 8.
GEOMETRY_IOU = .95
FLAT_IOU = .84
CHEVRON_IOU = .86
RASTER_MAX_AREA = .05
RASTER_MAX_SIDE = .3
SMALL_ON_BACKGROUND = 100
ICON_MAX = .05
ICON_SOURCES = ('textmask_icon', 'contrast_component', 'shape_residual')
BLOCK_GAP = 6
BLOCK_MAX_AREA = .45
BLOCK_MIN_DENSITY = .6
BLOCK_FILL = .7
BLOCK_NESTED = .9


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
    if w >= 16 and h >= 8:
        for frac in (.2, .28, .36, .45):
            notch = max(2, int(round(min(w, h)*frac)))
            score = _iou(local, _chevron(w, h, notch))
            if score > best[2]+.01:
                best = ('chevron', notch, score)
    return {'geometry': best[0], 'corner_radius_px': best[1], 'iou': round(best[2], 4)}


def _chevron(w, h, notch):
    """Right-pointing ribbon: a point on the right and a matching notch on the left."""
    notch = int(max(1, min(notch, w//2-1, h//2)))
    pts = np.array([[notch, 0], [w-1-notch, 0], [w-1, h//2], [w-1-notch, h-1],
                    [notch, h-1], [0, h//2]], np.int32)
    canvas = np.zeros((h, w), np.uint8)
    cv2.fillPoly(canvas, [pts], 1)
    return canvas.astype(bool)


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


def flat_region_candidates(rgb, *, grad_max=16., min_side=18, min_area=450):
    """Filled cards, ribbons and bars whose interior is flat enough to be a preset shape.

    SAM often returns the photo or the icon instead of the card. Low-gradient
    connected regions, with holes closed, are the card silhouettes.
    """
    blur = cv2.GaussianBlur(rgb, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt((gx**2).sum(2)+(gy**2).sum(2))
    edge = cv2.dilate((mag > grad_max).astype(np.uint8), np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((edge == 0).astype(np.uint8), 4)
    H, W = rgb.shape[:2]
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area or area > .28*H*W or min(w, h) < min_side:
            continue
        if w > .95*W and h > .5*H:
            continue
        mask = fill_holes(labels == i)
        x0, y0, bw, bh = _bbox(mask)
        geo = fit_geometry(mask[y0:y0+bh, x0:x0+bw])
        floor = CHEVRON_IOU if geo['geometry'] == 'chevron' else FLAT_IOU
        if geo['iou'] < floor:
            continue
        out.append((mask, {'source': 'flat_region'}))
    return out


def circle_candidates(rgb, *, min_radius=12, max_radius=36, residual_max=16.):
    """Solid icon discs. The glyph in the middle stays a cutout; the disc is an ellipse."""
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=26,
                             param1=90, param2=24, minRadius=min_radius, maxRadius=max_radius)
    if found is None:
        return []
    out = []
    for x, y, r in found[0]:
        x, y, r = float(x), float(y), float(r)
        if r+2 >= x or r+2 >= y or x+r+2 >= W or y+r+2 >= H:
            continue
        yy, xx = np.ogrid[:H, :W]
        dist = np.sqrt((xx-x)**2+(yy-y)**2)
        ring = (dist <= r) & (dist >= r*.62)
        if int(ring.sum()) < 30:
            continue
        values = rgb[ring].astype(np.float64)
        residual = float(np.percentile(np.abs(values-np.median(values, 0)).max(1), 80))
        if residual > residual_max:
            continue
        out.append((dist <= r, {'source': 'circle'}))
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


def merge_fragments(rgb, cands, *, max_side=48, gap=4, color_tol=18.):
    """Join tiny same-colour masks whose boxes nearly touch. A card is not a fragment."""
    small, large = [], []
    for cand in cands:
        if max(cand['bbox'][2:]) <= max_side:
            small.append({**cand, 'color': np.median(rgb[cand['mask']], 0) if cand['area'] else np.zeros(3)})
        else:
            large.append(cand)
    parent = list(range(len(small)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def near(a, b):
        ax, ay, aw, ah = a['bbox']
        bx, by, bw, bh = b['bbox']
        dx = max(0, max(ax, bx)-min(ax+aw, bx+bw))
        dy = max(0, max(ay, by)-min(ay+ah, by+bh))
        return dx <= gap and dy <= gap and np.abs(a['color']-b['color']).max() <= color_tol

    for i in range(len(small)):
        for j in range(i+1, len(small)):
            if near(small[i], small[j]):
                parent[find(j)] = find(i)
    groups = {}
    for i, cand in enumerate(small):
        groups.setdefault(find(i), []).append(cand)
    merged = list(large)
    for group in groups.values():
        if len(group) == 1:
            merged.append({key: group[0][key] for key in ('mask', 'bbox', 'area', 'meta')})
            continue
        mask = np.zeros_like(group[0]['mask'])
        for cand in group:
            mask |= cand['mask']
        merged.append({'mask': mask, 'bbox': _bbox(mask), 'area': int(mask.sum()),
                       'meta': {'source': 'merged_fragment'}})
    merged.sort(key=lambda cand: -cand['area'])
    return merged


def _without_icons(cands, icon_mask):
    """SAM must not slice a text-adjacent icon; icon_fallbacks owns that pixel."""
    if icon_mask is None or not icon_mask.any():
        return cands
    kept = []
    for cand in cands:
        overlap = int((cand['mask'] & icon_mask).sum())
        if overlap >= .5*cand['area']:
            continue
        mask = cand['mask'] & ~icon_mask
        if int(mask.sum()) < 120:
            continue
        kept.append({**cand, 'mask': mask, 'bbox': _bbox(mask), 'area': int(mask.sum())})
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


def residual_blobs(rgb, region, fill, bbox, *, threshold=38., min_area=25, badge=10., badge_growth=4.):
    """Off-fill blobs inside a native shape that no candidate described.

    A blob grows into the faint badge around it (a pale disc behind a glyph), so
    the icon is lifted whole instead of leaving its badge under the shape fill.
    """
    ys, xs = np.where(region)
    if not len(xs):
        return []
    deviation = np.abs(rgb[ys, xs].astype(float)-_predict(fill, xs, ys, bbox)).max(1)
    off = np.zeros(region.shape, np.uint8)
    off[ys[deviation > threshold], xs[deviation > threshold]] = 1
    off = cv2.morphologyEx(off, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    soft = np.zeros(region.shape, np.uint8)
    soft[ys[deviation > badge], xs[deviation > badge]] = 1
    H, W = region.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(off, 8)
    out = []
    for i in range(1, n):
        if stats[i][4] < min_area:
            continue
        blob = labels == i
        if any(float((blob & m).sum()) >= .8*stats[i][4] for m in out):
            continue
        bx, by, bw, bh = stats[i][:4]
        # Badges in a column often touch; look only near this glyph.
        m = int(round(.5*max(bw, bh)))
        x0, y0, x1, y1 = max(0, bx-m), max(0, by-m), min(W, bx+bw+m), min(H, by+bh+m)
        local = soft[y0:y1, x0:x1] | blob[y0:y1, x0:x1].astype(np.uint8)
        local = cv2.morphologyEx(local, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) | blob[y0:y1, x0:x1].astype(np.uint8)
        _, local_labels = cv2.connectedComponents(local, connectivity=8)
        ids = np.unique(local_labels[blob[y0:y1, x0:x1]])
        ids = ids[ids > 0]
        grown = np.zeros_like(blob)
        grown[y0:y1, x0:x1] = np.isin(local_labels, ids)
        grown = fill_holes(grown | blob)
        gx, gy, gw, gh = _bbox(grown)
        touches = gx <= x0 or gy <= y0 or gx+gw >= x1 or gy+gh >= y1
        if not touches and gw*gh <= badge_growth*bw*bh:
            blob = grown
        out.append(cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    return out


def _iou_floor(geometry, source):
    floor = FLAT_IOU if source == 'flat_region' else GEOMETRY_IOU
    if geometry == 'chevron':
        floor = min(floor, CHEVRON_IOU)
    return floor


def classify(rgb, sam_masks, *, text_mask, icon_mask, extra=()):
    """Return (elements, erase_mask). ``rgb`` is the text-free page.

    Preset-shaped flat regions become native shapes first. A nested region that
    is itself a preset stays a shape; other nested pieces are not separate
    rasters. Off-fill residue inside a shape becomes a cutout. Tiny fragments
    that still sit on the untouched background stay in that background.
    """
    H, W = rgb.shape[:2]
    page = H*W
    raw = [(m, meta) for m, meta in sam_masks if m.sum() >= 120 and m.sum() <= .45*page]
    raw += list(extra)
    cands = _without_icons(merge_fragments(rgb, _dedupe(raw)), icon_mask)
    taken = np.zeros((H, W), bool)
    absorbed = set()
    elements = []
    erase = np.zeros((H, W), bool)
    protect = cv2.dilate((icon_mask | text_mask).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    blur = cv2.GaussianBlur(rgb, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    calm = np.sqrt((gx**2).sum(2)+(gy**2).sum(2)) <= 18

    def add_raster(mask, record, children=()):
        if int(mask.sum()) < 25:
            return
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
        mask, (x, y, w, h) = c['mask'], c['bbox']
        children = [j for j in range(index+1, len(cands))
                    if (cands[j]['mask'] & mask).sum() >= .9*cands[j]['area']]
        child_union = np.zeros((H, W), bool)
        for j in children:
            child_union |= cands[j]['mask']
        geometry = fit_geometry(mask[y:y+h, x:x+w])
        thin = min(w, h) <= 4 and max(w, h) >= 12
        visible = mask & ~child_union & ~icon_mask & ~text_mask
        source = c['meta'].get('source', 'sam2')
        if source == 'circle':
            radius = max(3, min(w, h)//5)
            kernel = np.ones((radius, radius), np.uint8)
            visible = visible & ~(cv2.erode(mask.astype(np.uint8), kernel) > 0)
        own = visible & calm
        if not thin:
            eroded = own & (cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
            if int(eroded.sum()) >= 12:
                own = eroded
        if int(own.sum()) < 12:
            own = visible
        fill = fit_fill(rgb, own, c['bbox'])
        source = c['meta'].get('source', 'sam2')
        tol = SOLID_TOLERANCE*(2.5 if thin else 1)
        clean = fill is not None and (
            (fill['kind'] == 'solid' and fill['residual'] <= tol) or
            (fill['kind'] == 'linear' and fill['residual'] <= LINEAR_TOLERANCE))
        fits = thin or geometry['iou'] >= _iou_floor(geometry['geometry'], source)
        if c['area'] > .03*page or w > .75*W:
            # A large silhouette can look flat once icons are ignored and still
            # paint over a photo. Require the whole region to be one fill.
            eroded = mask & (cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
            full = fit_fill(rgb, eroded if int(eroded.sum()) >= 12 else mask, c['bbox'])
            limit = 8. if w > .75*W else 25.
            if full is None or full['residual'] > limit:
                fits = False
        nested_ok = all(cands[j]['area'] <= RASTER_MAX_AREA*page for j in children)
        c.update(children=children, child_union=child_union, geometry_fit=geometry, fill_fit=fill,
                 thin=thin, source=source,
                 promotable=bool(fits and clean and nested_ok and source != 'contrast_component'))

    def emit_shape(c):
        mask, (x, y, w, h) = c['mask'], c['bbox']
        geometry, fill = c['geometry_fit'], c['fill_fit']
        shape_children = [j for j in c['children'] if cands[j]['promotable'] and j not in absorbed]
        other_children = [j for j in c['children'] if j not in shape_children]
        shape_union = np.zeros((H, W), bool)
        for j in shape_children:
            shape_union |= cands[j]['mask']
        inner = mask & ~protect & ~shape_union
        inner &= cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        record = {'bbox': [x, y, w, h], 'area': c['area'], 'source': c['source'],
                  'geometry_fit': geometry, 'fill_fit': fill,
                  'kind': 'native_shape', 'geometry': 'rect' if c['thin'] else geometry['geometry'],
                  'id': f'el-{len(elements):03d}'}
        if record['geometry'] == 'roundRect':
            record['corner_radius_px'] = geometry['corner_radius_px']
        if fill['kind'] == 'solid':
            record['fill'] = fill['color']
        else:
            record['fill'] = fill['stops'][0]['color']
            record['fill_gradient'] = {'angle_deg': fill['angle_deg'], 'stops': fill['stops']}
        border = None if c['thin'] else _border(rgb, mask, fill)
        if border:
            record['line'] = {'color': border, 'width_px': 1}
        record['mask'] = mask
        elements.append(record)
        taken[:] |= mask & ~shape_union
        erase[:] |= mask
        absorbed.update(other_children)
        if c['thin']:
            return
        for blob in residual_blobs(rgb, inner, fill, c['bbox']):
            add_raster(blob, {'source': 'shape_residual', 'parent': record['id']})

    for index, c in enumerate(cands):
        if not c['promotable'] or index in absorbed:
            continue
        if float((c['mask'] & taken).sum())/max(c['area'], 1) > .8:
            continue
        emit_shape(c)
    for index, c in enumerate(cands):
        if not c['promotable'] and index not in absorbed:
            mask, (x, y, w, h) = c['mask'], c['bbox']
            if float((mask & taken).sum())/max(c['area'], 1) > .15:
                continue
            if max(w, h) <= SMALL_ON_BACKGROUND and float((mask & erase).sum()) < .35*c['area']:
                continue
            rasterable = c['area'] <= RASTER_MAX_AREA*page and max(w, h) <= RASTER_MAX_SIDE*max(W, H)
            # A large compact panel (a photo, a textured banner) becomes one block
            # later; a large sparse mask is page background and stays there.
            panel = c['area'] <= BLOCK_MAX_AREA*page and c['area'] >= BLOCK_MIN_DENSITY*w*h
            if not (rasterable or panel):
                continue
            record = {'bbox': [x, y, w, h], 'area': c['area'], 'source': c['source'],
                      'geometry_fit': c['geometry_fit'], 'fill_fit': c['fill_fit']}
            add_raster(mask | c['child_union'], record, c['children'])
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


def split_icons(elements, W):
    """(icons, texture) among raster elements. Icons are small, or come from an icon detector."""
    icons, texture = [], []
    for e in elements:
        if e['kind'] != 'raster':
            continue
        small = max(e['bbox'][2:]) <= ICON_MAX*W
        (icons if small or e.get('source') == 'textmask_icon' else texture).append(e)
    return icons, texture


def _near(a, b, gap):
    return (a[0] <= b[0]+b[2]+gap and b[0] <= a[0]+a[2]+gap and
            a[1] <= b[1]+b[3]+gap and b[1] <= a[1]+a[3]+gap)


def _union(a, b):
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0]+a[2], b[0]+b[2]), max(a[1]+a[3], b[1]+b[3])
    return [x0, y0, x1-x0, y1-y0]


def merge_blocks(members, W, H, *, gap=BLOCK_GAP, max_area=BLOCK_MAX_AREA, fill=BLOCK_FILL):
    """Group texture rasters whose boxes touch into rectangles: [(box, members)].

    Two groups merge if their pixels fill ``fill`` of the merged rectangle, if half
    of the smaller one's pixels are also the larger one's, or if the smaller box
    lies almost entirely inside the larger (sky and buildings of one photo). A wide
    band touching a row of separate cards does none of these.
    """
    groups = [(list(m['bbox']), [m], m['mask'].copy()) for m in sorted(members, key=lambda m: -m['area'])]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                if not _near(groups[i][0], groups[j][0], gap):
                    continue
                box = _union(groups[i][0], groups[j][0])
                if box[2]*box[3] > max_area*W*H:
                    continue
                x, y, w, h = box
                ma, mb = groups[i][2][y:y+h, x:x+w], groups[j][2][y:y+h, x:x+w]
                union = groups[i][2] | groups[j][2]
                a, b = groups[i][0], groups[j][0]
                small, large = (a, b) if a[2]*a[3] <= b[2]*b[3] else (b, a)
                overlapping = (float((ma & mb).sum()) >= .5*min(int(ma.sum()), int(mb.sum()))
                               or _contains(large, small, BLOCK_NESTED))
                if not overlapping and float((ma | mb).sum()) < fill*w*h:
                    continue
                groups[i] = (box, groups[i][1]+groups[j][1], union)
                del groups[j]
                merged = True
                break
            if merged:
                break
    return [(box, items) for box, items, _ in groups]


def _contains(outer, box, share=.9):
    x0, y0 = max(outer[0], box[0]), max(outer[1], box[1])
    x1, y1 = min(outer[0]+outer[2], box[0]+box[2]), min(outer[1]+outer[3], box[1]+box[3])
    inter = max(0, x1-x0)*max(0, y1-y0)
    return inter >= share*box[2]*box[3]


def blockify(elements, W, H):
    """Texture rasters become rectangular blocks; icons stay separate cutouts.

    A block's alpha is its fitted rect or roundRect, never the texture outline. A
    native shape or a non-detector icon lying inside a block is part of its pixels.
    Returns (elements, erase). Erase covers kept shapes, the pieces each block was
    built from (not its whole rectangle, which may clip something left on the
    background) and dilated icons.
    """
    icons, texture = split_icons(elements, W)
    blocks = []
    for box, items in merge_blocks(texture, W, H):
        x, y, w, h = box
        union = np.zeros((H, W), bool)
        for item in items:
            union |= item['mask']
        fit = fit_geometry(fill_holes(union[y:y+h, x:x+w]))
        mask = np.zeros((H, W), bool)
        record = {'kind': 'raster', 'role': 'block', 'source': 'block', 'bbox': box, 'area': int(w*h),
                  'members': len(items), 'absorbed': sum(i.get('absorbed', 0) for i in items)+len(items)-1}
        if fit['geometry'] == 'roundRect' and fit['iou'] >= .9:
            mask[y:y+h, x:x+w] = _rounded(w, h, fit['corner_radius_px'])
            record.update(geometry='roundRect', corner_radius_px=fit['corner_radius_px'], alpha='mask')
        else:
            mask[y:y+h, x:x+w] = True
            record.update(geometry='rect', alpha='rect')
        record['mask'] = mask
        record['pixels'] = union
        blocks.append(record)
    inside = lambda e: any(_contains(b['bbox'], e['bbox']) for b in blocks)
    kept_icons = []
    for e in icons:
        if inside(e) and e.get('source') not in ICON_SOURCES:
            continue
        e['role'] = 'icon'
        kept_icons.append(e)
    shapes = [e for e in elements if e['kind'] == 'native_shape' and not inside(e)]
    out = shapes+blocks+kept_icons
    for k, e in enumerate(out):
        e['id'] = f'el-{k:03d}'
    erase = np.zeros((H, W), bool)
    for e in shapes:
        erase |= e['mask']
    for e in blocks:
        erase |= (cv2.dilate(e.pop('pixels').astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & e['mask']
    for e in kept_icons:
        erase |= cv2.dilate(e['mask'].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return out, erase


def icon_erase(elements):
    """Dilated icon masks that fall inside a block: the block is cropped with them repaired."""
    blocks = [e['mask'] for e in elements if e.get('role') == 'block']
    if not blocks:
        return None
    within = np.logical_or.reduce(blocks)
    erase = np.zeros_like(within)
    for e in elements:
        if e.get('role') == 'icon':
            erase |= cv2.dilate(e['mask'].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    erase &= within
    return erase if erase.any() else None


def harden(crop, mask):
    """Hard edge. The box is not moved, and pixels outside the mask stay transparent.

    On a filled picture the outer pixel still holds the card color, so it is replaced
    by the next pixel in. An irregular cutout only drops partial alpha.
    """
    mask = np.asarray(mask, dtype=bool)
    rgb = np.array(crop[:, :, :3], copy=True)
    if mask.size and float(mask.mean()) >= .9 and min(mask.shape) > 4:
        rgb[0, :] = rgb[1, :]
        rgb[-1, :] = rgb[-2, :]
        rgb[:, 0] = rgb[:, 1]
        rgb[:, -1] = rgb[:, -2]
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha]), 0, 0


def cutout(rgb, element):
    """RGBA crop. Shrinks ``element['bbox']`` to the pixels actually kept."""
    x, y, w, h = (int(v) for v in element['bbox'])
    crop = rgb[y:y+h, x:x+w]
    if element.get('alpha') == 'rect':
        local = np.ones((h, w), bool)
    else:
        local = element['mask'][y:y+h, x:x+w].astype(bool)
    rgba, dx, dy = harden(crop, local)
    element['bbox'] = [x+dx, y+dy, int(rgba.shape[1]), int(rgba.shape[0])]
    return rgba
