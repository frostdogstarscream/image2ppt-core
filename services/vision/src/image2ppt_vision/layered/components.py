"""Rebuild the page from clean components instead of erasing and repairing.

OpenCV finds the designed components (cards, bands, title bars, ribbons,
badges) from three cues: calm connected regions, a two-colour split of a calm
region whose halves touch through a soft edge, and closed contours for a white
card on a white page that shows only its border. Containment between them is
the component tree. Each component is measured: a preset geometry plus a solid
or linear fill that reproduces its body becomes a native shape; anything else
is a picture cropped from the text-free page. SAM pictures (photos, charts,
logos) and the icons lifted off shapes join them.

Nothing under a component is reconstructed with a model. The page background
keeps the original pixels outside the components; under them it is a smooth
fill of the page surface, and a picture's own holes under its children are
filled the same way.
"""
import cv2
import numpy as np

from .inpaint import pull_push
from .segment import (CHEVRON_IOU, FLAT_IOU, LINEAR_TOLERANCE, NEST_SHARE, SHAPE_FULL_TOLERANCE, _border, _bbox,
                      _iou, _predict, assign_levels, fill_holes, fit_fill, fit_geometry, layer_erase, residual_blobs)

CALM = 16.
MIN_SIDE = 18
MIN_AREA = 450
MAX_AREA = .45
COMPONENT_IOU = FLAT_IOU
NATIVE_IOU = .9
SPLIT_LAB = 18.
CANNY = (12, 36)
DUPLICATE_IOU = .9
SAM_COVERED = .8
SAM_CALM = .6
LARGE = .03
LARGE_RESIDUAL = 25.
SKIP = 7
GROW = 4
GROW_RGB = 12.
SQUARE_RADIUS = 2
ICON_INK = 20
ICON_GAP = 7
SEE_THROUGH = .65
SEE_THROUGH_STD = 1.5
FRINGE = 5
FRINGE_MARGIN = 18.
BUSY = 28.
PHOTO_AREA = 5000
PHOTO_SIDE = 48


def _geometry_ok(mask, floor):
    x, y, w, h = _bbox(mask)
    fit = fit_geometry(mask[y:y+h, x:x+w])
    limit = min(floor, CHEVRON_IOU) if fit['geometry'] == 'chevron' else floor
    return fit if fit['iou'] >= limit else None


def _sized(mask, H, W):
    area = int(mask.sum())
    if area < MIN_AREA or area > MAX_AREA*H*W:
        return False
    x, y, w, h = _bbox(mask)
    return min(w, h) >= MIN_SIDE and not (w > .95*W and h > .5*H)


def regrow(rgb, mask, *, steps=GROW, tolerance=GROW_RGB):
    """``mask`` grown back over the band the edge detector took from it: a
    neighbouring pixel joins when it has the colour of the region pixels
    next to it, so the region stops at the designed edge."""
    H, W = mask.shape
    x, y, w, h = _bbox(mask)
    x0, y0, x1, y1 = max(0, x-steps), max(0, y-steps), min(W, x+w+steps), min(H, y+h+steps)
    img = rgb[y0:y1, x0:x1].astype(np.float32)
    part = mask[y0:y1, x0:x1].copy()
    k = np.ones((3, 3), np.float32)
    for _ in range(steps):
        m = part.astype(np.float32)
        count = cv2.filter2D(m, -1, k, borderType=cv2.BORDER_CONSTANT)
        mean = cv2.filter2D(img*m[..., None], -1, k, borderType=cv2.BORDER_CONSTANT)/np.maximum(count, 1)[..., None]
        grow = (count > 0) & ~part & (np.linalg.norm(img-mean, axis=2) <= tolerance)
        if not grow.any():
            break
        part |= grow
    out = mask.copy()
    out[y0:y1, x0:x1] = part
    return out


def calm_regions(rgb, grad):
    """Connected calm regions, holes closed. A region that is not one preset
    shape may be two blocks meeting through a soft edge: split it by colour."""
    H, W = rgb.shape[:2]
    edge = cv2.dilate((grad > CALM).astype(np.uint8), np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((edge == 0).astype(np.uint8), connectivity=4)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        mask = fill_holes(regrow(rgb, labels == i))
        if not _sized(mask, H, W):
            continue
        if _geometry_ok(mask, COMPONENT_IOU):
            out.append({'mask': mask, 'source': 'calm_region'})
            continue
        out += colour_split(rgb, lab, labels == i, H, W)
    return out


def colour_split(rgb, lab, region, H, W):
    values = lab[region]
    if len(values) < 2*MIN_AREA:
        return []
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, .5)
    _, assign, centres = cv2.kmeans(values, 2, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    if float(np.linalg.norm(centres[0]-centres[1])) < SPLIT_LAB:
        return []
    split = np.zeros(region.shape, np.int32)
    split[region] = assign[:, 0]+1
    out = []
    for k in (1, 2):
        n, labels, stats, _ = cv2.connectedComponentsWithStats((split == k).astype(np.uint8), connectivity=4)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
                continue
            mask = fill_holes(regrow(rgb, labels == i))
            if _sized(mask, H, W) and _geometry_ok(mask, COMPONENT_IOU):
                out.append({'mask': mask, 'source': 'colour_split'})
    return out


def contour_regions(rgb):
    """Closed outlines (a 1px border, a shadow edge) around rectangular cards,
    with their nesting from the contour hierarchy."""
    H, W = rgb.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    edges = cv2.morphologyEx(cv2.Canny(gray, *CANNY), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < MIN_AREA:
            continue
        mask = np.zeros((H, W), np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, -1)
        # Canny marks the page side of an edge as well; one pixel in is the component.
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
        if not _sized(mask, H, W):
            continue
        fit = _geometry_ok(mask, NATIVE_IOU)
        if fit and fit['geometry'] in ('rect', 'roundRect'):
            out.append({'mask': mask, 'source': 'contour', 'contour_parent': int(hierarchy[0][i][3])})
    return out


def component_tree(rgb, grad):
    """Deduplicated components with ``parent`` = the smallest component holding them."""
    cands = calm_regions(rgb, grad)+contour_regions(rgb)
    kept = []
    for c in sorted(cands, key=lambda c: -int(c['mask'].sum())):
        c['bbox'], c['area'] = _bbox(c['mask']), int(c['mask'].sum())
        if any(_near_iou(c, k) >= DUPLICATE_IOU for k in kept):
            continue
        kept.append(c)
    for i, c in enumerate(kept):
        holders = [j for j, k in enumerate(kept) if j != i and k['area'] > c['area'] and _share(c, k) >= NEST_SHARE]
        c['tree_parent'] = min(holders, key=lambda j: kept[j]['area']) if holders else None
    return kept


def _overlap_box(a, b):
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0]+a[2], b[0]+b[2]), max(a[1]+a[3], b[1]+b[3])
    return x0, y0, x1, y1


def _disjoint(a, b):
    return a[0] >= b[0]+b[2] or b[0] >= a[0]+a[2] or a[1] >= b[1]+b[3] or b[1] >= a[1]+a[3]


def _near_iou(a, b):
    if _disjoint(a['bbox'], b['bbox']):
        return 0.
    x0, y0, x1, y1 = _overlap_box(a['bbox'], b['bbox'])
    return _iou(a['mask'][y0:y1, x0:x1], b['mask'][y0:y1, x0:x1])


def _share(a, b):
    """Share of ``a`` inside ``b``."""
    if _disjoint(a['bbox'], b['bbox']):
        return 0.
    x0, y0, x1, y1 = _overlap_box(a['bbox'], b['bbox'])
    ma = a['mask'][y0:y1, x0:x1]
    return float((ma & b['mask'][y0:y1, x0:x1]).sum())/max(int(ma.sum()), 1)


def measure(rgb, comp, children, protect, calm):
    """Native-shape record for ``comp`` or None when its pixels are not one
    preset shape with one fill. ``children`` is the union of what lies on it."""
    mask = comp['mask']
    x, y, w, h = comp['bbox']
    fit = fit_geometry(mask[y:y+h, x:x+w])
    floor = CHEVRON_IOU if fit['geometry'] == 'chevron' else NATIVE_IOU
    if fit['iou'] < floor:
        return None
    if fit['geometry'] == 'roundRect' and fit['corner_radius_px'] <= SQUARE_RADIUS:
        fit = {**fit, 'geometry': 'rect', 'corner_radius_px': 0}
    inner = mask & (cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    body = inner & ~children & ~protect
    own = body & calm
    if int(own.sum()) < 60:
        own = body
    fill = fit_fill(rgb, own, comp['bbox'])
    if fill is None:
        return None
    limit = SHAPE_FULL_TOLERANCE if fill['kind'] == 'solid' else LINEAR_TOLERANCE
    blobs = residual_blobs(rgb, body, fill, comp['bbox'])
    lifted = np.zeros(mask.shape, bool)
    for blob in blobs:
        lifted |= blob
    ys, xs = np.where(body & ~lifted)
    if len(xs) < 30:
        return None
    deviation = np.abs(rgb[ys, xs].astype(float)-_predict(fill, xs, ys, comp['bbox'])).max(1)
    if float(np.percentile(deviation, 90)) > limit:
        return None
    if comp['area'] > LARGE*mask.size and float(np.percentile(deviation, 99)) > LARGE_RESIDUAL:
        return None
    record = {'kind': 'native_shape', 'source': comp['source'], 'bbox': list(comp['bbox']), 'area': comp['area'],
              'geometry': fit['geometry'], 'geometry_fit': fit, 'fill_fit': fill, 'mask': mask}
    if fit['geometry'] in ('roundRect', 'chevron'):
        record['corner_radius_px'] = fit['corner_radius_px']
    if fill['kind'] == 'solid':
        record['fill'] = fill['color']
    else:
        record['fill'] = fill['stops'][0]['color']
        record['fill_gradient'] = {'angle_deg': fill['angle_deg'], 'stops': fill['stops']}
    border = _border(rgb, mask, fill)
    if border:
        record['line'] = {'color': border, 'width_px': 1}
    record['icons'] = blobs
    return record


def rebuild(rgb, objects, W, H, grad, *, text_mask, icon_mask):
    """Elements for the page: OpenCV components as native shapes or pictures,
    plus the SAM pictures and icons in ``objects`` (``blockify`` output) that no
    component already explains. Returns (elements, erase)."""
    protect = cv2.dilate((text_mask | icon_mask).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    calm = grad <= CALM
    comps = component_tree(rgb, grad)
    icons = [o for o in objects if o.get('role') == 'icon']
    pictures = [o for o in objects if o['kind'] == 'raster' and o.get('role') != 'icon']
    shapes, fallbacks = [], []

    def lying_on(c, skip=None):
        children = np.zeros((H, W), bool)
        for k in comps:
            if k is not c and k['area'] < c['area'] and _share(k, c) >= NEST_SHARE:
                children |= k['mask']
        for o in icons+pictures:
            if o is not skip and o['area'] < c['area'] and not _disjoint(o['bbox'], c['bbox']):
                children |= o['mask'] & c['mask']
        return children

    for c in comps:
        record = measure(rgb, c, lying_on(c), protect, calm)
        if record and page_shows_through(rgb, record['mask']):
            record = None
        if record:
            shapes.append(record)
        else:
            fallbacks.append(c)
    # A SAM picture that one native shape already reproduces is that shape; a
    # calm SAM block (a ribbon, a tab, a disc) is measured like any component.
    kept = []
    for p in pictures:
        if any(_share(p, s) >= SAM_COVERED and _share(s, p) >= SAM_COVERED for s in shapes):
            continue
        if float(calm[p['mask']].mean()) >= SAM_CALM:
            record = measure(rgb, {'mask': p['mask'], 'bbox': p['bbox'], 'area': int(p['mask'].sum()),
                                   'source': 'sam_block'}, lying_on(p, p), protect, calm)
            if record and not page_shows_through(rgb, record['mask']):
                shapes.append(record)
                continue
        kept.append(p)
    pictures = kept
    for c in fallbacks:
        if any(_share(c, p) >= SAM_COVERED for p in pictures) or any(_share(c, s) >= SAM_COVERED and _share(s, c) >= SAM_COVERED for s in shapes):
            continue
        pictures.append({'kind': 'raster', 'role': 'block', 'source': c['source'], 'bbox': list(c['bbox']),
                         'area': c['area'], 'mask': c['mask'], 'alpha': 'mask', 'members': 1, 'absorbed': 0,
                         'geometry': 'outline', 'fallback': True})
    for s in shapes:
        # An icon holding the shape itself (its glow) lies beneath it and lifts nothing on it.
        above = [i for i in icons if _share(s, i) < .5]
        for blob in s.pop('icons'):
            if any(_iou(blob, i['mask']) >= .5 or _share({'mask': blob, 'bbox': _bbox(blob)}, i) >= NEST_SHARE
                   for i in above):
                continue
            icons.append({'kind': 'raster', 'role': 'icon', 'source': 'shape_residual', 'bbox': _bbox(blob),
                          'area': int(blob.sum()), 'mask': blob, 'alpha': 'mask', 'absorbed': 0})
    icons += uncovered_icons(shapes, pictures+icons, icon_mask & ~text_mask, H, W)
    pictures += embedded_pictures(rgb, grad, shapes+pictures+icons, H, W)
    out = shapes+pictures+icons
    for k, e in enumerate(out):
        e['id'] = f'el-{k:03d}'
    assign_levels(out)
    claim_fringe(rgb, out)
    return out, layer_erase(out, -1, H, W)


def page_shows_through(rgb, mask):
    """A large component whose broad shading still carries the page under it.

    Comparison is on a heavy blur, so icons and text holes on the component do
    not hide a wave that runs through it. A flat tint, or a designed gradient
    on a flat page, does not vary together with that surface."""
    H, W = mask.shape
    if int(mask.sum()) < LARGE*H*W:
        return False
    inner = cv2.erode(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    if int(inner.sum()) < 400:
        return False
    surface = smooth_under(rgb, mask)
    actual = cv2.GaussianBlur(rgb, (0, 0), 14)[inner].astype(np.float32).mean(1)
    under = cv2.GaussianBlur(surface, (0, 0), 14)[inner].astype(np.float32).mean(1)
    if float(actual.std()) < SEE_THROUGH_STD or float(under.std()) < SEE_THROUGH_STD:
        return False
    actual -= float(actual.mean())
    under -= float(under.mean())
    return float(np.corrcoef(actual, under)[0, 1]) >= SEE_THROUGH


def claim_fringe(rgb, elements, *, band=FRINGE):
    """Pixels of a child left just outside its mask (an anti-aliased edge, a
    notch the contour missed) join the child, so the parent crop is filled
    with the parent surface there instead of keeping a sliver of the child."""
    by_id = {e['id']: e for e in elements if 'id' in e}
    for child in elements:
        parent = by_id.get(child.get('parent'))
        if parent is None:
            continue
        others = np.zeros(child['mask'].shape, bool)
        for e in elements:
            if e is not child and e is not parent:
                others |= e['mask']
        ring = (cv2.dilate(child['mask'].astype(np.uint8), np.ones((2*band+1, 2*band+1), np.uint8)) > 0)
        ring &= ~child['mask'] & parent['mask'] & ~others
        if int(ring.sum()) < 8 or int(child['mask'].sum()) < 8:
            continue
        host = parent['mask'] & ~child['mask'] & ~others
        if int(host.sum()) < 8:
            continue
        child_color = np.median(rgb[child['mask']], 0)
        parent_color = np.median(rgb[host], 0)
        ys, xs = np.where(ring)
        pixels = rgb[ys, xs].astype(float)
        dist_child = np.linalg.norm(pixels-child_color, axis=1)
        dist_parent = np.linalg.norm(pixels-parent_color, axis=1)
        # The same colour as the child, and not the parent's own colour.
        closer = (dist_child <= FRINGE_MARGIN) & (dist_child+FRINGE_MARGIN < dist_parent)
        if int(closer.sum()) < 8:
            continue
        grow = np.zeros(child['mask'].shape, bool)
        grow[ys[closer], xs[closer]] = True
        child['mask'] = child['mask'] | grow
        child['bbox'], child['area'] = _bbox(child['mask']), int(child['mask'].sum())


def embedded_pictures(rgb, grad, elements, H, W):
    """A busy block inside a large component that no child covers (a photo the
    contour pass merged into the card) becomes its own picture."""
    added = []
    for host in elements:
        if host['area'] < LARGE*H*W:
            continue
        covered = np.zeros((H, W), bool)
        for other in elements:
            if other is not host and other['area'] < host['area'] and _share(other, host) >= .5:
                covered |= other['mask']
        busy = host['mask'] & ~covered & (grad > BUSY)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(busy.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            core = int(stats[i, cv2.CC_STAT_AREA])
            x, y, w, h = (int(v) for v in stats[i, :4])
            if core < PHOTO_AREA or min(w, h) < PHOTO_SIDE or core < .15*w*h:
                continue
            mask = np.zeros((H, W), bool)
            mask[y:y+h, x:x+w] = host['mask'][y:y+h, x:x+w] & ~covered[y:y+h, x:x+w]
            if int(mask.sum()) < PHOTO_AREA or any(_iou(mask, e['mask']) >= .5 for e in elements+added):
                continue
            added.append({'kind': 'raster', 'role': 'block', 'source': 'embedded_texture', 'bbox': _bbox(mask),
                          'area': int(mask.sum()), 'mask': mask, 'alpha': 'mask', 'members': 1, 'absorbed': 0,
                          'geometry': 'rect', 'fallback': True})
    return added


def uncovered_icons(shapes, lifted, icon_mask, H, W):
    """Icon ink on a native shape that nothing lifts. The shape's fill is
    measured around this ink, so without an icon of its own it would vanish."""
    covered = np.zeros((H, W), bool)
    for e in lifted:
        covered |= e['mask']
    on = np.zeros((H, W), bool)
    for s in shapes:
        on |= s['mask']
    ink = icon_mask & on & ~covered
    if int(ink.sum()) < ICON_INK:
        return []
    grouped = cv2.dilate(ink.astype(np.uint8), np.ones((ICON_GAP, ICON_GAP), np.uint8))
    n, labels = cv2.connectedComponents(grouped, connectivity=8)
    near = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out = []
    for i in range(1, n):
        mask = (labels == i) & near & on & ~covered
        if int((mask & ink).sum()) < ICON_INK:
            continue
        out.append({'kind': 'raster', 'role': 'icon', 'source': 'icon_on_shape', 'bbox': _bbox(mask),
                    'area': int(mask.sum()), 'mask': mask, 'alpha': 'mask', 'absorbed': 0})
    return out


def smooth_under(rgb, hole, *, skip=SKIP, pad=40, outlier=15.):
    """``rgb`` with ``hole`` filled from the surrounding surface only. The rim
    just outside the hole (an anti-aliased edge, a drop shadow) is not used,
    nor are pixels far from the surface's dominant colour around each hole
    (an arrow or a photo next to the card)."""
    if not hole.any():
        return rgb.copy()
    H, W = hole.shape
    k = np.ones((2*skip+1, 2*skip+1), np.uint8)
    out = rgb.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        x0, y0, x1, y1 = max(0, x-pad), max(0, y-pad), min(W, x+w+pad), min(H, y+h+pad)
        part = labels[y0:y1, x0:x1] == i
        crop = rgb[y0:y1, x0:x1].astype(np.float32)
        unknown = cv2.dilate(hole[y0:y1, x0:x1].astype(np.uint8), k) > 0
        if not (~unknown).any():
            unknown = hole[y0:y1, x0:x1]
        ring = (cv2.dilate(unknown.astype(np.uint8), k) > 0) & ~unknown
        if ring.sum() >= 8:
            far = np.linalg.norm(crop-np.median(crop[ring], 0), axis=2) > outlier
            known = ~unknown
            if (known & far).sum() < .5*known.sum():
                unknown = unknown | far
        if not (~unknown).any():
            continue
        filled = np.clip(np.rint(pull_push(crop, unknown)), 0, 255).astype(np.uint8)
        out[y0:y1, x0:x1][part] = filled[part]
    return out
