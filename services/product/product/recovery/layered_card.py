"""Recover the face of a detected frame after its independent photos have been removed."""

import cv2
import numpy as np
from PIL import Image
from ..contracts import intersection
from ..materialize import digest
from ..parallel_store import key
from .native_shapes import _fit_linear_gradient, estimate_canvas_field


def geometry_marks(source, text_rows):
    from .container_stage import _discover_candidates

    rgb = np.asarray(Image.open(source).convert('RGB'))
    canvas, _, scenery, flat = estimate_canvas_field(rgb)
    sha = digest(source)
    rows = []
    for kind, box in _discover_candidates(rgb, canvas, scenery, is_flat_canvas=flat):
        if kind != 'box':
            continue
        children = [
            row
            for row in text_rows
            if row.get('ocr_line_ids')
            and intersection(box, row['bbox']) >= 0.98 * row['bbox'][2] * row['bbox'][3]
        ]
        if len(children) < 2:
            continue
        rows.append(
            {
                'id': 'm-' + key([sha, 'card-frame', box])[:16],
                'label': 'card_frame',
                'bbox': box,
                'sources': [{'model': 'pixel-frame', 'label': 'card_frame'}],
            }
        )
    return rows


def contained_peers(box, peers, *, layered=False):
    surfaces = []
    for row in peers:
        bounds = row['bbox']
        if not row.get('owner_evidence') or intersection(box, bounds) < bounds[2]*bounds[3]:
            return False
        if row['kind'] == 'text' or row['kind'] == 'asset' and row.get('move_safety') == 'MOVABLE':
            continue
        if not (layered and row['kind'] == 'surface' and row.get('payload', {}).get('geometry') in ('rect', 'roundRect')):
            return False
        # A halo can extend outside a native shape's visible frame. The entire
        # proven restore area must belong to this parent before rebinding it.
        restore = row.get('restore_bbox', bounds)
        if intersection(box, restore) < restore[2]*restore[3]:
            return False
        surfaces.append(row)
    for surface in surfaces:
        for neighbor in peers:
            if neighbor is surface or not intersection(surface['bbox'], neighbor['bbox']):
                continue
            # Transparent text above a child shape keeps its existing delete proof.
            # Other overlapping shapes/photos need a separate occlusion proof.
            if neighbor['kind'] != 'text' or intersection(surface['bbox'], neighbor['bbox']) < neighbor['bbox'][2]*neighbor['bbox'][3]:
                return False
    return True


def bind_photo_backing(directory, record, frame_id, frame_box, field):
    """Bind a photo or native child to its measured parent fill without changing its ownership."""
    old = record['owner_evidence']
    path = directory / old['file']
    if digest(path) != old['sha256']:
        raise ValueError('photo ownership evidence changed')
    with np.load(path, allow_pickle=False) as data:
        contents = {name: data[name] for name in data.files}
    x, y, w, h = map(int, contents['restore_bbox'])
    left, top, width, height = frame_box
    if not (left <= x and top <= y and x + w <= left + width and y + h <= top + height):
        raise ValueError('child backing crosses native frame')
    contents['backing_patch'] = field[y - top : y - top + h, x - left : x - left + w].copy()
    new_path = directory / f"{record['id']}-{frame_id}-owner.npz"
    np.savez_compressed(new_path, **contents)
    record['owner_evidence'] = {
        'file': new_path.name,
        'sha256': digest(new_path),
        'method': ('native-card-under-verified-surface/1' if record.get('kind') == 'surface'
                   else 'native-card-under-verified-photo/1'),
        'prior': dict(old),
    }
    record['payload'] = {**record.get('payload', {}), 'background_parent_id': frame_id}


def native_child_faces(box, peers):
    """Raster footprint of opaque, already-verified rectangular child shapes."""
    x, y, w, h = box
    faces = np.zeros((h,w), bool)
    for row in peers:
        if row['kind'] != 'surface':
            continue
        ax, ay, aw, ah = map(round,row['bbox'])
        radius = min(aw/2, ah/2, row['payload'].get('corner_radius_px') or 0)
        yy, xx = np.ogrid[:ah, :aw]
        dx = np.maximum(radius-xx-.5, np.maximum(xx+.5-(aw-radius), 0))
        dy = np.maximum(radius-yy-.5, np.maximum(yy+.5-(ah-radius), 0))
        faces[ay-y:ay-y+ah,ax-x:ax-x+aw] |= dx*dx+dy*dy <= radius*radius
    return faces


def frame_layers(patch, parent, box, peers, baseline, ignore_boxes, *, child_faces=None):
    """Fit exposed card paper; retain unexplained interior art and leave photo holes transparent."""
    x, y, w, h = box
    sample = np.ones((h, w), bool)
    owned = np.zeros_like(sample)
    photos = np.zeros_like(sample)
    native_children = np.zeros_like(sample)
    if child_faces is None:
        child_faces = native_child_faces(box, peers)
    for r in peers:
        ref = r['owner_evidence']
        f = baseline / ref['file']
        if digest(f) != ref['sha256']:
            raise ValueError('card child source evidence changed')
        with np.load(f, allow_pickle=False) as proof:
            mask = proof['mask'][y : y + h, x : x + w].astype(bool)
        owned |= mask
        if r['kind'] == 'surface':
            native_children |= mask
            # Source-owner halos may extend beyond the opaque native face.
        if r['kind'] == 'asset':
            ax, ay, aw, ah = map(round, r['bbox'])
            occupied = mask[ay - y : ay - y + ah, ax - x : ax - x + aw]
            if occupied.shape != (ah, aw) or not occupied.all():
                return None
            photos |= mask
    for b in [r['bbox'] for r in peers] + list(ignore_boxes):
        ax, ay, aw, ah = map(round, b)
        sample[max(0, ay - y) : max(0, min(h, ay - y + ah)), max(0, ax - x) : max(0, min(w, ax - x + aw))] = (
            False
        )
    margin = max(2, round(min(w, h) * 0.018))
    sample[:margin] = sample[-margin:] = False
    sample[:, :margin] = sample[:, -margin:] = False
    smooth = cv2.GaussianBlur(patch, (5, 5), 0)
    sample &= np.max(np.abs(smooth.astype(float) - patch), axis=2) < 3
    if sample.sum() < 0.035 * w * h:
        return None
    colors = patch[sample].astype(int) // 16
    histogram = np.bincount(colors @ np.array([256, 16, 1]), minlength=4096)
    mode = int(histogram.argmax())
    center = np.median(patch[sample][colors @ np.array([256, 16, 1]) == mode], axis=0)
    sample &= np.abs(patch.astype(float) - center).max(2) < 22
    # Evidence for a backing must extend across the card rather than a photo or label.
    sy, sx = np.where(sample)
    if len(sx) < 0.035 * w * h or np.ptp(sx) < 0.75 * w or np.ptp(sy) < 0.55 * h:
        return None
    # Trim compression noise and text halos around the dominant paper colour.
    distance = np.abs(patch.astype(float) - center).max(2)
    sample &= distance <= max(5.0, float(np.percentile(distance[sample], 75)))
    fit = _fit_linear_gradient(patch, sample, 14)
    if fit is None:
        return None
    field, _, gradient = fit
    fill = np.rint(np.median(field[sample], axis=0)).astype('uint8')
    delta = np.abs(patch.astype(float) - field).max(2)
    # The exterior must meet a quiet page backing. Reject a picture or text crossing the edge.
    rim = np.zeros_like(sample)
    rim[[0, -1], :] = True
    rim[:, [0, -1]] = True
    exterior_error = np.abs(patch.astype(float) - parent).max(2)
    bad = (exterior_error > 35) & rim
    if bad.sum() > max(3, round(0.015 * rim.sum())):
        return None
    ink = (delta > 12) & ~photos
    # Keep complete islands such as photographs (including their light interiors).
    solid = (delta > 42) & ~photos
    contours, _ = cv2.findContours(solid.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        ax, ay, aw, ah = cv2.boundingRect(contour)
        if min(aw, ah) >= 12 and aw * ah < 0.55 * w * h and cv2.contourArea(contour) > 0.64 * aw * ah:
            local = np.zeros_like(sample, dtype='uint8')
            hull = cv2.convexHull(contour)
            # Large, nearly rectangular picture tiles can have pale details open to
            # their outer edge. Keep their full convex silhouette opaque on a new fill.
            picture_tile = (
                0.06 * w * h < aw * ah
                and 0.55 < aw / ah < 3.5
                and cv2.contourArea(hull) > 0.94 * aw * ah
                and cv2.contourArea(contour) > 0.78 * aw * ah
            )
            cv2.drawContours(local, [hull if picture_tile else contour], -1, 1, -1)
            solid |= local.astype(bool)
    ink |= solid
    # Where a rounded card meets its page, faint antialiasing belongs to the edge of
    # the shape. Keeping it as movable artwork makes the old outline follow the photo.
    border = np.ones_like(ink)
    border[margin:-margin, margin:-margin] = False
    edge_blend = border & (exterior_error <= 20) & (delta <= 26)
    # New retained artwork is above native shapes in the fixed compiler.
    # A previously hidden source pixel remains behind its opaque child. Moving
    # it into an upper picture would give that picture the wrong delete backing.
    ink &= ~(photos | native_children | child_faces | edge_blend)
    # Paper tint must follow a new card colour. Preserve full-opacity photo and label
    # interiors; express the low-contrast shading elsewhere as a translucent tint.
    opacity = np.where(solid, 1.0, np.minimum(delta / 170.0, 0.24))
    opacity = np.where(ink, opacity, 0.0)
    color = field + (patch.astype(float) - field) / np.maximum(opacity[..., None], 1 / 255)
    rgba = np.dstack(
        [np.clip(color, 0, 255).round().astype(np.uint8), (opacity * 255).round().astype(np.uint8)]
    )
    return fill, gradient, field, owned, ink, rgba
