"""Fit one exposed parent face for sibling shapes before any sibling is erased."""
import logging

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from .background_adapter import RecoveryBlocked
from .layerd_adapter import lower_backgrounds


def _inside(outer, inner):
    x, y, w, h = outer
    u, v, a, b = inner
    return x <= u and y <= v and u + a <= x + w and v + b <= y + h


def _parent_groups(candidates, series, size):
    page = (0, 0, *size)
    boxes = {tuple(box) for kind, box in candidates
             if kind == 'box' and min(box[2:]) > 0 and _inside(page, box)}
    parents = sorted(boxes, key=lambda b: (b[2]*b[3], b))
    children = {}
    for box in boxes:
        parent = next((p for p in parents if p != box and _inside(p, box)), None)
        if parent:
            children.setdefault(parent, set()).add(box)
    assigned = set()
    for members in series:
        members = sorted(set(map(tuple, members)) & boxes)
        if len(members) < 2 or assigned.intersection(members):
            continue
        parent = next((p for p in parents if all(_inside(p, b) and p != b for b in members)), (0, 0, *size))
        yield parent, members
        assigned.update(members)
    for parent in parents:
        siblings = sorted(children.get(parent, set()) - assigned)
        if len(siblings) > 1:
            yield parent, siblings


def _fit_face(rgb, visible, tile):
    """Interpolate low-frequency colour changes between actual samples of this face."""
    h, w = visible.shape
    centers, colors = [], []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            yy, xx = np.nonzero(visible[y:y+tile, x:x+tile])
            if len(xx) < max(3, tile*tile//5):
                continue
            pixels = rgb[y+yy, x+xx].astype(float)
            med = np.median(pixels, axis=0)
            if np.percentile(np.abs(pixels-med).max(1), 80) > 9:
                continue
            centers.append((x+np.median(xx), y+np.median(yy)))
            colors.append(med)
    if len(colors) < 7:
        return None
    centers, colors = np.asarray(centers), np.asarray(colors)
    design = np.column_stack((np.ones(len(centers)), centers))
    keep = np.ones(len(colors), bool)
    for _ in range(4):
        fit, _, rank, _ = np.linalg.lstsq(design[keep], colors[keep], rcond=None)
        if rank < 3:
            return None
        error = np.abs(design@fit - colors).max(1)
        limit = max(6.0, 3*np.median(error[keep]))
        good = error <= limit
        if good.sum() < 7:
            return None
        if np.array_equal(good, keep):
            break
        keep = good
    yy, xx = np.mgrid[:h, :w]
    if np.linalg.matrix_rank(design[keep]) < 3:
        return None
    valid = centers[keep]
    plane = np.stack((np.ones_like(xx), xx, yy), axis=-1) @ fit
    residuals = colors[keep] - design[keep] @ fit
    residual = LinearNDInterpolator(valid, residuals)(xx, yy)
    edge = ~np.isfinite(residual[..., 0])
    if edge.any():
        distance, closest = cKDTree(valid).query(np.column_stack((xx[edge], yy[edge])))
        values = residuals[closest].copy()
        values[distance > 3*tile] = np.nan
        residual[edge] = values
    return np.clip(plane + residual, 0, 255)


def shared_face_patches(plate, candidates, series, protected, *, source=None, layer_directory=None,
                        active_boxes=None):
    """Return validated per-child backdrops. A known group with no clean face is deferred."""
    h, w = plate.shape[:2]
    groups = [(p, boxes) for p, boxes in _parent_groups(candidates, series, (w,h))
              if active_boxes is None or active_boxes.intersection(boxes)]
    hypotheses = None
    patches = {}
    for parent, siblings in groups:
        px, py, pw, ph = parent
        scale = int(np.median([min(b[2:]) for b in siblings]))
        pad = max(4, scale//2)
        x0 = max(px, min(x for x, y, sw, sh in siblings)-pad)
        y0 = max(py, min(y for x, y, sw, sh in siblings)-pad)
        x1 = min(px+pw, max(x+sw for x, y, sw, sh in siblings)+pad)
        y1 = min(py+ph, max(y+sh for x, y, sw, sh in siblings)+pad)
        rgb = plate[y0:y1, x0:x1]
        visible = ~protected[y0:y1, x0:x1].copy()
        for x, y, sw, sh in siblings:
            visible[max(0,y-y0-1):y+sh-y0+1, max(0,x-x0-1):x+sw-x0+1] = False
        for kind, (x, y, sw, sh) in candidates:
            # Dividers and axes remain their own owner, even if attached to a bar.
            if kind in ('line', 'tick') and x < x1 and y < y1 and x+sw > x0 and y+sh > y0:
                visible[max(0,y-y0-1):y+sh-y0+1, max(0,x-x0-1):x+sw-x0+1] = False
        tile = max(4, min(scale//4, max(rgb.shape[:2])//5, 48))
        field = _fit_face(rgb, visible, tile)
        for box in siblings:
            if active_boxes is not None and box not in active_boxes:
                continue
            patches[box] = None
            x, y, sw, sh = box
            u, v = x-x0, y-y0
            border = np.zeros_like(visible)
            reach = max(3, min(tile, min(sw,sh)//3))
            border[max(0,v-reach):v+sh+reach, max(0,u-reach):u+sw+reach] = True
            border &= visible
            if border.sum() < max(12, min(sw, sh)):
                continue
            if field is not None:
                patch = field[v:v+sh, u:u+sw]
                errors = np.abs(rgb[border] - field[border]).max(1)
                finite = errors[np.isfinite(errors)]
                if (np.isfinite(patch).all() and len(finite) >= .75*len(errors)
                        and np.percentile(finite, 80) <= 6):
                    patches[box] = (patch, {'method':'shared-parent-samples/1', 'parent_bbox':list(parent),
                                            'context_error_p80':float(np.percentile(finite, 80))})
                    continue
            if layer_directory is None or source is None:
                continue
            if hypotheses is None:
                # Enumerate once. A reusable lower layer must be opaque across
                # every proposed scope and reconstruct this exact source image.
                union = np.zeros((h, w), bool)
                for _, members in groups:
                    for bx, by, bw, bh in members:
                        union[by:by+bh, bx:bx+bw] = True
                try:
                    hypotheses = lower_backgrounds(source, layer_directory, union)
                except (OSError, ValueError, RecoveryBlocked) as exc:
                    logging.getLogger(__name__).warning('Shared face LayerD unavailable: %s', type(exc).__name__)
                    hypotheses = []
            for lower in hypotheses:
                meta = lower.provenance
                crop = lower.image[y0:y1, x0:x1].astype(float)
                error = np.abs(rgb[border] - crop[border]).max(1)
                patch = crop[v:v+sh, u:u+sw]
                original = rgb[v:v+sh, u:u+sw].astype(float)
                face_color = np.median(rgb[border], axis=0)
                ink = np.abs(original-face_color).max(2) > 18
                if not ink.any() or np.mean(np.abs(original[ink]-patch[ink]).max(1)>12) < .85:
                    continue
                if float(np.percentile(error, 85)) > 7:
                    continue
                # Even a bound lower layer may contain a flat inpainted hole on
                # textured paper. Compare its fine detail with the exposed face.
                neighbors = border[:,1:] & border[:,:-1]
                if neighbors.any() and sw > 1:
                    visible_detail = np.median(np.abs(np.diff(rgb.astype(float), axis=1))[neighbors])
                    interior_detail = np.median(np.abs(np.diff(patch, axis=1)))
                    if visible_detail > 6 and interior_detail < .35*visible_detail:
                        continue
                patches[box] = (patch, {'method':'shared-parent-layerd/1', 'parent_bbox':list(parent),
                                        'context_error_p80':float(np.percentile(error, 80)),
                                        'manifest_sha256': meta['manifest_sha256'], 'below_z':meta['below_z']})
                break
    return patches


def bar_edge_backing(plate, box, patch, forbidden):
    """Remove coherent, low-contrast ringing attached to a verified bar's two sides.

    Leave the axis and top of the column alone. Contrast or protected content
    crossing either side interrupts the corresponding rows.
    """
    x, y, w, h = box
    reach = min(max(2, round(min(w,h)*.08)), 6, x//2, (plate.shape[1]-x-w)//2)
    if not reach:
        return None
    result = np.pad(patch, ((0,0), (reach,reach), (0,0)), mode='edge')
    owned = np.zeros(result.shape[:2], bool)
    for left in (True, False):
        edge = x if left else x+w-1
        direction = -1 if left else 1
        indices = edge + direction*np.arange(1, 2*reach+1)
        pixels = plate[y:y+h, indices].astype(float)
        desired = patch[:, :1] if left else patch[:, -1:]
        # The far strip must return to this parent. A neighbouring object
        # that continues through the far strip has separate ownership.
        outside = np.abs(pixels[:, reach:] - desired).max(2)
        gap = np.abs(pixels - desired).max(2)
        positions = (edge-x+reach) + direction*np.arange(1,reach+1)
        clear = ~forbidden[y:y+h,indices].any(1) & (outside.mean(1) <= 5)
        clear &= (np.max(gap[:, :reach], axis=1) <= 12)
        clear[:reach] = False
        clear[-reach:] = False
        if clear.sum() < .55*h:
            continue
        # Require a border-wide excess over the far strip's natural variation.
        baseline = np.median(outside[clear])
        if np.max(np.median(gap[clear,:reach],axis=0)) <= baseline + .75:
            continue
        owned[np.ix_(clear, positions)] = True
    if not owned.any():
        return None
    return [x-reach,y,w+2*reach,h], owned, result
