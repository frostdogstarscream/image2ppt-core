"""Discover inset phone screenshots from the source frame and its intrinsic text."""
from __future__ import annotations

import cv2
import numpy as np

from ..contracts import intersection


def _phone_shell(component: np.ndarray) -> bool:
    """A connected bezel has two thin long sides and a thicker central top notch."""
    h, w = component.shape
    middle = component[h // 6:4 * h // 5]
    side = max(2, w // 12)
    if min(middle[:, :side].any(1).mean(), middle[:, -side:].any(1).mean()) < 0.92:
        return False
    if not 0.025 <= component.mean() <= 0.25:
        return False
    top_depth = component[:max(3, h // 7)].sum(0)
    center = np.median(top_depth[2 * w // 5:3 * w // 5])
    shoulders = np.r_[top_depth[w // 8:w // 4], top_depth[3 * w // 4:7 * w // 8]]
    return np.median(shoulders) > 0 and center >= np.median(shoulders) + max(3, 0.025 * w)


def _shells(source: np.ndarray):
    h_img, w_img = source.shape[:2]
    gray = source.max(2)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((gray < 155).astype('uint8'), 8)
    for label, (x, y, w, h, _) in enumerate(stats[1:], 1):
        if not (w >= 40 and h >= 64 and 1.08 <= h / w <= 3.0):
            continue
        if x < 4 or y < 4 or x + w + 4 >= w_img or y + h + 4 >= h_img:
            continue
        component = labels[y:y+h, x:x+w] == label
        if not _phone_shell(component):
            continue
        pad = max(4, round(w * 0.04))
        x0, y0 = max(0, x-pad), max(0, y-pad)
        patch = gray[y0:min(h_img, y+h+pad), x0:min(w_img, x+w+pad)]
        _, weak = cv2.connectedComponents((patch < 220).astype('uint8'), connectivity=8)
        strong_y, strong_x = np.nonzero(component)
        weak_label = weak[y-y0+strong_y[0], x-x0+strong_x[0]]
        contour = cv2.findNonZero((weak == weak_label).astype('uint8'))
        bx, by, bw, bh = cv2.boundingRect(contour)
        if min(bx, by) == 0 or bx+bw >= patch.shape[1] or by+bh >= patch.shape[0]:
            continue  # The outline continues into something outside the search window.
        margin = max(2, round(w * 0.015))
        box = [int(x0+bx-margin), int(y0+by-margin), int(bw+2*margin), int(bh+2*margin)]
        if box[0] < 0 or box[1] < 0 or box[0]+box[2] > w_img or box[1]+box[3] > h_img:
            continue
        silhouette = np.zeros((box[3], box[2]), 'uint8')
        points = contour
        # A presentation card may clip the bottom of the screen. A faded rail
        # then ends above its partner. Confirm a real clipping edge in the
        # surrounding card before closing the two observed rails on that row.
        bottom = y0 + by + bh - 1
        if component[-max(3, h//10):, w//4:3*w//4].mean() < .025:
            side_strips = [source[bottom-3:bottom+2, x0+bx-5:x0+bx-1],
                           source[bottom-3:bottom+2, x0+bx+bw+1:x0+bx+bw+5]]
            clipped = any(strip.shape[:2] == (5, 4) and np.max(np.abs(np.diff(
                np.median(strip, axis=1), axis=0))) > 8 for strip in side_strips)
            if clipped:
                mid = (weak == weak_label)[by+bh//3:by+2*bh//3]
                rail_left = int(np.median(mid.argmax(1)))
                rail_right = int(mid.shape[1]-1-np.median(mid[:, ::-1].argmax(1)))
                bottom_corners = np.array([[[rail_left, by+bh-1]], [[rail_right, by+bh-1]]], dtype=np.int32)
                points = np.concatenate((contour, bottom_corners))
        hull = cv2.convexHull(points) + np.array([margin-bx, margin-by], dtype=np.int32)
        cv2.fillConvexPoly(silhouette, hull, 1)
        yield box, silhouette


def detect_device_frames(source: np.ndarray, text_items: list[dict], layout: dict | None) -> list[list[int]]:
    frames = []
    for box, _ in _shells(source):
        inner = [item for item in text_items if intersection(box, item['bbox']) >=
                 0.9 * max(1, item['bbox'][2] * item['bbox'][3])]
        image_hint = any(
            region.get('label') == 'image' and float(region.get('score', 0)) >= .45
            and intersection(box, region['bbox']) >= .6 * max(box[2]*box[3], region['bbox'][2]*region['bbox'][3])
            for region in (layout or {}).get('regions', [])
        )
        if len(inner) >= (2 if image_hint else 3):
            frames.append(box)
    return sorted(frames, key=lambda box: (box[1], box[0]))


def fit_device_backing(source: np.ndarray, box: list[int], protected: np.ndarray, peer_boxes=()):
    """Recheck shell pixels, then fit each row from both clear sides of the card.

    The row fit preserves a horizontal card edge behind a clipped phone. A frame
    hypothesis alone never licenses writing its bounding rectangle to the plate.
    """
    x, y, w, h = box
    # Recompute on a neighbourhood of the actual source; stored boxes are only proposals.
    pad = max(12, round(w * .1))
    x0, y0 = max(0, x-pad), max(0, y-pad)
    local = source[y0:min(source.shape[0], y+h+pad), x0:min(source.shape[1], x+w+pad)]
    expected = [x-x0, y-y0, w, h]
    silhouette = next((mask for rect, mask in _shells(local) if rect == expected), None)
    if silhouette is None:
        raise ValueError('source device bezel could not be verified')
    fringe = max(2, round(w * .015))
    owner = cv2.dilate(silhouette, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*fringe+1, 2*fringe+1))).astype(bool)
    if (protected[y:y+h, x:x+w] & owner).any():
        raise ValueError('device pixels overlap an independent object')
    width = max(4, min(8, w // 40))
    if x < width or x+w+width > source.shape[1]:
        raise ValueError('two source side strips are required')
    for peer in peer_boxes:
        if not intersection(box, peer):
            continue
        # OCR includes whitespace at the end of a line. Ignore only a shallow
        # grazing overlap backed by an uninterrupted clear gutter up to the bezel.
        px, py, pw, ph = peer
        if px < x and px+pw <= x+.04*w:
            gutter = source[max(y, int(py)):min(y+h, int(np.ceil(py+ph))), x-width:x]
        elif px >= x+.96*w and px+pw > x+w:
            gutter = source[max(y, int(py)):min(y+h, int(np.ceil(py+ph))), x+w:x+w+width]
        else:
            raise ValueError('device intersects a separate text or picture')
        if not gutter.size or gutter.min() < 220 or np.ptp(gutter, axis=0).max() > 12:
            raise ValueError('device and neighbour have no clear separating gutter')
    left = source[y:y+h, x-width:x].astype(np.float32)
    right = source[y:y+h, x+w:x+w+width].astype(np.float32)
    left_valid = ~protected[y:y+h, x-width:x]
    right_valid = ~protected[y:y+h, x+w:x+w+width]
    if min(left_valid.sum(1).min(), right_valid.sum(1).min()) < 3:
        raise ValueError('insufficient unowned backing samples on both sides')
    sides = []
    errors = []
    for strip, valid in ((left, left_valid), (right, right_valid)):
        color = np.nanmedian(np.where(valid[..., None], strip, np.nan), axis=1)
        errors.append(float(np.percentile(np.abs(strip - color[:, None]).max(2)[valid], 95)))
        sides.append(color)
    disagreement = float(np.percentile(np.abs(sides[0]-sides[1]).max(1), 95))
    if max(errors) > 4 or disagreement > 18:
        raise ValueError('side samples do not support a continuous device backing')
    t = (np.arange(w, dtype=np.float32) + (width+1)/2) / (w+width)
    backing = np.rint(sides[0][:, None]*(1-t[None, :, None]) + sides[1][:, None]*t[None, :, None]).astype('uint8')
    # The silhouette includes its intrinsic contents. Only observed colour changes
    # in the narrow outer fringe may transfer additional pixels to this object.
    delta = np.abs(source[y:y+h, x:x+w].astype(float)-backing).max(2)
    owner &= silhouette.astype(bool) | (delta > max(3, max(errors)))
    return backing, owner, {'method': 'source-device-bezel-and-paired-side-strips/1',
                            'side_p95_error': errors, 'side_color_disagreement_p95': disagreement,
                            'owner_pixels': int(owner.sum())}
