"""Preset geometry and fill fitting for LayerD alpha components."""
from __future__ import annotations

import math

import cv2
import numpy as np

LINEAR_TOLERANCE = 8.0
SHAPE_FULL_TOLERANCE = 6.0
CHEVRON_IOU = 0.86
NATIVE_IOU = 0.9
NEST_SHARE = 0.9
LARGE = 0.03
LARGE_RESIDUAL = 25.0
SQUARE_RADIUS = 2
SEMI_OPAQUE = 0.85
MIN_AREA = 450
MIN_SIDE = 18


def _hex(rgb):
    return "".join(f"{int(v):02X}" for v in np.clip(np.rint(rgb), 0, 255))


def bbox(mask):
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def fill_holes(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return mask
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    local = mask[y0:y1, x0:x1].astype(np.uint8)
    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(local)
    cv2.drawContours(filled, contours, -1, 1, -1)
    out = mask.copy()
    out[y0:y1, x0:x1] |= filled > 0
    return out


def _rounded(w, h, r):
    t = np.zeros((h, w), np.uint8)
    r = int(min(r, w // 2, h // 2))
    if r <= 0:
        t[:] = 1
        return t.astype(bool)
    cv2.rectangle(t, (r, 0), (w - 1 - r, h - 1), 1, -1)
    cv2.rectangle(t, (0, r), (w - 1, h - 1 - r), 1, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(t, (cx, cy), r, 1, -1)
    return t.astype(bool)


def _chevron(w, h, notch):
    notch = int(max(1, min(notch, w // 2 - 1, h // 2)))
    pts = np.array([[notch, 0], [w - 1 - notch, 0], [w - 1, h // 2], [w - 1 - notch, h - 1],
                    [notch, h - 1], [0, h // 2]], np.int32)
    canvas = np.zeros((h, w), np.uint8)
    cv2.fillPoly(canvas, [pts], 1)
    return canvas.astype(bool)


def fit_geometry(local):
    h, w = local.shape
    best = ("rect", 0, float(local.mean()))
    limit = min(w, h) // 2
    if limit >= 2:
        for r in sorted({max(1, int(limit * f)) for f in np.linspace(0.04, 1, 24)}):
            score = iou(local, _rounded(w, h, r))
            if score > best[2] + 1e-6:
                best = ("roundRect", r, score)
    e = np.zeros((h, w), np.uint8)
    cv2.ellipse(e, (((w - 1) / 2, (h - 1) / 2), (float(w), float(h)), 0.0), 1, -1)
    score = iou(local, e.astype(bool))
    if score > best[2] + 0.01:
        best = ("ellipse", 0, score)
    if w >= 16 and h >= 8:
        for frac in (0.2, 0.28, 0.36, 0.45):
            notch = max(2, int(round(min(w, h) * frac)))
            score = iou(local, _chevron(w, h, notch))
            if score > best[2] + 0.01:
                best = ("chevron", notch, score)
    return {"geometry": best[0], "corner_radius_px": best[1], "iou": round(best[2], 4)}


def fit_fill(rgb, own, box):
    ys, xs = np.where(own)
    if len(xs) < 12:
        return None
    values = rgb[ys, xs].astype(np.float64)
    median = np.median(values, 0)
    flat = float(np.percentile(np.abs(values - median).max(1), 90))
    result = {"kind": "solid", "color": _hex(median), "residual": round(flat, 3)}
    x, y, w, h = box
    if len(xs) >= 60 and max(w, h) >= 24:
        design = np.stack([(xs - x) / max(w, 1), (ys - y) / max(h, 1), np.ones(len(xs))], 1)
        coef, *_ = np.linalg.lstsq(design, values, rcond=None)
        linear = float(np.percentile(np.abs(values - design @ coef).max(1), 90))
        lum = coef[:2] @ np.array([0.299, 0.587, 0.114])
        span = float(np.abs(coef[:2]).max())
        if linear < flat * 0.7 and span >= 8:
            gx, gy = lum[0] / max(w, 1), lum[1] / max(h, 1)
            if abs(gx) + abs(gy) < 1e-9:
                channel = int(np.abs(coef[:2]).max(0).argmax())
                gx, gy = coef[0, channel] / max(w, 1), coef[1, channel] / max(h, 1)
            angle = math.degrees(math.atan2(gy, gx)) % 360
            corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
            direction = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            t = corners @ direction
            start, end = corners[t.argmin()], corners[t.argmax()]
            color = lambda p: np.array([p[0] / max(w, 1), p[1] / max(h, 1), 1]) @ coef
            result = {"kind": "linear", "angle_deg": round(angle, 2), "residual": round(linear, 3),
                      "flat_residual": round(flat, 3),
                      "stops": [{"offset": 0, "color": _hex(color(start))},
                                {"offset": 100000, "color": _hex(color(end))}]}
    return result


def predict(fill, xs, ys, box):
    if fill["kind"] == "solid":
        return np.array([int(fill["color"][i:i + 2], 16) for i in (0, 2, 4)], float)[None]
    x, y, w, h = box
    a = math.radians(fill["angle_deg"])
    d = np.array([math.cos(a), math.sin(a)])
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float) @ d
    t = (np.stack([xs - x, ys - y], 1) @ d - corners.min()) / max(np.ptp(corners), 1e-6)
    c0, c1 = (np.array([int(s["color"][i:i + 2], 16) for i in (0, 2, 4)], float) for s in fill["stops"])
    return c0[None] + np.clip(t, 0, 1)[:, None] * (c1 - c0)[None]


def border(rgb, mask, fill):
    ring = mask & ~(cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0)
    if ring.sum() < 20 or fill["kind"] != "solid":
        return None
    values = rgb[ring].astype(np.float64)
    median = np.median(values, 0)
    fill_rgb = np.array([int(fill["color"][i:i + 2], 16) for i in (0, 2, 4)], float)
    spread = float(np.percentile(np.abs(values - median).max(1), 75))
    if np.abs(median - fill_rgb).max() > 22 and spread <= 14:
        return _hex(median)
    return None


def residual_blobs(rgb, region, fill, box, *, threshold=38.0, min_area=25):
    ys, xs = np.where(region)
    if not len(xs):
        return []
    deviation = np.abs(rgb[ys, xs].astype(float) - predict(fill, xs, ys, box)).max(1)
    off = np.zeros(region.shape, np.uint8)
    off[ys[deviation > threshold], xs[deviation > threshold]] = 1
    off = cv2.morphologyEx(off, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(off, 8)
    out = []
    for i in range(1, n):
        if stats[i][4] < min_area:
            continue
        blob = labels == i
        out.append(cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    return out


def share(a, b):
    if not a.any() or not b.any():
        return 0.0
    return float((a & b).sum()) / max(int(a.sum()), 1)


def assign_levels(objects):
    order = sorted(objects, key=lambda e: -int(e["mask"].sum()))
    for i, e in enumerate(order):
        parent = None
        for c in reversed(order[:i]):
            if share(e["mask"], c["mask"]) >= NEST_SHARE:
                parent = c
                break
        e["level"] = parent["level"] + 1 if parent else 0
        e["parent"] = parent["id"] if parent and "id" in parent else None
    return objects


def measure_component(rgb, mask, *, protect, children=None, alpha=None):
    """Native-shape record or None. Semi-transparent alpha forces a picture."""
    children = children if children is not None else np.zeros(mask.shape, bool)
    mask = fill_holes(mask)
    area = int(mask.sum())
    if area < MIN_AREA:
        return None
    box = bbox(mask)
    x, y, w, h = box
    if min(w, h) < MIN_SIDE:
        return None
    if alpha is not None:
        opaque = float((alpha[mask] >= 250).mean())
        if opaque < SEMI_OPAQUE:
            return None
    fit = fit_geometry(mask[y:y + h, x:x + w])
    floor = CHEVRON_IOU if fit["geometry"] == "chevron" else NATIVE_IOU
    if fit["iou"] < floor:
        return None
    if fit["geometry"] == "roundRect" and fit["corner_radius_px"] <= SQUARE_RADIUS:
        fit = {**fit, "geometry": "rect", "corner_radius_px": 0}
    inner = mask & (cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    body = inner & ~children & ~protect
    if int(body.sum()) < 60:
        body = mask & ~children & ~protect
    fill = fit_fill(rgb, body, box)
    if fill is None:
        return None
    limit = SHAPE_FULL_TOLERANCE if fill["kind"] == "solid" else LINEAR_TOLERANCE
    blobs = residual_blobs(rgb, body, fill, box)
    lifted = np.zeros(mask.shape, bool)
    for blob in blobs:
        lifted |= blob
    ys, xs = np.where(body & ~lifted)
    if len(xs) < 30:
        return None
    deviation = np.abs(rgb[ys, xs].astype(float) - predict(fill, xs, ys, box)).max(1)
    if float(np.percentile(deviation, 90)) > limit:
        return None
    if area > LARGE * mask.size and float(np.percentile(deviation, 99)) > LARGE_RESIDUAL:
        return None
    record = {"kind": "native_shape", "bbox": box, "area": area, "geometry": fit["geometry"],
              "geometry_fit": fit, "fill_fit": fill, "mask": mask, "icons": blobs}
    if fit["geometry"] in ("roundRect", "chevron"):
        record["corner_radius_px"] = fit["corner_radius_px"]
    if fill["kind"] == "solid":
        record["fill"] = fill["color"]
    else:
        record["fill"] = fill["stops"][0]["color"]
        record["fill_gradient"] = {"angle_deg": fill["angle_deg"], "stops": fill["stops"]}
    edge = border(rgb, mask, fill)
    if edge:
        record["line"] = {"color": edge, "width_px": 1}
    return record
