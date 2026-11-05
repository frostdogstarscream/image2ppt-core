"""Fit clean source-backed surfaces and build strictly bounded text masks."""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw


def shape_mask(size, bbox, radius=0, polygon=None):
    image = Image.new("L", tuple(size), 0)
    draw = ImageDraw.Draw(image)
    x, y, width, height = bbox
    if polygon:
        draw.polygon([tuple(point) for point in polygon], fill=255)
    else:
        draw.rounded_rectangle((x, y, x + width - 1, y + height - 1), radius=radius, fill=255)
    return np.array(image) > 0


def color_hex(value):
    return "".join(f"{int(channel):02X}" for channel in np.clip(np.rint(value), 0, 255))


def gradient_pixels(model, bbox):
    x, y, width, height = bbox
    yy, xx = np.mgrid[y:y + height, x:x + width]
    if model["kind"] == "solid":
        return np.broadcast_to(model["colors"][0], (height, width, 3)).copy().astype(np.uint8)
    dx, dy = model["direction"]
    projection = (xx - x) * dx + (yy - y) * dy
    corners = [0, (width - 1) * dx, (height - 1) * dy,
               (width - 1) * dx + (height - 1) * dy]
    amount = np.clip((projection - min(corners)) / max(1e-6, max(corners) - min(corners)), 0, 1)
    stops = np.asarray(model["offsets"])
    colors = np.asarray(model["colors"])
    result = np.stack([np.interp(amount, stops, colors[:, channel]) for channel in range(3)], axis=2)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def fit_surface(source, bbox, allowed, *, max_samples=30000):
    """Choose the simplest solid/linear model meeting the saved quality limits."""
    x, y, width, height = bbox
    yy, xx = np.nonzero(allowed[y:y + height, x:x + width])
    if len(xx) < 60:
        raise ValueError("insufficient clean background samples")
    stride = max(1, math.ceil(len(xx) / max_samples))
    xx, yy = xx[::stride], yy[::stride]
    values = source[y + yy, x + xx].astype(np.float64)
    matrix = np.column_stack((np.ones(len(xx)), xx / max(1, width - 1), yy / max(1, height - 1)))
    keep = np.ones(len(xx), bool)
    for _ in range(6):
        coefficients = np.linalg.lstsq(matrix[keep], values[keep], rcond=None)[0]
        errors = np.abs(values - matrix @ coefficients).max(axis=1)
        new_keep = errors <= max(5, float(np.percentile(errors[keep], 90)))
        if new_keep.sum() < 60:
            break
        keep = new_keep
    # A minority color cannot explain a whole surface. Preserve the original
    # allowed sample population for quality measurement, including outliers.
    constant = {"kind": "solid", "colors": [np.median(values[keep], axis=0).tolist()], "offsets": [0]}
    candidates = [constant]
    pixel_derivatives = coefficients[1:].copy()
    pixel_derivatives[0] /= max(1, width - 1)
    pixel_derivatives[1] /= max(1, height - 1)
    _, singular, axes = np.linalg.svd(pixel_derivatives.T, full_matrices=False)
    direction = axes[0]
    dominant = int(np.argmax(np.abs(direction)))
    if direction[dominant] < 0:
        direction = -direction
    directions = [direction, np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                  np.array([2 ** -0.5, 2 ** -0.5]), np.array([-2 ** -0.5, 2 ** -0.5])]
    for offsets in ([0.0, 1.0], [0.0, 0.5, 1.0]):
        for axis in directions:
            dx, dy = axis
            projected = xx * dx + yy * dy
            corners = [0, (width - 1) * dx, (height - 1) * dy,
                       (width - 1) * dx + (height - 1) * dy]
            start, end = min(corners), max(corners)
            t = (projected - start) / max(1e-6, end - start)
            basis = np.stack([np.interp(t, offsets, np.eye(len(offsets))[i]) for i in range(len(offsets))], axis=1)
            colors = np.linalg.lstsq(basis, values, rcond=None)[0]
            candidates.append({"kind": "linear", "colors": np.clip(colors, 0, 255).tolist(),
                               "offsets": offsets, "direction": axis.tolist(),
                               "angle_deg": float(math.degrees(math.atan2(dy, dx)) % 360)})
    results = []
    for model in candidates:
        prediction = gradient_pixels(model, bbox)[yy, xx].astype(np.float64)
        differences = np.abs(prediction - values)
        score = {"mae": float(differences.mean()), "p95": float(np.percentile(differences, 95)),
                 "sample_count": len(xx), "inlier_fraction": float(keep.mean())}
        results.append((model, score))
        if score["mae"] <= 2 and score["p95"] <= 5:
            return model, score
    # Off-center highlights need a source-fitted middle stop. A fixed 50%
    # stop cannot describe a short fade followed by a long uniform color bar.
    for middle in (0.1, 0.2, 0.35, 0.65, 0.8, 0.9):
        for axis in directions[:3]:
            dx, dy = axis
            projected = xx * dx + yy * dy
            corners = [0, (width - 1) * dx, (height - 1) * dy,
                       (width - 1) * dx + (height - 1) * dy]
            t = (projected - min(corners)) / max(1e-6, max(corners) - min(corners))
            offsets = [0.0, middle, 1.0]
            basis = np.stack([np.interp(t, offsets, np.eye(3)[i]) for i in range(3)], axis=1)
            colors = np.linalg.lstsq(basis, values, rcond=None)[0]
            model = {"kind": "linear", "colors": np.clip(colors, 0, 255).tolist(), "offsets": offsets,
                     "direction": axis.tolist(), "angle_deg": float(math.degrees(math.atan2(dy, dx)) % 360)}
            differences = np.abs(gradient_pixels(model, bbox)[yy, xx].astype(np.float64) - values)
            score = {"mae": float(differences.mean()), "p95": float(np.percentile(differences, 95)),
                     "sample_count": len(xx), "inlier_fraction": float(keep.mean())}
            results.append((model, score))
            if score["mae"] <= 2 and score["p95"] <= 5:
                return model, score
    model, score = min(results, key=lambda item: item[1]["mae"])
    return model, score


def native_fill(model):
    properties = {"fill": color_hex(model["colors"][0])}
    if model["kind"] == "linear":
        properties["fill_gradient"] = {
            "angle_deg": model["angle_deg"],
            "stops": [{"offset": round(offset * 100000), "color": color_hex(color)}
                      for offset, color in zip(model["offsets"], model["colors"])]}
    return properties


def precise_text_masks(source, line, protected):
    """Color-consistent glyph seeds, bounded expansion and an independent ROI."""
    size = (source.shape[1], source.shape[0])
    scale = size[0] / 1672
    margin = max(2, round(3 * scale))
    x, y, width, height = line["bbox"]
    box = [max(0, x - margin), max(0, y - margin),
           min(size[0], x + width + margin) - max(0, x - margin),
           min(size[1], y + height + margin) - max(0, y - margin)]
    roi = shape_mask(size, box) & ~protected
    polygon = line.get("ocr_polygon")
    # A narrowed reviewed bbox supersedes a historic polygon that spans labels.
    if polygon and any(not (x - 2 <= px <= x + width + 2 and y - 2 <= py <= y + height + 2)
                       for px, py in polygon):
        polygon = None
    bounds = shape_mask(size, line["bbox"], polygon=polygon)
    foreground = np.array([int(line["style"]["color"][i:i + 2], 16) for i in (0, 2, 4)])
    distance = np.abs(source.astype(np.int16) - foreground).max(axis=2)
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
    contrast = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), max(1, height / 10)))
    core = bounds & roi & (distance <= 65) & (contrast >= 12)
    weak = roi & (distance <= 110) & (contrast >= 3)
    # Grow from the actual core only one pixel at a time. Unbounded connected
    # component propagation can swallow every inter-character gap in a row.
    glyph = core.copy()
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(max(1, min(2, round(height * 0.04)))):
        glyph |= cv2.dilate(glyph.astype(np.uint8), kernel).astype(bool) & weak
    glyph = cv2.dilate(glyph.astype(np.uint8), kernel).astype(bool) & roi
    authorized = cv2.dilate(glyph.astype(np.uint8), kernel).astype(bool) & roi
    density = float(glyph[bounds].mean()) if bounds.any() else 0
    status = "PASS" if core.any() and density <= 0.8 else "REVIEW_REQUIRED"
    return {"glyph": glyph, "core": core, "repair": authorized, "inspection": roi,
            "status": status, "coverage": density, "core_pixels": int(core.sum())}


def inspect_residue(source, rendered, core, inspection):
    """Independent ROI edges include source strokes omitted by a repair mask."""
    if not core.any():
        return {"status": "REVIEW_REQUIRED", "reason": "no_original_glyph_evidence"}
    original = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
    result = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY).astype(np.float32)
    signal = np.abs(result - cv2.GaussianBlur(result, (0, 0), 2))
    old_signal = np.abs(original - cv2.GaussianBlur(original, (0, 0), 2))
    checked = core & inspection & (old_signal > 12)
    if not checked.any():
        return {"status": "REVIEW_REQUIRED", "reason": "insufficient_independent_stroke_evidence"}
    fraction = float((signal[checked] > 6).mean())
    return {"status": "PASS" if fraction <= 0.01 else "REVIEW_REQUIRED",
            "residual_fraction": fraction, "checked_pixels": int(checked.sum())}


def edge_profiles(source, rendered, bbox):
    """Measure edge positions on four straight contour sections in source pixels."""
    x, y, width, height = map(int, bbox)
    height_limit, width_limit = source.shape[:2]
    result = {}
    for side in ("top", "bottom", "left", "right"):
        original, actual = [], []
        for offset in range(-4, 5):
            if side in {"top", "bottom"}:
                row = min(height_limit - 1, max(0, y + offset if side == "top" else y + height - 1 + offset))
                columns = slice(x + width // 4, x + width * 3 // 4)
                original.append(np.median(source[row, columns], axis=0))
                actual.append(np.median(rendered[row, columns], axis=0))
            else:
                column = min(width_limit - 1, max(0, x + offset if side == "left" else x + width - 1 + offset))
                rows = slice(y + height // 4, y + height * 3 // 4)
                original.append(np.median(source[rows, column], axis=0))
                actual.append(np.median(rendered[rows, column], axis=0))
        src_energy = np.abs(np.diff(np.asarray(original), axis=0)).max(axis=1)
        out_energy = np.abs(np.diff(np.asarray(actual), axis=0)).max(axis=1)
        if max(src_energy) < 3 or max(out_energy) < 3:
            result[side] = {"status": "LOW_CONTRAST", "delta": 0.0}
            continue
        src_weights = src_energy * (src_energy >= max(src_energy) * 0.5)
        out_weights = out_energy * (out_energy >= max(out_energy) * 0.5)
        delta = float((src_weights * np.arange(8)).sum() / src_weights.sum()
                      - (out_weights * np.arange(8)).sum() / out_weights.sum())
        result[side] = {"status": "MEASURED", "delta": delta}
    return result


def repair_seam_quality(before, after, authorized):
    increases = []
    for axis in (0, 1):
        edges = np.diff(authorized.astype(np.int8), axis=axis) != 0
        prior = np.abs(np.diff(before.astype(np.float32), axis=axis)).max(axis=2)
        current = np.abs(np.diff(after.astype(np.float32), axis=axis)).max(axis=2)
        increases.extend(np.maximum(current[edges] - prior[edges], 0).tolist())
    p95 = float(np.percentile(increases, 95)) if increases else 0.0
    return {"boundary_samples": len(increases), "new_jump_p95": p95,
            "status": "PASS" if p95 <= 3 else "REVIEW_REQUIRED"}
