"""Source-fitted basic-shape icon candidates with conservative PNG fallback."""
from __future__ import annotations

import cv2
import numpy as np


def organization_primitives(parameters):
    top_x, top_y, top_r, left_x, right_x, lower_y, lower_r, bar_y, stroke = parameters
    circles = [(top_x, top_y, top_r), (left_x, lower_y, lower_r), (right_x, lower_y, lower_r)]
    segments = [(top_x, top_y + top_r, top_x, bar_y),
                (left_x, bar_y, right_x, bar_y),
                (left_x, bar_y, left_x, lower_y - lower_r),
                (right_x, bar_y, right_x, lower_y - lower_r)]
    return circles, segments, stroke


def organization_mask(parameters, size):
    scale = 4
    image = np.zeros((size[1] * scale, size[0] * scale), np.uint8)
    circles, segments, stroke = organization_primitives(parameters)
    for x, y, radius in circles:
        cv2.circle(image, (round(x * scale), round(y * scale)), round(radius * scale), 255,
                   max(1, round(stroke * scale)), cv2.LINE_AA)
    for x1, y1, x2, y2 in segments:
        cv2.line(image, (round(x1 * scale), round(y1 * scale)), (round(x2 * scale), round(y2 * scale)),
                 255, max(1, round(stroke * scale)), cv2.LINE_AA)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA) >= 128


def contour_metrics(reference, candidate):
    union = (reference | candidate).sum()
    iou = float((reference & candidate).sum() / union) if union else 0
    kernel = np.ones((3, 3), np.uint8)
    reference_edge = reference & ~cv2.erode(reference.astype(np.uint8), kernel).astype(bool)
    candidate_edge = candidate & ~cv2.erode(candidate.astype(np.uint8), kernel).astype(bool)
    if not reference_edge.any() or not candidate_edge.any():
        return {"iou": iou, "contour_p95": float("inf")}
    distance = cv2.distanceTransform((~reference_edge).astype(np.uint8), cv2.DIST_L2, 3)
    reverse = cv2.distanceTransform((~candidate_edge).astype(np.uint8), cv2.DIST_L2, 3)
    return {"iou": iou, "contour_p95": float(np.percentile(np.concatenate((distance[candidate_edge], reverse[reference_edge])), 95))}


def fit_organization(source_crop):
    height, width = source_crop.shape[:2]
    # The reviewed symbol is blue line art. White image-edge halos are checked
    # by the actual-color render gate rather than counted as blue contour ink.
    reference = ((source_crop[:, :, 2].astype(int) - source_crop[:, :, 0].astype(int)) > 70)
    reference &= (source_crop[:, :, 2].astype(int) - source_crop[:, :, 1].astype(int)) > 25
    parameters = np.array([32, 10.5, 6.7, 10.5, 52.5, 46.5, 6.8, 28, 3.1])
    parameters[:8] *= width / 64
    parameters[8] *= width / 64
    best = contour_metrics(reference, organization_mask(parameters, (width, height)))
    for step in (1.0, 0.5, 0.25):
        for index in range(len(parameters)):
            original = parameters[index]
            winner = original
            for offset in (-2, -1, 0, 1, 2):
                trial = parameters.copy()
                trial[index] = original + offset * step
                if trial[index] <= 0:
                    continue
                score = contour_metrics(reference, organization_mask(trial, (width, height)))
                if score["iou"] > best["iou"]:
                    best = score
                    winner = trial[index]
            parameters[index] = winner
    return parameters, best, reference


def native_organization(node_id, bbox, parameters, color):
    x, y, _, _ = bbox
    circles, segments, stroke = organization_primitives(parameters)
    result = []
    # Native white under-strokes reproduce the source halo without an image.
    for layer, ink, thickness in (("halo", "FFFFFF", stroke + 1.5), ("ink", color, stroke)):
        for index, (cx, cy, radius) in enumerate(circles):
            result.append({"id": f"{node_id}-{layer}-circle-{index}", "type": "shape", "geometry": "ellipse",
                           "bbox": [x + cx - radius, y + cy - radius, radius * 2, radius * 2],
                           "fill": "none", "line": {"color": ink, "width_pt": thickness * 0.75}, "evidence_ids": []})
        for index, (x1, y1, x2, y2) in enumerate(segments):
            left = min(x1, x2) - thickness / 2
            top = min(y1, y2) - thickness / 2
            result.append({"id": f"{node_id}-{layer}-line-{index}", "type": "shape", "geometry": "roundRect",
                           "bbox": [x + left, y + top, abs(x2 - x1) + thickness, abs(y2 - y1) + thickness],
                           "fill": ink, "corner_radius_px": thickness / 2, "evidence_ids": []})
    return result
