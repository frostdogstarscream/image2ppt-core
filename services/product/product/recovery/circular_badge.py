"""Separate a sparse, single-colour symbol from a measured circular badge."""

import cv2
import numpy as np


def circular_badge(source, plate, box):
    x, y, w, h = map(round, box)
    if not (12 <= min(w, h) and max(w, h) <= 2 * min(w, h)):
        return
    margin = round(max(w, h) * 0.55)
    left, top = x - margin, y - margin
    if (
        min(left, top) < 0
        or x + w + margin > source.shape[1]
        or y + h + margin > source.shape[0]
    ):
        return
    tile = source[top : y + h + margin, left : x + w + margin].astype(float)
    crop = tile[margin : margin + h, margin : margin + w]
    face = np.median(crop.reshape(-1, 3), axis=0)
    near = np.max(np.abs(tile - face), axis=2) < 5
    contours, _ = cv2.findContours(
        near.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        bx, by, bw, bh = cv2.boundingRect(contour)
        if not (
            0 < bx <= margin
            and 0 < by <= margin
            and bx + bw >= margin + w
            and by + bh >= margin + h
            and bx + bw < tile.shape[1]
            and by + bh < tile.shape[0]
            and 0.9 < bw / bh < 1.1
        ):
            continue
        yy, xx = np.indices(tile.shape[:2])
        radius = ((xx - bx - (bw - 1) / 2) / (bw / 2)) ** 2 + (
            (yy - by - (bh - 1) / 2) / (bh / 2)
        ) ** 2
        observed = np.zeros(near.shape, "uint8")
        cv2.drawContours(observed, [contour], -1, 1, -1)
        disc = radius <= 1
        if (disc & observed.astype(bool)).sum() / (
            disc | observed.astype(bool)
        ).sum() < 0.95:
            continue
        ring = (radius > 1.12) & (radius < 1.30)
        if not ring.any():
            continue
        paper = np.median(tile[ring], axis=0)
        if (
            np.percentile(abs(tile[ring] - paper).max(1), 95) > 5
            or abs(face - paper).max() < 6
        ):
            continue
        diff = crop - face
        strong = np.linalg.norm(diff, axis=2) > 65
        if not 0.025 < strong.mean() < 0.55:
            continue
        ink = np.median(crop[strong], axis=0)
        direction = ink - face
        alpha = np.clip((diff @ direction) / max(1, direction @ direction), 0, 1)
        if (
            np.percentile(abs(diff - alpha[:, :, None] * direction).max(2)[strong], 95)
            > 20
        ):
            continue
        alpha[alpha < 0.06] = 0
        rgba = np.empty((h, w, 4), "uint8")
        rgba[:, :, :3] = np.rint(ink)
        rgba[:, :, 3] = np.rint(alpha * 255)
        patch = plate[top : y + h + margin, left : x + w + margin, :3]
        erase = radius <= 1.07
        if (
            np.minimum(abs(patch - face).max(2), abs(patch - paper).max(2))[erase].max()
            > 18
        ):
            continue
        return (
            [left + bx, top + by, bw, bh],
            bytes(np.rint(face).astype("uint8")).hex(),
            rgba,
            (left, top, erase, paper),
        )
