"""Measure a circular badge and separate its foreground symbol from the disc."""

import cv2
import numpy as np


def _faces(source, box, *, allow_overhang=False):
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
            0 < bx and (allow_overhang or bx <= margin)
            and 0 < by and (allow_overhang or by <= margin)
            and (allow_overhang or bx + bw >= margin + w)
            and (allow_overhang or by + bh >= margin + h)
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
        yield left, top, [left + bx, top + by, bw, bh], radius, face, paper


def circular_badge(source, plate, box):
    x, y, w, h = map(round, box)
    crop = source[y:y+h, x:x+w].astype(float)
    for left, top, bounds, radius, face, paper in _faces(source, box):
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
        patch = plate[top:top+radius.shape[0], left:left+radius.shape[1], :3]
        erase = radius <= 1.07
        if (
            np.minimum(abs(patch - face).max(2), abs(patch - paper).max(2))[erase].max()
            > 18
        ):
            continue
        return (
            bounds,
            bytes(np.rint(face).astype("uint8")).hex(),
            rgba,
            (left, top, erase, paper),
        )


def measure_symbol(source, box):
    """Find the complete symbol against its own disc, including enclosed holes.

    The detector locates the object. The source silhouette determines where to
    stop. Ink reaching the rim or another unaccounted-for mark prevents staging.
    """
    from .region_crop import _extract_rgba

    for left, top, disc, radius, face, paper in _faces(source, box, allow_overhang=True):
        x, y, w, h = disc
        tile = source[y:y+h, x:x+w]
        radial = radius[y-top:y-top+h, x-left:x-left+w]
        edge = (radial > .80) & (radial < .94)
        # Sample the same physical rim regardless of which detector located it.
        face = np.median(tile[edge], axis=0)
        delta = np.max(np.abs(tile - face), axis=2)
        noise = float(np.percentile(delta[edge], 95))
        if noise > 6 or np.any(delta[edge] > 15):
            continue
        inside = radial < .94
        support = inside & (delta > max(4, noise + 2))
        strong = inside & (delta > 30)
        if not .015 < strong.sum() / inside.sum() < .5:
            continue
        # Only components visibly connected to source detail belong to the symbol.
        _, labels = cv2.connectedComponents(support.astype('uint8'), connectivity=8)
        ink = np.isin(labels, np.unique(labels[strong])) & support
        if np.any(inside & ~ink & (delta > 12)):
            continue
        yy, xx = np.nonzero(ink)
        if not xx.size:
            continue
        ix, iy, iw, ih = cv2.boundingRect(ink.astype('uint8'))
        bx, by, bw, bh = box
        contained = ((xx+x >= bx-1) & (xx+x < bx+bw+1)
                     & (yy+y >= by-1) & (yy+y < by+bh+1))
        if contained.mean() < .98:
            continue
        backing = np.broadcast_to(face, tile.shape)
        rgba = _extract_rgba(tile, backing, ink)[iy:iy+ih, ix:ix+iw]
        return {'disc':disc, 'bbox':[x+ix,y+iy,iw,ih], 'face':bytes(face.round().astype('uint8')).hex(),
                'mask':ink[iy:iy+ih,ix:ix+iw], 'rgba':rgba}


def measure_flat_symbol(source, box):
    """Measure a sparse graphic on continuous paper. A detector box alone owns no pixels."""
    from .region_crop import _extract_rgba

    x, y, w, h = map(round, box)
    pad = max(2, round(min(w, h) * .12))
    if min(w, h) < 5 or min(x, y) < pad or x+w+pad > source.shape[1] or y+h+pad > source.shape[0]:
        return None
    tile = source[y-pad:y+h+pad, x-pad:x+w+pad]
    ring = np.ones(tile.shape[:2], bool)
    ring[pad:-pad, pad:-pad] = False
    face = np.rint(np.median(tile[ring], axis=0))
    delta = abs(tile - face).max(2)
    noise = float(np.percentile(delta[ring], 95))
    if noise > 5 or delta[ring].max() > 12:
        return None
    delta = delta[pad:-pad, pad:-pad]
    strong = delta > 30
    if not .015 < strong.mean() < .55:
        return None
    _, labels = cv2.connectedComponents((delta > max(3, noise+2)).astype('uint8'), connectivity=8)
    ink = (labels != 0) & np.isin(labels, np.unique(labels[strong]))
    if np.any(~ink & (delta > 12)):
        return None
    crop = tile[pad:-pad, pad:-pad]
    # A retained label, screenshot or photo is not a line icon. Require simple
    # source colours on the strong strokes, allowing antialiasing towards paper.
    strokes = crop[strong].astype(float) - face
    directions = strokes / np.linalg.norm(strokes, axis=1, keepdims=True)
    direction = np.median(directions, axis=0)
    if np.percentile(np.linalg.norm(directions - direction, axis=1), 95) > .22:
        return None
    ix, iy, iw, ih = cv2.boundingRect(ink.astype('uint8'))
    rgba = _extract_rgba(crop, np.broadcast_to(face, crop.shape), ink)
    return {'bbox':[x+ix, y+iy, iw, ih], 'face':bytes(face.astype('uint8')).hex(),
            'mask':ink[iy:iy+ih, ix:ix+iw], 'rgba':rgba[iy:iy+ih, ix:ix+iw]}


def bind_layer_foreground(directory, layers, source, source_sha):
    """Carry a verified two-layer page's foreground through incremental scene copies."""
    from pathlib import Path
    from PIL import Image
    from ..object_extract import load_page
    from ..materialize import digest, write

    page = load_page(layers)
    if len(page.layers) != 2:
        return
    with Image.open(page.source_path) as original:
        if not np.array_equal(np.asarray(original.convert("RGB")), source):
            return
    layer = page.layers[-1]
    write(
        Path(directory) / "badge-layer.json",
        {
            "source_sha256": source_sha,
            "path": str(layer.path.resolve()),
            "sha256": digest(layer.path),
        },
    )


def select_badge_ink(pixels, face, measured, layer, *, excluded=(), strict_holes=False):
    """Compare against source on the disc and check exposed holes/ink on a dark fill."""
    face = np.array(list(bytes.fromhex(face)), dtype=float)
    measured_alpha = measured[..., 3] / 255.0
    holes = measured_alpha < 0.06
    strokes = measured_alpha > 0.65
    trials = []
    best = measured
    selected = "color-unmix"
    best_score = float("inf")
    for name, candidate in (("color-unmix", measured), ("LayerD", layer)):
        if name in excluded:
            trials.append({"method": name, "status": "excluded-by-office"})
            continue
        if candidate is None:
            trials.append({"method": name, "status": "unavailable"})
            continue
        alpha = candidate[..., 3:4] / 255.0
        rgb = candidate[..., :3].astype(float)
        original = rgb * alpha + face * (1 - alpha)
        dark = rgb * alpha + 56 * (1 - alpha)
        source_error = float(np.abs(original - pixels).max(2).mean())
        spill = float(np.abs(dark[holes] - 56).max(1).mean()) if holes.any() else 0.0
        coverage = float(alpha[..., 0][strokes].mean()) if strokes.any() else 0.0
        valid = coverage >= 0.55 and spill <= 6 and source_error <= 12
        # Low-alpha edge pixels are visible antialiasing, even on a dark fill.
        # Test the backdrop only at pixels that the source footprint leaves empty.
        empty = measured[..., 3] == 0
        if strict_holes and empty.any():
            for background in (48, 230):
                contrast = rgb[empty] * alpha[empty] + background * (1 - alpha[empty])
                valid = valid and np.percentile(abs(contrast-background).max(1), 95) <= 8
            valid = valid and not np.any(alpha[..., 0][empty] > .06)
        score = source_error + spill
        trials.append(
            {
                "method": name,
                "status": "eligible" if valid else "rejected",
                "source_error": source_error,
                "dark_hole_error": spill,
                "stroke_coverage": coverage,
            }
        )
        if valid and score < best_score:
            best_score, selected, best = score, name, candidate
    return (best if np.isfinite(best_score) else None), {
        "selected": selected if np.isfinite(best_score) else "opaque-crop",
        "candidates": trials,
    }
