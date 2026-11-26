"""Estimate the backing from clean spaces within an OCR field on a smooth panel."""

import cv2
import numpy as np

from .background_adapter import RecoveryBlocked


def _sample_rows(source, box, protected):
    x, y, w, h = box
    top, bottom = max(0, y - 10), min(source.shape[0], y + h + 10)
    rgb = source[top:bottom, x : x + w].astype(float)
    xx = np.linspace(-1, 1, w)
    design = np.stack([np.ones(w), xx, xx * xx], axis=1)
    # Each band must contain multiple independently observed clear scanlines.
    # Reject a row with foreign ink, an icon edge or an abrupt change of surface.
    rows = {}
    for row, pixels in enumerate(rgb):
        legal = ~protected[top + row, x : x + w]
        train = legal & (np.arange(w) % 3 != 1)
        test = legal & ~train
        if train.sum() < max(9, w * 0.45) or test.sum() < 4:
            continue
        curve = design @ np.linalg.lstsq(design[train], pixels[train], rcond=None)[0]
        errors = np.abs(curve - pixels).max(1)
        if np.percentile(errors[test], 95) <= 4 and errors[legal].max() <= 13:
            rows[top + row] = curve
    before = [n for n in rows if n - 1 in rows and n <= y + 0.25 * h]
    after = [n for n in rows if n + 1 in rows and n >= y + 0.75 * h]
    if not before or not after:
        raise RecoveryBlocked("INSUFFICIENT_CONTEXT", "no verified clear rows bracketing text")
    a, b = max(before), min(after)
    slope = (rows[b] - rows[a]) / (b - a)
    if np.abs(slope).max() > 3:
        raise RecoveryBlocked("BACKGROUND_REPAIR_FAILED", "bracketing rows belong to different surfaces")
    errors = []
    for pos in (a - 1, b + 1):
        samples = source[pos, x : x + w][~protected[pos, x : x + w]]
        estimate = (rows[a] + (pos - a) * slope)[~protected[pos, x : x + w]]
        errors.extend(np.abs(estimate - samples).max(1))
    ground = np.clip(np.rint(rows[a][None, :, :] + (np.arange(y, y + h) - a)[:, None, None] * slope), 0, 255)
    return ground, np.asarray(errors)


def _fit_field(source, box, protected, color, glyph_height, *, line_polygons, row_samples=False):
    x, y, w, h = box
    if x < 0 or y < 0 or x + w > source.shape[1] or y + h > source.shape[0]:
        raise RecoveryBlocked("MASK_INCOMPLETE", "text field crosses source boundary")
    rgb = source[y : y + h, x : x + w].astype(np.float32)
    blocked = protected[y : y + h, x : x + w]
    own = np.zeros((h, w), np.uint8)
    for polygon in line_polygons or ():
        points = np.asarray(polygon, dtype=float) - [x, y]
        # OCR returns corner coordinates; sample pixel centres. In particular
        # a rectangular [x,x+w] field ends before column x+w.
        if len(np.unique(points[:, 0])) == len(np.unique(points[:, 1])) == 2:
            left, top = np.maximum(0, np.ceil(points.min(0) - 0.5).astype(int))
            right, bottom = np.maximum(0, np.ceil(points.max(0) - 0.5).astype(int))
            own[top:bottom, left:right] = 1
        else:
            cv2.fillPoly(own, [np.rint((points - 0.5) * 16).astype(np.int32)], 1, shift=4)
    if not own.any():
        raise RecoveryBlocked("OWNER_CONFLICT", "missing source OCR polygon for this field")
    own = own.astype(bool)
    fg = np.array(tuple(bytes.fromhex(color)), dtype=np.float32)
    # Strokes approach their foreground colour. Spaces between strokes are
    # farther from that colour, including white lettering on a coloured card.
    distance = np.linalg.norm(rgb - fg, axis=2)
    radius = min(13, max(4, round(glyph_height * 0.30)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    local_ground = cv2.morphologyEx(distance, cv2.MORPH_CLOSE, kernel)
    # Select flat paper samples away from the visible glyph cores and their
    # compression halo. A quadratic colour surface allows a gentle gradient
    # both along and across a paragraph.
    provisional = (local_ground - distance > 42).astype(np.uint8)
    clearance = cv2.distanceTransform(1 - provisional, cv2.DIST_L2, 5)
    yy, xx = np.indices((h, w))
    nx, ny = (xx + 0.5) / w, (yy + 0.5) / h
    design = np.stack([np.ones_like(nx), nx, ny, nx * ny, nx * nx, ny * ny], axis=2)
    samples = (clearance > min(8, max(4, radius))) & ~blocked
    if samples.sum() < 150:
        # Short, dense labels offer fewer gaps. Retain three clear pixels
        # between an anchor and any strong printed stroke, then validate it
        # against the same independent error measurements as a long line.
        samples = (clearance > 3.5) & ~blocked
    holdout = samples & ((xx + 2 * yy) % 5 == 0)
    train = samples & ~holdout
    if row_samples:
        ground, validation_errors = _sample_rows(source, box, protected)
    else:
        if train.sum() < 24 or holdout.sum() < 8:
            raise RecoveryBlocked("INSUFFICIENT_CONTEXT", "too few clean spaces in OCR field")
        for _ in range(4):
            coeff, _, rank, _ = np.linalg.lstsq(design[train], rgb[train], rcond=None)
            if rank != 6:
                raise RecoveryBlocked("INSUFFICIENT_CONTEXT", "clean samples do not span the field")
            ground = np.clip(np.rint(design @ coeff), 0, 255)
            error = np.abs(rgb - ground).max(2)
            med = np.median(error[train])
            keep = train & (error <= max(4, med + 3 * np.median(np.abs(error[train] - med))))
            if keep.sum() < 24 or np.array_equal(keep, train):
                break
            train = keep
        validation_errors = np.abs(rgb - ground).max(2)[holdout]
    delta = rgb - ground
    err = np.abs(delta).max(2)
    p95, p99 = map(float, np.percentile(validation_errors, [95, 99]))
    if p95 > 8 or p99 > 13:
        raise RecoveryBlocked("BACKGROUND_REPAIR_FAILED", "clean interletter samples disagree")
    direction = fg - ground
    alpha = (delta * direction).sum(2) / np.maximum(1, (direction * direction).sum(2))
    mix_error = np.abs(delta - alpha[:, :, None] * direction).max(2)
    stroke = (alpha >= 0.26) & (alpha <= 1.4) & (mix_error < 40) & (err > 30)
    # White or coloured emphasis may be rendered with a darker outline.
    # Recognise the second ink only on a light panel and when the field
    # already has independently confirmed foreground of its declared colour.
    if stroke.sum() >= 5 and np.percentile(ground, 5) > 208 and fg.mean() < 175:
        blue = (rgb[:, :, 2] - rgb[:, :, 0] > 48) & (rgb[:, :, 2] > rgb[:, :, 1] + 10)
        dark = (rgb.max(2) < 112) & (np.ptp(rgb, axis=2) < 80)
        stroke |= (blue | dark) & (err > 54)
    if (stroke & own).sum() < 5 or (stroke & blocked & own).any():
        raise RecoveryBlocked("PROTECTED_NEIGHBOR", "source strokes missing or overlap a peer")
    # Ignore independently protected neighbouring lines in the search margin.
    # A connected stroke crossing from the field into a protected owner still blocks it.
    _, components = cv2.connectedComponents(stroke.astype(np.uint8), connectivity=8)
    owned_labels = np.unique(components[own & stroke])
    stroke &= np.isin(components, owned_labels)
    if (stroke & blocked).any():
        raise RecoveryBlocked("PROTECTED_NEIGHBOR", "connected stroke crosses into a protected owner")
    solid = stroke & (alpha > 0.75) & (mix_error < 16)
    thickness = cv2.distanceTransform(solid.astype("uint8"), cv2.DIST_L2, 5)
    if thickness.max() > max(4, glyph_height * 0.4):
        raise RecoveryBlocked(
            "MASK_INCOMPLETE", "foreground contains a solid object larger than a printed stroke"
        )
    near = cv2.distanceTransform((~stroke).astype("uint8"), cv2.DIST_L2, 5)
    # Compression fringe and drop shadows must be attached to source ink.
    fringe = ((alpha > 0) & (mix_error <= 24)) | (err <= 30) | ((near <= 3) & (err <= 80))
    floor = 0 if validation_errors.max() == 0 and np.ptp(ground, axis=(0, 1)).max() == 0 else 3
    ink = (err > floor) & (stroke | (fringe & (near <= max(8, radius))))
    # Search margins extend past the OCR polygon. Exclude unrelated context
    # there unless a component is attached to the field's own measured ink.
    _, components = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    ink &= own | np.isin(components, np.unique(components[stroke]))
    noise = max(6.0, p99 + 2)
    clean_gap = (err <= noise) & (near > 3) & ~ink
    if ((err > 6) & ~ink & own & ~blocked & ~clean_gap).any():
        raise RecoveryBlocked("MASK_INCOMPLETE", "unexplained contrast remains outside corroborated strokes")
    mask = np.zeros(source.shape[:2], bool)
    mask[y : y + h, x : x + w] = ink & ~blocked
    reference = source.copy()
    reference[y : y + h, x : x + w] = ground.astype(np.uint8)
    # The reference outside the owned OCR field is the observed context;
    # no estimate or write is needed there except the supported glyph fringe.
    reference[y : y + h, x : x + w][~own & ~ink] = rgb[~own & ~ink]
    return (
        reference,
        mask,
        {
            "gap_noise_rgb": noise,
            "method": "interletter",
            "validation_p95_rgb": p95,
            "validation_pixels": len(validation_errors),
            "stroke_pixels": int(stroke.sum()),
            "sampling": "clear_rows" if row_samples else "interletter_spaces",
        },
    )


def _field_evidence(source, box, protected, color, glyph_height, *, line_polygons):
    try:
        return _fit_field(source, box, protected, color, glyph_height, line_polygons=line_polygons)
    except RecoveryBlocked as exc:
        if exc.code not in {"BACKGROUND_REPAIR_FAILED", "MASK_INCOMPLETE", "INSUFFICIENT_CONTEXT"}:
            raise
    try:
        return _fit_field(
            source, box, protected, color, glyph_height, line_polygons=line_polygons, row_samples=True
        )
    except RecoveryBlocked as exc:
        if exc.code == "PROTECTED_NEIGHBOR":
            raise
    # Small fields can have too few clean spaces. Re-estimate on the adjacent
    # actual paper. Ownership still comes from measured, connected strokes.
    x, y, w, h = box
    left, top = max(0, x - 7), max(0, y - 7)
    right, bottom = min(source.shape[1], x + w + 7), min(source.shape[0], y + h + 7)
    reference, mask, fit = _fit_field(
        source,
        [left, top, right - left, bottom - top],
        protected,
        color,
        glyph_height,
        line_polygons=line_polygons,
    )
    # Extend only to individually corroborated stroke/halo pixels. The stage
    # recomputes the same geometry and checks every previous owner's mask.
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise RecoveryBlocked("MASK_INCOMPLETE", "empty field")
    x0, y0 = min(x, int(xx.min())), min(y, int(yy.min()))
    fit["write_bbox"] = [x0, y0, max(x + w, int(xx.max()) + 1) - x0, max(y + h, int(yy.max()) + 1) - y0]
    fit["extra_sampling_margin"] = 7
    return reference, mask, fit


def interletter_backing(source, box, protected, color, glyph_height, *, line_polygons):
    reference, mask, fit = _field_evidence(
        source, box, protected, color, glyph_height, line_polygons=line_polygons
    )
    x, y, w, h = fit.get("write_bbox", box)
    window = np.s_[max(0, y - 3) : y + h + 3, max(0, x - 3) : x + w + 3]
    active = mask[window]
    observed = source[window]
    # Match the real paper at the edge of each repair. A polynomial measured
    # in thin letter spaces can extrapolate badly at the corner of a blue card.
    image = observed.astype(np.float32)
    image[active] = reference[window][active]
    legal = (~protected[window]).astype(np.float32)
    stencil = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], np.float32)
    weights = cv2.filter2D(legal, -1, stencil)
    if (weights[active] == 0).any():
        raise RecoveryBlocked("INSUFFICIENT_CONTEXT", "isolated writing site has no legal neighbour")
    for iteration in range(360):
        updated = cv2.filter2D(image * legal[:, :, None], -1, stencil) / np.maximum(weights[:, :, None], 0.25)
        step = np.max(np.abs(updated[active] - image[active]))
        image[active] = updated[active]
        if step < 0.04:
            break
    reference[window][active] = np.clip(np.rint(image[active]), 0, 255).astype(np.uint8)
    fit["boundary_interpolation_iterations"] = iteration + 1
    fit["boundary_interpolation_last_step"] = round(float(step), 4)
    return reference, mask, fit


def source_polygons(directory, graph, text):
    import json

    inventory = directory / "runtime-ocr-protection.json"
    if not inventory.is_file():
        raise RecoveryBlocked("OWNER_CONFLICT", "missing source OCR inventory")
    evidence = json.loads(inventory.read_text("utf8"))
    if evidence["source_sha256"] != graph.source_sha256:
        raise RecoveryBlocked("SOURCE_BINDING", "OCR inventory belongs to a different source")
    members = set(graph.text.members.get(text.id, []))
    polygons = [
        line["polygon"] for line in evidence["lines"] if line["id"] in members and line.get("polygon")
    ]
    if len(polygons) != len(members) or not polygons:
        raise RecoveryBlocked("OWNER_CONFLICT", "incomplete source OCR geometry")
    return polygons


def field_trial(graph, text, directory, trials):
    from PIL import Image
    from ..materialize import digest
    from ..parallel_contracts import EDITABLE_TEXT_METHOD
    from ..parallel_store import key
    from ..parallel_text_candidates import peer_protection
    from .text_mask_completion import text_search_box
    from .diagnostics import save_trial

    if (
        graph.text.method_version != EDITABLE_TEXT_METHOD
        or graph.ownership.get(text.id) != "RESOLVED"
        or digest(directory / "source.png") != graph.source_sha256
    ):
        raise RecoveryBlocked("OWNER_CONFLICT", "source or text ownership unverified")
    with Image.open(directory / "source.png") as image:
        source = np.array(image.convert("RGB"))
    style = next(s for s in graph.text.styles if s.id == text.style)
    box = text_search_box(graph, text, source)
    protected = peer_protection(
        text, graph, source, np.zeros(source.shape[:2], bool), baseline_directory=directory
    )
    reference, mask, fit = interletter_backing(
        source,
        box,
        protected,
        style.color,
        style.size_px,
        line_polygons=source_polygons(directory, graph, text),
    )
    return save_trial(
        trials / "interletter",
        source,
        mask,
        reference,
        protected,
        fit.get("write_bbox", box),
        reference,
        fit,
        {
            "candidate": "interletter",
            "method": "interletter",
            "mask_variant": "corroborated_strokes",
            "object_id": text.id,
            "source_sha256": graph.source_sha256,
            "scene_graph_sha256": key(graph.model_dump()),
        },
    )
