"""Recover independently editable fields on a shared, source-observed paint surface."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from ..contracts import intersection
from ..materialize import digest, write
from ..object_extract import rectangle
from ..parallel_rebuild import array_sha
from ..parallel_regions import regions
from ..parallel_store import key
from .candidate_stage import stage_text_pixels
from .region_crop import _separate_photo_caption


def contains(parent, child):
    return intersection(parent, child) >= 0.96 * child[2] * child[3]


def clipped(box, size):
    x, y, w, h = map(round, box)
    left, top = max(x, 0), max(y, 0)
    return [left, top, max(0, min(size[0], x + w) - left), max(0, min(size[1], y + h) - top)]


def plan_regions(graph, targets, inventory, source):
    """Photo interiors and different cells can never become one repair region."""
    elements = inventory.get("elements", [])
    target_ids = {t.id for t in targets}
    groups = {}
    for text in graph.text.texts:
        if text.id not in target_ids:
            continue
        parents = [
            e
            for e in elements
            if e["category"] in ("photo", "logo", "chart", "table", "card_shape", "ribbon_shape", "cell")
            and contains(e["bbox"], text.bbox)
        ]
        parent = min(parents, key=lambda e: e["bbox"][2] * e["bbox"][3], default=None)
        # A chart detector can enclose the separate captions underneath it.
        # A tight header observation permits a field trial, subject to all
        # subsequent colour, neighbor ownership and actual Office checks.
        if parent and parent["category"] == "chart" and any(
            e["category"] == "section_header" and contains(e["bbox"], text.bbox)
            and e["bbox"][2]*e["bbox"][3] <= 1.3*text.bbox[2]*text.bbox[3] for e in elements
        ):
            parent = None
        if parent:
            kind = parent["category"]
            box = clipped(parent["bbox"], graph.image_size)
            if kind in ("table", "logo", "chart"):
                continue
            if kind == "photo":
                split = _separate_photo_caption(
                    source, {"kinds": ["layout:image"], "box": box, "seed_boxes": [box]}, [text]
                )
                if "caption_boundary" not in split:
                    continue
                x, y, w, h = box
                top = split["caption_boundary"] + 1
                box = [x, top, w, y + h - top]
                if not contains(box, text.bbox):
                    continue
            name = [parent["id"], kind, box]
        else:
            # Unlabelled fields can also sit on measurable paper. The same
            # surface and ownership proofs apply before staging any replacement.
            title = any(e["category"] == "title" and contains(e["bbox"], text.bbox) for e in elements)
            x, y, w, h = text.bbox
            box = clipped([x - h, y - 0.45 * h, w + 2 * h, 1.9 * h], graph.image_size)
            name = [text.id, "title" if title else "field", box]
        gid = key([graph.source_sha256, name])[:20]
        row = groups.setdefault(
            gid, {"id": gid, "surface": name[0], "kind": name[1], "bbox": box, "members": []}
        )
        row["members"].append(text.id)
    return list(groups.values())


def _field(source, box, text_boxes):
    """Fit observed paper/strip colour, excluding all OCR rectangles during the fit."""
    x, y, w, h = box
    tile = source[y : y + h, x : x + w].astype(np.float32)
    sy, sx = np.indices((h, w))
    samples = np.ones((h, w), bool)
    margin = max(1, round(min(w, h) * 0.045))
    samples[:margin] = samples[-margin:] = False
    samples[:, :margin] = samples[:, -margin:] = False
    for b in text_boxes:
        tx, ty, tw, th = map(round, b)
        samples[
            max(0, ty - y - 1) : max(0, ty - y + th + 1), max(0, tx - x - 1) : max(0, tx - x + tw + 1)
        ] = False
    # Use spatially distributed smooth pixels. A narrow photograph edge, icon or
    # incidental patch of a similar colour is insufficient evidence for a surface.
    xs = sx / max(1, w - 1)
    ys = sy / max(1, h - 1)
    smooth = np.abs(tile - cv2.GaussianBlur(tile, (5, 5), 0)).max(2)
    usable = samples & (smooth < 5)
    if usable.sum() < max(20, 0.06 * w * h):
        raise ValueError("too little exposed colour on region")
    chosen = usable.copy()
    for _ in range(3):
        if (
            chosen.sum() < max(20, 0.65 * usable.sum())
            or np.ptp(xs[chosen]) < 0.55
            or np.ptp(ys[chosen]) < 0.33
        ):
            raise ValueError("no contiguous simple surface around fields")
        stride = max(1, int(chosen.sum() / 6000))
        A = np.column_stack((xs[chosen][::stride], ys[chosen][::stride], np.ones_like(xs[chosen][::stride])))
        coef = np.linalg.lstsq(A, tile[chosen][::stride], rcond=None)[0]
        field = xs[:, :, None] * coef[0] + ys[:, :, None] * coef[1] + coef[2]
        errors = np.abs(tile - field).max(2)
        limit = min(34, max(6, float(np.percentile(errors[chosen], 80)) * 1.8))
        chosen = usable & (errors <= limit)
    if np.mean(errors[usable] <= 27) < 0.80:
        raise ValueError("texture dominates measured surface")
    return field.clip(0, 255).astype(np.uint8)


def _external_ink(guard, source, field, box, peer, members):
    """Clip an oversized peer bbox only at a measured blank gutter between two labels."""
    x, y, w, h = box
    protected = rectangle(peer, (source.shape[1], source.shape[0]))[y : y + h, x : x + w].copy()
    for t in members:
        if not intersection(peer, t.bbox):
            continue
        tx, ty, tw, th = map(round, t.bbox)
        px, py, pw, ph = map(round, peer)
        if pw <= 0 or tw <= 0 or abs((py + ph / 2) - (ty + th / 2)) > 0.55 * max(ph, th):
            continue
        left_neighbor = px + pw / 2 < tx
        if not left_neighbor and px + pw / 2 <= tx + tw:
            continue
        boundary = tx if left_neighbor else tx + tw
        reach = max(2, round(0.38 * th))
        i0 = max(x, boundary - reach)
        i1 = min(x + w, boundary + reach)
        j0 = max(y, ty, py)
        j1 = min(y + h, ty + th, py + ph)
        if j1 <= j0 or i1 <= i0:
            continue
        # Full-height whitespace is the separation witness. A touching character
        # or line across the gap keeps the original protection rectangle intact.
        delta = abs(
            source[j0:j1, i0:i1].astype(int) - field[j0 - y : j1 - y, i0 - x : i1 - x].astype(int)
        ).max(2)
        gap = delta.max(0) < 10
        runs = np.flatnonzero(np.diff(np.r_[False, gap, False])).reshape(-1, 2)
        candidates = [i0 + (a + b) // 2 for a, b in runs if b - a >= max(2, round(0.07 * th))]
        if candidates:
            cut = min(candidates, key=lambda i: abs(i - boundary)) - x
            if left_neighbor:
                protected[:, cut:] = False
            else:
                protected[:, :cut] = False
    guard |= protected


def _complete_title_ink(tile, field, scope, guard, height):
    """On smooth paper follow whole glyph components that are anchored inside the OCR line."""
    contrast = np.abs(tile.astype(float) - field).max(2)
    radius = max(1, round(height * 0.12))
    near = cv2.dilate(scope.astype("uint8"), np.ones((2*radius+1, 2*radius+1), "uint8")).astype(bool)
    ring = near & ~scope & ~guard
    if not ring.any() or np.percentile(contrast[ring], 85) > 10:
        return np.zeros_like(scope)
    components = (contrast > 14) & near
    count, labels, stats, _ = cv2.connectedComponentsWithStats(components.astype("uint8"), 8)
    core_ids = set(np.unique(labels[scope & (contrast > 55)]))
    protected_ids = set(np.unique(labels[guard]))
    border = near & ~cv2.erode(near.astype("uint8"), np.ones((3,3), "uint8")).astype(bool)
    protected_ids.update(np.unique(labels[border]))
    ids = [i for i in range(1, count) if i in core_ids and i not in protected_ids
           and stats[i, cv2.CC_STAT_WIDTH] <= 1.25 * height
           and stats[i, cv2.CC_STAT_HEIGHT] <= 1.25 * height]
    return np.isin(labels, ids)


def measure_group(graph, group, source, baseline, lines):
    """Recompute bounded stroke ownership from source, foreground colours and the exposed surface."""
    members = [t for t in graph.text.texts if t.id in group["members"]]
    box = group["bbox"]
    x, y, w, h = box
    field = _field(source, box, [t.bbox for t in graph.text.texts] + [line["bbox"] for line in lines])
    tile = source[y : y + h, x : x + w]
    others = np.zeros((h, w), bool)
    included = {t.id for t in members}
    line_ids = {line for t in members for line in graph.text.members.get(t.id, [])}
    for b in [t.bbox for t in graph.text.texts if t.id not in included] + [
        v["bbox"] for v in lines if v["id"] not in line_ids
    ]:
        if intersection(b, box):
            _external_ink(others, source, field, box, b, members)
    for asset in graph.asset.objects:
        if not intersection(asset.bbox, box):
            continue
        rec = graph.selected.get(asset.id, {})
        if rec.get("retained_card_artwork"):
            continue  # Transferring its pixels requires a separate, bound ownership proof in staging.
        if (
            group["kind"] == "photo"
            and contains(asset.bbox, group["bbox"])
            and not rec
            and asset.kind == "illustration"
        ):
            continue
        others |= rectangle(asset.bbox, graph.image_size)[y : y + h, x : x + w]
    for rec in graph.selected.values():
        if rec.get("retained_card_artwork") or not intersection(rec["bbox"], box):
            continue
        ref = rec.get("owner_evidence")
        if not ref or digest(baseline / ref["file"]) != ref["sha256"]:
            raise ValueError("unknown previously committed pixels")
        with np.load(baseline / ref["file"], allow_pickle=False) as proof:
            others |= proof["mask"][y : y + h, x : x + w].astype(bool)
    used = np.zeros((h, w), bool)
    styles = {s.id: s for s in graph.text.styles}
    measured = {}
    for t in members:
        if graph.ownership.get(t.id) != "RESOLVED" or t.spans:
            raise ValueError("field ownership/style is not isolated")
        fg = np.frombuffer(bytes.fromhex(styles[t.style].color), dtype=np.uint8).astype(float)
        direction = fg - field.astype(float)
        deviation = tile.astype(float) - field
        den = (direction * direction).sum(2)
        alpha = (deviation * direction).sum(2) / np.maximum(1, den)
        color_error = np.abs(deviation - alpha[:, :, None] * direction).max(2)
        scope = rectangle(t.bbox, graph.image_size)[y : y + h, x : x + w]
        cores = scope & (alpha > 0.53) & (color_error < 32) & (den > 1000)
        if cores.sum() < max(3, 0.018 * scope.sum()) or (cores & others).any():
            raise ValueError("text ink missing or shared with a protected neighbor")
        tx, ty, tw, th = map(round, t.bbox)
        yy, xx = np.where(cores)
        if np.ptp(xx) < 0.48 * tw or np.ptp(yy) < 0.30 * th:
            raise ValueError("source strokes do not cover OCR field")
        # Extend only connected antialiased pixels consistent with both the
        # claimed text colour and the local surface. Preserve photos, rules and icons.
        radius = max(1, round(styles[t.style].size_px * 0.10))
        near = (
            cv2.dilate(
                cores.astype("uint8"),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
            )
            > 0
        )
        touching = cv2.dilate(cores.astype("uint8"), np.ones((3, 3), "uint8")) > 0
        mask = (
            near
            & scope
            & (alpha > 0.027)
            & ((color_error < 34) | (touching & (alpha > 0.45) & (color_error < 68)))
        )
        n, _, stats, _ = cv2.connectedComponentsWithStats(cores.astype("uint8"), 8)
        if any(v[cv2.CC_STAT_WIDTH] > max(th * 2.4, tw * 0.42) for v in stats[1:]) and len(t.text) > 3:
            raise ValueError("foreground includes non-letter geometry")
        from .marked_fields import footer_fields
        if group["kind"] in ("title", "field") and (getattr(styles[t.style], "role", None) == "title"
                or footer_fields(source, t.text, t.bbox)):
            mask |= _complete_title_ink(tile, field, scope, others, th)
        if (mask & (used | others)).any():
            raise ValueError("text ownership is not disjoint")
        used |= mask
        measured[t.id] = mask
    # Interpolate small variations of the exposed face (e.g. a blue overlay on
    # a photo) only where strokes were identified. Outside colours stay original.
    field_height = min(t.bbox[3] for t in members)
    halo_radius = max(1, round(field_height * 0.14))
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * halo_radius + 1, 2 * halo_radius + 1))
    surroundings = cv2.dilate(used.astype("uint8"), brush)
    fitted = cv2.inpaint(tile, surroundings, max(2, round(field_height * 0.18)), cv2.INPAINT_NS)
    discrepancy = abs(tile.astype(int) - fitted.astype(int)).max(2)
    # JPEG ringing and shadow can extend to the other side of the colour ramp.
    # Only low-contrast deviations connected to proven ink are attributed to it.
    # Sample beyond that fringe even when repairing the less-dilated write mask.
    added = np.zeros_like(used)
    for t in members:
        edge = (cv2.dilate(measured[t.id].astype("uint8"), brush) > 0) & ~used
        # OCR boxes can end inside the last stroke. Only extend a fringe
        # through connected colour evidence, never by claiming a rectangle.
        edge &= ~others & (discrepancy > 3) & (discrepancy <= 56)
        _, components = cv2.connectedComponents((edge | measured[t.id]).astype("uint8"), 8)
        edge &= np.isin(components, np.unique(components[measured[t.id]]))
        if (edge & added).any():
            raise ValueError("shared antialiased glyph edge")
        measured[t.id] |= edge
        added |= edge
    used |= added
    raw = source.copy()
    raw[y : y + h, x : x + w][used] = fitted[used]
    full_guard = np.zeros(source.shape[:2], bool)
    for peer in graph.text.texts:
        if peer.id not in included:
            full_guard |= rectangle(peer.bbox, graph.image_size)
    for asset in graph.asset.objects:
        if not graph.selected.get(asset.id, {}).get("retained_card_artwork"):
            full_guard |= rectangle(asset.bbox, graph.image_size)
    full_guard[y : y + h, x : x + w] = others
    return measured, raw, full_guard


def _project(mask, box, size):
    x, y, w, h = box
    result = np.zeros((size[1], size[0]), bool)
    result[y : y + h, x : x + w] = mask
    return result


def choose_region_background(group, graph, source, masks, fitted, guard, layers, inpaint, directory):
    """Compare any supplied lower layer against the observed face before spending an inpaint call."""
    from .layerd_adapter import lower_backgrounds
    from .background_adapter import RecoveryBlocked

    union = _project(np.logical_or.reduce(list(masks.values())), group["bbox"], graph.image_size)
    outline = cv2.dilate(union.astype("uint8"), np.ones((5, 5), "uint8")).astype(bool)
    ring = outline & ~union & ~guard
    gx, gy, gw, gh = group["bbox"]
    ring &= rectangle([gx, gy, gw, gh], graph.image_size)
    if ring.sum() < 10:
        return
    fallbacks = []
    if layers:
        try:
            fallbacks = [(v.image, "LayerD", v.provenance) for v in lower_backgrounds(source, layers, union)]
        except (RecoveryBlocked, ValueError, OSError):
            pass

    def acceptable(image):
        if image is None or image.shape != source.shape or image.dtype != np.uint8:
            return False
        # A lower layer that still contains the glyph, or lost the paint behind it,
        # cannot represent the revealed face. Use normal-view colour variation.
        seam = np.abs(image[ring].astype(float) - source[ring]).max(1)
        paint = np.abs(image[union].astype(float) - fitted[union]).max(1)
        return np.percentile(seam, 90) <= 12 and np.percentile(paint, 90) <= 19

    chosen = next((v for v in fallbacks if acceptable(v[0])), None)
    if chosen is None and inpaint:
        neighbors = fitted[ring].astype(float)
        spread = np.percentile(neighbors, 90, axis=0) - np.percentile(neighbors, 10, axis=0)
        if np.max(spread) > 30:
            try:
                image = inpaint(source.copy(), union)
                if acceptable(image):
                    chosen = (image, "LaMa", {})
            except (ValueError, OSError, RuntimeError):
                pass
    if chosen:
        image, method, provenance = chosen
        file = directory / ("region-background-" + group["id"] + ".npz")
        np.savez_compressed(file, clean=image[union])
        group["background"] = {
            "file": str(file.resolve()),
            "sha256": digest(file),
            "owner_mask": array_sha(union.astype("uint8")),
            "source": graph.source_sha256,
            "method": method,
            "provenance": provenance,
        }


def stage_region(group, graph, baseline, output, fonts, frame_overrides, source, lines):
    """Expose only a complete group; retry on the unchanged baseline if Office rejects it."""
    baseline, output = Path(baseline), Path(output)
    if digest(baseline / "source.png") != graph.source_sha256:
        raise ValueError("changed region source")
    records = regions(graph)[1]
    ids = group["members"]
    if any(records[i]["member_ids"] != [i] for i in ids):
        raise ValueError("indivisible graph relationship crosses text group")
    masks, raw, guard = measure_group(graph, group, source, baseline, lines)
    full = {i: _project(mask, group["bbox"], graph.image_size) for i, mask in masks.items()}
    union = np.logical_or.reduce(list(full.values()))
    backing = group.get("background")
    if backing:
        if (
            backing["source"] != graph.source_sha256
            or digest(Path(backing["file"])) != backing["sha256"]
            or backing["owner_mask"] != array_sha(union.astype("uint8"))
        ):
            raise ValueError("region background no longer matches source and ownership")
        with np.load(backing["file"], allow_pickle=False) as data:
            colors = data["clean"]
        if colors.dtype != np.uint8 or colors.shape != raw[union].shape:
            raise ValueError("invalid region backing samples")
        raw[union] = colors
    receipt = {
        "source": graph.source_sha256,
        "group": group,
        "union_sha256": array_sha(union.astype("uint8")),
        "patch_sha256": array_sha(raw[union]),
        "ocr_evidence": key(lines),
    }
    parents = {}
    for oid, mask in full.items():
        for aid, record in graph.selected.items():
            if not record.get("retained_card_artwork"):
                continue
            proof = record["owner_evidence"]
            if digest(baseline / proof["file"]) != proof["sha256"]:
                raise ValueError("retained artwork owner binding mismatch")
            with np.load(baseline / proof["file"], allow_pickle=False) as data:
                if (data["mask"].astype(bool) & mask).any():
                    if oid in parents:
                        raise ValueError("multiple card artworks share the same field")
                    parents[oid] = aid
    receipt["source_artwork"] = parents
    evidence_name = "region-" + group["id"] + ".json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="text-group-", dir=output.parent) as tmp:
        root = Path(tmp)
        write(root / "group.json", receipt)
        current = baseline
        for index, oid in enumerate(ids):
            text = next(t for t in graph.text.texts if t.id == oid)
            mask = full[oid]
            field_guard = guard | (union & ~mask)
            fb = (
                {"size": frame_overrides[oid], "method": "office_batch_auto_frame_repair/1"}
                if oid in frame_overrides
                else None
            )
            out = root / f"field-{index}"
            stage_text_pixels(
                graph,
                current,
                fonts,
                out,
                text=text,
                region=records[oid],
                source=source,
                box=list(map(round, text.bbox)),
                wanted=mask,
                written=raw,
                protected=field_guard,
                method="region-surface-text",
                report={"status": "PRECHECK_PASS", "region": group["id"]},
                completion_report=receipt,
                trial_directory=output,
                hashes={evidence_name: digest(root / "group.json")},
                frame_minimum=fb,
                fit_origin=True,
                source_artwork=parents.get(oid),
            )
            current = out
        write(current / evidence_name, receipt)
        current.rename(output)
    return output
