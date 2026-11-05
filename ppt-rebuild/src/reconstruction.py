"""Three-branch candidates, protected pixel replacements and reversible scenes.

Model decisions enter through a saved, source-hash-bound review file. This module
performs local OCR/CV/font matching only and never calls a model API.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from font_matcher import fit_visual
from object_registry import ObjectRegistry, validate_bbox
from ocr_engine import detect, group_lines, to_record
from slideir_adapter import sha256, write_json


def rectangle(bbox, size):
    validate_bbox(bbox, size)
    x, y, w, h = [int(v) for v in bbox]
    mask = np.zeros(size[::-1], bool)
    mask[y:y + h, x:x + w] = True
    return mask


def intersect(a, b):
    return max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(
        0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))


def padded(bbox, size, margin=3):
    x, y, w, h = [int(v) for v in bbox]
    left, top = max(0, x - margin), max(0, y - margin)
    return [left, top, min(size[0], x + w + margin) - left,
            min(size[1], y + h + margin) - top]


def background_model(rgb, bbox, excluded):
    """Interpolate clean lateral samples without crossing horizontal card edges."""
    height, width = rgb.shape[:2]
    x, y, w, h = [int(v) for v in bbox]
    side_colors, residuals = [], []
    for left, right in ((max(0, x-7), max(0, x-1)), (min(width, x+w+1), min(width, x+w+7))):
        if right <= left:
            continue
        samples = rgb[y:y+h, left:right].astype(float)
        allowed = ~excluded[y:y+h, left:right]
        rows = np.full((h, 3), np.nan)
        for row in range(h):
            values = samples[row][allowed[row]]
            if len(values):
                rows[row] = np.median(values, axis=0)
                residuals.extend(np.abs(values - rows[row]).max(axis=1).tolist())
        known = np.isfinite(rows[:, 0])
        if known.sum() < max(2, h // 2):
            continue
        for channel in range(3):
            rows[:, channel] = np.interp(np.arange(h), np.flatnonzero(known), rows[known, channel])
        side_colors.append(rows)
    if not side_colors:
        raise ValueError("insufficient unprotected lateral background samples")
    start, end = side_colors[0], side_colors[-1]
    amount = np.linspace(0, 1, w)[None, :, None]
    estimate = start[:, None, :] * (1-amount) + end[:, None, :] * amount
    return np.clip(estimate, 0, 255).round().astype(np.uint8), float(np.percentile(residuals, 75))


def layer_contributions(layers, mask):
    """Visible contribution is alpha multiplied by all higher transmittances."""
    remaining = np.ones(int(mask.sum()), np.float32)
    result = []
    for z, rgba in reversed(list(enumerate(layers))):
        alpha = rgba[..., 3][mask].astype(np.float32) / 255
        weight = alpha * remaining
        if weight.sum() > 0.01:
            result.append({"z": z, "file": f"layer-{z:02d}.png",
                           "visible_weight": round(float(weight.mean()), 6),
                           "alpha_coverage": round(float((alpha > 0.01).mean()), 6)})
        remaining *= 1 - alpha
    return sorted(result, key=lambda v: v["z"])


def extract_rgba(source, background, mask):
    """Unmix source over the estimated background, preserving straight alpha.

    Alpha is at least the value needed to keep recovered RGB inside the gamut.
    Fully transparent pixels carry zero RGB. Recomposition is measured separately.
    """
    src, bg = source.astype(np.float64), background.astype(np.float64)
    delta = src - bg
    minimum = np.where(delta >= 0, delta / np.maximum(255 - bg, 1), -delta / np.maximum(bg, 1)).max(axis=2)
    alpha = np.where(mask, np.clip(minimum + 0.02, 1 / 255, 1), 0)
    foreground = np.clip(bg + delta / np.maximum(alpha[..., None], 1 / 255), 0, 255)
    rgba = np.dstack((foreground.round().astype(np.uint8), (alpha * 255).round().astype(np.uint8)))
    rgba[~mask] = 0
    return rgba


def load_review(file, page):
    if file is None:
        return {"objects": [], "text_corrections": {}}
    payload = json.loads(Path(file).read_text(encoding="utf-8"))
    if payload.get("source_sha256") != sha256(page.source_path):
        raise ValueError("review file is not bound to this source image")
    if payload.get("image_size") != list(page.size):
        raise ValueError("review file image_size mismatch")
    for candidate in payload.get("objects", []):
        validate_bbox(candidate["bbox"], page.size)
        if candidate["kind"] not in {"image", "shape"}:
            raise ValueError("review objects must be image or shape candidates")
        if candidate["kind"] == "shape" and candidate.get("geometry", "rect") != "rect":
            raise ValueError("reviewed shape extraction currently accepts rectangles")
    return payload


def ocr_record(page, cache=None):
    if cache:
        result = json.loads(Path(cache).read_text(encoding="utf-8"))
        if result.get("source_sha256") != sha256(page.source_path):
            raise ValueError("cached OCR requires matching source_sha256")
        if result["image_size"] != list(page.size):
            raise ValueError("cached OCR size mismatch")
        return result
    lines = detect(page.source_path, min_score=0.0)
    result = to_record(lines, group_lines(lines), page.size)
    result["source_sha256"] = sha256(page.source_path)
    return result


def geometry_candidates(rgb, lines, images):
    """Conservative solid blue bullets and thin separators, outside text/icons."""
    color = rgb.astype(np.int16)
    blue = ((color[..., 2] - color[..., 0] > 35) &
            (color[..., 1] - color[..., 0] > 20) & (color[..., 2] > 110)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(blue, 8)
    result = []
    for x, y, w, h, pixels in stats[1:]:
        bbox = [int(x), int(y), int(w), int(h)]
        thin = (w >= 60 and h <= 3) or (h >= 30 and w <= 3)
        bullet = 5 <= w <= 15 and 5 <= h <= 15 and 0.6 < w / h < 1.6
        if not (thin or bullet) or pixels / (w * h) < 0.85:
            continue
        if any(intersect(bbox, entry["bbox"]) for entry in [*lines, *images]):
            continue
        if x == 0 or y == 0 or x + w >= rgb.shape[1] or y + h >= rgb.shape[0]:
            continue
        fill = np.median(rgb[y:y + h, x:x + w].reshape(-1, 3), axis=0).astype(int)
        result.append({"id": f"shape-{len(result):03d}", "kind": "shape", "bbox": bbox,
                       "geometry": "rect", "fill": "".join(f"{v:02X}" for v in fill),
                       "label": "Solid source bullet or separator", "origin": "local_color_components"})
    return result


def image_candidates(page, lines):
    """LayerD alpha regions propose bounded assets for later source review."""
    found = []
    for layer in page.layers[1:]:
        for region in layer.regions:
            x, y, w, h = region["bbox"]
            if not (20 <= w <= 180 and 20 <= h <= 150 and 0.25 <= w / h <= 4):
                continue
            bbox = padded([x, y, w, h], page.size)
            if any(intersect(bbox, line["bbox"]) for line in lines):
                continue
            if any(intersect(bbox, old["bbox"]) for old in found):
                continue
            found.append({"id": f"visual-{len(found):03d}", "kind": "image", "bbox": bbox,
                          "label": "LayerD alpha region pending semantic review", "reviewed": False,
                          "origin": "layerd_alpha_component", "source_layers": [layer.file]})
    return found


def prepare_candidates(page, directory, *, review_file=None, ocr_cache=None):
    directory = Path(directory)
    review = load_review(review_file, page)
    ocr = ocr_record(page, ocr_cache)
    write_json(directory / "ocr.json", ocr)
    rgb = np.array(Image.open(page.source_path).convert("RGB"))
    layer_pixels = [np.array(Image.open(layer.path).convert("RGBA")) for layer in page.layers]
    reviewed = review.get("objects", [])
    images = [dict(o, origin="conversation_review", reviewed=True) for o in reviewed if o["kind"] == "image"]
    proposals = image_candidates(page, ocr["lines"])
    write_json(directory / "visual-proposals.json", proposals)
    # Unreviewed region proposals remain source-owned; the saved review chooses assets.
    excluded_boxes = images
    text_candidates = []
    corrections = review.get("text_corrections", {})
    line_ids = {line["id"] for line in ocr["lines"]}
    if set(corrections) - line_ids:
        raise ValueError("text correction references unknown OCR line")
    for line in ocr["lines"]:
        corrected = corrections.get(line["id"], {})
        if corrected.get("original", line["text"]) != line["text"]:
            raise ValueError("text correction original does not match saved OCR")
        text = corrected.get("text", line["text"])
        if not text.strip():
            raise ValueError("empty text correction")
        text_bbox = corrected.get("bbox", line["bbox"])
        validate_bbox(text_bbox, page.size)
        if any(intersect(text_bbox, image["bbox"]) >= text_bbox[2] * text_bbox[3] * 0.95
               for image in excluded_boxes):
            continue
        item = {"id": "text-" + line["id"], "kind": "text", "bbox": padded(text_bbox, page.size, 2),
                "text": text, "ocr_text": line["text"], "ocr_score": line["score"],
                "origin": "local_ocr", "source_text_bbox": line["bbox"],
                "content_checked": corrected.get("checked", line["id"] in review.get("checked_lines", [])),
                "retain_raster": corrected.get("retain_raster", False)}
        if "style" in corrected:
            item["style_override"] = corrected["style"]
        text_candidates.append(item)
    shapes = [dict(o, origin="conversation_review") for o in reviewed if o["kind"] == "shape"]
    automatic = geometry_candidates(rgb, ocr["lines"], [*images, *shapes])
    candidates = [*images, *shapes, *automatic, *text_candidates]
    write_json(directory / "candidates.json", {
        "schema": "ReconstructionCandidates/1", "source_sha256": sha256(page.source_path),
        "image_size": list(page.size), "review": review, "objects": candidates})
    registry = ObjectRegistry(page.size, sha256(page.source_path))
    entries = [registry.add(candidate) for candidate in candidates]
    all_regions = np.zeros(page.size[::-1], np.uint16)
    for entry in entries:
        all_regions += rectangle(entry["bbox"], page.size)
    claimed = np.zeros(page.size[::-1], bool)
    replacements = {}
    for entry in entries:
        try:
            bbox = entry["bbox"]
            x, y, w, h = bbox
            own_region = rectangle(bbox, page.size)
            other_regions = all_regions > own_region.astype(np.uint16)
            if entry["kind"] == "image" and not entry.get("reviewed"):
                raise ValueError("visual candidate requires source review")
            if entry.get("retain_raster"):
                raise ValueError("review requires original raster content")
            if entry["kind"] == "text" and entry["ocr_score"] < 0.85 and not entry["content_checked"]:
                raise ValueError("low-confidence text requires source review")
            estimate, ring_error = background_model(rgb, bbox, other_regions)
            if ring_error > 12:
                raise ValueError(f"complex background ring ({ring_error:.2f})")
            crop = rgb[y:y + h, x:x + w]
            difference = np.abs(crop.astype(np.int16) - estimate.astype(np.int16)).max(axis=2)
            local_mask = difference > (14 if entry["kind"] == "image" else 30)
            if entry["kind"] == "shape":
                local_mask[:] = True
            if local_mask.sum() < 6:
                raise ValueError("no separated foreground pixels")
            local_mask = cv2.dilate(local_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            mask = np.zeros(page.size[::-1], bool)
            mask[y:y + h, x:x + w] = local_mask
            if np.any(mask & (claimed | other_regions)):
                raise ValueError("replacement mask intersects protected evidence")
            repair = estimate
            repair_method = "local_background_plane"
            # A lower layer is usable only when its actual pixels agree with the clean model.
            lower = layer_pixels[0][y:y + h, x:x + w]
            lower_error = float(np.abs(lower[..., :3].astype(np.int16) - estimate.astype(np.int16))[local_mask].mean())
            if lower_error <= 2.0 and np.all(lower[..., 3][local_mask] == 255):
                repair = lower[..., :3]
                repair_method = "validated_layerd_background"
            common = {"id": entry["id"], "bbox": bbox, "evidence_ids": [entry["evidence_id"]]}
            asset = None
            if entry["kind"] == "image":
                rgba = extract_rgba(crop, repair, local_mask)
                asset = Image.fromarray(rgba)
                node = dict(common, type="image", asset_id=entry["id"])
            elif entry["kind"] == "shape":
                node = dict(common, type="shape", geometry="rect", fill=entry["fill"])
            else:
                fit = fit_visual(rgb, estimate, entry["text"], bbox, dpi=96)
                entry["font_fit"] = fit
                if not fit.get("size_pt"):
                    raise ValueError("font matching failed")
                # Text color is the pixels farthest from the estimated background.
                core = crop[difference >= np.percentile(difference[local_mask], 85)]
                color = np.median(core, axis=0).round().astype(int)
                style = entry.get("style_override", {})
                node = dict(common, type="text", text=entry["text"],
                            font={"latin": "Arial", "ea": style.get("family", fit.get("family") or "微软雅黑"),
                                  "size_pt": style.get("size_pt", fit["size_pt"])},
                            bold=style.get("bold", bool(fit.get("bold"))),
                            color=style.get("color", "".join(f"{v:02X}" for v in color)),
                            align="left", wrap=False, vertical_anchor="middle")
            residual = float(np.abs(repair.astype(np.int16) - estimate.astype(np.int16))[local_mask].mean())
            if residual > 6:
                raise ValueError(f"background residue {residual:.2f}")
            mask_path = directory / "masks" / f"{entry['id']}.png"
            mask_path.parent.mkdir(exist_ok=True)
            Image.fromarray(local_mask.astype(np.uint8) * 255).save(mask_path)
            registry.transition(entry, "matched", mask_uri=str(mask_path.relative_to(directory)),
                                mask_origin=[x, y], mask_size=[w, h], source_layers=layer_contributions(layer_pixels, mask),
                                repair_method=repair_method, ring_error=round(ring_error, 3),
                                background_residue=round(residual, 3), protected_pixels_changed=0)
            claimed |= mask
            replacements[entry["id"]] = {"node": node, "bbox": list(bbox), "asset": asset, "mask": mask,
                                         "local_mask": local_mask, "repair": repair}
        except ValueError as error:
            registry.transition(entry, "reverted", reason=str(error))
    return registry, replacements, rgb, layer_pixels


def materialize(rgb, replacements, active):
    plate = rgb.copy()
    nodes, assets = [], {}
    for object_id, replacement in replacements.items():
        if object_id not in active:
            continue
        node = replacement["node"]
        x, y, w, h = replacement["bbox"]
        region = plate[y:y + h, x:x + w]
        region[replacement["local_mask"]] = replacement["repair"][replacement["local_mask"]]
        nodes.append(node)
        if replacement["asset"] is not None:
            assets[node["asset_id"]] = replacement["asset"]
    return Image.fromarray(plate), nodes, assets


def save_working_layers(directory, layers, rgb, replacements, active):
    """Patch all contributing layers, leaving the original pack untouched."""
    edited = [layer.copy() for layer in layers]
    for object_id in active:
        replacement = replacements[object_id]
        mask = replacement["mask"]
        x, y, w, h = replacement["bbox"]
        edited[0][y:y + h, x:x + w, :3][replacement["local_mask"]] = replacement["repair"][replacement["local_mask"]]
        edited[0][mask, 3] = 255
        for layer in edited[1:]:
            layer[mask, 3] = 0
    target = Path(directory) / "working-layers"
    target.mkdir()
    for z, layer in enumerate(edited):
        Image.fromarray(layer).save(target / f"layer-{z:02d}.png")
