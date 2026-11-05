"""Per-region fidelity measurement and bounded repair for a compiled SlideIR page.

measure  : compare the re-imported render against the source page, per node and on a
           coarse grid, so repair is driven by evidence instead of guesswork.
plate    : erase every native text box from the source page (OpenCV inpaint) so a
           region can be reverted to a crop without putting text back into a raster.
apply    : rewrite the IR - gradient fills for flat-fitted shapes, image assets re-cut
           from the plate, and nodes above the diff threshold converted to plate crops.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

THRESHOLD = 10.0  # a rebuilt region may stay native only below this mean channel difference


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def node_box(node):
    return node.get("source_text_bbox") or node["bbox"]


def crop_masks(ir, source, exclude_ids=()):
    """Text pixels are excluded from every fit: they are not background evidence."""
    mask = np.zeros(source.shape[:2], np.uint8)
    for node in ir["nodes"]:
        if node["type"] != "text" or node["id"] in exclude_ids:
            continue
        x, y, w, h = node_box(node)
        a, b = max(0, int(x) - 1), max(0, int(y) - 1)
        c, d = min(source.shape[1], math.ceil(x + w) + 1), min(source.shape[0], math.ceil(y + h) + 1)
        mask[b:d, a:c] = 255
    return mask


def _http_inpaint(image, mask, url, timeout=180):
    """Send the page and mask to a remote inpainting service and decode the reply."""
    import io
    import urllib.request
    import uuid

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG")
    mask_bytes = buffer.getvalue()

    boundary = uuid.uuid4().hex
    body = b""
    for name, payload in (("image", image_bytes), ("mask", mask_bytes)):
        body += b"--" + boundary.encode() + b"\r\n"
        body += f'Content-Disposition: form-data; name="{name}"; filename="{name}.png"\r\n'.encode()
        body += b"Content-Type: image/png\r\n\r\n" + payload + b"\r\n"
    body += b"--" + boundary.encode() + b"--\r\n"

    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Content-Length": str(len(body))},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return np.array(Image.open(io.BytesIO(response.read())).convert("RGB"))


def build_plate(ir, source_path, out_path, inpaint_url=None):
    source = rgb(source_path)
    mask = crop_masks(ir, source)
    # Grow the mask a little: glyph antialiasing lives just outside the OCR box.
    grown = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    if inpaint_url:
        try:
            plate = _http_inpaint(source, grown, inpaint_url)
        except Exception as exc:  # noqa: BLE001 - fall back rather than lose the page
            print(f"[plate] 远程修复失败，回退 OpenCV TELEA：{type(exc).__name__}: {exc}", flush=True)
            plate = None
        if plate is not None and plate.shape == source.shape:
            Image.fromarray(plate).save(out_path)
            return plate, grown
    plate = cv2.inpaint(cv2.cvtColor(source, cv2.COLOR_RGB2BGR), grown, 4, cv2.INPAINT_TELEA)
    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
    Image.fromarray(plate).save(out_path)
    return plate, grown


def _region_stats(source, mask, box):
    x, y, w, h = [int(round(v)) for v in box]
    x, y = max(0, x), max(0, y)
    w, h = min(w, source.shape[1] - x), min(h, source.shape[0] - y)
    if w <= 2 or h <= 2:
        return None
    crop = source[y:y + h, x:x + w].astype(np.float32)
    inner = mask[y:y + h, x:x + w] == 0
    if inner.sum() < 0.15 * w * h:
        margin = max(1, min(4, int(min(w, h) * 0.08)))
        inner = np.zeros((h, w), bool)
        inner[margin:h - margin, margin:w - margin] = True
    return crop, inner


def fit_fill(source, mask, box):
    """Return ('solid', hex) | ('gradient', angle, [(offset,hex)...]) | None."""
    stats = _region_stats(source, mask, box)
    if stats is None:
        return None
    crop, inner = stats
    values = crop[inner]
    if len(values) < 40:
        return None
    ys, xs = np.nonzero(inner)
    best = None
    # solid
    med = np.median(values, axis=0)
    solid_res = float(np.abs(values - med).mean())
    best = {"kind": "solid", "residual": solid_res, "color": med}
    # single-axis gradients
    for axis, name in ((0, "vertical"), (1, "horizontal")):
        coord = ys if axis == 0 else xs
        lo, hi = float(coord.min()), float(coord.max())
        if hi - lo < 8:
            continue
        t = (coord - lo) / (hi - lo)
        design = np.stack([np.ones_like(t), t], axis=1)
        coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
        pred = design @ coeff
        resid = float(np.abs(values - pred).mean())
        if best["kind"] == "solid" or resid < best["residual"]:
            if best["kind"] == "solid" or resid < best["residual"]:
                if resid < best["residual"]:
                    best = {"kind": "gradient", "axis": name, "residual": resid,
                            "start": np.clip(coeff[0], 0, 255), "end": np.clip(coeff[0] + coeff[1], 0, 255)}
    if best["kind"] == "solid":
        return best
    # A gradient only wins if it explains the region clearly better.
    if best["residual"] > 0.75 * solid_res:
        return {"kind": "solid", "residual": solid_res, "color": med}
    return best


def to_hex(rgb_values):
    return "".join(f"{int(round(v)):02X}" for v in np.clip(np.asarray(rgb_values), 0, 255))


def ink_mask(source, min_diff=24, dilate=2):
    """Pixels that carry ink: local contrast against a median background, plus edges."""
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    background = cv2.medianBlur(gray, 31)
    diff = np.abs(gray.astype(np.int16) - background.astype(np.int16))
    edges = cv2.Canny(gray, 40, 120) > 0
    mask = ((diff > min_diff) | edges).astype(np.uint8) * 255
    if dilate:
        mask = cv2.dilate(mask, np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8), iterations=1)
    return mask


def node_coverage_mask(ir, shape):
    mask = np.zeros(shape[:2], np.uint8)
    for node in ir["nodes"]:
        x, y, w, h = [int(round(v)) for v in node["bbox"]]
        x, y = max(0, x), max(0, y)
        mask[y:min(shape[0], y + h), x:min(shape[1], x + w)] = 255
    return mask


def coverage_report(ir, source):
    """How much visible content has no owning node at all - the silent-loss metric."""
    ink = ink_mask(source)
    owned = node_coverage_mask(ir, source.shape)
    missing = cv2.bitwise_and(ink, cv2.bitwise_not(owned))
    total_ink = int((ink > 0).sum())
    lost = int((missing > 0).sum())
    return {
        "ink_pixels": total_ink,
        "unowned_ink_pixels": lost,
        "unowned_fraction_of_ink": round(lost / max(1, total_ink), 4),
        "unowned_fraction_of_page": round(lost / (source.shape[0] * source.shape[1]), 4),
    }, missing


def backfill_missing(ir, source, plate, *, save_asset, min_area=90, pad=3, z=None,
                     budget=0.15, merge=9):
    """Add plate crops over ink that no node owns, so nothing disappears silently.

    Components are ranked by how much ink they carry and added while the union of
    backfilled crops stays inside ``budget`` (a fraction of the page), so a page
    cannot turn into one big picture just because the plan missed a few strokes.
    """
    missing = coverage_report(ir, source)[1]
    if merge:
        merged = cv2.dilate(missing, np.ones((merge, merge), np.uint8), iterations=1)
    else:
        merged = missing
    count, _labels, stats, _ = cv2.connectedComponentsWithStats((merged > 0).astype(np.uint8), 8)
    h, w = source.shape[:2]
    page_area = float(h * w)
    components = []
    for i in range(1, count):
        x, y, bw, bh, _area = (int(v) for v in stats[i])
        ink_area = int((missing[y:y + bh, x:x + bw] > 0).sum())
        if ink_area < min_area:
            continue
        density = ink_area / float(max(1, bw * bh))
        components.append((density, ink_area, x, y, bw, bh, i))
    # Prefer compact, content-dense regions: they restore the most visible detail per
    # pixel of raster spent. Long thin components (a rule crossing the page) sort last.
    components.sort(reverse=True)

    used = {n.get("z") for n in ir["nodes"] if isinstance(n.get("z"), int)}
    text_z = [n["z"] for n in ir["nodes"] if n["type"] == "text" and isinstance(n.get("z"), int)]
    cursor = (min(text_z) - 1) if text_z else (max(used) + 1 if used else 0)

    added = []
    taken_area = 0.0
    for density, ink_area, x, y, bw, bh, i in components:
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        area = (x1 - x0) * (y1 - y0) / page_area
        if taken_area + area > budget:
            continue
        taken_area += area
        while cursor in used and cursor > 0:
            cursor -= 1
        z_value = cursor if cursor > 0 else (max(used) + 1 if used else 0)
        used.add(z_value)
        cursor -= 1
        asset_id = f"backfill-{i:03d}"
        save_asset(asset_id, plate[y0:y1, x0:x1])
        node = {"id": asset_id, "type": "image", "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                "z": int(z_value), "evidence_ids": [], "asset_id": asset_id}
        ir["nodes"].append(node)
        added.append({"id": asset_id, "bbox": node["bbox"], "z": int(z_value), "ink_pixels": ink_area})
    return added


def renumber_with_backfill(ir, added_ids):
    """Renumber z densely, placing backfilled crops just below the text layer.

    The page's z values are usually already dense, so there is no free integer
    "just under the text". Renumbering preserves order and guarantees that
    backfilled artwork never covers native text.
    """
    added = set(added_ids)
    ordered = sorted(ir["nodes"], key=lambda n: (n.get("z", 0), 0 if n["id"] in added else 1))
    keep = [n for n in ordered if n["id"] not in added]
    fill = [n for n in ordered if n["id"] in added]
    insert_at = next((i for i, n in enumerate(keep) if n["type"] == "text"), len(keep))
    final = keep[:insert_at] + fill + keep[insert_at:]
    for index, node in enumerate(final):
        node["z"] = index
    ir["nodes"] = final
    return ir


def text_audit(ir, source, preview):
    """Is every text node a native text box, and is it actually visible on the render?

    A white label on a white patch is still "native" in the XML but unreadable in the
    deck, so visibility is measured from the re-imported render, not from the IR.
    """
    rows = []
    for node in ir["nodes"]:
        if node["type"] != "text":
            continue
        x, y, w, h = [int(round(v)) for v in (node.get("source_text_bbox") or node["bbox"])]
        x, y = max(0, x), max(0, y)
        w, h = min(w, source.shape[1] - x), min(h, source.shape[0] - y)
        if w < 3 or h < 3:
            continue
        src_crop = source[y:y + h, x:x + w].astype(np.float32)
        prev_crop = preview[y:y + h, x:x + w].astype(np.float32)
        src_ink = float(np.std(src_crop.mean(axis=2)))
        prev_ink = float(np.std(prev_crop.mean(axis=2)))
        rows.append({
            "id": node["id"],
            "text": (node.get("text") or "")[:24],
            "color": node.get("color"),
            "source_contrast": round(src_ink, 2),
            "render_contrast": round(prev_ink, 2),
            # The source had ink here but the render did not: the label is invisible.
            "invisible": bool(src_ink > 6 and prev_ink < 3.5),
        })
    invisible = [r for r in rows if r["invisible"]]
    return {
        "text_nodes": len(rows),
        "invisible": len(invisible),
        "invisible_ids": [r["id"] for r in invisible],
        "invisible_detail": invisible[:8],
    }


def measure(ir_path, source_path, preview_path, out_path, grid=(12, 7)):
    ir = load_json(ir_path)
    source = rgb(source_path)
    preview = rgb(preview_path)
    if preview.shape != source.shape:
        preview = np.array(Image.fromarray(preview).resize((source.shape[1], source.shape[0]), Image.LANCZOS))
    diff = np.abs(source.astype(np.int16) - preview.astype(np.int16)).mean(axis=2)
    text_mask = crop_masks(ir, source) > 0
    page_area = float(source.shape[0] * source.shape[1])

    rows = []
    for node in ir["nodes"]:
        x, y, w, h = [int(round(v)) for v in node_box(node)]
        x, y = max(0, x), max(0, y)
        w, h = min(w, source.shape[1] - x), min(h, source.shape[0] - y)
        if w <= 1 or h <= 1:
            continue
        window = diff[y:y + h, x:x + w]
        whole = float(window.mean())
        # Shape error must be read off background pixels only, otherwise a text-heavy
        # card looks "wrong" purely because the glyphs were re-set.
        background = window[~text_mask[y:y + h, x:x + w]]
        bg_area = int(background.size)
        bg_diff = float(background.mean()) if bg_area else 0.0
        rows.append({"id": node["id"], "type": node["type"], "bbox": [x, y, w, h],
                     "mean_diff": round(whole, 2), "bg_diff": round(bg_diff, 2),
                     "bg_area": bg_area, "area": w * h,
                     "contribution_to_page_mean": round(bg_diff * bg_area / page_area, 3)})

    contrib = sorted(rows, key=lambda r: -r["contribution_to_page_mean"])[:12]

    gh, gw = grid
    cells = []
    for gy in range(gh):
        for gx in range(gw):
            y0, y1 = gy * source.shape[0] // gh, (gy + 1) * source.shape[0] // gh
            x0, x1 = gx * source.shape[1] // gw, (gx + 1) * source.shape[1] // gw
            cells.append({"cell": f"{gx},{gy}", "mean_diff": round(float(diff[y0:y1, x0:x1].mean()), 2)})

    report = {
        "page_mean_diff": round(float(diff.mean()), 3),
        "threshold": THRESHOLD,
        "nodes": sorted(rows, key=lambda r: -r["mean_diff"]),
        "worst_nodes": sorted(rows, key=lambda r: -r["bg_diff"])[:12],
        "top_contributors": contrib,
        "grid": {"shape": [gh, gw], "cells": cells},
        "over_threshold": [r["id"] for r in rows if r["bg_diff"] > THRESHOLD],
    }
    if out_path:
        save_json(out_path, report)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("measure")
    pm.add_argument("--ir", required=True)
    pm.add_argument("--source", required=True)
    pm.add_argument("--preview", required=True)
    pm.add_argument("--out")

    pp = sub.add_parser("plate")
    pp.add_argument("--ir", required=True)
    pp.add_argument("--source", required=True)
    pp.add_argument("--out", required=True)
    pp.add_argument("--inpaint-url", help="Big-LaMa style POST /inpaint endpoint; falls back to OpenCV TELEA")

    pa = sub.add_parser("apply")
    pa.add_argument("--ir", required=True)
    pa.add_argument("--source", required=True)
    pa.add_argument("--report", required=True)
    pa.add_argument("--plate", required=True)
    pa.add_argument("--out", required=True)
    pa.add_argument("--threshold", type=float, default=THRESHOLD)
    pa.add_argument("--min-contribution", type=float, default=0.0)
    pa.add_argument("--max-raster", type=float, default=1.0,
                    help="cap on the union raster area, as a fraction of the page (e.g. 0.35)")
    pa.add_argument("--no-gradient", action="store_true")
    pa.add_argument("--backfill", action="store_true",
                    help="add plate crops over ink that no node owns, so nothing disappears")
    pa.add_argument("--backfill-min-area", type=int, default=90)
    pa.add_argument("--backfill-budget", type=float, default=0.15,
                    help="cap on the union area added by backfill, as a fraction of the page")

    pc = sub.add_parser("cover")
    pc.add_argument("--ir", required=True)
    pc.add_argument("--source", required=True)
    pc.add_argument("--out")

    pt = sub.add_parser("textaudit")
    pt.add_argument("--ir", required=True)
    pt.add_argument("--source", required=True)
    pt.add_argument("--preview", required=True)

    args = ap.parse_args()

    if args.cmd == "measure":
        report = measure(args.ir, args.source, args.preview, args.out)
        summary = {"page_mean_diff": report["page_mean_diff"],
                   "over_threshold": len(report["over_threshold"]),
                   "worst": [(r["id"], r["mean_diff"]) for r in report["worst_nodes"][:8]],
                   "top_contributors": [(r["id"], r["contribution_to_page_mean"]) for r in report["top_contributors"][:8]]}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "plate":
        ir = load_json(args.ir)
        plate, mask = build_plate(ir, args.source, args.out, inpaint_url=args.inpaint_url)
        print(json.dumps({"plate": args.out, "erased_pixels": int((mask > 0).sum()),
                          "fraction": round(float((mask > 0).mean()), 4)}, ensure_ascii=False))
        return 0

    if args.cmd == "textaudit":
        ir = load_json(args.ir)
        source = rgb(args.source)
        preview = rgb(args.preview)
        if preview.shape != source.shape:
            preview = np.array(Image.fromarray(preview.astype(np.uint8))
                               .resize((source.shape[1], source.shape[0]), Image.LANCZOS)).astype(np.uint8)
        print(json.dumps(text_audit(ir, source, preview), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "cover":
        ir = load_json(args.ir)
        source = rgb(args.source)
        report, missing = coverage_report(ir, source)
        if args.out:
            Image.fromarray(missing).save(args.out)
            report["visual"] = args.out
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    ir = load_json(args.ir)
    source = rgb(args.source)
    plate = rgb(args.plate)
    report = load_json(args.report) if Path(args.report).exists() else {"nodes": []}
    bg_by_id = {r["id"]: r.get("bg_diff", r.get("mean_diff", 0.0)) for r in report.get("nodes", [])}
    contrib_by_id = {r["id"]: r.get("contribution_to_page_mean", 0.0) for r in report.get("nodes", [])}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    import hashlib
    import shutil

    text_mask = crop_masks(ir, source)
    source_rel = Path(ir["source_image"])
    (out_dir / source_rel.parent).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source, out_dir / source_rel)

    reworked = {"gradient": [], "plate_asset": [], "reverted": [], "kept": [], "budget_refused": []}

    # Raster budget: start from the crops the plan already has and only add a revert
    # while the union stays under the cap, so a dense page cannot dissolve into images.
    page_h, page_w = source.shape[:2]
    raster_mask = np.zeros((page_h, page_w), bool)

    def coverage(mask):
        return float(mask.mean())

    def stamp(mask, box):
        x, y, w, h = [int(round(v)) for v in box]
        x, y = max(0, x), max(0, y)
        mask[y:min(page_h, y + h), x:min(page_w, x + w)] = True

    for node in ir["nodes"]:
        if node["type"] == "image":
            stamp(raster_mask, node["bbox"])

    # Decide the revert order up front: biggest contribution to the page mean first.
    candidates = [
        n for n in ir["nodes"]
        if n["type"] == "shape"
        and bg_by_id.get(n["id"], 0.0) > args.threshold
        and contrib_by_id.get(n["id"], 0.0) >= args.min_contribution
    ]
    candidates.sort(key=lambda n: -contrib_by_id.get(n["id"], 0.0))
    allowed = {}
    for node in candidates:
        trial = raster_mask.copy()
        stamp(trial, node_box(node))
        if coverage(trial) > args.max_raster:
            reworked["budget_refused"].append(node["id"])
            continue
        raster_mask = trial
        allowed[node["id"]] = True

    assets = {a["id"]: dict(a) for a in ir["assets"]}
    new_nodes = []

    def save_asset(asset_id, array):
        path = assets_dir / f"{asset_id}.png"
        Image.fromarray(array).save(path)
        assets[asset_id] = {**assets.get(asset_id, {}), "id": asset_id, "path": f"assets/{asset_id}.png",
                            "purpose": assets.get(asset_id, {}).get("purpose", "module"),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "contains_evidence_ids": assets.get(asset_id, {}).get("contains_evidence_ids", []),
                            "description": "Raster crop taken from the de-texted plate; internal pixels are not editable."}
        return f"assets/{asset_id}.png"

    for node in ir["nodes"]:
        node = json.loads(json.dumps(node))
        nid = node["id"]
        box = [int(round(v)) for v in node_box(node)]
        value = bg_by_id.get(nid, 0.0)
        contribution = contrib_by_id.get(nid, 0.0)

        if node["type"] == "text":
            new_nodes.append(node)
            reworked["kept"].append(nid)
            continue

        if node["type"] == "shape" and not args.no_gradient and node.get("geometry") in ("rect", "roundRect", "ellipse"):
            fit = fit_fill(source, text_mask, box)
            if fit and fit["kind"] == "gradient":
                node.pop("fill_gradient", None)
                node["fill_gradient"] = {
                    "angle_deg": 90 if fit["axis"] == "vertical" else 0,
                    "stops": [{"offset": 0, "color": to_hex(fit["start"])},
                              {"offset": 100000, "color": to_hex(fit["end"])}],
                }
                node["fill"] = to_hex(fit["start"])
                reworked["gradient"].append({"id": nid, "axis": fit["axis"], "residual": round(fit["residual"], 2),
                                             "stops": [node["fill_gradient"]["stops"][0]["color"],
                                                       node["fill_gradient"]["stops"][1]["color"]]})
            elif fit and fit["kind"] == "solid":
                node["fill"] = to_hex(fit["color"])

        # Only shapes are reverted: an image node is already a raster crop, and
        # replacing it would orphan the evidence its asset owns.
        if node["type"] == "shape" and allowed.get(nid):
            x, y, w, h = box
            x, y = max(0, x), max(0, y)
            w, h = min(w, source.shape[1] - x), min(h, source.shape[0] - y)
            if w > 1 and h > 1:
                crop = plate[y:y + h, x:x + w]
                asset_id = f"revert-{nid}"
                save_asset(asset_id, crop)
                new_nodes.append({"id": asset_id, "type": "image", "bbox": [x, y, w, h], "z": node.get("z", 0),
                                  "evidence_ids": node.get("evidence_ids", []), "asset_id": asset_id})
                reworked["reverted"].append({"id": nid, "was": node["type"], "bg_diff": round(value, 1),
                                             "contribution": round(contribution, 3)})
                continue

        if node["type"] == "image":
            x, y, w, h = box
            x, y = max(0, x), max(0, y)
            w, h = min(w, source.shape[1] - x), min(h, source.shape[0] - y)
            if w > 1 and h > 1:
                save_asset(node["asset_id"], plate[y:y + h, x:x + w])
                reworked["plate_asset"].append(nid)

        new_nodes.append(node)

    ir["nodes"] = new_nodes
    before_cover, _ = coverage_report(ir, source)
    backfilled = []
    if args.backfill:
        backfilled = backfill_missing(ir, source, plate, save_asset=save_asset,
                                      min_area=args.backfill_min_area,
                                      budget=args.backfill_budget)
        if backfilled:
            renumber_with_backfill(ir, [b["id"] for b in backfilled])
    after_cover, _ = coverage_report(ir, source)
    new_nodes = ir["nodes"]
    used = sorted({n.get("asset_id") for n in new_nodes if n.get("asset_id")})
    ir["assets"] = [assets[a] for a in used]
    save_json(out_dir / "slideir.json", ir)
    authoring = out_dir / "authoring.json"
    shutil.copyfile(Path(args.ir).parent / "authoring.json", authoring)
    doc = load_json(authoring)
    doc["ir_sha256"] = hashlib.sha256((out_dir / "slideir.json").read_bytes()).hexdigest()
    save_json(authoring, doc)

    print(json.dumps({"out": str(out_dir), "nodes": len(new_nodes),
                      "reverted": len(reworked["reverted"]), "gradient": len(reworked["gradient"]),
                      "plate_assets": len(reworked["plate_asset"]),
                      "raster_union": round(coverage(raster_mask), 4),
                      "budget_refused": len(reworked["budget_refused"]),
                      "unowned_ink_before": before_cover["unowned_fraction_of_ink"],
                      "unowned_ink_after": after_cover["unowned_fraction_of_ink"],
                      "backfilled": len(backfilled),
                      "backfilled_ids": [b["id"] for b in backfilled][:10],
                      "reverted_ids": reworked["reverted"][:12]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
