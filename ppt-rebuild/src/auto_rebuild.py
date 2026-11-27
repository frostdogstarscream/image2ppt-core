"""Fully automatic LayerD rebuild: no Cursor, no review JSON, no vision LLM."""
from __future__ import annotations

import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO / "src"))

from font_matcher import fit_visual
from geometry_fit import (MIN_AREA, assign_levels, bbox, fill_holes, measure_component, share)
from image2ppt.compiler import ArtifactBackend, execute
from layerd_loader import load_page, to_record, verify
from layerd_remote import decompose
from object_registry import ObjectRegistry
from ocr_engine import detect, group_lines, to_record as ocr_to_record
from reconstruction import extract_rgba
from slideir_adapter import export_ir, sha256, write_json
from text_mask import remote_inpaint


ICON_MAX = 80
TEXTURE_GRAD = 18.0
TEXTURE_SHARE = 0.35


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]


def text_mask_from_ocr(rgb, lines):
    H, W = rgb.shape[:2]
    mask = np.zeros((H, W), bool)
    for line in lines:
        x, y, w, h = [int(v) for v in line["bbox"]]
        x, y = max(0, x), max(0, y)
        w, h = min(w, W - x), min(h, H - y)
        if w < 3 or h < 3:
            continue
        crop = rgb[y:y + h, x:x + w].astype(np.int16)
        border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
        background = np.median(border, axis=0)
        ink = np.abs(crop - background).max(axis=2) > 40
        if ink.sum() < 8:
            continue
        mask[y:y + h, x:x + w] |= ink
    return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def layer_components(page):
    """Alpha connected components on every non-background LayerD layer."""
    comps = []
    for layer in page.layers:
        rgba = np.array(Image.open(layer.path).convert("RGBA"))
        alpha = rgba[..., 3]
        if layer.z == 0:
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > 24).astype(np.uint8), 8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < MIN_AREA:
                continue
            mask = labels == i
            comps.append({
                "layer_z": layer.z,
                "layer_file": layer.file,
                "mask": fill_holes(mask),
                "alpha": alpha,
                "rgba": rgba,
                "area": area,
                "bbox": bbox(mask),
            })
    comps.sort(key=lambda c: (-c["layer_z"], -c["area"]))
    return comps


def measure_page(rgb, comps, text_mask):
    protect = text_mask
    measured = []
    for index, comp in enumerate(comps):
        children = np.zeros(rgb.shape[:2], bool)
        for other in comps:
            if other is comp:
                continue
            if other["area"] < comp["area"] and share(other["mask"], comp["mask"]) >= 0.9:
                children |= other["mask"]
        # Icons already extracted later; keep residual blobs as children for fill.
        record = measure_component(rgb, comp["mask"] & ~text_mask, protect=protect,
                                   children=children, alpha=comp["alpha"])
        if record:
            record["id"] = f"shape-{index:03d}"
            record["origin"] = "layerd_alpha_measure"
            record["layer_z"] = comp["layer_z"]
            record["source_layers"] = [comp["layer_file"]]
            measured.append(record)
        else:
            mask = comp["mask"] & ~text_mask
            if int(mask.sum()) < MIN_AREA:
                continue
            measured.append({
                "id": f"image-{index:03d}",
                "kind": "raster",
                "role": "block",
                "bbox": bbox(mask),
                "area": int(mask.sum()),
                "mask": mask,
                "alpha": "mask",
                "rgba": comp["rgba"],
                "layer_z": comp["layer_z"],
                "origin": "layerd_alpha_picture",
                "source_layers": [comp["layer_file"]],
                "fallback": True,
            })
    # Residual icons on native shapes.
    icons = []
    for shape in [m for m in measured if m["kind"] == "native_shape"]:
        for blob in shape.pop("icons", []):
            if int(blob.sum()) < 20:
                continue
            icons.append({
                "id": f"icon-{len(icons):03d}",
                "kind": "raster",
                "role": "icon",
                "bbox": bbox(blob),
                "area": int(blob.sum()),
                "mask": blob,
                "alpha": "mask",
                "origin": "shape_residual",
                "layer_z": shape["layer_z"],
                "source_layers": shape["source_layers"],
            })
    # Compact high-contrast pieces that are not already owned.
    owned = np.zeros(rgb.shape[:2], bool)
    for item in measured + icons:
        owned |= item["mask"]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    smooth = cv2.cvtColor(cv2.medianBlur(rgb, 61), cv2.COLOR_RGB2LAB).astype(np.float32)
    fg = (np.linalg.norm(lab - smooth, axis=2) > 26) & ~owned & ~text_mask
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < 25 or max(w, h) > ICON_MAX or min(w, h) < 5:
            continue
        blob = labels == i
        icons.append({
            "id": f"icon-{len(icons):03d}",
            "kind": "raster",
            "role": "icon",
            "bbox": [x, y, w, h],
            "area": area,
            "mask": blob,
            "alpha": "mask",
            "origin": "contrast_icon",
            "layer_z": 1,
            "source_layers": [],
        })
    elements = measured + icons
    for k, e in enumerate(elements):
        e["id"] = e.get("id") or f"el-{k:03d}"
    assign_levels(elements)
    return elements


def lower_composite(layers, size):
    """Flatten LayerD layers back-to-front into an RGB page."""
    height, width = size[1], size[0]
    canvas = np.full((height, width, 3), 255, np.float32)
    for rgba in layers:
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        canvas = rgba[..., :3].astype(np.float32) * alpha + canvas * (1.0 - alpha)
    return np.clip(canvas, 0, 255).round().astype(np.uint8)


def gradient(rgb):
    blur = cv2.GaussianBlur(rgb, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt((gx ** 2).sum(2) + (gy ** 2).sum(2))


def hole_is_textured(rgb, hole, *, skip=7):
    if not hole.any():
        return False
    rim = cv2.dilate(hole.astype(np.uint8), np.ones((2 * skip + 1, 2 * skip + 1), np.uint8)) > 0
    known = rim & ~hole
    if known.sum() < 32:
        return True
    return float((gradient(rgb)[known] > TEXTURE_GRAD).mean()) >= TEXTURE_SHARE


def build_plate(source, layers, elements, text_mask, *, inpaint_url, inpaint=None):
    """Background keeps lower-layer pixels; textured holes may call BigLaMa."""
    inpaint = inpaint or remote_inpaint
    lower = layers[0][..., :3].copy() if layers else source.copy()
    if layers and layers[0].shape[2] == 4 and not np.all(layers[0][..., 3] == 255):
        lower = lower_composite(layers, (source.shape[1], source.shape[0]))
    plate = source.copy()
    erase = np.zeros(source.shape[:2], bool)
    for e in elements:
        erase |= e["mask"]
    erase |= text_mask
    calls = []
    if erase.any():
        if hole_is_textured(source, erase) and inpaint_url:
            patched = inpaint(source, erase, inpaint_url, timeout=300)
            plate[erase] = patched[erase]
            calls.append({"method": "big_lama_textured_hole", "pixels": int(erase.sum())})
        else:
            plate[erase] = lower[erase]
            calls.append({"method": "layerd_lower_pixels", "pixels": int(erase.sum())})
    return plate, calls


def text_nodes(rgb, plate, lines, occupied):
    nodes = []
    for line in lines:
        bbox_px = [int(v) for v in line["bbox"]]
        x, y, w, h = bbox_px
        if w < 4 or h < 4:
            continue
        own = _rect(bbox_px, (rgb.shape[1], rgb.shape[0]))
        if occupied.any() and share(own, occupied) >= 0.95:
            continue
        estimate = plate[y:y + h, x:x + w]
        fit = fit_visual(rgb, estimate, line["text"], bbox_px, dpi=96)
        if not fit.get("size_pt"):
            continue
        crop = rgb[y:y + h, x:x + w]
        difference = np.abs(crop.astype(np.int16) - estimate.astype(np.int16)).max(axis=2)
        local = difference > 30
        if local.sum() < 6:
            continue
        core = crop[difference >= np.percentile(difference[local], 85)]
        color = np.median(core, axis=0).round().astype(int)
        nodes.append({
            "id": "text-" + line["id"],
            "kind": "text",
            "bbox": bbox_px,
            "text": line["text"],
            "ocr_score": line["score"],
            "mask": own,
            "local_mask": local,
            "repair": estimate,
            "node": {
                "id": "text-" + line["id"],
                "type": "text",
                "bbox": bbox_px,
                "text": line["text"],
                "font": {"latin": "Arial", "ea": fit.get("family") or "微软雅黑",
                         "size_pt": fit["size_pt"]},
                "bold": bool(fit.get("bold")),
                "color": "".join(f"{v:02X}" for v in color),
                "align": "left",
                "wrap": False,
                "vertical_anchor": "middle",
                "evidence_ids": [f"ev-text-{line['id']}"],
            },
        })
    return nodes


def _rect(box, size):
    x, y, w, h = [int(v) for v in box]
    mask = np.zeros((size[1], size[0]), bool)
    mask[y:y + h, x:x + w] = True
    return mask


def element_replacements(source, plate, elements):
    replacements = {}
    for e in elements:
        x, y, w, h = e["bbox"]
        mask = e["mask"]
        local = mask[y:y + h, x:x + w]
        repair = plate[y:y + h, x:x + w].copy()
        if e["kind"] == "native_shape":
            node = {
                "id": e["id"],
                "type": "shape",
                "bbox": e["bbox"],
                "geometry": e["geometry"],
                "fill": e["fill"],
                "evidence_ids": [f"ev-{e['id']}"],
            }
            if e.get("corner_radius_px"):
                node["corner_radius_px"] = e["corner_radius_px"]
            if e.get("fill_gradient"):
                node["fill_gradient"] = e["fill_gradient"]
            if e.get("line"):
                node["line"] = {"color": e["line"]["color"],
                                "width_pt": round(e["line"]["width_px"] * 0.75, 3)}
            asset = None
        else:
            rgba_page = e.get("rgba")
            if rgba_page is not None:
                crop = rgba_page[y:y + h, x:x + w].copy()
                crop[~local] = 0
                asset = Image.fromarray(crop)
            else:
                asset = Image.fromarray(extract_rgba(source[y:y + h, x:x + w], repair, local))
            node = {
                "id": e["id"],
                "type": "image",
                "bbox": e["bbox"],
                "asset_id": e["id"],
                "evidence_ids": [f"ev-{e['id']}"],
            }
        replacements[e["id"]] = {
            "node": node,
            "bbox": list(e["bbox"]),
            "asset": asset,
            "mask": mask,
            "local_mask": local,
            "repair": repair,
            "kind": "shape" if e["kind"] == "native_shape" else "image",
            "label": e.get("origin", e["id"]),
            "level": e.get("level", 0),
        }
    return replacements


def compile_with_gate(directory, registry, replacements, source, *, mae_limit=10.0, timeout=600, probes=True):
    from rebuild_pipeline import compile_ir, edit_probes, measurements, pixels

    backend = ArtifactBackend(REPO, timeout)
    active = set(replacements)
    attempts = []
    notes = [
        "Automatic LayerD rebuild; deterministic paragraph grouping; no vision LLM",
        "Semi-transparent LayerD regions stay pictures; flat holes use lower-layer pixels",
    ]
    result = None
    revision = None
    for index in range(4):
        revision = directory / f"revisions/r{index:03d}"
        (revision / "source").mkdir(parents=True)
        shutil.copy2(directory / "source.png", revision / "source/page.png")
        plate = source.copy()
        nodes, assets = [], {}
        ordered = sorted(active, key=lambda i: (replacements[i].get("level", 0),
                                                replacements[i]["kind"] != "shape",
                                                -replacements[i]["bbox"][2] * replacements[i]["bbox"][3]))
        for object_id in ordered:
            replacement = replacements[object_id]
            x, y, w, h = replacement["bbox"]
            region = plate[y:y + h, x:x + w]
            region[replacement["local_mask"]] = replacement["repair"][replacement["local_mask"]]
            nodes.append(replacement["node"])
            if replacement["asset"] is not None:
                assets[replacement["node"]["asset_id"]] = replacement["asset"]
        ir = export_ir(revision, registry, nodes, Image.fromarray(plate), assets, notes=notes)
        write_json(revision / "registry-trial.json", registry.record())
        print(f"[auto] compile r{index:03d} objects={len(nodes)}", flush=True)
        result = compile_ir(backend, ir)
        rendered = pixels(result["preview"], registry.size)
        metrics = measurements(source, rendered, registry, active)
        failed = {r["id"] for r in metrics["regions"] if r["active"] and r["mae"] > mae_limit}
        attempts.append({"revision": str(revision), "ir": str(ir), "build": result, "metrics": metrics})
        write_json(revision / "evaluation.json", metrics)
        for entry in registry.objects:
            if entry["state"] == "matched":
                registry.transition(entry, "trial_rendered", revision=str(revision),
                                    pptx_sha256=sha256(result["pptx"]))
        if not failed:
            break
        for entry in registry.objects:
            if entry["id"] in failed and entry["id"] in active:
                registry.transition(entry, "reverted", reason="actual_pptx_roi_mae_exceeded",
                                    mae_limit=mae_limit)
                active.remove(entry["id"])
    else:
        raise RuntimeError("ROI gate did not converge; no committed delivery was produced")
    for entry in registry.objects:
        if entry["id"] in active:
            registry.transition(entry, "verified", revision=str(revision), mae_limit=mae_limit)
    probe_results = edit_probes(result, registry, replacements, active, directory / "edit-probes", backend) if probes else []
    if any(p["status"] == "FAIL" for p in probe_results):
        write_json(directory / "registry-uncommitted.json", registry.record())
        raise RuntimeError("direct PPTX edit probe failed; trial retained without commit")
    for entry in registry.objects:
        if entry["id"] in active:
            registry.transition(entry, "committed", revision=str(revision), pptx_sha256=sha256(result["pptx"]))
    registry.validate_owners(json.loads(ir.read_text(encoding="utf-8")))
    write_json(directory / "registry.json", registry.record())
    write_json(directory / "operations.json", registry.operations)
    counts = {kind: sum(e["kind"] == kind and e["id"] in active for e in registry.objects)
              for kind in ("text", "shape", "image")}
    return {
        "ir": str(ir),
        "pptx": result["pptx"],
        "preview": result["preview"],
        "quality": result.get("quality"),
        "metrics": metrics,
        "attempts": attempts,
        "edit_probes": probe_results,
        "counts": counts,
        "active": sorted(active),
        "pptx_sha256": sha256(result["pptx"]),
    }


def run_auto(image, output, *, pack=None, layerd_url="ssh://CMCC@192.168.1.103:18840",
             token_file=None, inpaint_url="ssh://CMCC@192.168.1.103:18841/inpaint",
             max_iterations=3, timeout=1200, probes=True, inpaint=None):
    """Build one page automatically. ``pack`` skips LayerD when a pack already exists."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    directory = output / f"auto-{stamp()}"
    directory.mkdir()
    image = Path(image).resolve(strict=True)
    shutil.copy2(image, directory / "input.png")
    if pack:
        pack_dir = Path(pack).resolve()
    else:
        if token_file is None:
            token_file = REPO / ".runtime" / "layerd" / "client-token"
        pack_dir, manifest = decompose(
            image, directory / f"layerd-{stamp()}",
            endpoint=layerd_url, token_file=token_file,
            max_iterations=max_iterations, timeout=timeout,
        )
        write_json(directory / "layerd-manifest.json", manifest)
    page = load_page(pack_dir)
    check = verify(page)
    write_json(directory / "input.json", {"pack": str(page.directory), **to_record(page), "recomposition": check})
    shutil.copy2(page.source_path, directory / "source.png")
    source = np.array(Image.open(page.source_path).convert("RGB"))
    layers = [np.array(Image.open(layer.path).convert("RGBA")) for layer in page.layers]
    lines = detect(page.source_path, min_score=0.0)
    ocr = ocr_to_record(lines, group_lines(lines), page.size)
    ocr["source_sha256"] = sha256(page.source_path)
    write_json(directory / "ocr.json", ocr)
    text_mask = text_mask_from_ocr(source, ocr["lines"])
    Image.fromarray((text_mask.astype(np.uint8) * 255)).save(directory / "text-mask.png")
    comps = layer_components(page)
    elements = measure_page(source, comps, text_mask)
    write_json(directory / "components.json", {
        "count": len(elements),
        "native_shapes": sum(e["kind"] == "native_shape" for e in elements),
        "rasters": sum(e["kind"] != "native_shape" for e in elements),
        "objects": [{k: v for k, v in e.items() if k not in ("mask", "rgba", "alpha", "icons", "geometry_fit", "fill_fit")}
                    for e in elements],
    })
    plate, fill_calls = build_plate(source, layers, elements, text_mask,
                                    inpaint_url=inpaint_url, inpaint=inpaint)
    Image.fromarray(plate).save(directory / "plate-draft.png")
    write_json(directory / "fill-calls.json", fill_calls)
    replacements = element_replacements(source, plate, elements)
    occupied = np.zeros(source.shape[:2], bool)
    for e in elements:
        if e["kind"] != "native_shape":
            occupied |= e["mask"]
    for text in text_nodes(source, plate, ocr["lines"], occupied):
        replacements[text["id"]] = {
            "node": text["node"],
            "bbox": text["bbox"],
            "asset": None,
            "mask": text["mask"],
            "local_mask": text["local_mask"],
            "repair": text["repair"],
            "kind": "text",
            "label": "local_ocr",
            "level": 10,
        }
    # Apply committed masks onto the plate once more so text holes are clean.
    for object_id, replacement in replacements.items():
        x, y, w, h = replacement["bbox"]
        plate[y:y + h, x:x + w][replacement["local_mask"]] = replacement["repair"][replacement["local_mask"]]
    Image.fromarray(plate).save(directory / "plate.png")
    registry = ObjectRegistry(page.size, sha256(page.source_path))
    for object_id, replacement in replacements.items():
        entry = registry.add({
            "id": object_id,
            "kind": replacement["kind"],
            "bbox": replacement["bbox"],
            "label": replacement.get("label", object_id),
            "text": replacement["node"].get("text") if replacement["kind"] == "text" else None,
            "content_checked": False,
        })
        registry.transition(entry, "matched", repair_method="auto_layerd",
                            mask_origin=replacement["bbox"][:2],
                            mask_size=replacement["bbox"][2:])
        replacement["node"]["evidence_ids"] = [entry["evidence_id"]]
    built = compile_with_gate(directory, registry, replacements, source,
                              timeout=timeout, probes=probes)
    summary = {
        "schema": "LayerDAutoDelivery/1",
        "directory": str(directory),
        "source": str(image),
        "source_sha256": sha256(page.source_path),
        "pack": str(page.directory),
        "recomposition": check,
        "fill_calls": fill_calls,
        "component_counts": {
            "native_shapes": sum(1 for e in elements if e["kind"] == "native_shape"),
            "rasters": sum(1 for e in elements if e["kind"] != "native_shape"),
            "text": built["counts"].get("text", 0),
        },
        **{k: built[k] for k in ("ir", "pptx", "preview", "metrics", "counts", "pptx_sha256", "edit_probes")},
        "status": "REVIEW_REQUIRED",
        "powerPoint_checked": False,
        "wps_checked": False,
        "human_acceptance": False,
        "office_acceptance": False,
    }
    write_json(directory / "delivery.json", summary)
    print(json.dumps({k: summary[k] for k in ("directory", "pptx", "counts", "status")}, ensure_ascii=False), flush=True)
    return summary
