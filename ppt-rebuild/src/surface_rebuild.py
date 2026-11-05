"""Versioned card/background reconstruction using the fixed SlideIR compiler."""
from __future__ import annotations

import argparse
import copy
import json
import hashlib
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np
from PIL import Image, ImageDraw

from background_models import (color_hex, fit_surface, gradient_pixels, inspect_residue,
                               native_fill, precise_text_masks, shape_mask, edge_profiles, repair_seam_quality)
from rebuild_pipeline import ArtifactBackend, REPO, ROOT, compile_ir, pixels, stamp
from reconstruction import intersect, padded, rectangle, extract_rgba
from slideir_adapter import sha256, write_json
from text_repair import NS, load_json, delete_text_copy
from ocr_engine import detect
from simple_icons import fit_organization, native_organization, contour_metrics


def pipeline_fingerprint():
    digest = hashlib.sha256()
    for name in ("surface_rebuild.py", "background_models.py", "simple_icons.py"):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def ancestry(regions, region):
    by_id = {item["id"]: item for item in regions}
    parents = []
    parent = region.get("parent_id")
    while parent:
        if parent in parents or parent not in by_id:
            raise ValueError("invalid surface hierarchy")
        parents.append(parent)
        parent = by_id[parent].get("parent_id")
    return parents


def rollback_review(review, failed_ids, reason):
    """Restore the full containing r3 card when one nested surface fails."""
    by_id = {region["id"]: region for region in review["regions"]}
    roots = set()
    for identifier in failed_ids:
        region = by_id[identifier]
        parents = ancestry(review["regions"], region)
        roots.add(parents[-1] if parents else identifier)
    revised = copy.deepcopy(review)
    revised.setdefault("baseline_card_fallbacks", []).extend(
        {**by_id[identifier], "reason": reason, "status": "R3_RETAINED_REVIEW_REQUIRED"}
        for identifier in sorted(roots))
    revised["regions"] = [region for region in revised["regions"]
                          if region["id"] not in roots
                          and not roots.intersection(ancestry(review["regions"], region))]
    return revised


def foreground_exclusions(ir, inventory, size):
    excluded = np.zeros((size[1], size[0]), bool)
    for line in inventory["lines"]:
        excluded |= rectangle(padded(line["bbox"], size, 4), size)
    for node in ir["nodes"]:
        if node["type"] == "image" and node["id"] != "background-image":
            excluded |= rectangle(padded(node["bbox"], size, 3), size)
        elif node["type"] == "shape" and node.get("fill") != "none":
            excluded |= rectangle(padded(node["bbox"], size, 2), size)
    return excluded


def surface_components(region):
    x, y, width, height = region["bbox"]
    result = [{"id": region["id"] + "-body", "bbox": region["bbox"],
               "radius": region["radius"], "part": "body"}]
    if region.get("header_height"):
        result.append({"id": region["id"] + "-header", "bbox": [x, y, width, region["header_height"]],
                       "radius": region["radius"], "part": "top"})
    if region.get("rail_width"):
        result.append({"id": region["id"] + "-rail", "bbox": [x, y, region["rail_width"], height],
                       "radius": region["radius"], "part": "left"})
    return result


def fit_region(source, region, regions, exclusions):
    size = (source.shape[1], source.shape[0])
    region_mask = shape_mask(size, region["bbox"], region["radius"])
    children = np.zeros(region_mask.shape, bool)
    for child in regions:
        if child.get("parent_id") == region["id"]:
            children |= rectangle(padded(child["bbox"], size, 2), size)
    models = []
    components = surface_components(region)
    local_detail = np.abs(source.astype(np.int16) - cv2.GaussianBlur(source, (5, 5), 0.8).astype(np.int16)).max(axis=2)
    for component in components:
        mask = shape_mask(size, component["bbox"], component["radius"])
        allowed = cv2.erode(mask.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
        allowed &= ~children
        if component["part"] == "body":
            for other in components[1:]:
                allowed &= ~rectangle(padded(other["bbox"], size, 3), size)
        if allowed.any():
            dominant = np.median(source[allowed], axis=0)
            background_colored = np.abs(source.astype(np.int16) - dominant).max(axis=2) <= 30
            # Clean gaps inside a dense text row are useful source samples.
            # The color/contrast tests exclude ink and separator antialiasing.
            allowed &= ((~exclusions) | background_colored) & background_colored & (local_detail <= 3)
        model, score = fit_surface(source, component["bbox"], allowed)
        models.append({**component, "model": model, "fit": score, "sample_mask": allowed})
    return region_mask, models


def model_nodes(region, models):
    nodes = []
    for component in models:
        node = {"id": component["id"], "type": "shape", "bbox": component["bbox"],
                "geometry": "roundRect" if component["radius"] else "rect",
                "corner_radius_px": component["radius"], "evidence_ids": [],
                **native_fill(component["model"])}
        nodes.append(node)
        if component["part"] in {"top", "left"}:
            x, y, width, height = node["bbox"]
            radius = component["radius"]
            # A second native rectangle squares only the interior corners.
            patch = [x, y + radius, width, height - radius] if component["part"] == "top" else [x + radius, y, width - radius, height]
            nodes.append({**copy.deepcopy(node), "id": node["id"] + "-interior", "bbox": patch,
                          "geometry": "rect", "corner_radius_px": 0})
    if region.get("border"):
        nodes.append({"id": region["id"] + "-border", "type": "shape", "bbox": region["bbox"],
                      "geometry": "roundRect", "corner_radius_px": region["radius"],
                      "fill": "none", "line": {"color": region["border_color"], "width_pt": 0.6},
                      "evidence_ids": []})
    nodes[0]["evidence_ids"] = ["ev-" + region["id"]]
    return nodes


def clean_page_plate(original, region_union, source, exclusions):
    """Restore a continuous neutral page surface and retain colored decoration."""
    height, width = original.shape[:2]
    clean = (source.min(axis=2) > 245) & (source.max(axis=2) - source.min(axis=2) < 7)
    clean &= ~region_union & ~exclusions
    model, score = fit_surface(source, [0, 0, width, height], clean)
    neutral = gradient_pixels(model, [0, 0, width, height])
    difference = np.abs(original.astype(np.int16) - neutral.astype(np.int16)).max(axis=2)
    preservation = (difference > 4) & ~region_union
    count, labels, stats, _ = cv2.connectedComponentsWithStats(preservation.astype(np.uint8), 8)
    strong_labels = set(np.unique(labels[(difference >= 16) & ~region_union]).tolist())
    selected = [index for index in range(1, count)
                if stats[index, cv2.CC_STAT_AREA] >= 30 and index in strong_labels]
    preservation = np.isin(labels, selected)
    core = preservation & (difference >= 16)
    fixed = cv2.dilate(core.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~region_union
    distance = cv2.distanceTransform((~fixed).astype(np.uint8), cv2.DIST_L2, 3)
    amount = np.clip(1 - distance / 6, 0, 1)
    amount[region_union] = 0
    result = np.rint(neutral.astype(np.float32) * (1 - amount[:, :, None])
                     + original.astype(np.float32) * amount[:, :, None]).astype(np.uint8)
    result[fixed] = original[fixed]
    return result, ~fixed, model, score


def border_color(source, region):
    x, y, width, height = region["bbox"]
    radius = region["radius"]
    # Side/bottom strokes are visible even when a blue header covers the top.
    values = np.concatenate((source[y + radius:y + height - radius, x:x + 2].reshape(-1, 3),
                             source[y + height - 2:y + height, x + radius:x + width - radius].reshape(-1, 3)))
    blue = values[(values[:, 2].astype(int) - values[:, 0].astype(int)) > 15]
    return color_hex(np.percentile(blue if len(blue) else values, 20, axis=0))


def extra_geometry(source, expected, exclusions, union, regions, old_nodes):
    """Recover source separators/bullets after the containing surface is owned."""
    size = (source.shape[1], source.shape[0])
    allowed = union & ~exclusions
    for region in regions:
        edge = shape_mask(size, region["bbox"], region["radius"])
        inner = cv2.erode(edge.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
        allowed &= ~(edge & ~inner)
    for node in old_nodes:
        if node["type"] == "shape":
            allowed &= ~rectangle(padded(node["bbox"], size, 2), size)
    delta = np.abs(source.astype(np.int16) - expected.astype(np.int16)).max(axis=2)
    ink = ((delta > 25) & allowed).astype(np.uint8)
    result = []
    for axis, kernel in (("v", (1, 24)), ("h", (32, 1))):
        opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((kernel[1], kernel[0]), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if (axis == "v" and width > 5) or (axis == "h" and height > 5):
                continue
            color = color_hex(np.median(source[labels == index], axis=0))
            result.append({"id": f"separator-{axis}-{len(result):03d}", "type": "shape", "geometry": "rect",
                           "bbox": [int(x), int(y), int(width), int(height)], "fill": color, "evidence_ids": []})
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if 5 <= width <= 16 and 5 <= height <= 16 and area / (width * height) > 0.65:
            result.append({"id": f"bullet-{len(result):03d}", "type": "shape", "geometry": "rect",
                           "bbox": [int(x), int(y), int(width), int(height)],
                           "fill": color_hex(np.median(source[labels == index], axis=0)), "evidence_ids": []})
    return result


def rewrite_package(source, target, *, keep=None, remove=None, transform=None):
    with ZipFile(source) as original:
        root = ET.fromstring(original.read("ppt/slides/slide1.xml"))
        tree = root.find("p:cSld/p:spTree", NS)
        for element in list(tree):
            nv = element.find(".//p:cNvPr", NS)
            if nv is None:
                continue
            name = nv.get("name")
            if (keep is not None and name not in keep) or (remove and name in remove):
                tree.remove(element)
            elif transform and name == transform["id"]:
                props = element.find("p:grpSpPr", NS) if element.tag.endswith("grpSp") else element.find("p:spPr", NS)
                xfrm = props.find("a:xfrm", NS)
                off = xfrm.find("a:off", NS)
                off.set("x", str(int(off.get("x")) + 20 * 9525))
                off.set("y", str(int(off.get("y")) + 12 * 9525))
                extent = xfrm.find("a:ext", NS)
                extent.set("cx", str(round(int(extent.get("cx")) * 0.9)))
                extent.set("cy", str(round(int(extent.get("cy")) * 0.9)))
        with ZipFile(target, "x", compression=ZIP_DEFLATED) as output:
            for entry in original.infolist():
                output.writestr(entry, ET.tostring(root, encoding="utf-8", xml_declaration=True)
                                if entry.filename == "ppt/slides/slide1.xml" else original.read(entry.filename))


def surface_quality(source, rendered, sample, bbox):
    x, y, width, height = bbox
    original = source[y:y + height, x:x + width].astype(np.float32)
    actual = rendered[y:y + height, x:x + width].astype(np.float32)
    valid = sample[y:y + height, x:x + width]
    weights = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 2)
    smooth_source = cv2.GaussianBlur(original * valid[:, :, None], (0, 0), 2) / np.maximum(weights[:, :, None], 1e-6)
    smooth_actual = cv2.GaussianBlur(actual * valid[:, :, None], (0, 0), 2) / np.maximum(weights[:, :, None], 1e-6)
    differences = np.abs(smooth_actual - smooth_source)[valid]
    raw = np.abs(actual - original)[valid]
    return {"mae": round(float(differences.mean()), 4), "p95": round(float(np.percentile(differences, 95)), 4),
            "raw_mae": round(float(raw.mean()), 4), "raw_p95": round(float(np.percentile(raw, 95)), 4),
            "method": "normalized_background_only_gaussian_sigma2"}


def make_revision(base_file, directory, plate, new_nodes, groups, regions, notes):
    base_file = Path(base_file)
    directory.mkdir()
    shutil.copytree(base_file.parent / "source", directory / "source")
    shutil.copytree(base_file.parent / "assets", directory / "assets")
    ir = copy.deepcopy(load_json(base_file))
    asset = next(asset for asset in ir["assets"] if asset["id"] == "plate")
    Image.fromarray(plate).save(directory / asset["path"])
    asset["sha256"] = sha256(directory / asset["path"])
    # Existing evidence stays with its prior owners. Newly inventoried background
    # surfaces get explicit evidence rather than reusing foreground text IDs.
    for region in regions:
        ir["evidence"].append({"id": "ev-" + region["id"], "kind": "shape", "bbox": region["bbox"],
                               "description": "Reviewed card background surface excluding foreground children"})
    ir["nodes"] = [ir["nodes"][0], *new_nodes, *ir["nodes"][1:]]
    for index, node in enumerate(ir["nodes"]):
        node["z"] = index
    ir["groups"] = [*groups, *ir["groups"]]
    ir["review_notes"] = [*ir["review_notes"], *notes]
    target = directory / "slideir.json"
    write_json(target, ir)
    write_json(directory / "authoring.json", {"tool": "Codex surface reconstruction", "model_id": "current-conversation",
               "created_at": stamp(), "ir_sha256": sha256(target), "decisions": notes})
    return target


def try_simple_icons(ir_file, result, source, review, directory, backend):
    records = []
    current = load_json(ir_file)
    candidate = copy.deepcopy(current)
    trials = []
    for node_id in review.get("icon_candidates", []):
        node = next(item for item in current["nodes"] if item["id"] == node_id)
        if node_id != "icon-organization":
            records.append({"id": node_id, "status": "PNG_RETAINED",
                            "reason": "source contour needs additional validated primitive template"})
            continue
        x, y, width, height = map(int, node["bbox"])
        crop = source[y:y + height, x:x + width]
        parameters, metrics, reference = fit_organization(crop)
        saved = review.get("primitive_parameters", {}).get(node_id)
        if saved:
            parameters = np.asarray(saved["parameters"], dtype=float)
        if metrics["iou"] < 0.85 or metrics["contour_p95"] > 1.5:
            records.append({"id": node_id, "status": "PNG_RETAINED", "reason": "primitive_contour_gate", **metrics})
            continue
        ink = crop[(crop[:, :, 2].astype(int) - crop[:, :, 0].astype(int)) > 100]
        color = saved["color"] if saved else color_hex(np.median(ink, axis=0))
        primitives = native_organization(node_id, node["bbox"], parameters, color)
        asset = next(item for item in candidate["assets"] if item["id"] == node["asset_id"])
        primitives[0]["evidence_ids"] = asset["contains_evidence_ids"]
        old_group = next(group for group in candidate["groups"] if node_id in group["member_ids"])
        index = next(index for index, item in enumerate(candidate["nodes"]) if item["id"] == node_id)
        candidate["nodes"] = [item for item in candidate["nodes"] if item["id"] not in old_group["member_ids"]]
        candidate["nodes"][index:index] = primitives
        old_group["member_ids"] = [item["id"] for item in primitives]
        candidate["assets"].remove(asset)
        trials.append({"id": node_id, "bbox": node["bbox"], "reference": reference, "parameters": parameters.tolist(),
                       "template_metrics": metrics, "primitive_count": len(primitives)})
    if trials:
        for index, node in enumerate(candidate["nodes"]):
            node["z"] = index
        trial_dir = directory / "icon-trial"
        trial_dir.mkdir()
        for name in ("source", "assets"):
            shutil.copytree(Path(ir_file).parent / name, trial_dir / name)
        trial_ir = trial_dir / "slideir.json"
        write_json(trial_ir, candidate)
        write_json(trial_dir / "authoring.json", {"tool": "Codex primitive icon trial", "model_id": "current-conversation",
                   "created_at": stamp(), "ir_sha256": sha256(trial_ir), "decisions": ["Source blue-contour fitting with preserved white halo"]})
        trial_result = compile_ir(backend, trial_ir)
        rendered = pixels(trial_result["preview"], (source.shape[1], source.shape[0]))
        all_passed = True
        for trial in trials:
            x, y, width, height = map(int, trial["bbox"])
            crop = rendered[y:y + height, x:x + width]
            blue = (crop[:, :, 2].astype(int) - crop[:, :, 0].astype(int) > 70)
            blue &= (crop[:, :, 2].astype(int) - crop[:, :, 1].astype(int) > 25)
            metrics = contour_metrics(trial["reference"], blue)
            mae = float(np.abs(crop.astype(np.int16) - source[y:y + height, x:x + width].astype(np.int16)).mean())
            passed = metrics["iou"] >= 0.92 and metrics["contour_p95"] <= 1.5 and mae <= 10
            all_passed &= passed
            records.append({key: value for key, value in trial.items() if key != "reference"} |
                           {"actual_metrics": metrics, "actual_mae": mae,
                            "status": "NATIVE" if passed else "PNG_RETAINED", "trial_ir": str(trial_ir.resolve())})
        if all_passed:
            ir_file, result = trial_ir, trial_result
    write_json(directory / "icon-conversions.json", records)
    return ir_file, result, records


def calibrate_contours(ir_file, source, rendered, regions, directory, backend):
    authored = load_json(ir_file)
    adjustments = []
    for region in regions:
        profiles = edge_profiles(source, rendered, region["bbox"])
        deltas = {side: float(np.clip(value["delta"], -3.5, 3.5)) for side, value in profiles.items()}
        if max(abs(value) for value in deltas.values()) < 0.4:
            continue
        x, y, width, height = region["bbox"]
        left, top = deltas["left"], deltas["top"]
        right, bottom = deltas["right"], deltas["bottom"]
        new_width, new_height = width + right - left, height + bottom - top
        if min(new_width, new_height) <= 0:
            continue
        for node in authored["nodes"]:
            if node["id"].startswith(region["id"] + "-"):
                nx, ny, nw, nh = node["bbox"]
                node["bbox"] = [x + left + (nx - x) * new_width / width,
                                y + top + (ny - y) * new_height / height,
                                nw * new_width / width, nh * new_height / height]
        adjustments.append({"id": region["id"], "profiles": profiles, "edge_delta_px": deltas})
    write_json(directory / "contour-adjustments.json", adjustments)
    if not adjustments:
        return ir_file, None
    revision = directory / "contour-calibrated"
    revision.mkdir()
    for name in ("source", "assets"):
        shutil.copytree(Path(ir_file).parent / name, revision / name)
    target = revision / "slideir.json"
    write_json(target, authored)
    write_json(revision / "authoring.json", {"tool": "Codex source-edge contour calibration", "model_id": "current-conversation",
               "created_at": stamp(), "ir_sha256": sha256(target), "decisions": ["Bounded source-vs-final-PPTX contour adjustment; text and icon positions preserved"]})
    return target, compile_ir(backend, target)


def rebuild_page(delivery, review, output, timeout=600):
    directory = Path(output) / (delivery["page"] + "-" + stamp())
    directory.mkdir(parents=True)
    ir = load_json(delivery["ir"])
    size = ir["canvas"]["source_px"]
    source_file = Path(delivery["ir"]).parent / ir["source_image"]
    if review["source_sha256"] != sha256(source_file) or review["image_size"] != size:
        raise ValueError("surface review source mismatch")
    source = pixels(source_file, size)
    plate_asset = next(asset for asset in ir["assets"] if asset["id"] == "plate")
    previous = pixels(Path(delivery["ir"]).parent / plate_asset["path"], size)
    inventory = load_json(Path(delivery["directory"]) / "text-inventory.json")
    exclusions = foreground_exclusions(ir, inventory, size)
    original_review = load_json(ROOT / "configs/reviews" / (delivery["page"][:2] + ".json"))
    existing_ids = {node["id"] for node in ir["nodes"]}
    missing_visuals = [item for item in original_review["objects"]
                       if item["kind"] == "image" and item["id"] not in existing_ids]
    fallback_mask = np.zeros(source.shape[:2], bool)
    for fallback in review.get("baseline_card_fallbacks", []):
        fallback_mask |= rectangle(padded(fallback["bbox"], size, 3), size)
    missing_visuals = [item for item in missing_visuals
                       if not rectangle(item["bbox"], size)[fallback_mask].any()]
    for visual in missing_visuals:
        exclusions |= rectangle(padded(visual["bbox"], size, 3), size)
    regions = sorted(copy.deepcopy(review["regions"]), key=lambda r: (len(ancestry(review["regions"], r)), r["bbox"][1], r["bbox"][0]))
    union = np.zeros(source.shape[:2], bool)
    models = []
    failures = []
    for region in regions:
        region["border_color"] = border_color(source, region)
        try:
            mask, components = fit_region(source, region, regions, exclusions)
            models.append((region, mask, components))
            # The source corner radius and stroke antialiasing can extend past
            # the fitted silhouette. Remove their complete reviewed footprint.
            union |= rectangle(padded(region["bbox"], size, 2), size)
        except ValueError as error:
            failures.append({"id": region["id"], "reason": str(error)})
    plate, authorized, page_model, page_score = clean_page_plate(previous, union, source, exclusions)
    plate[fallback_mask] = previous[fallback_mask]
    authorized[fallback_mask] = False
    expected = plate.copy()
    new_nodes = []
    groups = []
    records = []
    for region, mask, components in models:
        nodes = model_nodes(region, components)
        new_nodes.extend(nodes)
        if len(nodes) > 1:
            groups.append({"id": region["id"] + "-background", "member_ids": [node["id"] for node in nodes]})
        output_id = region["id"] + "-background" if len(nodes) > 1 else nodes[0]["id"]
        for component in components:
            x, y, width, height = component["bbox"]
            local = gradient_pixels(component["model"], component["bbox"])
            fill_mask = shape_mask(size, component["bbox"], component["radius"])
            if component["part"] == "top":
                fill_mask |= shape_mask(size, [x, y + component["radius"], width, height - component["radius"]])
            if component["part"] == "left":
                fill_mask |= shape_mask(size, [x + component["radius"], y, width - component["radius"], height])
            expected[y:y + height, x:x + width][fill_mask[y:y + height, x:x + width]] = local[fill_mask[y:y + height, x:x + width]]
        records.append({"id": region["id"], "parent_id": region.get("parent_id"), "bbox": region["bbox"],
                        "output_id": output_id,
                        "components": [{key: value for key, value in component.items() if key != "sample_mask"} for component in components]})
    geometry = extra_geometry(source, expected, exclusions, union, regions, ir["nodes"])
    for node in geometry:
        evidence_id = "ev-" + node["id"]
        node["evidence_ids"] = [evidence_id]
    new_nodes.extend(geometry)
    active_regions = [region for region, _, _ in models]
    notes = ["Clean native card gradients; source wording and existing text styles preserved",
             "Card backgrounds, text and icon groups are independently selectable",
             "Complex icon paths remain source RGBA; candidates are recorded for the next phase"]
    target = make_revision(delivery["ir"], directory / "revision", plate, new_nodes, groups, active_regions, notes)
    authored = load_json(target)
    for decoration in review.get("raster_decorations", []):
        x, y, width, height = decoration["bbox"]
        crop = previous[y:y + height, x:x + width]
        original_crop = source[y:y + height, x:x + width]
        blue = ((original_crop[:, :, 2].astype(int) - original_crop[:, :, 0].astype(int) > 40)
                & (original_crop[:, :, 1] < 170)).astype(np.uint8)
        contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        silhouette = np.zeros(blue.shape, np.uint8)
        cv2.drawContours(silhouette, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        relative = "assets/" + decoration["id"] + ".png"
        Image.fromarray(np.dstack((crop, silhouette))).save(target.parent / relative)
        authored["evidence"].append({"id": "ev-" + decoration["id"], "kind": "image", "bbox": decoration["bbox"],
                                     "description": "Source-shaped cleaned decorative banner retained as RGBA"})
        authored["assets"].append({"id": decoration["id"], "path": relative, "purpose": "module",
                                   "contains_evidence_ids": ["ev-" + decoration["id"]], "sha256": sha256(target.parent / relative)})
        authored["nodes"].insert(1 + len(new_nodes), {"id": decoration["id"], "type": "image", "bbox": decoration["bbox"],
                                  "asset_id": decoration["id"], "evidence_ids": [], "z": 0})
    for visual in missing_visuals:
        x, y, width, height = visual["bbox"]
        crop = source[y:y + height, x:x + width]
        background_crop = expected[y:y + height, x:x + width]
        mask = np.abs(crop.astype(np.int16) - background_crop.astype(np.int16)).max(axis=2) > 14
        rgba = extract_rgba(crop, background_crop, mask)
        relative = "assets/" + visual["id"] + ".png"
        Image.fromarray(rgba).save(target.parent / relative)
        evidence_id = "ev-" + visual["id"]
        background_asset = next(asset for asset in authored["assets"] if asset["id"] == "plate")
        if evidence_id in background_asset["contains_evidence_ids"]:
            background_asset["contains_evidence_ids"].remove(evidence_id)
        if not any(item["id"] == evidence_id for item in authored["evidence"]):
            authored["evidence"].append({"id": evidence_id, "kind": "image", "bbox": visual["bbox"], "description": visual["label"]})
        authored["assets"].append({"id": visual["id"], "path": relative, "purpose": "module",
                                   "contains_evidence_ids": [evidence_id], "sha256": sha256(target.parent / relative)})
        authored["nodes"].append({"id": visual["id"], "type": "image", "bbox": visual["bbox"],
                                  "asset_id": visual["id"], "evidence_ids": [], "z": max(node["z"] for node in authored["nodes"]) + 1})
    for node in geometry:
        authored["evidence"].append({"id": node["evidence_ids"][0], "kind": "shape", "bbox": node["bbox"],
                                     "description": "Source separator or bullet"})
    for index, node in enumerate(authored["nodes"]):
        node["z"] = index
    # Add evidence before compiling and rebind the same new, uncompiled authoring revision.
    target.write_text(json.dumps(authored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    authoring_file = target.parent / "authoring.json"
    authoring = load_json(authoring_file)
    authoring["ir_sha256"] = sha256(target)
    authoring_file.write_text(json.dumps(authoring, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Image.fromarray(union.astype(np.uint8) * 255).save(directory / "surface-rebuild-mask.png")
    Image.fromarray(authorized.astype(np.uint8) * 255).save(directory / "authorized-background-mask.png")
    Image.fromarray(plate).save(directory / "cleaned-base.png")
    Image.fromarray(expected).save(directory / "background-model-preview.png")
    overlay = Image.fromarray(source.copy())
    draw = ImageDraw.Draw(overlay)
    for region in regions:
        x, y, width, height = region["bbox"]
        draw.rectangle((x, y, x + width, y + height), outline="red", width=2)
        draw.text((x + 5, y + 4), region["id"], fill="red")
    overlay.save(directory / "reviewed-boundaries.png")
    write_json(directory / "surface-models.json", {"schema": "SurfaceModels/1", "source_sha256": review["source_sha256"],
               "page_model": page_model, "page_fit": page_score, "regions": records, "failures": failures})
    print(f"[surface] {delivery['page']} compile regions={len(models)} failures={len(failures)}", flush=True)
    backend = ArtifactBackend(REPO, timeout)
    result = compile_ir(backend, target)
    target, result, icon_records = try_simple_icons(target, result, source, review, directory, backend)
    text_names = [node["id"] for node in ir["nodes"] if node["type"] == "text"]
    deleted = directory / "all-text-deleted.pptx"
    delete_text_copy(result["pptx"], deleted, text_names)
    no_text = backend.render(deleted, directory / "all-text-deleted-render")
    background = directory / "background-only.pptx"
    keep = {"background-image", *[group["id"] for group in groups], *[node["id"] for node in new_nodes],
            *[item["id"] for item in review.get("raster_decorations", [])]}
    rewrite_package(result["pptx"], background, keep=keep)
    background_render = backend.render(background, directory / "background-only-render")
    actual_background = pixels(background_render["pages"][0]["path"], size)
    calibrated_ir, calibrated = calibrate_contours(target, source, actual_background, active_regions, directory, backend)
    if calibrated:
        target, result = calibrated_ir, calibrated
        deleted = directory / "all-text-deleted-calibrated.pptx"
        delete_text_copy(result["pptx"], deleted, text_names)
        no_text = backend.render(deleted, directory / "all-text-deleted-calibrated-render")
        background = directory / "background-only-calibrated.pptx"
        rewrite_package(result["pptx"], background, keep=keep)
        background_render = backend.render(background, directory / "background-only-calibrated-render")
        actual_background = pixels(background_render["pages"][0]["path"], size)
    checks = []
    for region, mask, components in models:
        sample = np.logical_or.reduce([component["sample_mask"] for component in components])
        score = surface_quality(source, actual_background, sample, region["bbox"])
        contour = edge_profiles(source, actual_background, region["bbox"])
        distances = [abs(value["delta"]) for value in contour.values() if value["status"] == "MEASURED"]
        contour_p95 = float(np.percentile(distances, 95)) if distances else None
        checks.append({"id": region["id"], **score, "contour_sections": contour, "contour_position_p95": contour_p95,
                       "contour_status": "PASS" if contour_p95 is not None and contour_p95 <= 1.5 else "REVIEW_REQUIRED",
                       "status": "PASS" if score["mae"] <= 2 and score["p95"] <= 5 else "REVIEW_REQUIRED"})
    outside = int(np.any(plate != previous, axis=2)[~authorized].sum())
    if outside:
        raise ValueError("background write escaped authorized mask")
    seam = repair_seam_quality(previous, actual_background, authorized)
    naked = pixels(no_text["pages"][0]["path"], size)
    protected = np.zeros(source.shape[:2], bool)
    for node in ir["nodes"]:
        if node["type"] == "image" and node["id"] != "background-image":
            asset = next(asset for asset in ir["assets"] if asset["id"] == node["asset_id"])
            alpha = np.array(Image.open(Path(delivery["ir"]).parent / asset["path"]).convert("RGBA"))[:, :, 3]
            x, y, width, height = map(int, node["bbox"])
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_NEAREST)
            protected[y:y + height, x:x + width] |= alpha > 16
    (directory / "text-masks").mkdir()
    text_checks = []
    for line in inventory["lines"]:
        masks = precise_text_masks(source, line, protected)
        for kind in ("glyph", "repair", "inspection"):
            Image.fromarray(masks[kind].astype(np.uint8) * 255).save(directory / "text-masks" / f"{line['id']}-{kind}.png")
        residue = inspect_residue(source, naked, masks["core"], masks["inspection"])
        text_checks.append({"id": line["id"], "mask_coverage": masks["coverage"], "mask_status": masks["status"], **residue})
    lab = cv2.cvtColor(naked, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=3, tileGridSize=(16, 16)).apply(lab[:, :, 0])
    enhanced = directory / "all-text-deleted-enhanced.png"
    Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)).save(enhanced)
    hits = []
    for file in (Path(no_text["pages"][0]["path"]), enhanced):
        for hit in detect(file, min_score=0.45):
            if any(intersect(hit.bbox, line["bbox"]) > hit.bbox[2] * hit.bbox[3] * 0.5 for line in inventory["lines"]):
                hits.append({"text": hit.text, "bbox": hit.bbox, "preview": str(file)})
    ownership = []
    for node in ir["nodes"]:
        if node["id"] == "background-image" or node["id"].endswith("-frame"):
            continue
        candidates = [region for region in regions if intersect(node["bbox"], region["bbox"]) >= node["bbox"][2] * node["bbox"][3] * 0.5]
        owner = min(candidates, key=lambda region: region["bbox"][2] * region["bbox"][3])["id"] if candidates else None
        ownership.append({"id": node["id"], "container_id": owner, "role": node["type"], "bbox": node["bbox"]})
    write_json(directory / "object-ownership.json", ownership)
    complex_icons = []
    converted = {record["id"] for record in icon_records if record["status"] == "NATIVE"}
    for node in ir["nodes"]:
        if node["type"] == "image" and node["id"] != "background-image" and node["id"] not in converted:
            asset = next(asset for asset in ir["assets"] if asset["id"] == node["asset_id"])
            complex_icons.append({"id": node["id"], "bbox": node["bbox"], "source_asset_sha256": asset["sha256"],
                                  "output": "RGBA_PNG", "svg_matching": "DEFERRED",
                                  "next_stage_requirement": "licensed local SVG candidate and validated native path support"})
    write_json(directory / "complex-icon-candidates.json", complex_icons)
    quality_failures = [record["id"] for record in checks if record["status"] != "PASS"]
    quality_failures.extend(record["id"] + "-contour" for record in checks
                            if record["contour_position_p95"] is not None and record["contour_position_p95"] > 1.5)
    if seam["status"] != "PASS":
        quality_failures.append("repair_boundary_seam")
    quality_failures.extend(record["id"] for record in text_checks
                            if record["status"] != "PASS" or record["mask_status"] != "PASS")
    if hits:
        quality_failures.append("independent_ocr_residue")
    if failures:
        quality_failures.extend(record["id"] for record in failures)
    summary = {"schema": "SurfaceDelivery/1", "page": delivery["page"], "directory": str(directory.resolve()),
               "ir": str(target.resolve()), **result, "source_sha256": review["source_sha256"],
               "native_card_backgrounds": len(models), "native_background_shapes": len(new_nodes),
               "icon_conversions": icon_records,
               "surface_checks": checks, "text_checks": text_checks, "ocr_hits": hits,
               "failures": failures, "outside_authorized_changed_pixels": outside,
               "repair_seam": seam,
               "quality_failures": quality_failures,
               "baseline_card_fallbacks": review.get("baseline_card_fallbacks", []),
               "submission_status": "VERIFIED" if not quality_failures else "QUARANTINED",
               "deleted_text_pptx": str(deleted.resolve()), "deleted_text_preview": no_text["pages"][0]["path"],
               "background_preview": background_render["pages"][0]["path"],
               "background_pptx": str(background.resolve()),
               "powerpoint_checked": False, "wps_checked": False, "human_acceptance": False, "status": "REVIEW_REQUIRED"}
    write_json(directory / "delivery.json", summary)
    if quality_failures:
        card_failures = {item["id"] for item in checks
                         if item["status"] != "PASS"
                         or (item["contour_position_p95"] is not None and item["contour_position_p95"] > 1.5)}
        card_failures.update(item["id"] for item in failures)
        if card_failures:
            revised = rollback_review(review, card_failures, ", ".join(quality_failures))
            write_json(directory / "rollback-review.json", revised)
            recovered = rebuild_page(delivery, revised, output, timeout)
            recovered["rollback_diagnostics"] = str((directory / "delivery.json").resolve())
            return recovered
        raise ValueError("surface or independent text acceptance failed: " + ", ".join(quality_failures))
    print(f"[surface] {delivery['page']} checks={[(c['id'],c['mae']) for c in checks]} ocr={hits}", flush=True)
    return summary


def rebuild_batch(input_file, output, reviews, pages=None, resume=None, timeout=600):
    original = load_json(input_file)
    deliveries = original["pages"]
    if pages:
        deliveries = [item for item in deliveries if item["page"][:2] in pages]
    directory = Path(output).resolve() / ("surfaces-" + stamp())
    directory.mkdir(parents=True)
    completed = {}
    if resume:
        previous = load_json(resume)
        for item in previous["pages"]:
            if sha256(item["ir"]) != item["ir_sha256"] or sha256(item["pptx"]) != item["pptx_sha256"]:
                raise ValueError("resume artifact hash mismatch")
            completed[item["page"]] = item
    results = []
    failures = []
    for delivery in deliveries:
        try:
            if delivery["page"] in completed:
                result = completed[delivery["page"]]
                review_file = Path(reviews) / (delivery["page"][:2] + ".json")
                if (result["source_sha256"] != delivery["source_sha256"]
                        or result.get("baseline_ir_sha256") != sha256(delivery["ir"])
                        or result.get("review_sha256") != sha256(review_file)
                        or result.get("pipeline_sha256") != pipeline_fingerprint()):
                    raise ValueError("resume source, review or pipeline changed; rerun page explicitly")
            else:
                review_file = Path(reviews) / (delivery["page"][:2] + ".json")
                review = load_json(review_file)
                result = rebuild_page(delivery, review, directory / "pages", timeout)
                result["ir_sha256"] = sha256(result["ir"])
                result["pptx_sha256"] = sha256(result["pptx"])
                result["baseline_ir_sha256"] = sha256(delivery["ir"])
                result["review_sha256"] = sha256(review_file)
                result["pipeline_sha256"] = pipeline_fingerprint()
            results.append(result)
        except Exception as error:
            failures.append({"page": delivery["page"], "reason": str(error)})
            print(f"[surface] {delivery['page']} FAILED {error}", flush=True)
        write_json(directory / f"progress-{len(results) + len(failures):02d}.json", {"pages": results, "failures": failures})
    summary = {"schema": "SurfaceBatch/1", "directory": str(directory), "input": str(Path(input_file).resolve()),
               "pages": results, "failures": failures, "status": "REVIEW_REQUIRED"}
    if not failures:
        from image2ppt.compiler import execute
        backend = ArtifactBackend(REPO, timeout)
        summary["deck"] = execute([backend.node, ROOT / "compiler_bridge.mjs", "compose", directory / "deck",
                                   *[item["ir"] for item in results]], REPO, timeout)
    write_json(directory / "batch.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--pages", nargs="+")
    parser.add_argument("--reviews", default=str(ROOT / "configs/surface-reviews"))
    parser.add_argument("--output", default=str(ROOT / "outputs/surfaces"))
    parser.add_argument("--resume")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    result = rebuild_batch(args.input, args.output, args.reviews, args.pages, args.resume, args.timeout)
    print(json.dumps({"directory": result["directory"], "pages": len(result["pages"]), "failures": result["failures"]}, ensure_ascii=False))
    raise SystemExit(2 if result["failures"] else 0)
