"""Extract large colored planes and small repeated marks from a cleaned plate.

Geometry comes from connected source pixels, never page-specific coordinates.
The model reviews candidate IDs before a build. Existing assets are copied
byte-for-byte; the original IR and plate are retained in their old run.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from model_slideir_loop import digest, write_json


def _components(mask: np.ndarray) -> list[dict]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [{"label": index, "bbox": [int(x), int(y), int(w), int(h)],
             "area": int(area), "labels": labels}
            for index, (x, y, w, h, area) in enumerate(stats[1:], 1)]


def detect(plate: Path, *, row_count: int, checkbox_count: int) -> dict:
    rgb = np.asarray(Image.open(plate).convert("RGB"), dtype=np.int16)
    height, width = rgb.shape[:2]
    chroma = rgb[:, :, 2] - rgb[:, :, 0]
    blue_green = rgb[:, :, 2] - rgb[:, :, 1]
    mask = (chroma > 7) & (blue_green > 3)
    candidates = [c for c in _components(mask) if c["area"] > width * height * .025]
    vertical = [c for c in candidates if c["bbox"][3] > c["bbox"][2] * 1.2]
    if len(vertical) != 1:
        raise ValueError(f"expected one vertical plane, found {len(vertical)}")
    left = vertical[0]
    lx, ly, lw, lh = left["bbox"]
    horizontal = [c for c in candidates if c is not left and
                  c["bbox"][2] > lw * 2 and c["bbox"][2] > c["bbox"][3] * 5]
    rows = sorted([c for c in horizontal if c["bbox"][1] < ly + lh],
                  key=lambda c: c["bbox"][1])
    banner_options = [c for c in horizontal if c["bbox"][1] >= ly + lh]
    if len(rows) != row_count or len(banner_options) != 1:
        raise ValueError(f"plane count mismatch: rows={len(rows)}, banners={len(banner_options)}")
    banner = banner_options[0]
    bx, by, bw, bh = banner["bbox"]
    # Neighboring blue slashes at the same y are parts of the banner module.
    extras = [c for c in _components(mask) if c["label"] != banner["label"] and
              c["area"] > width * height * .0005 and
              abs(c["bbox"][1] - by) < bh * .25 and
              c["bbox"][3] > bh * .7 and
              (abs(c["bbox"][0] + c["bbox"][2] - bx) < bw * .06 or
               abs(c["bbox"][0] - (bx + bw)) < bw * .06)]
    banner_parts = [banner, *extras]
    bx0 = min(c["bbox"][0] for c in banner_parts)
    by0 = min(c["bbox"][1] for c in banner_parts)
    bx1 = max(c["bbox"][0] + c["bbox"][2] for c in banner_parts)
    by1 = max(c["bbox"][1] + c["bbox"][3] for c in banner_parts)
    dark = (chroma > 50) & (rgb[:, :, 0] < 160)
    checkboxes = sorted([c for c in _components(dark)
                         if lx < c["bbox"][0] < lx + lw * .4 and
                         ly + lh * .15 < c["bbox"][1] < ly + lh * .8 and
                         .6 < c["bbox"][2] / max(1, c["bbox"][3]) < 1.6 and
                         80 < c["area"] < lw * lh * .01 and
                         10 < c["bbox"][2] < lw * .2 and
                         10 < c["bbox"][3] < lh * .1],
                        key=lambda c: c["bbox"][1])
    if len(checkboxes) != checkbox_count:
        raise ValueError(f"checkbox count mismatch: {len(checkboxes)}")
    ordered = [("left-panel", [left], left["bbox"])] + [
        (f"right-panel-{index + 1:02d}", [row], row["bbox"])
        for index, row in enumerate(rows)] + [
        ("bottom-banner", banner_parts, [bx0, by0, bx1 - bx0, by1 - by0])]
    ordered += [(f"checkbox-{index + 1:02d}", [box], box["bbox"])
                for index, box in enumerate(checkboxes)]
    return {"source_plate_sha256": digest(plate), "image_size": [width, height],
            "threshold": {"blue_minus_red": 7, "blue_minus_green": 3},
            "objects": [{"candidate_id": name, "bbox": box,
                         "component_labels": [c["label"] for c in parts],
                         "component_areas": [c["area"] for c in parts]}
                        for name, parts, box in ordered]}


def build(ir_path: Path, candidates_path: Path, review_path: Path | None,
          output: Path, *, diagnostic: bool = False) -> dict:
    if output.exists():
        raise ValueError("refusing to overwrite extraction output")
    ir_path = ir_path.resolve(strict=True)
    original = json.loads(ir_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    review = (json.loads(review_path.read_text(encoding="utf-8"))
              if review_path else None)
    ids = {obj["candidate_id"] for obj in candidates["objects"]}
    if not diagnostic:
        if review is None:
            raise ValueError("model asset review is required for a final build")
        accepted = {obj["candidate_id"] for obj in review.get("objects", [])
                    if obj.get("accept") is True}
        if review.get("schema") != "asset-review/1" or accepted != ids or len(review["objects"]) != len(ids):
            raise ValueError("model has not accepted every detected plane and checkbox")
        if review.get("source_sha256") != original["source_sha256"]:
            raise ValueError("model asset review source mismatch")
    plate_entry = next(a for a in original["assets"]
                       if a["id"] == original["background"]["asset_id"])
    plate_path = ir_path.parent / plate_entry["path"]
    if digest(plate_path) != candidates["source_plate_sha256"] or digest(plate_path) != plate_entry["sha256"]:
        raise ValueError("cleaned plate or detector input changed")
    if not diagnostic and review.get("plate_sha256") != digest(plate_path):
        raise ValueError("model asset review plate mismatch")
    rgb = np.asarray(Image.open(plate_path).convert("RGB"))
    signed = rgb.astype(np.int16)
    chroma = signed[:, :, 2] - signed[:, :, 0]
    mask = ((chroma > 7) & (signed[:, :, 2] - signed[:, :, 1] > 3)).astype(np.uint8)
    dark = ((chroma > 50) & (signed[:, :, 0] < 160)).astype(np.uint8)
    _, plane_labels, _, _ = cv2.connectedComponentsWithStats(mask, 8)
    _, dark_labels, _, _ = cv2.connectedComponentsWithStats(dark, 8)
    object_masks = {}
    check_mask = np.zeros(mask.shape, np.uint8)
    for obj in candidates["objects"]:
        name = obj["candidate_id"]
        if name.startswith("checkbox-"):
            label = obj["component_labels"][0]
            component = (dark_labels == label).astype(np.uint8)
            check_mask |= cv2.dilate(component, np.ones((3, 3), np.uint8))
            object_masks[name] = component
        else:
            object_masks[name] = np.isin(plane_labels, obj["component_labels"]).astype(np.uint8)
    # Remove the three glyphs from the panel's texture before extracting it.
    panel_rgb = cv2.inpaint(rgb, cv2.dilate(check_mask, np.ones((7, 7), np.uint8)), 4,
                            cv2.INPAINT_TELEA)
    plane_union = np.zeros(mask.shape, np.uint8)
    for name, component in object_masks.items():
        if not name.startswith("checkbox-"):
            plane_union |= cv2.dilate(component, np.ones((5, 5), np.uint8))
    base = cv2.inpaint(rgb, plane_union, 5, cv2.INPAINT_TELEA)
    output.mkdir(parents=True)
    shutil.copytree(ir_path.parent / "source", output / "source")
    shutil.copytree(ir_path.parent / "assets", output / "assets")
    result = copy.deepcopy(original)
    Image.fromarray(base).save(output / plate_entry["path"])
    plate_entry_new = next(a for a in result["assets"] if a["id"] == plate_entry["id"])
    plate_entry_new["sha256"] = digest(output / plate_entry["path"])
    inventory = []
    for index, obj in enumerate(candidates["objects"]):
        name = obj["candidate_id"]
        x, y, w, h = obj["bbox"]
        margin = 3
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(rgb.shape[1], x + w + margin), min(rgb.shape[0], y + h + margin)
        component = object_masks[name]
        alpha = cv2.dilate(component, np.ones((5, 5), np.uint8)) * 255
        layer_rgb = rgb if name.startswith("checkbox-") else panel_rgb if name == "left-panel" else rgb
        rgba = np.dstack([layer_rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1]]).astype(np.uint8)
        path = output / "assets" / (name + ".png")
        Image.fromarray(rgba, "RGBA").save(path)
        aid, eid = "plane-" + name, "plane-evidence-" + name
        box = [x0, y0, x1 - x0, y1 - y0]
        result["evidence"].append({"id": eid, "kind": "shape", "bbox": box,
                                   "description": "Detected from the hash-bound cleaned plate; " +
                                   ("diagnostic candidate " if diagnostic else "model-reviewed ") + name})
        result["assets"].append({"id": aid, "path": "assets/" + path.name,
                                  "purpose": "module", "sha256": digest(path),
                                  "contains_evidence_ids": [eid]})
        result["nodes"].append({"id": name, "type": "image", "bbox": box,
                                "z": -len(candidates["objects"]) + index,
                                "asset_id": aid, "evidence_ids": []})
        inventory.append({"id": name, "bbox": box, "asset_path": str(path),
                          "asset_sha256": digest(path),
                          "source_component_area": obj["component_areas"],
                          "transparent_corner": int(rgba[0, 0, 3]) == 0})
    result["review_notes"] = [note.replace(
        "Full-slide plate retained as raster background because it carries layout bands, checkbox glyphs, banner shape and the page number 02 (l023).",
        "The residual plate retains the page number and minor decoration; large planes and checkbox glyphs were extracted as " +
        ("unreviewed diagnostic candidates." if diagnostic else "model-reviewed transparent modules."))
        for note in result["review_notes"]]
    write_json(output / "slideir.json", result)
    write_json(output / "authoring.json", {"format_version": "1", "tool": "source-bound-plane-extraction",
                                            "ir_sha256": digest(output / "slideir.json"),
                                            "parent_ir_sha256": digest(ir_path),
                                            "candidate_report_sha256": digest(candidates_path),
                                            "model_review_sha256": digest(review_path) if review_path else None,
                                            "diagnostic_only": diagnostic})
    report = {"status": "DIAGNOSTIC_REVIEW_PENDING" if diagnostic else "BUILT",
              "source_sha256": result["source_sha256"],
              "old_plate_sha256": digest(plate_path), "new_plate_sha256": plate_entry_new["sha256"],
              "inventory": inventory}
    write_json(output / "extraction.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["detect", "build"], required=True)
    parser.add_argument("--plate", type=Path)
    parser.add_argument("--ir", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-count", type=int, default=5)
    parser.add_argument("--checkbox-count", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "detect":
        report = detect(args.plate, row_count=args.row_count,
                        checkbox_count=args.checkbox_count)
        write_json(args.output, report)
        print(json.dumps({"status": "DETECTED", "objects": len(report["objects"]),
                          "output": str(args.output)}))
    else:
        report = build(args.ir, args.candidates, args.review, args.output,
                       diagnostic=args.diagnostic)
        print(json.dumps({"status": report["status"], "objects": len(report["inventory"]),
                          "output": str(args.output)}))
