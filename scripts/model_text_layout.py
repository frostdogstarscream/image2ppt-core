"""Source-bound typography calibration for a recorded, model-authored SlideIR.

Only pixels removed from a hash-bound cleaned background plate are measured.
The model's wording and object plan remain the input; ambiguous regions stay as
they were and receive a diagnostic rather than a guessed edit.
"""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ppt-rebuild/src"))
from font_matcher import FONT_DIR, Ink, _render, fit_text  # noqa: E402


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ink(mask: np.ndarray, box: list[float]) -> Ink | None:
    x, y, w, h = map(lambda v: int(round(v)), box)
    crop = mask[y:y + h, x:x + w]
    ys, xs = np.nonzero(crop)
    if len(xs) < 12:
        return None
    return Ink([x + int(xs.min()), y + int(ys.min()),
                int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
               len(xs), "", float(crop.mean()))


def _split_gap(mask: np.ndarray, box: list[float]) -> tuple[int, int] | None:
    x, y, w, h = map(lambda v: int(round(v)), box)
    columns = mask[y:y + h, x:x + w].any(axis=0)
    gaps = []
    start = None
    for index, visible in enumerate([*columns, True]):
        if not visible and start is None:
            start = index
        if visible and start is not None:
            if start >= 15 and index <= w * .48 and index - start >= 12:
                gaps.append((start, index))
            start = None
    return max(gaps, key=lambda gap: gap[1] - gap[0]) if gaps else None


def _font_density(text: str, file: str, size_px: float) -> float:
    font = ImageFont.truetype(str(FONT_DIR / file), max(1, round(size_px)))
    width = max(64, round(size_px * (len(text) + 2)))
    height = max(64, round(size_px * 3))
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).text((round(size_px), round(size_px)), text,
                               font=font, fill=255)
    pixels = np.asarray(image) > 40
    ys, xs = np.nonzero(pixels)
    if not len(xs):
        return 0.0
    return float(pixels[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean())


def calibrate(ir: dict, source: Path, assets_base: Path) -> tuple[dict, dict]:
    """Return a new IR and an auditable report; never mutate model or source files."""
    result = copy.deepcopy(ir)
    report = {"status": "SKIPPED", "source_sha256": _hash(source),
              "input_ir_canvas": result.get("canvas"), "rows": [], "splits": []}
    if report["source_sha256"] != result.get("source_sha256"):
        raise ValueError("source hash differs from model SlideIR")
    plates = [a for a in result.get("assets", [])
              if a.get("id") == result.get("background", {}).get("asset_id")
              and a.get("purpose") == "cleaned_background"]
    if len(plates) != 1:
        report["reason"] = "no unique cleaned background plate"
        return result, report
    plate = assets_base / plates[0]["path"]
    if _hash(plate) != plates[0]["sha256"]:
        raise ValueError("cleaned background plate hash mismatch")
    with Image.open(source) as src, Image.open(plate) as bg:
        if src.size != bg.size or list(src.size) != result["canvas"]["source_px"]:
            raise ValueError("source, plate and IR canvas dimensions differ")
        src_rgb = np.asarray(src.convert("RGB"), dtype=np.int16)
        bg_rgb = np.asarray(bg.convert("RGB"), dtype=np.int16)
    sw, sh = result["canvas"]["source_px"]
    tw, th = result["canvas"]["target_px"]
    if abs(tw / sw - th / sh) > .002:
        raise ValueError("anisotropic target canvas prevents font conversion")
    # Artifact Tool uses CSS pixels: 1 pt = 96/72 target px, then scale to source.
    dpi_source = 96 * sw / tw
    mask = np.max(np.abs(src_rgb - bg_rgb), axis=2) > 35
    evidence = {e["id"]: e for e in result["evidence"]}
    groups = {member for group in result["groups"] for member in group["member_ids"]}
    calibrated = []
    occupied_z = {n["z"] for n in result["nodes"]}
    next_z = max(occupied_z) + 1
    for node in result["nodes"]:
        if node["type"] != "text":
            calibrated.append(node)
            continue
        items = [evidence[eid] for eid in node["evidence_ids"]]
        row = {"node_id": node["id"], "text": node["text"],
               "source_bbox": node["bbox"], "model_size_pt": node["font"]["size_pt"],
               "source_px_per_pt": round(dpi_source / 72, 6),
               "target_px_per_pt": round(96 / 72, 6), "emu_per_target_px": 9525}
        parts = [node]
        # The original OCR/model sometimes merges a two-digit index and heading.
        # Split only when exact evidence wording and a wide source-ink gap agree.
        if (len(items) == 1 and items[0]["kind"] == "text"
                and items[0]["text"] == node["text"] and node["id"] not in groups
                and not node.get("runs") and not node.get("paragraphs")
                and len(node["text"]) > 2 and node["text"][:2].isdigit()
                and not node["text"][2].isdigit()):
            gap = _split_gap(mask, items[0]["bbox"])
            if gap:
                original = items[0]
                ex, ey, ew, eh = original["bbox"]
                left_box = [ex, ey, gap[0], eh]
                right_box = [ex + gap[1], ey, ew - gap[1], eh]
                if _ink(mask, left_box) and _ink(mask, right_box):
                    source_id = original["id"]
                    result["evidence"].remove(original)
                    left_ev = {**original, "id": source_id + "-num", "text": node["text"][:2],
                               "bbox": left_box,
                               "description": "Source OCR line " + source_id + "; exact prefix"}
                    right_ev = {**original, "id": source_id + "-title", "text": node["text"][2:],
                                "bbox": right_box,
                                "description": "Source OCR line " + source_id + "; exact suffix"}
                    result["evidence"].extend([left_ev, right_ev])
                    number = {**copy.deepcopy(node), "id": node["id"] + "-num",
                              "text": left_ev["text"], "bbox": left_box,
                              "evidence_ids": [left_ev["id"]]}
                    title = {**copy.deepcopy(node), "id": node["id"] + "-title",
                             "text": right_ev["text"], "bbox": right_box,
                             "evidence_ids": [right_ev["id"]], "z": next_z}
                    if "source_text_bbox" in node:
                        number["source_text_bbox"] = left_box
                        title["source_text_bbox"] = right_box
                    next_z += 1
                    parts = [number, title]
                    report["splits"].append({"node": node["id"], "evidence": source_id,
                                             "parts": [number["id"], title["id"]],
                                             "gap_source_px": [ex + gap[0], ex + gap[1]],
                                             "text_concatenation_preserved": True})
        for part in parts:
            ink = _ink(mask, part["bbox"])
            entry = dict(row, node_id=part["id"], text=part["text"], bbox=part["bbox"])
            if not ink:
                entry["status"] = "NO_ISOLATED_SOURCE_INK"
                report["rows"].append(entry)
                calibrated.append(part)
                continue
            entry["source_ink_bbox"] = ink.bbox
            family = "Arial" if part["text"].isascii() else "微软雅黑"
            fits = fit_text(part["text"], ink, dpi=dpi_source,
                            size_range=(8, 72, .5), families=[family])
            ix, iy, iw, ih = ink.bbox
            source_density = float(mask[iy:iy + ih, ix:ix + iw].mean())
            ranked = sorted(((f.score + .15 * abs(_font_density(part["text"], f.file,
                                                               f.size_px) - source_density), f)
                             for f in fits), key=lambda candidate: candidate[0])
            entry["source_ink_density"] = round(source_density, 4)
            entry["candidate_fits"] = [{"family": f.family, "bold": f.bold,
                                         "size_pt": f.size_pt, "geometry_score": f.score,
                                         "score": round(score, 4),
                                         "rendered_ink_px": f.rendered}
                                        for score, f in ranked[:3]]
            within_box = [(score, f) for score, f in ranked
                          if f.rendered[0] <= part["bbox"][2] + 3
                          and f.rendered[1] <= part["bbox"][3] + 3]
            best = within_box[0][1] if within_box else None
            if best is None or within_box[0][0] > .12:
                entry["status"] = "FIT_UNCERTAIN"
            else:
                old_size = part["font"]["size_pt"]
                model_pixels = _render(part["text"], FONT_DIR / best.file,
                                       max(1, round(old_size * dpi_source / 72)))
                entry["model_rendered_ink_px"] = list(model_pixels)
                entry["model_width_ratio_to_source_ink"] = round(
                    model_pixels[0] / max(1, ink.bbox[2]), 3)
                entry["model_predicted_overflow"] = (
                    model_pixels[0] > part["bbox"][2] + 3 or
                    model_pixels[1] > part["bbox"][3] + 3)
                part["font"]["size_pt"] = best.size_pt
                # Font geometry and emphasis are independent decisions. Preserve
                # the model/reviewer's explicit bold and any runs/paragraphs.
                entry.update(status="FITTED", fitted_size_pt=best.size_pt,
                             size_change_pt=round(best.size_pt - old_size, 2),
                             candidate_bold=best.bold,
                             preserved_bold=part.get("bold"),
                             projected_target_ink_px=[round(v * tw / sw, 2)
                                                      for v in best.rendered])
            report["rows"].append(entry)
            calibrated.append(part)
    result["nodes"] = calibrated
    if report["splits"]:
        result["review_notes"] = [note.replace(
            "kept as single text nodes to avoid inventing split geometry.",
            "split at source-measured ink gaps after model response; see text-layout.json.")
            for note in result.get("review_notes", [])]
        result["review_notes"].append(
            "Text/evidence split provenance and source ink measurements are recorded in text-layout.json.")
    report["status"] = "MEASURED"
    report["source_size_px"] = [sw, sh]
    report["target_size_px"] = [tw, th]
    report["target_size_emu"] = [round(tw * 9525), round(th * 9525)]
    return result, report


def write_comparisons(source: Path, preview: Path, report: dict, destination: Path) -> None:
    """Save fixed-region source/render pairs and pixel errors for every text node."""
    destination.mkdir(parents=True, exist_ok=False)
    with Image.open(source) as src, Image.open(preview) as rendered:
        src = src.convert("RGB")
        rendered = rendered.convert("RGB").resize(src.size, Image.Resampling.BILINEAR)
        for row in report["rows"]:
            x, y, w, h = [int(round(v)) for v in row["bbox"]]
            box = (max(0, x - 8), max(0, y - 8), min(src.width, x + w + 8),
                   min(src.height, y + h + 8))
            original = src.crop(box)
            actual = rendered.crop(box)
            difference = np.abs(np.asarray(original, dtype=np.int16) -
                                np.asarray(actual, dtype=np.int16))
            row["render_bbox_mae"] = round(float(difference.mean()), 3)
            row["render_crop_bbox"] = list(box)
            pair = Image.new("RGB", (original.width * 2, original.height), "white")
            pair.paste(original, (0, 0))
            pair.paste(actual, (original.width, 0))
            filename = row["node_id"] + ".png"
            pair.save(destination / filename)
            row["comparison_file"] = str(destination / filename)
