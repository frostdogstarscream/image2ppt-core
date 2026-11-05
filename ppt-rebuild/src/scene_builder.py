"""M5a — scene graph.

One model-independent description of the slide in reference pixels (1672×941).
Everything downstream (PPTX export, evaluation) reads this file only, so a change
in the exporter can never silently change what was decided here.

Objects carry their provenance: which layer they came from, how confident the fit
was, and which edit level they reach. A text box that could not be cleaned keeps
its pixels as an image object with an explicit reason instead of a ghost.
"""
from __future__ import annotations

import json
from pathlib import Path

CANVAS = (1672, 941)
DPI = 125.4


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build(page: str, *, ocr_path: Path, fonts_path: Path, mask_path: Path,
          plate_path: Path, source_path: Path, workdir: Path) -> dict:
    ocr = _load(ocr_path)
    fonts = {fit["box"]: fit for fit in _load(fonts_path)["fits"]}
    mask = {entry["box"]: entry for entry in _load(mask_path)["boxes"]}
    canvas = list(CANVAS)
    if list(ocr["image_size"]) != canvas:
        raise ValueError(f"canvas {canvas} does not match OCR image size {ocr['image_size']}")

    objects = []
    for box in ocr["boxes"]:
        fit = fonts.get(box["id"], {})
        decision = mask.get(box["id"], {})
        status = decision.get("status", "unknown")
        if status.startswith("keep_raster") or status.startswith("refused") or status == "no_ink":
            # 计划 §M3：擦不干净就整块留作图片，绝不出现"删了字但背景损坏"
            reason = status.split(":")[0]
            objects.append({
                "id": f"raster_{box['id']}",
                "type": "image",
                "bbox_px": box["bbox"],
                "z": 40,
                "source_layers": [f"layer-{decision.get('owner_layer', 0):02d}.png"],
                "editable_level": "raster",
                "reason": reason,
                "crop_from": "source",
            })
            continue
        if not fit.get("size_pt"):
            continue
        objects.append({
            "id": f"text_{box['id']}",
            "type": "text",
            "text": box["text"],
            "bbox_px": box["bbox"],
            "font_family": fit.get("font_family_final"),
            "font_size_pt": fit.get("size_pt"),
            "bold": bool(fit.get("bold")),
            "color": (fit.get("ink") or {}).get("color"),
            "align": "left",
            "z": 30,
            "source_layers": [f"layer-{decision.get('owner_layer', 0):02d}.png"],
            "confidence": round(1.0 - (fit.get("fit_score") or 0.0), 4),
            "editable_level": "native",
            "erase_status": status,
            "residual_after": decision.get("residual_after"),
        })

    objects.sort(key=lambda item: (item["z"], item["bbox_px"][1], item["bbox_px"][0]))
    for index, item in enumerate(objects):
        item["id"] = f"{item['id']}" if item["id"].count("_") >= 1 else item["id"]
        item["order"] = index

    return {
        "schema": "Scene/1",
        "slide_id": page,
        "canvas_px": canvas,
        "dpi": DPI,
        "background": {
            "file": str(plate_path),
            "bbox_px": [0, 0, canvas[0], canvas[1]],
            "editable_level": "raster",
            "note": "去字后的底图：文字已按笔画级掩码清除，不含任何可读文字",
        },
        "source": str(source_path),
        "objects": objects,
        "counts": {
            "text": sum(1 for o in objects if o["type"] == "text"),
            "image": sum(1 for o in objects if o["type"] == "image"),
        },
    }


def validate(scene: dict) -> list[str]:
    """Cheap invariants that must hold before export."""
    problems = []
    canvas = scene["canvas_px"]
    ids = set()
    for obj in scene["objects"]:
        if obj["id"] in ids:
            problems.append(f"duplicate id {obj['id']}")
        ids.add(obj["id"])
        x, y, w, h = obj["bbox_px"]
        if w <= 0 or h <= 0:
            problems.append(f"{obj['id']}: empty bbox")
        if x < -1 or y < -1 or x + w > canvas[0] + 1 or y + h > canvas[1] + 1:
            problems.append(f"{obj['id']}: bbox outside canvas")
        if obj["type"] == "text" and not obj["text"].strip():
            problems.append(f"{obj['id']}: empty text")
        if obj["type"] == "text" and not obj.get("font_size_pt"):
            problems.append(f"{obj['id']}: missing font size")
    return problems
