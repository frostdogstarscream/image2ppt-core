"""M2a — text detection and recognition.

PP-OCRv4 (det / cls / rec ONNX) runs locally; no network, no model download.
Output is one record per detected line with polygon, bbox, text and confidence.
Line grouping into text boxes is a separate, testable step so that a bad grouping
never silently rewrites the recognised text.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MODELS = Path(__file__).resolve().parents[1] / "models"

_ENGINE = None


def engine():
    """RapidOCR bound to the local ONNX files, with implicit downloads disabled."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    from rapidocr import RapidOCR
    from rapidocr.utils.download_file import DownloadFile
    from rapidocr.utils.typings import OCRVersion

    DownloadFile.run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("runtime model download disabled"))
    missing = [name for name in ("det.onnx", "cls.onnx", "rec.onnx") if not (MODELS / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing OCR models in {MODELS}: {', '.join(missing)}")
    _ENGINE = RapidOCR(params={
        "Det.model_path": str(MODELS / "det.onnx"),
        "Cls.model_path": str(MODELS / "cls.onnx"),
        "Rec.model_path": str(MODELS / "rec.onnx"),
        "EngineConfig.onnxruntime.intra_op_num_threads": 4,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "Det.ocr_version": OCRVersion.PPOCRV4,
        "Rec.ocr_version": OCRVersion.PPOCRV4,
        "Global.log_level": "error",
    })
    return _ENGINE


@dataclass
class Line:
    id: str
    text: str
    score: float
    polygon: list[list[float]]
    bbox: list[int]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0


def _bbox_from_polygon(polygon) -> list[int]:
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    left, top = int(np.floor(min(xs))), int(np.floor(min(ys)))
    return [left, top, int(np.ceil(max(xs))) - left, int(np.ceil(max(ys))) - top]


def detect(image_path: str | Path, *, min_score: float = 0.0) -> list[Line]:
    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB"))
    result = engine()(rgb[:, :, ::-1].copy())  # RapidOCR expects BGR
    lines: list[Line] = []
    rows = zip(result.boxes, result.txts, result.scores) if result.boxes is not None else []
    for index, (polygon, text, score) in enumerate(rows):
        text = (text or "").strip()
        score = float(score or 0.0)
        if not text or score < min_score:
            continue
        polygon = polygon.tolist() if hasattr(polygon, "tolist") else polygon
        polygon = [[float(x), float(y)] for x, y in polygon]
        lines.append(Line(id=f"l{index:03d}", text=text, score=round(score, 4),
                          polygon=polygon, bbox=_bbox_from_polygon(polygon)))
    return lines


def group_lines(lines: list[Line], *, max_gap_ratio: float = 0.55) -> list[dict]:
    """Group recognised lines into text boxes.

    Lines join when they are horizontally aligned, vertically adjacent within
    ``max_gap_ratio`` of the taller line's height, and of a similar type size.
    The grouping never edits line text: each box keeps its member line ids, so a
    reviewer can trace every character back to one OCR line.
    """
    ordered = sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0]))
    used: set[str] = set()
    boxes: list[dict] = []
    for line in ordered:
        if line.id in used:
            continue
        members = [line]
        used.add(line.id)
        changed = True
        while changed:
            changed = False
            current = members[-1]
            cx, cy = current.center
            best = None
            for candidate in ordered:
                if candidate.id in used:
                    continue
                px, py = candidate.center
                height = max(current.height, candidate.height)
                gap = candidate.bbox[1] - (current.bbox[1] + current.bbox[3])
                if not (-0.6 * height <= gap <= max_gap_ratio * height):
                    continue
                if abs(px - cx) > (current.bbox[2] + candidate.bbox[2]) * 0.5:
                    continue
                if candidate.height > current.height * 2 or current.height > candidate.height * 2:
                    continue
                if best is None or abs(py - cy) < abs(best.center[1] - cy):
                    best = candidate
            if best is not None:
                members.append(best)
                used.add(best.id)
                changed = True
        xs = [m.bbox[0] for m in members]
        ys = [m.bbox[1] for m in members]
        right = max(m.bbox[0] + m.bbox[2] for m in members)
        bottom = max(m.bbox[1] + m.bbox[3] for m in members)
        boxes.append({
            "id": f"tb{len(boxes):03d}",
            "lines": [m.id for m in members],
            "text": "\n".join(m.text for m in members),
            "bbox": [min(xs), min(ys), right - min(xs), bottom - min(ys)],
            "line_count": len(members),
            "score": round(sum(m.score for m in members) / len(members), 4),
        })
    boxes.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    for index, box in enumerate(boxes):
        box["id"] = f"tb{index:03d}"
    return boxes


def to_record(lines: list[Line], boxes: list[dict], image_size) -> dict:
    return {
        "schema": "OCRLines/1",
        "image_size": list(image_size),
        "engine": "PP-OCRv4 (det/cls/rec onnx, local)",
        "line_count": len(lines),
        "box_count": len(boxes),
        "lines": [
            {"id": l.id, "text": l.text, "score": l.score, "bbox": l.bbox, "polygon": l.polygon}
            for l in lines
        ],
        "boxes": boxes,
        "note": "box 的 text 由成员 line 原样拼接，分组不改写任何字符",
    }
