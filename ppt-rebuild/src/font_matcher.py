"""M2b — font size, family and colour recovered by reverse rendering.

The size is not guessed from the bounding box height: candidate sizes are rendered
with real font files and compared against the ink actually present in the page.
Only fonts that exist on this machine are candidates, and the result keeps the
runner-up scores so a reviewer can see how close the decision was.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path("C:/Windows/Fonts")
CANDIDATE_FILES = {
    "微软雅黑": ["msyh.ttc", "msyhbd.ttc"],
    "黑体": ["simhei.ttf"],
    "宋体": ["simsun.ttc"],
    "等线": ["Deng.ttf", "Dengb.ttf"],
    "Arial": ["arial.ttf", "arialbd.ttf"],
}

# 1672 px wide page exported at 13.333 in -> 125.4 dpi, so 1 pt = 125.4/72 px.
DEFAULT_DPI = 125.4


@dataclass
class Ink:
    bbox: list[int]
    pixels: int
    color: str
    coverage: float


@dataclass
class Fit:
    family: str
    file: str
    bold: bool
    size_px: float
    size_pt: float
    score: float
    rendered: list[int]
    target: list[int]


def ink_bbox(rgb: np.ndarray, bbox, *, tolerance: int = 40) -> Ink | None:
    """Ink of one text box: pixels that differ from the local background."""
    x, y, w, h = [int(v) for v in bbox]
    x, y = max(0, x), max(0, y)
    w, h = min(w, rgb.shape[1] - x), min(h, rgb.shape[0] - y)
    if w < 3 or h < 3:
        return None
    crop = rgb[y:y + h, x:x + w].astype(np.int16)
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    background = np.median(border, axis=0)
    distance = np.abs(crop - background).max(axis=2)
    mask = distance > tolerance
    if mask.sum() < 12:
        return None
    ys, xs = np.nonzero(mask)
    ink_box = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    # Colour: the most common colour among the darkest third of ink pixels, which
    # avoids antialiased edges dragging the estimate toward the background.
    values = crop[mask]
    luminance = values.mean(axis=1)
    dark = values[luminance <= np.percentile(luminance, 35)]
    colour = np.median(dark, axis=0).round().astype(int)
    return Ink(
        bbox=[ink_box[0] + x, ink_box[1] + y, ink_box[2], ink_box[3]],
        pixels=int(mask.sum()),
        color="".join(f"{v:02X}" for v in np.clip(colour, 0, 255)),
        coverage=round(float(mask.mean()), 4),
    )


def _render(text: str, font_path: Path, size_px: int) -> tuple[int, int]:
    font = ImageFont.truetype(str(font_path), size_px)
    canvas = Image.new("L", (size_px * (len(text) + 2), size_px * 3), 0)
    ImageDraw.Draw(canvas).text((size_px, size_px), text, font=font, fill=255)
    array = np.array(canvas) > 40
    if not array.any():
        return 0, 0
    ys, xs = np.nonzero(array)
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _size_for_width(text: str, font_path: Path, target_w: int, bounds) -> list[float]:
    """Rendered width grows with size, so search the size that matches the ink width.

    Returns a few candidate sizes around the crossing point (plus the bounds) so the
    caller can still judge height.
    """
    low, high = bounds
    for _ in range(9):
        mid = (low + high) / 2.0
        width, _ = _render(text, font_path, max(1, int(round(mid))))
        if width < target_w:
            low = mid
        else:
            high = mid
    centre = (low + high) / 2.0
    return [max(1, int(round(centre + delta))) for delta in (-2, -1, 0, 1, 2)]


def fit_text(text: str, ink: Ink, *, dpi: float = DEFAULT_DPI,
             size_range=(6, 72, 0.5), families: list[str] | None = None) -> list[Fit]:
    """Render candidates and score them against the measured ink box."""
    target_w, target_h = ink.bbox[2], ink.bbox[3]
    start, stop, _step = size_range
    bounds = (start * dpi / 72.0, stop * dpi / 72.0)
    families = families or list(CANDIDATE_FILES)
    fits: list[Fit] = []
    for family in families:
        for filename in CANDIDATE_FILES.get(family, []):
            path = FONT_DIR / filename
            if not path.exists():
                continue
            bold = "bd" in filename.lower() or "b." in filename.lower()
            best: Fit | None = None
            for size_px in _size_for_width(text, path, target_w, bounds):
                width, height = _render(text, path, size_px)
                if width == 0:
                    continue
                # Width matters most for CJK lines (advance is an exact multiple of em);
                # height is noisier because of ascender/descender room.
                score = abs(width - target_w) / max(1, target_w) * 0.75 + abs(height - target_h) / max(1, target_h) * 0.25
                if best is None or score < best.score:
                    best = Fit(family=family, file=filename, bold=bold, size_px=size_px,
                               size_pt=round(size_px * 72.0 / dpi, 2), score=round(score, 4),
                               rendered=[width, height], target=[target_w, target_h])
            if best is not None:
                fits.append(best)
    fits.sort(key=lambda f: f.score)
    return fits


def fit_box(rgb: np.ndarray, box: dict, *, dpi: float = DEFAULT_DPI,
            size_range=(6, 72, 0.5), families: list[str] | None = None) -> dict:
    """Fit one OCR text box; returns the ranked candidates plus the ink measurement."""
    text = box["text"].replace("\n", "")
    ink = ink_bbox(rgb, box["bbox"])
    if ink is None:
        return {"box": box["id"], "ink": None, "candidates": [], "size_ok": False,
                "family_ambiguous": True, "size_pt": None, "family": None, "bold": None}
    candidates = fit_text(text, ink, dpi=dpi, size_range=size_range, families=families)
    best = candidates[0] if candidates else None
    # CJK families share the same em advance, so width cannot separate them. Report
    # the ambiguity instead of inventing a per-box family: the page-level vote decides.
    runner_up = candidates[1] if len(candidates) > 1 else None
    family_ambiguous = bool(runner_up and (runner_up.score - best.score) < 0.05) if best else True
    return {
        "box": box["id"],
        "text": box["text"],
        "bbox": box["bbox"],
        "ink": {"bbox": ink.bbox, "pixels": ink.pixels, "coverage": ink.coverage, "color": ink.color},
        "size_pt": best.size_pt if best else None,
        "family": best.family if best else None,
        "bold": best.bold if best else None,
        "fit_score": best.score if best else None,
        "size_ok": bool(best and best.score <= 0.12),
        "family_ambiguous": family_ambiguous,
        "candidates": [
            {"family": c.family, "size_pt": c.size_pt, "bold": c.bold,
             "score": c.score, "rendered": c.rendered, "target": c.target}
            for c in candidates[:4]
        ],
    }


def vote_family(fits: list[dict]) -> dict:
    """One page-level family wins by vote; a box may only dissent with strong evidence."""
    tally: dict[str, int] = {}
    for fit in fits:
        family = fit.get("family")
        if family:
            tally[family] = tally.get(family, 0) + 1
    total = sum(tally.values())
    if not total:
        return {"family": None, "share": 0.0, "tally": {}}
    winner = max(tally.items(), key=lambda kv: kv[1])[0]
    assigned = 0
    for fit in fits:
        family = fit.get("family")
        if family == winner or not family:
            fit["font_family_final"] = winner
        else:
            gap = None
            if fit.get("candidates") and len(fit["candidates"]) > 1:
                own = fit["candidates"][0]["score"]
                best_winner = next((c["score"] for c in fit["candidates"] if c["family"] == winner), None)
                gap = (own - best_winner) if best_winner is not None else None
            # Keep a different family only when it wins by a wide margin.
            if gap is not None and gap > 0.15:
                fit["font_family_final"] = family
                fit["family_dissent"] = True
            else:
                fit["font_family_final"] = winner
        assigned += fit.get("font_family_final") == winner
    return {"family": winner, "share": round(assigned / max(1, len(fits)), 3), "tally": tally}
