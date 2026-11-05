"""M6 — visual validation.

Renders are compared against the source at the reference resolution. Three kinds of
result are reported separately, because a single number cannot tell "looks similar"
from "is editable":

  visual      MAE / PSNR / SSIM of the whole page
  text        error measured on text pixels only (the renderer floor lives here)
  structure   per-object error, so a broken object can be named instead of averaged

Nothing here decides acceptance on its own.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str | Path, size=None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size and image.size != tuple(size):
            image = image.resize(tuple(size), Image.LANCZOS)
        return np.array(image).astype(np.float32)


def metrics(source: np.ndarray, render: np.ndarray) -> dict:
    diff = np.abs(source - render)
    mae = float(diff.mean())
    mse = float((diff ** 2).mean())
    psnr = float(10 * np.log10(255.0 ** 2 / max(mse, 1e-9)))
    return {"mae": round(mae, 3), "psnr": round(psnr, 2), "ssim": round(ssim(source, render), 4)}


def _box_mean(array: np.ndarray, radius: int) -> np.ndarray:
    """Uniform-window mean via integral image (no scipy/cv2 needed)."""
    padded = np.pad(array, radius, mode="reflect")
    integral = padded.cumsum(0).cumsum(1)
    integral = np.pad(integral, ((1, 0), (1, 0)))
    k = 2 * radius + 1
    height, width = array.shape
    total = (integral[k:k + height, k:k + width]
             - integral[0:height, k:k + width]
             - integral[k:k + height, 0:width]
             + integral[0:height, 0:width])
    return total / (k * k)


def ssim(source: np.ndarray, render: np.ndarray, radius: int = 5) -> float:
    a = source.mean(axis=2)
    b = render.mean(axis=2)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = _box_mean(a, radius), _box_mean(b, radius)
    var_a = _box_mean(a * a, radius) - mu_a ** 2
    var_b = _box_mean(b * b, radius) - mu_b ** 2
    cov = _box_mean(a * b, radius) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float((numerator / denominator).mean())


def text_mask_from_scene(scene: dict, shape) -> np.ndarray:
    mask = np.zeros(shape[:2], bool)
    for obj in scene["objects"]:
        if obj["type"] != "text":
            continue
        x, y, w, h = [int(round(v)) for v in obj["bbox_px"]]
        x, y = max(0, x), max(0, y)
        mask[y:y + h, x:x + w] = True
    return mask


def per_object(scene: dict, source: np.ndarray, render: np.ndarray) -> list[dict]:
    diff = np.abs(source - render).mean(axis=2)
    rows = []
    for obj in scene["objects"]:
        x, y, w, h = [int(round(v)) for v in obj["bbox_px"]]
        x, y = max(0, x), max(0, y)
        w, h = min(w, source.shape[1] - x), min(h, source.shape[0] - y)
        if w <= 1 or h <= 1:
            continue
        rows.append({
            "id": obj["id"],
            "type": obj["type"],
            "editable_level": obj.get("editable_level"),
            "bbox": [x, y, w, h],
            "mae": round(float(diff[y:y + h, x:x + w].mean()), 2),
        })
    rows.sort(key=lambda r: -r["mae"])
    return rows


def evaluate(scene: dict, source_path, render_path, *, out_dir=None) -> dict:
    size = tuple(scene["canvas_px"])
    source = load_rgb(source_path, size)
    render = load_rgb(render_path, size)
    mask = text_mask_from_scene(scene, source.shape)
    page = metrics(source, render)
    text_pixels = float(np.abs(source[mask] - render[mask]).mean()) if mask.any() else 0.0
    non_text = float(np.abs(source[~mask] - render[~mask]).mean()) if (~mask).any() else 0.0
    objects = per_object(scene, source, render)
    worst = objects[0] if objects else None

    result = {
        "schema": "VisualEval/1",
        "page": scene["slide_id"],
        "canvas_px": list(size),
        "visual": page,
        "text": {
            "coverage": round(float(mask.mean()), 4),
            "mae_on_text_pixels": round(text_pixels, 3),
            "mae_on_non_text_pixels": round(non_text, 3),
            "contribution_to_page_mae": round(text_pixels * float(mask.mean()), 3),
            "note": "文字像素误差主要由渲染器差异决定，不作为可优化指标",
        },
        "structure": {
            "objects": len(objects),
            "worst": worst,
            "over_10": [o["id"] for o in objects if o["mae"] > 10],
        },
        "objects": objects,
    }
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        diff = np.abs(source - render).mean(axis=2)
        heat = np.zeros((*diff.shape, 3), np.uint8)
        heat[..., 0] = np.clip(diff * 3, 0, 255)
        heat[..., 1] = np.clip(255 - diff * 3, 0, 255)
        Image.fromarray(heat).save(out / "diff-heatmap.png")
        side = Image.new("RGB", (size[0], size[1] * 2 + 8), "white")
        side.paste(Image.open(source_path).convert("RGB").resize(size, Image.LANCZOS), (0, 0))
        side.paste(Image.open(render_path).convert("RGB").resize(size, Image.LANCZOS), (0, size[1] + 8))
        side.save(out / "compare.png")
        result["artifacts"] = {"heatmap": str(out / "diff-heatmap.png"), "compare": str(out / "compare.png")}
    return result
