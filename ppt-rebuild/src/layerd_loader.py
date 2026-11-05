"""M1 — LayerD layer parser.

Reads the manifest plus the full-size RGBA layers, validates them, recomposes the
page back-to-front and checks that recomposition against the service's own
``composite.png``. Region candidates are emitted as *regions*, never as semantic
objects: connectivity is pixel evidence, not a claim about what the thing is.

Coordinates are always full-page pixels; a layer is never treated as if its
content started at (0, 0).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ALPHA_TOL = 2.0


@dataclass
class Layer:
    file: str
    path: Path
    z: int
    role: str
    size: tuple[int, int]
    mode: str
    alpha_bbox: list[int]
    alpha_coverage: float
    partial_alpha_fraction: float
    sha256: str
    alpha_mode: str = "straight"
    regions: list[dict] = field(default_factory=list)


@dataclass
class Page:
    page_id: str
    directory: Path
    size: tuple[int, int]
    layer_order: str
    source_path: Path
    composite_path: Path | None
    layers: list[Layer]

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256((self.directory / "manifest.json").read_bytes()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detect_alpha_mode(rgba: np.ndarray) -> str:
    """Straight vs premultiplied alpha.

    PNG stores straight alpha, so that is the default. The premultiplied test only
    looks at *semi-transparent* pixels: premultiplied data cannot have a colour
    channel above alpha, while straight data routinely does (a bright glyph edge
    keeps its colour at alpha=20). Testing fully opaque or fully transparent pixels
    proves nothing, which is why they are excluded.
    """
    alpha = rgba[..., 3]
    semi = (alpha > 8) & (alpha < 64)
    if semi.sum() < 16:
        return "straight"
    rgb = rgba[..., :3]
    above = (rgb > alpha[..., None] + ALPHA_TOL).any(axis=2)
    return "straight" if above[semi].mean() > 0.25 else "premultiplied"


def layer_alpha_bbox(rgba: np.ndarray, threshold: int = 8) -> list[int]:
    mask = rgba[..., 3] > threshold
    if not mask.any():
        return [0, 0, 0, 0]
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def region_candidates(rgba: np.ndarray, *, min_area: int = 64, threshold: int = 24) -> list[dict]:
    """Connected alpha regions as pixel evidence only."""
    mask = (rgba[..., 3] > threshold).astype(np.uint8)
    if not mask.any():
        return []
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    regions = []
    height, width = mask.shape
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area < min_area:
            continue
        regions.append({
            "bbox": [x, y, w, h],
            "pixels": area,
            "fill_ratio": round(area / float(max(1, w * h)), 4),
            "touches_edge": x == 0 or y == 0 or x + w >= width or y + h >= height,
        })
    regions.sort(key=lambda r: -r["pixels"])
    return regions


def load_layer(path: Path, meta: dict) -> Layer:
    with Image.open(path) as image:
        image = image.convert("RGBA")
        array = np.array(image)
        size = image.size
        mode = image.mode
    if size != tuple(meta["size"]):
        raise ValueError(f"{path.name}: size {size} does not match manifest {meta['size']}")
    return Layer(
        file=meta["file"],
        path=path,
        z=int(meta["z"]),
        role=meta.get("role", "unknown"),
        size=size,
        mode=mode,
        alpha_bbox=[int(v) for v in meta.get("alpha_bbox", layer_alpha_bbox(array))],
        alpha_coverage=float(meta.get("alpha_coverage", 0.0)),
        partial_alpha_fraction=float(meta.get("partial_alpha_fraction", 0.0)),
        sha256=meta.get("sha256", _sha256(path)),
        alpha_mode=detect_alpha_mode(array),
        regions=region_candidates(array),
    )


def load_page(page_dir: str | Path) -> Page:
    directory = Path(page_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    size = tuple(int(v) for v in manifest["image_size"])
    layers = [load_layer(directory / meta["file"], meta) for meta in manifest["layers"]]
    layers.sort(key=lambda layer: layer.z)
    if [layer.z for layer in layers] != list(range(len(layers))):
        raise ValueError("manifest z values must be a dense 0..n-1 sequence")
    source = directory / "source.png"
    if not source.exists():
        raise FileNotFoundError(f"{directory}: source.png missing")
    composite = directory / "composite.png"
    with Image.open(source) as image:
        if image.size != size:
            raise ValueError(f"{directory}: source size {image.size} != manifest {size}")
    return Page(
        page_id=manifest.get("id") or directory.name,
        directory=directory,
        size=size,
        layer_order=manifest.get("layer_order", "back_to_front"),
        source_path=source,
        composite_path=composite if composite.exists() else None,
        layers=layers,
    )


def recompose(page: Page, background=(255, 255, 255)) -> np.ndarray:
    """Flatten back-to-front onto an opaque background, honouring alpha convention."""
    width, height = page.size
    canvas = np.zeros((height, width, 3), np.float32)
    canvas[..., :] = np.array(background, np.float32)
    for layer in page.layers:
        with Image.open(layer.path) as image:
            rgba = np.array(image.convert("RGBA")).astype(np.float32)
        alpha = rgba[..., 3:4] / 255.0
        rgb = rgba[..., :3]
        if layer.alpha_mode == "premultiplied":
            canvas = rgb + canvas * (1.0 - alpha)
        else:
            canvas = rgb * alpha + canvas * (1.0 - alpha)
    return np.clip(canvas, 0, 255).round().astype(np.uint8)


def verify(page: Page) -> dict:
    """Recomposition must reproduce the service's own composite, else we misread the layers."""
    result = {"page": page.directory.name, "size": list(page.size),
              "layers": len(page.layers), "composite_available": page.composite_path is not None}
    rendered = recompose(page).astype(np.int16)
    mse = float(((rendered - rendered) ** 2).mean())  # placeholder replaced below
    if page.composite_path is not None:
        with Image.open(page.composite_path) as image:
            composite = np.array(image.convert("RGB")).astype(np.int16)
        if composite.shape != rendered.shape:
            raise ValueError(f"{page.directory}: composite size mismatch")
        diff = np.abs(rendered - composite)
        mae = float(diff.mean())
        mse = float((diff.astype(np.float32) ** 2).mean())
        result.update({
            "mae": round(mae, 3),
            "psnr": round(float(10 * np.log10(255.0 ** 2 / max(mse, 1e-9))), 2),
            "max_channel_error": int(diff.max()),
        })
    with Image.open(page.source_path) as image:
        source = np.array(image.convert("RGB")).astype(np.int16)
    result["source_mae"] = round(float(np.abs(rendered - source).mean()), 3)
    return result


def relocate_semantic_bbox(region_bbox: list[int], layer: Layer) -> list[int]:
    """Regions already live in full-page coordinates; this asserts the invariant."""
    x, y, w, h = region_bbox
    if x < 0 or y < 0 or x + w > layer.size[0] or y + h > layer.size[1]:
        raise ValueError(f"{layer.file}: region outside layer bounds")
    return [x, y, w, h]


def to_record(page: Page) -> dict:
    return {
        "schema": "LayerDParse/1",
        "page": page.directory.name,
        "manifest_sha256": page.manifest_sha256,
        "image_size": list(page.size),
        "layer_order": page.layer_order,
        "source": page.source_path.name,
        "composite": page.composite_path.name if page.composite_path else None,
        "layers": [
            {
                "file": layer.file,
                "z": layer.z,
                "role": layer.role,
                "size": list(layer.size),
                "mode": layer.mode,
                "alpha_mode": layer.alpha_mode,
                "alpha_bbox": layer.alpha_bbox,
                "alpha_coverage": layer.alpha_coverage,
                "partial_alpha_fraction": layer.partial_alpha_fraction,
                "sha256": layer.sha256,
                "region_count": len(layer.regions),
                "regions": layer.regions,
                "region_note": "连通区域是像素证据，不是语义对象",
            }
            for layer in page.layers
        ],
    }
