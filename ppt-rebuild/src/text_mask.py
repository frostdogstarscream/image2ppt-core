"""M3 — stroke-level text mask and layered erasing.

Three ideas, in the order the plan requires them:

1. The mask is built from glyph pixels (colour distance against the local
   background inside the OCR box), never from the box itself.
2. Whether a layer owns the text is decided by alpha overlap, and whether the
   layers *below* can carry the hole is decided by their coverage — that is what
   separates "erase for free" from "erase and repair".
3. A ghost check runs on the recomposed page: if text-coloured pixels survive in
   the masked area, the erase is refused rather than shipped.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

TEXT_COLOR_TOL = 40


@dataclass
class MaskResult:
    mask: np.ndarray
    bbox: list[int]
    pixels: int
    ratio_vs_bbox: float
    color: str


def _local_background(crop: np.ndarray, margin: int = 2) -> np.ndarray:
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    return np.median(border, axis=0)


def stroke_mask(rgb: np.ndarray, box_bbox, *, tolerance: int = TEXT_COLOR_TOL) -> MaskResult | None:
    """Glyph pixels of one OCR text box, at stroke level.

    The mask is deliberately restricted to pixels that differ from the local
    background *and* fall inside the OCR box, so erasing cannot touch neighbouring
    icons or borders.
    """
    x, y, w, h = [int(round(v)) for v in box_bbox]
    x, y = max(0, x), max(0, y)
    w, h = min(w, rgb.shape[1] - x), min(h, rgb.shape[0] - y)
    if w < 3 or h < 3:
        return None
    crop = rgb[y:y + h, x:x + w].astype(np.int16)
    background = _local_background(crop)
    distance = np.abs(crop - background).max(axis=2)
    local = distance > tolerance
    if local.sum() < 8:
        return None
    # Text colour is the population *farthest* from the background, not the darkest:
    # a white label on a blue banner would otherwise be measured as the banner colour.
    values = crop[local]
    local_distance = distance[local]
    core = values[local_distance >= np.percentile(local_distance, 75)]
    colour = np.median(core, axis=0).round().astype(int)
    full = np.zeros(rgb.shape[:2], bool)
    full[y:y + h, x:x + w] = local
    return MaskResult(mask=full, bbox=[x, y, w, h], pixels=int(local.sum()),
                      ratio_vs_bbox=round(float(local.sum()) / (w * h), 4),
                      color="".join(f"{v:02X}" for v in np.clip(colour, 0, 255)))


def mask_ratio_ok(result: MaskResult, *, low: float = 0.15, high: float = 0.85) -> tuple[bool, str]:
    """Reject a mask that has collapsed into the whole box (or nearly nothing)."""
    if result.pixels < 8:
        return False, "mask_too_small"
    if result.ratio_vs_bbox < low:
        return False, f"mask_too_sparse({result.ratio_vs_bbox})"
    if result.ratio_vs_bbox > high:
        return False, f"mask_covers_box({result.ratio_vs_bbox})"
    return True, "ok"


def layer_coverage(mask: np.ndarray, alpha: np.ndarray) -> float:
    region = mask.sum()
    if region == 0:
        return 0.0
    return float(((alpha > 24) & mask).sum() / region)


def owning_layer(mask: np.ndarray, layers: list[tuple[int, np.ndarray]]) -> tuple[int, dict]:
    """Which layer actually carries this text.

    An opaque background covers everything, so "highest coverage" would always
    return it. The text is *visible*, therefore it must sit on the topmost layer
    that still covers the whole mask.
    """
    scores = {z: layer_coverage(mask, alpha) for z, alpha in layers}
    covering = [z for z, s in scores.items() if s >= 0.90]
    if covering:
        owner = max(covering)
        rule = "topmost_full_coverage"
    else:
        candidates = {z: s for z, s in scores.items() if s > 0.05}
        owner = max(candidates, key=candidates.get) if candidates else None
        rule = "max_partial_coverage"
    return owner, {"per_layer": {str(z): round(s, 3) for z, s in scores.items()},
                   "rule": rule, "owner": owner}


def below_coverage(mask: np.ndarray, layers: list[tuple[int, np.ndarray]], owner_z: int) -> float:
    """Do the layers underneath fill the hole opaquely? If so, erasing is free."""
    region = mask.sum()
    if region == 0:
        return 0.0
    stacked = np.zeros(mask.shape, np.uint8)
    for z, alpha in layers:
        if z >= owner_z:
            continue
        stacked = np.maximum(stacked, alpha)
    return float(((stacked > 200) & mask).sum() / region)


def erase_by_alpha(rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgba.copy()
    out[mask, 3] = 0
    return out


def erase_by_inpaint(rgba: np.ndarray, mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """Keep alpha, replace the glyph pixels with the surrounding object colour."""
    out = rgba.copy()
    if not mask.any():
        return out
    rgb = out[..., :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    out[..., :3] = np.where(mask[..., None], cv2.cvtColor(filled, cv2.COLOR_BGR2RGB), rgb)
    return out


def ghost_ratio(recomposed: np.ndarray, mask: np.ndarray, text_color: str, *, tolerance: int = 26) -> float:
    """Fraction of masked pixels that still look like the original text colour."""
    if mask.sum() == 0:
        return 0.0
    target = np.array([int(text_color[i:i + 2], 16) for i in (0, 2, 4)], np.int16)
    region = recomposed[mask].astype(np.int16)
    close = np.abs(region - target).max(axis=1) < tolerance
    return float(close.mean())


def composite(layers: list[tuple[int, np.ndarray]], size) -> np.ndarray:
    """Flatten edited RGBA arrays back-to-front (straight alpha, as PNG stores it)."""
    height, width = size[1], size[0]
    canvas = np.full((height, width, 3), 255, np.float32)
    for _z, rgba in sorted(layers):
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        canvas = rgba[..., :3].astype(np.float32) * alpha + canvas * (1.0 - alpha)
    return np.clip(canvas, 0, 255).round().astype(np.uint8)


def remote_inpaint(image: np.ndarray, mask: np.ndarray, url: str, timeout: int = 180) -> np.ndarray:
    """Ask a Big-LaMa style service to repair a region. Raises on any failure."""
    import io
    import urllib.request
    import uuid

    def encode(array, mode):
        buffer = io.BytesIO()
        Image.fromarray(array, mode if mode else None).save(buffer, format="PNG")
        return buffer.getvalue()

    image_bytes = encode(image, "RGB")
    mask_bytes = encode((mask.astype(np.uint8) * 255), "L")
    boundary = uuid.uuid4().hex
    body = b""
    for name, payload in (("image", image_bytes), ("mask", mask_bytes)):
        body += b"--" + boundary.encode() + b"\r\n"
        body += f'Content-Disposition: form-data; name="{name}"; filename="{name}.png"\r\n'.encode()
        body += b"Content-Type: image/png\r\n\r\n" + payload + b"\r\n"
    body += b"--" + boundary.encode() + b"--\r\n"
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Content-Length": str(len(body))},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return np.array(Image.open(io.BytesIO(response.read())).convert("RGB"))


def topmost_opaque_below(layers: list[tuple[int, np.ndarray]], mask: np.ndarray, owner_z: int) -> int | None:
    """The layer that supplies the pixels seen through the erased hole."""
    best = None
    for z, rgba in sorted(layers):
        if z >= owner_z:
            continue
        if layer_coverage(mask, rgba[..., 3]) >= 0.98:
            best = z
    return best


def inpaint_rgb(image: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Repair an RGB page over a mask.

    The exporter ships the flattened plate, so the repair has to land on that
    composite. Fixing one layer underneath leaves the other layers' halo visible,
    which is exactly the faint ghost this deck shows.
    """
    if not mask.any():
        return image.copy()
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, (mask.astype(np.uint8) * 255), radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)


def grow(mask: np.ndarray, px: int = 3) -> np.ndarray:
    """Dilate a mask: glyph antialiasing and matte halos live just outside the stroke."""
    if px <= 0:
        return mask
    kernel = np.ones((px * 2 + 1, px * 2 + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def residual_contrast(image: np.ndarray, mask: np.ndarray, *, ring_px: int = 15, tolerance: int = 12) -> float:
    """How much of the masked area still stands out from its surroundings.

    This is the metric that actually detects a ghost: after erasing, the region
    should look like the background ring around it. Colour-matching against the
    original text colour misses faint residue, which is what a coarse matte leaves.
    """
    if mask.sum() == 0:
        return 0.0
    kernel = np.ones((ring_px * 2 + 1, ring_px * 2 + 1), np.uint8)
    ring = (cv2.dilate(mask.astype(np.uint8), kernel) - mask.astype(np.uint8)) > 0
    if not ring.any():
        return 0.0
    background = np.median(image[ring].astype(np.float32), axis=0)
    deviation = np.abs(image[mask].astype(np.float32) - background).max(axis=1)
    return float((deviation > tolerance).mean())


def describe(result: MaskResult) -> dict:
    return {"bbox": result.bbox, "pixels": result.pixels,
            "ratio_vs_bbox": result.ratio_vs_bbox, "color": result.color}
