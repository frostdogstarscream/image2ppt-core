"""Synthetic tests for automatic LayerD rebuild. No GPU or remote services."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_rebuild import build_plate, measure_page, text_mask_from_ocr
from geometry_fit import measure_component


def _pack(directory, layers, source):
    directory = Path(directory)
    directory.mkdir(parents=True)
    Image.fromarray(source).save(directory / "source.png")
    records = []
    for z, rgba in enumerate(layers):
        name = f"layer-{z:02d}.png"
        Image.fromarray(rgba).save(directory / name)
        alpha = rgba[..., 3]
        ys, xs = np.where(alpha > 8)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)] if len(xs) else [0, 0, 0, 0]
        records.append({
            "file": name, "z": z, "role": "background" if z == 0 else "foreground",
            "size": [source.shape[1], source.shape[0]], "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
            "alpha_mode": "straight", "alpha_bbox": bbox,
            "alpha_coverage": float((alpha > 8).mean()), "partial_alpha_fraction": float(((alpha > 8) & (alpha < 250)).mean()),
        })
    Image.fromarray(source).save(directory / "composite.png")
    manifest = {
        "id": directory.name,
        "image_size": [source.shape[1], source.shape[0]],
        "layer_order": "back_to_front",
        "alpha_mode": "straight",
        "normalized_sha256": hashlib.sha256((directory / "source.png").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256((directory / "source.png").read_bytes()).hexdigest(),
        "layers": records,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return directory


def test_nested_card_measures_three_native_levels():
    H, W = 360, 640
    source = np.full((H, W, 3), (246, 248, 252), np.uint8)
    bg = np.dstack([source, np.full((H, W), 255, np.uint8)])
    card = np.zeros((H, W, 4), np.uint8)
    cv2.rectangle(card, (40, 40), (400, 300), (255, 255, 255, 255), -1)
    cv2.rectangle(card, (40, 40), (400, 300), (200, 210, 225, 255), 1)
    header = np.zeros((H, W, 4), np.uint8)
    header[60:120, 60:380] = (20, 90, 200, 255)
    badge = np.zeros((H, W, 4), np.uint8)
    badge[75:105, 80:110] = (240, 150, 30, 255)
    source = source.copy()
    source[card[..., 3] > 0] = card[..., :3][card[..., 3] > 0]
    source[header[..., 3] > 0] = header[..., :3][header[..., 3] > 0]
    source[badge[..., 3] > 0] = badge[..., :3][badge[..., 3] > 0]
    comps = [
        {"layer_z": 1, "layer_file": "layer-01.png", "mask": card[..., 3] > 0, "alpha": card[..., 3],
         "rgba": card, "area": int((card[..., 3] > 0).sum()), "bbox": [40, 40, 361, 261]},
        {"layer_z": 2, "layer_file": "layer-02.png", "mask": header[..., 3] > 0, "alpha": header[..., 3],
         "rgba": header, "area": int((header[..., 3] > 0).sum()), "bbox": [60, 60, 320, 60]},
        {"layer_z": 3, "layer_file": "layer-03.png", "mask": badge[..., 3] > 0, "alpha": badge[..., 3],
         "rgba": badge, "area": int((badge[..., 3] > 0).sum()), "bbox": [80, 75, 30, 30]},
    ]
    empty = np.zeros((H, W), bool)
    elements = measure_page(source, comps, empty)
    shapes = [e for e in elements if e["kind"] == "native_shape"]
    assert len(shapes) >= 3
    levels = sorted(e["level"] for e in shapes[:3])
    assert levels == [0, 1, 2]


def test_gradient_band_gets_fill_gradient():
    H, W = 200, 400
    rgb = np.full((H, W, 3), 246, np.uint8)
    t = np.linspace(0, 1, 300)[None, :, None]
    band = np.rint((1 - t) * np.array([20, 90, 200]) + t * np.array([40, 180, 230])).astype(np.uint8)
    rgb[60:140, 50:350] = band
    mask = np.zeros((H, W), bool)
    mask[60:140, 50:350] = True
    alpha = mask.astype(np.uint8) * 255
    record = measure_component(rgb, mask, protect=np.zeros((H, W), bool), alpha=alpha)
    assert record is not None
    assert record["kind"] == "native_shape"
    assert "fill_gradient" in record


def test_semi_transparent_card_stays_picture():
    H, W = 240, 320
    rgb = np.full((H, W, 3), 240, np.uint8)
    rgb[40:200, 40:280] = (20, 90, 200)
    # Wave texture under a translucent white card.
    for x in range(W):
        rgb[180:, x] = (200, 220, 240) if (x // 8) % 2 == 0 else (180, 210, 235)
    mask = np.zeros((H, W), bool)
    mask[40:200, 40:280] = True
    alpha = np.zeros((H, W), np.uint8)
    alpha[mask] = 140
    record = measure_component(rgb, mask, protect=np.zeros((H, W), bool), alpha=alpha)
    assert record is None
    comps = [{"layer_z": 1, "layer_file": "l1", "mask": mask, "alpha": alpha,
              "rgba": np.dstack([np.full((H, W, 3), 255, np.uint8), alpha]),
              "area": int(mask.sum()), "bbox": [40, 40, 240, 160]}]
    elements = measure_page(rgb, comps, np.zeros((H, W), bool))
    assert len(elements) == 1
    assert elements[0]["kind"] == "raster" and elements[0].get("fallback")


def test_textured_hole_calls_inpaint_flat_hole_does_not():
    H, W = 120, 160
    flat = np.full((H, W, 3), 245, np.uint8)
    texture = flat.copy()
    rng = np.random.default_rng(0)
    texture[:] = rng.integers(40, 220, texture.shape, dtype=np.uint8)
    hole = np.zeros((H, W), bool)
    hole[30:90, 40:120] = True
    bg = np.dstack([flat, np.full((H, W), 255, np.uint8)])
    elements = [{"mask": hole, "kind": "native_shape"}]
    calls = {"n": 0}

    def fake_inpaint(image, mask, url, timeout=180):
        calls["n"] += 1
        out = image.copy()
        out[mask] = (10, 10, 10)
        return out

    plate, report = build_plate(flat, [bg], elements, np.zeros((H, W), bool),
                                inpaint_url="http://example/inpaint", inpaint=fake_inpaint)
    assert calls["n"] == 0
    assert report[0]["method"] == "layerd_lower_pixels"
    assert (plate[hole] == flat[hole]).all() or (plate[30:90, 40:120] == flat[30:90, 40:120]).all()

    plate2, report2 = build_plate(texture, [bg], elements, np.zeros((H, W), bool),
                                  inpaint_url="http://example/inpaint", inpaint=fake_inpaint)
    assert calls["n"] == 1
    assert report2[0]["method"] == "big_lama_textured_hole"
    assert (plate2[hole] == (10, 10, 10)).all()


def test_auto_run_with_local_pack(tmp_path, monkeypatch):
    pytest.importorskip("rapidocr")
    if not (ROOT / "models" / "det.onnx").exists():
        pytest.skip("ppt-rebuild OCR models missing")
    from auto_rebuild import run_auto

    H, W = 360, 640
    source = np.full((H, W, 3), (246, 248, 252), np.uint8)
    cv2.rectangle(source, (40, 40), (400, 220), (255, 255, 255), -1)
    cv2.rectangle(source, (40, 40), (400, 220), (200, 210, 225), 1)
    source[60:110, 60:380] = (20, 90, 200)
    cv2.putText(source, "TITLE", (80, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    bg = np.dstack([np.full((H, W, 3), (246, 248, 252), np.uint8), np.full((H, W), 255, np.uint8)])
    card = np.zeros((H, W, 4), np.uint8)
    cv2.rectangle(card, (40, 40), (400, 220), (255, 255, 255, 255), -1)
    header = np.zeros((H, W, 4), np.uint8)
    header[60:110, 60:380] = (20, 90, 200, 255)
    pack = _pack(tmp_path / "pack", [bg, card, header], source)
    image = tmp_path / "page.png"
    Image.fromarray(source).save(image)

    def fake_compile(backend, ir):
        preview = tmp_path / "preview.png"
        Image.fromarray(source).save(preview)
        pptx = tmp_path / "result.pptx"
        pptx.write_bytes(b"PK\x03\x04dummy")
        return {"pptx": str(pptx), "preview": str(preview), "quality": {}}

    monkeypatch.setattr("rebuild_pipeline.compile_ir", fake_compile)
    monkeypatch.setattr("rebuild_pipeline.edit_probes", lambda *a, **k: [])
    monkeypatch.setattr(
        "rebuild_pipeline.pixels",
        lambda path, size: np.array(Image.open(path).convert("RGB").resize(tuple(size))),
    )
    result = run_auto(image, tmp_path / "out", pack=pack, inpaint_url="", probes=False, timeout=60)
    assert result["status"] == "REVIEW_REQUIRED"
    assert Path(result["directory"], "delivery.json").exists()
    components = json.loads(Path(result["directory"], "components.json").read_text(encoding="utf-8"))
    assert components["native_shapes"] >= 1
