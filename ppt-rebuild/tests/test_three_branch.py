"""Regressions for input integrity, ownership and atomic pixel replacements."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from _helpers import SRC
from layerd_loader import load_page
from object_registry import ObjectRegistry, validate_bbox
from reconstruction import extract_rgba, layer_contributions, materialize, prepare_candidates
from slideir_adapter import export_ir, sha256
from text_mask import below_coverage


@pytest.fixture
def pack(tmp_path):
    rgb = np.full((48, 80, 3), 245, np.uint8)
    rgb[16:24, 15:42] = 20
    Image.fromarray(rgb).save(tmp_path / "source.png")
    Image.fromarray(rgb).save(tmp_path / "composite.png")
    rgba = np.dstack((rgb, np.full(rgb.shape[:2], 255, np.uint8)))
    Image.fromarray(rgba).save(tmp_path / "layer-00.png")
    manifest = {"image_size": [80, 48], "layer_order": "back_to_front",
                "source_sha256": sha256(tmp_path / "source.png"),
                "layers": [{"file": "layer-00.png", "z": 0, "role": "background",
                            "size": [80, 48], "sha256": sha256(tmp_path / "layer-00.png")}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    return tmp_path


def change_manifest(pack, mutation):
    file = pack / "manifest.json"
    record = json.loads(file.read_text("utf8"))
    mutation(record)
    file.write_text(json.dumps(record), encoding="utf8")


@pytest.mark.parametrize("field", ["source.png", "layer-00.png"])
def test_tampered_input_fails(pack, field):
    image = Image.open(pack / field)
    image.putpixel((0, 0), (1, 2, 3, 255) if image.mode == "RGBA" else (1, 2, 3))
    image.save(pack / field)
    with pytest.raises(ValueError, match="sha256"):
        load_page(pack)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(layer_order="front_to_back"),
    lambda r: r.update(alpha_mode="premultiplied"),
    lambda r: r["layers"][0].update(z=2),
    lambda r: r["layers"][0].update(size=[79, 48]),
    lambda r: r["layers"][0].update(file="../escape.png"),
])
def test_pack_contract_errors(pack, mutation):
    change_manifest(pack, mutation)
    with pytest.raises(ValueError):
        load_page(pack)


def test_uploaded_and_normalized_hashes_are_distinct(pack):
    normalized = sha256(pack / "source.png")
    uploaded = "a" * 64
    change_manifest(pack, lambda r: r.update(source_sha256=uploaded))
    (pack / "request.json").write_text(json.dumps({"source_sha256": uploaded, "normalized_sha256": normalized}), encoding="utf8")
    assert load_page(pack).size == (80, 48)
    (pack / "request.json").write_text(json.dumps({"source_sha256": uploaded, "normalized_sha256": "b" * 64}), encoding="utf8")
    with pytest.raises(ValueError, match="normalized sha256"):
        load_page(pack)


def test_dark_rgba_png_uses_declared_straight_alpha(pack):
    layer = np.full((48, 80, 4), [10, 20, 30, 40], np.uint8)
    Image.fromarray(layer).save(pack / "layer-00.png")
    change_manifest(pack, lambda r: r["layers"][0].update(sha256=sha256(pack / "layer-00.png")))
    assert load_page(pack).layers[0].alpha_mode == "straight"


def test_effective_alpha_uses_composition():
    alpha = np.full((3, 4), 150, np.uint8)
    assert below_coverage(np.ones((3, 4), bool), [(0, alpha), (1, alpha)], 2) == 1.0


def test_visible_layer_weights():
    first = np.full((2, 3, 4), [1, 2, 3, 255], np.uint8)
    second = np.full((2, 3, 4), [1, 2, 3, 128], np.uint8)
    rows = layer_contributions([first, second], np.ones((2, 3), bool))
    assert abs(sum(r["visible_weight"] for r in rows) - 1) < 1e-5
    assert rows[0]["visible_weight"] < rows[1]["visible_weight"]


def test_transparent_asset_recomposes_source():
    source = np.full((10, 14, 3), [25, 110, 180], np.uint8)
    background = np.full_like(source, 240)
    mask = np.zeros((10, 14), bool)
    mask[2:8, 3:11] = True
    rgba = extract_rgba(source, background, mask)
    alpha = rgba[..., 3:4].astype(float) / 255
    recomposed = rgba[..., :3] * alpha + background * (1 - alpha)
    assert np.abs(recomposed[mask] - source[mask]).max() <= 1
    assert np.all(rgba[~mask] == 0)


def test_rollback_restores_original_and_keeps_patch_origin():
    source = np.full((30, 50, 3), 120, np.uint8)
    mask = np.ones((5, 8), bool)
    replacement = {"bbox": [10, 10, 8, 5], "node": {"id": "text", "type": "text", "bbox": [12, 11, 8, 5]},
                   "local_mask": mask, "repair": np.full((5, 8, 3), 255, np.uint8), "asset": None}
    plate, nodes, _ = materialize(source, {"text": replacement}, set())
    assert np.array_equal(np.array(plate), source) and not nodes
    plate, _, _ = materialize(source, {"text": replacement}, {"text"})
    assert np.all(np.array(plate)[10:15, 10:18] == 255)
    assert np.all(np.array(plate)[15] == 120)


def test_font_failure_never_erases_source(pack, tmp_path, monkeypatch):
    import reconstruction
    page = load_page(pack)
    ocr = {"source_sha256": sha256(page.source_path), "image_size": [80, 48],
           "lines": [{"id": "l000", "bbox": [14, 15, 29, 10], "text": "文字", "score": 1}], "boxes": []}
    monkeypatch.setattr(reconstruction, "ocr_record", lambda *a: ocr)
    monkeypatch.setattr(reconstruction, "fit_visual", lambda *a, **k: {"size_pt": None})
    out = tmp_path / "trial"
    out.mkdir()
    registry, replacements, source, _ = prepare_candidates(page, out)
    assert not replacements
    assert registry.objects[0]["state"] == "reverted"
    assert registry.objects[0]["reason"] == "font matching failed"
    plate, nodes, _ = materialize(source, replacements, set())
    assert np.array_equal(np.array(plate), source) and not nodes


def test_ownership_and_dynamic_canvas(pack, tmp_path):
    registry = ObjectRegistry([80, 48], sha256(pack / "source.png"))
    entry = registry.add({"id": "text", "kind": "text", "bbox": [10, 10, 20, 10], "text": "文字"})
    registry.transition(entry, "reverted", reason="font failed")
    revision = tmp_path / "revision"
    (revision / "source").mkdir(parents=True)
    (revision / "source/page.png").write_bytes((pack / "source.png").read_bytes())
    ir_path = export_ir(revision, registry, [], Image.open(pack / "source.png"), {}, notes=[])
    ir = json.loads(ir_path.read_text("utf8"))
    assert ir["canvas"]["source_px"] == [80, 48]
    assert registry.validate_owners(ir)["ev-text"] == "background-image"
    ir["nodes"].append({"id": "duplicate", "type": "shape", "evidence_ids": ["ev-text"]})
    with pytest.raises(ValueError, match="duplicate"):
        registry.validate_owners(ir)


@pytest.mark.parametrize("bbox", [[0, 0, 0, 10], [float("nan"), 0, 1, 1], [79, 0, 2, 1], [0.5, 0, 2, 1]])
def test_coordinate_errors(bbox):
    with pytest.raises(ValueError):
        validate_bbox(bbox, [80, 48])


def test_render_gate_rolls_back_pixels_and_object_together(pack, tmp_path, monkeypatch):
    import rebuild_pipeline as pipeline
    page = load_page(pack)
    original_hash = sha256(page.source_path)
    source = np.array(Image.open(page.source_path).convert("RGB"))
    layers = [np.array(Image.open(page.layers[0].path))]

    def candidates(*args, **kwargs):
        registry = ObjectRegistry(page.size, original_hash)
        entry = registry.add({"id": "shape-test", "kind": "shape", "bbox": [14, 15, 29, 10]})
        registry.transition(entry, "matched")
        mask = np.zeros(source.shape[:2], bool)
        mask[15:25, 14:43] = True
        replacement = {"bbox": entry["bbox"], "mask": mask, "local_mask": np.ones((10, 29), bool),
                       "repair": np.full((10, 29, 3), 245, np.uint8), "asset": None,
                       "node": {"id": entry["id"], "bbox": entry["bbox"], "type": "shape",
                                "geometry": "rect", "fill": "141414", "evidence_ids": [entry["evidence_id"]]}}
        return registry, {entry["id"]: replacement}, source, layers

    calls = []

    def compiler(backend, ir):
        folder = ir.parent / "compiled"
        folder.mkdir()
        preview = source.copy()
        if not calls:
            preview[15:25, 14:43] = [0, 255, 0]
        Image.fromarray(preview).save(folder / "preview.png")
        (folder / "result.pptx").write_bytes(b"synthetic compiler stub")
        (folder / "quality.json").write_text("{}", encoding="utf8")
        calls.append(ir)
        return {"pptx": str(folder / "result.pptx"), "preview": str(folder / "preview.png"),
                "quality": str(folder / "quality.json")}

    monkeypatch.setattr(pipeline, "prepare_candidates", candidates)
    monkeypatch.setattr(pipeline, "compile_ir", compiler)
    monkeypatch.setattr(pipeline, "ArtifactBackend", lambda *a: object())
    result = pipeline.run_page(pack.parent, pack.name, tmp_path / "deliveries", probes=False, align=False)
    assert len(calls) == 2
    assert result["counts"]["shape"] == 0
    final = Path(result["ir"]).parent
    assert np.array_equal(np.array(Image.open(final / "assets/plate.png")), source)
    registry = json.loads((Path(result["directory"]) / "registry.json").read_text("utf8"))
    assert registry["objects"][0]["state"] == "reverted"
    assert registry["objects"][0]["final_owner"] == "background-image"
    assert sha256(page.source_path) == original_hash
    assert len(list((Path(result["directory"]) / "revisions").iterdir())) == 2


def test_owned_masks_protect_neighbouring_icon(pack, tmp_path, monkeypatch):
    import reconstruction
    from slideir_adapter import write_json
    page = load_page(pack)
    review = tmp_path / "review.json"
    write_json(review, {"source_sha256": sha256(page.source_path), "image_size": [80, 48],
                        "objects": [{"id": "icon", "kind": "image", "bbox": [12, 14, 20, 12]}]})
    ocr = {"image_size": [80, 48], "lines": [{"id": "l0", "bbox": [28, 15, 16, 10], "text": "文字", "score": 1}]}
    monkeypatch.setattr(reconstruction, "ocr_record", lambda *a: ocr)
    out = tmp_path / "protected"
    out.mkdir()
    registry, replacements, source, _ = prepare_candidates(page, out, review_file=review)
    assert any(e["state"] == "reverted" and "protected" in e.get("reason", "") for e in registry.objects)
    masks = [v["mask"] for v in replacements.values()]
    if masks:
        assert np.max(np.sum(masks, axis=0)) <= 1


def test_diagnostic_edits_preserve_unrelated_package_members(tmp_path):
    from zipfile import ZipFile
    from edit_checks import edit_final_copy, NS
    import xml.etree.ElementTree as ET

    source = tmp_path / "source.pptx"
    xml = f'''<p:sld xmlns:p="{NS['p']}" xmlns:a="{NS['a']}"><p:cSld><p:spTree>
      <p:grpSp><p:nvGrpSpPr><p:cNvPr id="1" name="icon-group"/></p:nvGrpSpPr></p:grpSp>
      <p:sp><p:nvSpPr><p:cNvPr id="2" name="title"/></p:nvSpPr><p:txBody><a:p><a:r><a:t>原文</a:t></a:r></a:p></p:txBody></p:sp>
      </p:spTree></p:cSld></p:sld>'''
    with ZipFile(source, "x") as z:
        z.writestr("ppt/slides/slide1.xml", xml)
        z.writestr("unchanged.bin", b"original bytes")
    original_hash = sha256(source)
    edited = tmp_path / "edited.pptx"
    edit_final_copy(source, edited, delete_groups=["icon-group"], replace_text=("title", "编辑验证"))
    with ZipFile(edited) as z:
        assert z.read("unchanged.bin") == b"original bytes"
        root = ET.fromstring(z.read("ppt/slides/slide1.xml"))
        assert not root.findall(".//p:grpSp", NS)
        assert root.find(".//a:t", NS).text == "编辑验证"
    assert sha256(source) == original_hash
    with pytest.raises(ValueError, match="expected one"):
        edit_final_copy(source, tmp_path / "missing.pptx", delete_groups=["unknown-group"])
