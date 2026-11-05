"""Regressions for bounded masks, gradient coordinates and surface ownership."""
import copy

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from _helpers import SRC
from background_models import fit_surface, gradient_pixels, precise_text_masks, inspect_residue, shape_mask
from surface_rebuild import ancestry, model_nodes, clean_page_plate, surface_quality, rollback_review
from simple_icons import contour_metrics, native_organization


def test_failed_nested_card_restores_its_whole_r3_container():
    review = {"regions": [
        {"id": "a", "bbox": [0, 0, 200, 200]},
        {"id": "b", "bbox": [5, 5, 100, 100], "parent_id": "a"},
        {"id": "c", "bbox": [10, 10, 20, 20], "parent_id": "b"},
        {"id": "d", "bbox": [250, 0, 200, 200]}]}
    revised = rollback_review(review, {"c"}, "contour failed")
    assert [item["id"] for item in revised["regions"]] == ["d"]
    assert revised["baseline_card_fallbacks"][0]["id"] == "a"
    assert len(review["regions"]) == 4


def test_wide_surface_gradient_direction_uses_pixel_coordinates():
    model = {"kind": "linear", "colors": [[200, 220, 240], [245, 250, 255]],
             "offsets": [0, 1], "direction": [0.1, 0.995], "angle_deg": 84.26}
    source = gradient_pixels(model, [0, 0, 800, 80])
    fitted, score = fit_surface(source, [0, 0, 800, 80], np.ones((80, 800), bool))
    assert score["mae"] <= 2
    assert score["p95"] <= 5
    assert fitted["kind"] == "linear"
    assert abs(fitted["angle_deg"] - model["angle_deg"]) < 5


def test_off_center_three_stop_fade_is_supported():
    model = {"kind": "linear", "colors": [[255, 255, 255], [225, 241, 254], [225, 241, 254]],
             "offsets": [0, 0.1, 1], "direction": [1, 0], "angle_deg": 0}
    source = gradient_pixels(model, [0, 0, 900, 60])
    fitted, score = fit_surface(source, [0, 0, 900, 60], np.ones((60, 900), bool))
    assert score["mae"] <= 2 and score["p95"] <= 5


def test_dense_text_does_not_turn_into_a_solid_repair_rectangle():
    image = Image.new("RGB", (500, 90), (245, 250, 254))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 28)
    draw.text((10, 20), "智慧工地与企业服务测试", font=font, fill=(20, 30, 40))
    source = np.array(image)
    protected = np.zeros((90, 500), bool)
    protected[:, 200:205] = True
    line = {"bbox": [8, 20, 340, 40], "style": {"color": "141E28"}}
    masks = precise_text_masks(source, line, protected)
    assert masks["core_pixels"] > 0
    assert 0 < masks["coverage"] < 0.8
    assert not masks["repair"][protected].any()
    assert np.all(masks["glyph"] <= masks["repair"])


def test_no_stroke_evidence_requires_review():
    source = np.full((40, 100, 3), 250, np.uint8)
    result = inspect_residue(source, source, np.zeros((40, 100), bool), np.ones((40, 100), bool))
    assert result["status"] == "REVIEW_REQUIRED"


def test_independent_inspection_uses_original_strokes():
    source = np.full((50, 100, 3), 250, np.uint8)
    source[20:35, 40:44] = 10
    core = np.zeros((50, 100), bool)
    core[20:35, 40:44] = True
    region = np.ones((50, 100), bool)
    assert inspect_residue(source, source, core, region)["status"] != "PASS"
    assert inspect_residue(source, np.full_like(source, 250), core, region)["status"] == "PASS"


def test_surface_hierarchy_rejects_cycles():
    regions = [{"id": "a", "parent_id": "b"}, {"id": "b", "parent_id": "a"}]
    with pytest.raises(ValueError, match="hierarchy"):
        ancestry(regions, regions[0])


def test_one_surface_has_one_native_shape_and_one_evidence_owner():
    region = {"id": "card", "bbox": [10, 20, 200, 100], "radius": 12, "border": False}
    components = [{"id": "card-body", "bbox": region["bbox"], "radius": 12, "part": "body",
                   "model": {"kind": "solid", "colors": [[230, 240, 250]], "offsets": [0]}}]
    nodes = model_nodes(region, components)
    assert len(nodes) == 1
    assert nodes[0]["evidence_ids"] == ["ev-card"]


def test_global_neutral_repair_preserves_colored_decoration():
    source = np.full((100, 200, 3), 253, np.uint8)
    source[80:90, 10:100] = (60, 150, 210)
    old = source.copy()
    old[20:40, 70:140] = (250, 251, 252)
    union = np.zeros((100, 200), bool)
    exclusions = np.zeros_like(union)
    cleaned, authorized, _, _ = clean_page_plate(old, union, source, exclusions)
    assert np.array_equal(cleaned[80:90, 10:100], old[80:90, 10:100])
    assert np.array_equal(cleaned[~authorized], old[~authorized])


def test_mask_corners_and_contour_checks_are_explicit():
    mask = shape_mask((200, 100), [10, 10, 100, 60], 12)
    assert mask[40, 60] and not mask[10, 10]
    assert contour_metrics(mask, mask)["iou"] == 1
    primitives = native_organization("icon", [10, 10, 64, 58], [32, 11, 6.5, 11, 52, 46, 6.5, 29, 3.5], "0066AA")
    assert len(primitives) <= 32
    assert all(node["type"] == "shape" for node in primitives)
