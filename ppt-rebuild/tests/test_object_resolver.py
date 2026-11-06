"""Ownership contracts reject uncertainty rather than authorize bbox erasure."""
import copy
import json

import numpy as np
import pytest
from PIL import Image

from _helpers import SRC
from object_resolver import (MASK_ROLES, authorize_repair, digest, from_ir, load_mask,
                             mask_ref, migrate_v1, registry_from_ir, resolve, validate_ir_owners, xywh_to_xyxy)


def legacy():
    return {"schema": "ObjectRegistry/1", "source_sha256": "a"*64,
            "canvas_px": [20, 15], "coordinate_space": "source_pixels_xywh",
            "objects": [{"id": "title", "kind": "text", "bbox": [2, 3, 5, 4],
                         "state": "committed", "final_owner": "title", "evidence_id": "e1"}]}


def saved_mask(tmp_path, name, array, **kwargs):
    file = tmp_path / (name + ".png")
    Image.fromarray(array).save(file)
    return mask_ref(file, [20, 15], root=tmp_path, **kwargs)


def writable(tmp_path):
    registry = migrate_v1(legacy()); obj = registry["objects"][0]
    pixels = np.zeros((15, 20), np.uint8); pixels[3:7, 2:7] = 255
    obj["action"] = "replace"
    obj["masks"] = {role: saved_mask(tmp_path, role, np.zeros_like(pixels) if role == "protection" else pixels)
                    for role in MASK_ROLES}
    return registry


def test_migration_is_read_only_and_does_not_invent_masks(tmp_path):
    old = legacy(); before = copy.deepcopy(old); new = migrate_v1(old)
    assert old == before
    assert new["schema"] == "ObjectRegistry/2"
    assert new["objects"][0]["bbox_xyxy"] == [2, 3, 7, 7]
    assert new["objects"][0]["status"] == "COMMITTED"
    assert all(m["status"] == "UNKNOWN" for m in new["objects"][0]["masks"].values())
    with pytest.raises(ValueError, match="not authorized"):
        authorize_repair(new, ["title"], tmp_path)
    new["objects"][0]["action"] = "replace"
    assert resolve(new, tmp_path)["conflicts"][0]["type"] == "UNKNOWN_REPAIR_AUTHORIZATION"


@pytest.mark.parametrize("box", [[2, 3, -1, 2], [19, 1, 2, 2], [float('nan'), 0, 1, 1]])
def test_bad_boxes_are_rejected(box):
    with pytest.raises(ValueError): xywh_to_xyxy(box, [20, 15])


def test_exact_resolved_pixels_only_are_authorized(tmp_path):
    registry = writable(tmp_path)
    allowed = authorize_repair(registry, ["title"], tmp_path)
    assert allowed.sum() == 20
    assert allowed[3:7, 2:7].all()
    assert registry["objects"][0]["status"] == "COMMITTED"  # resolver copies


def test_same_contribution_has_one_owner(tmp_path):
    registry = writable(tmp_path)
    other = copy.deepcopy(registry["objects"][0]); other.update(id="duplicate", output_owner="duplicate", action="retain")
    registry["objects"].append(other)
    assert any(c["type"] == "DUPLICATE_CONTRIBUTION_OWNER" for c in resolve(registry, tmp_path)["conflicts"])
    with pytest.raises(ValueError): authorize_repair(registry, ["title"], tmp_path)


def test_alpha_overlap_between_distinct_contributions_is_allowed(tmp_path):
    registry = writable(tmp_path); first = registry["objects"][0]
    second = copy.deepcopy(first)
    second.update(id="card", kind="shape", output_owner="card", evidence_ids=["e2"], contribution_ids=["e2"])
    first["transaction_id"] = second["transaction_id"] = "shared-card"
    alpha = np.zeros((15, 20), np.uint8); alpha[3:7, 2:7] = 128
    first["masks"]["visible"] = saved_mask(tmp_path, "alpha", alpha, semantics="alpha_coverage")
    registry["objects"].append(second)
    assert not resolve(registry, tmp_path)["conflicts"]
    assert authorize_repair(registry, ["title", "card"], tmp_path).sum() == 20
    second.pop("transaction_id")
    assert any(c["type"] == "SHARED_REPAIR_REQUIRES_TRANSACTION" for c in resolve(registry, tmp_path)["conflicts"])


def test_protected_arrow_blocks_text_repair(tmp_path):
    registry = writable(tmp_path); obj = registry["objects"][0]
    protected = np.zeros((15, 20), np.uint8); protected[4, 4] = 255
    obj["masks"]["protection"] = saved_mask(tmp_path, "protected", protected)
    assert any(c["type"] == "REPAIR_INTERSECTS_PROTECTION" for c in resolve(registry, tmp_path)["conflicts"])
    with pytest.raises(ValueError): authorize_repair(registry, ["title"], tmp_path)


def test_inspection_cannot_shrink_away_visible_source_strokes(tmp_path):
    registry = writable(tmp_path)
    mask = np.zeros((15, 20), np.uint8); mask[3:6, 2:7] = 255
    registry["objects"][0]["masks"]["inspection"] = saved_mask(tmp_path, "shrunk", mask)
    assert any(c["type"] == "INSPECTION_OMITS_VISIBLE_EVIDENCE" for c in resolve(registry, tmp_path)["conflicts"])


def test_mask_offset_dimensions_hash_and_semantics(tmp_path):
    mask = np.full((4, 5), 255, np.uint8)
    ref = saved_mask(tmp_path, "local", mask, origin_px=(2, 3))
    assert load_mask(ref, tmp_path, [20, 15])[3:7, 2:7].all()
    wrong = copy.deepcopy(ref); wrong["origin_px"] = [19, 3]
    with pytest.raises(ValueError, match="outside"): load_mask(wrong, tmp_path, [20, 15])
    Image.fromarray(np.zeros((4, 5), np.uint8)).save(tmp_path / "local.png")
    with pytest.raises(ValueError, match="hash"): load_mask(ref, tmp_path, [20, 15])
    with pytest.raises(ValueError, match="only 0 and 255"):
        saved_mask(tmp_path, "invalid-binary", np.full((4, 5), 128, np.uint8))


def ir_fixture(tmp_path):
    Image.new("RGB", (20, 15), "white").save(tmp_path / "source.png")
    return {"canvas": {"source_px": [20, 15]}, "source_image": "source.png",
            "source_sha256": digest(tmp_path / "source.png"), "background": {"color": "FFFFFF"},
            "nodes": [{"id": "title", "type": "text", "bbox": [2, 3, 5, 4], "z": 2, "evidence_ids": ["e1"]},
                      {"id": "icon", "type": "image", "bbox": [9, 3, 3, 4], "z": 1, "evidence_ids": [], "asset_id": "asset1"}],
            "assets": [{"id": "asset1", "contains_evidence_ids": ["e2"]}],
            "evidence": [{"id": "e1"}, {"id": "e2"}],
            "groups": [{"id": "card-edit-group", "member_ids": ["title", "icon"]}]}


def test_edit_groups_do_not_imply_semantic_parent_or_transaction_dependency(tmp_path):
    registry = registry_from_ir(ir_fixture(tmp_path)); obj = registry["objects"][0]
    assert obj["edit_group_ids"] == ["card-edit-group"]
    assert obj["semantic_parent_id"] is None and obj["transaction_dependencies"] == []
    registry["objects"][0]["transaction_dependencies"] = ["icon"]
    registry["objects"][1]["transaction_dependencies"] = ["title"]
    assert any(c["type"] == "CYCLIC_TRANSACTION_DEPENDENCY" for c in resolve(registry, tmp_path)["conflicts"])


def test_shadow_outputs_do_not_mutate_ir_or_authorize_legacy_boxes(tmp_path):
    irfile = tmp_path / "slideir.json"
    irfile.write_text(json.dumps(ir_fixture(tmp_path)), encoding="utf-8")
    before = digest(irfile)
    registry = from_ir(irfile, tmp_path / "shadow")
    assert before == digest(irfile)
    assert {p.name for p in (tmp_path / "shadow").iterdir()} == {
        "objects.json", "conflicts.json", "ownership-decisions.json", "overlap.png"}
    assert not registry["conflicts"]
    assert all(not d["repair_authorized"] for d in registry["decisions"])
    assert registry["review_required"]


def test_missing_evidence_is_a_real_conflict(tmp_path):
    ir = ir_fixture(tmp_path); ir["nodes"].pop()
    registry = resolve(registry_from_ir(ir), tmp_path)
    assert registry["conflicts"] == [{"type": "EVIDENCE_COVERAGE_MISMATCH", "object_ids": [], "missing": ["e2"], "extra": []}]


def test_ir_owner_validation_has_narrow_scope_and_rejects_duplicates(tmp_path):
    ir = ir_fixture(tmp_path)
    good = validate_ir_owners(ir)
    assert good['status'] == 'PASS'
    assert good['scope'] == 'evidence_ownership_only' and not good['pixel_masks_verified']
    ir['evidence'].append({'id': 'e1'})
    assert validate_ir_owners(ir)['status'] == 'FAIL'
    ir = ir_fixture(tmp_path)
    ir['nodes'][0]['evidence_ids'].append('e2')
    assert validate_ir_owners(ir)['status'] == 'FAIL'
