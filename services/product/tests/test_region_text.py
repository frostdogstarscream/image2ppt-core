from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from product.materialize import digest
from product.recovery.region_artwork import transfer_text_ink
from product.recovery.region_text import measure_group, plan_regions, choose_region_background


def card_fixture():
    source = np.empty((104, 235, 3), dtype="uint8")
    source[:] = (236, 242, 245)
    source[:, :, 1] += np.arange(235, dtype="uint8")[None, :] // 35
    pristine = source.copy()
    texts = []
    for i, (word, origin) in enumerate([("ALPHA", (38, 40)), ("BETA", (38, 75))]):
        cv2.putText(source, word, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (24, 24, 24), 1, cv2.LINE_AA)
        (width, height), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        texts.append(
            SimpleNamespace(
                id="t-" + str(i),
                text=word,
                bbox=[origin[0], origin[1] - height, width, height + 3],
                spans=[],
                style="black",
            )
        )
    graph = SimpleNamespace(
        image_size=[235, 104],
        source_sha256="fixture",
        text=SimpleNamespace(
            texts=texts,
            styles=[SimpleNamespace(id="black", color="181818", size_px=16)],
            members={t.id: [t.id + "-ocr"] for t in texts},
        ),
        asset=SimpleNamespace(objects=[]),
        selected={},
        ownership={t.id: "RESOLVED" for t in texts},
    )
    inventory = {"elements": [{"id": "card", "category": "card_shape", "bbox": [5, 5, 220, 90]}]}
    return source, pristine, graph, inventory


def test_shared_card_has_disjoint_text_claims_and_preserves_other_content(tmp_path):
    source, pristine, graph, inventory = card_fixture()
    source[27:32, 111:140] = [12, 13, 14]
    graph.asset.objects.append(SimpleNamespace(id="icon", kind="icon", bbox=[110, 26, 32, 8]))
    (group,) = plan_regions(graph, graph.text.texts, inventory, source)
    assert group["members"] == ["t-0", "t-1"]
    masks, clean, guard = measure_group(graph, group, source, tmp_path, [])
    first, second = masks.values()
    assert first.any() and second.any() and not (first & second).any()
    assert guard[28, 114]
    np.testing.assert_array_equal(clean[27:32, 111:140], source[27:32, 111:140])
    for text in graph.text.texts:
        x, y, w, h = text.bbox
        error = np.abs(clean[y : y + h, x : x + w].astype(float) - pristine[y : y + h, x : x + w])
        assert np.percentile(error, 99) < 12


def test_failed_member_rejects_group_before_any_publication(tmp_path):
    source, _, graph, inventory = card_fixture()
    text = graph.text.texts[-1]
    graph.asset.objects.append(SimpleNamespace(id="badge", kind="logo", bbox=text.bbox))
    (group,) = plan_regions(graph, graph.text.texts, inventory, source)
    with pytest.raises(ValueError, match="protected neighbor"):
        measure_group(graph, group, source, tmp_path, [])
    assert list(tmp_path.iterdir()) == []


def test_embedded_photo_text_requires_an_observed_caption_band():
    source, _, graph, _ = card_fixture()
    inventory = {"elements": [{"id": "photo", "category": "photo", "bbox": [1, 1, 200, 95]}]}
    assert plan_regions(graph, graph.text.texts, inventory, source) == []


def test_lower_layer_must_match_visible_paint_and_remove_source_glyphs(tmp_path, monkeypatch):
    from product.recovery import layerd_adapter

    source, pristine, graph, inventory = card_fixture()
    (group,) = plan_regions(graph, graph.text.texts, inventory, source)
    masks, fitted, guard = measure_group(graph, group, source, tmp_path, [])
    alternatives = [
        layerd_adapter.LowerBackground(im, {"identifier": i})
        for i, im in enumerate([source, np.zeros_like(source), pristine])
    ]
    monkeypatch.setattr(layerd_adapter, "lower_backgrounds", lambda *a, **kw: alternatives)
    choose_region_background(group, graph, source, masks, fitted, guard, "fixture", None, tmp_path)
    assert group["background"]["method"] == "LayerD"
    assert group["background"]["provenance"] == {"identifier": 2}


def test_text_on_retained_artwork_updates_bound_asset_and_owner(tmp_path):
    bg = np.full((15, 30, 3), 245, dtype=np.uint8)
    art = np.full((9, 20, 4), [25, 83, 134, 255], dtype=np.uint8)
    art[3:6, 6:14, :3] = 245
    src = bg.copy()
    src[3:12, 5:25] = art[:, :, :3]
    Image.fromarray(src).save(tmp_path / "source.png")
    Image.fromarray(art).save(tmp_path / "ink.png")
    prior = np.zeros((15, 30), bool)
    prior[3:12, 5:25] = True
    box = [5, 3, 20, 9]
    np.savez_compressed(tmp_path / "ink.npz", mask=prior, restore_bbox=box, backing_patch=bg[3:12, 5:25])
    src_sha = digest(tmp_path / "source.png")
    record = {
        "retained_card_artwork": True,
        "asset_reference": {
            "path": "ink.png",
            "bbox": box,
            "sha256": digest(tmp_path / "ink.png"),
            "source_sha256": src_sha,
        },
        "owner_evidence": {"file": "ink.npz", "sha256": digest(tmp_path / "ink.npz")},
    }
    mask = np.zeros_like(prior)
    mask[6:9, 11:19] = True
    original = record["asset_reference"]["sha256"]
    repair = np.full((3, 8, 3), [25, 83, 134], dtype=np.uint8)
    transfer_text_ink(tmp_path, record, mask, [11, 6, 8, 3], repair, src_sha)
    assert record["asset_reference"]["sha256"] != original
    updated = np.array(Image.open(tmp_path / "ink.png"))
    assert np.array_equal(updated[3:6, 6:14, :3], repair)
    assert np.array_equal(updated[0], art[0])
    with np.load(tmp_path / "ink.npz", allow_pickle=False) as proof:
        np.testing.assert_array_equal(proof["mask"], prior & ~mask)
    with pytest.raises(ValueError, match="exclusively own"):
        transfer_text_ink(tmp_path, record, mask, [11, 6, 8, 3], repair, src_sha)

def test_group_inpaint_retries_only_fields_that_failed_layer_trial():
    from types import SimpleNamespace
    from product.runtime.batch_backbone import _retry_joint_text

    source = np.zeros((4, 8, 3), dtype=np.uint8)
    masks = {key: np.zeros(source.shape[:2], bool) for key in ('one', 'two')}
    masks['one'][1, 2] = masks['two'][1, 5] = True
    members = [(i, SimpleNamespace(id=key)) for i, key in enumerate(masks)]
    layer = SimpleNamespace(image=source.copy(), provenance={'below_z': 1})
    calls = []

    def repair(image, mask):
        np.testing.assert_array_equal(mask, masks['two'])
        calls.append('inpaint')
        return source.copy()

    def probe(obj, *, folder, inpaint):
        inpaint(source, masks[obj.id])
        provider = inpaint.receipt['provider']
        calls.append((obj.id, provider))
        ok = obj.id == 'one' or provider == 'lama'
        return {'status': 'PRECHECK_PASS' if ok else 'BLOCKED'}

    outcomes = [None, None]
    _retry_joint_text(members, masks, [layer], source, repair, probe, outcomes)
    assert calls == [('one', 'layerd'), ('two', 'layerd'), 'inpaint', ('two', 'lama')]
    assert all(item['status'] == 'PRECHECK_PASS' for item in outcomes)
