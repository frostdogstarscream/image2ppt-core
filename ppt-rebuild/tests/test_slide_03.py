"""P0 tests — page 03, the first sample named by the plan (§5)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from _helpers import ROOT, dump, load_config, page_dir

from layerd_loader import detect_alpha_mode, load_page, recompose, region_candidates, to_record, verify


def _provenance_installed() -> bool:
    """The loader refuses a pack without a receipt: request.json or the baseline catalog.

    Review packs copied by hand from a service job are missing request.json, and
    fabricating the receipt would defeat the check, so those tests skip the same way
    the other private-dataset tests do.
    """
    directory: Path = page_dir("03")
    return (directory / "request.json").exists() or (ROOT / "configs" / "layerd-baseline.json").exists()


requires_provenance = pytest.mark.skipif(
    not _provenance_installed(),
    reason="LayerD provenance receipt not installed (request.json or configs/layerd-baseline.json)",
)


@requires_provenance
def test_m1_page_loads_with_manifest():
    page = load_page(page_dir("03"))
    assert page.size == (1672, 941)
    assert page.layer_order == "back_to_front"
    assert [layer.z for layer in page.layers] == list(range(len(page.layers)))
    assert page.layers[0].role == "background"
    assert sum(1 for layer in page.layers if layer.role == "foreground") >= 1
    record = to_record(page)
    assert record["schema"] == "LayerDParse/1"
    assert len(record["layers"]) == len(page.layers)


@requires_provenance
def test_m1_recomposition_matches_service_composite():
    """If we cannot reproduce composite.png we have misread the layers."""
    cfg = load_config()
    page = load_page(page_dir("03"))
    report = verify(page)
    assert report["composite_available"] is True
    assert report["mae"] <= cfg["layerd"]["recompose_mae_tolerance"], dump(report)
    assert report["psnr"] > 25, dump(report)


@requires_provenance
def test_m1_regions_are_pixel_evidence_not_objects():
    page = load_page(page_dir("03"))
    for layer in page.layers:
        for region in layer.regions:
            x, y, w, h = region["bbox"]
            # every region is in full-page coordinates
            assert 0 <= x < page.size[0] and 0 <= y < page.size[1]
            assert x + w <= page.size[0] and y + h <= page.size[1]
            assert 0 < region["fill_ratio"] <= 1.0
    record = to_record(page)
    assert "region_note" in record["layers"][0]
    assert "语义对象" in record["layers"][0]["region_note"]


def test_m1_alpha_mode_detection():
    straight = np.zeros((8, 8, 4), np.uint8)
    straight[1:7, 1:7] = (200, 40, 40, 40)       # bright colour at low alpha -> straight
    premul = np.zeros((8, 8, 4), np.uint8)
    premul[1:7, 1:7] = (10, 20, 30, 40)          # colour never above alpha -> premultiplied
    opaque = np.zeros((8, 8, 4), np.uint8)
    opaque[..., :3] = 120
    opaque[..., 3] = 255                          # no semi-transparent pixels: default straight
    assert detect_alpha_mode(straight) == "straight"
    assert detect_alpha_mode(premul) == "premultiplied"
    assert detect_alpha_mode(opaque) == "straight"


@requires_provenance
def test_m1_recompose_is_deterministic():
    page = load_page(page_dir("03"))
    first = recompose(page)
    second = recompose(page)
    assert np.array_equal(first, second)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
