"""Offline tests: no network, no credentials.

Run with pytest, or directly:  python tests/test_offline.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image  # noqa: E402

from image2ppt_planner.client import ChatReply  # noqa: E402
from image2ppt_planner.config import Settings  # noqa: E402
from image2ppt_planner.planner import compact_inventory, plan_page, prepare_image  # noqa: E402


def _tmpdir() -> Path:
    """Scratch dir under the repository root.

    Neither the OS temp area nor every repo subdirectory is writable under the
    current file sandbox, so the caller can point PLANNER_TEST_TMP elsewhere.
    """
    import os

    base = Path(os.environ.get("PLANNER_TEST_TMP") or Path(__file__).resolve().parents[3] / ".tmp-test")
    base.mkdir(parents=True, exist_ok=True)
    path = base / uuid.uuid4().hex[:12]
    path.mkdir()
    return path


INVENTORY = {
    "schema": "CVElementInventory",
    "version": "1.0",
    "source_sha256": "a" * 64,
    "image_size": [200, 120],
    "texts": [
        {"id": "t001", "text": "总体思路", "bbox": [10, 8, 80, 16]},
        {"id": "t002", "text": "分步推进既有系统升级", "bbox": [10, 40, 150, 12]},
    ],
    "candidates": [
        {"id": "cv-000", "bbox": [0, 0, 200, 30], "geometry_hint": "polygon", "ocr_inside": ["t001"]},
    ],
}

VALID_PLAN = {
    "schema": "ElementPlan/2",
    "source_sha256": "a" * 64,
    "image_size": [200, 120],
    "background_color": "FFFFFF",
    "objects": [
        {"id": "band", "kind": "native_shape", "bbox": [0, 0, 200, 30], "z": 0,
         "geometry": "rect", "fill": "0B5FA5"},
    ],
    "text_decisions": [
        {"id": "t001", "bbox": [10, 8, 80, 16], "text": "总体思路", "color": "FFFFFF",
         "action": "native", "bold": True, "z": 10, "font_size_px": 16,
         "font_family": "微软雅黑", "font_latin": "微软雅黑", "font_confidence": 0.8,
         "font_evidence": "无衬线、等宽笔画，与雅黑一致"},
        {"id": "t002", "bbox": [10, 40, 150, 12], "text": "分步推进既有系统升级", "color": "111111",
         "action": "native", "bold": False, "z": 11, "font_size_px": 12,
         "font_family": "微软雅黑", "font_latin": "微软雅黑", "font_confidence": 0.7,
         "font_evidence": "正文黑体，字重较细"},
    ],
    "limitations": [],
}


class StubClient:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls = 0

    async def chat(self, messages):
        self.calls += 1
        text = self.replies.pop(0) if self.replies else "{}"
        return ChatReply(
            text=text, model="stub", usage={"total_tokens": 1}, finish_reason="stop",
            request_sha256="0" * 64, raw={"choices": [{"message": {"content": text}}]},
        )


def _settings(tmp: Path) -> Settings:
    return Settings(api_key="test-key-not-real", artifacts_root=tmp, max_attempts=3)


def _image(tmp: Path, size=(400, 240)) -> Path:
    path = tmp / "page.png"
    Image.new("RGB", size, "white").save(path)
    return path


def test_compact_inventory_keeps_ids_and_bboxes():
    payload = compact_inventory(INVENTORY, ["微软雅黑"])
    assert [t["id"] for t in payload["texts"]] == ["t001", "t002"]
    assert payload["texts"][0]["bbox"] == [10, 8, 80, 16]
    assert payload["texts"][0]["px"] == round(16 * 0.78, 1)
    assert payload["candidates"][0]["id"] == "cv-000"


def test_prepare_image_returns_data_url():
    tmp = _tmpdir()
    url, note = prepare_image(_image(tmp), _settings(tmp))
    assert url.startswith("data:image/")
    assert note["bytes"] > 0
    assert note["mime"] in {"image/png", "image/jpeg"}


def test_plan_page_accepts_valid_reply():
    tmp = _tmpdir()
    client = StubClient([json.dumps(VALID_PLAN, ensure_ascii=False)])
    result = asyncio.run(plan_page(_settings(tmp), _image(tmp), INVENTORY, out_dir=tmp / "run", client=client))
    assert result.attempts == 1
    assert len(result.plan["text_decisions"]) == 2
    assert client.calls == 1
    assert (tmp / "run" / "plan.json").exists()
    assert (tmp / "run" / "provenance.json").exists()


def test_plan_page_repairs_after_validator_error():
    tmp = _tmpdir()
    broken = json.loads(json.dumps(VALID_PLAN))
    broken["text_decisions"] = broken["text_decisions"][:1]  # drop one OCR id
    client = StubClient([json.dumps(broken, ensure_ascii=False), json.dumps(VALID_PLAN, ensure_ascii=False)])
    result = asyncio.run(plan_page(_settings(tmp), _image(tmp), INVENTORY, out_dir=tmp / "run2", client=client))
    assert result.attempts == 2
    assert client.calls == 2
    assert result.errors and "cover exactly once" in result.errors[0]


def test_plan_page_gives_up_after_budget():
    tmp = _tmpdir()
    client = StubClient(["not json", "still not json", "nope"])
    try:
        asyncio.run(plan_page(_settings(tmp), _image(tmp), INVENTORY, out_dir=tmp / "run3", client=client))
    except RuntimeError as exc:
        assert "未产出可校验的计划" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError after the retry budget")
    assert client.calls == 3


def test_malformed_gradient_is_rejected():
    """The service admission gate rejects wrong gradient keys; the planner must too."""
    tmp = _tmpdir()
    bad = json.loads(json.dumps(VALID_PLAN))
    bad["objects"][0].pop("fill", None)
    bad["objects"][0]["fill_gradient"] = {
        "type": "linear", "angle": 0,
        "stops": [{"pos": 0, "color": "005BAC"}, {"pos": 1, "color": "E0F0FF"}],
    }
    client = StubClient([json.dumps(bad, ensure_ascii=False), json.dumps(VALID_PLAN, ensure_ascii=False)])
    result = asyncio.run(plan_page(_settings(tmp), _image(tmp), INVENTORY, out_dir=tmp / "run4", client=client))
    assert result.attempts == 2
    assert any("element plan fields" in e or "fill_gradient" in e for e in result.errors)


def test_well_formed_gradient_is_accepted():
    tmp = _tmpdir()
    good = json.loads(json.dumps(VALID_PLAN))
    good["objects"][0].pop("fill", None)
    good["objects"][0]["fill_gradient"] = {
        "angle_deg": 90,
        "stops": [{"offset": 0, "color": "005BAC"}, {"offset": 100000, "color": "E0F0FF"}],
    }
    client = StubClient([json.dumps(good, ensure_ascii=False)])
    result = asyncio.run(plan_page(_settings(tmp), _image(tmp), INVENTORY, out_dir=tmp / "run5", client=client))
    assert result.attempts == 1
    assert result.plan["objects"][0]["fill_gradient"]["angle_deg"] == 90


def test_raster_retained_text_is_rejected():
    """A label kept inside a crop is unreadable by an editor, so the planner must refuse it."""
    tmp = _tmpdir()
    bad = json.loads(json.dumps(VALID_PLAN))
    bad["objects"].append({
        "id": "crop-title", "kind": "raster_asset", "bbox": [0, 0, 100, 24], "z": 5,
        "mask": "rect", "contains_text_ids": ["t001"],
    })
    bad["text_decisions"][0]["action"] = "retained_in_asset"
    client = StubClient([json.dumps(bad, ensure_ascii=False), json.dumps(VALID_PLAN, ensure_ascii=False)])
    result = asyncio.run(plan_page(_settings(tmp), _image(tmp), INVENTORY, out_dir=tmp / "run6", client=client))
    assert result.attempts == 2
    assert any("retained_in_asset" in e or "text_retained" in e for e in result.errors)
    assert all(t["action"] == "native" for t in result.plan["text_decisions"])


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
