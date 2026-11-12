"""Focused protocol and offline tests; no external model is contacted."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from model_slideir_loop import ROOT, digest, patch, run, seal
from model_text_layout import calibrate

SCRATCH = ROOT / ".tmp-test"
SCRATCH.mkdir(exist_ok=True)


class ModelLoopTest(unittest.TestCase):
    def test_patch_is_local_and_keeps_input(self):
        original = {"nodes": [{"id": "t", "type": "text", "bbox": [1, 2, 30, 10],
                               "text": "real source"}], "review_notes": []}
        changed = patch(original, {"patches": [{"id": "t", "changes": {"bbox": [2, 2, 30, 10]}}]})
        self.assertEqual(original["nodes"][0]["bbox"][0], 1)
        self.assertEqual(changed["nodes"][0]["bbox"][0], 2)
        with self.assertRaises(ValueError):
            patch(original, {"patches": [{"id": "t", "changes": {"text": "invented"}}]})

    def test_font_patch_preserves_other_font_fields(self):
        original = {"nodes": [{"id": "t", "type": "text", "text": "source",
                               "font": {"latin": "Arial", "ea": "Microsoft YaHei",
                                        "size_pt": 20}}], "review_notes": []}
        changed = patch(original, {"patches": [{"id": "t", "changes":
                          {"font": {"size_pt": 18}}}]})
        self.assertEqual(changed["nodes"][0]["font"],
                         {"latin": "Arial", "ea": "Microsoft YaHei", "size_pt": 18})
        self.assertEqual(original["nodes"][0]["font"]["size_pt"], 20)
        with self.assertRaisesRegex(ValueError, "invalid local font patch"):
            patch(original, {"patches": [{"id": "t", "changes":
                         {"font": {"text": "invented"}}}]})

    def test_source_bound_split_keeps_exact_text_and_unique_evidence(self):
        font_file = Path("C:/Windows/Fonts/msyhbd.ttc")
        if not font_file.exists():
            self.skipTest("font matcher needs the installed Microsoft YaHei font")
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            root = Path(tmp)
            (root / "assets").mkdir()
            plate = root / "assets/plate.png"
            source = root / "page.png"
            Image.new("RGB", (400, 100), "white").save(plate)
            image = Image.new("RGB", (400, 100), "white")
            draw = ImageDraw.Draw(image)
            draw.text((30, 25), "02", font=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 35),
                      fill="#004080")
            draw.text((115, 25), "标题", font=ImageFont.truetype(str(font_file), 35),
                      fill="#004080")
            image.save(source)
            ir = {"source_sha256": digest(source), "canvas": {"source_px": [400, 100],
                  "target_px": [400, 100]}, "background": {"asset_id": "plate"},
                  "assets": [{"id": "plate", "purpose": "cleaned_background",
                              "path": "assets/plate.png", "sha256": digest(plate)}],
                  "evidence": [{"id": "e", "kind": "text", "text": "02标题",
                                "bbox": [30, 20, 250, 50]}], "groups": [],
                  "nodes": [{"id": "merged", "type": "text", "text": "02标题",
                             "bbox": [30, 20, 250, 50], "z": 1,
                             "evidence_ids": ["e"],
                             "font": {"latin": "Arial", "ea": "Microsoft YaHei",
                                      "size_pt": 40}}], "review_notes": []}
            calibrated, report = calibrate(ir, source, root)
            self.assertEqual(len(report["splits"]), 1)
            self.assertEqual([n["text"] for n in calibrated["nodes"]], ["02", "标题"])
            self.assertEqual([e["text"] for e in calibrated["evidence"]], ["02", "标题"])
            self.assertEqual({e for n in calibrated["nodes"] for e in n["evidence_ids"]},
                             {e["id"] for e in calibrated["evidence"]})
            self.assertEqual(ir["nodes"][0]["text"], "02标题")

    def test_seal_rejects_model_provenance_change(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            page = Path(tmp) / "page.png"
            Image.new("RGB", (64, 36), "white").save(page)
            ir = {"schema_version": "0.1", "canvas": {"source_px": [64, 36]},
                  "assets": [], "review_notes": [],
                  "source_sha256": "0" * 64}
            with self.assertRaises(ValueError):
                seal(ir, page, {})
            del ir["source_sha256"]
            self.assertEqual(seal(ir, page, {})["source_sha256"], digest(page))

    def test_seal_normalizes_only_protocol_format(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            page = Path(tmp) / "page.png"
            Image.new("RGB", (64, 36), "white").save(page)
            ir = {"schema_version": "SlideIR 0.1", "canvas": {"source_px": [64, 36]},
                  "background": {"color": "#FFFFFF"}, "assets": [],
                  "nodes": [{"id": "t", "text": "observed", "color": "#112233"}],
                  "review_notes": "source uncertain"}
            result = seal(ir, page, {})
            self.assertEqual(result["schema_version"], "0.1")
            self.assertEqual(result["background"]["color"], "FFFFFF")
            self.assertEqual(result["nodes"][0]["text"], "observed")
            self.assertEqual(result["review_notes"], ["source uncertain"])
            self.assertEqual(ir["background"]["color"], "#FFFFFF")

    def test_offline_real_input_never_calls_model(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            root = Path(tmp)
            page = root / "page.png"
            Image.new("RGB", (96, 54), "white").save(page)
            ocr = root / "ocr.json"
            ocr.write_text(json.dumps([{"text": "observed", "bbox": [1, 1, 20, 10]}]))
            args = argparse.Namespace(source=str(page), ocr=str(ocr), assets_dir=None,
                                      output=str(root / "run"), base_url="", model="",
                                      api_key_env="NO_SUCH_KEY", timeout=1, live=False)
            result = run(args, model_call=lambda *a: self.fail("model called offline"))
            self.assertEqual(result["status"], "OFFLINE_READY")
            self.assertEqual(json.loads((root / "run/input-manifest.json").read_text())["ocr"][0]["text"], "observed")

    def test_ocr_mismatch_refused_before_output(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            root = Path(tmp)
            page = root / "page.png"
            Image.new("RGB", (96, 54), "white").save(page)
            ocr = root / "ocr.json"
            ocr.write_text(json.dumps({"source_sha256": "0" * 64, "image_size": [96, 54]}))
            args = argparse.Namespace(source=str(page), ocr=str(ocr), assets_dir=None,
                                      output=str(root / "run"), base_url="", model="",
                                      api_key_env="NO_SUCH_KEY", timeout=1, live=False)
            with self.assertRaisesRegex(ValueError, "OCR source hash"):
                run(args)
            self.assertFalse((root / "run").exists())

    def test_unknown_model_response_is_not_retried(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            root = Path(tmp)
            page = root / "page.png"
            Image.new("RGB", (96, 54), "white").save(page)
            calls = []

            def unknown(*args):
                calls.append(1)
                raise TimeoutError("uncertain paid response")

            args = argparse.Namespace(source=str(page), ocr=None, assets_dir=None,
                                      output=str(root / "run"), base_url="https://fixture.invalid/v1",
                                      model="fixture", api_key_env="FIXTURE_KEY",
                                      timeout=1, live=True)
            from unittest.mock import patch as mock_patch
            with mock_patch.dict("os.environ", {"FIXTURE_KEY": "fixture-only"}):
                result = run(args, model_call=unknown)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result["status"], "MODEL_RESPONSE_UNKNOWN")
            self.assertEqual(json.loads((root / "run/turn-0/reservation.json").read_text())["status"],
                             "UNKNOWN_UNTIL_RECORDED")

    def test_synthetic_fixture_compiles_through_model_boundary(self):
        fixture = ROOT / "examples/synthetic"
        original = json.loads((fixture / "slideir.json").read_text(encoding="utf-8"))
        original.pop("source_sha256")  # Runner seals the normalized image itself.
        replies = iter([original, {"patches": []}])

        def fake_model(*_):
            return {"answer": next(replies), "model": "synthetic-fixture", "usage": {}}

        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            args = argparse.Namespace(source=str(fixture / "source/page.png"), ocr=None,
                                      assets_dir=str(fixture / "assets"),
                                      output=str(Path(tmp) / "run"), base_url="https://fixture.invalid/v1",
                                      model="synthetic-fixture", api_key_env="FIXTURE_KEY",
                                      timeout=1, live=True)
            from unittest.mock import patch as mock_patch
            with mock_patch.dict("os.environ", {"FIXTURE_KEY": "fixture-only"}):
                result = run(args, model_call=fake_model)
            self.assertEqual(result["status"], "REVIEW_REQUIRED", result["attempts"])
            self.assertEqual(result["model_calls"], 2)
            self.assertTrue(Path(result["best_build"]["pptx"]).is_file())
            self.assertTrue(Path(result["best_build"]["preview"]).is_file())


if __name__ == "__main__":
    unittest.main()
