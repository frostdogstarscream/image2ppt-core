"""The authoring sidecar gate: a record must describe the IR it ships with."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from image2ppt.contracts import AuthoringRecord
from image2ppt.providers import ManualDraftProvider
from image2ppt.storage import digest, write_json


def ir_body():
    return {"schema_version": "0.1", "source_image": "source/page.png", "nodes": [], "assets": []}


def record(ir_path, **overrides):
    body = {"format_version": "1", "tool": "codex", "model_id": "gpt-x",
            "created_at": "2026-10-14T00:00:00", "ir_sha256": digest(ir_path),
            "node_decisions": {}, "font_candidates": [], "unresolved_ids": [], "review_notes": []}
    body.update(overrides)
    return body


class AuthoringGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.ir = self.dir / "slideir.json"
        self.record_path = self.dir / "authoring.json"
        write_json(self.ir, ir_body())

    def tearDown(self):
        self.tmp.cleanup()

    def test_absent_record_keeps_legacy_runs_replayable(self):
        self.assertIsNone(ManualDraftProvider().load(self.ir).authoring)

    def test_matching_record_is_carried(self):
        write_json(self.record_path, record(self.ir))
        authoring = ManualDraftProvider().load(self.ir).authoring
        self.assertEqual((authoring.tool, authoring.model_id), ("codex", "gpt-x"))

    def test_record_that_no_longer_describes_the_ir_is_rejected(self):
        write_json(self.record_path, record(self.ir))
        write_json(self.ir, {**ir_body(), "review_notes": ["authored"]})
        with self.assertRaisesRegex(ValueError, "does not describe this IR"):
            ManualDraftProvider().load(self.ir)

    def test_rerecording_restores_the_run(self):
        write_json(self.record_path, record(self.ir))
        write_json(self.ir, {**ir_body(), "review_notes": ["authored"]})
        write_json(self.record_path, record(self.ir))
        self.assertEqual(ManualDraftProvider().load(self.ir).authoring.tool, "codex")

    def test_record_rejects_unknown_fields(self):
        with self.assertRaises(Exception):
            AuthoringRecord.model_validate(record(self.ir, surprise=1))

    def test_record_rejects_a_malformed_hash(self):
        with self.assertRaises(Exception):
            AuthoringRecord.model_validate(record(self.ir, ir_sha256="nope"))


if __name__ == "__main__":
    unittest.main()
