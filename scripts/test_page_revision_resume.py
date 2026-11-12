"""Offline checks for response reuse and rate-limit behavior."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from moma_page_analysis import retry_after_seconds, run as model_call
from model_slideir_loop import ROOT
from run_moma_page_revision import _answer

SCRATCH = ROOT / ".tmp-test"


class PageRevisionResumeTest(unittest.TestCase):
    def test_supplied_answer_is_reused_without_call(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            answer = Path(tmp) / "answer.json"
            answer.write_text("{}", encoding="utf-8")
            with patch("run_moma_page_revision._phase") as call:
                found, result = _answer("style", source=answer, ir=answer,
                                        phase_dir=Path(tmp) / "style", key=None,
                                        supplied=answer)
            self.assertEqual(found, answer)
            self.assertEqual(result["status"], "REUSED")
            call.assert_not_called()

    def test_recent_429_defers_without_call(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            attempt = Path(tmp) / "attempt-01"
            attempt.mkdir()
            (attempt / "reservation.json").write_text("{}", encoding="utf-8")
            (attempt / "failure.json").write_text(
                json.dumps({"status": "HTTP_ERROR", "http_status": 429,
                            "retry_after_seconds": 180}), encoding="utf-8")
            with patch("run_moma_page_revision._phase") as call:
                with self.assertRaisesRegex(RuntimeError, "rate limited"):
                    _answer("assets", source=attempt, ir=attempt,
                            phase_dir=Path(tmp), key="ephemeral")
            call.assert_not_called()

    def test_unknown_response_is_never_retried(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            attempt = Path(tmp) / "attempt-01"
            attempt.mkdir()
            (attempt / "reservation.json").write_text("{}", encoding="utf-8")
            (attempt / "failure.json").write_text(
                '{"status":"RESPONSE_UNKNOWN"}', encoding="utf-8")
            with patch("run_moma_page_revision._phase") as call:
                with self.assertRaisesRegex(RuntimeError, "unknown"):
                    _answer("assets", source=attempt, ir=attempt,
                            phase_dir=Path(tmp), key="ephemeral")
            call.assert_not_called()

    def test_retry_after_parses_delta_and_date(self):
        self.assertEqual(retry_after_seconds("120"), 120)
        self.assertEqual(retry_after_seconds("invalid"), None)
        self.assertIsNotNone(retry_after_seconds("Wed, 21 Oct 2037 07:28:00 GMT"))

    def test_429_record_is_definitive_and_retains_retry_after(self):
        fixture = ROOT / "examples/synthetic"
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            output = Path(tmp) / "call"
            response = SimpleNamespace(status_code=429,
                                       headers={"Content-Type": "application/json",
                                                "Retry-After": "120"},
                                       json=lambda: {"error": {"code": "rate_limit"}})
            args = SimpleNamespace(phase="style", source=fixture / "source/page.png",
                                   ir=fixture / "slideir.json", output=output,
                                   candidates=None, plate=None, preview=None,
                                   max_tokens=100, timeout=2)
            with patch("moma_page_analysis.requests.post", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                    model_call(args, api_key="fixture-only")
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            reservation = json.loads((output / "reservation.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["retry_after_seconds"], 120)
            self.assertEqual(reservation["status"], "HTTP_ERROR_RECORDED")


if __name__ == "__main__":
    unittest.main()
