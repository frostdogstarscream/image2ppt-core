import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path


def load_candidates():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "region_candidates_fixture", root / "scripts/region_candidates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(value).hexdigest()


def candidate():
    return {
        "candidate_id": "p5-test",
        "region_id": "P5-example",
        "page": "P5",
        "decision": "hybrid",
        "state": "ACCEPTED",
        "baseline": {
            "label": "baseline",
            "artifacts": [{"label": "base", "path": "base.txt", "sha256": digest(b"base")}],
            "metrics": {"page_mean_pixel_difference": 12.0, "region_mean_pixel_difference": None},
        },
        "candidate": {
            "label": "candidate",
            "target_fidelity": "pixel",
            "edit_goals": ["section_title"],
            "artifacts": [{"label": "candidate", "path": "candidate.txt", "sha256": digest(b"candidate")}],
            "metrics": {"page_mean_pixel_difference": 11.0, "edit_target_coverage": 1.0},
        },
        "coverage": {"section_title": {"status": "covered", "evidence": "native title"}},
        "budget": {"nodes": 10, "text_nodes": 4, "images": 1, "shapes": 5, "limits": {}},
        "history": [
            {"state": "PROPOSED", "basis": "test"},
            {"state": "VALIDATED_STRUCTURE", "basis": "test"},
            {"state": "TRIAL_BUILT", "basis": "test"},
            {"state": "REVIEW_REQUIRED", "basis": "test"},
            {"state": "ACCEPTED", "basis": "test"},
        ],
        "outcome": {"status": "ACCEPTED", "human_acceptance": True, "reason": "human accepted"},
    }


def records(value):
    return {"schema_version": "region-candidates/0.1", "candidates": [value]}


def test_candidate_state_chain_and_report(tmp_path):
    module = load_candidates()
    value = records(candidate())
    assert module.validate(value) == []
    text = module.report(value)
    assert "p5-test" in text and "ACCEPTED" in text
    assert "12.0" in text and "11.0" in text


def test_rejects_illegal_transition_and_terminal_without_acceptance():
    module = load_candidates()
    value = candidate()
    value["history"] = [
        {"state": "PROPOSED", "basis": "test"},
        {"state": "TRIAL_BUILT", "basis": "test"},
    ]
    value["state"] = "TRIAL_BUILT"
    value["outcome"] = {"status": "TRIAL_BUILT", "human_acceptance": False, "reason": ""}
    errors = module.validate(records(value))
    assert any("illegal transition" in error for error in errors)

    value = candidate()
    value["outcome"]["human_acceptance"] = False
    errors = module.validate(records(value))
    assert any("human_acceptance=true" in error for error in errors)


def test_rejects_unsafe_artifact_and_bad_hash():
    module = load_candidates()
    value = candidate()
    value["candidate"]["artifacts"][0]["path"] = "../outside.txt"
    errors = module.validate(records(value))
    assert any("repo-relative" in error for error in errors)

    value = candidate()
    value["baseline"]["artifacts"][0]["sha256"] = "not-a-hash"
    errors = module.validate(records(value))
    assert any("sha256 invalid" in error for error in errors)


def test_verify_files_detects_match_and_tamper(tmp_path):
    module = load_candidates()
    (tmp_path / "base.txt").write_bytes(b"base")
    (tmp_path / "candidate.txt").write_bytes(b"candidate")
    value = records(candidate())
    result = module.verify_files(value, tmp_path)
    assert result["ok"] is True
    assert all(row["sha256_match"] for row in result["rows"])

    (tmp_path / "candidate.txt").write_bytes(b"tampered")
    result = module.verify_files(value, tmp_path)
    assert result["ok"] is False
    assert any(not row["sha256_match"] for row in result["rows"])
