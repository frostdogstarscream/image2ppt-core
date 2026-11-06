"""Publication safety, exact rollback and restart behavior without a renderer."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from _helpers import SRC
import reconstruction_transaction as transactions
from reconstruction_transaction import ReconstructionTransaction, RunLock, scene_fingerprint


def scene(directory, color="112233"):
    directory.mkdir()
    (directory / "assets").mkdir()
    (directory / "assets" / "icon.png").write_bytes(b"transparent-icon-without-text")
    (directory / "source.png").write_bytes(b"unchanged-original-source")
    asset_hash = hashlib.sha256((directory / "assets" / "icon.png").read_bytes()).hexdigest()
    data = {"schema": "SlideIR/0.1", "source_image": "source.png",
            "assets": [{"id": "icon", "path": "assets/icon.png", "sha256": asset_hash}],
            "nodes": [{"id": "text", "type": "text", "text": "项目27项", "color": color},
                      {"id": "card", "type": "shape", "geometry": "roundRect"},
                      {"id": "icon", "type": "image", "asset_id": "icon"}],
            "groups": [{"id": "card-group", "member_ids": ["text", "card", "icon"]}]}
    (directory / "slideir.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (directory / "authoring.json").write_text('{"ir_hash":"fixture"}', encoding="utf-8")
    return directory / "slideir.json"


def setup(tmp_path):
    baseline = scene(tmp_path / "r6")
    candidate = scene(tmp_path / "proposed", "445566")
    root = tmp_path / "run"
    tx = ReconstructionTransaction(root, "card-1", baseline, {"source": "frozen"},
                                   ["render", "residue"], ["text"])
    return tx, baseline, candidate, root


def passed(_):
    return {"status": "PASS"}


def verify(tx):
    return tx.verify({"render": "PASS", "residue": {"status": "PASS"}}, passed)


def test_success_keeps_entire_baseline_and_publishes_only_verified_scene(tmp_path):
    tx, base, candidate, root = setup(tmp_path)
    initial = scene_fingerprint(base)
    with RunLock(root):
        frozen = tx.prepare(candidate)
        assert frozen != candidate and scene_fingerprint(frozen) == scene_fingerprint(candidate)
        assert not (root / "active.json").exists()
        assert verify(tx)["state"] == "VERIFIED"
        commit = tx.commit()
        assert json.loads((root / "active.json").read_text()) == commit
    assert initial == scene_fingerprint(base)
    assert len(json.loads(Path(commit["ir"]).read_text(encoding="utf-8"))["nodes"]) == 3


@pytest.mark.parametrize("value", [None, "PENDING", "UNKNOWN", "NEEDS_REVIEW", "FAIL"])
def test_missing_or_unknown_required_gate_cannot_publish(tmp_path, value):
    tx, _, candidate, root = setup(tmp_path)
    tx.prepare(candidate)
    result = tx.verify({"render": "PASS", "residue": value}, passed)
    assert result["state"] == "NEEDS_REVIEW"
    with pytest.raises(ValueError, match="required gates"):
        tx.commit()
    assert not (root / "active.json").exists()


def test_owner_conflict_fails_before_publication(tmp_path):
    tx, _, candidate, root = setup(tmp_path)
    tx.prepare(candidate)
    def conflict(_):
        raise ValueError("Duplicate visible contribution owner")
    receipt = tx.verify({"render": "PASS", "residue": "PASS"}, conflict)
    assert receipt["failed_required"] == ["ownership"]
    with pytest.raises(ValueError):
        tx.commit()
    assert not (root / "active.json").exists()


def test_validator_without_explicit_pass_is_not_accepted(tmp_path):
    tx, _, candidate, _ = setup(tmp_path)
    tx.prepare(candidate)
    assert tx.verify({"render": "PASS", "residue": "PASS"}, lambda _: None)["state"] == "NEEDS_REVIEW"


def test_changed_parent_after_verify_rejects_commit(tmp_path):
    tx, base, candidate, root = setup(tmp_path)
    tx.prepare(candidate)
    verify(tx)
    data = json.loads(base.read_text(encoding="utf-8"))
    data["nodes"][0]["text"] = "unapproved parent change"
    base.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Parent scene changed"):
        tx.commit()
    assert not (root / "active.json").exists()


def test_modified_candidate_after_verify_rejects_commit(tmp_path):
    tx, _, candidate, _ = setup(tmp_path)
    frozen = tx.prepare(candidate)
    verify(tx)
    (frozen.parent / "source.png").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="snapshot changed"):
        tx.commit()


def test_rollback_restores_native_objects_assets_groups_and_never_r3(tmp_path):
    tx, base, candidate, root = setup(tmp_path)
    tx.prepare(candidate)
    tx.verify({"render": "FAIL", "residue": "PASS"}, passed)
    receipt = tx.rollback("native candidate visually failed")
    assert scene_fingerprint(receipt["ir"]) == scene_fingerprint(base)
    restored = json.loads(Path(receipt["ir"]).read_text(encoding="utf-8"))
    assert [n["type"] for n in restored["nodes"]] == ["text", "shape", "image"]
    assert restored["groups"][0]["member_ids"] == ["text", "card", "icon"]
    assert not receipt["editable_degradation"]
    assert not (root / "active.json").exists()
    assert tx.rollback("retry") == receipt
    with pytest.raises(ValueError, match="terminal"):
        verify(tx)


@pytest.mark.parametrize("stage", ["prepare", "verify", "publish"])
def test_restart_after_each_complete_stage_is_idempotent(tmp_path, stage):
    tx, base, candidate, root = setup(tmp_path)
    frozen = tx.prepare(candidate)
    if stage in {"verify", "publish"}:
        verify(tx)
    if stage == "publish":
        first = tx.commit()
    resumed = ReconstructionTransaction(root, "card-1", base, {"source": "frozen"},
                                        ["render", "residue"], ["text"])
    assert resumed.prepare(candidate) == frozen
    if stage == "prepare":
        verify(resumed)
    final = resumed.commit()
    assert resumed.commit() == final
    if stage == "publish":
        assert first == final
    assert len(list((root / "transactions" / "card-1").glob("attempt-*"))) == 1


def test_interruption_after_atomic_pointer_recovers_audit_without_republish(tmp_path, monkeypatch):
    tx, _, candidate, root = setup(tmp_path)
    tx.prepare(candidate)
    verify(tx)
    original = transactions._atomic_write
    def fail_receipt(path, value):
        if Path(path).name == "commit.json":
            raise RuntimeError("injected process interruption")
        original(path, value)
    monkeypatch.setattr(transactions, "_atomic_write", fail_receipt)
    with pytest.raises(RuntimeError):
        tx.commit()
    pointer_bytes = (root / "active.json").read_bytes()
    pointer_time = (root / "active.json").stat().st_mtime_ns
    monkeypatch.setattr(transactions, "_atomic_write", original)
    tx.commit()
    assert (root / "active.json").read_bytes() == pointer_bytes
    assert (root / "active.json").stat().st_mtime_ns == pointer_time
    assert (tx.directory / "commit.json").exists()


def test_prepare_interruption_retries_into_new_attempt(tmp_path, monkeypatch):
    tx, _, candidate, _ = setup(tmp_path)
    original = transactions._atomic_write
    monkeypatch.setattr(transactions, "_atomic_write", lambda *_: (_ for _ in ()).throw(RuntimeError("interrupt")))
    with pytest.raises(RuntimeError):
        tx.prepare(candidate)
    monkeypatch.setattr(transactions, "_atomic_write", original)
    tx.prepare(candidate)
    verify(tx)
    assert tx.commit()["state"] == "COMMITTED"
    assert len(list(tx.directory.glob("attempt-*"))) == 2


def test_run_lock_excludes_other_writer_and_releases(tmp_path):
    with RunLock(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            with RunLock(tmp_path):
                pass
    with RunLock(tmp_path):
        pass


def test_changed_resume_inputs_and_stale_parent_pointer_rejected(tmp_path):
    tx, base, candidate, root = setup(tmp_path)
    tx.prepare(candidate)
    altered = ReconstructionTransaction(root, "card-1", base, {"source": "changed"},
                                        ["render", "residue"], ["text"])
    with pytest.raises(ValueError, match="inputs changed"):
        altered.prepare(candidate)
    (root / "active.json").write_text('{"tx_id":"another-writer"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Active parent changed"):
        verify(tx)


def test_path_traversal_is_not_copied_into_snapshot(tmp_path):
    tx, _, candidate, _ = setup(tmp_path)
    data = json.loads(candidate.read_text(encoding="utf-8"))
    data["source_image"] = "../../outside.png"
    candidate.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        tx.prepare(candidate)
