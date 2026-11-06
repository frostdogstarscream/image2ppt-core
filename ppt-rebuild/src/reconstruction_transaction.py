"""Immutable reconstruction snapshots and a single atomic publication pointer.

The renderer writes outside the frozen scene directories. Only a fully verified
scene can become active; rollback refers to the complete accepted parent scene.
Caller holds RunLock for the whole run, including compile/render and publication.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class RunLock:
    """OS-released exclusive lock; a crashed process cannot leave a stale lock."""

    def __init__(self, directory):
        self.path = Path(directory) / ".writer.lock"
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise RuntimeError("Another process is writing this reconstruction run") from error
        self.stream = stream
        return self

    def __exit__(self, *_):
        if self.stream is not None:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


def scene_fingerprint(ir_file):
    """Bind all IR fields, referenced source/asset bytes and authoring metadata."""
    ir_file = Path(ir_file).resolve()
    ir = _read(ir_file)
    paths = {ir_file.name: ir_file}
    references = [a["path"] for a in ir.get("assets", [])]
    if ir.get("source_image"):
        references.append(ir["source_image"])
    if (ir_file.parent / "authoring.json").exists():
        references.append("authoring.json")
    for relative in references:
        ref = Path(relative)
        resolved = (ir_file.parent / ref).resolve()
        if ref.is_absolute() or not resolved.is_relative_to(ir_file.parent):
            raise ValueError("Scene reference escapes immutable scene directory")
        if not resolved.is_file():
            raise ValueError("Missing scene source or asset: " + str(relative))
        paths[ref.as_posix()] = resolved
    files = {name: _hash(path) for name, path in sorted(paths.items())}
    for asset in ir.get("assets", []):
        if asset.get("sha256") and files[Path(asset["path"]).as_posix()] != asset["sha256"]:
            raise ValueError("Scene asset hash mismatch: " + asset["id"])
    return {"schema": "SceneFingerprint/1", "ir_name": ir_file.name,
            "files": files, "sha256": _digest(files)}


def _snapshot(ir_file, destination):
    ir_file, destination = Path(ir_file).resolve(), Path(destination)
    before = scene_fingerprint(ir_file)
    destination.mkdir(parents=True, exist_ok=False)
    for relative in before["files"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ir_file.parent / relative, target)
    target_ir = destination / ir_file.name
    if scene_fingerprint(ir_file) != before or scene_fingerprint(target_ir) != before:
        raise ValueError("Scene changed during immutable snapshot")
    return target_ir, before


def _status(value):
    return value.get("status") if isinstance(value, dict) else value


class ReconstructionTransaction:
    """Prepare -> verify -> publish, with immutable accepted-parent rollback.

    ``directory`` is the page's transaction root; its active.json points to the
    latest accepted scene. ``input_fingerprint`` binds config/reviews/environment.
    ``owner_validator`` must raise on invalid ownership or return explicit PASS.
    """

    def __init__(self, directory, tx_id, baseline_ir, input_fingerprint,
                 required_gates, object_ids=()):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", tx_id) or tx_id in {".", ".."}:
            raise ValueError("Invalid transaction identifier")
        if not required_gates or len(set(required_gates)) != len(required_gates):
            raise ValueError("Required gates must be nonempty and unique")
        self.root = Path(directory).resolve()
        self.directory = self.root / "transactions" / tx_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tx_id = tx_id
        self.baseline_ir = Path(baseline_ir).resolve()
        self.input_fingerprint = input_fingerprint
        self.required_gates = sorted(required_gates)
        self.object_ids = list(object_ids)
        self.prepare_file = self.directory / "prepare.json"
        self.active_file = self.root / "active.json"

    @property
    def candidate_ir(self):
        return Path(_read(self.prepare_file)["candidate_ir"])

    def _record(self):
        record = _read(self.prepare_file)
        expected = {"tx_id": self.tx_id, "input_fingerprint": self.input_fingerprint,
                    "required_gates": self.required_gates, "object_ids": self.object_ids}
        if any(record[key] != value for key, value in expected.items()):
            raise ValueError("Resume transaction identity or inputs changed")
        for field in ("baseline", "candidate"):
            if scene_fingerprint(record[field + "_ir"]) != record[field + "_fingerprint"]:
                raise ValueError("Immutable " + field + " snapshot changed")
        return record

    def _parent_valid(self, record, current_parent_ir=None):
        parent = current_parent_ir or self.baseline_ir
        if scene_fingerprint(parent) != record["baseline_fingerprint"]:
            raise ValueError("Parent scene changed since prepare")
        active = _read(self.active_file) if self.active_file.exists() else None
        if active and active.get("tx_id") == self.tx_id:
            if active.get("scene_sha256") != record["candidate_fingerprint"]["sha256"]:
                raise ValueError("Published transaction fingerprint mismatch")
            return
        if _digest(active) != record["parent_pointer_sha256"]:
            raise ValueError("Active parent changed since prepare")

    def prepare(self, candidate_ir):
        if self.prepare_file.exists():
            record = self._record()
            self._parent_valid(record)
            if scene_fingerprint(candidate_ir) != record["candidate_fingerprint"]:
                raise ValueError("Existing transaction candidate changed; use a new transaction ID")
            return Path(record["candidate_ir"])
        active = _read(self.active_file) if self.active_file.exists() else None
        baseline = scene_fingerprint(self.baseline_ir)
        if active and active.get("scene_sha256") != baseline["sha256"]:
            raise ValueError("Baseline is not the currently accepted scene")
        # Incomplete attempt directories are preserved for diagnostics. A retry
        # gets a fresh directory and only prepare.json names a complete attempt.
        attempt = self.directory / ("attempt-" + uuid.uuid4().hex)
        base_ir, base_fp = _snapshot(self.baseline_ir, attempt / "baseline")
        new_ir, new_fp = _snapshot(candidate_ir, attempt / "candidate")
        record = {"schema": "ReconstructionPrepare/1", "state": "STAGED",
                  "tx_id": self.tx_id, "input_fingerprint": self.input_fingerprint,
                  "object_ids": self.object_ids, "required_gates": self.required_gates,
                  "original_parent_ir": str(self.baseline_ir),
                  "baseline_ir": str(base_ir), "baseline_fingerprint": base_fp,
                  "candidate_ir": str(new_ir), "candidate_fingerprint": new_fp,
                  "parent_pointer_sha256": _digest(active)}
        _atomic_write(self.prepare_file, record)
        return new_ir

    def verify(self, gates, owner_validator, current_parent_ir=None):
        record = self._record()
        if (self.directory / "rollback.json").exists():
            raise ValueError("Rolled-back transaction is terminal; use a new transaction ID")
        self._parent_valid(record, current_parent_ir)
        missing = [name for name in self.required_gates if _status(gates.get(name)) != "PASS"]
        try:
            ownership = owner_validator(_read(record["candidate_ir"]))
        except Exception as error:
            ownership = {"status": "FAIL", "error_type": type(error).__name__, "reason": str(error)}
        if _status(ownership) != "PASS":
            missing.append("ownership")
        # A validator is not allowed to mutate the scene under evaluation.
        self._record()
        receipt = {"schema": "ReconstructionVerification/1", "tx_id": self.tx_id,
                   "state": "VERIFIED" if not missing else "NEEDS_REVIEW",
                   "gates": gates, "ownership": ownership, "failed_required": missing,
                   "candidate_sha256": record["candidate_fingerprint"]["sha256"],
                   "parent_sha256": record["baseline_fingerprint"]["sha256"]}
        _atomic_write(self.directory / "acceptance.json", receipt)
        return receipt

    def commit(self):
        record = self._record()
        if (self.directory / "rollback.json").exists():
            raise ValueError("Rolled-back transaction cannot commit")
        accepted = _read(self.directory / "acceptance.json")
        if (accepted.get("state") != "VERIFIED" or accepted.get("failed_required")
                or any(_status(accepted.get("gates", {}).get(key)) != "PASS" for key in self.required_gates)
                or _status(accepted.get("ownership")) != "PASS"
                or accepted.get("candidate_sha256") != record["candidate_fingerprint"]["sha256"]
                or accepted.get("parent_sha256") != record["baseline_fingerprint"]["sha256"]):
            raise ValueError("All required gates and ownership must pass before publication")
        self._parent_valid(record)
        receipt = {"schema": "ReconstructionCommit/1", "state": "COMMITTED",
                   "tx_id": self.tx_id, "ir": record["candidate_ir"],
                   "scene_sha256": record["candidate_fingerprint"]["sha256"],
                   "parent_sha256": record["baseline_fingerprint"]["sha256"],
                   "input_fingerprint": self.input_fingerprint,
                   "acceptance_sha256": _hash(self.directory / "acceptance.json")}
        active = _read(self.active_file) if self.active_file.exists() else None
        if active and active.get("tx_id") == self.tx_id:
            if active != receipt:
                raise ValueError("Published receipt changed")
        else:
            _atomic_write(self.active_file, receipt)
        # active.json is authoritative. If interrupted here, commit() recovers
        # this audit receipt without publishing the scene a second time.
        _atomic_write(self.directory / "commit.json", receipt)
        return receipt

    def rollback(self, reason):
        record = self._record()
        self._parent_valid(record)
        active = _read(self.active_file) if self.active_file.exists() else None
        if active and active.get("tx_id") == self.tx_id:
            raise ValueError("Committed transaction is immutable; rollback requires a new transaction")
        path = self.directory / "rollback.json"
        if path.exists():
            return _read(path)
        receipt = {"schema": "ReconstructionRollback/1", "state": "ROLLED_BACK",
                   "tx_id": self.tx_id, "reason": reason,
                   "ir": record["baseline_ir"],
                   "scene_sha256": record["baseline_fingerprint"]["sha256"],
                   "editable_degradation": False, "restored": "complete_accepted_parent_scene"}
        _atomic_write(path, receipt)
        return receipt
