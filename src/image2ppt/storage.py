"""Content verification, atomic metadata and an expendable SQLite index."""
import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from .contracts import FileRecord


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(file):
    with Path(file).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(file):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate JSON field: " + k)
            result[k] = v
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(Path(file).read_text(encoding="utf-8-sig"), object_pairs_hook=pairs, parse_constant=invalid)


def write_json(file, value, *, exclusive=False):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with file.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = file.with_name(file.name + "." + uuid4().hex + ".tmp")
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, file)


def local_file(base, relative):
    from pathlib import PureWindowsPath
    if not isinstance(relative, str) or not relative or ":" in relative or Path(relative).is_absolute() or PureWindowsPath(relative).is_absolute():
        raise ValueError("expected confined relative path")
    root = Path(base).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    if path == root or not path.is_file():
        raise ValueError("expected local file")
    return path


def inventory(base, paths):
    return {str(p): {"sha256": digest(local_file(base, str(p))), "bytes": local_file(base, str(p)).stat().st_size} for p in paths}


def verify_files(base, files):
    if not files:
        raise ValueError("empty file manifest")
    for name, value in files.items():
        record = value if isinstance(value, FileRecord) else FileRecord.model_validate(value)
        file = local_file(base, name)
        if file.stat().st_size != record.bytes or digest(file) != record.sha256:
            raise ValueError("artifact hash mismatch: " + name)


@contextmanager
def run_lock(directory):
    """OS-owned lock is released even when the process crashes."""
    file = Path(directory) / ".lock"
    with file.open("a+b") as stream:
        if file.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def index_run(run, state, *, database=None):
    run = Path(run)
    with closing(sqlite3.connect(database or run.parent / "index.sqlite3", timeout=0.1)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, path TEXT, status TEXT, updated TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS revisions(run_id TEXT, revision TEXT, build TEXT, hash TEXT, PRIMARY KEY(run_id,revision))")
        db.execute("CREATE TABLE IF NOT EXISTS stages(run_id TEXT, name TEXT, key TEXT, data TEXT, PRIMARY KEY(run_id,name))")
        db.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?,?)", (run.name, str(run), state['status'], now()))
        for name, stage in state.get("stages", {}).items():
            db.execute("INSERT OR REPLACE INTO stages VALUES(?,?,?,?)", (run.name, name, stage['key'], json.dumps(stage)))
        for build in sorted((run / "revisions").glob("r*/build.json")):
            db.execute("INSERT OR REPLACE INTO revisions VALUES(?,?,?,?)", (run.name, build.parent.name, str(build), digest(build)))


def update_index(run, state):
    """Derived-index failure must not change primary state or mask its error."""
    import logging
    import time
    for attempt in range(2):
        try:
            index_run(run, state)
            return
        except (sqlite3.Error, OSError) as error:
            code = getattr(error, 'sqlite_errorcode', None)
            busy = code is not None and (code & 255) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
            if busy and attempt == 0:
                time.sleep(0.05)
                continue
            diagnostic = {'at':now(), 'status':'DEGRADED', 'type':type(error).__name__,
                          'message':str(error), 'sqlite_code':code, 'attempts':attempt+1,
                          'action':'Original index preserved; rebuild explicitly into a new file.'}
            try:
                write_json(Path(run)/'index-warning.json', diagnostic)
            except OSError as warning_error:
                logging.getLogger(__name__).warning('Could not persist index warning: %s', warning_error)
            logging.getLogger(__name__).warning('Derived SQLite index unavailable: %s', error)
            return


def rebuild_index(directory, output):
    """Explicit rebuild to a NEW database; original damaged/locked file is untouched."""
    directory, output = Path(directory).resolve(strict=True), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch(exist_ok=False)
    from .contracts import RunState
    count = 0
    for file in sorted(directory.glob('run-*/run.json')):
        state = RunState.model_validate(read_json(file)).model_dump(exclude_none=True)
        index_run(file.parent, state, database=output)
        count += 1
    if count == 0:
        raise ValueError('no run records found; new empty database retained for diagnosis')
    return {'database':str(output), 'runs_indexed':count, 'original_preserved':True}
