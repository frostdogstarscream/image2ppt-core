"""Durable, versioned MCP operations and immutable evidence references."""
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
import time

from ..materialize import digest, write
from ..parallel_store import key
from ..resource_lease import lease


class RuntimeConflict(ValueError):
    pass


class RuntimeState:
    def __init__(self, job):
        self.job = Path(job).resolve()
        self.root = self.job/'runtime'
        self.root.mkdir(exist_ok=True)
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS operations (
                    id TEXT PRIMARY KEY, fingerprint TEXT, state TEXT, result TEXT);
                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY, path TEXT, sha TEXT, role TEXT, delivered INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, payload TEXT);
            ''')

    @contextmanager
    def db(self):
        with sqlite3.connect(self.root/'runtime.sqlite', timeout=60) as db:
            yield db

    def read(self):
        file = self.root/'state.json'
        return json.loads(file.read_text('utf8')) if file.exists() else {
            'version': 0, 'pages': [], 'candidates': {}, 'status': 'RUNNING'}

    def remaining_seconds(self):
        session=self.root/'session.json'
        if not session.exists():return None
        deadline=json.loads(session.read_text('utf8')).get('deadline')
        return max(0,round(deadline-time.time())) if deadline is not None else None

    def event(self, payload):
        with self.db() as db:
            db.execute('INSERT INTO events(created,payload) VALUES(?,?)',
                       (time.time(), json.dumps(payload, ensure_ascii=False)))

    def register(self, path, role):
        path = Path(path).resolve()
        if not path.is_relative_to(self.job) or not path.is_file():
            raise ValueError('evidence must be an existing job file')
        sha = digest(path)
        identifier = 'e-'+key({'path': str(path.relative_to(self.job)), 'sha': sha, 'role': role})[:32]
        with self.db() as db:
            db.execute('INSERT OR IGNORE INTO evidence(id,path,sha,role) VALUES(?,?,?,?)',
                       (identifier, str(path.relative_to(self.job)), sha, role))
        return {'id': identifier, 'sha256': sha, 'role': role,'name':path.name}

    def evidence(self, identifier, delivered=False):
        with self.db() as db:
            row = db.execute('SELECT path,sha,role,delivered FROM evidence WHERE id=?', (identifier,)).fetchone()
            if not row:
                raise ValueError('unknown evidence reference')
            path = (self.job/row[0]).resolve()
            if not path.is_relative_to(self.job) or digest(path) != row[1]:
                raise RuntimeConflict('evidence bytes changed')
            if delivered:
                db.execute('UPDATE evidence SET delivered=1 WHERE id=?', (identifier,))
        return path, {'id': identifier, 'sha256': row[1], 'role': row[2],'name':path.name,
                      'provided_to_agent': bool(row[3]) or delivered}

    def operation_directory(self, token):
        legacy = self.root/'operations'/token
        return legacy if legacy.exists() else self.root/'o'/key(token)[:12]

    def mutate(self, operation, arguments, expected_version, token, execute):
        if not re.fullmatch(r'[A-Za-z0-9_-]{8,100}', token):
            raise ValueError('an 8-100 character idempotency key is required')
        fingerprint = key({'operation': operation, 'arguments': arguments, 'parent': expected_version})
        with lease('runtime-job-'+key(str(self.job)), capacity=1):
            with self.db() as db:
                row = db.execute('SELECT fingerprint,state,result FROM operations WHERE id=?', (token,)).fetchone()
            if row:
                if row[0] != fingerprint:
                    raise RuntimeConflict('idempotency key was used for different arguments')
                recovery = self.operation_directory(token)/'completed.json'
                if row[1] == 'COMPLETED' or recovery.exists():
                    result = json.loads(row[2]) if row[1] == 'COMPLETED' else json.loads(recovery.read_text('utf8'))
                    for path, sha in result.get('files', {}).items():
                        if digest(self.job/path) != sha:
                            raise RuntimeConflict('completed operation artifact changed')
                    if row[1] != 'COMPLETED':
                        head = self.read()
                        if head['version'] not in (result['state']['version']-1, result['state']['version']):
                            raise RuntimeConflict('recovery receipt conflicts with newer head')
                        write(self.root/'state.json', result['state'])
                        with self.db() as db:
                            db.execute('UPDATE operations SET state=?,result=? WHERE id=?',
                                       ('COMPLETED', json.dumps(result, ensure_ascii=False), token))
                    return {**result['response'],'remaining_seconds':self.remaining_seconds()}
                raise RuntimeConflict('operation outcome UNKNOWN; inspect receipts before a new action')
            current = self.read()
            if expected_version is not None and current['version'] != expected_version:
                raise RuntimeConflict('scene version conflict; reload job context')
            if (self.job/'cancel.requested').exists():
                raise InterruptedError('job cancelled')
            if current['status'] != 'RUNNING':
                raise RuntimeConflict('job has already finalized')
            directory = self.operation_directory(token)
            directory.mkdir(parents=True, exist_ok=False)
            write(directory/'input.json', {'tool': operation, 'arguments': arguments,
                  'parent_version': current['version'], 'fingerprint': fingerprint})
            with self.db() as db:
                db.execute('INSERT INTO operations VALUES(?,?,?,NULL)', (token, fingerprint, 'RESERVED'))
            self.event({'tool': operation, 'state': 'RUNNING', 'operation_id': token})
            try:
                result = execute(current, directory)
                current['version'] += 1
                result = {**result, 'version': current['version'],'remaining_seconds':self.remaining_seconds()}
                # A recovery receipt is written before publishing the mutable head.
                files = {str(p.relative_to(self.job)): digest(p) for p in directory.rglob('*') if p.is_file()}
                receipt = {'state': current, 'response': result, 'files': files}
                write(directory/'completed.json', receipt)
                write(self.root/'state.json', current)
                with self.db() as db:
                    db.execute('UPDATE operations SET state=?,result=? WHERE id=?',
                               ('COMPLETED', json.dumps(receipt, ensure_ascii=False), token))
                self.event({'tool': operation, 'state': 'COMPLETED', 'version': current['version']})
                return result
            except BaseException as exc:
                with self.db() as db:
                    db.execute('UPDATE operations SET state=? WHERE id=?', ('UNKNOWN', token))
                self.event({'tool': operation, 'state': 'UNKNOWN', 'error_type': type(exc).__name__})
                if isinstance(exc,StopIteration):
                    raise ValueError('object reference is absent from the current scene; inspect current object IDs') from None
                raise
