"""Durable DAG, exactly-once call reservation and generation-level publication."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from .materialize import digest,write


def key(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


class UnknownCall(RuntimeError):pass
class Conflict(RuntimeError):pass
class BudgetExceeded(RuntimeError):pass


class Store:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        with self.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS tasks(k TEXT PRIMARY KEY, branch TEXT, state TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS dependencies(task TEXT, parent TEXT, PRIMARY KEY(task,parent));
            CREATE TABLE IF NOT EXISTS calls(k TEXT PRIMARY KEY, state TEXT, response TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS evidence(k TEXT PRIMARY KEY, path TEXT, sha TEXT);
            CREATE TABLE IF NOT EXISTS candidates(k TEXT PRIMARY KEY, object_id TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS budgets(object_key TEXT, kind TEXT, fingerprint TEXT, PRIMARY KEY(object_key,kind,fingerprint));
            CREATE TABLE IF NOT EXISTS commits(generation INTEGER PRIMARY KEY, manifest TEXT, sha TEXT);
            CREATE TABLE IF NOT EXISTS head(id INTEGER PRIMARY KEY CHECK(id=1), generation INTEGER NOT NULL);
            INSERT OR IGNORE INTO head VALUES(1,0);
            ''')

    @contextmanager
    def db(self):
        db=sqlite3.connect(self.root/'parallel.sqlite',timeout=60)
        try:
            db.execute('PRAGMA busy_timeout=60000');yield db;db.commit()
        except BaseException:db.rollback();raise
        finally:db.close()

    def task(self,k,branch,state,payload,parents=()):
        with self.db() as db:
            db.execute('INSERT OR REPLACE INTO tasks VALUES(?,?,?,?)',(k,branch,state,json.dumps(payload,ensure_ascii=False)))
            db.executemany('INSERT OR IGNORE INTO dependencies VALUES(?,?)',[(k,p) for p in parents])

    def dependents(self,k):
        with self.db() as db:
            return {r[0] for r in db.execute('WITH RECURSIVE sub(k) AS (SELECT ? UNION SELECT task FROM dependencies JOIN sub ON parent=sub.k) SELECT k FROM sub',(k,))}

    def consume(self,object_key,kind,fingerprint,limit):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM budgets WHERE object_key=? AND kind=? AND fingerprint=?',(object_key,kind,fingerprint)).fetchone():return False
            if db.execute('SELECT count(*) FROM budgets WHERE object_key=? AND kind=?',(object_key,kind)).fetchone()[0]>=limit:
                raise BudgetExceeded(kind+' exhausted for '+object_key)
            db.execute('INSERT INTO budgets VALUES(?,?,?)',(object_key,kind,fingerprint))
        return True

    def candidate(self,object_id,payload):
        k=key(payload)
        with self.db() as db:db.execute('INSERT OR IGNORE INTO candidates VALUES(?,?,?)',(k,object_id,json.dumps(payload,ensure_ascii=False)))
        return k

    def call(self,identity,actor,system,context,images,contract,lease):
        if (self.root/'cancel.json').exists():raise InterruptedError('model dispatch cancelled; retain cached results')
        # Include exact image bytes, prompt, contract and model, never API keys.
        cfg=getattr(actor,'cfg',None)
        transport_binding=cfg.describe() if cfg is not None and hasattr(cfg,'describe') else getattr(actor,'request_binding',{})
        k=key({'identity':identity,'model':actor.model,'endpoint':getattr(cfg,'base_url',getattr(actor,'base_url',None)),
               'request_settings':transport_binding,
               'system':system,'context':context,'images':[digest(p) for p in images],'contract':contract.model_json_schema()})
        raw=self.root/'raw'/f'{k}.json'
        network=self.root/'raw'/f'{k}.network.json';recover=False
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT state,response FROM calls WHERE k=?',(k,)).fetchone()
            if row:
                if row[0]=='COMPLETED':
                    if not raw.is_file() or digest(raw)!=row[1]:raise ValueError('raw response hash mismatch')
                    return self.validated(contract,json.loads(raw.read_text(encoding='utf-8')),raw)
                # A raw receipt may have landed before its DB update. Resume parsing.
                if raw.is_file():
                    result=json.loads(raw.read_text(encoding='utf-8'))
                    db.execute("UPDATE calls SET state='COMPLETED',response=? WHERE k=?",(digest(raw),k))
                    return self.validated(contract,result,raw)
                if network.is_file() and hasattr(actor,'ask_receipted'):recover=True
                else:raise UnknownCall('response unknown; identical paid call will not be repeated: '+k)
            else:db.execute('INSERT INTO calls VALUES(?,?,?,?)',(k,'RESERVED',None,time.time()))
        try:
            if recover:
                # A receipt exists: this method is contractually a local replay.
                result=actor.ask_receipted(system,context,images,contract,network)
            else:
                with lease:
                    result=actor.ask_receipted(system,context,images,contract,network) if hasattr(actor,'ask_receipted') else actor.ask(system,context,images,contract)
            write(raw,result)
            with self.db() as db:db.execute("UPDATE calls SET state='COMPLETED',response=? WHERE k=?",(digest(raw),k))
        except BaseException:
            with self.db() as db:db.execute("UPDATE calls SET state='UNKNOWN' WHERE k=? AND state!='COMPLETED'",(k,))
            raise
        # Schema failures keep immutable raw responses, not a corrected fake receipt.
        return self.validated(contract,result,raw)

    def validated(self,contract,payload,raw):
        result=contract.model_validate(payload)
        if hasattr(result,'references'):
            # Raw provider references are not trusted file paths. The original
            # JSON stays unchanged, and this explicit adapter records the mapping.
            supplied=list(result.references);result.references=[str(Path(raw).resolve())]
            network=Path(raw).with_name(Path(raw).stem+'.network.json')
            if network.is_file():result.references.append(str(network.resolve()))
            if supplied:
                record={'schema':'ModelReferenceAdapter/1','raw_reply_sha256':digest(raw),
                        'supplied_references':supplied,'filesystem_authority':False}
                ref=self.root/'reference-adapters'/(key(record)+'.json')
                if not ref.exists():write(ref,record)
                result.references.append(str(ref.resolve()))
        return result

    def evidence(self,record):
        k=key(record);file=self.root/'evidence'/f'{k}.json'
        if not file.exists():write(file,record)
        if key(json.loads(file.read_text(encoding='utf-8')))!=k:raise ValueError('evidence corrupted')
        with self.db() as db:db.execute('INSERT OR IGNORE INTO evidence VALUES(?,?,?)',(k,str(file),digest(file)))
        return str(file)

    def head(self):
        with self.db() as db:return db.execute('SELECT generation FROM head WHERE id=1').fetchone()[0]

    def publish(self,manifest,expected):
        # All referenced pages/assets/QA must exist before any visibility change.
        for p,sha in manifest['files'].items():
            if digest(p)!=sha:raise ValueError('generation artifact hash mismatch')
        generation=expected+1;file=self.root/'generations'/f'{generation}-{key(manifest)}.json'
        write(file,manifest)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT generation FROM head WHERE id=1').fetchone()[0]!=expected:raise Conflict('generation CAS conflict')
            db.execute('INSERT INTO commits VALUES(?,?,?)',(generation,str(file),digest(file)))
            db.execute('UPDATE head SET generation=? WHERE id=1 AND generation=?',(generation,expected))
        return generation

    def committed(self):
        with self.db() as db:
            row=db.execute('SELECT manifest,sha FROM commits JOIN head ON commits.generation=head.generation WHERE head.id=1').fetchone()
        if not row:return None
        if digest(row[0])!=row[1]:raise ValueError('committed manifest corrupted')
        record=json.loads(Path(row[0]).read_text(encoding='utf-8'))
        for p,sha in record['files'].items():
            if digest(p)!=sha:raise ValueError('committed artifact corrupted')
        return record
