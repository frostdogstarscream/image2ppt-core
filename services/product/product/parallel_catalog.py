"""Derived stable-asset index. Committed scene manifests remain authoritative."""
import json,sqlite3
from pathlib import Path
from .materialize import digest
from .parallel_store import Store


class Catalog:
    def __init__(self,root):
        self.path=Path(root)/'catalog.sqlite';self.path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path) as db:db.execute('CREATE TABLE IF NOT EXISTS stable(source TEXT,id TEXT,record TEXT,PRIMARY KEY(source,id))')

    def register(self,job):
        store=Store(Path(job)/'state');commit=store.committed()
        if commit is None:return
        with sqlite3.connect(self.path) as db:
            for row in commit.get('stable_assets',[]):
                if row.get('automatic_quality')!='PASS' or row['path'] not in commit['files'] or commit['files'][row['path']]!=row['sha256']:
                    raise ValueError('stable index lacks committed quality evidence')
                record={**row,'commit_job':str(Path(job).resolve()),'generation':store.head()}
                db.execute('INSERT OR REPLACE INTO stable VALUES(?,?,?)',(row['source_sha256'],row['id'],json.dumps(record,ensure_ascii=False)))

    def lookup(self,source):
        with sqlite3.connect(self.path) as db:rows=db.execute('SELECT record FROM stable WHERE source=?',(source,)).fetchall()
        result=[]
        for (payload,) in rows:
            row=json.loads(payload)
            try:
                store=Store(Path(row['commit_job'])/'state');commit=store.committed()
                if store.head()!=row['generation'] or not commit or commit['files'].get(row['path'])!=row['sha256']:continue
                if digest(row['path'])!=row['sha256']:continue
                result.append(row)
            except (OSError,ValueError,sqlite3.Error):continue
        return result
