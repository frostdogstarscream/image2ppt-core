"""Stage a new GPU service tree, including the SAME cross-service lease code."""
import argparse,hashlib,json,shutil
from pathlib import Path

def prepare(target):
    services=Path(__file__).resolve().parent;target=Path(target).resolve()
    target.mkdir(parents=True,exist_ok=False)
    for name in ('layerd','segmentation'):
        (target/name).mkdir()
        for file in (services/name).iterdir():
            if file.is_file():shutil.copyfile(file,target/name/file.name)
    shutil.copyfile(services/'resource_lease.py',target/'resource_lease.py')
    # Docker's restricted build context receives an exact generated copy.
    shutil.copyfile(services/'resource_lease.py',target/'layerd/resource_lease.py')
    records={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file()}
    (target/'staging-manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(records),'shared_lease_sha256':records['resource_lease.py']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();prepare(a.output)
