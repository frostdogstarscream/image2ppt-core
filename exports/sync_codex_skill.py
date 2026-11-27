"""Sync the Codex flavor only; --check performs no writes or deletions."""
import argparse
import hashlib
import json
from pathlib import Path
from skill_files import skill_files

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    source=ROOT/'skills/image2ppt';target=ROOT/'.agents/skills/image2ppt'
    expected=skill_files(source)
    for name in expected:
        destination=target/name
        destination.resolve().relative_to(target.resolve())
        if not args.check:
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes((source/name).read_bytes())
    mismatches=[name for name in expected if not (target/name).is_file() or (target/name).read_bytes()!=(source/name).read_bytes()]
    extras=sorted(set(skill_files(target))-set(expected)) if target.exists() and (target/'SKILL.md').exists() else []
    print(json.dumps({'status':'PASS' if not mismatches and not extras else 'FAIL','check_only':args.check,
        'source':str(source),'target':str(target),'mismatches':mismatches,'unexpected_files':extras,
        'sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in expected}},indent=2))
    if mismatches or extras:raise SystemExit(1)


if __name__=='__main__':main()
