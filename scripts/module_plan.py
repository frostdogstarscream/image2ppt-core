"""Prepare, materialize and verify source-bound ModulePlan sidecars."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"services/vision/src"))
from image2ppt_vision.module_plan import prepare_module_job, apply_module_job, verify_module_page

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="command", required=True)
q = sub.add_parser("prepare")
for field in ("ir", "output", "model", "base-url"):
    q.add_argument("--"+field, required=True)
q = sub.add_parser("apply")
q.add_argument("--job", required=True); q.add_argument("--output", required=True)
q = sub.add_parser("verify")
q.add_argument("--folder", required=True); q.add_argument("--pptx")
q.add_argument("--slide", type=int, default=1)
args = vars(p.parse_args()); command = args.pop("command")
if command == "prepare":
    args["irfile"] = args.pop("ir")
result = {"prepare": prepare_module_job, "apply": apply_module_job, "verify": verify_module_page}[command](**args)
print(json.dumps(result, ensure_ascii=False))
