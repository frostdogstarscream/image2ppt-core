"""Source-pixel graphics completion without altering existing nodes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"services/vision/src"))
from image2ppt_vision.source_graphics import materialize, verify

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="command", required=True)
q = sub.add_parser("apply")
q.add_argument("--folder", required=True)
q.add_argument("--output", required=True)
q = sub.add_parser("verify")
q.add_argument("--folder", required=True)
q.add_argument("--pptx")
args = vars(p.parse_args())
command = args.pop("command")
print(json.dumps((materialize if command == "apply" else verify)(**args), ensure_ascii=False))
