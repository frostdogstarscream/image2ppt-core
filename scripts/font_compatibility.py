"""Create bounded font-size candidates and select them using actual PPT previews."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"services/vision/src"))
from image2ppt_vision.font_compatibility import constrain, select, verify_selection

parser = argparse.ArgumentParser()
commands = parser.add_subparsers(dest="command", required=True)
p = commands.add_parser("constrain")
p.add_argument("--ir", required=True)
p.add_argument("--output", required=True)
p = commands.add_parser("select")
for field in ("baseline", "candidate", "baseline-pptx", "baseline-preview",
              "candidate-pptx", "candidate-preview", "output"):
    p.add_argument("--"+field, required=True)
p.add_argument("--baseline-slide", type=int, default=1)
p = commands.add_parser("verify")
for field in ("baseline", "selected", "baseline-pptx", "baseline-preview",
              "selected-pptx", "selected-preview", "report"):
    p.add_argument("--"+field, required=True)
p.add_argument("--baseline-slide", type=int, default=1)
args = vars(parser.parse_args())
action = args.pop("command")
if action == "constrain":
    args["irfile"] = args.pop("ir")
if action == "verify":
    report = Path(args.pop("report"))
    if report.exists():
        raise FileExistsError("new verification report required")
    result = verify_selection(**args)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    result = {k: v for k, v in result.items() if k != "regions"}
else:
    result = (constrain if action == "constrain" else select)(**args)
print(json.dumps(result, ensure_ascii=False))
