"""Apply a source-bound visual style review without changing model-authored text."""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from pathlib import Path

from model_slideir_loop import digest, patch, write_json
from model_text_layout import calibrate


def apply(ir_path: Path, answer_path: Path, output: Path) -> dict:
    ir_path = ir_path.resolve(strict=True)
    answer_path = answer_path.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise ValueError("refusing to overwrite style output")
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    if answer.get("schema") != "style-review/1" or answer.get("source_sha256") != ir["source_sha256"]:
        raise ValueError("style review schema or source mismatch")
    text_nodes = {n["id"]: n for n in ir["nodes"] if n["type"] == "text"}
    items = answer.get("items")
    if (not isinstance(items, list) or len(items) != len(text_nodes)
            or {item.get("id") for item in items if isinstance(item, dict)} != set(text_nodes)):
        raise ValueError("style review must cover each text node exactly once")
    patches, audit = [], []
    for item in items:
        if not isinstance(item.get("bold"), bool):
            raise ValueError("style bold must be boolean")
        changes = {"bold": item["bold"]}
        runs = item.get("runs")
        if runs is not None:
            if (not isinstance(runs, list) or not runs or
                    not all(isinstance(run, dict) and
                            set(run) <= {"text", "bold", "color"} and
                            isinstance(run.get("text"), str) and run["text"] and
                            ("bold" not in run or isinstance(run["bold"], bool)) and
                            ("color" not in run or isinstance(run["color"], str) and
                             re.fullmatch(r"[0-9a-fA-F]{6}", run["color"]))
                            for run in runs) or
                    "".join(run["text"] for run in runs) != text_nodes[item["id"]]["text"]):
                raise ValueError("style runs must preserve exact source text")
            changes["runs"] = runs
        patches.append({"id": item["id"], "changes": changes})
        audit.append({"id": item["id"], "text": text_nodes[item["id"]]["text"],
                      "before_bold": text_nodes[item["id"]].get("bold"),
                      "reviewed_bold": item["bold"], "runs": runs,
                      "evidence": item.get("evidence"),
                      "confidence": item.get("confidence")})
    updated = patch(ir, {"patches": patches, "review_notes": [
        "Text emphasis reviewed against source image by a hash-bound MoMA vision response."]})
    source = ir_path.parent / ir["source_image"]
    updated, calibration = calibrate(updated, source, ir_path.parent)
    output.mkdir(parents=True)
    for name in ("source", "assets"):
        shutil.copytree(ir_path.parent / name, output / name)
    write_json(output / "slideir.json", updated)
    write_json(output / "authoring.json", {"format_version": "1", "tool": "model-style-review",
                                            "ir_sha256": digest(output / "slideir.json"),
                                            "parent_ir_sha256": digest(ir_path),
                                            "model_answer_sha256": digest(answer_path)})
    report = {"status": "APPLIED", "source_sha256": ir["source_sha256"],
              "model_answer_sha256": digest(answer_path),
              "red_present": answer.get("red_present"), "rows": audit,
              "calibration_status": calibration["status"]}
    write_json(output / "style-audit.json", report)
    write_json(output / "text-layout.json", calibration)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--answer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = apply(args.ir, args.answer, args.output)
    print(json.dumps({"status": result["status"], "rows": len(result["rows"]),
                      "output": str(args.output)}, ensure_ascii=False))
