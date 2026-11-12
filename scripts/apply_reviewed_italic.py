"""Apply a hash-bound model render review's numbered italic correction."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from model_slideir_loop import digest, patch, write_json


def apply(ir_path: Path, preview: Path, review_path: Path, output: Path) -> dict:
    ir_path = ir_path.resolve(strict=True)
    preview = preview.resolve(strict=True)
    review_path = review_path.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise ValueError("refusing to overwrite italic correction")
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (review.get("schema") != "render-review/1" or
            review.get("source_sha256") != ir["source_sha256"] or
            review.get("preview_sha256") != digest(preview)):
        raise ValueError("render review is not bound to this source and preview")
    issues = review.get("style_issues")
    if (not isinstance(issues, list) or len(issues) != 1 or
            not isinstance(issues[0], str) or
            "01-05" not in issues[0] or "斜体" not in issues[0]):
        raise ValueError("model review does not identify the numbered italic issue")
    nodes = {node["id"]: node for node in ir["nodes"]}
    ids = [f"t-row{i}-num" for i in range(1, 6)]
    if any(nodes.get(node_id, {}).get("text") != f"0{i}" for i, node_id in enumerate(ids, 1)):
        raise ValueError("numbered text node mapping changed")
    updated = patch(ir, {"patches": [{"id": node_id, "changes": {"italic": True}}
                                     for node_id in ids],
                         "review_notes": ["Number italics corrected from hash-bound MoMA render review."]})
    output.mkdir(parents=True)
    for name in ("source", "assets"):
        shutil.copytree(ir_path.parent / name, output / name)
    write_json(output / "slideir.json", updated)
    write_json(output / "authoring.json", {"format_version": "1",
                                            "tool": "model-render-style-correction",
                                            "ir_sha256": digest(output / "slideir.json"),
                                            "parent_ir_sha256": digest(ir_path),
                                            "review_sha256": digest(review_path),
                                            "reviewed_preview_sha256": digest(preview)})
    report = {"status": "APPLIED_REVIEW_REQUIRED", "italic_node_ids": ids,
              "source_sha256": ir["source_sha256"],
              "review_sha256": digest(review_path),
              "reviewed_preview_sha256": digest(preview)}
    write_json(output / "italic-audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply(args.ir, args.preview, args.review, args.output),
                     ensure_ascii=False, indent=2))
