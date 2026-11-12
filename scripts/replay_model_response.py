"""Compile a recorded model response offline after deterministic protocol sealing."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from model_slideir_loop import compiler, digest, mae, seal, write_json


def replay(run_dir: Path, response_path: Path, output: Path,
           known_asset_ir: Path | None = None) -> dict:
    run_dir = run_dir.resolve(strict=True)
    response_path = response_path.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise ValueError("refusing to overwrite replay output")
    if not response_path.is_relative_to(run_dir):
        raise ValueError("response must belong to the selected run")
    manifest = json.loads((run_dir / "input-manifest.json").read_text(encoding="utf-8"))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    source = run_dir / "source/page.png"
    if digest(source) != manifest["source_sha256"]:
        raise ValueError("source bytes changed since model call")
    for asset in manifest["assets"].values():
        if digest(run_dir / asset["path"]) != asset["sha256"]:
            raise ValueError("asset bytes changed since model call")
    ir = seal(response["answer"], source, manifest["assets"])
    normalizations = ["schema_version SlideIR 0.1 -> 0.1",
                      "six-digit hex colors: remove leading #",
                      "review_notes string -> one-item array"]
    if known_asset_ir:
        known = json.loads(known_asset_ir.resolve(strict=True).read_text(encoding="utf-8"))
        provenance = {a["id"]: a for a in known["assets"]}
        for asset in ir["assets"]:
            saved = provenance.get(asset["id"])
            if not saved or saved["sha256"] != asset["sha256"] or saved["path"] != asset["path"]:
                raise ValueError("trusted asset metadata does not match model asset")
            asset["purpose"] = saved["purpose"]
        normalizations.append("asset purpose copied from hash-matched prior asset manifest")
        backgrounds = [a for a in ir["assets"] if a["purpose"] == "cleaned_background"]
        for asset in backgrounds:
            owners = [n for n in ir["nodes"] if n.get("type") == "image" and
                      n.get("asset_id") == asset["id"]]
            if len(owners) == 1 and owners[0]["bbox"] == [0, 0, *manifest["source_size"]] and not owners[0]["evidence_ids"]:
                ir["nodes"].remove(owners[0])
                ir["background"]["asset_id"] = asset["id"]
                normalizations.append("full-page cleaned plate node -> background.asset_id")
    output.mkdir(parents=True)
    shutil.copytree(run_dir / "source", output / "source")
    shutil.copytree(run_dir / "assets", output / "assets")
    ir_file = output / "slideir.json"
    write_json(ir_file, ir)
    write_json(output / "authoring.json", {"format_version": "1",
                                            "tool": "recorded-model-response-replay",
                                            "model_id": response["model"],
                                            "created_at": datetime.now(timezone.utc).isoformat(),
                                            "ir_sha256": digest(ir_file),
                                            "source_response_sha256": digest(response_path),
                                            "normalization": normalizations})
    try:
        validation = compiler(ir_file, "validate")
        build = compiler(ir_file, "compile")
        score = mae(source, Path(build["preview"]))
        result = {"status": "REVIEW_REQUIRED", "ir": str(ir_file), "build": build,
                  "validation": validation, "page_mae": score,
                  "human_acceptance": False, "model_calls": 0}
    except Exception as exc:
        result = {"status": "FAILED", "ir": str(ir_file), "error": str(exc)[-3000:],
                  "human_acceptance": False, "model_calls": 0}
    write_json(output / "replay.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--known-asset-ir", help="Prior IR only for hash-matched asset purpose metadata")
    args = parser.parse_args()
    try:
        result = replay(Path(args.run), Path(args.response), Path(args.output),
                        Path(args.known_asset_ir) if args.known_asset_ir else None)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] != "FAILED" else 2
    except Exception as exc:
        print(f"replay-model-response: {type(exc).__name__}: {str(exc)[-500:]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
