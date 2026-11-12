"""Run a three-vision-call page revision without Codex planning inside the loop.

Input is a previously model-authored SlideIR plus its frozen source and assets.
This is an automatic revision pipeline, not fresh screenshot recognition.
"""
from __future__ import annotations

import argparse
import getpass
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from apply_model_style import apply as apply_style
from extract_background_planes import build as build_planes, detect as detect_planes
from model_slideir_loop import compiler, digest, mae, write_json
from moma_page_analysis import run as call_model


def _phase(name: str, *, source: Path, ir: Path, output: Path, key: str,
           plate: Path | None = None, preview: Path | None = None,
           candidates: Path | None = None) -> dict:
    options = SimpleNamespace(phase=name, source=source, ir=ir, output=output,
                              plate=plate, preview=preview, candidates=candidates,
                              timeout=240, max_tokens=5000)
    return call_model(options, api_key=key)


def _answer(name: str, *, source: Path, ir: Path, phase_dir: Path,
            key: str | None, supplied: Path | None = None,
            plate: Path | None = None, preview: Path | None = None,
            candidates: Path | None = None) -> tuple[Path, dict]:
    """Reuse recorded responses; a 429 attempt is never overwritten or hot retried."""
    if supplied is not None:
        path = supplied.resolve(strict=True)
        return path, {"status": "REUSED", "phase": name, "answer_sha256": digest(path)}
    attempts = sorted(phase_dir.glob("attempt-*/reservation.json"))
    for reservation in attempts:
        answer = reservation.parent / "answer.json"
        if answer.exists():
            return answer, {"status": "REUSED", "phase": name,
                            "answer_sha256": digest(answer)}
    if attempts:
        last = attempts[-1].parent
        failure_file = last / "failure.json"
        if not failure_file.exists():
            raise RuntimeError(f"{name} response is unknown; inspect {last} before retry")
        failure = json.loads(failure_file.read_text(encoding="utf-8"))
        if failure.get("http_status") == 429:
            delay = failure.get("retry_after_seconds")
            if not isinstance(delay, (int, float)) or delay < 0:
                delay = 60
            eligible = datetime.fromtimestamp(failure_file.stat().st_mtime,
                                              timezone.utc) + timedelta(seconds=delay)
            if datetime.now(timezone.utc) < eligible:
                raise RuntimeError(f"{name} rate limited; next attempt after {eligible.isoformat()}")
        elif failure.get("status") == "RESPONSE_UNKNOWN":
            raise RuntimeError(f"{name} response is unknown; inspect {last} before retry")
    if not key:
        raise RuntimeError(f"{name} needs an ephemeral API key; completed phases are retained")
    index = len(attempts) + 1
    output = phase_dir / f"attempt-{index:02d}"
    result = _phase(name, source=source, ir=ir, output=output, key=key,
                    plate=plate, preview=preview, candidates=candidates)
    return output / "answer.json", result


def run(ir_path: Path, output: Path, *, key: str | None,
        row_count: int = 5, checkbox_count: int = 3,
        style_answer: Path | None = None, asset_answer: Path | None = None,
        review_answer: Path | None = None) -> dict:
    ir_path = ir_path.resolve(strict=True)
    output = output.resolve()
    input_ir = json.loads(ir_path.read_text(encoding="utf-8"))
    source = ir_path.parent / input_ir["source_image"]
    plate_asset = next(a for a in input_ir["assets"]
                       if a["id"] == input_ir["background"]["asset_id"])
    plate = ir_path.parent / plate_asset["path"]
    if digest(source) != input_ir["source_sha256"] or digest(plate) != plate_asset["sha256"]:
        raise ValueError("frozen image or plate hash mismatch")
    identity = {"input_ir_sha256": digest(ir_path),
                                      "source_sha256": digest(source),
                                      "plate_sha256": digest(plate),
                                      "model": "qwen/qwen3.8-max",
                                      "planned_visual_calls": ["style", "assets", "review"]}
    input_record = output / "input.json"
    if output.exists():
        if not input_record.exists() or json.loads(input_record.read_text(encoding="utf-8")) != identity:
            raise ValueError("existing page revision has a different frozen input")
    else:
        output.mkdir(parents=True)
        write_json(input_record, identity)
    style_response, first = _answer("style", source=source, ir=ir_path,
                                    phase_dir=output / "model-style", key=key,
                                    supplied=style_answer)
    style_dir = output / "style"
    if style_dir.exists():
        recorded = json.loads((style_dir / "authoring.json").read_text(encoding="utf-8"))
        if recorded["model_answer_sha256"] != digest(style_response):
            raise ValueError("style answer changed during resume")
    else:
        apply_style(ir_path, style_response, style_dir)
    candidates = detect_planes(plate, row_count=row_count,
                               checkbox_count=checkbox_count)
    candidate_file = output / "plane-candidates.json"
    if candidate_file.exists():
        if json.loads(candidate_file.read_text(encoding="utf-8")) != candidates:
            raise ValueError("plane candidates changed during resume")
    else:
        write_json(candidate_file, candidates)
    assets_response, second = _answer("assets", source=source,
                                      ir=style_dir / "slideir.json",
                                      plate=plate, candidates=candidate_file,
                                      phase_dir=output / "model-assets", key=key,
                                      supplied=asset_answer)
    extract_dir = output / "extracted"
    if extract_dir.exists():
        recorded = json.loads((extract_dir / "authoring.json").read_text(encoding="utf-8"))
        if recorded["model_review_sha256"] != digest(assets_response):
            raise ValueError("asset answer changed during resume")
    else:
        build_planes(style_dir / "slideir.json", candidate_file,
                     assets_response, extract_dir)
    final_ir = extract_dir / "slideir.json"
    validation = compiler(final_ir, "validate")
    compiled_record = output / "compiled.json"
    if compiled_record.exists():
        compiled = json.loads(compiled_record.read_text(encoding="utf-8"))
        if not Path(compiled["pptx"]).exists() or not Path(compiled["preview"]).exists():
            raise ValueError("recorded compilation is missing its artifacts")
    else:
        compiled = compiler(final_ir, "compile")
        write_json(compiled_record, compiled)
    preview = Path(compiled["preview"])
    review_context = output / "review-context.json"
    context = {"extraction": str(extract_dir / "extraction.json"),
               "style_audit": str(style_dir / "style-audit.json"),
               "validation": validation, "preview_sha256": digest(preview)}
    if review_context.exists():
        if json.loads(review_context.read_text(encoding="utf-8")) != context:
            raise ValueError("review context changed during resume")
    else:
        write_json(review_context, context)
    review_response, third = _answer("review", source=source, ir=final_ir,
                                     preview=preview, candidates=review_context,
                                     phase_dir=output / "model-review", key=key,
                                     supplied=review_answer)
    review = json.loads(review_response.read_text(encoding="utf-8"))
    if review.get("schema") != "render-review/1" or review.get("source_sha256") != digest(source) or review.get("preview_sha256") != digest(preview):
        raise ValueError("final visual review is not bound to the compiled preview")
    model_results = [first, second, third]
    result = {"status": "REVIEW_REQUIRED",
              "model_calls_this_run": sum(r.get("status") != "REUSED" for r in model_results),
              "visual_review_phases": 3,
              "model_results": model_results, "final_ir": str(final_ir),
              "pptx": compiled["pptx"], "preview": compiled["preview"],
              "page_mae": mae(source, preview), "visual_review": review,
              "human_acceptance": False, "powerpoint_checked": False,
              "wps_checked": False}
    write_json(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-count", type=int)
    parser.add_argument("--checkbox-count", type=int)
    parser.add_argument("--style-answer", type=Path)
    parser.add_argument("--asset-answer", type=Path)
    parser.add_argument("--review-answer", type=Path)
    args = parser.parse_args()
    secret = getpass.getpass("MoMA key (memory only; blank to reuse recorded phases): ").strip()
    print(json.dumps(run(args.ir, args.output, key=secret,
                         row_count=args.row_count or 5,
                         checkbox_count=args.checkbox_count or 3,
                         style_answer=args.style_answer,
                         asset_answer=args.asset_answer,
                         review_answer=args.review_answer),
                     ensure_ascii=False, indent=2))
