"""Evaluate a two-page research job against private edit cases and field references."""

import argparse, collections, hashlib, json, pathlib, shutil, sys
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "product"))
from probe_fields import run as probe_edits
from product.runtime.compiler import compile_local
from product.runtime.office import check_local
from product.parallel_pipeline import effective_texts
from product.parallel_contracts import SceneGraph
from product.scorecard import score_page
from product.resource_lease import lease


def read(f):
    return json.loads(f.read_text(encoding="utf-8-sig"))


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def overlap(box, target):
    x, y, w, h = box
    a, b, c, d = target
    shared = max(0, min(x + w, a + c) - max(x, a)) * max(
        0, min(y + h, b + d) - max(y, b)
    )
    return shared / max(1, w * h + c * d - shared)


def field_evidence(state, cases):
    evidence = []
    for pg, label, target, _, _ in cases:
        page = str(pg)
        marks = (
            read(pathlib.Path(state["experiment_marks"][page]))["rows"]
            if page in state.get("experiment_marks", {})
            else []
        )
        planned = state.get("marked_plans", {}).get(page, [])

        def near(rows):
            hits = sorted(
                rows, key=lambda row: overlap(row["bbox"], target), reverse=True
            )
            return [
                {
                    **{
                        k: r[k]
                        for k in (
                            "id",
                            "label",
                            "role",
                            "route",
                            "source_text",
                            "move_with",
                            "bbox",
                        )
                        if k in r
                    },
                    "iou": round(overlap(r["bbox"], target), 3),
                }
                for r in hits[:3]
                if overlap(r["bbox"], target) > 0
            ]

        evidence.append(
            {
                "page": pg,
                "target": label,
                "bbox": target,
                "marks": near(marks),
                "plan": near(planned),
            }
        )
    return evidence


def run(job, out, references, cases):
    out.mkdir(exist_ok=True, parents=True)
    state = read(job / "runtime/state.json")
    result = read(job / "result.json") if (job / "result.json").exists() else {}
    rows = state.get("pages", [])
    if len(rows) != 2 or not all(rows):
        raise ValueError("both pages must be initialized")
    irs = [pathlib.Path(r["ir"]) for r in rows]
    c = state.get("composition", {})
    deck = out / "two-pages.pptx"
    if c:
        original = pathlib.Path(c["artifacts"]["pptx"])
        if sha(original) != c["files"][str(original)]:
            raise ValueError("composition hash mismatch")
        origin = "agent-composed"
        shutil.copyfile(original, deck)
    else:
        if not result:
            raise ValueError("wait for finalization or agent composition")
        gs = [
            SceneGraph.model_validate_json(
                pathlib.Path(r["scene"]).read_text(encoding="utf8")
            )
            for r in rows
        ]
        with lease("compiler"):
            artifact = compile_local(
                irs,
                out / "from-committed-scenes",
                [effective_texts(g, ir) for g, ir in zip(gs, irs)],
            )
        shutil.copyfile(artifact["pptx"], deck)
        origin = "diagnostic composition of committed scenes"
    d = sha(deck)
    checks = None
    for f in job.rglob("checks.json"):
        v = read(f)
        if (
            v.get("status") == "PASS"
            and v.get("pptx_sha256") == d
            and all(
                (
                    pathlib.Path(v["office_directory"]) / f"page-{i:02d}-before.png"
                ).is_file()
                for i in (1, 2)
            )
        ):
            checks = v
            break
    if checks is None:
        checks = check_local(deck, irs, out / "office")
    pages = []
    for i, r in enumerate(rows, 1):
        source = pathlib.Path(r["snapshot"]["path"])
        ref = read(references / f"reference-fields-{i}.json")
        render = pathlib.Path(checks["office_directory"]) / f"page-{i:02d}-before.png"
        score = score_page(
            source,
            deck,
            slide=i,
            checks=checks if checks.get("office", {}).get("pptx_sha256") == d else None,
            render_path=render if render.exists() else None,
            ocr=ref,
        )
        pages.append(
            {
                k: score.get(k)
                for k in (
                    "fidelity",
                    "native_editable_fraction",
                    "movable_asset_fraction",
                    "object_counts",
                    "duplicate_picture_fraction",
                    "raster_plate_content_fraction",
                    "non_logo_matched_lines",
                    "non_logo_reference_lines",
                )
            }
        )
        if render.exists():
            shutil.copyfile(render, out / f"page-{i}.png")
            sheet = Image.new("RGB", (1672, 500), "white")
            draw = ImageDraw.Draw(sheet)
            for j, f in enumerate((source, render)):
                im = Image.open(f).convert("RGB")
                im.thumbnail((836, 470))
                sheet.paste(im, (j * 836, 30))
                draw.text((j * 836 + 10, 7), ("Source", "PowerPoint")[j], fill="black")
            sheet.save(out / f"page-{i}-comparison.png")
    with lease("office"):
        probe_edits(deck, out / "edit-tests", cases)
    edits = read(out / "edit-tests/report.json")
    counts = collections.Counter()
    attempted = collections.Counter()
    strategies = collections.Counter()
    trace = collections.Counter()
    for f in (job / "runtime").rglob("receipt.json"):
        a = read(f)
        if "model" in a and "input_sha256" in a:
            counts[a["model"] + "/" + a["status"]] += 1
    for f in (job / "runtime").rglob("strategy-comparison.json"):
        a = read(f)
        for row in a["attempts"]:
            attempted[row["strategy"] + "/" + row["status"]] += 1
        if a["ranked"]:
            strategies[a["ranked"][0]["strategy"]] += 1
    for f in (job / "runtime").rglob("trace/selection.json"):
        trace[read(f)["status"]] += 1
    report = {
        "job": job.name,
        "status": result.get("status", state.get("status")),
        "reason": result.get("reason"),
        "origin": origin,
        "sha256": d,
        "office": checks["status"],
        "office_detail": {
            k: checks.get(k) for k in ("failure", "reason", "error_type") if k in checks
        },
        "pages": pages,
        "edits": edits,
        "model_receipts": dict(counts),
        "strategy_attempts": dict(attempted),
        "best_ranked_by_attempt": dict(strategies),
        "svg_decisions": dict(trace),
        "human_acceptance": "unreviewed",
    }
    report["field_evidence"] = field_evidence(state, cases)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(
        origin,
        checks["status"],
        [(p["non_logo_matched_lines"], p["non_logo_reference_lines"]) for p in pages],
        [r["status"] for r in edits],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("job", "out", "references", "cases"):
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    args = parser.parse_args()
    run(args.job, args.out, args.references, read(args.cases))
