"""Measure Artifact Tool text-origin correction for the vision renderer profile.

Builds probe slides from installed fonts, compiles them with this project's
compiler, measures ink origins in the reimported preview, and writes a candidate
profile. It never overwrites the bundled vision profile: a human must review the
candidate and install it explicitly.

This is a host-side tool. It does not call a model API and does not write into
task outputs.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SIZE = (1672, 941)
TARGET_SIZE = (1280, 720)
SCALE = TARGET_SIZE[0] / SOURCE_SIZE[0]
SIZES = (8, 12, 16, 24, 32, 40)
SCRIPTS = {"cjk": "纪检智能分析", "latin": "Ag09Bypx"}
FAMILIES = (
    ("msyh.ttc", "微软雅黑", False), ("msyhbd.ttc", "微软雅黑", True),
    ("Deng.ttf", "等线", False), ("Dengb.ttf", "等线", True),
    ("simhei.ttf", "黑体", False), ("simsun.ttc", "宋体", False),
    ("arial.ttf", "Arial", False), ("arialbd.ttf", "Arial", True),
)
ROW_HEIGHT = 78
COLUMNS = {"cjk": 60, "latin": 900}


def node_executable():
    configured = os.environ.get("IMAGE2PPT_NODE")
    if configured:
        return configured
    return str(Path(os.environ["USERPROFILE"]) /
               ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe")


def python_json(command):
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    text = result.stdout
    start = text.find("{")
    if start < 0:
        raise RuntimeError("command did not return JSON: " + text[-500:])
    return json.loads(text[start:])


def font_probe(font_path, family, bold, slide_dir, index):
    """Write one source page and its SlideIR with 12 isolated text samples."""
    slide_dir.mkdir(parents=True, exist_ok=False)
    (slide_dir / "source").mkdir()
    page = Image.new("RGB", SOURCE_SIZE, "white")
    draw = ImageDraw.Draw(page)
    nodes = []
    evidence = []
    samples = []
    z = 1
    has_cjk = family != "Arial"
    for row, size in enumerate(SIZES):
        for script, text in SCRIPTS.items():
            if script == "cjk" and not has_cjk:
                continue
            font = ImageFont.truetype(str(font_path), size)
            x = COLUMNS[script]
            y = 10 + row * 2 * ROW_HEIGHT + (ROW_HEIGHT if script == "latin" else 0)
            draw.text((x, y), text, font=font, fill=(0, 0, 0))
            box = font.getbbox(text)
            ink = [x + box[0], y + box[1], box[2] - box[0], box[3] - box[1]]
            samples.append({"id": "%s-%s-%d" % (script, size, index), "script": script,
                            "size_px": size, "source_ink_bbox": ink,
                            "slide_index": index})
            node_id = "text-%s-%02d-%02d" % (script, size, index)
            evidence_id = "ev-" + node_id
            evidence.append({"id": evidence_id, "kind": "text",
                             "bbox": ink, "text": text})
            nodes.append({
                "id": node_id, "type": "text",
                "bbox": [x, y, 700, max(24, int(size * 1.5))], "z": z,
                "text": text, "font": {"latin": family, "ea": family,
                                       "size_pt": round(size * SCALE * .75, 4)},
                "bold": bold, "color": "000000", "align": "left", "wrap": False,
                "source_text_bbox": [x, y, 700, max(24, int(size * 1.5))],
                "evidence_ids": [evidence_id],
            })
            z += 1
    source = slide_dir / "source/page.png"
    page.save(source)
    ir = {"schema_version": "0.1", "source_image": "source/page.png",
          "source_sha256": sha256(source),
          "canvas": {"source_px": list(SOURCE_SIZE), "target_px": list(TARGET_SIZE)},
          "background": {"color": "FFFFFF"}, "review_status": "needs_review",
          "review_notes": ["Origin-profile probe; not a real slide reconstruction."],
          "evidence": evidence, "assets": [], "groups": [], "nodes": nodes}
    (slide_dir / "slideir.json").write_text(json.dumps(ir, ensure_ascii=False), encoding="utf-8")
    return slide_dir / "slideir.json", samples


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rendered_ink(page, bbox, source_size, target_size):
    """Locate the rendered ink box inside the scaled search window."""
    x, y, w, h = bbox
    sx = target_size[0] / source_size[0]
    sy = target_size[1] / source_size[1]
    left = max(0, int((x - 12) * sx)); top = max(0, int((y - 12) * sy))
    right = min(page.width, int((x + w + 12) * sx)); bottom = min(page.height, int((y + h + 18) * sy))
    crop = np.asarray(page.crop((left, top, right, bottom)).convert("RGB"))
    mask = np.min(crop, axis=2) < 200
    if not mask.any():
        return None
    yy, xx = np.where(mask)
    return [left + int(xx.min()), top + int(yy.min()),
            int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="candidate profile JSON")
    parser.add_argument("--report", help="per-sample report JSON")
    parser.add_argument("--work", help="probe workspace (default under .build)")
    parser.add_argument("--renderer-version", default=None)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("refusing to overwrite: " + str(output))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = Path(args.work).resolve() if args.work else ROOT / ".build" / ("origin-profile-" + stamp)
    work.mkdir(parents=True, exist_ok=False)
    fonts_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    irs, samples = [], []
    for index, (filename, family, bold) in enumerate(FAMILIES):
        font_path = fonts_dir / filename
        if not font_path.is_file():
            print("skip missing font", filename, file=sys.stderr)
            continue
        ir, rows = font_probe(font_path, family, bold, work / ("slide-%02d" % index), index)
        irs.append(ir)
        for row in rows:
            row.update(family=family, bold=bold, font_sha256=sha256(font_path))
        samples.extend(rows)
    if not irs:
        raise SystemExit("no probe fonts installed")
    composed = python_json([node_executable(), str(ROOT / "scripts/compose.mjs"),
                            *map(str, irs), "--output", str(work / "deck")])
    preview_dir = Path(composed["preview"])
    pages = sorted(preview_dir.glob("page-*.png"))
    if len(pages) != len(irs):
        raise SystemExit("preview page count mismatch")
    measured = []
    by_slide = {}
    for sample in samples:
        by_slide.setdefault(sample["slide_index"], []).append(sample)
    for slide_index, page_path in enumerate(pages):
        with Image.open(page_path) as page:
            page = page.convert("RGB")
            for sample in by_slide.get(slide_index, []):
                ink = rendered_ink(page, sample["source_ink_bbox"], SOURCE_SIZE, TARGET_SIZE)
                if ink is None:
                    raise SystemExit("no rendered ink for " + sample["id"])
                scale_y = SOURCE_SIZE[1] / TARGET_SIZE[1]
                scale_x = SOURCE_SIZE[0] / TARGET_SIZE[0]
                source = sample["source_ink_bbox"]
                sample["rendered_ink_bbox"] = ink
                sample["dy_source_px"] = round(source[1] - ink[1] * scale_y, 4)
                sample["dx_source_px"] = round(source[0] - ink[0] * scale_x, 4)
                measured.append(sample)
    curves = []
    for family, bold in sorted({(row["family"], row["bold"]) for row in measured}):
        rows = [row for row in measured if row["family"] == family and row["bold"] == bold]
        sizes = np.array([row["size_px"] for row in rows], float)
        deltas = np.array([row["dy_source_px"] for row in rows], float)
        slope, intercept = np.polyfit(sizes, deltas, 1)
        residual = deltas - (slope * sizes + intercept)
        sha = rows[0]["font_sha256"]
        curves.append({
            "family": family, "bold": bold, "dx_source_px": 0,
            "dy_slope": round(float(slope), 6), "dy_intercept": round(float(intercept), 6),
            "size_range_px": [int(sizes.min()), int(sizes.max())],
            "initial_residual_max_px": round(float(np.max(np.abs(residual))), 4),
            "font_sha256": sha,
            "measured_dx_mean_px": round(float(np.mean([row["dx_source_px"] for row in rows])), 4),
        })
    version = args.renderer_version
    if version is None:
        doctor = python_json([node_executable(), str(ROOT / "scripts/main.mjs"), "doctor"])
        version = doctor["artifact_version"]
    max_residual = max(curve["initial_residual_max_px"] for curve in curves)
    install_ready = len(measured) >= 20 and max_residual <= 2.0
    profile = {"purpose": "Renderer-specific PIL-origin to final-PPTX text-origin calibration",
               "renderer": "Artifact Tool active Windows runtime",
               "source_canvas": list(SOURCE_SIZE), "target_canvas": list(TARGET_SIZE),
               "curves": curves,
               "profile_id": "artifact-tool-windows-cjk-origin-%s-candidate" % version,
               "artifact_tool_version": version,
               "calibration_method": "Probe deck compiled by this project, reimported preview, "
                                     "ink-origin delta against PIL source; vertical correction only.",
               "validated_samples": len(measured),
               "validation": {"sizes_px": list(SIZES), "scripts": sorted(SCRIPTS),
                              "mean_absolute_dy_source_px": round(float(np.mean(
                                  [abs(row["dy_source_px"]) for row in measured])), 4),
                              "max_absolute_dy_source_px": round(float(np.max(
                                  [abs(row["dy_source_px"]) for row in measured])), 4),
                              "install_ready": install_ready,
                              "quality_gate": {"max_curve_residual_px": max_residual,
                                               "limit_px": 2.0,
                                               "minimum_samples": 20}},
               "font_identification_verified": False, "powerPoint_checked": False,
               "wps_checked": False, "human_acceptance": False,
               "install": "review this candidate; do not replace the bundled profile automatically"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.report:
        report = {"schema": "RendererOriginProbe/1", "artifact_tool_version": version,
                  "work": str(work), "deck": composed["pptx"], "samples": measured,
                  "human_acceptance": False}
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps({"candidate": str(output), "samples": len(measured),
                      "curves": len(curves), "renderer": version}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
