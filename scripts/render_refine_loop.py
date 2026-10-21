"""Bounded render-measure-patch loop for high-confidence text nodes.

This is the outer closed loop: compile the calibrated IR, reimport/render the
final PPTX, compare each eligible text node against the source pixels, patch
only its origin and font size within a small budget, compile again, and keep a
patch only when the re-rendered ink actually improves.

It never edits the input IR or the submitted element plan. The bounded output
is a new directory with the patched IR, its authoring sidecar and a report.
"""
import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
INK_THRESHOLD = 30
MIN_FOREGROUND_PIXELS = 12
MIN_PATCH_SHIFT_PX = .5
MIN_SIZE_CHANGE = .01
MIN_INK_GAIN = .2
MAX_NODE_REGRESSION = .02


def node_executable():
    import os
    return os.environ.get("IMAGE2PPT_NODE") or str(
        Path(os.environ["USERPROFILE"]) /
        ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe")


def parse_json_output(text):
    start = text.find("{")
    if start < 0:
        raise RuntimeError("compiler did not return JSON: " + text[-500:])
    return json.loads(text[start:])


def compile_ir(ir_path):
    result = subprocess.run([node_executable(), str(ROOT / "scripts/main.mjs"),
                             "compile", "--input", str(ir_path)],
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    data = parse_json_output(result.stdout)
    return data


def copy_input(ir_path, output):
    ir_path = Path(ir_path).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise SystemExit("refusing to overwrite: " + str(output))
    output.mkdir(parents=True)
    for name in ("source", "assets"):
        source = ir_path.parent / name
        if source.exists():
            shutil.copytree(source, output / name)
    shutil.copyfile(ir_path, output / "slideir.json")
    return output / "slideir.json"


def digest(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_ir(path, ir, original_sha):
    path = Path(path)
    path.write_text(json.dumps(ir, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    authoring = {"tool": "render-refine-loop", "ir_sha256": digest(path),
                 "loop_input_ir_sha256": original_sha,
                 "review_notes": ir.get("review_notes", [])}
    (path.parent / "authoring.json").write_text(
        json.dumps(authoring, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def page_mae(source, preview):
    """Whole-page drift locator; used only as a no-regression guard."""
    rendered = preview.resize(source.size, Image.Resampling.BILINEAR)
    return float(np.abs(np.asarray(source, float) -
                        np.asarray(rendered.convert("RGB"), float)).mean())


def public(metric):
    return {key: value for key, value in metric.items() if not key.startswith("_")}


def ink_mask(crop):
    values = crop.reshape(-1, 3).astype(float)
    border = np.concatenate([values[:crop.shape[1]], values[-crop.shape[1]:],
                             values[::crop.shape[1]], values[crop.shape[1]-1::crop.shape[1]]])
    background = np.median(border, axis=0)
    delta = np.max(np.abs(crop.astype(float) - background), axis=2)
    mask = delta > INK_THRESHOLD
    if mask.sum() < MIN_FOREGROUND_PIXELS:
        return None
    return mask


def measure(source, preview, node):
    source_size = source.size
    target_size = preview.size
    sx = target_size[0] / source_size[0]
    sy = target_size[1] / source_size[1]
    sbox = node.get("source_text_bbox", node["bbox"])
    # Compare the same source region before and after a patch. Cropping the
    # rendered preview around the current node bbox would move the comparison
    # window with the patch and hide the very offset being corrected.
    pad = 12
    sl = max(0, int(sbox[0]) - pad); st = max(0, int(sbox[1]) - pad)
    sr = min(source.width, int(sbox[0] + sbox[2]) + pad)
    sb = min(source.height, int(sbox[1] + sbox[3]) + pad)
    scrop = np.asarray(source.crop((sl, st, sr, sb)).convert("RGB"))
    smask = ink_mask(scrop)
    if smask is None:
        return None
    rl = max(0, int(sl * sx)); rt = max(0, int(st * sy))
    rr = min(preview.width, int(sr * sx)); rb = min(preview.height, int(sb * sy))
    rcrop = np.asarray(preview.crop((rl, rt, rr, rb)).convert("RGB"))
    rmask = ink_mask(rcrop)
    if rmask is None:
        return None
    syy, sxx = np.where(smask); ryy, rxx = np.where(rmask)
    s_cx = sl + float(sxx.mean()); s_cy = st + float(syy.mean())
    r_cx = (rl + float(rxx.mean())) / sx; r_cy = (rt + float(ryy.mean())) / sy
    s_w = float(sxx.max() - sxx.min() + 1); s_h = float(syy.max() - syy.min() + 1)
    r_w = float(rxx.max() - rxx.min() + 1) / sx
    r_h = float(ryy.max() - ryy.min() + 1) / sy
    rendered_resized = np.asarray(
        Image.fromarray(rcrop).resize((scrop.shape[1], scrop.shape[0]),
                                      Image.Resampling.BILINEAR))
    difference = np.abs(scrop.astype(float) - rendered_resized.astype(float))
    return {"source_centroid": [round(s_cx, 3), round(s_cy, 3)],
            "rendered_centroid": [round(r_cx, 3), round(r_cy, 3)],
            "source_ink_size": [round(s_w, 3), round(s_h, 3)],
            "rendered_ink_size": [round(r_w, 3), round(r_h, 3)],
            "ink_mae": round(float(difference[smask].mean()), 4),
            "bbox_mae": round(float(difference.mean()), 4),
            "_source_crop": scrop, "_rendered_crop": rendered_resized,
            "_mask": smask}


def shift_image(image, dx, dy):
    out = np.zeros_like(image)
    height, width = image.shape[:2]
    x0, x1 = max(0, dx), min(width, width + dx)
    y0, y1 = max(0, dy), min(height, height + dy)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = image[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def scale_center(image, factor):
    if abs(factor - 1) < 1e-9:
        return image
    height, width = image.shape[:2]
    scaled = np.asarray(Image.fromarray(
        np.clip(image, 0, 255).astype(np.uint8)).resize(
        (max(1, round(width * factor)), max(1, round(height * factor))),
        Image.Resampling.BILINEAR))
    out = np.zeros_like(image)
    y = (height - scaled.shape[0]) // 2
    x = (width - scaled.shape[1]) // 2
    sy0, sx0 = max(0, -y), max(0, -x)
    dy0, dx0 = max(0, y), max(0, x)
    h = min(scaled.shape[0] - sy0, height - dy0)
    w = min(scaled.shape[1] - sx0, width - dx0)
    if h > 0 and w > 0:
        out[dy0:dy0 + h, dx0:dx0 + w] = scaled[sy0:sy0 + h, sx0:sx0 + w]
    return out


def propose_patch(metric, node, *, max_shift, max_size_change):
    """Search translations/scale in the fixed source region, not by centroid.

    Glyph centroids are dominated by font-shape differences; the direct MAE
    search asks the question the page metric asks, and it is cheap because it
    moves the already-rendered crop before spending another compile.
    """
    if not metric:
        return None
    source = metric["_source_crop"].astype(float)
    rendered = metric["_rendered_crop"].astype(float)
    mask = metric["_mask"]
    base = float(np.abs(source[mask] - rendered[mask]).mean())
    limit = int(round(max_shift))
    shifts = range(-limit, limit + 1)
    factors = [1.0]
    if max_size_change > 0:
        factors = [1 - max_size_change, 1 - max_size_change / 2, 1,
                   1 + max_size_change / 2, 1 + max_size_change]
    best = (base, 0, 0, 1.0)
    for factor in factors:
        scaled = scale_center(rendered, factor)
        for dx in shifts:
            for dy in shifts:
                moved = shift_image(scaled, dx, dy)
                value = float(np.abs(source[mask] - moved[mask]).mean())
                if value < best[0] - 1e-9:
                    best = (value, dx, dy, factor)
    gain = base - best[0]
    if gain < MIN_INK_GAIN:
        return None
    if best[1] == 0 and best[2] == 0 and abs(best[3] - 1) < 1e-9:
        return None
    return {"dx": float(best[1]), "dy": float(best[2]),
            "size_factor": round(best[3], 5),
            "expected_ink_mae": round(best[0], 4),
            "basis": "fixed_region_alignment_search"}


def apply_patch(node, patch, canvas):
    width = min(node["bbox"][2], canvas[0])
    height = min(node["bbox"][3], canvas[1])
    x = min(max(0, node["bbox"][0] + patch["dx"]), canvas[0] - width)
    y = min(max(0, node["bbox"][1] + patch["dy"]), canvas[1] - height)
    node["bbox"] = [x, y, width, height]
    node["font"]["size_pt"] = round(node["font"]["size_pt"] * patch["size_factor"], 4)
    return node


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", required=True, help="calibrated SlideIR")
    parser.add_argument("--calibration", help="calibration.json from the vision service")
    parser.add_argument("--output", required=True, help="new directory for the bounded result")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-shift", type=float, default=2.0, help="source pixels per round")
    parser.add_argument("--max-size-change", type=float, default=.02)
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 3:
        raise SystemExit("rounds must be 1..3")
    ir_path = copy_input(args.ir, args.output)
    original_sha = digest(ir_path)
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    rows = {}
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        rows = {row["id"]: row for row in calibration.get("diagnostics", [])
                if row.get("kind") == "typography" and row.get("applied")}
    eligible = []
    for node in ir["nodes"]:
        if node["type"] != "text":
            continue
        if rows:
            row = rows.get(node["id"])
            if not row or "font" not in row.get("applied_fields", []):
                continue
        elif not node.get("source_text_bbox"):
            continue
        eligible.append(node["id"])
    if not eligible:
        raise SystemExit("no eligible high-confidence text nodes")
    write_ir(ir_path, ir, original_sha)
    source = Image.open(ir_path.parent / ir["source_image"]).convert("RGB")
    canvas = ir["canvas"]["target_px"]
    history = []
    current = ir
    for round_index in range(1, args.rounds + 1):
        compiled = compile_ir(ir_path)
        preview = Image.open(Path(compiled["output_dir"]) / "preview.png").convert("RGB")
        before_page = page_mae(source, preview)
        before = {node_id: measure(source, preview, next(n for n in current["nodes"]
                                                         if n["id"] == node_id))
                  for node_id in eligible}
        patches = {}
        candidate = copy.deepcopy(current)
        for node in candidate["nodes"]:
            if node["id"] not in eligible:
                continue
            patch = propose_patch(before[node["id"]], node,
                                  max_shift=args.max_shift,
                                  max_size_change=args.max_size_change)
            if patch:
                apply_patch(node, patch, canvas)
                patches[node["id"]] = patch
        if not patches:
            history.append({"round": round_index, "patches": 0,
                            "accepted": [], "reverted": [],
                            "reason": "no_patch_within_budget"})
            break
        write_ir(ir_path, candidate, original_sha)
        compiled_after = compile_ir(ir_path)
        preview_after = Image.open(Path(compiled_after["output_dir"]) / "preview.png").convert("RGB")
        after_page = page_mae(source, preview_after)
        after = {node_id: measure(source, preview_after,
                                  next(n for n in candidate["nodes"] if n["id"] == node_id))
                 for node_id in eligible}
        if after_page > before_page + .05:
            write_ir(ir_path, current, original_sha)
            history.append({"round": round_index, "patches": len(patches),
                            "accepted": [], "reverted": [],
                            "page_guard_reverted": True,
                            "page_mae_before": round(before_page, 4),
                            "page_mae_after": round(after_page, 4)})
            break
        accepted, reverted = [], []
        for node in candidate["nodes"]:
            if node["id"] not in patches:
                continue
            node_id = node["id"]
            b, a = before[node_id], after[node_id]
            if not b or not a:
                reverted.append({"id": node_id, "reason": "measurement_lost"})
                original = next(n for n in current["nodes"] if n["id"] == node_id)
                node.update(copy.deepcopy(original)); continue
            gain = b["ink_mae"] - a["ink_mae"]
            regression = (a["ink_mae"] - b["ink_mae"]) / max(b["ink_mae"], .001)
            if gain >= MIN_INK_GAIN and regression <= MAX_NODE_REGRESSION:
                accepted.append({"id": node_id, "patch": patches[node_id],
                                 "before": public(b), "after": public(a)})
            else:
                original = next(n for n in current["nodes"] if n["id"] == node_id)
                node.update(copy.deepcopy(original))
                reverted.append({"id": node_id, "reason": "no_rendered_improvement",
                                 "before_ink_mae": b["ink_mae"],
                                 "after_ink_mae": a["ink_mae"]})
        if not accepted:
            write_ir(ir_path, current, original_sha)
            history.append({"round": round_index, "patches": len(patches),
                            "accepted": [], "reverted": reverted})
            break
        write_ir(ir_path, candidate, original_sha)
        current = candidate
        history.append({"round": round_index, "patches": len(patches),
                        "accepted": accepted, "reverted": reverted,
                        "page_mae_before": round(before_page, 4),
                        "page_mae_after": round(after_page, 4)})
    final = compile_ir(ir_path)
    report = {"schema": "RenderRefineLoop/1", "input_ir": str(Path(args.ir).resolve()),
              "input_ir_sha256": original_sha, "calibration": args.calibration,
              "eligible_nodes": eligible, "rounds_requested": args.rounds,
              "rounds": history, "max_shift_source_px": args.max_shift,
              "max_size_change": args.max_size_change,
              "final_ir_sha256": digest(ir_path), "final_preview": final["preview"],
              "final_pptx": final["pptx"], "final_quality": final["quality"],
              "preview_origin": "reimported_final_pptx",
              "actual_render_verified": False, "human_acceptance": False}
    (ir_path.parent / "render-loop.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(ir_path.parent), "eligible": len(eligible),
                      "rounds": len(history), "final_preview": final["preview"],
                      "report": str(ir_path.parent / "render-loop.json")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
