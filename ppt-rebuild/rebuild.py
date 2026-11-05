"""ppt-rebuild driver.

Subcommands map to the plan's modules:
  layers   M1  parse the LayerD pack                 -> outputs/scenes/<page>_layers.json
  ocr      M2  text detection + line grouping        -> outputs/scenes/<page>_ocr.json
  mask     M3  stroke mask + layered erase           -> outputs/scenes/<page>_mask.json
  scene    M5  build the scene graph                 -> outputs/scenes/<page>_scene.json
  export   M5  write the PPTX
  eval     M6  render + MAE/PSNR/SSIM + diff maps
  all      run the chain for one page
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs"


def _scene_path(page: str, kind: str) -> Path:
    target = OUT / "scenes"
    target.mkdir(parents=True, exist_ok=True)
    return target / f"{page}_{kind}.json"


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {path.relative_to(ROOT)}")


def cmd_layers(args) -> int:
    from layerd_loader import load_page, to_record, verify

    page = load_page(Path(args.pages_root) / args.page)
    record = to_record(page)
    record["recomposition"] = verify(page)
    print(json.dumps({k: record[k] for k in ("page", "image_size", "layer_order")}, ensure_ascii=False))
    print(json.dumps(record["recomposition"], ensure_ascii=False))
    for layer in record["layers"]:
        print(f"  z={layer['z']} {layer['file']:<14} role={layer['role']:<11} "
              f"regions={layer['region_count']:<3} alpha_mode={layer['alpha_mode']}")
    _write(_scene_path(args.page, "layers"), record)
    mae = record["recomposition"].get("mae")
    if mae is not None and mae > args.tolerance:
        print(f"  !! 重组 MAE {mae} 超过容差 {args.tolerance}", file=sys.stderr)
        return 2
    return 0


def cmd_ocr(args) -> int:
    from ocr_engine import detect, group_lines, to_record
    from PIL import Image

    page_dir = Path(args.pages_root) / args.page
    source = page_dir / "source.png"
    with Image.open(source) as image:
        size = image.size
    lines = detect(source, min_score=args.min_score)
    boxes = group_lines(lines, max_gap_ratio=args.max_gap)
    record = to_record(lines, boxes, size)
    print(json.dumps({k: record[k] for k in ("line_count", "box_count", "engine")}, ensure_ascii=False))
    for box in boxes[:6]:
        print(f"  {box['id']} {box['bbox']} lines={box['line_count']} {box['text'][:28]!r}")
    _write(_scene_path(args.page, "ocr"), record)
    return 0


def _load_scene(page: str, kind: str) -> dict:
    path = _scene_path(page, kind)
    if not path.exists():
        raise SystemExit(f"缺少 {path.name}，请先运行对应步骤")
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_fonts(args) -> int:
    from font_matcher import fit_box, vote_family
    from PIL import Image
    import numpy as np

    ocr = _load_scene(args.page, "ocr")
    source = Path(args.pages_root) / args.page / "source.png"
    with Image.open(source) as image:
        rgb = np.array(image.convert("RGB"))
    fits = [fit_box(rgb, box, size_range=(args.min_pt, args.max_pt, 0.5)) for box in ocr["boxes"]]
    vote = vote_family(fits)
    sizes = [fit["size_pt"] for fit in fits if fit.get("size_pt")]
    print(json.dumps({
        "boxes": len(fits),
        "family_vote": vote,
        "size_pt_distinct": len(set(round(s, 1) for s in sizes)),
        "size_pt_range": [round(min(sizes), 1), round(max(sizes), 1)] if sizes else None,
        "size_insufficient": sum(1 for fit in fits if not fit.get("size_ok")),
        "family_dissent": sum(1 for fit in fits if fit.get("family_dissent")),
    }, ensure_ascii=False))
    for fit in fits[:6]:
        print(f"  {fit['box']} {fit.get('size_pt')}pt -> {fit.get('font_family_final')} "
              f"bold={fit.get('bold')} fit={fit.get('fit_score')} {str(fit.get('text'))[:20]!r}")
    _write(_scene_path(args.page, "fonts"),
           {"schema": "FontFit/1", "image_size": ocr["image_size"], "dpi": 125.4,
            "family_vote": vote, "fits": fits})
    return 0


def cmd_mask(args) -> int:
    import numpy as np
    from PIL import Image

    from layerd_loader import load_page
    from text_mask import (below_coverage, composite, describe, erase_by_alpha, erase_by_inpaint,
                           grow, inpaint_rgb, layer_coverage, mask_ratio_ok, owning_layer, remote_inpaint,
                           residual_contrast, stroke_mask, topmost_opaque_below)

    page = load_page(Path(args.pages_root) / args.page)
    ocr = _load_scene(args.page, "ocr")
    working = {layer.z: np.array(Image.open(layer.path).convert("RGBA")) for layer in page.layers}
    alphas = [(z, rgba[..., 3]) for z, rgba in working.items()]
    source = np.array(Image.open(page.source_path).convert("RGB"))

    union_mask = np.zeros(page.size[::-1], bool)
    boxes, summary = [], {"erased_alpha": 0, "erased_inpaint": 0, "repaired_lama": 0,
                          "repaired_local": 0, "refused": 0, "keep_raster": 0}
    for box in ocr["boxes"]:
        result = stroke_mask(source, box["bbox"], tolerance=args.tolerance)
        if result is None:
            boxes.append({"box": box["id"], "status": "no_ink"}); summary["refused"] += 1; continue
        ok, reason = mask_ratio_ok(result, low=args.ratio_min, high=args.ratio_max)
        if not ok:
            # 计划 §8.1：宁可交给审核，也不整框擦
            boxes.append({"box": box["id"], "status": f"refused:{reason}", "mask": describe(result)})
            summary["refused"] += 1
            continue
        owner_z, evidence = owning_layer(result.mask, alphas)
        if owner_z is None:
            boxes.append({"box": box["id"], "status": "refused:no_owner", "mask": describe(result),
                          "layer_evidence": evidence})
            summary["refused"] += 1
            continue
        repair_mask = grow(result.mask, args.repair_grow)
        union_mask |= repair_mask
        below = below_coverage(repair_mask, alphas, owner_z)
        mode = "alpha" if below > args.free_erase_coverage else "inpaint"
        before = composite(list(working.items()), page.size)
        residual_before = residual_contrast(before, repair_mask, tolerance=args.residual_tolerance)
        if mode == "alpha":
            working[owner_z] = erase_by_alpha(working[owner_z], repair_mask)
        else:
            working[owner_z] = erase_by_inpaint(working[owner_z], result.mask)
        after = composite(list(working.items()), page.size)
        residual_after = residual_contrast(after, repair_mask, tolerance=args.residual_tolerance)
        status = f"erased_{mode}"
        summary[f"erased_{mode}"] += 1

        # The layer below almost always keeps a faint impression of the glyphs (the
        # coarse matte does not reach the halo), so the hole is filled unconditionally:
        # waiting for a residual threshold lets faint ghosts through.
        repaired_layer = None
        fix_z = topmost_opaque_below(list(working.items()), repair_mask, owner_z)
        if fix_z is not None:
            working[fix_z] = erase_by_inpaint(working[fix_z], repair_mask, radius=args.repair_radius)
            after = composite(list(working.items()), page.size)
            residual_after = residual_contrast(after, repair_mask, tolerance=args.residual_tolerance)
            repaired_layer = fix_z
            status = "erased_and_repaired"
            summary["repaired_local"] += 1
        if residual_after > args.ghost_threshold and args.inpaint_url:
            repaired = remote_inpaint(after, result.mask, args.inpaint_url)
            if repaired.shape == after.shape:
                fix_z = topmost_opaque_below(list(working.items()), result.mask, owner_z)
                if fix_z is not None:
                    target = working[fix_z].copy()
                    target[result.mask, :3] = repaired[result.mask]
                    working[fix_z] = target
                    after = composite(list(working.items()), page.size)
                residual_after = residual_contrast(after, result.mask)
                status = "repaired_lama"
                summary["repaired_lama"] += 1
        if residual_after > args.ghost_threshold:
            # 计划 §M3：擦不干净就回退为图片对象，绝不带着字影出片
            summary["keep_raster"] += 1
            status = f"keep_raster:{residual_after:.2f}"
        boxes.append({
            "box": box["id"], "text": box["text"], "status": status, "mask": describe(result),
            "owner_layer": owner_z, "below_coverage": round(below, 3), "erase_mode": mode,
            "repaired_layer": repaired_layer,
            "residual_before": round(residual_before, 3), "residual_after": round(residual_after, 3),
            "layer_evidence": evidence,
        })

    workdir = ROOT / "workdir" / args.page
    erased_dir = workdir / "layers-erased"
    erased_dir.mkdir(parents=True, exist_ok=True)
    for layer in page.layers:
        Image.fromarray(working[layer.z]).save(erased_dir / layer.file)
    plate = composite(list(working.items()), page.size)
    # Repair the flattened page: the plate is what the exporter ships, and a layer
    # left with a halo would still be visible through every transparent layer above it.
    plate = inpaint_rgb(plate, union_mask, radius=args.repair_radius)
    Image.fromarray(plate).save(workdir / "plate.png")
    for entry in boxes:
        result = entry.get("mask")
        if not result:
            continue
        x, y, w, h = result["bbox"]
        m = np.zeros(plate.shape[:2], bool)
        m[y:y + h, x:x + w] = True
        entry["plate_residual"] = round(residual_contrast(plate, grow(m, args.repair_grow),
                                                          tolerance=args.residual_tolerance), 3)

    summary["plate_residual_max"] = round(max((b.get("plate_residual") or 0) for b in boxes), 3)
    record = {"schema": "TextErase/1", "page": args.page, "image_size": list(page.size),
              "boxes": boxes, "summary": summary,
              "outputs": {"layers": str(erased_dir.relative_to(ROOT)), "plate": str((workdir / 'plate.png').relative_to(ROOT))}}
    print(json.dumps(summary, ensure_ascii=False))
    statuses: dict[str, int] = {}
    for entry in boxes:
        head = entry["status"].split(":")[0]
        statuses[head] = statuses.get(head, 0) + 1
    print(json.dumps({"statuses": statuses}, ensure_ascii=False))
    _write(_scene_path(args.page, "mask"), record)
    return 0


def cmd_scene(args) -> int:
    from scene_builder import build, validate

    page_dir = Path(args.pages_root) / args.page
    workdir = ROOT / "workdir" / args.page
    scene = build(
        args.page,
        ocr_path=_scene_path(args.page, "ocr"),
        fonts_path=_scene_path(args.page, "fonts"),
        mask_path=_scene_path(args.page, "mask"),
        plate_path=workdir / "plate.png",
        source_path=page_dir / "source.png",
        workdir=workdir,
    )
    problems = validate(scene)
    print(json.dumps({"counts": scene["counts"], "problems": problems}, ensure_ascii=False))
    _write(_scene_path(args.page, "scene"), scene)
    return 0 if not problems else 4


def cmd_export(args) -> int:
    return cmd_run(args)


def cmd_run(args) -> int:
    from rebuild_pipeline import run_page

    review = args.review
    default_review = ROOT / "configs" / "reviews" / f"{args.page[:2]}.json"
    if review is None and default_review.exists():
        review = default_review
    result = run_page(args.pages_root, args.page, args.output, review_file=review,
                      ocr_cache=args.ocr_cache, timeout=args.timeout,
                      probes=not args.skip_probes, align=not args.skip_alignment)
    print(json.dumps({k: result[k] for k in ("directory", "pptx", "preview", "counts", "status")}, ensure_ascii=False))
    return 0


def cmd_batch(args) -> int:
    from rebuild_pipeline import batch

    result = batch(args.pages_root, args.output, reviews=args.reviews, pages=args.pages,
                   resume=args.resume, timeout=args.timeout, probes=not args.skip_probes)
    print(json.dumps({"directory": result["directory"], "pages": len(result["pages"]),
                      "failures": result["failures"], "deck": result.get("deck")}, ensure_ascii=False))
    return 2 if result["failures"] else 0


def cmd_baseline(args) -> int:
    from rebuild_pipeline import baseline
    from slideir_adapter import write_json

    result = baseline(args.pages_root)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def cmd_check_edits(args) -> int:
    from rebuild_pipeline import verify_deliveries

    result = verify_deliveries(args.input, args.output, timeout=args.timeout)
    print(json.dumps(result, ensure_ascii=False))
    return 2 if result["status"] == "FAIL" else 0


def cmd_repair_text(args) -> int:
    from text_repair import repair_batch

    result = repair_batch(args.input,args.output,pages=args.pages,reviews=args.reviews,
                          inpaint_url=args.inpaint_url,timeout=args.timeout)
    print(json.dumps({"directory":result["directory"],"pages":len(result["pages"]),"deck":result["deck"]},ensure_ascii=False))
    return 2 if any(page["coverage"]["pending_paragraphs"] for page in result["pages"]) else 0


def cmd_rebuild_surfaces(args) -> int:
    from surface_rebuild import rebuild_batch

    result = rebuild_batch(args.input, args.output, args.reviews, args.pages, args.resume, args.timeout)
    print(json.dumps({"directory": result["directory"], "pages": len(result["pages"]),
                      "failures": result["failures"]}, ensure_ascii=False))
    return 2 if result["failures"] else 0


def cmd_eval(args) -> int:
    from visual_eval import evaluate

    scene = _load_scene(args.page, "scene")
    page_dir = Path(args.pages_root) / args.page
    render = Path(args.render)
    out = ROOT / "outputs" / "preview" / args.page
    result = evaluate(scene, page_dir / "source.png", render, out_dir=out)
    print(json.dumps({"visual": result["visual"], "text": {k: result["text"][k] for k in
                     ("coverage", "mae_on_text_pixels", "mae_on_non_text_pixels")},
                     "structure": {"objects": result["structure"]["objects"],
                                   "over_10": result["structure"]["over_10"][:6]}}, ensure_ascii=False))
    _write(_scene_path(args.page, "eval"), result)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rebuild")
    parser.add_argument("--pages-root", default=str(ROOT.parent / "outputs" / "layerd-review-20261104"))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("layers", help="M1 layer parse")
    p.add_argument("page")
    p.add_argument("--tolerance", type=float, default=3.0)
    p.set_defaults(func=cmd_layers)

    p = sub.add_parser("ocr", help="M2 text detection and line grouping")
    p.add_argument("page")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--max-gap", type=float, default=0.55)
    p.set_defaults(func=cmd_ocr)

    p = sub.add_parser("fonts", help="M2 font size/family/colour by reverse rendering")
    p.add_argument("page")
    p.add_argument("--min-pt", type=float, default=6.0)
    p.add_argument("--max-pt", type=float, default=72.0)
    p.set_defaults(func=cmd_fonts)

    p = sub.add_parser("mask", help="M3 stroke mask and layered erase")
    p.add_argument("page")
    p.add_argument("--tolerance", type=int, default=40)
    p.add_argument("--ratio-min", type=float, default=0.15)
    p.add_argument("--ratio-max", type=float, default=0.85)
    p.add_argument("--free-erase-coverage", type=float, default=0.98)
    p.add_argument("--ghost-threshold", type=float, default=0.05)
    p.add_argument("--repair-radius", type=int, default=6)
    p.add_argument("--repair-grow", type=int, default=3)
    p.add_argument("--residual-tolerance", type=int, default=8)
    p.add_argument("--inpaint-url", default="")
    p.set_defaults(func=cmd_mask)

    p = sub.add_parser("scene", help="M5 build the scene graph")
    p.add_argument("page")
    p.set_defaults(func=cmd_scene)

    for command in ("run", "all", "export"):
        p = sub.add_parser(command, help="three-branch reconstruction through fixed SlideIR compiler")
        p.add_argument("page")
        p.add_argument("--review", help="source-hash-bound candidate and text review JSON")
        p.add_argument("--ocr-cache", help="source-hash-bound OCR JSON")
        p.add_argument("--output", default=str(OUT / "rebuild"))
        p.add_argument("--timeout", type=int, default=600)
        p.add_argument("--skip-probes", action="store_true")
        p.add_argument("--skip-alignment", action="store_true")
        p.set_defaults(func=cmd_run)

    p = sub.add_parser("batch", help="versioned multi-page reconstruction and compose")
    p.add_argument("--pages", nargs="+", help="page prefixes, in delivery order")
    p.add_argument("--reviews", default=str(ROOT / "configs" / "reviews"))
    p.add_argument("--output", default=str(OUT / "rebuild"))
    p.add_argument("--resume", help="previous batch.json or progress JSON")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--skip-probes", action="store_true")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("baseline", help="read-only validation of all saved LayerD packs")
    p.add_argument("--output", help="new baseline receipt path")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("check-edits", help="direct final-PPTX deletion and native-trial text replacement")
    p.add_argument("input", help="delivery.json or batch.json")
    p.add_argument("--output", default=str(OUT / "rebuild"))
    p.add_argument("--timeout", type=int, default=600)
    p.set_defaults(func=cmd_check_edits)

    p = sub.add_parser("repair-text", help="text-first inpainting and paragraph reconstruction")
    p.add_argument("input", help="source-backed LayerDPPT batch.json")
    p.add_argument("--pages", nargs="+")
    p.add_argument("--reviews",default=str(ROOT / "configs" / "text-repair-reviews"))
    p.add_argument("--output",default=str(OUT / "text-repair"))
    p.add_argument("--inpaint-url",default="http://127.0.0.1:18842/inpaint")
    p.add_argument("--timeout",type=int,default=600)
    p.set_defaults(func=cmd_repair_text)

    p = sub.add_parser("rebuild-surfaces", help="native card backgrounds with bounded masks and source ownership")
    p.add_argument("input", help="frozen text-repair batch.json")
    p.add_argument("--pages", nargs="+")
    p.add_argument("--reviews", default=str(ROOT / "configs/surface-reviews"))
    p.add_argument("--output", default=str(OUT / "surfaces"))
    p.add_argument("--resume", help="prior surface batch or progress JSON with verified artifact hashes")
    p.add_argument("--timeout", type=int, default=600)
    p.set_defaults(func=cmd_rebuild_surfaces)

    p = sub.add_parser("eval", help="M6 render comparison")
    p.add_argument("page")
    p.add_argument("--render", required=True, help="PNG rendered from the exported PPTX")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    print(f"[{args.command}] page={getattr(args, 'page', 'batch')}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
