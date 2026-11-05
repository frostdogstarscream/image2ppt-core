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

    args = parser.parse_args(argv)
    print(f"[{args.command}] page={args.page}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
