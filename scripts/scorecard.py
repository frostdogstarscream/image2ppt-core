"""Offline page scorecard for a delivered PPTX.

Usage:
  python scripts/scorecard.py --source page.png --pptx result.pptx [--render office-preview.png]
         [--ocr ocr.json | --ocr-manifest .runtime/ocr-models.json [--ocr-python-path DIR]]
         [--slide 1] [--out scorecard.json] [--owner-preview owner.png]

Measurements only. No model calls; local OCR is optional and offline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'services/product'))

from product import scorecard  # noqa: E402


def _ocr(args, cache: Path) -> dict | None:
    if args.ocr:
        return json.loads(Path(args.ocr).read_text(encoding='utf-8'))
    if not args.ocr_manifest:
        return None
    if args.ocr_python_path:
        sys.path.insert(0, str(Path(args.ocr_python_path).resolve()))
    from product.parallel_branches import local_ocr, snapshot
    snap = snapshot(args.source, cache / 'snapshot')
    return local_ocr(snap, cache, args.ocr_manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True)
    parser.add_argument('--pptx', required=True)
    parser.add_argument('--render')
    parser.add_argument('--ocr')
    parser.add_argument('--ocr-manifest')
    parser.add_argument('--ocr-python-path')
    parser.add_argument('--slide', type=int, default=1)
    parser.add_argument('--out')
    parser.add_argument('--owner-preview')
    parser.add_argument('--cache', default=str(ROOT / 'outputs/scorecard-cache'))
    args = parser.parse_args()
    ocr = _ocr(args, Path(args.cache))
    result = scorecard.score_page(args.source, args.pptx, render_path=args.render, ocr=ocr, slide=args.slide)
    if args.owner_preview:
        scorecard.save_owner_preview(args.source, args.pptx, args.owner_preview, args.slide)
        result['owner_preview'] = args.owner_preview
    if args.out:
        scorecard.write_json(args.out, result)
    summary = {k: result[k] for k in ('object_counts', 'visible_area_fraction', 'content_fraction',
                                      'native_editable_content_fraction', 'editable_content_fraction',
                                      'movable_raster_content_fraction', 'duplicate_raster_content_fraction',
                                      'raster_plate_content_fraction', 'native_content_fraction')}
    for k in ('fidelity', 'text'):
        if k in result:
            summary[k] = {kk: vv for kk, vv in result[k].items() if kk != 'unmatched'}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
