"""Write source / before / after text crops and fixed-region error deltas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from model_slideir_loop import digest, write_json


def compare(source: Path, before: Path, after: Path, layout: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("refusing to overwrite comparison output")
    report = json.loads(layout.read_text(encoding="utf-8"))
    if digest(source) != report["source_sha256"]:
        raise ValueError("comparison source does not match calibration source")
    output.mkdir(parents=True)
    result = {"source_sha256": digest(source), "before_sha256": digest(before),
              "after_sha256": digest(after), "rows": []}
    with Image.open(source) as src, Image.open(before) as old, Image.open(after) as new:
        src = src.convert("RGB")
        old = old.convert("RGB").resize(src.size, Image.Resampling.BILINEAR)
        new = new.convert("RGB").resize(src.size, Image.Resampling.BILINEAR)
        for row in report["rows"]:
            x, y, w, h = [int(round(v)) for v in row["bbox"]]
            box = (max(0, x - 8), max(0, y - 8), min(src.width, x + w + 8),
                   min(src.height, y + h + 8))
            crops = [image.crop(box) for image in (src, old, new)]
            original = np.asarray(crops[0], dtype=np.int16)
            errors = [round(float(np.abs(original - np.asarray(crop, dtype=np.int16)).mean()), 3)
                      for crop in crops[1:]]
            pair = Image.new("RGB", (crops[0].width * 3, crops[0].height), "white")
            for index, crop in enumerate(crops):
                pair.paste(crop, (index * crop.width, 0))
            filename = row["node_id"] + ".png"
            pair.save(output / filename)
            result["rows"].append({"node_id": row["node_id"], "text": row["text"],
                                   "crop_bbox": list(box), "before_mae": errors[0],
                                   "after_mae": errors[1],
                                   "delta_mae": round(errors[1] - errors[0], 3),
                                   "comparison_file": str(output / filename)})
    result["improved_regions"] = sum(r["delta_mae"] < 0 for r in result["rows"])
    result["regressed_regions"] = sum(r["delta_mae"] > 0 for r in result["rows"])
    write_json(output / "comparison.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "before", "after", "layout", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.source, args.before, args.after, args.layout, args.output),
                     ensure_ascii=False, indent=2))
