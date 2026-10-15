"""Decide per region whether a rebuild earned its place, or a crop did better.

Native text is editable; a crop is pixel-exact. That is a real trade and it does
not resolve the same way everywhere: measuring the 14-page deck, every region
scored 2.3-4.2 as a crop and 8.9-50.1 rebuilt. Rebuilding costs fidelity that
adjusting the geometry does not recover, because the text is re-set in a
different face however well it is positioned.

So a rebuilt region is kept only when it stays close to what the crop cost.
A region over the threshold goes back to a crop, which is why a page can end up
part native and part raster.

  python scripts/fidelity_gate.py cases.json
  python scripts/fidelity_gate.py cases.json --threshold 10

cases.json is a list of {"name", "source", "preview", "bbox": [x,y,w,h]}, where
source is the page image and preview the PNG re-imported from the final PPTX.
"""
import argparse
import json
import sys

from PIL import Image, ImageChops, ImageStat

DEFAULT_THRESHOLD = 10.0


def drift(source, preview, bbox):
    """Mean absolute pixel difference over the region, across the three channels."""
    with Image.open(preview) as rendered:
        page = rendered.convert("RGB")
        if page.size != source.size:
            page = page.resize(source.size, Image.LANCZOS)
        x, y, w, h = bbox
        a = source.crop((x, y, x + w, y + h))
        b = page.crop((x, y, x + w, y + h))
        return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", help="JSON list of regions to judge")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="drift a rebuilt region may reach before it is reverted (default %(default)s)")
    args = parser.parse_args(argv)

    cases = json.loads(open(args.cases, encoding="utf-8").read())
    verdicts = []
    for case in cases:
        with Image.open(case["source"]) as page:
            value = drift(page.convert("RGB"), case["preview"], case["bbox"])
        verdicts.append({"name": case.get("name"), "drift": round(value, 2),
                         "verdict": "keep_native" if value <= args.threshold else "revert_to_crop"})
    verdicts.sort(key=lambda v: -v["drift"])
    for v in verdicts:
        print("%8.2f  %-20s %s" % (v["drift"], v["verdict"], v["name"]))
    kept = sum(1 for v in verdicts if v["verdict"] == "keep_native")
    print("\n%d/%d 保留原生，阈值 %.1f" % (kept, len(verdicts), args.threshold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
