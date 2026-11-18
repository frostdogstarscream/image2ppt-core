import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compare(combined, baseline):
    actual = read(combined / "report.json")
    control = read(baseline / "report.json")
    for directory, report in ((combined, actual), (baseline, control)):
        if (
            hashlib.sha256((directory / "two-pages.pptx").read_bytes()).hexdigest()
            != report["sha256"]
        ):
            raise ValueError("Deck does not match the evaluation report")
    comparison = []
    for n in (1, 2):
        a = np.asarray(Image.open(combined / f"page-{n}.png").convert("RGB"))
        b = np.asarray(Image.open(baseline / f"page-{n}.png").convert("RGB"))
        if a.shape != b.shape:
            raise ValueError(f"Slide {n} render dimensions differ")
        error = np.abs(a.astype(np.int16) - b.astype(np.int16))
        comparison.append(
            {
                "page": n,
                "size": [a.shape[1], a.shape[0]],
                "changed_pixels": int(np.any(error, axis=2).sum()),
                "mean_channel_error": float(error.mean()),
                "max_channel_error": int(error.max()),
            }
        )
    with ZipFile(combined / "two-pages.pptx") as p:
        damaged = p.testzip()
        if damaged:
            raise ValueError(f"Invalid ZIP member: {damaged}")
        svgs = [
            n for n in p.namelist() if n.startswith("ppt/media/") and n.endswith(".svg")
        ]
    metadata = {
        "combined": actual["sha256"],
        "baseline": control["sha256"],
        "powerpoint_renders": comparison,
        "embedded_svg_count": len(svgs),
    }
    (combined / "baseline-comparison.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare final PowerPoint renders for the paired experiment"
    )
    parser.add_argument("combined", type=Path)
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()
    compare(args.combined, args.baseline)
