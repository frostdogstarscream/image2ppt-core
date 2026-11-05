"""Per-slide raster budget + how much of each page the de-texting actually repainted."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

prs = Presentation("product.pptx")
W, H = prs.slide_width, prs.slide_height
src_files = sorted(Path("source").glob("*.png")) if any(Path(".").glob("source")) else sorted(Path("source").glob("*.png"))
media_dir = Path("pptx-extract") / "ppt" / "media"

rows = []
for index, slide in enumerate(prs.slides, start=1):
    detail_area = 0.0
    detail_count = 0
    bg = None
    for shape in slide.shapes:
        if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
            continue
        frac = (shape.width * shape.height) / (W * H)
        if frac >= 0.95:
            bg = shape.image.blob
        else:
            detail_area += frac
            detail_count += 1

    src = Image.open(src_files[index - 1]).convert("RGB")
    entry = {"slide": index, "detail_crops": detail_count,
             "detail_crop_area_fraction": round(detail_area, 4)}
    if bg is not None:
        import io
        with Image.open(io.BytesIO(bg)) as im:
            background = im.convert("RGB").resize(src.size)
        a = np.asarray(src, dtype=np.int16)
        b = np.asarray(background, dtype=np.int16)
        diff = np.abs(a - b).mean(axis=2)
        entry["background_vs_source_mean"] = round(float(diff.mean()), 2)
        entry["repainted_fraction_gt30"] = round(float((diff > 30).mean()), 4)
        entry["repainted_fraction_gt80"] = round(float((diff > 80).mean()), 4)
    rows.append(entry)

print(json.dumps(rows, ensure_ascii=False, indent=2))
