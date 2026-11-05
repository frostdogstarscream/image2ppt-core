"""Map every picture on each slide to its media file, pixel size and slide coverage."""
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


def walk(shapes):
    for shape in shapes:
        yield shape
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from walk(shape.shapes)


prs = Presentation(sys.argv[1])
W, H = prs.slide_width, prs.slide_height
extract = Path(sys.argv[2])
media_dir = extract / "ppt" / "media"
index_by_sha = {hashlib.sha256(f.read_bytes()).hexdigest(): f.name for f in media_dir.iterdir()}

rows = []
for slide_no, slide in enumerate(prs.slides, start=1):
    for shape in walk(slide.shapes):
        if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
            continue
        blob = shape.image.blob
        media = index_by_sha.get(hashlib.sha256(blob).hexdigest())
        px = None
        if media:
            with Image.open(media_dir / media) as im:
                px = f"{im.mode} {im.width}x{im.height}"
        frac = (shape.width * shape.height) / (W * H) if shape.width and shape.height else 0
        rows.append({
            "slide": slide_no,
            "shape": shape.name,
            "media": media,
            "px": px,
            "fraction_of_slide": round(frac, 4),
            "bytes": len(blob),
        })

print(json.dumps({
    "pictures": len(rows),
    "full_bleed": [r for r in rows if r["fraction_of_slide"] >= 0.95],
    "top12_by_area": sorted(rows, key=lambda r: -r["fraction_of_slide"])[:12],
}, ensure_ascii=False, indent=2))
