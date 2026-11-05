"""Build a VisionAnalysis-shaped input for the pilot from the product deck's own text boxes.

This is not OCR. It reuses the text lines and boxes that already exist in the
reference PPTX so the pilot isolates planner quality from OCR quality; the CV
element candidates are still computed from the real page image.
"""
import hashlib
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

EMU_PER_PX = 7288
pptx_path, image_path, out_path, slide_no = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

image = Path(image_path)
sha = hashlib.sha256(image.read_bytes()).hexdigest()

from PIL import Image
with Image.open(image) as im:
    width, height = im.size

prs = Presentation(pptx_path)
slide = prs.slides[slide_no - 1]
objects = []
n = 0
for shape in slide.shapes:
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        continue
    if not (shape.has_text_frame and shape.text_frame.text.strip()):
        continue
    n += 1
    x = round(shape.left / EMU_PER_PX, 2)
    y = round(shape.top / EMU_PER_PX, 2)
    w = round(shape.width / EMU_PER_PX, 2)
    h = round(shape.height / EMU_PER_PX, 2)
    objects.append({
        "id": f"t{n:03d}",
        "kind": "text",
        "text": shape.text_frame.text.strip(),
        "bbox": [max(0.0, x), max(0.0, y), max(1.0, w), max(1.0, h)],
    })

analysis = {
    "schema": "VisionAnalysis",
    "version": "1.0",
    "image_size": [width, height],
    "coordinate_space": "exif_transposed_source_pixels",
    "inputs": {"original": {"sha256": sha, "original_sha256": sha}},
    "objects": objects,
    "stages": [],
}
Path(out_path).write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"image": str(image), "sha256": sha[:16], "size": [width, height],
                  "text_objects": len(objects), "out": out_path}, ensure_ascii=False, indent=2))
