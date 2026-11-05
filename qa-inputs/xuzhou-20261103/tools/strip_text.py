"""Ghost-text probe: strip every native text box from selected slides, keep rasters and
shapes, re-save as a new file. Re-rendering the copy shows whether the background image
still carries the source text underneath the editable text.
"""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

src, dst = sys.argv[1], sys.argv[2]
pages = {int(x) for x in sys.argv[3].split(",")}

prs = Presentation(src)
removed = {}
for index, slide in enumerate(prs.slides, start=1):
    if index not in pages:
        continue
    keep = []
    gone = 0
    for shape in list(slide.shapes):
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            keep.append(shape)
            continue
        try:
            has_text = shape.has_text_frame and shape.text_frame.text.strip()
        except Exception:
            has_text = False
        if has_text:
            shape._element.getparent().remove(shape._element)
            gone += 1
        else:
            keep.append(shape)
    removed[index] = gone
prs.save(dst)
print({"saved": str(Path(dst).resolve()), "removed_text_shapes": removed})
