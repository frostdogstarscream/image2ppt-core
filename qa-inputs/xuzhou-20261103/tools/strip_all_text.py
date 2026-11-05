"""Strip every native text box on all 12 slides and render, to prove no text is baked into rasters."""
import sys

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

src, dst = sys.argv[1], sys.argv[2]
prs = Presentation(src)
removed = {}
for index, slide in enumerate(prs.slides, start=1):
    count = 0
    for shape in list(slide.shapes):
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            continue
        try:
            has_text = shape.has_text_frame and shape.text_frame.text.strip()
        except Exception:
            has_text = False
        if has_text:
            shape._element.getparent().remove(shape._element)
            count += 1
    removed[index] = count
prs.save(dst)
print("removed", removed)
