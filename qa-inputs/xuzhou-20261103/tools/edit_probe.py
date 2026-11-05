"""Edit probe: lengthen one title and one paragraph, then re-render.
Shows what an actual user edit does to a box that has wrap="none" and no autofit.
"""
import sys

from pptx import Presentation

src, dst = sys.argv[1], sys.argv[2]
prs = Presentation(src)
edits = []

slide3 = prs.slides[2]
for shape in slide3.shapes:
    if not (shape.has_text_frame and shape.text_frame.text.strip()):
        continue
    text = shape.text_frame.text
    if text.strip() == "园区治理":
        shape.text_frame.paragraphs[0].runs[0].text = "园区治理与安全监管"
        edits.append(("slide3-title", text, "园区治理与安全监管"))
        break

slide1 = prs.slides[0]
for shape in slide1.shapes:
    if not (shape.has_text_frame and shape.text_frame.text.strip()):
        continue
    if shape.text_frame.text.startswith("携手推进徐州高新区"):
        para = shape.text_frame.paragraphs[0]
        para.runs[0].text = "携手推进徐州高新区“AI+”应用与智能化升级，共同探索可持续的建设运营机制。"
        edits.append(("slide1-subtitle", "携手推进…智能化升级", "…共同探索可持续的建设运营机制。"))
        break

prs.save(dst)
print({"saved": dst, "edits": edits})
