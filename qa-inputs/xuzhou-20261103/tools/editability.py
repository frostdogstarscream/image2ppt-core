"""Editability details: paragraph fragmentation, alignment, autofit, run fonts."""
import json
import sys
from collections import Counter
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Pt

TERMINAL = "。！？：；…）)】」”’"

prs = Presentation(sys.argv[1])
rows = []
totals = Counter()

for index, slide in enumerate(prs.slides, start=1):
    frag = 0
    boxes = 0
    algn = Counter()
    autofit = Counter()
    wrap = Counter()
    families = Counter()
    brs = 0
    for shape in slide.shapes:
        if not (shape.has_text_frame and shape.text_frame.text.strip()):
            continue
        boxes += 1
        text = shape.text_frame.text.strip()
        if len(text) >= 3 and text[-1] not in TERMINAL:
            frag += 1
        for para in shape.text_frame.paragraphs:
            algn[str(para.alignment)] += 1
            for run in para.runs:
                fam = run.font.name or "(inherit)"
                families[fam] += 1
        body = shape.text_frame._txBody.find(
            "{http://schemas.openxmlformats.org/drawingml/2006/main}bodyPr")
        if body is not None:
            child = [c.tag.split("}")[1] for c in body if c.tag.endswith(("normAutofit", "spAutoFit", "noAutofit"))]
            autofit[child[0] if child else "(none)"] += 1
            wrap[body.get("wrap") or "(default)"] += 1
        brs += len(shape.text_frame._txBody.findall(
            ".//{http://schemas.openxmlformats.org/drawingml/2006/main}br"))
    rows.append({"slide": index, "text_boxes": boxes, "fragment_boxes": frag,
                 "fragment_share": round(frag / boxes, 3) if boxes else 0,
                 "paragraph_alignment": dict(algn), "autofit": dict(autofit),
                 "wrap": dict(wrap), "fonts": dict(families), "soft_breaks": brs})
    totals.update({"boxes": boxes, "fragments": frag})
    for k, v in algn.items():
        totals["algn:" + k] += v
    for k, v in autofit.items():
        totals["autofit:" + k] += v
    for k, v in families.items():
        totals["font:" + k] += v

print(json.dumps({"per_slide": rows, "totals": dict(totals)}, ensure_ascii=False, indent=2))
