"""Reverse-engineer the generator from the OOXML it left behind.

Reads slide XML directly (no model) and reports:
  * the descr taxonomy the builder writes on pictures
  * geometry classes used for native shapes
  * text-box paragraph structure (how many lines per box)
  * the pixel -> EMU factor and whether all coordinates are integer pixels
  * colour / bold vocabulary actually recovered
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
root = Path("pptx-extract")
slides = sorted(root.glob("ppt/slides/slide*.xml"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))

from lxml import etree

descr = Counter()
geom = Counter()
desc_by_slide = defaultdict(Counter)
ratios = []
run_colors = Counter()
bold = Counter()
para_counts = Counter()
boxes = 0
paras_total = 0
single_line_boxes = 0
multipara = []

for slide in slides:
    tree = etree.parse(str(slide))
    get = lambda el, tag: el.findall(tag, NS)
    n = int(re.search(r"(\d+)", slide.stem).group(1))
    for pic in get(tree, ".//p:pic"):
        c = pic.find(".//p:cNvPr", NS)
        d = (c.get("descr") or "(none)") if c is not None else "(none)"
        descr[d] += 1
        desc_by_slide[n][d] += 1
        xfrm = pic.find(".//a:xfrm", NS)
        if xfrm is not None:
            off, ext = xfrm.find("a:off", NS), xfrm.find("a:ext", NS)
            for v in (off.get("x"), off.get("y"), ext.get("cx"), ext.get("cy")):
                ratios.append(int(v) / 7288.0)
    for sp in get(tree, ".//p:sp"):
        for g in sp.findall(".//a:prstGeom", NS):
            geom["prst:" + (g.get("prst") or "?")] += 1
        for g in sp.findall(".//a:custGeom", NS):
            geom["custGeom"] += 1
        body = sp.find(".//p:txBody", NS)
        if body is None:
            continue
        txt = "".join(t.text or "" for t in body.findall(".//a:t", NS)).strip()
        if not txt:
            continue
        boxes += 1
        ps = body.findall("a:p", NS)
        paras_total += len(ps)
        para_counts[len(ps)] += 1
        if len(ps) == 1:
            single_line_boxes += 1
        else:
            multipara.append((n, len(ps), txt[:28]))
        for r in body.findall(".//a:r", NS):
            rPr = r.find("a:rPr", NS)
            if rPr is None:
                continue
            bold[rPr.get("b") or "0"] += 1
            for c in rPr.findall(".//a:srgbClr", NS):
                run_colors[c.get("val")] += 1

ratios = sorted(ratios)
integer_share = sum(1 for v in ratios if abs(v - round(v)) < 0.02) / len(ratios)

report = {
    "pictures_descr": dict(descr),
    "descr_by_slide": {k: dict(v) for k, v in sorted(desc_by_slide.items())},
    "shape_geometry": dict(geom),
    "text": {
        "boxes": boxes,
        "paragraphs": paras_total,
        "single_paragraph_boxes": single_line_boxes,
        "multi_paragraph_boxes": boxes - single_line_boxes,
        "paragraph_count_histogram": dict(sorted(para_counts.items())),
    },
    "geometry_mapping": {
        "emu_per_pixel": 7288,
        "share_of_coords_on_integer_pixels": round(integer_share, 4),
    },
    "runs": {"bold": dict(bold), "distinct_colors": len(run_colors), "top_colors": run_colors.most_common(12)},
}
Path("reverse-engineering.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2)[:3000])
print("\nmulti-paragraph examples:", multipara[:6])
