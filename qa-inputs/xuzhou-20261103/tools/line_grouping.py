"""How lines are grouped into text boxes, and where a sentence really spans two boxes."""
import json
import re
from collections import Counter
from pathlib import Path

from lxml import etree

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}
TERMINAL = "。！？：；…）)】」”’"
root = Path("pptx-extract")
slides = sorted(root.glob("ppt/slides/slide*.xml"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))

suspected = []
stats = Counter()
for slide in slides:
    n = int(re.search(r"(\d+)", slide.stem).group(1))
    tree = etree.parse(str(slide))
    boxes = []
    for sp in tree.findall(".//p:sp", NS):
        body = sp.find(".//p:txBody", NS)
        if body is None:
            continue
        paras = []
        for p in body.findall("a:p", NS):
            t = "".join(x.text or "" for x in p.findall(".//a:t", NS))
            if t.strip():
                paras.append(t.strip())
        if paras:
            boxes.append(paras)
    for i, box in enumerate(boxes[:-1]):
        last = box[-1]
        nxt = boxes[i + 1][0]
        if last[-1] in TERMINAL:
            stats["ends_terminal"] += 1
            continue
        # a short line with no punctuation at all is a label/title, not a broken sentence
        has_punct = any(c in last for c in "，、；。：")
        if len(last) >= 12 and has_punct:
            stats["sentence_runs_into_next_box"] += 1
            suspected.append((n, last, nxt))
        else:
            stats["label_or_short"] += 1

out = {
    "stats": dict(stats),
    "suspected_sentence_split_across_boxes": [
        {"slide": s, "box_a_tail": a, "box_b_head": b} for s, a, b in suspected
    ],
}
Path("line-grouping.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out["stats"], ensure_ascii=False))
print("suspected splits:", len(suspected))
for s, a, b in suspected[:10]:
    print(f"  p{s}: ...{a[-16:]}  ||  {b[:16]}...")
