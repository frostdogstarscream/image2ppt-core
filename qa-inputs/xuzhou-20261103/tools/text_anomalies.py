"""Flag traditional-character slips and odd spacing in native text (OCR error hunt)."""
import json
import re
from pathlib import Path

TRAD = set("與為會於後個這裡說麼應時數據務資統網條關驗檢測營銷產專級區園華農醫衛環境險風該當開發實設備標準選項計劃總體業團隊規執進機構組織職責權審優調評銜處屬單雙萬億線絡")

audit = json.loads(Path("audit-structure.json").read_text(encoding="utf-8"))
lines = []
for slide in audit["slides"]:
    for text in slide["native_text"].split("\n"):
        bad = [c for c in text if c in TRAD]
        if bad:
            lines.append(f"- slide {slide['slide']}: {text}   <<< {' '.join(bad)}")

space = []
for slide in audit["slides"]:
    for text in slide["native_text"].split("\n"):
        if re.search(r"[\u4e00-\u9fff] {1,}[\u4e00-\u9fff]|、 |[（(] | [5-9]G", text):
            space.append(f"- slide {slide['slide']}: {text}")

out = ["# 繁体/异体字命中", "", *lines, "", "# 可疑空格", "", *space]
Path("text-anomalies.md").write_text("\n".join(out), encoding="utf-8")
print("hits", len(lines), "spacing", len(space))
