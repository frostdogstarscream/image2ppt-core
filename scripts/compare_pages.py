"""Per-page source-vs-render comparison sheets for a composed deck.

Left half is the source page, right half is the PNG re-imported from the final
PPTX. The mean pixel difference is a rough locator for drift, not an accuracy
score: a faithful rebuild still differs wherever text is re-set natively.
"""
import json
import sys
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageStat

cases = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
report = []
for case in cases:
    with Image.open(case["source"]) as s, Image.open(case["preview"]) as p:
        source = s.convert("RGB")
        rendered = p.convert("RGB")
        if rendered.size != source.size:
            rendered = rendered.resize(source.size)
        difference = sum(ImageStat.Stat(ImageChops.difference(source, rendered)).mean) / 3
        gap, header = 16, 28
        sheet = Image.new("RGB", (source.width * 2 + gap, source.height + header), "white")
        sheet.paste(source, (0, header))
        sheet.paste(rendered, (source.width + gap, header))
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 9), "source %s" % case.get("page", ""), fill="black")
        draw.text((source.width + gap + 4, 9), "rendered from final PPTX", fill="black")
        name = "page-%03d.png" % case["slide"]
        sheet.save(out / name)
    report.append({"slide": case["slide"], "page": case.get("page"),
                   "mean_pixel_difference": difference, "sheet": name})
print(json.dumps({"pages": report}, ensure_ascii=False, indent=2))
