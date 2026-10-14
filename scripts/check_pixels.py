"""Meaningful synthetic checks for residue in pixels of actual PPTX previews."""
import json
import sys
from pathlib import Path
from PIL import Image, ImageChops, ImageStat


def mean_white_difference(file, region):
    with Image.open(file) as image:
        crop = image.convert("RGB").crop(region)
        return sum(ImageStat.Stat(ImageChops.difference(crop, Image.new("RGB", crop.size, "white"))).mean) / 3


cases = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checks = []
for case in cases:
    region = None
    if case["action"] == "delete":
        region = (72, 52, 1172, 120)
    elif case["node"] == "delivery-card":
        region = (72, 250, 99, 520)
    elif case["node"] == "visual-module":
        region = (672, 230, 693, 526)
    if region:
        mean = mean_white_difference(case["preview"], region)
        if mean > 1:
            raise ValueError(f"{case['node']} 原位置有残留，平均像素差 {mean}")
        checks.append({"node": case["node"], "region": region, "mean_difference_from_white": mean, "passed": True})
with Image.open(sys.argv[2]) as source, Image.open(sys.argv[3]) as output:
    if source.size != output.size:
        raise ValueError("源图和实际文件预览尺寸不同。")
    mean = sum(ImageStat.Stat(ImageChops.difference(source.convert("RGB"), output.convert("RGB"))).mean) / 3
    checks.append({"synthetic_source_preview_mean_pixel_difference": mean,
                   "note": "仅记录夹具回渲染差异，不作为真实图片准确率。"})
print(json.dumps(checks, ensure_ascii=False, indent=2))

