"""Review a source-graphics candidate against receipt-bound actual-file renders."""
import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"services/vision/src"))
from image2ppt_vision.font_compatibility import bound_preview
from image2ppt_vision.source_graphics import compare_pixels, verify
from image2ppt_vision.storage import digest, read_json, write_json

p = argparse.ArgumentParser()
for field in ("folder", "before-folder", "before-deck", "after-deck", "output"):
    p.add_argument("--"+field, required=True)
a = p.parse_args()
folder, before, after, output = map(Path, (a.folder, a.before_deck, a.after_deck, a.output))
if output.exists():
    raise FileExistsError("new review directory required")
ir = read_json(folder/"slideir.json")
report = read_json(folder/"source-graphics.json")
baseline_ir = Path(a.before_folder)/"slideir.json"
if digest(baseline_ir) != report["parent_ir_sha256"]:
    raise ValueError("wrong comparison baseline")
bindings = [
    bound_preview(baseline_ir, before/"deck.pptx", before/"preview/page-001.png"),
    bound_preview(folder/"slideir.json", after/"deck.pptx", after/"preview/page-001.png")]
source = Image.open(folder/ir["source_image"]).convert("RGB").resize(
    tuple(ir["canvas"]["target_px"]), Image.Resampling.LANCZOS)
old, new = [Image.open(f/"preview/page-001.png").convert("RGB") for f in (before, after)]
verification = verify(folder, after/"deck.pptx")
pixels = compare_pixels(source, old, new, report["bars"], ir["canvas"]["source_px"])
output.mkdir(parents=True)
write_json(output/"verification.json", {"structure": verification, "pixels": pixels, "bindings": bindings})
shutil.copyfile(after/"deck.pptx", output/"第05页_源图图形补全_待复核.pptx")
new.save(output/"preview.png")
shutil.copyfile(folder/"module-overlay.png", output/"module-overlay.png")
# Whole-page comparisons plus close crops. Crops are diagnostic views, not slide assets.
comparison = Image.new("RGB", (old.width, old.height*3+90), "white")
draw = ImageDraw.Draw(comparison)
for index, (label, image) in enumerate((("SOURCE", source), ("BEFORE", old), ("AFTER", new))):
    y = index*(old.height+30)
    draw.text((10, y+8), label, fill="black")
    comparison.paste(image, (0, y+30))
comparison.save(output/"source-before-after.png")
regions = pixels["regions"]
canvas = Image.new("RGB", (900, 60*len(regions)+30), "white")
draw = ImageDraw.Draw(canvas)
for col, label in enumerate(("SOURCE", "BEFORE", "AFTER")):
    draw.text((col*300+10, 8), label, fill="black")
for row, region in enumerate(regions):
    x, y, w, h = region["target_roi"]
    for col, image in enumerate((source, old, new)):
        crop = image.crop((x, y, x+w, y+h))
        crop.thumbnail((280, 42))
        canvas.paste(crop, (col*300+10, row*60+38))
canvas.save(output/"bar-regions.png")
print({"structure": verification["status"], "pixels": pixels["status"],
       "bars": len(regions), "before_mae": pixels["global_before_mae"],
       "after_mae": pixels["global_after_mae"],
       "outside_changed_pixels": pixels["outside_changed_pixels"]})
