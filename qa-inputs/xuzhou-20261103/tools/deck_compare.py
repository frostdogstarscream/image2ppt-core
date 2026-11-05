"""Deck-level comparison: our gated build vs the reference product, page by page.

Raster coverage is measured as the union of image-node rectangles, not the sum,
so overlapping crops cannot push the number past 100%.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"D:\Code\image2PPT")
DECK = ROOT / ".tmp-test" / "deck"
SRC = ROOT / "qa-inputs" / "xuzhou-20261103" / "source"
REF = ROOT / "qa-inputs" / "xuzhou-20261103" / "render-product"


def rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB")).astype(np.int16)


def newest_preview(page_dir):
    outs = sorted((page_dir / "h5" / "outputs").iterdir())
    return outs[-1] / "preview.png"


def union_fraction(ir, w, h):
    mask = np.zeros((h, w), bool)
    for node in ir["nodes"]:
        if node["type"] != "image":
            continue
        x, y, bw, bh = [int(round(v)) for v in node["bbox"]]
        x, y = max(0, x), max(0, y)
        mask[y:min(h, y + bh), x:min(w, x + bw)] = True
    return float(mask.mean())


rows = []
for page in sorted(DECK.iterdir()):
    if not page.is_dir():
        continue
    pid = page.name
    src_file = next(SRC.glob(f"{pid}_*.png"))
    source = rgb(src_file)
    h, w = source.shape[:2]

    our_preview = rgb(newest_preview(page))
    if our_preview.shape != source.shape:
        our_preview = np.array(Image.fromarray(our_preview.astype(np.uint8))
                               .resize((w, h), Image.LANCZOS)).astype(np.int16)
    ours = float(np.abs(source - our_preview).mean())

    ref_preview = rgb(REF / f"page-0{pid}.png")
    if ref_preview.shape != source.shape:
        ref_preview = np.array(Image.fromarray(ref_preview.astype(np.uint8))
                               .resize((w, h), Image.LANCZOS)).astype(np.int16)
    ref = float(np.abs(source - ref_preview).mean())

    ir = json.loads((page / "h5" / "slideir.json").read_text(encoding="utf-8"))
    texts = sum(1 for n in ir["nodes"] if n["type"] == "text")
    shapes = sum(1 for n in ir["nodes"] if n["type"] == "shape")
    images = sum(1 for n in ir["nodes"] if n["type"] == "image")
    rows.append({"page": pid, "ref": round(ref, 2), "ours": round(ours, 2),
                 "delta": round(ours - ref, 2), "raster_union": round(100 * union_fraction(ir, w, h), 1),
                 "texts": texts, "shapes": shapes, "images": images})

print(f"{'页':>3} {'image2ppt':>10} {'我们(闸门)':>10} {'差':>7} {'切片并集':>9} {'文字':>5} {'形状':>5} {'图片':>5}")
for r in rows:
    print(f"{r['page']:>3} {r['ref']:>10.2f} {r['ours']:>10.2f} {r['delta']:>+7.2f} "
          f"{r['raster_union']:>8.1f}% {r['texts']:>5} {r['shapes']:>5} {r['images']:>5}")

n = len(rows)
print()
print(f"平均：image2ppt {sum(r['ref'] for r in rows)/n:.2f} | 我们 {sum(r['ours'] for r in rows)/n:.2f} "
      f"| 平均切片并集 {sum(r['raster_union'] for r in rows)/n:.1f}%")
print(f"我们更低的页：{sum(1 for r in rows if r['delta']<0)}/{n}")
print(f"文字全部原生：{all(r['texts']>0 for r in rows)} | 合计文字节点 {sum(r['texts'] for r in rows)}")

(ROOT / ".tmp-test" / "deck-compare.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
