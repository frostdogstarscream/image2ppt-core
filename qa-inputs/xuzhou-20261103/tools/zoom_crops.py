"""Side-by-side zoom crops: source page vs re-imported render, for close inspection."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

SOURCE = Path(sys.argv[1])
RENDER = Path(sys.argv[2])
OUT = Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)

# (name, page, left, top, right, bottom, zoom)
REGIONS = [
    ("p03-card1", 3, (18, 232, 420, 460), 2.0),
    ("p03-header", 3, (10, 20, 760, 175), 2.0),
    ("p03-wave", 3, (0, 760, 1672, 941), 1.0),
    ("p01-cover", 1, (0, 0, 1672, 941), 0.85),
    ("p09-card", 9, (18, 232, 420, 460), 2.0),
]

src_files = sorted(SOURCE.glob("*.png"))
for name, page, box, zoom in REGIONS:
    src = Image.open(src_files[page - 1]).convert("RGB")
    ren = Image.open(RENDER / f"page-{page:03d}.png").convert("RGB").resize(src.size)
    a = src.crop(box)
    b = ren.crop(box)
    w, h = a.size
    w2, h2 = int(w * zoom), int(h * zoom)
    a = a.resize((w2, h2), Image.LANCZOS)
    b = b.resize((w2, h2), Image.LANCZOS)
    gap, header = 10, 22
    sheet = Image.new("RGB", (w2 * 2 + gap, h2 + header), "white")
    sheet.paste(a, (0, header))
    sheet.paste(b, (w2 + gap, header))
    d = ImageDraw.Draw(sheet)
    d.text((3, 6), f"SOURCE p{page}", fill="black")
    d.text((w2 + gap + 3, 6), f"PRODUCT p{page}", fill="black")
    sheet.save(OUT / f"{name}.png")
    print(name, sheet.size)
