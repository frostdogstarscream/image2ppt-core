"""Audit: does any text survive on the de-texted plate?

Anything OCR still reads on the plate will end up baked into a raster crop, which
means the reader cannot edit it. This is the "is all text editable" measure.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from image2ppt_vision.models import recognize

directory = Path(sys.argv[1])
rows = []
for plate in sorted(directory.glob("*.png")):
    rgb = np.array(Image.open(plate).convert("RGB"))
    objects = recognize(rgb)
    found = []
    for obj in objects:
        text = (obj.get("text") or "").strip()
        score = float(obj.get("score") or 0)
        if len(text) < 1 or score < 0.55:
            continue
        found.append({"text": text, "score": round(score, 2), "bbox": obj.get("bbox")})
    rows.append({"page": plate.stem, "leftover_texts": len(found), "sample": found[:8]})

total = sum(r["leftover_texts"] for r in rows)
Path("/data/pilot/plate-audit.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
for r in rows:
    print(f"{r['page']:>10}  leftover {r['leftover_texts']:>3}  " +
          " | ".join(t["text"][:12] for t in r["sample"][:4]))
print(f"\nTOTAL leftover texts on plates: {total}")
