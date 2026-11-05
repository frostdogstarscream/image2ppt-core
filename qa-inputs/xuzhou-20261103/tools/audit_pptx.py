"""Structural audit of a produced PPTX: what is native, what is raster, what is text inside a picture.

Usage:
  python audit_pptx.py <product.pptx> <out.json> [--slides-dir <dir>]

Reports per-slide primitive counts, native text inventory, raster union coverage
and a per-slide "native text vs visible text" hint. No rendering, no model calls.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

EMU_EMU_PER_INCH = 914400


def walk(shapes, depth=0):
    """Yield (shape, depth) recursively through groups."""
    for shape in shapes:
        yield shape, depth
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from walk(shape.shapes, depth + 1)


def shape_kind(shape):
    st = shape.shape_type
    if st == MSO_SHAPE_TYPE.PICTURE:
        return "picture"
    if st == MSO_SHAPE_TYPE.GROUP:
        return "group"
    if st == MSO_SHAPE_TYPE.TABLE:
        return "table"
    if st == MSO_SHAPE_TYPE.CHART:
        return "chart"
    if st == MSO_SHAPE_TYPE.LINE or shape.element.tag.endswith("}cxnSp"):
        return "connector"
    try:
        if shape.has_text_frame and shape.text_frame.text.strip():
            return "text"
    except Exception:
        pass
    return "shape"


def bbox(shape):
    try:
        if shape.left is None or shape.top is None or shape.width is None or shape.height is None:
            return None
        return (shape.left, shape.top, shape.width, shape.height)
    except Exception:
        return None


def main():
    pptx = Path(sys.argv[1])
    out = Path(sys.argv[2])
    prs = Presentation(str(pptx))
    W, H = prs.slide_width, prs.slide_height
    slide_area = W * H

    report = {
        "file": str(pptx),
        "slide_size_emu": {"cx": W, "cy": H, "inches": [round(W / EMU_EMU_PER_INCH, 3), round(H / EMU_EMU_PER_INCH, 3)]},
        "slides": [],
        "totals": Counter(),
    }

    for index, slide in enumerate(prs.slides, start=1):
        kinds = Counter()
        texts = []
        font_sizes = Counter()
        font_names = Counter()
        pics = []      # (bbox, name)
        text_boxes = []  # (bbox, text, size_pt)
        groups = []
        runs = 0

        for shape, depth in walk(slide.shapes):
            kind = shape_kind(shape)
            kinds[kind] += 1
            if kind == "picture":
                box = bbox(shape)
                if box:
                    pics.append(box)
            elif kind == "text":
                runs += len(shape.text_frame.paragraphs)
                t = shape.text_frame.text
                texts.append(t)
                box = bbox(shape)
                sizes = []
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        if run.font.size is not None:
                            sizes.append(run.font.size.pt)
                            font_sizes[round(run.font.size.pt, 2)] += 1
                        if run.font.name:
                            font_names[run.font.name] += 1
                text_boxes.append((box, t, sizes))
            elif kind == "group":
                groups.append(len(shape.shapes))

        # raster union coverage at low resolution
        scale = 0.05
        gw, gh = max(1, int(W * scale / 9525)), max(1, int(H * scale / 9525))
        mask = np.zeros((gh, gw), dtype=bool)
        for (l, t, w, h) in pics:
            x0 = int(round(l / W * gw)); x1 = int(round((l + w) / W * gw))
            y0 = int(round(t / H * gh)); y1 = int(round((t + h) / H * gh))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(gw, x1), min(gh, y1)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = True
        raster_frac = float(mask.mean())

        # native text that sits on top of a raster region (ghost-text risk)
        over_raster = 0
        for box, _t, _s in text_boxes:
            if not box:
                continue
            l, t, w, h = box
            cx = min(gw - 1, max(0, int((l + w / 2) / W * gw)))
            cy = min(gh - 1, max(0, int((t + h / 2) / H * gh)))
            if mask[cy, cx]:
                over_raster += 1

        all_text = "\n".join(texts)
        entry = {
            "slide": index,
            "kinds": dict(kinds),
            "text_shapes": kinds.get("text", 0),
            "paragraphs": runs,
            "native_chars": len(all_text.replace("\n", "")),
            "raster_union_fraction": round(raster_frac, 4),
            "pictures": len(pics),
            "largest_picture_fraction": round(max([(w * h) / slide_area for (_l, _t, w, h) in pics], default=0.0), 4),
            "groups": len(groups),
            "group_members": groups[:10],
            "text_boxes_over_raster": over_raster,
            "font_sizes_pt": dict(sorted(font_sizes.items())),
            "font_names": dict(font_names),
            "native_text": all_text,
        }
        report["totals"].update(kinds)
        report["slides"].append(entry)

    report["totals"] = dict(report["totals"])
    report["totals"]["native_chars_all_slides"] = sum(s["native_chars"] for s in report["slides"])
    report["totals"]["mean_raster_union_fraction"] = round(
        sum(s["raster_union_fraction"] for s in report["slides"]) / len(report["slides"]), 4)
    report["totals"]["distinct_font_sizes"] = sorted({k for s in report["slides"] for k in s["font_sizes_pt"]})

    # media inventory from the package
    import zipfile
    with zipfile.ZipFile(pptx) as z:
        names = z.namelist()
        media = [n for n in names if n.startswith("ppt/media/")]
        workbooks = [n for n in names if n.startswith("ppt/embeddings/")]
        report["package"] = {
            "media_files": len(media),
            "media_bytes": sum(z.getinfo(n).file_size for n in media),
            "embedded_workbooks": len(workbooks),
            "notes_slides": len([n for n in names if n.startswith("ppt/notesSlides/notesSlide")]),
            "has_chart_parts": len([n for n in names if n.startswith("ppt/charts/chart")]),
            "slide_layouts_used": len([n for n in names if n.startswith("ppt/slideLayouts/slideLayout")]),
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "slides": len(report["slides"]),
        "totals": report["totals"],
        "package": report["package"],
        "per_slide": [
            {k: s[k] for k in ("slide", "kinds", "native_chars", "raster_union_fraction",
                               "largest_picture_fraction", "groups", "text_boxes_over_raster")}
            for s in report["slides"]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
