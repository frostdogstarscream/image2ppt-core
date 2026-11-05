"""M5b — PPTX export with python-pptx.

The slide keeps the source aspect ratio (no stretching). Text is a real text box
with zero insets and word wrap off, so what the reader edits is what was measured.
Raster objects are placed above the background and below nothing else.

A scene object is only written once: the exporter never copies the same visible
content from both a native object and a leftover raster.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt


def _emu(px: float, dpi: float) -> Emu:
    return Emu(int(round(px / dpi * 914400)))


def export(scene: dict, out_path: str | Path, *, workdir: Path, source_path: Path) -> dict:
    canvas_w, canvas_h = scene["canvas_px"]
    dpi = float(scene["dpi"])
    deck = Presentation()
    deck.slide_width = _emu(canvas_w, dpi)
    deck.slide_height = _emu(canvas_h, dpi)
    slide = deck.slides.add_slide(deck.slide_layouts[6])

    background = Path(scene["background"]["file"])
    slide.shapes.add_picture(str(background), 0, 0, deck.slide_width, deck.slide_height)

    written = {"text": 0, "image": 0}
    for obj in sorted(scene["objects"], key=lambda o: o["order"]):
        x, y, w, h = obj["bbox_px"]
        if obj["type"] == "image":
            crop_from = obj.get("crop_from", "source")
            origin = Path(scene["source"]) if crop_from == "source" else background
            with Image.open(origin) as image:
                crop = image.convert("RGB").crop((int(x), int(y), int(x + w), int(y + h)))
            asset = workdir / "assets"
            asset.mkdir(parents=True, exist_ok=True)
            crop_path = asset / f"{obj['id']}.png"
            crop.save(crop_path)
            slide.shapes.add_picture(str(crop_path), _emu(x, dpi), _emu(y, dpi), _emu(w, dpi), _emu(h, dpi))
            written["image"] += 1
            continue

        box = slide.shapes.add_textbox(_emu(x, dpi), _emu(y, dpi), _emu(w, dpi), _emu(h, dpi))
        frame = box.text_frame
        frame.word_wrap = False
        frame.margin_left = 0
        frame.margin_right = 0
        frame.margin_top = 0
        frame.margin_bottom = 0
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        lines = obj["text"].split("\n")
        for index, line in enumerate(lines):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.alignment = PP_ALIGN.LEFT
            run = paragraph.add_run()
            run.text = line
            font = run.font
            font.size = Pt(obj["font_size_pt"])
            font.bold = bool(obj.get("bold"))
            if obj.get("font_family"):
                font.name = obj["font_family"]
                # East-Asian face must be set explicitly or PowerPoint falls back.
                rPr = run._r.get_or_add_rPr()
                ea = rPr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}ea")
                if ea is None:
                    from lxml import etree
                    ea = etree.SubElement(rPr, "{http://schemas.openxmlformats.org/drawingml/2006/main}ea")
                ea.set("typeface", obj["font_family"])
            if obj.get("color"):
                font.color.rgb = RGBColor.from_string(obj["color"])
        written["text"] += 1

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deck.save(out_path)
    return {"pptx": str(out_path), "written": written, "slides": len(deck.slides)}
