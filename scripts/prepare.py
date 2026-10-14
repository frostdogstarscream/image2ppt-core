"""Preserve a source image, normalize orientation and make analysis/crop assets."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
Image.MAX_IMAGE_PIXELS = 40_000_000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(source):
    source = Path(source).resolve(strict=True)
    if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        raise ValueError("v0.1 接收页面图片；PDF 请先提供页面图片。")
    if source.stat().st_size > 100_000_000:
        raise ValueError("输入图片超过 100 MB。")
    with Image.open(source) as raw:
        if raw.width * raw.height > Image.MAX_IMAGE_PIXELS:
            raise ValueError("输入图片超过 4000 万像素。")
        if getattr(raw, "n_frames", 1) != 1:
            raise ValueError("请逐页提供图片；不静默忽略多帧输入。")
        original_size = list(raw.size)
        orientation = raw.getexif().get(274, 1)
        normalized = ImageOps.exif_transpose(raw)
        composited = normalized.mode == "RGBA" or "transparency" in normalized.info
        if composited:
            rgba = normalized.convert("RGBA")
            base = Image.new("RGBA", rgba.size, "white")
            normalized = Image.alpha_composite(base, rgba).convert("RGB")
        else:
            normalized = normalized.convert("RGB")
        run = ROOT / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
        (run / "source").mkdir(parents=True, exist_ok=False)
        (run / "assets").mkdir()
        shutil.copy2(source, run / "source" / ("original" + source.suffix.lower()))
        page = run / "source" / "page.png"
        normalized.save(page)
        analysis = normalized.copy()
        analysis.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
        analysis.save(run / "source" / "analysis.png")
        w, h = normalized.size
        target = [1280, round(1280 * h / w)]
        if not 96 <= target[1] <= 8192:
            raise ValueError("页面比例不适合原型画布，请先裁出幻灯片区域。")
        ir = {
            "schema_version": "0.1", "source_image": "source/page.png", "source_sha256": digest(page),
            "canvas": {"source_px": [w, h], "target_px": target}, "background": {"color": "FFFFFF"},
            "review_status": "needs_review", "review_notes": ["等待 Codex 看图、建立内容清单并选择背景与字体。"],
            "evidence": [], "assets": [], "groups": [], "nodes": [],
        }
        save_json(run / "slideir.json", ir)
        metadata = {"source_name": source.name, "original_sha256": digest(source), "page_sha256": digest(page),
                    "original_size": original_size, "source_px": [w, h], "exif_orientation": orientation,
                    "analysis_px": list(analysis.size), "analysis_to_page_scale": [w / analysis.width, h / analysis.height],
                    "transparency_composited_on": "white" if composited else None, "status": "awaiting_codex"}
        save_json(run / "input.json", metadata)
        return {"run_dir": str(run), "ir": str(run / "slideir.json"), "page": str(page), "status": "awaiting_codex"}


def crop(source, bbox, output):
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists() or output == source:
        raise ValueError("裁剪输出必须是新文件。")
    x, y, w, h = bbox
    with Image.open(source) as image:
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > image.width or y + h > image.height:
            raise ValueError("裁剪范围必须在原图内。")
        output.parent.mkdir(parents=True, exist_ok=True)
        image.crop((x, y, x + w, y + h)).save(output, format="PNG")
    return {"asset": str(output), "sha256": digest(output), "bbox": bbox}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("source")
    p = sub.add_parser("crop")
    p.add_argument("source")
    p.add_argument("--bbox", nargs=4, type=int, required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare(args.source) if args.command == "prepare" else crop(args.source, args.bbox, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

