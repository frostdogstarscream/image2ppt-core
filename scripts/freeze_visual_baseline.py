"""Freeze existing real-page artifacts; measurements are not human acceptance."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services/vision/src"))
from image2ppt_vision.storage import digest, read_json, write_json
from image2ppt_vision.text_ledger import verify_ledger
from image2ppt_vision.font_plan import verify_font_export


def font_check(ir, pptx, slide):
    try:
        return verify_font_export(ir, pptx, slide=slide)
    except ValueError as error:
        return {"status": "REVIEW_REQUIRED", "first_failure": str(error),
                "meaning": "Existing baseline defect retained; not an accepted font reference."}


def confined(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path escapes baseline")
    return path


def verify(folder):
    folder = Path(folder).resolve()
    manifest = read_json(folder / "manifest.json")
    listed = set()
    for item in manifest["files"]:
        path = confined(folder, item["path"])
        if item["path"] in listed:
            raise ValueError("duplicate manifest path")
        listed.add(item["path"])
        if not path.is_file() or digest(path) != item["sha256"]:
            raise ValueError("baseline file changed: " + item["path"])
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*")
              if p.is_file() and p != folder / "manifest.json"}
    if actual != listed:
        raise ValueError("baseline file inventory changed")
    checks = []
    for index, page in enumerate(manifest["pages"], 1):
        checks.append({
            "page": page,
            "text": verify_ledger(folder / "pages" / page, folder / "baseline.pptx", slide=index),
            "font": font_check(folder / "pages" / page / "slideir.json",
                               folder / "baseline.pptx", index),
        })
    return {"status": "PASS", "files": len(listed), "pages": checks,
            "meaning": "Snapshot integrity and text export; inspect separate font results",
            "human_acceptance": False}


def measure(source, preview, ir, destination):
    import numpy as np
    from PIL import Image
    src = Image.open(source).convert("RGB")
    rendered = Image.open(preview).convert("RGB")
    if src.size != tuple(ir["canvas"]["source_px"]):
        raise ValueError("source image size mismatch")
    if rendered.size != tuple(ir["canvas"]["target_px"]):
        raise ValueError("preview size mismatch")
    # Compare in the actual render coordinate system; record source resampling.
    reference = src.resize(rendered.size, Image.Resampling.LANCZOS)
    delta = np.abs(np.asarray(reference, dtype=float) - np.asarray(rendered, dtype=float))
    canvas = Image.new("RGB", (rendered.width * 2, rendered.height), "white")
    canvas.paste(reference, (0, 0))
    canvas.paste(rendered, (rendered.width, 0))
    canvas.save(destination)
    sx, sy = rendered.width / src.width, rendered.height / src.height
    regions = []
    for node in ir["nodes"]:
        x, y, w, h = node["bbox"]
        x0, y0 = max(0, int(x*sx)), max(0, int(y*sy))
        x1, y1 = min(rendered.width, int(np.ceil((x+w)*sx))), min(rendered.height, int(np.ceil((y+h)*sy)))
        region = delta[y0:y1, x0:x1]
        regions.append({"id": node["id"], "type": node["type"], "bbox_source_px": node["bbox"],
                        "mean_absolute_rgb_error": round(float(region.mean()), 5) if region.size else None})
    return {"source_size": list(src.size), "preview_size": list(rendered.size),
            "source_resampling": "Pillow LANCZOS to actual preview size",
            "mean_absolute_rgb_error": round(float(delta.mean()), 5),
            "p95_absolute_rgb_error": float(np.percentile(delta, 95)),
            "regions": regions,
            "interpretation": "Diagnostic distances, not accuracy or acceptance; overlapping ROIs include surrounding/occluding objects."}


def freeze(input_root, deck_root, pages, output):
    input_root, deck_root, output = map(lambda p: Path(p).resolve(), (input_root, deck_root, output))
    if output.exists():
        raise FileExistsError("new baseline directory required")
    if not pages or len(set(pages)) != len(pages):
        raise ValueError("nonempty unique ordered pages required")
    if pages != ["02", "03", "05"]:
        raise ValueError("this acceptance baseline is scoped to ordered fixtures 02, 03, 05")
    # Validate original artifacts before creating a snapshot.
    checks = []
    for i, page in enumerate(pages, 1):
        folder = confined(input_root, page)
        checks.append({"page": page,
                       "text": verify_ledger(folder, deck_root / "deck.pptx", slide=i),
                       "font": font_check(folder / "slideir.json", deck_root / "deck.pptx", i)})
    output.mkdir(parents=True)
    origins = []

    def copy(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        origins.append({"path": dst.relative_to(output).as_posix(), "origin": str(src.resolve()),
                        "sha256": digest(src)})

    copy(deck_root / "deck.pptx", output / "baseline.pptx")
    for name in ("summary.json", "source-review-notes.json"):
        if (input_root / name).is_file():
            copy(input_root / name, output / "prior-evidence" / name)
    for name in ("quality.json", "text-ledger-verification.json"):
        copy(deck_root / name, output / "prior-evidence" / name)
    measurements = []
    for i, page in enumerate(pages, 1):
        folder = input_root / page
        for file in folder.rglob("*"):
            if file.is_file():
                copy(file, output / "pages" / page / file.relative_to(folder))
        preview = output / "previews" / f"{page}.png"
        copy(deck_root / "preview" / f"page-{i:03d}.png", preview)
        ir = read_json(folder / "slideir.json")
        comparison = output / "comparisons" / f"{page}.png"
        comparison.parent.mkdir(exist_ok=True)
        metrics = measure(folder / ir["source_image"], preview, ir, comparison)
        measurements.append({"page": page, "objects": dict(Counter(n["type"] for n in ir["nodes"])),
                             "visual": metrics})
    for base in (ROOT / "scripts", ROOT / "services/vision/src/image2ppt_vision"):
        for file in base.rglob("*"):
            if file.is_file() and file.suffix in (".py", ".mjs") and "__pycache__" not in file.parts:
                copy(file, output / "code-snapshot" / file.relative_to(ROOT))
    write_json(output / "origins.json", origins)
    write_json(output / "measurements.json", measurements)
    write_json(output / "export-checks.json", checks)
    policy = {
        "schema": "VisualAcceptance/1", "baseline_role": "regression reference, not ground truth",
        "text": {"selected_text_must_match": True, "truth_labels": "not_fully_verified",
                 "pending_entries": 24, "pending_is_error_count": False},
        "attributes": {"native_text_count": 270, "font_xml_checked": True,
                       "font_result": {c["page"]: c["font"]["status"] for c in checks},
                       "geometry_color_layer_source_accuracy": "REVIEW_REQUIRED"},
        "visual": {"global_and_object_roi_distances": "diagnostic_only",
                   "font_stage": "fixed text, geometry, colors, assets, z-order; only declared font changes",
                   "later_patch_gate": "bounded object/property scope; compare target and unaffected regions separately",
                   "automatic_acceptance": False},
        "powerpoint_checked": False, "wps_checked": False, "human_acceptance": False,
    }
    write_json(output / "acceptance.json", policy)
    (output / "README.md").write_text(
        "# 固定样例与验收基准\n\n"
        "固定页码顺序：02、03、05。此包是回归参照，不是已经验收合格的标准答案。\n\n"
        "- `pages/`：原图、素材、文字台账、IR、authoring 及识别证据。\n"
        "- `baseline.pptx` 与 `previews/`：既有实际 PPTX 和该文件的回渲染结果。\n"
        "- `comparisons/`：左侧原图缩放、右侧实际 PPTX 回渲染。\n"
        "- `export-checks.json`：本次重新检查的文字与字体声明一致性。\n"
        "- `measurements.json`：整页及对象区域像素差异；不等同于准确率。\n"
        "- `acceptance.json`：内容、属性、视觉三类验收规则与未验证状态。\n"
        "- `origins.json`、`manifest.json`：原始路径、文件哈希与完整性清单。\n"
        "- `code-snapshot/`：建立基准时的相关程序源码。\n\n"
        "270 个原生文本导出一致；275 条台账记录中仍有 24 条待核对，"
        "其中 5 条由图片保留。未做 PowerPoint/WPS 打开编辑及人工验收。\n\n"
        "后续阶段创建独立目录，禁止覆盖本包。检查：\n"
        "```powershell\npython -X utf8 scripts/freeze_visual_baseline.py verify --folder <本包目录>\n```\n",
        encoding="utf8")
    files = [{"path": p.relative_to(output).as_posix(), "sha256": digest(p), "bytes": p.stat().st_size}
             for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "manifest.json", {"schema": "VisualBaseline/1",
               "created_at": datetime.now(timezone.utc).isoformat(), "pages": pages, "files": files})
    return verify(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("freeze")
    for field in ("input", "deck", "output"):
        command.add_argument("--" + field, required=True)
    command.add_argument("--pages", nargs="+", default=["02", "03", "05"])
    command = sub.add_parser("verify")
    command.add_argument("--folder", required=True)
    args = parser.parse_args()
    result = verify(args.folder) if args.command == "verify" else freeze(args.input, args.deck, args.pages, args.output)
    print(json.dumps(result, ensure_ascii=False))
