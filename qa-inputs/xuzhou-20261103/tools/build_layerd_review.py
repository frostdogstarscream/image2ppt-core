"""Assemble a review pack from the layerd output: readable strips + an index."""
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("outputs/layerd-review-20261104")
SRC_NAMES = Path("qa-inputs/xuzhou-20261103/source")
FONT = "C:/Windows/Fonts/msyh.ttc"

W, H = 900, 507  # 16:9 thumbnails


def f(size, bold=False):
    return ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc" if bold else FONT, size)


def checker(size, step=16):
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    for y in range(0, size[1], step):
        for x in range(0, size[0], step):
            if (x // step + y // step) % 2:
                d.rectangle([x, y, x + step - 1, y + step - 1], fill=(233, 233, 233))
    return img


def flatten(path, size=(W, H)):
    """Composite RGBA over a checkerboard so transparency is visible."""
    with Image.open(path) as im:
        im = im.convert("RGBA").resize(size, Image.LANCZOS)
    canvas = checker(size).convert("RGBA")
    canvas.alpha_composite(im)
    return canvas.convert("RGB")


def strip_for(page_dir: Path, manifest: dict, title: str) -> Path:
    layers = manifest["layers"]
    rows = [("原图 source", page_dir / "source.png")]
    for layer in layers:
        rows.append((f"{layer['file']}  role={layer['role']}  z={layer['z']}  bbox={layer['alpha_bbox']}",
                     page_dir / layer["file"]))
    header = 44
    label_h = 26
    sheet = Image.new("RGB", (W, header + len(rows) * (H + label_h)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((12, 10), title, font=f(20, True), fill=(20, 30, 40))
    y = header
    for label, path in rows:
        if not path.exists():
            continue
        d.text((12, y + 4), label, font=f(14), fill=(90, 104, 118))
        sheet.paste(flatten(path), (0, y + label_h))
        y += H + label_h
    out = page_dir / "对比图.png"
    sheet.save(out)
    return out


def main() -> int:
    rows = []
    for page_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit()):
        manifest = json.loads((page_dir / "manifest.json").read_text(encoding="utf-8"))
        num = page_dir.name
        source_names = list(SRC_NAMES.glob(f"{num}_*.png"))
        pretty = source_names[0].stem if source_names else num
        target = ROOT / pretty
        if not target.exists():
            shutil.move(str(page_dir), str(target))
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        strip_for(target, manifest, f"{pretty} · {len(manifest['layers'])} 层 · {manifest['image_size'][0]}×{manifest['image_size'][1]}")
        fg = [l for l in manifest["layers"] if l["role"] == "foreground"]
        rows.append({
            "page": pretty,
            "layers": len(manifest["layers"]),
            "foreground": len(fg),
            "coverage": round(sum(l["alpha_coverage"] for l in fg) * 100, 1),
            "sizes": sorted({tuple(l["size"]) for l in manifest["layers"]}),
        })

    lines = ["# layerd 分层结果 · 独立审查包", "",
             f"共 {len(rows)} 页，每页一个目录，目录内包含：", "",
             "| 文件 | 说明 |", "|---|---|",
             "| `source.png` | 送进 layerd 的原图 |",
             "| `layer-00.png` | 背景层（z=0，全幅 opaque） |",
             "| `layer-01..03.png` | 前景层（RGBA，带 alpha，z 递增） |",
             "| `manifest.json` | 官方清单：sha256、role、alpha_bbox、alpha_coverage |",
             "| `composite.png` | 服务端按层合成的还原图 |",
             "| `contact.png` | 服务端生成的层预览图 |",
             "| `对比图.png` | 我做的纵向对照条（原图 + 每层，透明处为棋盘格） |",
             "", "## 逐页概览", "",
             "| 页 | 总层数 | 前景层 | 前景覆盖率 | 目录 |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['page']} | {r['layers']} | {r['foreground']} | {r['coverage']}% | `{r['page']}/` |")
    lines += ["", "## 审查时值得看的几点", "",
              "1. **背景层（layer-00）被剥掉了什么**——卡片、横幅、文字是否都消失了，残留多少。",
              "2. **前景层的 alpha 边界**——是紧贴图形轮廓，还是带了一圈底色（这会决定它能不能直接当元素用）。",
              "3. **文字落在哪一层**——layerd 声明 `native_text: false`，文字应当在某个前景层里，看看是哪一层。",
              "4. **前景层之间是否重叠**——04/05/07/10 页有 3 个前景层，覆盖率合计超过 40%。",
              "5. **composite.png 与原图的差异**——服务端合成回去的图能看出分层丢了什么。",
              ""]
    (ROOT / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"打包完成：{len(rows)} 页")
    for r in rows:
        print(f"  {r['page']}  {r['layers']} 层  前景 {r['foreground']}  覆盖 {r['coverage']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
