"""Draw the current two-machine workflow."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "C:/Windows/Fonts/msyh.ttc"
FONT_B = "C:/Windows/Fonts/msyhbd.ttc"
OUT = Path("qa-inputs/xuzhou-20261103/diagram")
OUT.mkdir(parents=True, exist_ok=True)


def f(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


W, H = 1680, 980
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

INK = (26, 36, 46)
MUTED = (104, 118, 132)
LINE = (148, 162, 176)
SRV = ((232, 241, 252), (58, 118, 190))       # server lane
LOC = ((232, 246, 240), (40, 150, 106))       # local lane
ART = ((255, 249, 232), (198, 160, 74))       # artifact
PILL = ((240, 244, 249), (150, 165, 180))


def zone(x, y, w, h, title, palette):
    d.rounded_rectangle([x, y, x + w, y + h], radius=16, fill=palette[0] + (255,) if False else tuple(int(c * 0.35 + 255 * 0.65) for c in palette[0]),
                        outline=palette[1], width=2)
    d.text((x + 22, y + 14), title, font=f(19, True), fill=palette[1])


def box(x, y, w, h, title, body, palette, tsize=20, bsize=15):
    fill = tuple(int(c * 0.25 + 255 * 0.75) for c in palette[0])
    d.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=fill, outline=palette[1], width=2)
    d.text((x + 16, y + 12), title, font=f(tsize, True), fill=INK)
    d.text((x + 16, y + 12 + tsize + 10), body, font=f(bsize), fill=MUTED)


def pill(x, y, text, palette=PILL):
    w = d.textlength(text, font=f(13)) + 22
    d.rounded_rectangle([x, y, x + w, y + 24], radius=12, fill=palette[0], outline=palette[1], width=1)
    d.text((x + 11, y + 4), text, font=f(13), fill=INK)
    return w


def arrow(x1, y1, x2, y2, color=LINE, width=2, head=8):
    d.line([(x1, y1), (x2, y2)], fill=color, width=width)
    if y1 == y2:
        s = 1 if x2 > x1 else -1
        d.polygon([(x2, y2), (x2 - s * head, y2 - head * 0.6), (x2 - s * head, y2 + head * 0.6)], fill=color)
    else:
        s = 1 if y2 > y1 else -1
        d.polygon([(x2, y2), (x2 - head * 0.6, y2 - s * head), (x2 + head * 0.6, y2 - s * head)], fill=color)


# header
d.text((70, 34), "当前的图片 → 可编辑 PPTX 工作流", font=f(30, True), fill=INK)
d.text((70, 76), "两段式：服务器负责“看懂”，本机负责“画出来”，中间用 SlideIR 交接", font=f(16), fill=MUTED)

# lanes
LX, LW = 70, 700
RX, RW = 910, 700
zone(LX, 120, LW, 420, "服务器 36.134.45.38 · 一次性容器（现有服务不动）", SRV)
zone(RX, 120, RW, 640, "本机 Windows · Node + Artifact 运行时", LOC)

# server steps
sx, sw = LX + 26, LW - 52
box(sx, 178, sw, 74, "① OCR 与图形候选", "PP-OCRv4 文字 + OpenCV 候选 → inventory.json", SRV)
arrow(sx + sw // 2, 252, sx + sw // 2, 282)
box(sx, 282, sw, 92, "② 模型规划  qwen3.8-max（关闭思考）", "ElementPlan/2：一行一节点、逐 run 取色、字号按行高反推", SRV)

gy = 384
gx = sx
for label in ("字段准入", "语义校验", "文字必须原生"):
    gx += pill(gx, gy, label) + 8
d.text((sx + 4, gy + 30), "三道校验闸：任一不过就带着错误打回重做（最多 3 次）", font=f(13), fill=MUTED)

arrow(sx + sw // 2, 374, sx + sw // 2, 442)
box(sx, 442, sw, 74, "③ 元素重建 reconstruct_elements", "→ calibrated/slideir.json（约 67 节点/页）", SRV)

# handoff
hx, hy = 786, 452
d.rounded_rectangle([hx, hy - 26, hx + 118, hy + 26], radius=8, fill=ART[0], outline=ART[1], width=2)
d.text((hx + 14, hy - 18), "SlideIR", font=f(16, True), fill=INK)
d.text((hx + 14, hy + 2), "每页一份 JSON", font=f(12), fill=MUTED)
arrow(LX + LW - 4, hy, hx - 4, hy)
d.line([(hx + 59, hy - 26), (hx + 59, 216)], fill=LINE, width=2)
arrow(hx + 59, 216, RX + 26, 216)

# local steps
lx, lw = RX + 26, RW - 52
box(lx, 178, lw, 74, "④ 去字底图", "OCR 掩码 + TELEA 修复 → plate.png（擦掉约 21% 像素）", LOC)
arrow(lx + lw // 2, 252, lx + lw // 2, 282)
box(lx, 282, lw, 92, "⑤ 精修", "渐变拟合 · 保真闸门(5) · 密度回填(20%) · z 重排", LOC)
arrow(lx + lw // 2, 374, lx + lw // 2, 404)
box(lx, 404, lw, 74, "⑥ 编译（固定编译器）", "→ result.pptx + preview.png + quality.json", LOC)
arrow(lx + lw // 2, 478, lx + lw // 2, 508)
box(lx, 508, lw, 86, "⑦ 测量与审计", "整页/文字/非文字像素差，逐区域贡献排序", LOC)

gy = 604
gx = lx
for label in ("覆盖检查", "文字可见性", "底图残字"):
    gx += pill(gx, gy, label) + 8
d.text((lx + 4, gy + 30), "三道出片前审计：内容丢失、白字隐形、文字被烘焙", font=f(13), fill=MUTED)

arrow(lx + lw // 2, 594, lx + lw // 2, 660)
box(lx, 660, lw, 74, "⑧ 合成与交付", "compose 12 页 → deck.pptx，结构校验", LOC)

# bottom strip
by = 808
d.line([(70, by - 26), (W - 70, by - 26)], fill=(226, 231, 236), width=1)
d.text((70, by), "三个旋钮 → 三版成品（12 页平均像素差 / 切片占比，对手 image2ppt 为 13.11 / 60.9%）", font=f(16, True), fill=INK)
rows = [
    ("A 保真优先", "闸门 5 · 不回填", "12.89 / 43.3%"),
    ("B 可编辑优先", "闸门 5 · 切片封顶 35%", "13.73 / 27.8%"),
    ("C 补全优先（推荐）", "闸门 5 · 回填 20%", "12.38 / 53.6%"),
]
cx = 78
for name, cfg, score in rows:
    d.rounded_rectangle([cx, by + 34, cx + 470, by + 106], radius=10,
                        fill=((240, 246, 252) if "C " not in name else (232, 246, 240)), outline=(170, 182, 194), width=1)
    d.text((cx + 14, by + 44), name, font=f(17, True), fill=INK)
    d.text((cx + 14, by + 72), cfg, font=f(13), fill=MUTED)
    d.text((cx + 300, by + 52), score, font=f(20, True), fill=(30, 120, 80))
    cx += 490

d.text((70, by + 124), "耗时：服务端 54 分钟（12 页）+ 本机 25 分钟 ≈ 80 分钟出一册；改代码后只重跑本机，约 25 分钟。",
       font=f(14), fill=MUTED)

img.save(OUT / "workflow.png")
print("saved", img.size)
