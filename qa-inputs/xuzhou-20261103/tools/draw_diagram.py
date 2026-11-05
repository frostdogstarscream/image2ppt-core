"""Draw the reverse-engineered pipeline diagram for the image2ppt.com conversion."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path("diagram")
OUT.mkdir(exist_ok=True)

FONT = "C:/Windows/Fonts/msyh.ttc"
FONT_B = "C:/Windows/Fonts/msyhbd.ttc"


def f(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


W, H = 1500, 1180
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

INK = (28, 38, 48)
MUTED = (108, 122, 136)
LINE = (150, 165, 180)

C_MODEL = ((227, 240, 255), (47, 127, 209))
C_DET = ((230, 246, 239), (46, 158, 107))
C_IO = ((240, 243, 247), (140, 153, 168))
C_ART = ((255, 252, 240), (200, 172, 90))
C_TEXT = ((245, 240, 252), (128, 96, 190))


def box(x, y, w, h, title, body=None, palette=C_IO, dashed=False, radius=10,
        tsize=21, bsize=16, align="center"):
    fill, stroke = palette
    if dashed:
        d.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill, outline=stroke, width=1)
        # crude dash overlay: skip, use dotted corners instead
    else:
        d.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill, outline=stroke, width=2)
    tw = d.textlength(title, font=f(tsize, True))
    if align == "center":
        d.text((x + (w - tw) / 2, y + (12 if body else (h - tsize) / 2 - 3)), title, font=f(tsize, True), fill=INK)
    else:
        d.text((x + 18, y + (12 if body else (h - tsize) / 2 - 3)), title, font=f(tsize, True), fill=INK)
    if body:
        for i, ln in enumerate(body):
            lw = d.textlength(ln, font=f(bsize))
            xx = (x + (w - lw) / 2) if align == "center" else (x + 18)
            d.text((xx, y + 44 + i * (bsize + 7)), ln, font=f(bsize), fill=MUTED)


def arrow(x1, y1, x2, y2, color=LINE, width=2, head=9):
    d.line([(x1, y1), (x2, y2)], fill=color, width=width)
    if x1 == x2:
        s = 1 if y2 > y1 else -1
        d.polygon([(x2, y2), (x2 - head * 0.62, y2 - s * head), (x2 + head * 0.62, y2 - s * head)], fill=color)
    else:
        s = 1 if x2 > x1 else -1
        d.polygon([(x2, y2), (x2 - s * head, y2 - head * 0.62), (x2 - s * head, y2 + head * 0.62)], fill=color)


def elbow(x1, y1, x2, y2, color=LINE):
    """orthogonal connector: down, across, then arrow into target"""
    my = (y1 + y2) / 2
    d.line([(x1, y1), (x1, my)], fill=color, width=2)
    d.line([(x1, my), (x2, my)], fill=color, width=2)
    arrow(x2, my, x2, y2, color)


# ---- header
d.text((70, 34), "image2ppt.com 的转换架构", font=f(30, True), fill=INK)
d.text((70, 74), "由成品 PPTX 的 OOXML 反推 · 每条链路后面括号内是成品里的实测证据", font=f(16), fill=MUTED)

LX, LW = 70, 470
RX, RW = 640, 730

# ---- input
box(LX, 120, LW, 58, "输入：页面图 PNG 1672×941", palette=C_IO)
arrow(LX + LW / 2, 178, LX + LW / 2, 226)

# ---- 1 understanding
box(LX, 226, LW, 82, "① 页面理解（视觉模型）",
    ["每页产出：文本行 · 线条 · 色块 · 图标 · 照片"], palette=C_MODEL)
arrow(LX + LW, 267, RX - 12, 267)
box(RX, 226, RW, 82, "元素清单",
    ["每个元素：bbox（像素）· 文字 · 颜色 · 粗细 · 类别"], palette=C_ART)
arrow(LX + LW / 2, 308, LX + LW / 2, 356)

# ---- 2 background
box(LX, 356, LW, 82, "② 背景净化：擦掉文字",
    ["按周边颜色补齐，重绘成一张干净底图"], palette=C_DET)
arrow(LX + LW, 397, RX - 12, 397)
box(RX, 356, RW, 82, "满幅底图 image-N-1.jpg",
    ["1672×941 · 仅 39–70 KB · 平均重绘 14.1% 像素"], palette=C_ART)
arrow(LX + LW / 2, 438, LX + LW / 2, 486)

# ---- 3 classify
box(LX, 486, LW, 82, "③ 元素分类（descr 标签）",
    ["决定这一块是画成矢量，还是切下来"], palette=C_MODEL)

# fork
fy = 568
left_cx, right_cx = LX + 109, LX + LW - 109
d.line([(LX + LW / 2, 568), (LX + LW / 2, fy)], fill=LINE, width=2)
d.line([(left_cx, fy), (right_cx, fy)], fill=LINE, width=2)
arrow(left_cx, fy, left_cx, 606)
arrow(right_cx, fy, right_cx, 606)

box(LX, 606, 218, 120, "纯色块 · 直线",
    ["→ 画成矢量形状", "roundRect 65 · line 33", "rect 2 · custGeom 2"], palette=C_DET, tsize=19, bsize=15)
box(LX + 252, 606, 218, 120, "图标 · 渐变块 · 照片",
    ["→ 从底图上切片", "icon 87 · container 25", "image 128 · salvage 9"], palette=C_ART, tsize=19, bsize=15)

# converge
cy = 762
arrow(left_cx, 726, left_cx, cy)
arrow(right_cx, 726, right_cx, cy)
d.line([(left_cx, cy), (right_cx, cy)], fill=LINE, width=2)
midx = LX + LW / 2
arrow(midx, cy, midx, 772)

# ---- 4 text
box(LX, 772, LW, 82, "④ 文字原生化：一行一框",
    ["字号按行高反推，颜色逐 run 取色"], palette=C_TEXT)
arrow(LX + LW, 813, RX - 12, 813)
box(RX, 772, RW, 82, "文本框的真实参数",
    ['wrap="none" · 内边距 0 · 微软雅黑 · 295 框 / 374 段'], palette=C_ART)
arrow(LX + LW / 2, 854, LX + LW / 2, 902)

# ---- 5 assemble
box(LX, 902, LW, 82, "⑤ 组装",
    ["底图 → 切片 → 形状 → 图标 → 文字（文字恒在最上层）"], palette=C_DET)
arrow(LX + LW, 943, RX - 12, 943)
box(RX, 902, RW, 82, "几样东西它干脆没有",
    ["0 组合 · 0 表格 · 0 图表 · 0 模型调用记录"], palette=C_ART)
arrow(LX + LW / 2, 984, LX + LW / 2, 1032)

box(LX, 1032, LW, 56, "输出：PPTX（12 页）", palette=C_IO)

# ---- coordinate footnote
ny = 1116
d.line([(70, ny - 18), (W - 70, ny - 18)], fill=(226, 231, 236), width=1)
d.text((70, ny), "坐标系：全程用源图像素，最后乘 7288 得到 EMU（90.4% 的坐标正好落在整数像素上）；",
       font=f(15), fill=MUTED)
d.text((70, ny + 24), "画布高固定 7.5 in → 125.4 dpi → 字号 pt = 行高像素 × 72 ÷ 125.4，这就是同一页出现 13.1/13.2/13.3 pt 的原因。",
       font=f(15), fill=MUTED)

img.save(OUT / "pipeline.png")
print("saved", img.size)
