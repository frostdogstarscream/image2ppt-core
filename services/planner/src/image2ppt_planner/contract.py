"""The ElementPlan/2 contract as a JSON Schema, plus the prompts that produce it.

The schema here is guidance for the model; the authoritative gate is
``image2ppt_vision.supervised_elements.validate_element_plan``, which is run on
every response before it is accepted.
"""
from __future__ import annotations

GEOMETRIES = [
    "rect", "roundRect", "ellipse", "line", "chevron", "pentagon", "rightArrow",
    "leftArrow", "trapezoid", "parallelogram", "hexagon", "diamond", "triangle",
]

COLOR = {
    "type": "string",
    "pattern": "^[0-9a-fA-F]{6}$",
    "description": "six hex digits WITHOUT a leading '#', e.g. 0B5FA5",
}

BBOX = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 4,
    "maxItems": 4,
    "description": "[x, y, width, height] in source-image pixels",
}

OBJECT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "kind", "bbox", "z"],
    "properties": {
        "id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,70}$"},
        "kind": {"enum": ["native_shape", "raster_asset"]},
        "bbox": BBOX,
        "z": {"type": "integer"},
        "candidate_id": {"type": "string"},
        "geometry": {"enum": GEOMETRIES},
        "fill": {"type": "string"},
        "fill_gradient": {
            "type": "object",
            "additionalProperties": False,
            "required": ["angle_deg", "stops"],
            "properties": {
                "angle_deg": {"type": "number", "description": "0 = left->right, 90 = top->bottom"},
                "stops": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["offset", "color"],
                        "properties": {
                            "offset": {"type": "number", "minimum": 0, "maximum": 100000},
                            "color": COLOR,
                        },
                    },
                },
            },
        },
        "line": {
            "type": "object",
            "additionalProperties": False,
            "required": ["color", "width_pt"],
            "properties": {
                "color": COLOR,
                "width_pt": {"type": "number", "minimum": 0, "maximum": 20},
            },
        },
        "mask": {"enum": ["rect", "ellipse"]},
        "contains_text_ids": {"type": "array", "items": {"type": "string"}},
    },
}

TEXT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "bbox", "text", "color", "action", "bold", "z", "font_size_px"],
    "properties": {
        "id": {"type": "string"},
        "bbox": BBOX,
        "text": {"type": "string", "minLength": 1, "maxLength": 5000},
        "color": COLOR,
        "action": {"enum": ["native", "retained_in_asset"]},
        "bold": {"type": "boolean"},
        "z": {"type": "integer"},
        "font_size_px": {"type": "number", "minimum": 1, "maximum": 400},
        "font_family": {"type": "string"},
        "font_latin": {"type": "string"},
        "font_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "font_evidence": {"type": "string", "minLength": 1, "maxLength": 500},
        "runs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                    "color": COLOR,
                    "bold": {"type": "boolean"},
                },
            },
        },
    },
}

ELEMENT_PLAN_V2 = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema", "source_sha256", "image_size", "background_color", "objects", "text_decisions"],
    "properties": {
        "schema": {"const": "ElementPlan/2"},
        "source_sha256": {"type": "string"},
        "image_size": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        "background_color": COLOR,
        "objects": {"type": "array", "minItems": 1, "maxItems": 400, "items": OBJECT},
        "text_decisions": {"type": "array", "items": TEXT},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
}


SYSTEM_PROMPT = """你是"页面重建规划器"。输入是一张幻灯片图片和一份元素清单，输出必须是**一份严格合法的 ElementPlan/2 JSON**，不允许任何解释文字、Markdown 代码块或注释。

# 硬性规则（违反即整份计划作废）

1. `text_decisions` 必须**逐条覆盖**清单 `texts` 里每一个 id，不多不少。`id` 原样照抄清单，`bbox` 和 `text` 也原样照抄，**不允许改写文字、不允许合并两条、不允许自己编 id**。
2. 每条文字的决定 `action`：
   - `native`：文字做成可编辑文本框（**默认、几乎总是选这个**）；
   - `retained_in_asset`：**会直接被打回**。只有这段文字确实压在照片/插画上、完全无法分离时才能用，且必须有一个 `raster_asset` 的 `contains_text_ids` 里包含它。**读者无法编辑它，所以能拆就拆。**
3. `objects` 里每个对象的 `bbox` 必须在页面范围内。`raster_asset` 的**面积必须小于整页面积的 18%**（约小于 0.18 × 宽 × 高）。想用一张满幅图铺底会被直接拒绝——不要这么做。
4. 能用纯色或单轴渐变的规则图形（矩形、圆角矩形、椭圆、直线、箭头、梯形等）必须用 `native_shape`；照片、图标、插画、带阴影/多段渐变/不规则轮廓的块才用 `raster_asset`。
5. `native_shape` 必须给 `geometry`（取值见下），并给 `fill` 或 `fill_gradient`；直线给 `fill: "none"`，并且 **`line` 对象必须同时包含 `color` 和 `width_pt` 两个字段**（`width_pt` 是 0–20 的数字）——缺一个整份计划会被拒绝。
   渐变必须是这个精确形状（字段名一个字母都不能改）：
   `"fill_gradient": {"angle_deg": 90, "stops": [{"offset": 0, "color": "0B5FA5"}, {"offset": 100000, "color": "FFFFFF"}]}`
   —— `angle_deg` 0 表示从左到右、90 表示从上到下；`offset` 取 0–100000，首尾必须是 0 和 100000。
   不要用 `type` / `angle` / `pos` / `position` 这些别的写法。
6. `font_size_px` 取该行文字在源图里的**字高像素**（清单里的 `px` 是估计值，可以微调但不要离谱）。全部字号请尽量聚成少数几档（例如正文 24/28、标题 40/56），不要每条一个值。
7. `z` 是层叠次序，整数。背景小、文字大。所有 `native` 文字的 `z` 必须**大于**任何与它重叠的 `raster_asset` 的 `z`。
8. 颜色一律写**六位十六进制、不带 `#`**（例如 `0B5FA5`、`FFFFFF`）。写成 `#0B5FA5` 会被校验器直接拒绝。从图片上取真实颜色，不要用主题色猜测。
9. 字体只能从清单 `fonts` 里选。`native` 文字必须同时给出 `font_family`、`font_confidence`（0–1 之间的小数）、`font_evidence`（一句中文说明判断依据，例如"笔画粗细均匀、无衬线、字形方正，符合雅黑"）。
10. `schema` 固定填 `"ElementPlan/2"`，`source_sha256` 和 `image_size` 原样照抄清单。
11. `id` 只能用 `A-Za-z0-9_-`，长度 1–70，且全局唯一。
12. `objects` 里可以给 `candidate_id` 指向清单 `candidates` 里的某个几何候选，表示"这块就是这个候选"，能显著提高几何精度；给了 `candidate_id` 时 `bbox` 应尽量与候选一致。

# geometry 允许值

rect, roundRect, ellipse, line, chevron, pentagon, rightArrow, leftArrow, trapezoid, parallelogram, hexagon, diamond, triangle

# 输出格式

只输出一个 JSON 对象，结构如下（这是结构示意，字段必须齐全）：

```json
{
  "schema": "ElementPlan/2",
  "source_sha256": "<照抄>",
  "image_size": [1672, 941],
  "background_color": "FFFFFF",
  "objects": [
    {"id": "bg-band-top", "kind": "native_shape", "bbox": [0, 0, 1672, 96], "z": 0,
     "geometry": "rect", "fill": "0B5FA5"},
    {"id": "icon-crane", "kind": "raster_asset", "bbox": [62, 224, 66, 59], "z": 4, "mask": "rect"}
  ],
  "text_decisions": [
    {"id": "t001", "bbox": [80, 12, 280, 42], "text": "总体思路", "color": "1A5FA8",
     "action": "native", "bold": true, "z": 20, "font_size_px": 42,
     "font_family": "微软雅黑", "font_latin": "微软雅黑", "font_confidence": 0.8,
     "font_evidence": "无衬线、笔画等宽、字形方正，与雅黑一致"}
  ],
  "limitations": []
}
```

只输出 JSON，不要 ```json 围栏，不要任何前后说明。"""


def build_user_prompt(inventory_compact: dict) -> str:
    import json

    return (
        "以下是这一页的元素清单。请按系统提示生成 ElementPlan/2。\n\n"
        "```json\n"
        + json.dumps(inventory_compact, ensure_ascii=False, separators=(",", ":"))
        + "\n```\n\n"
        "要求：逐条覆盖 texts 中的每个 id；raster_asset 不得超过整页面积 18%；只输出 JSON。"
    )


def repair_prompt(error: str, attempt: int) -> str:
    return (
        f"你上一次输出没有通过服务端校验（第 {attempt} 次尝试）。校验器报错如下：\n\n"
        f"{error}\n\n"
        "请据此修正后**重新输出完整的** ElementPlan/2 JSON。"
        "特别注意：text_decisions 必须覆盖清单中每一条 OCR id 且 bbox/text 不得改动；"
        "raster_asset 面积不得超过整页 18%；id 必须符合 [A-Za-z0-9_-]{1,70}；只输出 JSON。"
    )
