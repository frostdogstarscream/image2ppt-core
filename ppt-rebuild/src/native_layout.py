"""Editing relationships and layout measurements in the source slide's coordinate system."""

from functools import lru_cache
import math
import re

import cv2
import numpy as np
from PIL import ImageFont

from native_plan import overlap


def center(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


def row_partner(box, icons):
    x, y, w, h = box
    choices = []
    for icon in icons:
        u, v, s, t = icon["bbox"]
        gap = x - u - s
        dy = abs(y + h / 2 - v - t / 2)
        if -0.15 * h <= gap < max(4 * h, 1.6 * s) and dy <= max(h, t) * 0.9:
            choices.append((dy + max(0, gap) * 0.12, icon["id"]))
    return min(choices)[1] if choices else None


@lru_cache(maxsize=128)
def _face(bold, pixels):
    for name in (("msyhbd.ttc" if bold else "msyh.ttc"), "NotoSansCJK-Regular.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, pixels)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_field(words, rows, para, module, readings, image_size):
    boxes = [r["bbox"] for r in rows]
    x = min(b[0] for b in boxes)
    y = min(b[1] for b in boxes)
    right = max(b[0] + b[2] for b in boxes)
    bottom = max(b[1] + b[3] for b in boxes)
    field = [x, y, right - x, bottom - y]
    own = {r["id"] for r in rows}
    host = module["bbox"]
    carriers = [
        o["bbox"]
        for o in module["objects"]
        if o["kind"] == "shape"
        and o.get("geometry") in ("rect", "roundRect", "pentagon", "chevron", "rightArrow")
        and overlap(o["bbox"], field) >= 0.88 * field[2] * field[3]
    ]
    if carriers:
        host = min(carriers, key=lambda b: b[2] * b[3])
    h = float(np.median([b[3] for b in boxes]))
    others = [r["bbox"] for r in readings.values() if r["id"] not in own]
    limits = [
        host[0] + host[2],
        image_size[0],
        *[
            b[0]
            for b in others
            if b[0] >= right and min(bottom, b[1] + b[3]) - max(y, b[1]) > 0.3 * min(h, b[3])
        ],
    ]
    available = max(field[2], min(limits) - x - max(2, h * 0.16))
    below = [b[1] for b in others if b[1] >= bottom and b[0] < right and b[0] + b[2] > x]
    lower = min([host[1] + host[3], image_size[1], *below])
    size = max(1, h * 0.58)
    bold = bool(para.get("bold"))
    advance = max(_face(bold, 96).getlength(line) for line in words.split("\n")) / 96
    fixed = len(rows) == 1 and para.get("role") in ("label", "title")
    icons = [o for o in module["objects"] if o["kind"] == "icon"]
    partner = row_partner(field, icons) if len(rows) == 1 else None
    anchor = next((o for o in icons if o["id"] == partner), None)
    repeat_row = (
        anchor is not None
        and para.get("role") != "title"
        and sum(
            abs(center(o["bbox"])[0] - center(anchor["bbox"])[0]) < anchor["bbox"][2] * 0.65
            for o in icons
        )
        >= 3
    )
    if repeat_row:
        # Preserve the full edit area to the right of a list marker, including room for later typing.
        row_height = min(max(field[3], lower - y), image_size[1] - y)
        size = min(size, available * 0.75 / max(0.1, advance * 1.06))
        if row_height >= 1.5 * h:
            size = min(size, row_height * 0.75 / (2 * 1.25))
        return [x, y, min(available, image_size[0] - x), row_height], max(1, size), True
    if fixed:
        width = min(available, max(field[2], advance * size / 0.75 * 1.12))
        size = min(size, width * 0.75 / max(0.1, advance * 1.06), max(field[3], lower - y) * 0.60)
        # Keep the original optical centre for centred text that gains room.
        if para.get("align") == "center":
            x = max(host[0], x - (width - field[2]) / 2)
        height = min(max(field[3], size * 1.65), max(field[3], lower - y))
    else:
        width = min(available, max(field[2] * 1.12, float(para.get("width") or 0)))
        height = max(field[3], lower - y)
        # Include a safety margin for Office's CJK advances and mixed-script fallback.
        for _ in range(30):
            lines = sum(
                max(1, math.ceil(_face(bold, 96).getlength(line) / 96 * size / 0.75 * 1.05 / width))
                for line in words.split("\n")
            )
            if lines * size * 1.72 <= height or size <= h * 0.40:
                break
            size *= 0.96
    return (
        [x, y, min(width, image_size[0] - x), min(height, image_size[1] - y)],
        max(1, size),
        not fixed,
    )


def workflow_groups(module, nodes, readings):
    """Confirm explicit groups or infer columns from a sequence of visible step numbers."""
    by = {n["id"]: n for n in nodes}
    owners = {i: n["id"] for n in nodes if n["type"] == "text" for i in n["evidence_ids"]}
    groups = []
    for g in module.get("edit_groups", []):
        if not isinstance(g, dict):
            continue
        ids = {owners.get(i, i) for i in g.get("members", []) if isinstance(i, str)}
        if ids <= by.keys() and any(by[i]["type"] == "text" for i in ids) and len(ids) >= 2:
            groups.append(ids)
    # Model proposals that share members are ambiguous. Fall back to measured columns.
    if groups and len(set.union(*groups)) == sum(map(len, groups)):
        return groups
    numbers = sorted(
        [n for n in nodes if n["type"] == "text" and re.fullmatch(r"0*\d{1,2}", n["text"])],
        key=lambda n: n["bbox"][0],
    )
    if len(numbers) < 3 or any(
        int(b["text"]) != int(a["text"]) + 1 for a, b in zip(numbers, numbers[1:])
    ):
        return []
    height = float(np.median([readings[n["evidence_ids"][0]]["bbox"][3] for n in numbers]))
    if max(n["bbox"][1] for n in numbers) - min(n["bbox"][1] for n in numbers) > height:
        return []
    starts = [n["bbox"][0] for n in numbers]
    stride = float(np.median(np.diff(starts)))
    if stride < 2 * height:
        return []
    groups = [set() for _ in numbers]
    flow_lines = {
        o["id"]
        for o in module["objects"]
        if o.get("geometry") == "line" or o.get("geometry", "").endswith("Arrow")
    }
    for n in nodes:
        b = n["bbox"]
        if b[2] > 0.94 * stride or center(b)[1] < numbers[0]["bbox"][1] - height * 0.45:
            continue
        if (
            n["id"].split("-")[0] in flow_lines
            or n.get("geometry", "").endswith("Arrow")
            or n.get("geometry") == "line"
        ):
            continue
        cx = center(b)[0]
        idx = min(range(len(starts)), key=lambda i: abs(cx - (starts[i] + stride * 0.24)))
        if abs(cx - (starts[idx] + stride * 0.24)) <= stride * 0.49:
            groups[idx].add(n["id"])
    return [g for g in groups if len(g) >= 2]


def painted_overlap(first, second):
    if not overlap(first["bbox"], second["bbox"]):
        return False
    if not first.get("outline") and second.get("outline"):
        first, second = second, first
    if not first.get("outline"):
        return True
    # The empty centre of a routed connector cannot obscure a card inside its bend.
    x, y, w, h = first["bbox"]
    vertices = np.rint(np.asarray(first["outline"]) * [w, h] / 100000 + [x, y]).astype(np.int32)
    left, top, width, height = second["bbox"]
    bounds = (math.floor(left), math.floor(top), math.ceil(left + width) - math.floor(left),
              math.ceil(top + height) - math.floor(top))
    if cv2.pointPolygonTest(vertices, (float(left), float(top)), False) >= 0:
        return True
    return any(cv2.clipLine(bounds, tuple(start), tuple(end))[0]
               for start, end in zip(vertices, np.roll(vertices, 1, axis=0)))


def order_edit_groups(module, nodes, readings):
    """Keep shared backdrop first. Reorder siblings only when the moved layers do not overlap."""
    bundles = workflow_groups(module, nodes, readings)
    workflow = (
        bool(bundles)
        and sum(
            any(
                n["id"] in bundle and n["type"] == "text" and n["text"].strip().isdecimal()
                for n in nodes
            )
            for bundle in bundles
        )
        >= 3
    )
    if not bundles:
        # Pair a row label with an icon only if the relation is unambiguous.
        icons = [o for o in module["objects"] if o["kind"] == "icon"]
        pairs = {}
        for n in nodes:
            if n["type"] == "text":
                partner = row_partner(readings[n["evidence_ids"][0]]["bbox"], icons)
                if partner:
                    pairs.setdefault(partner, []).append(n["id"])
        bundles = [{icon, *labels} for icon, labels in pairs.items() if len(labels) == 1]
    ordered = list(nodes)
    children = []
    for idx, ids in enumerate(bundles, 1):
        members = [n for n in ordered if n["id"] in ids]
        if len(members) < 2:
            continue
        last = max(ordered.index(n) for n in members)
        if any(
            painted_overlap(n, other)
            for n in members
            for other in ordered[ordered.index(n) + 1 : last]
            if other["id"] not in ids
        ):
            continue
        group_id = f"item-{module['id']}-{idx:02}"
        ordered = [n for n in ordered[:last] if n["id"] not in ids] + members + ordered[last + 1 :]
        values = [n["text"].strip() for n in members if n["type"] == "text"]
        number = next((s for s in values if s.isdecimal()), "")
        label = next((s for s in values if not s.isdecimal()), "")
        caption = (number + " " + label).strip().splitlines()[0] if label else group_id
        children.append(
            dict(id=group_id, member_ids=[n["id"] for n in members], display_name=caption[:64])
        )
    parent_of = {key: g["id"] for g in children for key in g["member_ids"]}
    members = list(dict.fromkeys(parent_of.get(n["id"], n["id"]) for n in ordered))
    return ordered, children, members, workflow


def overflow_adjustments(ir, measurement):
    by = {n["id"]: n for n in ir["nodes"] if n["type"] == "text"}
    changed = {}
    for m in measurement["fields"]:
        ratio = min(m["width"] / max(0.01, m["ink_width"]), m["height"] / max(0.01, m["ink_height"]))
        if ratio < 0.98 and m["id"] in by:
            original = by[m["id"]]["font"]["size_pt"]
            changed[m["id"]] = max(1, original * max(0.80, ratio * 0.97))
    return changed
