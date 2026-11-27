"""Materialize a component plan on a blank canvas. Only bounded visual assets can read source pixels."""

from pathlib import Path
import math
import re
from subprocess import SubprocessError

import cv2
import numpy as np
from PIL import Image

from layerd_loader import load_page
from layerd_remote import decompose
from native_plan import overlap
from native_layout import fit_field, order_edit_groups, row_partner
from native_matte import optional_matte
from reconstruction import extract_rgba
from slideir_adapter import sha256
from text_mask import remote_inpaint

GEOMETRIES = {
    "rect",
    "roundRect",
    "ellipse",
    "line",
    "pentagon",
    "chevron",
    "rightArrow",
    "leftArrow",
    "upArrow",
    "downArrow",
    "diamond",
    "triangle",
    "hexagon",
    "trapezoid",
    "parallelogram",
}


def rgb(value, default):
    value = str(value or "").strip("#")
    return value.upper() if re.fullmatch("[0-9a-fA-F]{6}", value) else default


def shape_style(row):
    fill = rgb(row.get("fill"), "none" if row.get("fill") == "none" else "FFFFFF")
    style = {"geometry": row.get("geometry", "rect"), "fill": fill}
    if style["geometry"] not in GEOMETRIES:
        raise ValueError("Unsupported shape geometry: " + str(style["geometry"]))
    end = rgb(row.get("end_color"), None)
    opacity = float(row.get("opacity", 1))
    if end and fill != "none":
        style["fill_gradient"] = {
            "angle_deg": float(row.get("angle", 0)),
            "stops": [
                {"offset": 0, "color": fill, "opacity": opacity},
                {"offset": 100000, "color": end, "opacity": float(row.get("end_opacity", opacity))},
            ],
        }
    elif fill != "none" and opacity != 1:
        style["fill_opacity"] = opacity
    stroke = rgb(row.get("stroke"), None)
    if stroke:
        style["line"] = {"color": stroke, "width_pt": float(row.get("stroke_width", 1)) * 0.75}
    return style


def photo_asset(crop, box, texts, bands, inpaint_url, notes):
    """Separate typeset overlays from the photograph, keeping photographed text."""
    arr = np.array(crop.convert("RGB"))
    mask = np.zeros(arr.shape[:2], np.uint8)
    left, top, width, height = box
    for text in texts:
        x, y, w, h = text["bbox"]
        x0, y0, x1, y1 = (
            max(0, x - left),
            max(0, y - top),
            min(width, x + w - left),
            min(height, y + h - top),
        )
        if x0 < x1 and y0 < y1:
            region = arr[y0:y1, x0:x1].astype(float)
            ink = np.array(list(bytes.fromhex(text["color"])))
            distance = np.max(np.abs(region - ink), axis=2)
            mask[y0:y1, x0:x1] |= cv2.dilate(
                (distance < 85).astype(np.uint8), np.ones((3, 3), np.uint8)
            )
    for band in bands:
        x, y, w, h = band["bbox"]
        if overlap(box, band["bbox"]) < 0.8 * w * h:
            continue
        x0, y0, x1, y1 = (
            max(0, x - left),
            max(0, y - top),
            min(width, x + w - left),
            min(height, y + h - top),
        )
        if x0 >= x1 or y0 >= y1:
            continue
        opacity = float(band.get("opacity", 1))
        if opacity >= 0.95 or band.get("end_color"):
            mask[y0:y1, x0:x1] = 1
        else:
            tint = np.array(list(bytes.fromhex(rgb(band.get("fill"), "FFFFFF"))))
            region = arr[y0:y1, x0:x1].astype(float)
            arr[y0:y1, x0:x1] = np.clip((region - opacity * tint) / (1 - opacity), 0, 255).round()
    if mask.any():
        if inpaint_url:
            try:
                return Image.fromarray(remote_inpaint(arr, mask.astype(bool), inpaint_url))
            except (OSError, RuntimeError, ValueError, SubprocessError) as error:
                notes.append("Local photo repair used after " + type(error).__name__)
        arr = cv2.inpaint(arr, mask, 3, cv2.INPAINT_TELEA)
        notes.append("Occluded photograph pixels were locally estimated")
    return Image.fromarray(arr)


def layer_icon(page, box, target):
    x, y, w, h = box
    for layer in reversed(page.layers):
        with Image.open(layer.path) as im:
            rgba = im.convert("RGBA").crop((x, y, x + w, y + h))
        pixels = np.asarray(rgba)
        opaque = pixels[:, :, 3] > 180
        clear = pixels[:, :, 3] < 16
        if 0.025 < opaque.mean() < 0.91 and clear.mean() > 0.07:
            err = np.abs(pixels[:, :, :3].astype(float) - target)
            spread = np.linalg.norm(target - np.median(target.reshape(-1, 3), axis=0), axis=2)
            salient = spread > max(28, np.percentile(spread, 65))
            if (
                salient.any()
                and (opaque & salient).sum() / salient.sum() > 0.78
                and np.median(err[opaque]) < 22
            ):
                return rgba
    return None


def measured_icon(image, box, neighbors, circular=False):
    """Snap a coarse model box to a nearby bounded ink component before cropping."""
    x, y, w, h = box
    margin = round(0.78 * max(w, h))
    left, top = max(0, x - margin), max(0, y - margin)
    right, bottom = min(image.width, x + w + margin), min(image.height, y + h + margin)
    view = np.asarray(image.crop((left, top, right, bottom))).astype(np.int16)
    if circular:
        # Short proposals often cover only the top half of a circular badge. Measure
        # its source outline independently of the white drawing inside.
        gray = cv2.cvtColor(view.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        scale = max(w, h)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            1,
            max(5, scale * 0.35),
            param1=65,
            param2=max(14, scale * 0.21),
            minRadius=max(4, round(scale * 0.34)),
            maxRadius=max(6, round(scale * 0.68)),
        )
        if circles is not None:
            options = []
            for cx, cy, radius in circles[0]:
                cx, cy = cx + left, cy + top
                dist = ((cx - x - w / 2) / w) ** 2 + ((cy - y - h / 2) / max(w, h)) ** 2
                pad = math.ceil(radius) + 2
                l, t = max(0, round(cx) - pad), max(0, round(cy) - pad)
                candidate = [
                    l,
                    t,
                    min(image.width, round(cx) + pad + 1) - l,
                    min(image.height, round(cy) + pad + 1) - t,
                ]
                if dist < 0.27 and not any(
                    overlap(candidate, n["bbox"]) > 0.20 * n["bbox"][2] * n["bbox"][3]
                    and overlap(box, n["bbox"]) < 0.12 * n["bbox"][2] * n["bbox"][3]
                    for n in neighbors
                ):
                    options.append((dist + 0.16 * abs(radius * 2 / scale - 1.04), candidate))
            if options:
                return min(options)[1]
    ring = np.median(np.concatenate((view[:, 0], view[:, -1], view[0], view[-1])), axis=0)
    ink = (np.max(np.abs(view - ring), axis=2) > 45).astype(np.uint8)
    for line in neighbors:
        u, v, s, t = line["bbox"]
        if overlap(box, line["bbox"]) < 0.20 * s * t:
            ink[max(0, v - top) : max(0, v + t - top), max(0, u - left) : max(0, u + s - left)] = 0
    join = max(1, round(min(w, h) * 0.12))
    fused = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * join + 1, 2 * join + 1))
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(fused)
    options = []
    for u, v, s, t, area in stats[1:]:
        raw = [int(left + u), int(top + v), int(s), int(t)]
        distance = ((raw[0] + s / 2 - x - w / 2) / w) ** 2 + ((raw[1] + t / 2 - y - h / 2) / h) ** 2
        if any(
            overlap(raw, line["bbox"]) > 0.20 * line["bbox"][2] * line["bbox"][3]
            and overlap(box, line["bbox"]) < 0.12 * line["bbox"][2] * line["bbox"][3]
            for line in neighbors
        ):
            continue
        if (
            0.25 * w <= s <= 1.52 * w
            and 0.28 * h <= t <= min(2.6 * h, 1.65 * max(w, h))
            and area > 0.07 * w * h
            and distance < 0.92
        ):
            options.append((distance, raw))
    if not options:
        return box
    _, (x, y, w, h) = min(options)
    pad = max(2, round(0.08 * min(w, h)))
    l, t = max(0, x - pad), max(0, y - pad)
    return [l, t, min(image.width, x + w + pad) - l, min(image.height, y + h + pad) - t]


def join_body_lines(paragraphs, readings):
    # Models often split a wrapped line into two fields. Preserve the column and
    # text style, joining only the next aligned physical line of the same field.
    result = []
    for paragraph in paragraphs:
        current = dict(paragraph, lines=list(paragraph["lines"]))
        if result:
            prev = result[-1]
            a = readings[prev["lines"][-1]]["bbox"]
            b = readings[current["lines"][0]]["bbox"]
            if (
                prev.get("role") == current.get("role") == "body"
                and prev.get("color") == current.get("color")
                and prev.get("bold") == current.get("bold")
                and abs(a[0] - b[0]) < 0.8 * max(a[3], b[3])
                and 0.76 < a[3] / b[3] < 1.31
                and -0.25 * a[3] <= b[1] - a[1] - a[3] <= 0.85 * a[3]
            ):
                prev["lines"].extend(current["lines"])
                prev["list"] = (
                    prev.get("list")
                    or current.get("list")
                    or readings[current["lines"][0]]["text"].startswith(("·", "•"))
                )
                continue
        result.append(current)
    return result


def separate_icon_rows(paragraphs, readings, objects):
    """Text aligned to distinct left-hand icons represents independently editable rows."""
    result = []
    icons = [o for o in objects if o["kind"] == "icon"]
    for para in paragraphs:
        pieces = []
        for key in para["lines"]:
            partner = row_partner(readings[key]["bbox"], icons)
            previous = next((p for p, _ in reversed(pieces) if p), None)
            if pieces and partner and partner != previous:
                result.append(dict(para, lines=[a[1] for a in pieces]))
                pieces = []
            pieces.append((partner, key))
        if pieces:
            result.append(dict(para, lines=[a[1] for a in pieces]))
    return result


def routed_line(obj, image, texts, base):
    """Use source-supported runs. A return line's bounding rectangle is not a diagonal."""
    x, y, w, h = base["bbox"]
    if min(w, h) < max(w, h) * 0.035:
        return None
    path = obj.get("path")
    covered = [t["bbox"] for t in texts if overlap(t["bbox"], base["bbox"])]
    if not path and not covered:
        return None
    view = np.asarray(image, dtype=np.int16)
    W, H = image.size
    stroke=rgb(obj.get("stroke"), None)
    if stroke and (obj.get("arrow_end") or len(path or [])>2) and w>3*h:
        margin=max(2,round(.11*h))
        l,t=max(0,x-margin),max(0,y-margin)
        r,b=min(W,x+w+margin),min(H,y+h+margin)
        crop=view[t:b,l:r]
        tint=np.array(list(bytes.fromhex(stroke)))
        ink=(np.max(abs(crop-tint),axis=2)<48).astype(np.uint8)
        for tx,ty,tw,th in covered:
            ink[max(0,ty-t):max(0,ty+th-t),max(0,tx-l):max(0,tx+tw-l)]=0
        outlines,_=cv2.findContours(ink,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        traces=[]
        for contour in outlines:
            u,v,s,hh=cv2.boundingRect(contour)
            if not (s>.45*w or (hh>.34*h and min(abs(l+u-x),abs(l+u-x-w))<.11*w)):
                continue
            area=cv2.contourArea(contour)
            if not 0<area/(s*hh)<.3 or area>6*(s+hh) or min(s,hh)<2:
                continue
            vertices=cv2.approxPolyDP(contour,.65,True).reshape(-1,2)
            if not 3<=len(vertices)<=256:
                continue
            local=np.rint((vertices-[u,v])*100000/[s,hh]).astype(int).tolist()
            traces.append(dict(base,id=base["id"] if not traces else base["id"]+"-"+str(len(traces)),
                               bbox=[l+u,t+v,s,hh],evidence_ids=[] if traces else base["evidence_ids"],
                               geometry="line",outline=local,fill=stroke))
        if any(n["bbox"][2]>.45*w for n in traces):
            return traces
    # Coloured or grey line on a contrasting ground: measure the change across its normal.
    output = []
    coords = []
    for pt in path or []:
        if (
            isinstance(pt, list)
            and len(pt) == 2
            and all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in pt)
        ):
            coords.append((pt[0] * W / 1000, pt[1] * H / 1000))
    if len(coords) < 2:
        coords = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
    for a, b in zip(coords, coords[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        horizontal = abs(dx) >= abs(dy)
        if max(abs(dx), abs(dy)) < 3:
            continue
        if min(abs(dx), abs(dy)) > max(abs(dx), abs(dy)) * 0.18:
            continue
        arr = view if horizontal else view.transpose(1, 0, 2)
        p, q = (a, b) if horizontal else (a[::-1], b[::-1])
        lo, hi = max(0, round(min(p[0], q[0]))), min(arr.shape[1], round(max(p[0], q[0])))
        normal = round((p[1] + q[1]) / 2)
        margin = max(4, round(min(w, h) * 0.16))
        best = None
        for v in range(max(3, normal - margin), min(arr.shape[0] - 3, normal + margin + 1)):
            if hi - lo < 3:
                continue
            band = arr[v, lo:hi]
            contrast = (
                np.minimum(
                    np.max(abs(band - arr[v - 3, lo:hi]), axis=1),
                    np.max(abs(band - arr[v + 3, lo:hi]), axis=1),
                )
                > 12
            )
            for tx, ty, tw, th in covered:
                along, across, span, depth = (tx, ty, tw, th) if horizontal else (ty, tx, th, tw)
                if across <= v < across + depth:
                    contrast[max(0, along - lo) : max(0, along + span - lo)] = False
            changes = np.diff(np.r_[False, contrast, False].astype(int))
            for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
                score = (end - start) / (1 + 0.012 * abs(v - normal))
                if end - start >= max(6, (hi - lo) * 0.30) and (best is None or score > best[0]):
                    best = (score, int(lo + start), int(lo + end), v, band[(start + end) // 2])
        if best:
            _, start, end, v, color = best
            box = [start, v, end - start, 1.5] if horizontal else [v, start, 1.5, end - start]
            output.append(
                dict(
                    base,
                    id=base["id"] if not output else base["id"] + "-" + str(len(output)),
                    evidence_ids=[] if output else base["evidence_ids"],
                    bbox=box,
                    geometry="rect",
                    fill="".join(f"{c:02X}" for c in color),
                )
            )
    # Recover arrowheads at the outer edges, even when the proposal stops at the elbow.
    if output and (obj.get("arrow_end") or "arrow" in obj["label"].lower()):
        target = rgb(obj.get("stroke"), output[0]["fill"])
        tint = np.array(list(bytes.fromhex(target)), dtype=float)
        pad = max(4, round(h * 0.12))
        l, t = max(0, x - pad), max(0, y - pad)
        region = view[t : min(H, y + h + pad), l : min(W, x + w + pad)]
        mask = (np.max(abs(region - tint), axis=2) < 48).astype(np.uint8)
        core = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(core)
        for index, (u, v, s, hh, area) in enumerate(stats[1:], 1):
            if not (4 <= hh <= h * 0.14 and 5 <= s <= h * 0.2 and 0.65 < s / hh < 2.8):
                continue
            bx, by = int(l + u), int(t + v)
            if min(abs(bx - x), abs(bx - x - w)) > max(2 * s, w * 0.04):
                continue
            if any(overlap([bx, by, s, hh], b) for b in covered):
                continue
            widths = (labels[v : v + hh, u : u + s] == index).sum(axis=0)
            cut = max(1, s // 3)
            asym = widths[:cut].mean() - widths[-cut:].mean()
            if abs(asym) < 0.15 * hh:
                continue
            output.append(
                dict(
                    base,
                    id=base["id"] + "-tip" + str(index),
                    evidence_ids=[],
                    bbox=[bx, by, int(s), int(hh)],
                    geometry="rightArrow" if asym > 0 else "leftArrow",
                    fill=target,
                )
            )
    return output


def build_canvas(
    directory,
    source,
    plan,
    *,
    pack,
    layerd_url,
    token_file,
    inpaint_url,
    max_iterations,
    timeout,
    scenery=None,
):
    with Image.open(source) as im:
        image = im.convert("RGB")
    W, H = image.size
    assets_dir = directory / "assets"
    assets_dir.mkdir()
    nodes, groups, assets, evidence, remaining, notes, attempts = [], [], [], [], [], [], []
    unresolved = []
    readings = {t["id"]: t for t in plan["readings"]}
    editable_ink = {
        i: dict(readings[i], color=rgb(p.get("color"), "172641"))
        for module in plan["modules"]
        for p in module["paragraphs"]
        for i in p["lines"]
    }
    for row in readings.values():
        evidence.append(dict(id=row["id"], kind="text", text=row["text"], bbox=row["bbox"]))
    background = plan["background"]
    color = rgb(background.get("color"), "FFFFFF")
    page_background = {"color": color}
    if scenery is not None:
        import shutil

        shutil.copyfile(scenery, assets_dir / "scenery.png")
        assets.append(
            dict(
                id="scenery",
                path="assets/scenery.png",
                purpose="cleaned_background",
                sha256=sha256(scenery),
                contains_evidence_ids=[],
            )
        )
        page_background["asset_id"] = "scenery"
    elif str(background.get("scenery") or "").lower() not in ("", "none", "null", "无"):
        notes.append("Scenery simplified to editable background: " + str(background["scenery"]))
    if scenery is None and rgb(background.get("end_color"), None):
        nodes.append(
            dict(
                id="canvas-gradient",
                type="shape",
                bbox=[0, 0, W, H],
                evidence_ids=[],
                **shape_style(dict(background, fill=color, geometry="rect")),
            )
        )
    page = None
    if pack:
        page = load_page(pack)
        with Image.open(page.source_path) as ref:
            if not np.array_equal(np.asarray(ref.convert("RGB")), np.asarray(image)):
                raise ValueError("Layer pack does not match the input page")
    remote_error = None
    rejected_icon_readings = []
    for module in plan["modules"]:
        first = len(nodes)
        if module.get("planning_warning"):
            notes.append(module["id"] + ": local text planning: " + module["planning_warning"])
        native_text = {}
        inside = {entry["line"]: entry["asset"] for entry in module["inside"]}
        paragraphs = separate_icon_rows(
            join_body_lines(module["paragraphs"], readings), readings, module["objects"]
        )
        for para in paragraphs:
            for key in para["lines"]:
                ink = rgb(para.get("color"), None)
                if ink is None:
                    x, y, w, h = readings[key]["bbox"]
                    patch = np.asarray(image.crop((x, y, x + w, y + h)), dtype=float)
                    ground = np.median(
                        np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]]), axis=0
                    )
                    delta = np.max(np.abs(patch - ground), axis=2)
                    strokes = patch[delta >= np.percentile(delta, 86)]
                    ink = "".join(f"{int(c):02X}" for c in np.median(strokes, axis=0))
                native_text[key] = dict(readings[key], color=ink)
        for obj in module["objects"]:
            key, box = obj["id"], obj["bbox"]
            if obj["kind"] == "icon":
                box = measured_icon(image, box, native_text.values(), obj.get("geometry") == "ellipse")
            kind = "shape" if obj["kind"] == "shape" else "image"
            evidence.append(dict(id="ev-" + key, kind=kind, description=obj["label"], bbox=box))
            base = dict(id=key, bbox=box, type=kind, evidence_ids=["ev-" + key])
            if kind == "shape":
                if obj.get("geometry") == "line":
                    segments = routed_line(obj, image, native_text.values(), base)
                    if segments is not None:
                        nodes.extend(segments)
                        if not segments:
                            unresolved.extend(base["evidence_ids"])
                            notes.append(key + ": route lacks source-supported segments")
                        continue
                style = shape_style(obj)
                if style["geometry"] in ("triangle", "rightArrow", "leftArrow", "upArrow", "downArrow"):
                    direction = str(obj.get("direction", "")).lower()
                    if not direction:
                        match = re.search(r"(up|down|left|right)[_ -]?arrow", obj["label"], re.I)
                        direction = match.group(1).lower() if match else ""
                    if direction in ("up", "down", "left", "right"):
                        style["geometry"] = direction + "Arrow"
                nodes.append(dict(base, **style))
                continue
            x, y, w, h = box
            crop = image.crop((x, y, x + w, y + h))
            if w * h >= 0.8 * W * H:
                raise ValueError("A full-slide asset cannot be used in native synthesis")
            overlay = [
                t
                for t in editable_ink.values()
                if overlap(box, t["bbox"]) > 0.35 * t["bbox"][2] * t["bbox"][3]
            ]
            bands = [
                s
                for s in module["objects"]
                if s["kind"] == "shape"
                and s["bbox"][2] * s["bbox"][3] < 0.8 * w * h
                and any(
                    overlap(s["bbox"], t["bbox"]) > 0.6 * t["bbox"][2] * t["bbox"][3] for t in overlay
                )
            ]
            image_notes = []
            if obj["kind"] == "photo" or overlay:
                crop = photo_asset(crop, box, overlay, bands, inpaint_url, image_notes)
            else:
                pixels = np.asarray(crop).astype(float)
                matte = layer_icon(page, box, pixels) if page else None
                border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
                bg = np.median(border, axis=0)
                uniform = np.percentile(np.max(np.abs(border - bg), axis=1), 85) < 16
                if matte is None and not uniform and not page and layerd_url and remote_error is None:
                    target = directory / (key + "-source.png")
                    crop.save(target)
                    try:
                        result, _ = decompose(
                            target,
                            directory / (key + "-layers"),
                            endpoint=layerd_url,
                            token_file=token_file,
                            max_iterations=max_iterations,
                            timeout=timeout,
                        )
                        matte = layer_icon(load_page(result), [0, 0, w, h], pixels)
                        attempts.append({"id": key, "model": "LayerD", "accepted": matte is not None})
                    except (OSError, ValueError, RuntimeError, SubprocessError) as error:
                        remote_error = type(error).__name__ + ": " + str(error)[:180]
                if matte is None and uniform:
                    mask = np.max(np.abs(pixels - bg), axis=2) > 7
                    matte = Image.fromarray(
                        extract_rgba(pixels, np.broadcast_to(bg, pixels.shape), mask)
                    )
                if matte is None:
                    matte = optional_matte(crop, obj, min(timeout, 120), attempts)
                if matte is not None and obj["kind"] == "icon":
                    # Compare visible detail against the source before accepting a single layer.
                    # In particular, one layer may contain only the blue disc behind a white glyph.
                    canvas = Image.new("RGB", crop.size, tuple(bg.astype(int)))
                    canvas.paste(matte.convert("RGB"), mask=matte.getchannel("A"))
                    distance = np.max(abs(np.asarray(canvas, dtype=float) - pixels), axis=2)
                    if np.any(cv2.boxFilter((distance > 34).astype(np.float32), -1, (5, 5)) > 0.72):
                        matte = None
                        image_notes.append("Kept complete icon crop after detail comparison")
                if matte is not None:
                    crop = matte
                else:
                    image_notes.append(
                        "Retained bounded raster; " + (remote_error or "no isolated foreground")
                    )
            owned = [i for i, a in inside.items() if a == key]
            for i in owned:
                claim = next(c for c in module["inside"] if c["line"] == i)
                if obj["kind"] == "icon" and "icon artwork" in str(claim.get("reason", "")).lower():
                    rejected_icon_readings.append(dict(line=i, ocr=readings[i]["text"], asset=key))
                    continue
                remaining.append(
                    {"asset": key, "bbox": readings[i]["bbox"], "text": readings[i]["text"]}
                )
            if obj.get("embedded"):
                remaining.append({"asset": key, "bbox": box, "description": obj["embedded"]})
            relative = "assets/" + key + ".png"
            crop.save(directory / relative)
            assets.append(
                dict(
                    id=key,
                    path=relative,
                    purpose="module",
                    sha256=sha256(directory / relative),
                    contains_evidence_ids=["ev-" + key, *owned],
                )
            )
            nodes.append(dict(base, asset_id=key, evidence_ids=[]))
            notes.extend(key + ": " + n for n in image_notes)
        for para in paragraphs:
            rows = [native_text[i] for i in para["lines"]]
            spelling = [
                (
                    r["candidate"]
                    if r.get("confirmed") and not r.get("preserve_source_spelling")
                    else r["text"]
                )
                for r in rows
            ]
            words = spelling[0]
            for word in spelling[1:]:
                if para.get("list"):
                    words += "\n"
                elif (
                    words[-1].isascii()
                    and words[-1].isalnum()
                    and word[0].isascii()
                    and word[0].isalnum()
                ):
                    words += " "
                words += word
            field, size, wrap = fit_field(words, rows, para, module, native_text, (W, H))
            nodes.append(
                dict(
                    id="text-" + rows[0]["id"],
                    type="text",
                    bbox=field,
                    text=words,
                    font={"latin": "Arial", "ea": "Microsoft YaHei", "size_pt": size},
                    color=rows[0]["color"],
                    bold=bool(para.get("bold")),
                    align=para.get("align", "left"),
                    wrap=wrap,
                    evidence_ids=para["lines"],
                    text_override_reason="Line assembly and recorded source readings in component-plan.json",
                )
            )
        ordered, children, members, workflow = order_edit_groups(module, nodes[first:], readings)
        nodes[first:] = ordered
        groups.extend(children)
        # PowerPoint's GroupItems flattens nested shapes when selecting via its edit API.
        # Expose each numbered workflow step directly; preserve whole-card movement elsewhere.
        if module["id"] != "page" and len(members) >= 2 and not workflow:
            groups.append(dict(id="group-" + module["id"], member_ids=members))
    for z, node in enumerate(nodes):
        node["z"] = z
    ir = dict(
        schema_version="0.1",
        source_image="source.png",
        source_sha256=sha256(source),
        canvas={"source_px": [W, H], "target_px": [W, H]},
        background=page_background,
        review_status="needs_review",
        review_notes=notes,
        nodes=nodes,
        groups=groups,
        evidence=evidence,
        unresolved_evidence_ids=unresolved,
        assets=assets,
    )
    return ir, dict(
        icon_strokes_misread_as_text=rejected_icon_readings,
        embedded_text=remaining,
        object_notes=notes,
        model_attempts=attempts,
        groups=groups,
        unconfirmed_readings=[
            r
            for r in readings.values()
            if r.get("confirmed") is False or r["score"] is None or r["score"] < 0.94
        ],
    )
