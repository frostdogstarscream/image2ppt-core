"""Small OOXML adapter: preserve real grouping and East Asian font properties.

This does not author slides. It adapts an Artifact Tool-produced package and
audits its native objects using the standard library.
"""
import argparse
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

SLIDE_PART = "ppt/slides/slide{}.xml"
SLIDE_PART_RE = re.compile(r"^ppt/slides/slide(\d+)\.xml$")


def q(name):
    prefix, local = name.split(":")
    return "{" + NS[prefix] + "}" + local


def _text_properties(tag, style):
    props = ET.Element(q(tag), {"sz": str(round(style["font"]["size_pt"] * 100)),
                               "b": "1" if style.get("bold") else "0", "dirty": "0"})
    fill = ET.SubElement(props, q("a:solidFill"))
    ET.SubElement(fill, q("a:srgbClr"), {"val": style["color"]})
    for role, family in (("latin", "latin"), ("ea", "ea"), ("cs", "ea")):
        ET.SubElement(props, q("a:" + role), {"typeface": style["font"][family]})
    return props


def _replace_paragraphs(item, node, ir):
    body = item.find("p:txBody", NS)
    if body is None:
        raise ValueError("原生文本框缺少 txBody: " + node["id"])
    for existing in list(body.findall("a:p", NS)):
        body.remove(existing)
    scale = ir["canvas"]["target_px"][0] / ir["canvas"]["source_px"][0]
    for spec in node["paragraphs"]:
        paragraph = ET.SubElement(body, q("a:p"))
        attrs = {"algn": {"left": "l", "center": "ctr", "right": "r"}[spec.get("align", "left")]}
        if "left_indent_px" in spec:
            attrs["marL"] = str(round(spec["left_indent_px"] * scale * 9525))
        props = ET.SubElement(paragraph, q("a:pPr"), attrs)
        for field, tag in (("line_spacing_pt", "lnSpc"), ("space_before_pt", "spcBef"), ("space_after_pt", "spcAft")):
            if field in spec:
                spacing = ET.SubElement(props, q("a:" + tag))
                ET.SubElement(spacing, q("a:spcPts"), {"val": str(round(spec[field] * 100))})
        props.append(_text_properties("a:defRPr", spec))
        for run in spec.get("runs", [{"text": spec["text"]}]):
            style = {**spec, **run}
            element = ET.SubElement(paragraph, q("a:r"))
            element.append(_text_properties("a:rPr", style))
            text = ET.SubElement(element, q("a:t"), {"{http://www.w3.org/XML/1998/namespace}space": "preserve"})
            text.text = run["text"]
        paragraph.append(_text_properties("a:endParaRPr", spec))


def adapt_slide(xml, ir):
    """Apply IR-driven OOXML fixes to one already-parsed slide tree, in place.

    Zero text insets, East Asian/latin run fonts, explicit line spacing, and
    real PPTX groups. One IR describes exactly one slide.
    """
    tree = xml.find("p:cSld/p:spTree", NS)
    objects = {}
    for item in tree:
        nv = item.find(".//p:cNvPr", NS)
        if nv is not None:
            objects[nv.get("name")] = item
    ids = [int(e.get("id", "0")) for e in xml.findall(".//p:cNvPr", NS)]
    sw, sh = ir["canvas"]["source_px"]
    tw, th = ir["canvas"]["target_px"]
    scale = tw / sw

    def _set_alpha(clr_el, opacity):
        if clr_el is None or opacity is None:
            return
        val = max(0, min(100000, round(opacity * 100000)))
        if val >= 100000:
            return
        for old_a in list(clr_el.findall("a:alpha", NS)):
            clr_el.remove(old_a)
        ET.SubElement(clr_el, q("a:alpha"), {"val": str(val)})

    for node in ir["nodes"]:
        item = objects.get(node["id"])
        if item is None:
            raise ValueError("导出对象没有保留节点名称: " + node["id"])
        if node["type"] == "shape":
            sp_pr = item.find("p:spPr", NS)
            if sp_pr is not None:
                if node.get("geometry") == "custom" and node.get("custom_paths"):
                    prst = sp_pr.find("a:prstGeom", NS)
                    if prst is not None:
                        p_idx = list(sp_pr).index(prst)
                        sp_pr.remove(prst)
                        cg = ET.Element(q("a:custGeom"))
                        for tag_name in ("a:avLst", "a:gdLst", "a:ahLst", "a:cxnLst"):
                            ET.SubElement(cg, q(tag_name))
                        ET.SubElement(cg, q("a:rect"), {"l": "l", "t": "t", "r": "r", "b": "b"})
                        path_lst = ET.SubElement(cg, q("a:pathLst"))
                        for p in node["custom_paths"]:
                            pa = ET.SubElement(path_lst, q("a:path"), {
                                "w": str(max(1, round(p["width"] * 100))),
                                "h": str(max(1, round(p["height"] * 100))),
                            })
                            for cmd in p["commands"]:
                                ctype = cmd["type"]
                                if ctype == "close":
                                    ET.SubElement(pa, q("a:close"))
                                elif ctype == "cubicBezierTo":
                                    cb = ET.SubElement(pa, q("a:cubicBezTo"))
                                    for kx, ky in (("x1", "y1"), ("x2", "y2"), ("x", "y")):
                                        ET.SubElement(cb, q("a:pt"), {
                                            "x": str(round(cmd[kx] * 100)),
                                            "y": str(round(cmd[ky] * 100)),
                                        })
                                else:
                                    tag = "a:moveTo" if ctype == "moveTo" else "a:lnTo"
                                    el = ET.SubElement(pa, q(tag))
                                    ET.SubElement(el, q("a:pt"), {
                                        "x": str(round(cmd["x"] * 100)),
                                        "y": str(round(cmd["y"] * 100)),
                                    })
                        sp_pr.insert(p_idx, cg)
                if node.get("fill_opacity") is not None:
                    _set_alpha(sp_pr.find("a:solidFill/a:srgbClr", NS), node["fill_opacity"])
                if node.get("fill_gradient"):
                    gs_list = sp_pr.findall("a:gradFill/a:gsLst/a:gs", NS)
                    for gs_spec, gs_el in zip(node["fill_gradient"]["stops"], gs_list):
                        _set_alpha(gs_el.find("a:srgbClr", NS), gs_spec.get("opacity"))
                line_spec = node.get("line")
                ln_el = sp_pr.find("a:ln", NS)
                if line_spec and ln_el is not None:
                    _set_alpha(ln_el.find("a:solidFill/a:srgbClr", NS), line_spec.get("opacity"))
                    for end_key, end_tag in (("head_end", "a:headEnd"), ("tail_end", "a:tailEnd")):
                        for old_end in list(ln_el.findall(end_tag, NS)):
                            ln_el.remove(old_end)
                    insert_pos = len(list(ln_el))
                    for idx_c, ch in enumerate(list(ln_el)):
                        if ch.tag == q("a:extLst"):
                            insert_pos = idx_c
                            break
                    for end_key, end_tag in (("head_end", "a:headEnd"), ("tail_end", "a:tailEnd")):
                        end_type = line_spec.get(end_key)
                        if end_type and end_type != "none":
                            ln_el.insert(insert_pos, ET.Element(q(end_tag), {"type": end_type, "w": "med", "len": "med"}))
                            insert_pos += 1
                sh_spec = node.get("shadow")
                if sh_spec:
                    for old_eff in list(sp_pr.findall("a:effectLst", NS)):
                        sp_pr.remove(old_eff)
                    blur_emu = max(0, round(sh_spec["blur_px"] * scale * 9525))
                    dx = sh_spec["offset_x_px"] * scale
                    dy = sh_spec["offset_y_px"] * scale
                    dist_emu = max(0, round(math.hypot(dx, dy) * 9525))
                    dir_deg = math.degrees(math.atan2(dy, dx)) % 360.0 if (dx or dy) else 90.0
                    dir_60k = round(dir_deg * 60000) % 21600000
                    eff = ET.Element(q("a:effectLst"))
                    outer = ET.SubElement(eff, q("a:outerShdw"), {
                        "blurRad": str(blur_emu),
                        "dist": str(dist_emu),
                        "dir": str(dir_60k),
                        "algn": "tl",
                        "rotWithShape": "0",
                    })
                    srgb = ET.SubElement(outer, q("a:srgbClr"), {"val": sh_spec["color"]})
                    _set_alpha(srgb, sh_spec["opacity"])
                    insert_at = len(list(sp_pr))
                    for idx_c, ch in enumerate(list(sp_pr)):
                        if ch.tag in (q("a:scene3d"), q("a:sp3d"), q("a:extLst")):
                            insert_at = idx_c
                            break
                    sp_pr.insert(insert_at, eff)
            continue
        if node["type"] == "image":
            clip = node.get("clip_geometry")
            cr = node.get("corner_radius_px")
            if clip in ("ellipse", "roundRect") or (cr is not None and cr > 0):
                sp_pr = item.find("p:spPr", NS)
                prst = sp_pr.find("a:prstGeom", NS) if sp_pr is not None else None
                if prst is not None:
                    for ch in list(prst):
                        prst.remove(ch)
                    if clip == "ellipse":
                        prst.set("prst", "ellipse")
                        ET.SubElement(prst, q("a:avLst"))
                    else:
                        prst.set("prst", "roundRect")
                        min_side = max(1.0, min(float(node["bbox"][2]), float(node["bbox"][3])))
                        radius = float(cr) if cr is not None else min_side * 0.08
                        adj = min(50000, max(0, round(radius / min_side * 100000)))
                        av = ET.SubElement(prst, q("a:avLst"))
                        ET.SubElement(av, q("a:gd"), {"name": "adj", "fmla": f"val {adj}"})
            continue
        if node["type"] == "table":
            for rpr in item.findall(".//a:rPr", NS) + item.findall(".//a:defRPr", NS) + item.findall(".//a:endParaRPr", NS):
                for key, field in [("a:latin", "latin"), ("a:ea", "ea")]:
                    font = rpr.find(key, NS)
                    if font is None:
                        font = ET.SubElement(rpr, q(key))
                    font.set("typeface", node["font"][field])
            continue
        if node["type"] == "text":
            # Artifact Tool currently omits zero insets from bodyPr. OOXML
            # defaults would reintroduce 9.6px horizontal / 4.8px vertical
            # margins on import, clipping dense restored text.
            body = item.find("p:txBody/a:bodyPr", NS)
            if body is not None:
                # IR insets are target pixels; OOXML bodyPr uses EMU at 96 dpi.
                insets = node.get("insets", [0, 0, 0, 0])
                for key, value in zip(("lIns", "rIns", "tIns", "bIns"), insets):
                    body.set(key, str(round(value * 9525)))
                if "vertical_anchor" in node:
                    body.set("anchor", {"top": "t", "middle": "ctr",
                                        "bottom": "b"}[node["vertical_anchor"]])
            if node.get("paragraphs"):
                _replace_paragraphs(item, node, ir)
                continue
            if "line_spacing_pt" in node:
                for paragraph in item.findall("p:txBody/a:p", NS):
                    props = paragraph.find("a:pPr", NS)
                    if props is None:
                        props = ET.Element(q("a:pPr"))
                        paragraph.insert(0, props)
                    spacing = props.find("a:lnSpc", NS)
                    if spacing is None:
                        spacing = ET.Element(q("a:lnSpc"))
                        props.insert(0, spacing)
                    for child in list(spacing):
                        spacing.remove(child)
                    ET.SubElement(spacing, q("a:spcPts"), {"val": str(round(node["line_spacing_pt"] * 100))})
            for rpr in item.findall(".//a:rPr", NS) + item.findall(".//a:defRPr", NS) + item.findall(".//a:endParaRPr", NS):
                for key, field in [("a:latin", "latin"), ("a:ea", "ea")]:
                    font = rpr.find(key, NS)
                    if font is None:
                        font = ET.SubElement(rpr, q(key))
                    font.set("typeface", node["font"][field])
    sw, sh = ir["canvas"]["source_px"]
    tw, th = ir["canvas"]["target_px"]
    for group in ir["groups"]:
        members = [objects[id] for id in group["member_ids"]]
        first = min(list(tree).index(member) for member in members)
        member_nodes = [n for n in ir["nodes"] if n["id"] in group["member_ids"]]
        left = min(n["bbox"][0] for n in member_nodes)
        top = min(n["bbox"][1] for n in member_nodes)
        right = max(n["bbox"][0] + n["bbox"][2] for n in member_nodes)
        bottom = max(n["bbox"][1] + n["bbox"][3] for n in member_nodes)
        x, y = round(left / sw * tw * 9525), round(top / sh * th * 9525)
        w, h = round((right - left) / sw * tw * 9525), round((bottom - top) / sh * th * 9525)
        grp = ET.Element(q("p:grpSp"))
        nv = ET.SubElement(grp, q("p:nvGrpSpPr"))
        new_id = max(ids) + 1
        ids.append(new_id)
        ET.SubElement(nv, q("p:cNvPr"), {"id": str(new_id), "name": group["id"]})
        ET.SubElement(nv, q("p:cNvGrpSpPr"))
        ET.SubElement(nv, q("p:nvPr"))
        props = ET.SubElement(grp, q("p:grpSpPr"))
        transform = ET.SubElement(props, q("a:xfrm"))
        for tag, attrs in [("a:off", {"x": x, "y": y}), ("a:ext", {"cx": w, "cy": h}),
                           ("a:chOff", {"x": x, "y": y}), ("a:chExt", {"cx": w, "cy": h})]:
            ET.SubElement(transform, q(tag), {k: str(v) for k, v in attrs.items()})
        for member in sorted(members, key=lambda e: list(tree).index(e)):
            tree.remove(member)
            grp.append(member)
        tree.insert(first, grp)
    return xml


def adapt(source, target, irfiles):
    """Adapt a package where slide N is adapted against the Nth IR.

    A single-IR call reproduces the original single-slide adapter exactly.
    The IR order defines the slide order.
    """
    irs = [json.loads(Path(f).read_text(encoding="utf-8-sig")) for f in irfiles]
    target = Path(target)
    if target.exists():
        raise ValueError("禁止覆盖已有 PPTX。")
    with ZipFile(source) as package:
        by_index = {}
        for name in package.namelist():
            match = SLIDE_PART_RE.match(name)
            if match:
                by_index[int(match.group(1))] = name
        if len(by_index) != len(irs):
            raise ValueError("源包有 %d 页，但提供了 %d 个 IR；页数与 IR 必须一致。"
                             % (len(by_index), len(irs)))
        if sorted(by_index) != list(range(1, len(irs) + 1)):
            raise ValueError("幻灯片部件不是连续的 slide1..slideN: " + ", ".join(sorted(by_index)))
        rewritten = {}
        for index in range(1, len(irs) + 1):
            part = by_index[index]
            xml = ET.fromstring(package.read(part))
            rewritten[part] = ET.tostring(adapt_slide(xml, irs[index - 1]), encoding="utf-8", xml_declaration=True)
        with ZipFile(target, "x", compression=ZIP_DEFLATED) as output:
            for entry in package.infolist():
                output.writestr(entry, rewritten.get(entry.filename) or package.read(entry.filename))


def inspect(file):
    with ZipFile(file) as package:
        if package.testzip() is not None:
            raise ValueError("PPTX CRC 校验失败。")
        # Slide order is numeric (slide2 precedes slide10), never lexicographic.
        indexed = {}
        for name in package.namelist():
            match = SLIDE_PART_RE.match(name)
            if match:
                indexed[int(match.group(1))] = name
        slides = [indexed[index] for index in sorted(indexed)]
        result = []
        for name in slides:
            root = ET.fromstring(package.read(name))
            result.append({"part": name, "text_nodes": len(root.findall(".//p:sp/p:txBody", NS)),
                           "shapes": len(root.findall(".//p:sp", NS)) + len(root.findall(".//p:cxnSp", NS)), "images": len(root.findall(".//p:pic", NS)),
                           "groups": len(root.findall(".//p:grpSp", NS)),
                           "text": [t.text or "" for t in root.findall(".//a:t", NS)],
                           "names": [e.get("name") for e in root.findall(".//p:cNvPr", NS)]})
        return {"slide_count": len(slides), "crc_passed": True, "slides": result}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["adapt", "inspect"])
    p.add_argument("source")
    p.add_argument("--target")
    p.add_argument("--ir", action="append", help="IR for slide 1, 2, ...; repeat once per slide")
    args = p.parse_args()
    if args.command == "adapt":
        adapt(args.source, args.target, args.ir)
    print(json.dumps(inspect(args.target if args.command == "adapt" else args.source), ensure_ascii=False, indent=2))
