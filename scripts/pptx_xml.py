"""Small OOXML adapter: preserve real grouping and East Asian font properties.

This does not author slides. It adapts an Artifact Tool-produced package and
audits its native objects using the standard library.
"""
import argparse
import json
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
    for node in ir["nodes"]:
        item = objects.get(node["id"])
        if item is None:
            raise ValueError("导出对象没有保留节点名称: " + node["id"])
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
                           "shapes": len(root.findall(".//p:sp", NS)), "images": len(root.findall(".//p:pic", NS)),
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
