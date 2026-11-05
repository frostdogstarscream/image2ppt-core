"""Direct final-PPTX card edits with old-site and package-integrity checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image

from background_models import gradient_pixels, shape_mask, precise_text_masks
from rebuild_pipeline import ArtifactBackend, REPO, pixels, stamp
from slideir_adapter import write_json, sha256
from surface_rebuild import rewrite_package, ancestry
from text_repair import NS, load_json
from text_repair_acceptance import verify_styles, delete_deck_text


def verify_native_shape_properties(pptx, ir, slide=1):
    with ZipFile(pptx) as package:
        root = ET.fromstring(package.read(f"ppt/slides/slide{slide}.xml"))
    shapes = {element.find("p:nvSpPr/p:cNvPr", NS).get("name"): element
              for element in root.findall(".//p:sp", NS)}
    records = []
    for node in ir["nodes"]:
        if node["type"] != "shape" or not node["id"].startswith(("surface-", "separator-", "bullet-", "icon-organization-")):
            continue
        props = shapes[node["id"]].find("p:spPr", NS)
        geometry = props.find("a:prstGeom", NS)
        assert geometry is not None and geometry.get("prst") == node["geometry"], node["id"]
        if node["geometry"] == "roundRect" and node.get("corner_radius_px"):
            adjustment = geometry.find("a:avLst/a:gd", NS)
            actual = int(adjustment.get("fmla").split()[-1])
            desired = round(node["corner_radius_px"] / min(node["bbox"][2:]) * 100000)
            assert abs(actual - desired) <= 2, node["id"]
        if node.get("fill_gradient"):
            gradient = props.find("a:gradFill", NS)
            assert gradient is not None, node["id"]
            stops = gradient.findall("a:gsLst/a:gs", NS)
            planned = node["fill_gradient"]["stops"]
            assert len(stops) == len(planned), node["id"]
            for stop, expected in zip(stops, planned):
                assert int(stop.get("pos")) == expected["offset"], node["id"]
                assert stop.find("a:srgbClr", NS).get("val") == expected["color"], node["id"]
        if node.get("line"):
            assert int(props.find("a:ln", NS).get("w")) == round(node["line"]["width_pt"] * 12700), node["id"]
        records.append({"id": node["id"], "geometry": node["geometry"], "status": "PASS"})
    return records


def change_card_style(source, target, node_id):
    with ZipFile(source) as package:
        root = ET.fromstring(package.read("ppt/slides/slide1.xml"))
        shape = next(item for item in root.findall(".//p:sp", NS)
                     if item.find("p:nvSpPr/p:cNvPr", NS).get("name") == node_id)
        props = shape.find("p:spPr", NS)
        for child in list(props):
            if child.tag in {"{" + NS["a"] + "}" + name for name in ("solidFill", "gradFill", "noFill")}:
                props.remove(child)
        fill = ET.Element("{" + NS["a"] + "}solidFill")
        ET.SubElement(fill, "{" + NS["a"] + "}srgbClr", {"val": "DDEEDD"})
        props.insert(2, fill)
        line = props.find("a:ln", NS)
        if line is not None:
            props.remove(line)
        line = ET.SubElement(props, "{" + NS["a"] + "}ln", {"w": "19050"})
        line_fill = ET.SubElement(line, "{" + NS["a"] + "}solidFill")
        ET.SubElement(line_fill, "{" + NS["a"] + "}srgbClr", {"val": "CC5500"})
        with ZipFile(target, "x", compression=ZIP_DEFLATED) as output:
            for entry in package.infolist():
                output.writestr(entry, ET.tostring(root, encoding="utf-8", xml_declaration=True)
                                if entry.filename == "ppt/slides/slide1.xml" else package.read(entry.filename))
    with ZipFile(source) as before, ZipFile(target) as after:
        assert all(before.read(name) == after.read(name) for name in before.namelist() if name != "ppt/slides/slide1.xml")
    return {"fill": "DDEEDD", "border": "CC5500", "border_width_pt": 1.5}


def ancestor_background(base, records, region):
    result = base.copy()
    parents = set(ancestry(records, region))
    size = (base.shape[1], base.shape[0])
    for parent in records:
        if parent["id"] not in parents:
            continue
        for part in parent["components"]:
            x, y, width, height = part["bbox"]
            mask = shape_mask(size, part["bbox"], part["radius"])
            if part["part"] == "top":
                mask |= shape_mask(size, [x, y + part["radius"], width, height - part["radius"]])
            elif part["part"] == "left":
                mask |= shape_mask(size, [x + part["radius"], y, width - part["radius"], height])
            local = gradient_pixels(part["model"], part["bbox"])
            selected = mask[y:y + height, x:x + width]
            result[y:y + height, x:x + width][selected] = local[selected]
    return result


def check_page(delivery, directory, backend):
    ir = load_json(delivery["ir"])
    size = ir["canvas"]["source_px"]
    models = load_json(Path(delivery["directory"]) / "surface-models.json")["regions"]
    parent_ids = {item["parent_id"] for item in models if item.get("parent_id")}
    # A leaf verifies its original site without suppressing independently
    # selectable child cards. Root cards are exercised on page three.
    leaves = [item for item in models if item["id"] not in parent_ids and item["bbox"][3] > 45]
    region = next((item for item in leaves if item.get("parent_id")), leaves[0])
    before_hash = sha256(delivery["pptx"])
    directory.mkdir()
    background_only = Path(delivery.get("background_pptx", Path(delivery["directory"]) / "background-only.pptx"))
    deleted = directory / "card-deleted.pptx"
    rewrite_package(background_only, deleted, remove={region["output_id"]})
    rendered = backend.render(deleted, directory / "card-deleted-render")
    actual = pixels(rendered["pages"][0]["path"], size)
    base = pixels(Path(delivery["directory"]) / "cleaned-base.png", size)
    expected = ancestor_background(base, models, region)
    mask = shape_mask(size, region["bbox"], region["components"][0]["radius"])
    mask = cv2.erode(mask.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
    # Separators and bullets have their own source evidence and stay editable.
    for node in ir["nodes"]:
        if node["type"] == "shape" and not node["id"].startswith("surface-"):
            x, y, width, height = [round(value) for value in node["bbox"]]
            mask[max(0, y - 3):y + height + 3, max(0, x - 3):x + width + 3] = False
    differences = np.abs(actual.astype(np.int16) - expected.astype(np.int16))[mask]
    old_site_mae = float(differences.mean())
    old_site_p95 = float(np.percentile(differences, 95))
    moved = directory / "card-moved-shrunk.pptx"
    rewrite_package(delivery["pptx"], moved, transform={"id": region["output_id"]})
    moved_render = backend.render(moved, directory / "card-moved-shrunk-render")
    styled = directory / "card-fill-border-changed.pptx"
    body_id = region["components"][0]["id"]
    style = change_card_style(delivery["pptx"], styled, body_id)
    styled_render = backend.render(styled, directory / "card-fill-border-render")
    with ZipFile(moved) as package:
        root = ET.fromstring(package.read("ppt/slides/slide1.xml"))
        changed = next(element for element in root.find("p:cSld/p:spTree", NS)
                       if element.find(".//p:cNvPr", NS) is not None
                       and element.find(".//p:cNvPr", NS).get("name") == region["output_id"])
        xfrm = changed.find("p:grpSpPr/a:xfrm", NS) if changed.tag.endswith("grpSp") else changed.find("p:spPr/a:xfrm", NS)
        recorded_transform = {tag: xfrm.find("a:" + tag, NS).attrib for tag in ("off", "ext")}
    with ZipFile(delivery["pptx"]) as original, ZipFile(moved) as edited:
        original_root = ET.fromstring(original.read("ppt/slides/slide1.xml"))
        original_tree = original_root.find("p:cSld/p:spTree", NS)
        edited_tree = root.find("p:cSld/p:spTree", NS)
        for before_element, after_element in zip(original_tree, edited_tree):
            identifier = before_element.find(".//p:cNvPr", NS)
            if identifier is None or identifier.get("name") != region["output_id"]:
                assert ET.tostring(before_element) == ET.tostring(after_element)
                continue
            original_xfrm = (before_element.find("p:grpSpPr/a:xfrm", NS)
                             if before_element.tag.endswith("grpSp") else before_element.find("p:spPr/a:xfrm", NS))
            before_off = original_xfrm.find("a:off", NS)
            before_extent = original_xfrm.find("a:ext", NS)
            assert int(recorded_transform["off"]["x"]) == int(before_off.get("x")) + 20 * 9525
            assert int(recorded_transform["off"]["y"]) == int(before_off.get("y")) + 12 * 9525
            assert int(recorded_transform["ext"]["cx"]) == round(int(before_extent.get("cx")) * 0.9)
            assert int(recorded_transform["ext"]["cy"]) == round(int(before_extent.get("cy")) * 0.9)
        assert all(original.read(name) == edited.read(name) for name in original.namelist()
                   if name != "ppt/slides/slide1.xml")
    if sha256(delivery["pptx"]) != before_hash:
        raise ValueError("direct edit probe altered delivery")
    return {"page": delivery["page"], "target": region["id"], "output_id": region["output_id"],
            "delete_old_site_mae": old_site_mae, "delete_old_site_p95": old_site_p95,
            "old_site_status": "PASS" if old_site_mae <= 2 and old_site_p95 <= 5 else "REVIEW_REQUIRED",
            "delete_preview": rendered["pages"][0]["path"], "move_px": [20, 12], "scale": 0.9,
            "transform": recorded_transform, "moved_pptx": str(moved.resolve()),
            "moved_preview": moved_render["pages"][0]["path"], "changed_style": style,
            "styled_pptx": str(styled.resolve()), "styled_preview": styled_render["pages"][0]["path"],
            "original_preserved": True}


def audit_batch(file):
    batch = load_json(file)
    baseline = {item["page"]: item for item in load_json(batch["input"])["pages"]}
    directory = Path(batch["directory"]) / ("acceptance-" + stamp())
    directory.mkdir()
    backend = ArtifactBackend(REPO, 600)
    deleted = directory / "all-native-text-deleted.pptx"
    delete_deck_text(batch["deck"]["pptx"], deleted, batch["pages"])
    deletion_render = backend.render(deleted, directory / "all-native-text-deleted-render")
    pages = []
    for index, delivery in enumerate(batch["pages"], 1):
        ir = load_json(delivery["ir"])
        original = load_json(baseline[delivery["page"]]["ir"])
        original_text = {node["id"]: {key: value for key, value in node.items() if key != "z"}
                         for node in original["nodes"] if node["type"] == "text"}
        final_text = {node["id"]: {key: value for key, value in node.items() if key != "z"}
                      for node in ir["nodes"] if node["type"] == "text"}
        assert original_text == final_text, "surface rebuild changed existing paragraph text or styles"
        mask_directory = directory / (delivery["page"][:2] + "-masks")
        mask_directory.mkdir()
        source = pixels(Path(delivery["ir"]).parent / ir["source_image"], ir["canvas"]["source_px"])
        protection = np.zeros(source.shape[:2], bool)
        for node in original["nodes"]:
            if node["type"] != "image" or node["id"] == "background-image":
                continue
            asset = next(item for item in original["assets"] if item["id"] == node["asset_id"])
            alpha = np.array(Image.open(Path(baseline[delivery["page"]]["ir"]).parent / asset["path"]).convert("RGBA"))[:, :, 3]
            x, y, width, height = map(int, node["bbox"])
            protection[y:y + height, x:x + width] |= cv2.resize(alpha, (width, height), interpolation=cv2.INTER_NEAREST) > 16
        Image.fromarray(protection.astype(np.uint8) * 255).save(mask_directory / "protected-visuals.png")
        inventory = load_json(Path(baseline[delivery["page"]]["directory"]) / "text-inventory.json")
        for line in inventory["lines"]:
            masks = precise_text_masks(source, line, protection)
            assert not masks["repair"][protection].any()
            for kind in ("glyph", "core", "repair", "inspection"):
                Image.fromarray(masks[kind].astype(np.uint8) * 255).save(mask_directory / f"{line['id']}-{kind}.png")
        styles = verify_styles(batch["deck"]["pptx"], ir, index)
        native_shapes = verify_native_shape_properties(batch["deck"]["pptx"], ir, index)
        actual = pixels(deletion_render["pages"][index - 1]["path"], ir["canvas"]["source_px"])
        single = pixels(delivery["deleted_text_preview"], ir["canvas"]["source_px"])
        composed_mae = float(np.abs(actual.astype(np.int16) - single.astype(np.int16)).mean())
        assert composed_mae <= 0.1
        record = {"page": delivery["page"], "style_checks": styles, "native_shape_checks": native_shapes,
                  "baseline_paragraphs_preserved": len(final_text), "masks": str(mask_directory.resolve()),
                  "composed_deleted_background_mae": composed_mae,
                  "deleted_text_preview": deletion_render["pages"][index - 1]["path"]}
        if delivery["page"][:2] in {"03", "04", "10"}:
            record["card_edit_probe"] = check_page(delivery, directory / delivery["page"][:2], backend)
        pages.append(record)
    result = {"schema": "SurfaceAcceptance/1", "directory": str(directory.resolve()), "pages": pages,
              "deleted_text_pptx": str(deleted.resolve()), "powerpoint_checked": False,
              "wps_checked": False, "human_acceptance": False}
    write_json(directory / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch")
    result = audit_batch(parser.parse_args().batch)
    print(json.dumps({"directory": result["directory"], "pages": len(result["pages"])}, ensure_ascii=False))
