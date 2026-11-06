"""Direct delivered-PPTX edits, independently compiled expectations and honest QA.

Matching an independently edited IR establishes export consistency only. It does
not establish that the hidden background is clean; that needs a source-bound,
separately verified clean-plate receipt. Original packages are always read-only.
"""
from __future__ import annotations

import copy
import json
import math
import posixpath
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from rebuild_pipeline import compile_ir, pixels
from slideir_adapter import sha256, write_json
from text_repair import NS
from qa_status import aggregate_checks, classify_edit_probe, summarize_edit_report


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _name(element):
    found = element.find(".//p:cNvPr", NS)
    return found.get("name") if found is not None else None


def target_members(ir, identifier):
    group = next((g for g in ir.get("groups", []) if g["id"] == identifier), None)
    if group:
        return group["member_ids"]
    if any(n["id"] == identifier for n in ir["nodes"]):
        return [identifier]
    # Source decorations may have separate native connected components. Move
    # those together in the diagnostic, without claiming a real PPTX group.
    components = [n["id"] for n in ir["nodes"] if n["id"].startswith(identifier+"-")
                  and n["id"][len(identifier)+1:].isdigit()]
    return components or [identifier]


def target_bbox(ir, identifier):
    ids = set(target_members(ir, identifier))
    boxes = [n["bbox"] for n in ir["nodes"] if n["id"] in ids]
    if not boxes or len(boxes) != len(ids):
        raise ValueError("Unknown probe target: " + identifier)
    x, y = min(b[0] for b in boxes), min(b[1] for b in boxes)
    return [x, y, max(b[0]+b[2] for b in boxes)-x, max(b[1]+b[3] for b in boxes)-y]


def replacement_strings(node, long=False):
    """Keep every paragraph and run, including emphasized runs, non-empty."""
    count = [len(p.get("runs") or [{}]) for p in node.get("paragraphs") or [node]]
    value = "编辑验证内容用于检查更长段落是否产生溢出" if long else "测"
    return [[value for _ in range(number)] for number in count]


def edited_ir(ir, target, action, delta=(20, 12)):
    result = copy.deepcopy(ir)
    ids = set(target_members(ir, target))
    if action == "delete":
        removed = [n for n in result["nodes"] if n["id"] in ids]
        result["nodes"] = [n for n in result["nodes"] if n["id"] not in ids]
        deleted_claims = {e for n in removed for e in n.get("evidence_ids", [])}
        used = {n["asset_id"] for n in result["nodes"] if n.get("asset_id")}
        if result.get("background", {}).get("asset_id"):
            used.add(result["background"]["asset_id"])
        for asset in result.get("assets", []):
            if asset["id"] not in used:
                deleted_claims.update(asset.get("contains_evidence_ids", []))
        result["assets"] = [a for a in result.get("assets", []) if a["id"] in used]
        result["groups"] = [{**g, "member_ids": [i for i in g["member_ids"] if i not in ids]}
                            for g in result.get("groups", [])]
        result["groups"] = [g for g in result["groups"] if len(g["member_ids"]) >= 2]
        # This is a diagnostic scene with intentionally removed source content,
        # not an incomplete reconstruction. Retain shared evidence if another
        # surviving object owns it, and drop only the target's exclusive claims.
        retained_claims = {e for n in result["nodes"] for e in n.get("evidence_ids", [])}
        retained_claims.update(e for a in result["assets"] for e in a.get("contains_evidence_ids", []))
        exclusive = deleted_claims - retained_claims
        result["evidence"] = [e for e in result.get("evidence", []) if e["id"] not in exclusive]
        result["unresolved_evidence_ids"] = [e for e in result.get("unresolved_evidence_ids", []) if e not in exclusive]
    else:
        for node in result["nodes"]:
            if node["id"] not in ids:
                continue
            if action == "move":
                node["bbox"][0] += delta[0]
                node["bbox"][1] += delta[1]
            elif action == "recolor":
                node.pop("fill_gradient", None)
                node["fill"] = "DDEEDD"
                node["line"] = {"color": "CC5500", "width_pt": 1.5}
            elif action in {"replace_short", "replace_long"}:
                rows = replacement_strings(node, action == "replace_long")
                for paragraph, strings in zip(node.get("paragraphs") or [node], rows):
                    if paragraph.get("runs"):
                        for run, text in zip(paragraph["runs"], strings):
                            run["text"] = text
                    paragraph["text"] = "".join(strings)
                node["text"] = "\n".join(p["text"] for p in node.get("paragraphs") or [node])
                node["text_override_reason"] = "Diagnostic edit probe; original delivery preserved"
            else:
                raise ValueError("Unsupported edit probe action")
    result["review_status"] = "needs_review"
    result.setdefault("review_notes", []).append("Diagnostic edit expectation only; not a replacement deliverable")
    return result


def _find(root, target):
    shapes = [s for s in root.iter() if s.tag in {"{"+NS["p"]+"}"+t for t in ("sp", "pic", "grpSp")}
              and _name(s) == target]
    if len(shapes) != 1:
        raise ValueError("Expected exactly one actual PPTX target: " + target)
    return shapes[0]


def slide_part(package, slide_number=1):
    """Resolve display order through presentation relationships, not filenames."""
    if not isinstance(slide_number, int) or isinstance(slide_number, bool) or slide_number < 1:
        raise ValueError("slide_number must be a positive one-based integer")
    presentation = ET.fromstring(package.read("ppt/presentation.xml"))
    slides = presentation.findall("p:sldIdLst/p:sldId", NS)
    if not slides:
        # Read compatibility for the old minimal single-slide unit fixture.
        actual = [n for n in package.namelist() if n.startswith("ppt/slides/")
                  and n.endswith(".xml") and "/_rels/" not in n]
        if slide_number == 1 and actual == ["ppt/slides/slide1.xml"]:
            return actual[0]
        raise ValueError("Presentation slide order is missing")
    if slide_number > len(slides):
        raise ValueError("Requested slide is absent from presentation")
    rid = slides[slide_number-1].get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
    rels = ET.fromstring(package.read("ppt/_rels/presentation.xml.rels"))
    found = [r for r in rels if r.get("Id") == rid]
    if len(found) != 1 or found[0].get("TargetMode") == "External" or not found[0].get("Type", "").endswith("/slide"):
        raise ValueError("Invalid slide relationship")
    target = found[0].get("Target", "")
    part = posixpath.normpath(target.lstrip("/") if target.startswith("/") else posixpath.join("ppt", target))
    if not part.startswith("ppt/slides/") or part not in package.namelist():
        raise ValueError("Invalid slide package part")
    return part


def _without_text(element):
    result = copy.deepcopy(element)
    for text in result.findall(".//a:t", NS):
        text.text = ""
    return ET.tostring(result)


def rewrite_probe(source, output, ir, target, action, delta=(20, 12), slide_number=1):
    """Change one actual object, preserving all other package parts and XML."""
    members = target_members(ir, target)
    actual_group = any(g["id"] == target for g in ir.get("groups", []))
    if len(members) > 1 and not actual_group:
        previous = source
        for index, member in enumerate(members):
            next_file = (Path(output) if index == len(members)-1 else
                         Path(output).with_name(Path(output).stem+f"-component-{index+1}.pptx"))
            rewrite_probe(previous, next_file, ir, member, action, delta, slide_number)
            previous = next_file
        return {"protected_xml": "PASS", "package_parts": "PASS",
                "semantic_component_count": len(members), "actual_pptx_group": False}
    with ZipFile(source) as package:
        part = slide_part(package, slide_number)
        root = ET.fromstring(package.read(part))
        before = copy.deepcopy(root)
        selected = _find(root, target)
        original_target = copy.deepcopy(selected)
        parent = next(p for p in root.iter() if selected in list(p))
        if action == "delete":
            parent.remove(selected)
        elif action == "move":
            props = selected.find("p:grpSpPr/a:xfrm", NS) if selected.tag.endswith("grpSp") else selected.find("p:spPr/a:xfrm", NS)
            off = props.find("a:off", NS)
            pres = ET.fromstring(package.read("ppt/presentation.xml"))
            dimensions = pres.find("p:sldSz", NS)
            scale = [int(dimensions.get(k))/ir["canvas"]["source_px"][i] for i, k in enumerate(("cx", "cy"))]
            # Child offsets are measured in the group's child coordinate system.
            ancestor = parent
            while ancestor is not None:
                if ancestor.tag.endswith("grpSp"):
                    transform = ancestor.find("p:grpSpPr/a:xfrm", NS)
                    ext, child = transform.find("a:ext", NS), transform.find("a:chExt", NS)
                    for i, axis in enumerate(("cx", "cy")):
                        scale[i] *= int(child.get(axis))/int(ext.get(axis))
                ancestor = next((p for p in root.iter() if ancestor in list(p)), None)
            for i, axis in enumerate(("x", "y")):
                off.set(axis, str(int(off.get(axis))+round(delta[i]*scale[i])))
        elif action == "recolor":
            props = selected.find("p:spPr", NS)
            if props is None:
                raise ValueError("Recolor target must be a native shape")
            for child in list(props):
                if child.tag in {"{"+NS["a"]+"}"+s for s in ("solidFill", "gradFill", "noFill", "ln")}:
                    props.remove(child)
            fill = ET.Element("{"+NS["a"]+"}solidFill")
            ET.SubElement(fill, "{"+NS["a"]+"}srgbClr", {"val": "DDEEDD"})
            props.insert(2, fill)
            line = ET.SubElement(props, "{"+NS["a"]+"}ln", {"w": "19050"})
            ET.SubElement(ET.SubElement(line, "{"+NS["a"]+"}solidFill"), "{"+NS["a"]+"}srgbClr", {"val": "CC5500"})
        elif action in {"replace_short", "replace_long"}:
            node = next(n for n in ir["nodes"] if n["id"] == target)
            replacements = replacement_strings(node, action == "replace_long")
            paragraphs = selected.findall("p:txBody/a:p", NS)
            if len(paragraphs) != len(replacements):
                raise ValueError("Actual paragraph structure differs from IR")
            for paragraph, strings in zip(paragraphs, replacements):
                texts = paragraph.findall("a:r/a:t", NS)
                if len(texts) != len(strings):
                    raise ValueError("Actual run structure differs from IR")
                for text, replacement in zip(texts, strings):
                    text.text = replacement
            if _without_text(original_target) != _without_text(selected):
                raise ValueError("Text replacement changed paragraph or run style")
        else:
            raise ValueError("Unsupported edit probe action")
        # Removing the affected object from both trees leaves precisely the
        # untouched document; this checks nested groups as well as top-level XML.
        clean_before, clean_after = copy.deepcopy(before), copy.deepcopy(root)
        for tree in (clean_before, clean_after):
            matches = [e for e in tree.iter() if e.tag in {"{"+NS["p"]+"}"+t for t in ("sp", "pic", "grpSp")}
                       and _name(e) == target]
            for obj in matches:
                next(p for p in tree.iter() if obj in list(p)).remove(obj)
        if ET.tostring(clean_before) != ET.tostring(clean_after):
            raise ValueError("Edit changed protected XML")
        with ZipFile(output, "x", compression=ZIP_DEFLATED) as out:
            for entry in package.infolist():
                out.writestr(entry, ET.tostring(root, encoding="utf-8", xml_declaration=True)
                             if entry.filename == part else package.read(entry.filename))
    with ZipFile(source) as before_zip, ZipFile(output) as after_zip:
        if before_zip.namelist() != after_zip.namelist() or any(before_zip.read(n) != after_zip.read(n)
                for n in before_zip.namelist() if n != part):
            raise ValueError("Edit changed unrelated package parts")
    result = {"protected_xml": "PASS", "package_parts": "PASS", "slide_part": part,
              "slide_number": slide_number}
    if action.startswith("replace"):
        result["run_styles_preserved"] = "PASS"
    return result


def _rect(shape, box, pad=3):
    result = np.zeros(shape[:2], bool)
    x, y, w, h = box
    result[max(0, math.floor(y)-pad):min(shape[0], math.ceil(y+h)+pad),
           max(0, math.floor(x)-pad):min(shape[1], math.ceil(x+w)+pad)] = True
    return result


def baseline_contribution(before, baseline_deleted, box):
    """Freeze old visible contribution before examining any edited candidate."""
    footprint = np.abs(before.astype(np.int16)-baseline_deleted.astype(np.int16)).max(axis=2) > 1
    overflow = footprint & ~_rect(before.shape, box, pad=0)
    return footprint, {"method": "baseline_minus_independent_baseline_target_deletion",
                       "visible_ink_bbox_xyxy": mask_bbox(footprint),
                       "visible_pixels": int(footprint.sum()),
                       "baseline_overflow": bool(overflow.any()),
                       "baseline_overflow_pixels": int(overflow.sum())}


def shifted_mask(mask, dx, dy):
    """Translate a frozen mask without wrapping around canvas edges."""
    dx, dy = int(round(dx)), int(round(dy))
    result = np.zeros_like(mask)
    height, width = mask.shape
    x0, x1 = max(0, -dx), min(width, width-dx)
    y0, y1 = max(0, -dy), min(height, height-dy)
    if x0 < x1 and y0 < y1:
        result[y0+dy:y1+dy, x0+dx:x1+dx] = mask[y0:y1, x0:x1]
    return result


def replacement_overflow(actual, baseline_deleted, box):
    """New ink outside the frame remains overflow, even over old overflow ink."""
    new_ink = np.abs(actual.astype(np.int16)-baseline_deleted.astype(np.int16)).max(axis=2) > 1
    outside = new_ink & ~_rect(actual.shape, box)
    return {"overflow_detected": bool(outside.any()), "new_overflow_pixels": int(outside.sum()),
            "visible_ink_bbox_xyxy": mask_bbox(new_ink),
            "overflow_basis": "edited_minus_frozen_baseline_deletion_outside_original_frame"}


def mask_bbox(mask):
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)] if len(x) else None


def region_checks(before, actual, expected, old_box, new_box=None, baseline_footprint=None):
    """Authorize old overflow only from the baseline, never candidate differences."""
    old = _rect(before.shape, old_box)
    new = _rect(before.shape, new_box) if new_box else np.zeros_like(old)
    if baseline_footprint is not None:
        if baseline_footprint.shape != old.shape or baseline_footprint.dtype != bool:
            raise ValueError("Baseline contribution mask has invalid shape or type")
        old |= baseline_footprint
        if new_box:
            new |= shifted_mask(baseline_footprint, new_box[0]-old_box[0], new_box[1]-old_box[1])
    allowed = old | new
    interior = _rect(before.shape, old_box, pad=0)
    edge = old & ~_rect(before.shape, old_box, pad=-3)
    error = np.abs(actual.astype(np.int16)-expected.astype(np.int16)) if expected is not None else None
    changed = np.abs(actual.astype(np.int16)-before.astype(np.int16)).max(axis=2) > 1
    records = {}
    for name, mask in (("old", old), ("new", new), ("interior", interior), ("edge", edge), ("outside", ~allowed)):
        records[name] = {"pixels": int(mask.sum()), "mae_vs_expected": float(error[mask].mean()) if error is not None and mask.any() else None,
                         "p95_vs_expected": float(np.percentile(error[mask], 95)) if error is not None and mask.any() else None,
                         "changed_pixels": int((changed & mask).sum())}
    consistent = all(v["mae_vs_expected"] is None or v["mae_vs_expected"] <= .1 for v in records.values())
    outside = records["outside"]["changed_pixels"]
    return {"regions": records, "export_consistency": "REVIEW_REQUIRED" if expected is None else "PASS" if consistent else "FAIL",
            "outside_changed_pixels": outside, "protected_pixels": "PASS" if not outside else "FAIL",
            "actual_changed_pixels": int(changed.sum()), "scope": "export_consistency_not_clean_plate_correctness"}


def clean_plate_evidence(delivery, ir, target):
    receipt = delivery.get("verified_clean_plate")
    if not receipt:
        return {"status": "REVIEW_REQUIRED", "reason": "independent_source_bound_clean_plate_receipt_absent"}
    try:
        checked = read(receipt["report_path"])
        valid = (sha256(receipt["report_path"]) == receipt["report_sha256"]
                 and checked.get("status") == "PASS"
                 and checked.get("source_sha256") == ir.get("source_sha256")
                 and target in checked.get("verified_target_ids", [])
                 and checked.get("method") != "export_consistency")
    except (OSError, KeyError, ValueError):
        valid = False
    return {"status": "PASS" if valid else "REVIEW_REQUIRED",
            "reason": "independent_clean_plate_receipt" if valid else "clean_plate_receipt_unbound_or_incomplete",
            "receipt": receipt.get("report_path")}


def _expected(delivery, ir, output, backend):
    output.mkdir()
    base = Path(delivery["ir"]).parent
    for name in {ir["source_image"], *[a["path"] for a in ir["assets"]]}:
        src, dest = base/name, output/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    target = output / "slideir.json"
    write_json(target, ir)
    write_json(output / "authoring.json", {"tool": "r7 independent diagnostic IR edit", "model_id": "none",
               "ir_sha256": sha256(target), "decisions": ["Export consistency only; no clean-plate inference"]})
    return compile_ir(backend, target)


def probe_specs(ir, page):
    text = [n for n in ir["nodes"] if n["type"] == "text"]
    specs = []
    if page == "02":
        target = "s02-sidebar-label"
        specs.append((target, "move"))
        ids = target_members(ir, target)
        members = [n for n in ir["nodes"] if n["id"] in ids]
        if len(members) == len(ids) and all(n["type"] == "shape" for n in members):
            specs.append((target, "recolor"))
    if page == "08":
        specs.extend((target, "move") for target in
                     ("s08-feedback-arrow-1", "s08-feedback-arrow-2", "s08-feedback-arrow-3", "s08-summary-chevrons"))
    if page in {"01", "03", "04", "10", "11"} and not text:
        specs.extend(("MISSING_NATIVE_TEXT", action) for action in ("delete", "move", "replace_short", "replace_long"))
    if page in {"01", "03", "04", "10", "11"} and text:
        target = max(text, key=lambda n: n["font"]["size_pt"] if page == "01" else len(n.get("paragraphs", [])))["id"]
        specs.extend((target, action) for action in ("delete", "move", "replace_short", "replace_long"))
        emphasized = next((n for n in text if any(p.get("runs") for p in n.get("paragraphs", []))), None)
        if page == "03" and emphasized and emphasized["id"] != target:
            specs.append((emphasized["id"], "replace_short"))
    if page == "03":
        groups = {g["id"] for g in ir.get("groups", [])}
        nodes = {n["id"] for n in ir["nodes"]}
        for name in ("icon-crane", "icon-building", "icon-laptop", "icon-organization", "icon-analysis", "icon-handshake"):
            target = name+"-group" if name+"-group" in groups else name
            specs.append((target, "move"))  # Missing required targets are reported, never silently skipped.
    if page in {"03", "04", "10"}:
        cards = [n for n in ir["nodes"] if n["type"] == "shape" and n["id"].startswith("surface-")
                 and n["bbox"][2] > 80 and n["bbox"][3] > 40 and not n["id"].endswith("border")]
        if cards:
            card = min(cards, key=lambda n: n["bbox"][2]*n["bbox"][3])
            group = next((g["id"] for g in ir.get("groups", []) if card["id"] in g["member_ids"]), card["id"])
            specs.extend(((group, "move"), (card["id"], "recolor")))
        else:
            specs.append(("MISSING_NATIVE_CARD", "recolor"))
    return specs


def neighbor_protection(ir, target, shape):
    """Declared foreground regions select evidence; rectangles do not prove ink.

    A backdrop underneath the target is not a neighboring object. The rectangles
    are diagnostic candidates only, never permission or confirmed collision.
    """
    ids = set(target_members(ir, target))
    target_nodes = [n for n in ir["nodes"] if n["id"] in ids]
    old = target_bbox(ir, target)
    mask = np.zeros(shape[:2], bool)
    records = []
    for node in ir["nodes"]:
        if node["id"] in ids:
            continue
        box = node["bbox"]
        contains = (box[0] <= old[0] and box[1] <= old[1]
                    and box[0]+box[2] >= old[0]+old[2]
                    and box[1]+box[3] >= old[1]+old[3])
        behind = all(node.get("z", 0) < n.get("z", 0) for n in target_nodes)
        if contains and node["type"] in {"shape", "image"} and behind:
            continue
        region = _rect(shape, box, pad=0)
        mask |= region
        records.append({"id": node["id"], "bbox": box})
    return mask, records


def _delete_neighbor_copy(source, output, ir, target, neighbors, slide_number):
    """Delete explicit foreground nodes in a diagnostic copy, retaining target."""
    target_ids = set(target_members(ir, target))
    removed_ids = {row["id"] for row in neighbors}
    if target_ids & removed_ids:
        raise ValueError("Neighbor deletion selected a protected target member")
    visible_tags = {"{"+NS["p"]+"}"+name for name in ("sp", "pic", "grpSp", "graphicFrame", "cxnSp")}
    with ZipFile(source) as package:
        part = slide_part(package, slide_number)
        root = ET.fromstring(package.read(part))
        target_xml = {identifier: ET.tostring(_find(root, identifier)) for identifier in target_ids}
        selected = [_find(root, identifier) for identifier in sorted(removed_ids)]
        parents = {child: parent for parent in root.iter() for child in parent}
        affected_groups = set()
        removed_descendants = {child for selected_node in selected for child in selected_node.iter()}
        for node in selected:
            parent = parents.get(node)
            while parent is not None:
                if parent.tag == "{"+NS["p"]+"}grpSp":
                    affected_groups.add(parent)
                parent = parents.get(parent)
        preserved = {}
        group_properties = {}
        for node in root.iter():
            if node.tag not in visible_tags or node in removed_descendants:
                continue
            name = _name(node)
            if node in affected_groups:
                group_properties[node] = b"".join(ET.tostring(child) for child in node if child.tag not in visible_tags)
            elif name:
                if name in preserved:
                    raise ValueError("Ambiguous protected object name in neighbor diagnostic")
                preserved[name] = ET.tostring(node)
        for node in selected:
            parents[node].remove(node)
        emptied = []
        for node in reversed(list(root.iter())):
            if node in affected_groups and not any(child.tag in visible_tags for child in node):
                emptied.append(_name(node))
                parents[node].remove(node)
        for node, before_props in group_properties.items():
            if node in set(root.iter()) and b"".join(ET.tostring(child) for child in node if child.tag not in visible_tags) != before_props:
                raise ValueError("Neighbor diagnostic changed retained group properties")
        for identifier, expected in {**preserved, **target_xml}.items():
            if ET.tostring(_find(root, identifier)) != expected:
                raise ValueError("Neighbor diagnostic changed protected XML: " + identifier)
        with ZipFile(output, "x", compression=ZIP_DEFLATED) as out:
            for entry in package.infolist():
                out.writestr(entry, ET.tostring(root, encoding="utf-8", xml_declaration=True)
                             if entry.filename == part else package.read(entry.filename))
    with ZipFile(source) as before_zip, ZipFile(output) as after_zip:
        if before_zip.namelist() != after_zip.namelist() or any(
                before_zip.read(name) != after_zip.read(name) for name in before_zip.namelist() if name != part):
            raise ValueError("Neighbor diagnostic changed unrelated package parts")
    return {"status": "PASS", "protected_xml": "PASS", "package_parts": "PASS",
            "slide_part": part, "slide_number": slide_number, "retained_target_ids": sorted(target_ids),
            "removed_neighbor_ids": sorted(removed_ids), "removed_empty_groups": emptied}


def frozen_neighbor_evidence(delivery, ir, target, original, before, output, backend,
                             slide_number, baseline_deck_render):
    """Measure frozen visible neighbors from a real-PPTX deletion diagnostic.

    Missing diagnostics are an evidence gap, never confirmation of a collision.
    The diagnostic's foreground mask is independent of the edited candidate.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    original_hash, ir_hash = sha256(original), sha256(delivery["ir"])
    _, neighbors = neighbor_protection(ir, target, before.shape)
    record = {"schema": "NeighborContributionEvidence/1", "target": target,
              "original_pptx": str(original), "original_pptx_sha256": original_hash,
              "original_ir": str(delivery["ir"]), "original_ir_sha256": ir_hash,
              "slide_number": slide_number, "source_sha256": ir.get("source_sha256"),
              "method": "actual_pptx_minus_actual_foreground_neighbor_deletion",
              "scope": "declared_foreground_nodes_excluding_containing_backing_surfaces",
              "neighbor_ids": [row["id"] for row in neighbors]}
    contribution = None
    try:
        if neighbors:
            diagnostic = output / "neighbors-deleted.pptx"
            record["package_check"] = _delete_neighbor_copy(original, diagnostic, ir, target, neighbors, slide_number)
            rendered = backend.render(diagnostic, output / "render")
            record["unmodified_pages"] = unchanged_page_checks(baseline_deck_render, rendered, slide_number)
            if record["unmodified_pages"]["status"] != "PASS":
                raise ValueError("Neighbor diagnostic changed another rendered page")
            preview = rendered["pages"][slide_number-1]["path"]
            without = pixels(preview, ir["canvas"]["source_px"])
            contribution = np.abs(before.astype(np.int16)-without.astype(np.int16)).max(axis=2) > 1
            record.update(diagnostic_pptx=str(diagnostic), diagnostic_pptx_sha256=sha256(diagnostic),
                          preview=preview, preview_sha256=sha256(preview))
        else:
            contribution = np.zeros(before.shape[:2], bool)
            record["reason"] = "no_declared_foreground_neighbors"
        maskfile = output / "frozen-neighbor-contribution.png"
        Image.fromarray(contribution.astype(np.uint8)*255).save(maskfile)
        record.update(status="PASS", mask=str(maskfile), mask_sha256=sha256(maskfile),
                      visible_pixels=int(contribution.sum()), visible_bbox_xyxy=mask_bbox(contribution))
    except Exception as error:
        contribution = None
        record.update(status="REVIEW_REQUIRED", reason="independent_neighbor_contribution_unavailable",
                      error_type=type(error).__name__, error=str(error))
    if sha256(original) != original_hash or sha256(delivery["ir"]) != ir_hash:
        raise ValueError("Neighbor diagnostic modified frozen original input")
    reportfile = output / "report.json"
    write_json(reportfile, record)
    return contribution, {**record, "report_path": str(reportfile), "report_sha256": sha256(reportfile)}


def unchanged_page_checks(baseline, actual, selected_page):
    if len(baseline["pages"]) != len(actual["pages"]):
        return {"status": "FAIL", "reason": "rendered_slide_count_changed", "pages": []}
    records = []
    for index, (a, b) in enumerate(zip(baseline["pages"], actual["pages"]), 1):
        if index == selected_page:
            continue
        with Image.open(a["path"]) as im:
            size = im.size
        difference = np.abs(pixels(a["path"], size).astype(np.int16)-pixels(b["path"], size).astype(np.int16))
        changed = int((difference.max(axis=2) > 1).sum())
        records.append({"slide_number": index, "status": "PASS" if changed == 0 else "FAIL",
                        "changed_pixels": changed, "mae": float(difference.mean())})
    return {"status": "FAIL" if any(r["status"] == "FAIL" for r in records) else "PASS",
            "reason": "unmodified_pages_rendered" if records else "single_slide_package",
            "pages": records}


def audit_page(delivery, output, backend, expected_probes=None, deck_pptx=None,
               slide_number=1, baseline_deck_render=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ir = read(delivery["ir"])
    size = ir["canvas"]["source_px"]
    original = Path(deck_pptx or delivery["pptx"])
    original_hash = sha256(original)
    if deck_pptx:
        baseline_deck_render = baseline_deck_render or backend.render(original, output / "baseline-deck-render")
        if not 1 <= slide_number <= len(baseline_deck_render["pages"]):
            raise ValueError("Target slide absent from baseline deck render")
        before = pixels(baseline_deck_render["pages"][slide_number-1]["path"], size)
    else:
        if slide_number != 1:
            raise ValueError("A single-page probe requires slide_number=1")
        before = pixels(delivery["preview"], size)
        baseline_deck_render = {"pages": [{"path": delivery["preview"]}]}
    specs = list(expected_probes if expected_probes is not None else probe_specs(ir, delivery["page"][:2]))
    reports = []
    deletion_cache = {}
    neighbor_cache = {}
    (output / "baseline-deletions").mkdir()
    for index, (target, action) in enumerate(specs, 1):
        directory = output / f"{index:02d}-{target}-{action}"
        directory.mkdir()
        result = {"target": target, "action": action, "slide_number": slide_number,
                  "clean_plate_correctness": clean_plate_evidence(delivery, ir, target)}
        members = [n for n in ir["nodes"] if n["id"] in target_members(ir, target)]
        result["fill_editability"] = ("RASTER_IMAGE_NO_NATIVE_FILL" if any(n["type"] == "image" for n in members)
                                      else "NATIVE_FILL" if members and all(n["type"] == "shape" for n in members)
                                      else "NOT_APPLICABLE")
        result["actual_pptx_group"] = any(g["id"] == target for g in ir.get("groups", []))
        result["component_ids"] = [n["id"] for n in members]
        try:
            old = target_bbox(ir, target)
            result["original_bbox"] = old
            if target not in deletion_cache:
                deletion = _expected(delivery, edited_ir(ir, target, "delete"),
                                     output / "baseline-deletions" / target, backend)
                deleted_pixels = pixels(deletion["preview"], size)
                footprint, footprint_record = baseline_contribution(before, deleted_pixels, old)
                mask_path = output / "baseline-deletions" / target / "frozen-visible-contribution.png"
                Image.fromarray(footprint.astype(np.uint8)*255).save(mask_path)
                footprint_record.update(mask=str(mask_path), mask_sha256=sha256(mask_path),
                                        baseline_pptx_sha256=original_hash,
                                        baseline_ir_sha256=sha256(delivery["ir"]),
                                        deletion_preview=deletion["preview"])
                write_json(mask_path.parent / "contribution.json", footprint_record)
                deletion_cache[target] = (deletion, deleted_pixels, footprint, footprint_record)
            deletion, deleted_pixels, footprint, footprint_record = deletion_cache[target]
            result["baseline_contribution"] = footprint_record
            result["baseline_overflow"] = footprint_record["baseline_overflow"]
            delta = (20, 12)
            if action == "move":
                delta = (20 if old[0]+old[2]+23 < size[0] else -20,
                         12 if old[1]+old[3]+15 < size[1] else -12)
            edited = directory / "edited.pptx"
            result.update(rewrite_probe(original, edited, ir, target, action, delta, slide_number))
            rendering = backend.render(edited, directory / "actual-render")
            result["unmodified_pages"] = unchanged_page_checks(baseline_deck_render, rendering, slide_number)
            if not 1 <= slide_number <= len(rendering["pages"]):
                raise ValueError("Target slide absent after editing")
            result.update(pptx=str(edited), preview=rendering["pages"][slide_number-1]["path"])
            actual_pixels = pixels(result["preview"], size)
            if action.startswith("replace"):
                # Report visible overflow even when the independent compiler
                # rejects the deliberately overlong diagnostic expectation.
                result.update(replacement_overflow(actual_pixels, deleted_pixels, old))
            new = [old[0]+delta[0], old[1]+delta[1], old[2], old[3]] if action == "move" else None
            # These measurements remain available even if independent export fails.
            result.update(region_checks(before, actual_pixels, None, old, new, baseline_footprint=footprint))
            rectangles, neighbors = neighbor_protection(ir, target, before.shape)
            Image.fromarray(rectangles.astype(np.uint8)*255).save(directory / "neighbor-regions.png")
            result["neighbor_regions"] = neighbors
            if action.startswith("replace"):
                if target not in neighbor_cache:
                    neighbor_cache[target] = frozen_neighbor_evidence(delivery, ir, target, original, before,
                        output / "neighbor-evidence" / target, backend, slide_number, baseline_deck_render)
                protected, neighbor_record = neighbor_cache[target]
                result["neighbor_evidence"] = neighbor_record
                ink = np.abs(actual_pixels.astype(np.int16)-deleted_pixels.astype(np.int16)).max(axis=2) > 1
                # Existing overlap is not an edit-introduced collision.
                collision = ink & protected & ~footprint if protected is not None else None
                possible = ink & rectangles & ~footprint
                result["potential_neighbor_region_pixels"] = int(possible.sum())
                result["protected_overlap_pixels"] = int(collision.sum()) if collision is not None else None
                # Count only intersections with independently measured foreground;
                # this count does not assign a pixel's ownership among neighbors.
                result["overlap_count"] = sum(bool((collision & _rect(before.shape, n["bbox"], pad=0)).any()) for n in neighbors) if collision is not None else None
                bounds = {"schema": "EditableTextBounds/1", "object_id": target,
                          "declared_bbox_xywh": old, "visible_ink_bbox_xyxy": mask_bbox(ink),
                          "baseline_visible_ink_bbox_xyxy": footprint_record["visible_ink_bbox_xyxy"],
                          "edit_safe_bbox_xyxy": [old[0], old[1], old[0]+old[2], old[1]+old[3]],
                          "basis": "actual_declared_container_no_qa_expansion", "source_sha256": ir.get("source_sha256"),
                          "verification": "FAIL" if result["overflow_detected"] or (collision is not None and collision.any())
                          else "REVIEW_REQUIRED" if collision is None else "PASS",
                          "neighbor_evidence_report_sha256": neighbor_record["report_sha256"]}
                write_json(directory / "text-bounds.json", bounds)
                result["text_bounds"] = bounds
            heat = np.abs(actual_pixels.astype(np.int16)-before.astype(np.int16)).max(axis=2).astype(np.uint8)
            Image.fromarray(heat).save(directory / "changed-pixels.png")
            expected_ir = edited_ir(ir, target, action, delta)
            expected = deletion if action == "delete" else _expected(delivery, expected_ir, directory / "expected", backend)
            measurement = region_checks(before, actual_pixels, pixels(expected["preview"], size), old, new,
                                        baseline_footprint=footprint)
            result.update(measurement, expected_preview=expected["preview"])
            result.setdefault("overflow_detected", False)
            if action == "replace_long" and result["overflow_detected"]:
                allowed = _rect(before.shape, old) | footprint
                changed = np.abs(actual_pixels.astype(np.int16)-before.astype(np.int16)).max(axis=2) > 1
                outside_capacity = int((changed & ~allowed & ~ink).sum())
                result["capacity_overflow"] = {"verified": measurement["export_consistency"] == "PASS"
                                                and collision is not None and not collision.any() and outside_capacity == 0,
                                               "reason": "UNSUPPORTED_REFLOW", "outside_capacity_pixels": outside_capacity,
                                               "protected_overlap_pixels": int(collision.sum()) if collision is not None else None,
                                               "neighbor_evidence_status": neighbor_record["status"]}
        except Exception as error:
            result.update(error_type=type(error).__name__, reason=str(error))
        verdict = classify_edit_probe(result)
        result.update(status=verdict["status"], checks=verdict["checks"], failures=verdict["failures"],
                      review_required=verdict["review_required"], not_applicable=verdict["not_applicable"])
        write_json(directory / "report.json", result)
        reports.append(result)
    if sha256(original) != original_hash:
        raise ValueError("Edit probes modified the original delivered PPTX")
    result = {"schema": "R7EditChecks/1", "page": delivery["page"], "original_sha256": original_hash,
              "status": aggregate_checks(reports)["status"], "source_pptx": str(original),
              "slide_number": slide_number, "final_deck_probe": bool(deck_pptx), "expected_probes": specs,
              "probes": reports, "powerpoint_checked": False, "wps_checked": False,
              "human_acceptance": False, "original_preserved": True}
    if not reports:
        result["reason"] = "page_has_no_representative_edit_probe; deck-wide deletion checked separately"
    result.update({key: value for key, value in summarize_edit_report(result, specs).items() if key != "probes"})
    write_json(output / "report.json", result)
    return result
