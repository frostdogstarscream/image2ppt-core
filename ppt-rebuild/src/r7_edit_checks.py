"""Direct delivered-PPTX edits, independently compiled expectations and honest QA.

Matching an independently edited IR establishes export consistency only. It does
not establish that the hidden background is clean; that needs a source-bound,
separately verified clean-plate receipt. Original packages are always read-only.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from rebuild_pipeline import compile_ir, pixels
from slideir_adapter import sha256, write_json
from text_repair import NS


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


def rewrite_probe(source, output, ir, target, action, delta=(20, 12)):
    """Change one actual object, preserving all other package parts and XML."""
    members = target_members(ir, target)
    actual_group = any(g["id"] == target for g in ir.get("groups", []))
    if len(members) > 1 and not actual_group:
        previous = source
        for index, member in enumerate(members):
            next_file = (Path(output) if index == len(members)-1 else
                         Path(output).with_name(Path(output).stem+f"-component-{index+1}.pptx"))
            rewrite_probe(previous, next_file, ir, member, action, delta)
            previous = next_file
        return {"protected_xml": "PASS", "package_parts": "PASS",
                "semantic_component_count": len(members), "actual_pptx_group": False}
    part = "ppt/slides/slide1.xml"
    with ZipFile(source) as package:
        root = ET.fromstring(package.read(part))
        before = copy.deepcopy(root)
        selected = _find(root, target)
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
    return {"protected_xml": "PASS", "package_parts": "PASS"}


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
            "overflow_basis": "edited_minus_frozen_baseline_deletion_outside_original_frame"}


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
    error = np.abs(actual.astype(np.int16)-expected.astype(np.int16))
    changed = np.abs(actual.astype(np.int16)-before.astype(np.int16)).max(axis=2) > 1
    records = {}
    for name, mask in (("old", old), ("new", new), ("interior", interior), ("edge", edge), ("outside", ~allowed)):
        records[name] = {"pixels": int(mask.sum()), "mae_vs_expected": float(error[mask].mean()) if mask.any() else None,
                         "p95_vs_expected": float(np.percentile(error[mask], 95)) if mask.any() else None,
                         "changed_pixels": int((changed & mask).sum())}
    consistent = all(v["mae_vs_expected"] is None or v["mae_vs_expected"] <= .1 for v in records.values())
    outside = records["outside"]["changed_pixels"]
    return {"regions": records, "export_consistency": "PASS" if consistent else "FAIL",
            "outside_changed_pixels": outside, "protected_pixels": "PASS" if not outside else "REVIEW_REQUIRED",
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


def audit_page(delivery, output, backend):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ir = read(delivery["ir"])
    size = ir["canvas"]["source_px"]
    before = pixels(delivery["preview"], size)
    original_hash = sha256(delivery["pptx"])
    reports = []
    deletion_cache = {}
    (output / "baseline-deletions").mkdir()
    for index, (target, action) in enumerate(probe_specs(ir, delivery["page"][:2]), 1):
        directory = output / f"{index:02d}-{target}-{action}"
        directory.mkdir()
        result = {"target": target, "action": action, "clean_plate_correctness": clean_plate_evidence(delivery, ir, target)}
        members = [n for n in ir["nodes"] if n["id"] in target_members(ir, target)]
        result["fill_editability"] = ("RASTER_IMAGE_NO_NATIVE_FILL" if any(n["type"] == "image" for n in members)
                                      else "NATIVE_FILL" if members and all(n["type"] == "shape" for n in members)
                                      else "NOT_APPLICABLE")
        result["actual_pptx_group"] = any(g["id"] == target for g in ir.get("groups", []))
        result["component_ids"] = [n["id"] for n in members]
        try:
            old = target_bbox(ir, target)
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
            result.update(rewrite_probe(delivery["pptx"], edited, ir, target, action, delta))
            rendering = backend.render(edited, directory / "actual-render")
            result.update(pptx=str(edited), preview=rendering["pages"][0]["path"])
            actual_pixels = pixels(result["preview"], size)
            if action.startswith("replace"):
                # Report visible overflow even when the independent compiler
                # rejects the deliberately overlong diagnostic expectation.
                result.update(replacement_overflow(actual_pixels, deleted_pixels, old))
            expected_ir = edited_ir(ir, target, action, delta)
            expected = deletion if action == "delete" else _expected(delivery, expected_ir, directory / "expected", backend)
            new = [old[0]+delta[0], old[1]+delta[1], old[2], old[3]] if action == "move" else None
            measurement = region_checks(before, actual_pixels, pixels(expected["preview"], size), old, new,
                                        baseline_footprint=footprint)
            result.update(measurement, expected_preview=expected["preview"])
            result.setdefault("overflow_detected", False)
            result["run_styles_preserved"] = "PASS" if action.startswith("replace") else "NOT_APPLICABLE"
            result["status"] = ("PASS" if measurement["export_consistency"] == "PASS"
                                and measurement["protected_pixels"] == "PASS"
                                and measurement["actual_changed_pixels"] > 0
                                and not result["overflow_detected"]
                                and result["clean_plate_correctness"]["status"] == "PASS" else "REVIEW_REQUIRED")
        except Exception as error:
            result.update(status="REVIEW_REQUIRED", error_type=type(error).__name__, reason=str(error))
        write_json(directory / "report.json", result)
        reports.append(result)
    if sha256(delivery["pptx"]) != original_hash:
        raise ValueError("Edit probes modified the original delivered PPTX")
    result = {"schema": "R7EditChecks/1", "page": delivery["page"], "original_sha256": original_hash,
              "status": "PASS" if reports and all(r["status"] == "PASS" for r in reports) else "REVIEW_REQUIRED",
              "probes": reports, "powerpoint_checked": False, "wps_checked": False,
              "human_acceptance": False, "original_preserved": True}
    if not reports:
        result["reason"] = "page_has_no_representative_edit_probe; deck-wide deletion checked separately"
    write_json(output / "report.json", result)
    return result
