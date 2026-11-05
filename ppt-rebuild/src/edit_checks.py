"""Diagnostic edits of final PPTX copies; slide creation stays in the compiler."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def name(element):
    value = element.find(".//p:cNvPr", NS)
    return value.get("name") if value is not None else None


def edit_final_copy(source, target, *, delete_groups=(), replace_text=None):
    """Preserve every other ZIP member and fail on missing/duplicate targets."""
    with ZipFile(source) as original:
        part = "ppt/slides/slide1.xml"
        root = ET.fromstring(original.read(part))
        tree = root.find("p:cSld/p:spTree", NS)
        for group_id in delete_groups:
            matches = [e for e in tree.findall("p:grpSp", NS) if name(e) == group_id]
            if len(matches) != 1:
                raise ValueError(f"expected one final-PPTX group {group_id}")
            tree.remove(matches[0])
        if replace_text:
            object_id, text = replace_text
            matches = [e for e in root.findall(".//p:sp", NS) if name(e) == object_id]
            if len(matches) != 1:
                raise ValueError(f"expected one native text object {object_id}")
            text_elements = matches[0].findall(".//a:t", NS)
            if not text_elements:
                raise ValueError("native text object has no editable text")
            for index, element in enumerate(text_elements):
                element.text = text if index == 0 else ""
        with ZipFile(target, "x", compression=ZIP_DEFLATED) as output:
            for entry in original.infolist():
                payload = ET.tostring(root, encoding="utf-8", xml_declaration=True) if entry.filename == part else original.read(entry.filename)
                output.writestr(entry, payload)


def check_deleted_visuals(result, registry, replacements, active, directory, backend):
    import numpy as np
    from PIL import Image
    from image2ppt.quality import inspect_pptx
    from slideir_adapter import sha256

    selected = [e for e in registry.objects if e["id"] in active and e["kind"] == "image"]
    if not selected:
        return None
    directory = Path(directory)
    target = directory / "delete-visual-groups.pptx"
    group_ids = [e["id"] + "-group" for e in selected]
    before = inspect_pptx(result["pptx"])
    edit_final_copy(result["pptx"], target, delete_groups=group_ids)
    after = inspect_pptx(target)
    old_names = {e["name"] for e in before["slides"][0]["objects"]}
    new_names = {e["name"] for e in after["slides"][0]["objects"]}
    expected_removed = {v for entry in selected for v in (entry["id"], entry["id"]+"-frame", entry["id"]+"-group")}
    if old_names - new_names != expected_removed or new_names - old_names:
        raise ValueError("visual deletion changed unrelated object inventory")
    original_objects = {e["name"]: e for e in before["slides"][0]["objects"]}
    for remaining in after["slides"][0]["objects"]:
        original_object = original_objects[remaining["name"]]
        if remaining["transform"] != original_object["transform"] or remaining["text"] != original_object["text"]:
            raise ValueError("visual deletion changed a neighbouring object")
    rendered = backend.render(target, directory / "delete-visual-groups-render")
    with Image.open(rendered["pages"][0]["path"]) as image:
        if image.size != tuple(registry.size):
            raise ValueError("visual deletion preview canvas mismatch")
        rgb = np.array(image.convert("RGB"))
    scores = []
    for entry in selected:
        x, y, w, h = entry["bbox"]
        repair = replacements[entry["id"]]["repair"]
        # Check the whole old rectangle, including the faint edge outside the mask.
        mae = float(np.abs(rgb[y:y+h, x:x+w].astype(np.int16) - repair.astype(np.int16)).mean())
        scores.append({"id": entry["id"], "old_site_mae": round(mae, 4), "status": "PASS" if mae <= 6 else "FAIL"})
    return {"id": "delete-visual-groups", "status": "PASS" if all(s["status"] == "PASS" for s in scores) else "FAIL",
            "method": "direct_final_pptx_group_deletion_and_reimport", "original_sha256": sha256(result["pptx"]),
            "edited_pptx": str(target), "render": rendered, "objects": scores}
