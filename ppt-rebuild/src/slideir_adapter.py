"""Adapt validated reconstruction decisions to the maintained SlideIR compiler."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def export_ir(directory, registry, nodes, plate, assets, *, notes):
    """All paths are confined to one new revision; retained evidence belongs to plate."""
    directory = Path(directory)
    (directory / "assets").mkdir(parents=True, exist_ok=True)
    plate.save(directory / "assets/plate.png")
    active = {n["id"] for n in nodes}
    retained = [e["evidence_id"] for e in registry.objects if e["id"] not in active]
    evidence = [{"id": "ev-page-remainder", "kind": "image",
                 "bbox": [0, 0, *registry.size], "description": "Source content outside committed replacement masks"}]
    for entry in registry.objects:
        item = {"id": entry["evidence_id"], "kind": entry["kind"], "bbox": entry["bbox"]}
        if entry["kind"] == "text":
            item["text"] = entry["text"]
        else:
            item["description"] = entry.get("label", entry["id"])
        evidence.append(item)
    ir_assets = [{"id": "plate", "path": "assets/plate.png", "purpose": "cleaned_background",
                  "sha256": sha256(directory / "assets/plate.png"),
                  "contains_evidence_ids": ["ev-page-remainder", *retained]}]
    ir_nodes = [{"id": "background-image", "type": "image", "bbox": [0, 0, *registry.size],
                 "z": 0, "asset_id": "plate", "evidence_ids": []}]
    groups = []
    for index, original in enumerate(nodes):
        node = dict(original, z=index + 1)
        if node["type"] == "image":
            asset_id = node["asset_id"]
            relative = f"assets/{asset_id}.png"
            assets[asset_id].save(directory / relative)
            ir_assets.append({"id": asset_id, "path": relative, "purpose": "module",
                              "sha256": sha256(directory / relative),
                              "contains_evidence_ids": node.pop("evidence_ids")})
            node["evidence_ids"] = []
            # A transparent image and an invisible frame form a real movable group.
            # The frame has no evidence; the image owns the source icon exactly once.
            frame_id = node["id"] + "-frame"
            ir_nodes.append(dict(node, z=index * 2 + 1))
            ir_nodes.append({"id": frame_id, "type": "shape", "geometry": "rect",
                             "bbox": node["bbox"], "z": index * 2 + 2,
                             "evidence_ids": [], "fill": "none"})
            groups.append({"id": node["id"] + "-group", "member_ids": [node["id"], frame_id]})
        else:
            node["z"] = index * 2 + 1
            ir_nodes.append(node)
    ir = {"schema_version": "0.1", "source_image": "source/page.png",
          "source_sha256": registry.source_sha256,
          "canvas": {"source_px": registry.size, "target_px": registry.size},
          "background": {"color": "FFFFFF"},
          "review_status": "needs_review", "review_notes": notes,
          "evidence": evidence, "assets": ir_assets, "nodes": ir_nodes, "groups": groups}
    registry.validate_owners(ir)
    target = directory / "slideir.json"
    write_json(target, ir)
    write_json(directory / "authoring.json", {
        "tool": "Codex image2ppt / LayerD three-branch reconstruction",
        "model_id": "current-conversation", "created_at": datetime.now(timezone.utc).isoformat(),
        "ir_sha256": sha256(target),
        "decisions": ["LayerD straight-alpha source evidence; one final owner per object",
                      "Replacement pixels and native object are committed together after final-PPTX rendering",
                      *notes]})
    return target
