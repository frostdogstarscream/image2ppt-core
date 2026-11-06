"""R7 read-only evidence adapter and bounded pixel-write authorization.

Registry/1 remains unchanged. Registry/2 describes evidence, never authorizes a
write merely because an OCR or node bounding box exists. Call authorize_repair
before applying a patch; retained baseline objects need no inferred masks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MASK_ROLES = ("visible", "repair", "protection", "inspection")
STATE_MAP = {"candidate": "DISCOVERED", "matched": "RESOLVED",
             "trial_rendered": "STAGED", "verified": "VERIFIED",
             "committed": "COMMITTED", "reverted": "ROLLED_BACK"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def xywh_to_xyxy(box, canvas):
    if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
        raise ValueError("Invalid source-pixel bbox")
    x, y, w, h = box
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > canvas[0] + 1e-6 or y + h > canvas[1] + 1e-6:
        raise ValueError("Source bbox outside canvas")
    return [x, y, x + w, y + h]


def unknown_masks():
    return {role: {"status": "UNKNOWN", "reason": "No verified pixel evidence"} for role in MASK_ROLES}


def _object(identifier, kind, box, evidence, owner, **extra):
    return {"id": identifier, "kind": kind, "bbox_xyxy": box,
            "status": "BASELINE_RETAINED", "action": "retain",
            "evidence_ids": list(evidence), "contribution_ids": list(evidence),
            "output_owner": owner, "semantic_parent_id": None,
            "transaction_dependencies": [], "edit_group_ids": [],
            "masks": unknown_masks(), **extra}


def migrate_v1(record):
    """Return a new /2 record; never change legacy data or invent pixel masks."""
    if record.get("schema") != "ObjectRegistry/1":
        raise ValueError("Expected ObjectRegistry/1")
    if record.get("coordinate_space") != "source_pixels_xywh":
        raise ValueError("Unknown legacy coordinate space")
    size = record["canvas_px"]
    objects = []
    for old in record["objects"]:
        if old["state"] not in STATE_MAP:
            raise ValueError("Unknown legacy state")
        entry = _object(old["id"], old["kind"], xywh_to_xyxy(old["bbox"], size),
                        [old["evidence_id"]], old.get("final_owner", "background-image"),
                        status=STATE_MAP[old["state"]], legacy_state=old["state"],
                        z_order=old.get("z", 0), source_layers=copy.deepcopy(old.get("source_layers", [])))
        objects.append(entry)
    return {"schema": "ObjectRegistry/2", "source_sha256": record["source_sha256"],
            "canvas_px": list(size), "coordinate_space": "source_pixels_xyxy",
            "objects": objects, "migration": {"from": "ObjectRegistry/1", "read_only": True},
            "conflicts": [], "decisions": [], "review_required": True}


def registry_from_ir(ir, *, source_sha256=None):
    """Adapt existing SlideIR ownership without treating its boxes as masks."""
    size = ir["canvas"]["source_px"]
    assets = {a["id"]: a for a in ir["assets"]}
    groups = {}
    for group in ir.get("groups", []):
        for member in group["member_ids"]:
            groups.setdefault(member, []).append(group["id"])
    objects = []
    for node in ir["nodes"]:
        evidence = list(node.get("evidence_ids", []))
        if node["type"] == "image":
            evidence += assets[node["asset_id"]].get("contains_evidence_ids", [])
        objects.append(_object(node["id"], node["type"], xywh_to_xyxy(node["bbox"], size),
                               evidence, node["id"], z_order=node["z"],
                               edit_group_ids=sorted(groups.get(node["id"], [])),
                               source_layers=[]))
    if ir["background"].get("asset_id"):
        evidence = assets[ir["background"]["asset_id"]].get("contains_evidence_ids", [])
        objects.append(_object("ir-background", "background", [0, 0, *size], evidence, "ir-background", z_order=-1))
    return {"schema": "ObjectRegistry/2", "source_sha256": source_sha256 or ir.get("source_sha256"),
            "canvas_px": list(size), "coordinate_space": "source_pixels_xyxy",
            "expected_evidence_ids": [e["id"] for e in ir["evidence"]],
            "objects": objects, "conflicts": [], "decisions": [], "review_required": True}


def mask_ref(path, canvas_px, origin_px=(0, 0), semantics="binary", *, root=None, method="reviewed_source_mask"):
    """Describe a saved single-channel mask. 0/255 binary differs from alpha."""
    path = Path(path).resolve()
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError("Mask must be uint8 single-channel L")
        size = list(image.size)
    reference = {"status": "VERIFIED", "path": str(path.relative_to(Path(root).resolve()) if root else path),
                 "sha256": digest(path), "size_px": size, "origin_px": list(origin_px),
                 "canvas_px": list(canvas_px), "semantics": semantics, "method": method}
    load_mask(reference, root or path.parent, canvas_px)
    return reference


def load_mask(reference, base_dir, canvas_px):
    if reference.get("status") != "VERIFIED":
        raise ValueError("UNKNOWN mask cannot authorize repair")
    if reference.get("canvas_px") != list(canvas_px):
        raise ValueError("Mask canvas mismatch")
    path = Path(reference["path"])
    if not path.is_absolute():
        root = Path(base_dir).resolve()
        path = (root / path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Mask path escapes evidence directory")
    if digest(path) != reference["sha256"]:
        raise ValueError("Mask hash mismatch")
    with Image.open(path) as image:
        if image.mode != "L" or list(image.size) != reference["size_px"]:
            raise ValueError("Mask mode or dimensions mismatch")
        local = np.asarray(image)
    origin = reference["origin_px"]
    if len(origin) != 2 or any(type(v) is not int for v in origin):
        raise ValueError("Mask origin must contain integer source pixels")
    x, y = origin; w, h = reference["size_px"]
    xywh_to_xyxy([x, y, w, h], canvas_px)
    semantics = reference.get("semantics")
    if semantics not in ("binary", "alpha_coverage"):
        raise ValueError("Unknown mask semantics")
    if semantics == "binary" and not np.isin(local, [0, 255]).all():
        raise ValueError("Binary masks must contain only 0 and 255")
    result = np.zeros((canvas_px[1], canvas_px[0]), np.uint8)
    result[y:y+h, x:x+w] = local
    return result


def resolve(registry, base_dir):
    """Deterministic ownership and repair checks. Retention never implies PASS."""
    if registry.get("schema") != "ObjectRegistry/2" or registry.get("coordinate_space") != "source_pixels_xyxy":
        raise ValueError("Expected ObjectRegistry/2 source_pixels_xyxy")
    result = copy.deepcopy(registry); size = result["canvas_px"]
    objects = result["objects"]; ids = [o["id"] for o in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate registry object id")
    conflicts = []; decisions = []; claims = {}; arrays = {}
    def conflict(kind, involved, **details):
        conflicts.append({"type": kind, "object_ids": sorted(set(involved)), **details})
    for obj in objects:
        left, top, right, bottom = obj["bbox_xyxy"]
        xywh_to_xyxy([left, top, right-left, bottom-top], size)
        if obj.get("action") not in ("retain", "replace", "superseded"):
            raise ValueError("Unknown registry action")
        if obj.get("action") == "superseded":
            continue
        for contribution in obj.get("contribution_ids", []):
            claims.setdefault(contribution, []).append((obj["id"], obj["output_owner"]))
        parent = obj.get("semantic_parent_id")
        dependencies = obj.get("transaction_dependencies", [])
        unknown = [d for d in dependencies + ([parent] if parent else []) if d not in ids or d == obj["id"]]
        if unknown:
            conflict("INVALID_RELATION_REFERENCE", [obj["id"]], targets=unknown)
        loaded = {}
        for role in MASK_ROLES:
            ref = obj.get("masks", {}).get(role, {"status": "UNKNOWN"})
            if ref.get("status") != "VERIFIED":
                continue
            try:
                loaded[role] = load_mask(ref, base_dir, size)
                if role != "visible" and ref["semantics"] != "binary":
                    raise ValueError("Only visible masks may use alpha_coverage")
            except (ValueError, OSError, KeyError) as error:
                conflict("INVALID_MASK", [obj["id"]], role=role, reason=str(error))
        arrays[obj["id"]] = loaded
        if obj["action"] == "replace":
            missing = sorted(set(MASK_ROLES) - set(loaded))
            if missing:
                conflict("UNKNOWN_REPAIR_AUTHORIZATION", [obj["id"]], roles=missing)
            elif not loaded["visible"].any() or not loaded["repair"].any() or not loaded["inspection"].any():
                conflict("EMPTY_PIXEL_EVIDENCE", [obj["id"]])
            else:
                if np.any((loaded["visible"] > 0) & (loaded["inspection"] == 0)):
                    conflict("INSPECTION_OMITS_VISIBLE_EVIDENCE", [obj["id"]])
                if np.any((loaded["visible"] > 0) & (loaded["repair"] == 0)):
                    conflict("REPAIR_OMITS_VISIBLE_EVIDENCE", [obj["id"]])
    for contribution, writers in sorted(claims.items()):
        if len(writers) > 1:
            conflict("DUPLICATE_CONTRIBUTION_OWNER", [w[0] for w in writers], contribution_id=contribution)
    expected = result.get("expected_evidence_ids")
    if expected is not None and set(claims) != set(expected):
        conflict("EVIDENCE_COVERAGE_MISMATCH", [], missing=sorted(set(expected)-set(claims)), extra=sorted(set(claims)-set(expected)))
    replacing = [o for o in objects if o["action"] == "replace" and "repair" in arrays.get(o["id"], {})]
    for obj in replacing:
        repair = arrays[obj["id"]]["repair"] > 0
        for other in objects:
            if other["action"] == "superseded":
                continue
            protection = arrays.get(other["id"], {}).get("protection")
            if protection is not None and np.any(repair & (protection > 0)):
                conflict("REPAIR_INTERSECTS_PROTECTION", [obj["id"], other["id"]])
        for other in replacing:
            if other["id"] <= obj["id"]:
                continue
            if np.any(repair & (arrays[other["id"]]["repair"] > 0)) and not (
                    obj.get("transaction_id") and obj.get("transaction_id") == other.get("transaction_id")):
                conflict("SHARED_REPAIR_REQUIRES_TRANSACTION", [obj["id"], other["id"]])
    # Dependency cycles must not leave a writer waiting on itself through a parent.
    graph = {o["id"]: o.get("transaction_dependencies", []) for o in objects}
    def cyclic(node, visiting, done):
        if node in visiting: return True
        if node in done or node not in graph: return False
        visiting.add(node)
        if any(cyclic(child, visiting, done) for child in graph[node]): return True
        visiting.remove(node); done.add(node)
        return False
    if any(cyclic(node, set(), set()) for node in graph):
        conflict("CYCLIC_TRANSACTION_DEPENDENCY", list(graph))
    for obj in objects:
        relevant = [c for c in conflicts if not c["object_ids"] or obj["id"] in c["object_ids"]]
        if obj["action"] == "superseded":
            status = "SUPERSEDED"
        elif relevant:
            status = "NEEDS_REVIEW"
        elif obj["action"] == "replace":
            status = "RESOLVED"
        else:
            status = "BASELINE_RETAINED"
        obj["status"] = status
        decisions.append({"object_id": obj["id"], "status": status,
                          "repair_authorized": status == "RESOLVED",
                          "reason": "Retained baseline is not pixel validation" if status == "BASELINE_RETAINED" else None})
    result.update(conflicts=conflicts, decisions=decisions,
                  review_required=bool(conflicts) or any(o["status"] == "BASELINE_RETAINED" for o in objects))
    return result


def authorize_repair(registry, object_ids, base_dir):
    """Return the union of explicitly resolved repair pixels, or fail closed."""
    resolved = resolve(registry, base_dir)
    objects = {o["id"]: o for o in resolved["objects"]}
    if not object_ids or len(set(object_ids)) != len(object_ids):
        raise ValueError("Repair requires unique nonempty object ids")
    mask = np.zeros(tuple(reversed(resolved["canvas_px"])), bool)
    for identifier in object_ids:
        obj = objects.get(identifier)
        if obj is None or obj["status"] != "RESOLVED":
            reasons = [c for c in resolved["conflicts"] if not c["object_ids"] or identifier in c["object_ids"]]
            raise ValueError(f"Repair not authorized for {identifier}: {reasons}")
        mask |= load_mask(obj["masks"]["repair"], base_dir, resolved["canvas_px"]) > 0
    return mask


def validate_ir_owners(ir):
    """Check final evidence ownership only, never claim unknown masks passed QA."""
    try:
        for name in ("nodes", "assets", "evidence"):
            ids = [item["id"] for item in ir[name]]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate " + name + " id")
        resolved = resolve(registry_from_ir(ir), Path('.'))
        conflicts = resolved['conflicts']
        return {"status": "FAIL" if conflicts else "PASS", "conflicts": conflicts,
                "scope": "evidence_ownership_only", "pixel_masks_verified": False}
    except (KeyError, TypeError, ValueError) as error:
        return {"status": "FAIL", "conflicts": [{"type": "INVALID_IR_OWNERSHIP", "reason": str(error)}],
                "scope": "evidence_ownership_only", "pixel_masks_verified": False}


def from_ir(irfile, output):
    """Write a shadow-only registry and overlap visualization; source stays read-only."""
    irfile = Path(irfile); output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ir = json.loads(irfile.read_text(encoding="utf-8"))
    source = (irfile.parent / ir["source_image"]).resolve()
    source_hash = digest(source)
    if ir.get("source_sha256") and ir["source_sha256"] != source_hash:
        raise ValueError("IR source hash mismatch")
    registry = registry_from_ir(ir, source_sha256=source_hash)
    registry["baseline_ir_sha256"] = digest(irfile)
    registry = resolve(registry, irfile.parent)
    with Image.open(source) as original:
        if list(original.size) != registry["canvas_px"]:
            raise ValueError("Source canvas mismatch")
        overlay = original.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    problematic = {i for c in registry["conflicts"] for i in c["object_ids"]}
    for obj in registry["objects"]:
        if obj["id"] in ("background-image", "ir-background"):
            continue
        draw.rectangle(obj["bbox_xyxy"], outline="red" if obj["id"] in problematic else "#E69F00", width=1)
    overlay.save(output / "overlap.png")
    registry["visualization_note"] = "Amber outlines are candidate extents, not verified pixel masks or erase permission"
    for name, data in [("objects.json", registry), ("conflicts.json", registry["conflicts"]),
                       ("ownership-decisions.json", registry["decisions"])]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return registry
