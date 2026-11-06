"""One source-bound, measured cover-container candidate; never relax QA masks."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import uuid
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np

from object_resolver import authorization_digest, validate_ir_owners
from qa_status import aggregate_checks
from r7_edit_checks import NS, _find, audit_page, slide_part
from rebuild_pipeline import compile_ir, pixels
from reconstruction_transaction import ReconstructionTransaction, _snapshot, scene_fingerprint
from slideir_adapter import sha256, write_json


SOURCE_SHA256 = "cba343522fd7f87baf705fe56812bd9d3b56557008d898bb84fcf099cb13d1e7"
TARGET = "paragraph-l003"
OLD_BBOX = [142, 169, 453, 212]
NEW_BBOX = [142, 157, 453, 236]
CREATION_EXTENSION_URI = "{FF2B5EF4-FFF2-40B4-BE49-F238E27FC236}"
CREATION_NAMESPACE = "http://schemas.microsoft.com/office/drawing/2014/main"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _container(package_path, bbox, source_size):
    with ZipFile(package_path) as package:
        node = _find(ET.fromstring(package.read(slide_part(package))), TARGET)
        dimensions = ET.fromstring(package.read("ppt/presentation.xml")).find("p:sldSz", NS)
    node = copy.deepcopy(node)
    transform = node.find("p:spPr/a:xfrm", NS)
    if transform is None or dimensions is None:
        raise ValueError("Cover text container geometry is missing")
    off, ext = transform.find("a:off", NS), transform.find("a:ext", NS)
    if off is None or ext is None:
        raise ValueError("Cover text container offsets are missing")
    scale_x = int(dimensions.get("cx"))/source_size[0]
    scale_y = int(dimensions.get("cy"))/source_size[1]
    observed = [int(off.get("x")), int(off.get("y")), int(ext.get("cx")), int(ext.get("cy"))]
    expected = [round(bbox[0]*scale_x), round(bbox[1]*scale_y), round(bbox[2]*scale_x), round(bbox[3]*scale_y)]
    if any(abs(a-b) > 1 for a, b in zip(observed, expected)):
        raise ValueError("Actual cover OOXML geometry differs from the declared container")
    # Every effective layout property stays byte-equivalent apart from y and h.
    off.set("y", "AUTHORIZED_Y")
    ext.set("cy", "AUTHORIZED_HEIGHT")
    regenerated = []
    # Artifact export generates a fresh creation UUID on every compilation.
    # Normalize only this metadata attribute under its precise documented OOXML
    # extension location; all effective properties and every other attribute stay.
    path = "p:nvSpPr/p:cNvPr/a:extLst/a:ext"
    for extension in node.findall(path, NS):
        if extension.get("uri") != CREATION_EXTENSION_URI:
            continue
        for creation in extension.findall("{"+CREATION_NAMESPACE+"}creationId"):
            value = creation.get("id", "")
            try:
                uuid.UUID(value.strip("{}"))
            except (ValueError, AttributeError) as error:
                raise ValueError("Cover creation identity is not a valid UUID") from error
            regenerated.append({"extension_uri": CREATION_EXTENSION_URI, "namespace": CREATION_NAMESPACE,
                                "attribute": "id", "value": value})
            creation.set("id", "REGENERATED_CREATION_UUID")
    return ET.tostring(node), {"bbox_xywh": bbox, "actual_emu": observed,
                              "expected_emu": expected, "slide_size_emu": dict(dimensions.attrib),
                              "regenerated_identity_metadata": regenerated}


def container_gate(before_pptx, after_pptx, old_bbox, new_bbox, size):
    before, before_geometry = _container(before_pptx, old_bbox, size)
    after, after_geometry = _container(after_pptx, new_bbox, size)
    return {"status": "PASS" if before == after else "FAIL",
            "reason": "only_y_and_height_authorized", "before": before_geometry, "after": after_geometry,
            "text_runs_and_layout_preserved": before == after}


def short_safety_gate(report):
    probes = report.get("probes", [])
    if len(probes) != 1 or (probes[0].get("target"), probes[0].get("action")) != (TARGET, "replace_short"):
        return {"status": "FAIL", "reason": "required_cover_short_probe_missing"}
    probe = probes[0]
    checks = {key: probe.get(key) for key in ("export_consistency", "protected_xml", "package_parts",
                                             "protected_pixels", "run_styles_preserved", "unmodified_pages")}
    checks.update(effect={"status": "PASS" if probe.get("actual_changed_pixels", 0) > 0 else "FAIL"},
                  boundary={"status": "PASS" if probe.get("outside_changed_pixels") == 0 else "FAIL"},
                  overflow={"status": "PASS" if probe.get("overflow_detected") is False else "FAIL"},
                  neighbors={"status": "PASS" if probe.get("protected_overlap_pixels") == 0 else "FAIL"},
                  execution={"status": "FAIL" if probe.get("error_type") or probe.get("failures") else "PASS"})
    result = aggregate_checks(checks)
    result["clean_plate_correctness"] = probe.get("clean_plate_correctness")
    result["scope"] = "container_geometry_safety; independent_background_evidence_reported_separately"
    return result


def attempt_cover(page, base_ir, base_delivery, attemptdir, backend, frozen_validator):
    """Try exactly one candidate and publish only after actual render/edit gates.

    Caller's page run lock covers this operation. The clean-plate question remains
    in final edit QA because no background pixels or assets change in this repair.
    """
    base_ir, attemptdir = Path(base_ir), Path(attemptdir)
    original = _read(base_ir)
    initial_fp = scene_fingerprint(base_ir)
    result = {**base_delivery, "ir": str(base_ir), "action": "BASELINE_RETAINED", "attempts": []}
    if not page.get("page", "").startswith("01") or original.get("source_sha256") != SOURCE_SHA256:
        result["cover_bounds"] = {"status": "REVIEW_REQUIRED", "reason": "cover_source_not_in_reviewed_scope"}
        return result
    targets = [n for n in original["nodes"] if n["id"] == TARGET]
    if len(targets) != 1 or targets[0].get("type") != "text":
        result["cover_bounds"] = {"status": "FAIL", "reason": "cover_target_missing_or_ambiguous"}
        return result
    box = targets[0]["bbox"]
    if box not in (OLD_BBOX, NEW_BBOX):
        result["cover_bounds"] = {"status": "REVIEW_REQUIRED", "reason": "cover_container_outside_reviewed_delta", "bbox": box}
        return result
    directory = attemptdir / "cover-container"
    directory.mkdir(parents=True, exist_ok=False)
    size = original["canvas"]["source_px"]
    # Re-running an accepted container still checks current actual geometry/edit
    # evidence, but cannot create another candidate or transaction.
    if box == NEW_BBOX:
        try:
            _, geometry = _container(base_delivery["pptx"], NEW_BBOX, size)
            report = audit_page({**page, **base_delivery, "ir": str(base_ir)}, directory / "short-probe", backend,
                                expected_probes=[(TARGET, "replace_short")])
            safety = short_safety_gate(report)
            result["cover_bounds"] = {**safety, "reason": "ALREADY_SATISFIED_CONTAINER", "geometry": geometry}
            if safety["status"] == "PASS":
                result["action"] = "NO_OP"
        except Exception as error:
            result["cover_bounds"] = {"status": "FAIL", "reason": str(error), "error_type": type(error).__name__}
        if scene_fingerprint(base_ir) != initial_fp:
            raise ValueError("Immutable cover parent changed during NO_OP verification")
        write_json(directory / "result.json", result)
        return result

    tx = None
    attempt = {"number": 1, "object_id": TARGET, "old_bbox": OLD_BBOX, "new_bbox": NEW_BBOX}
    try:
        candidate_ir, _ = _snapshot(base_ir, directory / "proposed")
        candidate = copy.deepcopy(original)
        next(n for n in candidate["nodes"] if n["id"] == TARGET)["bbox"] = list(NEW_BBOX)
        candidate_ir.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
        authoring = candidate_ir.parent / "authoring.json"
        if authoring.exists():
            record = _read(authoring)
            record["ir_sha256"] = sha256(candidate_ir)
            record.setdefault("decisions", []).append("Source-bound r7.1 cover container: same center, y/h only")
            authoring.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        grant = {"schema": "SceneChangeAuthorization/1", "status": "PASS", "tx_id": "cover-container-1",
                 "object_ids": [TARGET], "source_sha256": SOURCE_SHA256, "parent_scene_sha256": initial_fp["sha256"],
                 "allowed_fields": {TARGET: {"bbox": {"before": OLD_BBOX, "after": NEW_BBOX}}},
                 "evidence_files": {str(base_ir.resolve()): sha256(base_ir),
                                    str(Path(base_delivery["pptx"]).resolve()): sha256(base_delivery["pptx"]),
                                    str(Path(base_delivery["preview"]).resolve()): sha256(base_delivery["preview"])}}
        grant["authorization_sha256"] = authorization_digest(grant)
        write_json(directory / "authorization.json", grant)
        tx = ReconstructionTransaction(directory, grant["tx_id"], base_ir,
                {"source": SOURCE_SHA256, "parent": initial_fp["sha256"], "authorization": grant["authorization_sha256"]},
                ["frozen_scene", "actual_container", "unchanged_visual", "short_edit_safety"], [TARGET], authorization=grant)
        frozen = tx.prepare(candidate_ir)
        trial = compile_ir(backend, frozen)
        before = pixels(base_delivery["preview"], size)
        after = pixels(trial["preview"], size)
        diff = np.abs(after.astype(np.int16)-before.astype(np.int16))
        changed = int((diff.max(axis=2) > 1).sum())
        gates = {"frozen_scene": frozen_validator(original, _read(frozen), allowed_fields=grant["allowed_fields"]),
                 "actual_container": container_gate(base_delivery["pptx"], trial["pptx"], OLD_BBOX, NEW_BBOX, size),
                 "unchanged_visual": {"status": "PASS" if changed == 0 else "FAIL", "changed_pixels": changed,
                                      "mae": float(diff.mean()), "threshold": "max_rgb_difference_gt_1"}}
        probe_report = audit_page({**page, **trial, "ir": str(frozen)}, directory / "short-probe", backend,
                                  expected_probes=[(TARGET, "replace_short")])
        gates["short_edit_safety"] = short_safety_gate(probe_report)
        gatefile = directory / "gates.json"
        write_json(gatefile, gates)
        verdict = tx.verify(gates, validate_ir_owners, evidence_files={"render": trial["preview"], "qa": gatefile,
                                                                     "short_probe": directory / "short-probe" / "report.json"})
        attempt.update(gates=gates, verification=verdict)
        if verdict["state"] == "VERIFIED":
            receipt = tx.commit()
            attempt.update(state="COMMITTED", receipt=receipt)
            result.update(trial, ir=str(receipt["ir"]), action="UPDATED")
        else:
            receipt = tx.rollback("Single cover candidate failed required gates")
            attempt.update(state="ROLLED_BACK", receipt=receipt)
        result["cover_bounds"] = {"schema": "EditableTextBounds/1", "status": aggregate_checks(gates)["status"],
                                  "object_id": TARGET, "original_declared_bbox_xywh": OLD_BBOX,
                                  "candidate_declared_bbox_xywh": NEW_BBOX,
                                  "basis": "same_center_actual_container_and_short_edit_render",
                                  "source_sha256": SOURCE_SHA256, "neighbor_blue_line_y": 407,
                                  "candidate_to_blue_line_px": 14, "attempt_limit": 1}
    except Exception as error:
        attempt.update(state="REJECTED", reason=str(error), error_type=type(error).__name__)
        if tx is not None and tx.prepare_file.exists():
            attempt["receipt"] = tx.rollback("Cover candidate exception: " + str(error))
            attempt["state"] = "ROLLED_BACK"
        result["cover_bounds"] = {"status": "FAIL", "reason": str(error), "error_type": type(error).__name__}
    if scene_fingerprint(base_ir) != initial_fp:
        raise ValueError("Immutable cover parent changed")
    result["attempts"].append(attempt)
    write_json(directory / "result.json", result)
    return result
