"""One evidence-aware status contract for probes, pages and delivery summaries."""
from collections import Counter


def normalize_check(value, *, missing_reason="required_check_missing"):
    record = dict(value) if isinstance(value, dict) else {"status": value}
    status = record.get("status")
    if status == "UNKNOWN":
        record.update(status="REVIEW_REQUIRED", original_status="UNKNOWN")
        record.setdefault("reason", "evidence_unknown")
    elif not isinstance(status, str) or status not in {"PASS", "FAIL", "REVIEW_REQUIRED", "NOT_APPLICABLE"}:
        record.update(status="FAIL", original_status=status,
                      reason=missing_reason if status is None else "invalid_status")
    elif status == "NOT_APPLICABLE" and not record.get("reason"):
        record.update(status="FAIL", original_status=status, reason="not_applicable_reason_missing")
    return record


def _leaf_findings(record, path):
    """Traverse only the shared checks tree, retaining findings under FAIL groups."""
    children = record.get("checks")
    findings = []
    if isinstance(children, dict) and children:
        items = children.items()
    elif isinstance(children, (list, tuple)) and children:
        items = ((child.get("name", str(index)) if isinstance(child, dict) else str(index), child)
                 for index, child in enumerate(children))
    else:
        items = []
    for name, child in items:
        findings.extend(_leaf_findings(normalize_check(child), [*path, str(name)]))
    # A parent can explicitly fail for an additional reason not represented by
    # children. Never drop that fact merely because it also contains subchecks.
    if not findings or (record["status"] in {"FAIL", "REVIEW_REQUIRED", "NOT_APPLICABLE"}
                        and not any(child["status"] == record["status"] for child in findings)):
        detail = {key: value for key, value in record.items()
                  if key not in {"checks", "failures", "review_required", "not_applicable", "probes", "path", "name"}}
        findings.append({**detail, "name": "/".join(path), "path": list(path)})
    return findings


def aggregate_checks(checks):
    items = checks.items() if isinstance(checks, dict) else enumerate(checks)
    records = [{**normalize_check(value), "name": str(name)} for name, value in items]
    if not records:
        records = [{"name": "coverage", "status": "REVIEW_REQUIRED", "reason": "no_applicable_checks_defined"}]
    leaves = [leaf for record in records for leaf in _leaf_findings(record, [record["name"]])]
    failures = [r for r in leaves if r["status"] == "FAIL"]
    pending = [r for r in leaves if r["status"] == "REVIEW_REQUIRED"]
    na = [r for r in leaves if r["status"] == "NOT_APPLICABLE"]
    status = ("FAIL" if failures else "REVIEW_REQUIRED" if pending else
              "NOT_APPLICABLE" if all(r["status"] == "NOT_APPLICABLE" for r in records) else "PASS")
    result = {"status": status, "checks": records, "failures": failures,
              "review_required": pending, "not_applicable": na}
    if status == "NOT_APPLICABLE":
        result["reason"] = "all_declared_checks_not_applicable"
    return result


def classify_edit_probe(probe):
    """Classify measured failures independently of missing cleanliness evidence.

    Capacity is a deliberately narrow diagnostic exception: measured layout
    overflow with matching exported content, no neighbor collision and no other
    out-of-bound change. A boolean overflow flag or arbitrary exception cannot
    establish this exception.
    """
    checks = {}
    capacity = probe.get("capacity_overflow", {})
    neighbor_evidence = probe.get("neighbor_evidence", {
        "status": "REVIEW_REQUIRED", "reason": "independent_neighbor_contribution_missing"})
    neighbor_status = normalize_check(neighbor_evidence)["status"]
    capacity_diagnostic = (probe.get("action") == "replace_long"
                           and probe.get("overflow_detected") is True
                           and capacity.get("reason") == "UNSUPPORTED_REFLOW"
                           and capacity.get("outside_capacity_pixels") == 0
                           and probe.get("export_consistency") == "PASS"
                           and not probe.get("error_type"))
    expected_capacity = (capacity_diagnostic and capacity.get("verified") is True
                         and capacity.get("protected_overlap_pixels") == 0 and neighbor_status == "PASS")
    pending_capacity = (capacity_diagnostic and capacity.get("verified") is False
                        and capacity.get("protected_overlap_pixels") is None and neighbor_status == "REVIEW_REQUIRED")
    capacity_reason = "UNSUPPORTED_REFLOW" if expected_capacity else "capacity_safety_pending_neighbor_evidence"
    for name in ("export_consistency", "protected_xml", "package_parts", "protected_pixels"):
        if name == "protected_pixels" and (expected_capacity or pending_capacity):
            checks[name] = {"status": "REVIEW_REQUIRED", "reason": capacity_reason}
        else:
            checks[name] = probe.get(name)
    if "unmodified_pages" in probe:
        checks["unmodified_pages"] = probe["unmodified_pages"]
    if probe.get("action", "").startswith("replace"):
        checks["run_styles_preserved"] = probe.get("run_styles_preserved")
        checks["neighbor_evidence"] = neighbor_evidence
        if not isinstance(probe.get("overflow_detected"), bool):
            checks["overflow_measurement"] = {"status": "FAIL", "reason": "required_overflow_measurement_missing"}
    checks["clean_plate_correctness"] = probe.get("clean_plate_correctness", {
        "status": "REVIEW_REQUIRED", "reason": "independent_clean_plate_evidence_missing"})
    changed = probe.get("actual_changed_pixels")
    checks["edit_effect"] = {"status": "PASS" if isinstance(changed, (int, float)) and changed > 0 else "FAIL",
                             "reason": "edit_changed_pixels" if changed else "edit_effect_missing_or_zero"}
    outside = probe.get("outside_changed_pixels")
    checks["authorized_boundary"] = ({"status": "REVIEW_REQUIRED", "reason": capacity_reason}
        if (expected_capacity or pending_capacity) and isinstance(outside, (int, float)) and outside > 0 else
        {"status": "PASS" if outside == 0 else "FAIL", "reason": "boundary_contained" if outside == 0 else
         "boundary_measurement_missing" if outside is None else "outside_authorized_pixels"})
    if probe.get("overflow_detected"):
        checks["text_capacity"] = {"status": "REVIEW_REQUIRED" if expected_capacity or pending_capacity else "FAIL",
                                   "reason": capacity_reason if expected_capacity or pending_capacity else "text_overflow"}
    overlap = probe.get("protected_overlap_pixels")
    if isinstance(overlap, (int, float)) and overlap > 0:
        checks["neighbor_collision"] = {"status": "FAIL", "reason": "edited_ink_overlaps_protected_object"}
    if probe.get("error_type"):
        checks["execution"] = {"status": "FAIL", "reason": "edit_execution_failed", "error_type": probe["error_type"]}
    # Existing callers may provide a stronger explicit failure. Do not weaken it.
    if "status" in probe:
        checks["reported_status"] = normalize_check({"status": probe["status"], "reason": probe.get("reason")})
    return aggregate_checks(checks)


def summarize_edit_report(report, expected):
    probes = report.get("probes", [])
    required = Counter(tuple(pair) for pair in expected)
    actual = Counter((p.get("target"), p.get("action")) for p in probes)
    checks = {f"probe-{i}:{p.get('target')}:{p.get('action')}": classify_edit_probe(p)
              for i, p in enumerate(probes)}
    checks["required_probe_inventory"] = {
        "status": "PASS" if actual == required else "FAIL",
        "reason": "required_probes_present" if actual == required else "missing_or_duplicate_required_edit_probe",
        "missing": [list(v) for v in (required-actual).elements()],
        "unexpected": [list(v) for v in (actual-required).elements()]}
    if not required:
        checks["probe_coverage"] = {"status": "REVIEW_REQUIRED", "reason": "no_representative_edit_probe_defined"}
    result = aggregate_checks(checks)
    result["probes"] = probes
    return result
