import copy

import pytest
from _helpers import SRC
from qa_status import normalize_check, aggregate_checks, classify_edit_probe, summarize_edit_report


@pytest.mark.parametrize("value,expected", [("PASS", "PASS"), ("FAIL", "FAIL"),
    ("REVIEW_REQUIRED", "REVIEW_REQUIRED"), ("UNKNOWN", "REVIEW_REQUIRED"),
    (None, "FAIL"), ("made-up", "FAIL"), ([], "FAIL"), ({"status": {}}, "FAIL"), ("NOT_APPLICABLE", "FAIL"),
    ({"status": "NOT_APPLICABLE", "reason": "no_native_fill"}, "NOT_APPLICABLE")])
def test_normalization_distinguishes_unknown_malformed_and_explicit_na(value, expected):
    assert normalize_check(value)["status"] == expected


def test_hard_failure_cannot_be_hidden_by_missing_evidence_at_any_level():
    object_check = aggregate_checks({"boundary": "FAIL", "clean": "UNKNOWN"})
    page_check = aggregate_checks([object_check, "PASS"])
    deck_check = aggregate_checks([page_check, "REVIEW_REQUIRED"])
    assert object_check["status"] == page_check["status"] == deck_check["status"] == "FAIL"
    assert aggregate_checks([])["status"] == "REVIEW_REQUIRED"
    assert aggregate_checks([{"status": "NOT_APPLICABLE", "reason": "no_fill"}])["status"] == "NOT_APPLICABLE"


def passing_probe(action="replace_short"):
    return {"target": "text", "action": action, "export_consistency": "PASS", "protected_pixels": "PASS",
            "actual_changed_pixels": 10, "outside_changed_pixels": 0, "protected_xml": "PASS", "package_parts": "PASS",
            "run_styles_preserved": "PASS", "overflow_detected": False, "neighbor_evidence": {"status": "PASS"},
            "clean_plate_correctness": {"status": "PASS"}}


def test_short_overflow_inside_old_contribution_still_fails():
    probe = passing_probe()
    probe.update(overflow_detected=True, outside_changed_pixels=0)
    assert classify_edit_probe(probe)["status"] == "FAIL"


def test_long_overflow_boolean_does_not_waive_any_execution_error():
    probe = passing_probe("replace_long")
    probe.update(overflow_detected=True, error_type="PermissionError")
    assert classify_edit_probe(probe)["status"] == "FAIL"
    del probe["protected_xml"]
    assert classify_edit_probe(probe)["status"] == "FAIL"


def test_only_fully_measured_isolated_capacity_overflow_is_review_required():
    probe = passing_probe("replace_long")
    probe.update(overflow_detected=True, outside_changed_pixels=20, protected_pixels="FAIL",
                 capacity_overflow={"verified": True, "reason": "UNSUPPORTED_REFLOW",
                                    "outside_capacity_pixels": 0, "protected_overlap_pixels": 0})
    assert classify_edit_probe(probe)["status"] == "REVIEW_REQUIRED"
    for change in ({"protected_xml": "FAIL"}, {"package_parts": "FAIL"}, {"error_type": "ValueError"},
                   {"protected_overlap_pixels": 1}, {"export_consistency": "FAIL"}):
        changed = {**copy.deepcopy(probe), **change}
        assert classify_edit_probe(changed)["status"] == "FAIL"


def test_report_keeps_missing_probe_and_missing_clean_plate_separate():
    probe = passing_probe()
    probe["clean_plate_correctness"] = {"status": "REVIEW_REQUIRED", "reason": "missing_independent_evidence"}
    assert summarize_edit_report({"probes": [probe]}, [("text", "replace_short")])["status"] == "REVIEW_REQUIRED"
    assert summarize_edit_report({"probes": []}, [("text", "replace_short")])["status"] == "FAIL"
    assert summarize_edit_report({"probes": [probe, probe]}, [("text", "replace_short")])["status"] == "FAIL"


def test_reported_failure_and_unknown_measurement_are_not_ignored():
    probe = passing_probe(); probe["status"] = "FAIL"
    assert classify_edit_probe(probe)["status"] == "FAIL"
    probe = passing_probe(); probe["protected_pixels"] = "UNKNOWN"
    assert classify_edit_probe(probe)["status"] == "REVIEW_REQUIRED"


def test_nested_failure_and_pending_both_reach_delivery_lists_without_duplicates():
    probe = aggregate_checks({"boundary": {"status": "FAIL", "reason": "outside_authorized_pixels"},
                              "clean_plate": {"status": "REVIEW_REQUIRED", "reason": "independent_evidence_absent"},
                              "fill": {"status": "NOT_APPLICABLE", "reason": "text_has_no_native_fill"}})
    page = aggregate_checks({"text-probe": probe})
    deck = aggregate_checks({"page-01": page})
    assert probe["status"] == page["status"] == deck["status"] == "FAIL"
    assert len(deck["failures"]) == len(deck["review_required"]) == len(deck["not_applicable"]) == 1
    assert deck["failures"][0]["path"] == ["page-01", "text-probe", "boundary"]
    assert deck["review_required"][0]["reason"] == "independent_evidence_absent"
    assert deck["review_required"][0]["name"] == "page-01/text-probe/clean_plate"


def test_wrapper_failure_is_preserved_when_children_only_show_pending():
    result = aggregate_checks({"probe": {"status": "FAIL", "reason": "execution_failed",
        "checks": [{"name": "background", "status": "REVIEW_REQUIRED", "reason": "no_evidence"}]}})
    assert result["status"] == "FAIL"
    assert result["failures"][0]["reason"] == "execution_failed"
    assert result["review_required"][0]["reason"] == "no_evidence"


def test_all_na_summary_remains_explicitly_na_when_nested():
    leaf = aggregate_checks({"fill": {"status": "NOT_APPLICABLE", "reason": "no_native_fill"}})
    parent = aggregate_checks({"object": leaf})
    assert parent["status"] == "NOT_APPLICABLE" and parent["reason"]
    assert len(parent["not_applicable"]) == 1


def test_measured_capacity_with_missing_neighbor_proof_is_pending_and_unverified():
    probe=passing_probe('replace_long')
    probe.update(overflow_detected=True, outside_changed_pixels=12, protected_pixels='FAIL',
                 protected_overlap_pixels=None, neighbor_evidence={'status':'REVIEW_REQUIRED','reason':'no_render_proof'},
                 capacity_overflow={'verified':False,'reason':'UNSUPPORTED_REFLOW','outside_capacity_pixels':0,
                                    'protected_overlap_pixels':None,'neighbor_evidence_status':'REVIEW_REQUIRED'})
    result=classify_edit_probe(probe)
    assert result['status']=='REVIEW_REQUIRED' and not result['failures']
    assert probe['capacity_overflow']['verified'] is False
    for change in ({'error_type':'PermissionError'}, {'package_parts':'FAIL'}, {'export_consistency':'UNKNOWN'},
                   {'action':'replace_short'}):
        assert classify_edit_probe({**probe,**change})['status']=='FAIL'
