import copy
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest

from _helpers import SRC
import r71_cover as cover
from reconstruction_transaction import scene_fingerprint
from slideir_adapter import sha256


def package(path, bbox, style="77.25", creation_id=None, extension_uri=None):
    x, y, w, h = [int(v*10) for v in bbox]
    ns = cover.NS
    identity = (f'<a:extLst><a:ext uri="{extension_uri or cover.CREATION_EXTENSION_URI}">'
                f'<c:creationId xmlns:c="{cover.CREATION_NAMESPACE}" id="{creation_id}"/>'
                '</a:ext></a:extLst>') if creation_id is not None else ''
    with ZipFile(path, "x") as out:
        out.writestr("ppt/presentation.xml", f'<p:presentation xmlns:p="{ns["p"]}"><p:sldSz cx="10000" cy="6000"/></p:presentation>')
        out.writestr("ppt/slides/slide1.xml", f'<p:sld xmlns:p="{ns["p"]}" xmlns:a="{ns["a"]}"><p:cSld><p:spTree>'
            f'<p:sp><p:nvSpPr><p:cNvPr id="1" name="{cover.TARGET}">{identity}</p:cNvPr></p:nvSpPr><p:spPr><a:xfrm>'
            f'<a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm></p:spPr><p:txBody>'
            f'<a:bodyPr anchor="ctr" lIns="0" rIns="0" tIns="0" bIns="0" wrap="none"><a:noAutofit/></a:bodyPr>'
            f'<a:p><a:r><a:rPr sz="{style}"/><a:t>数智赋能，</a:t></a:r></a:p>'
            '<a:p><a:r><a:rPr b="0"/><a:t>智启高新</a:t></a:r></a:p></p:txBody></p:sp>'
            '</p:spTree></p:cSld></p:sld>')
    return path


def setup(tmp_path, monkeypatch, box=None, visual_change=False, style_change=False, probe_fail=False):
    base = tmp_path / "baseline"; base.mkdir()
    image = np.full((600, 1000, 3), 255, np.uint8)
    image[170:185, 150:190] = [10, 50, 100]
    Image.fromarray(image).save(base / "source.png")
    source_hash = sha256(base / "source.png")
    monkeypatch.setattr(cover, "SOURCE_SHA256", source_hash)
    node = {"id": cover.TARGET, "type": "text", "bbox": box or list(cover.OLD_BBOX), "z": 10,
            "font": {"size_pt": 77.25}, "text": "数智赋能，\n智启高新"}
    ir = {"source_image": "source.png", "source_sha256": source_hash, "canvas": {"source_px": [1000, 600]},
          "nodes": [node], "evidence": [], "assets": [], "groups": []}
    path = base / "slideir.json"; path.write_text(json.dumps(ir), encoding="utf-8")
    package(base / "result.pptx", node["bbox"])
    Image.fromarray(image).save(base / "preview.png")
    delivery = {"ir": str(path), "pptx": str(base / "result.pptx"), "preview": str(base / "preview.png")}
    def compile_candidate(backend, frozen):
        frozen = Path(frozen); data = json.loads(frozen.read_text(encoding="utf-8"))
        render = frozen.parent / "rendered"; render.mkdir()
        package(render / "result.pptx", data["nodes"][0]["bbox"], "66" if style_change else "77.25")
        actual = image.copy()
        if visual_change:
            actual[171, 151] = 0
        Image.fromarray(actual).save(render / "preview.png")
        return {"pptx": str(render / "result.pptx"), "preview": str(render / "preview.png")}
    monkeypatch.setattr(cover, "compile_ir", compile_candidate)
    monkeypatch.setattr(cover, "validate_ir_owners", lambda ir: {"status": "PASS"})
    def probe(delivery, output, backend, expected_probes):
        output.mkdir()
        assert expected_probes == [(cover.TARGET, "replace_short")]
        result = {"probes": [{"target": cover.TARGET, "action": "replace_short", "export_consistency": "PASS",
            "protected_xml": "PASS", "package_parts": "PASS", "protected_pixels": "PASS", "run_styles_preserved": "PASS",
            "unmodified_pages": {"status": "PASS"}, "actual_changed_pixels": 5, "outside_changed_pixels": 0,
            "overflow_detected": probe_fail, "protected_overlap_pixels": 0,
            "clean_plate_correctness": {"status": "REVIEW_REQUIRED", "reason": "source_clean_plate_missing"}}]}
        (output / "report.json").write_text(json.dumps(result), encoding="utf-8")
        return result
    monkeypatch.setattr(cover, "audit_page", probe)
    def frozen(before, after, allowed_fields):
        assert allowed_fields == {cover.TARGET: {"bbox": {"before": cover.OLD_BBOX, "after": cover.NEW_BBOX}}}
        restored = copy.deepcopy(after); restored["nodes"][0]["bbox"] = list(cover.OLD_BBOX)
        assert restored == before
        return {"status": "PASS"}
    return path, delivery, frozen


def test_single_container_candidate_publishes_with_bound_actual_evidence(tmp_path, monkeypatch):
    path, delivery, validator = setup(tmp_path, monkeypatch)
    original = scene_fingerprint(path)
    result = cover.attempt_cover({"page": "01_封面"}, path, delivery, tmp_path / "run", None, validator)
    assert result["action"] == "UPDATED"
    assert len(result["attempts"]) == 1 and result["attempts"][0]["state"] == "COMMITTED"
    receipt = result["attempts"][0]["verification"]
    assert {"render", "qa", "short_probe"} <= receipt["evidence_files"].keys()
    assert json.loads(Path(result["ir"]).read_text(encoding="utf-8"))["nodes"][0]["bbox"] == cover.NEW_BBOX
    assert scene_fingerprint(path) == original
    assert result["cover_bounds"]["status"] == "PASS"


@pytest.mark.parametrize("failure", ["visual_change", "style_change", "probe_fail"])
def test_candidate_failure_restores_parent_without_second_attempt(tmp_path, monkeypatch, failure):
    path, delivery, validator = setup(tmp_path, monkeypatch, **{failure: True})
    original = scene_fingerprint(path)
    result = cover.attempt_cover({"page": "01_封面"}, path, delivery, tmp_path / "run", None, validator)
    assert result["action"] == "BASELINE_RETAINED" and result["ir"] == str(path)
    assert result["pptx"] == delivery["pptx"]
    assert len(result["attempts"]) == 1 and result["attempts"][0]["state"] == "ROLLED_BACK"
    assert not (tmp_path / "run" / "cover-container" / "active.json").exists()
    assert result["cover_bounds"]["status"] == "FAIL" and scene_fingerprint(path) == original


def test_existing_container_is_noop_after_actual_geometry_and_short_check(tmp_path, monkeypatch):
    path, delivery, validator = setup(tmp_path, monkeypatch, box=list(cover.NEW_BBOX))
    monkeypatch.setattr(cover, "compile_ir", lambda *a: pytest.fail("NO_OP constructed a candidate"))
    result = cover.attempt_cover({"page": "01_封面"}, path, delivery, tmp_path / "run", None, validator)
    assert result["action"] == "NO_OP" and result["attempts"] == []
    assert not (tmp_path / "run" / "cover-container" / "transactions").exists()


def test_source_and_nonreviewed_geometry_cannot_receive_cover_delta(tmp_path, monkeypatch):
    path, delivery, validator = setup(tmp_path, monkeypatch, box=[142, 160, 453, 230])
    result = cover.attempt_cover({"page": "01_封面"}, path, delivery, tmp_path / "run", None, validator)
    assert result["action"] == "BASELINE_RETAINED" and result["cover_bounds"]["status"] == "REVIEW_REQUIRED"
    monkeypatch.setattr(cover, "SOURCE_SHA256", "unrelated-source")
    result = cover.attempt_cover({"page": "01_封面"}, path, delivery, tmp_path / "run2", None, validator)
    assert result["cover_bounds"]["reason"] == "cover_source_not_in_reviewed_scope"


def test_short_gate_does_not_treat_missing_or_failing_measurement_as_success():
    assert cover.short_safety_gate({"probes": []})["status"] == "FAIL"
    record = {"target": cover.TARGET, "action": "replace_short", "clean_plate_correctness": {"status": "REVIEW_REQUIRED"}}
    assert cover.short_safety_gate({"probes": [record]})["status"] == "FAIL"


def test_regenerated_creation_uuid_is_recorded_without_hiding_style_change(tmp_path):
    old_id='{427588D1-52C5-46EB-B1DA-9EFF335599CC}'
    new_id='{9E1D5C98-3A99-42ED-B022-19F8B2401CF8}'
    before=package(tmp_path/'before.pptx',cover.OLD_BBOX,creation_id=old_id)
    after=package(tmp_path/'after.pptx',cover.NEW_BBOX,creation_id=new_id)
    result=cover.container_gate(before,after,cover.OLD_BBOX,cover.NEW_BBOX,[1000,600])
    assert result['status']=='PASS'
    assert result['before']['regenerated_identity_metadata'][0]['value']==old_id
    assert result['after']['regenerated_identity_metadata'][0]['value']==new_id
    different_style=package(tmp_path/'different-style.pptx',cover.NEW_BBOX,style='66',creation_id=new_id)
    assert cover.container_gate(before,different_style,cover.OLD_BBOX,cover.NEW_BBOX,[1000,600])['status']=='FAIL'


def test_creation_identity_normalization_does_not_ignore_unrecognized_extensions(tmp_path):
    before=package(tmp_path/'before.pptx',cover.OLD_BBOX,creation_id='{427588D1-52C5-46EB-B1DA-9EFF335599CC}',
                   extension_uri='unrecognized-extension')
    after=package(tmp_path/'after.pptx',cover.NEW_BBOX,creation_id='{9E1D5C98-3A99-42ED-B022-19F8B2401CF8}',
                  extension_uri='unrecognized-extension')
    assert cover.container_gate(before,after,cover.OLD_BBOX,cover.NEW_BBOX,[1000,600])['status']=='FAIL'
    malformed=package(tmp_path/'invalid-id.pptx',cover.NEW_BBOX,creation_id='not-a-uuid')
    with pytest.raises(ValueError,match='valid UUID'):
        cover._container(malformed,cover.NEW_BBOX,[1000,600])
