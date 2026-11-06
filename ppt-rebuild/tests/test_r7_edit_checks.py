"""Diagnostic edits preserve neighboring XML and report evidentiary limits."""
import json
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from _helpers import SRC
from r7_edit_checks import (NS, edited_ir, rewrite_probe, region_checks,
                            clean_plate_evidence, target_bbox, probe_specs,
                            baseline_contribution, replacement_overflow, shifted_mask)
from slideir_adapter import sha256
from PIL import Image
import r7_edit_checks


def example():
    run1 = {"text": "项目", "bold": False, "color": "123456"}
    run2 = {"text": "27项", "bold": True, "color": "0088AA"}
    paragraph = {"text": "项目27项", "runs": [run1, run2]}
    return {"canvas": {"source_px": [100, 100]}, "source_sha256": "source-hash",
            "nodes": [{"id": "text", "type": "text", "bbox": [10, 10, 35, 10], "text": "项目27项",
                       "paragraphs": [paragraph], "font": {"size_pt": 12}, "evidence_ids": ["t"]},
                      {"id": "icon", "type": "image", "bbox": [5, 50, 10, 15], "asset_id": "a", "evidence_ids": []},
                      {"id": "frame", "type": "shape", "bbox": [5, 50, 10, 15], "evidence_ids": []},
                      {"id": "surface-card", "type": "shape", "bbox": [40, 40, 30, 30],
                       "fill_gradient": {"stops": []}, "fill": "778899", "evidence_ids": ["c"]}],
            "assets": [{"id": "a", "path": "icon.png", "contains_evidence_ids": ["i"]}],
            "evidence": [{"id": "t", "kind": "text", "text": "项目27项"}, {"id": "i", "kind": "image"}, {"id": "c", "kind": "shape"}],
            "groups": [{"id": "icon-group", "member_ids": ["icon", "frame"]}],
            "background": {"color": "FFFFFF"}}


def pptx(path):
    xml = f'''<p:sld xmlns:p="{NS['p']}" xmlns:a="{NS['a']}"><p:cSld><p:spTree>
    <p:sp><p:nvSpPr><p:cNvPr id="1" name="text"/></p:nvSpPr><p:spPr><a:xfrm><a:off x="100" y="100"/><a:ext cx="350" cy="100"/></a:xfrm><a:prstGeom prst="rect"/></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr b="0"/><a:t>项目</a:t></a:r><a:r><a:rPr b="1"/><a:t>27项</a:t></a:r></a:p></p:txBody></p:sp>
    <p:sp><p:nvSpPr><p:cNvPr id="2" name="surface-card"/></p:nvSpPr><p:spPr><a:xfrm><a:off x="400" y="400"/><a:ext cx="300" cy="300"/></a:xfrm><a:prstGeom prst="rect"/><a:solidFill><a:srgbClr val="778899"/></a:solidFill></p:spPr></p:sp>
    </p:spTree></p:cSld></p:sld>'''
    with ZipFile(path, "x") as out:
        out.writestr("ppt/slides/slide1.xml", xml)
        out.writestr("ppt/presentation.xml", f'<p:presentation xmlns:p="{NS["p"]}"><p:sldSz cx="1000" cy="1000"/></p:presentation>')
        out.writestr("ppt/media/original.png", b"protected-original-media")
    return path


@pytest.mark.parametrize("action", ["delete", "move", "replace_short", "replace_long", "recolor"])
def test_actual_package_edit_preserves_other_files_and_neighbor_xml(tmp_path, action):
    source = pptx(tmp_path / "original.pptx")
    before = sha256(source)
    target = "surface-card" if action == "recolor" else "text"
    result = rewrite_probe(source, tmp_path / "edited.pptx", example(), target, action)
    assert result["protected_xml"] == result["package_parts"] == "PASS"
    assert result["slide_number"] == 1 and result["slide_part"] == "ppt/slides/slide1.xml"
    if action.startswith("replace"):
        assert result["run_styles_preserved"] == "PASS"
    assert sha256(source) == before
    with ZipFile(tmp_path / "edited.pptx") as archive:
        root = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
        assert archive.read("ppt/media/original.png") == b"protected-original-media"
        if action.startswith("replace"):
            assert [p.get("b") for p in root.findall(".//a:rPr", NS)] == ["0", "1"]
            assert all(t.text for t in root.findall(".//a:t", NS))
        if action == "move":
            assert root.find(".//a:off", NS).attrib == {"x": "300", "y": "220"}


def test_expected_group_move_updates_all_members_without_rewriting_original():
    ir = example()
    changed = edited_ir(ir, "icon-group", "move")
    assert target_bbox(changed, "icon-group") == [25, 62, 10, 15]
    assert target_bbox(ir, "icon-group") == [5, 50, 10, 15]
    assert changed["nodes"][0] == ir["nodes"][0]


def test_diagnostic_deletion_removes_exclusive_evidence_and_unused_asset():
    original = example()
    changed = edited_ir(original, "icon-group", "delete")
    assert changed["assets"] == [] and changed["groups"] == []
    assert changed["unresolved_evidence_ids"] == []
    assert {e["id"] for e in changed["evidence"]} == {"t", "c"}
    assert [n["id"] for n in changed["nodes"]] == ["text", "surface-card"]
    assert {e["id"] for e in original["evidence"]} == {"t", "i", "c"}


def test_diagnostic_text_delete_does_not_leave_unprocessed_source_evidence():
    changed = edited_ir(example(), "text", "delete")
    assert {e["id"] for e in changed["evidence"]} == {"i", "c"}
    assert changed["unresolved_evidence_ids"] == []
    assert all("t" not in n.get("evidence_ids", []) for n in changed["nodes"])


def test_mixed_run_replacement_preserves_emphasis_without_empty_runs():
    original = example()
    changed = edited_ir(original, "text", "replace_short")["nodes"][0]
    assert changed["text"] == "测测"
    assert [r["bold"] for r in changed["paragraphs"][0]["runs"]] == [False, True]
    assert [r["color"] for r in changed["paragraphs"][0]["runs"]] == ["123456", "0088AA"]
    assert original["nodes"][0]["text"] == "项目27项"


def test_expected_recolor_removes_gradient_and_sets_native_border():
    changed = edited_ir(example(), "surface-card", "recolor")["nodes"][-1]
    assert "fill_gradient" not in changed
    assert changed["fill"] == "DDEEDD" and changed["line"]["width_pt"] == 1.5


def test_edges_and_outside_are_checked_without_cropping():
    before = np.full((50, 50, 3), 255, np.uint8)
    actual = before.copy(); expected = before.copy()
    actual[10, 10] = 0  # Old outline, not card interior.
    result = region_checks(before, actual, expected, [10, 10, 20, 20])
    assert result["export_consistency"] == "FAIL"
    assert result["regions"]["edge"]["changed_pixels"] == 1
    actual = before.copy(); actual[45, 45] = 0
    result = region_checks(before, actual, actual, [10, 10, 20, 20])
    assert result["export_consistency"] == "PASS"
    assert result["protected_pixels"] == "FAIL" and result["outside_changed_pixels"] == 1


def test_matching_export_does_not_prove_clean_background():
    before = np.full((50, 50, 3), 255, np.uint8)
    before[15:20, 15:20] = 0  # Old residual pixels could be present in both.
    result = region_checks(before, before, before, [10, 10, 20, 20])
    assert result["export_consistency"] == "PASS"
    assert clean_plate_evidence({}, example(), "text")["status"] == "REVIEW_REQUIRED"


def test_old_overflow_removal_is_authorized_by_frozen_baseline_not_candidate():
    deleted = np.full((80, 100, 3), 255, np.uint8)
    before = deleted.copy(); before[12:16, 12:16] = 0; before[12:16, 35:39] = 0
    footprint, record = baseline_contribution(before, deleted, [10, 10, 20, 10])
    assert record["baseline_overflow"] and record["baseline_overflow_pixels"] == 16
    short = deleted.copy(); short[12:16, 12:14] = 0
    result = region_checks(before, short, short, [10, 10, 20, 10], baseline_footprint=footprint)
    assert result["export_consistency"] == "PASS" and result["outside_changed_pixels"] == 0
    assert replacement_overflow(short, deleted, [10, 10, 20, 10])["overflow_detected"] is False
    short[70, 80] = 0
    assert region_checks(before, short, short, [10, 10, 20, 10], baseline_footprint=footprint)["outside_changed_pixels"] == 1


def test_move_translates_only_frozen_old_contribution():
    deleted = np.full((80, 100, 3), 255, np.uint8)
    before = deleted.copy(); before[12:16, 12:16] = 0; before[12:16, 35:39] = 0
    footprint, _ = baseline_contribution(before, deleted, [10, 10, 20, 10])
    moved = deleted.copy(); moved[shifted_mask(footprint, 20, 12)] = 0
    result = region_checks(before, moved, moved, [10, 10, 20, 10], [30, 22, 20, 10], footprint)
    assert result["outside_changed_pixels"] == 0
    moved[65, 85] = 0
    assert region_checks(before, moved, moved, [10, 10, 20, 10], [30, 22, 20, 10], footprint)["outside_changed_pixels"] == 1


def test_new_long_ink_in_old_overflow_is_still_reported_as_new_overflow():
    deleted = np.full((80, 100, 3), 255, np.uint8)
    before = deleted.copy(); before[12:16, 35:39] = 0
    footprint, _ = baseline_contribution(before, deleted, [10, 10, 20, 10])
    actual = before.copy()
    comparison = region_checks(before, actual, actual, [10, 10, 20, 10], baseline_footprint=footprint)
    assert comparison["outside_changed_pixels"] == 0
    overflow = replacement_overflow(actual, deleted, [10, 10, 20, 10])
    assert overflow["overflow_detected"] and overflow["new_overflow_pixels"] == 16


def test_clean_plate_requires_source_hash_target_and_independent_receipt(tmp_path):
    path = tmp_path / "clean.json"
    record = {"status": "PASS", "source_sha256": "source-hash", "verified_target_ids": ["text"],
              "method": "source_bound_stroke_and_surface_validation"}
    path.write_text(json.dumps(record), encoding="utf-8")
    delivery = {"verified_clean_plate": {"report_path": str(path), "report_sha256": sha256(path)}}
    assert clean_plate_evidence(delivery, example(), "text")["status"] == "PASS"
    assert clean_plate_evidence(delivery, example(), "icon")["status"] == "REVIEW_REQUIRED"
    record["method"] = "export_consistency"
    path.write_text(json.dumps(record), encoding="utf-8")
    delivery["verified_clean_plate"]["report_sha256"] = sha256(path)
    assert clean_plate_evidence(delivery, example(), "text")["status"] == "REVIEW_REQUIRED"


def test_six_required_page_three_icons_are_not_silently_skipped():
    probes = probe_specs(example(), "03")
    icons = [target for target, action in probes if target.startswith("icon-") and action == "move"]
    assert len(icons) == 6


def test_page_two_native_label_has_move_and_fill_but_raster_only_moves():
    ir = example()
    ir["nodes"].append({"id": "s02-sidebar-label", "type": "shape", "bbox": [5, 5, 30, 10]})
    assert probe_specs(ir, "02") == [("s02-sidebar-label", "move"), ("s02-sidebar-label", "recolor")]
    ir["nodes"][-1]["type"] = "image"
    assert probe_specs(ir, "02") == [("s02-sidebar-label", "move")]


def test_page_two_missing_label_is_explicit_probe_and_compound_is_not_fake_group():
    ir = example()
    assert probe_specs(ir, "02") == [("s02-sidebar-label", "move")]
    with pytest.raises(ValueError, match="Unknown probe target"):
        target_bbox(ir, "s02-sidebar-label")
    ir["nodes"].extend([{"id": "s02-sidebar-label-1", "type": "shape", "bbox": [5, 5, 25, 10]},
                        {"id": "s02-sidebar-label-2", "type": "shape", "bbox": [32, 5, 3, 10]}])
    assert target_bbox(ir, "s02-sidebar-label") == [5, 5, 30, 10]
    moved = edited_ir(ir, "s02-sidebar-label", "move")
    assert target_bbox(moved, "s02-sidebar-label") == [25, 17, 30, 10]
    assert ir["groups"] == moved["groups"]


def test_page_eight_all_four_decorations_are_required_even_on_p0_fallback():
    probes = probe_specs(example(), "08")
    assert probes == [("s08-feedback-arrow-1", "move"), ("s08-feedback-arrow-2", "move"),
                      ("s08-feedback-arrow-3", "move"), ("s08-summary-chevrons", "move")]


def test_multiple_native_components_move_together_without_claiming_actual_group(tmp_path):
    source = pptx(tmp_path / "original.pptx")
    with ZipFile(source) as z:
        parts = {name: z.read(name) for name in z.namelist()}
    parts["ppt/slides/slide1.xml"] = parts["ppt/slides/slide1.xml"].replace(b'name="text"', b'name="s02-sidebar-label-1"').replace(b'name="surface-card"', b'name="s02-sidebar-label-2"')
    native = tmp_path / "native.pptx"
    with ZipFile(native, "x") as z:
        for name, data in parts.items(): z.writestr(name, data)
    ir = example()
    ir["nodes"] = [{"id": "s02-sidebar-label-1", "type": "shape", "bbox": [10, 10, 35, 10]},
                   {"id": "s02-sidebar-label-2", "type": "shape", "bbox": [40, 40, 30, 30]}]
    ir["groups"] = []
    result = rewrite_probe(native, tmp_path / "moved.pptx", ir, "s02-sidebar-label", "move")
    assert result["semantic_component_count"] == 2 and result["actual_pptx_group"] is False
    with ZipFile(tmp_path / "moved.pptx") as z:
        root = ET.fromstring(z.read("ppt/slides/slide1.xml"))
        assert [n.attrib for n in root.findall(".//a:off", NS)] == [{"x": "300", "y": "220"}, {"x": "600", "y": "520"}]


def deck_with_reordered_parts(tmp_path):
    single = pptx(tmp_path / "single.pptx")
    with ZipFile(single) as archive:
        slide = archive.read("ppt/slides/slide1.xml")
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    path = tmp_path / "deck.pptx"
    with ZipFile(path, "x") as archive:
        archive.writestr("ppt/slides/slide7.xml", slide)
        archive.writestr("ppt/slides/slide22.xml", slide.replace(b'name="text"', b'name="other-page-text"'))
        archive.writestr("ppt/presentation.xml", f'<p:presentation xmlns:p="{NS["p"]}" xmlns:r="{rel_ns}">'
                          '<p:sldIdLst><p:sldId id="256" r:id="first"/><p:sldId id="257" r:id="second"/></p:sldIdLst>'
                          '<p:sldSz cx="1000" cy="1000"/></p:presentation>')
        archive.writestr("ppt/_rels/presentation.xml.rels", f'<Relationships xmlns="{pkg_ns}">'
                          f'<Relationship Id="first" Type="{rel_ns}/slide" Target="slides/slide22.xml"/>'
                          f'<Relationship Id="second" Type="{rel_ns}/slide" Target="slides/slide7.xml"/></Relationships>')
        archive.writestr("ppt/media/original.png", b"original")
    return path


def test_final_deck_edit_uses_presentation_order_and_preserves_other_slide(tmp_path):
    original = deck_with_reordered_parts(tmp_path)
    output = tmp_path / "edited-deck.pptx"
    result = rewrite_probe(original, output, example(), "text", "replace_short", slide_number=2)
    assert result["slide_part"] == "ppt/slides/slide7.xml"
    with ZipFile(original) as before, ZipFile(output) as after:
        assert before.read("ppt/slides/slide22.xml") == after.read("ppt/slides/slide22.xml")
        root = ET.fromstring(after.read("ppt/slides/slide7.xml"))
        assert [v.text for v in root.findall(".//a:t", NS)] == ["测", "测"]
    with pytest.raises(ValueError, match="absent"):
        rewrite_probe(original, tmp_path / "invalid.pptx", example(), "text", "delete", slide_number=3)


def test_boundary_measurement_survives_absent_independent_expectation():
    before = np.full((50, 50, 3), 255, np.uint8)
    actual = before.copy(); actual[49, 49] = 0
    result = region_checks(before, actual, None, [10, 10, 20, 20])
    assert result["export_consistency"] == "REVIEW_REQUIRED"
    assert result["outside_changed_pixels"] == 1 and result["protected_pixels"] == "FAIL"


@pytest.mark.parametrize("expected_error", [False, True])
def test_audit_final_deck_keeps_target_slide_and_unmodified_page_evidence(tmp_path, monkeypatch, expected_error):
    original = deck_with_reordered_parts(tmp_path)
    original_hash = sha256(original)
    ir = example(); ir["nodes"][0]["z"] = 10
    irfile = tmp_path / "ir.json"; irfile.write_text(json.dumps(ir), encoding="utf-8")
    blank = np.full((100, 100, 3), 255, np.uint8)
    before = blank.copy(); before[12:15, 12:15] = 0
    actual = blank.copy(); actual[12:15, 12:14] = 0
    for name, data in (("before", before), ("actual", actual), ("blank", blank)):
        Image.fromarray(data).save(tmp_path / (name+".png"))
    def expectation(delivery, changed_ir, output, backend):
        output.mkdir()
        retained = any(n["id"] == "text" for n in changed_ir["nodes"])
        if retained and expected_error:
            raise PermissionError("expected render inaccessible")
        return {"preview": str(tmp_path / ("actual.png" if retained else "blank.png"))}
    monkeypatch.setattr(r7_edit_checks, "_expected", expectation)
    class Backend:
        def render(self, path, output):
            return {"pages": [{"path": str(tmp_path / "blank.png")}, {"path": str(tmp_path / "actual.png")}]}
    result = r7_edit_checks.audit_page({"ir": str(irfile), "page": "01", "pptx": str(original),
                                       "preview": str(tmp_path / "before.png")}, tmp_path / "audit", Backend(),
        expected_probes=[("text", "replace_short")], deck_pptx=original, slide_number=2,
        baseline_deck_render={"pages": [{"path": str(tmp_path / "blank.png")}, {"path": str(tmp_path / "before.png")}]})
    assert result["final_deck_probe"] and result["slide_number"] == 2
    probe = result["probes"][0]
    assert probe["actual_changed_pixels"] == 3 and probe["outside_changed_pixels"] == 0
    assert probe["unmodified_pages"]["status"] == "PASS"
    assert result["status"] == ("FAIL" if expected_error else "REVIEW_REQUIRED")
    assert sha256(original) == original_hash


def test_unmodified_page_visual_change_fails_even_if_target_is_correct(tmp_path):
    a = np.full((5, 5, 3), 255, np.uint8); b = a.copy(); b[0, 0] = 0
    Image.fromarray(a).save(tmp_path / "a.png"); Image.fromarray(b).save(tmp_path / "b.png")
    result = r7_edit_checks.unchanged_page_checks(
        {"pages": [{"path": str(tmp_path / "a.png")}, {"path": str(tmp_path / "a.png")}]},
        {"pages": [{"path": str(tmp_path / "b.png")}, {"path": str(tmp_path / "a.png")}]}, 2)
    assert result["status"] == "FAIL" and result["pages"][0]["changed_pixels"] == 1


def test_neighbor_deletion_keeps_target_and_prunes_only_newly_empty_groups(tmp_path):
    source=pptx(tmp_path/'original.pptx')
    with ZipFile(source) as archive:
        parts={name:archive.read(name) for name in archive.namelist()}
    root=ET.fromstring(parts['ppt/slides/slide1.xml'])
    tree=root.find('p:cSld/p:spTree',NS)
    card=r7_edit_checks._find(root,'surface-card');tree.remove(card)
    group=ET.SubElement(tree,'{'+NS['p']+'}grpSp')
    props=ET.SubElement(group,'{'+NS['p']+'}nvGrpSpPr')
    ET.SubElement(props,'{'+NS['p']+'}cNvPr',{'id':'10','name':'neighbor-group'})
    ET.SubElement(group,'{'+NS['p']+'}grpSpPr');group.append(card)
    # This unrelated empty container must not be cleaned up by the diagnostic.
    empty=ET.SubElement(tree,'{'+NS['p']+'}grpSp')
    props=ET.SubElement(empty,'{'+NS['p']+'}nvGrpSpPr')
    ET.SubElement(props,'{'+NS['p']+'}cNvPr',{'id':'11','name':'unrelated-empty-group'})
    ET.SubElement(empty,'{'+NS['p']+'}grpSpPr')
    parts['ppt/slides/slide1.xml']=ET.tostring(root)
    grouped=tmp_path/'grouped.pptx'
    with ZipFile(grouped,'x') as archive:
        for name,data in parts.items():archive.writestr(name,data)
    result=r7_edit_checks._delete_neighbor_copy(grouped,tmp_path/'neighbors-deleted.pptx',example(),
        'text',[{'id':'surface-card'}],1)
    assert result['status']=='PASS' and result['removed_empty_groups']==['neighbor-group']
    with ZipFile(tmp_path/'neighbors-deleted.pptx') as archive:
        after=ET.fromstring(archive.read('ppt/slides/slide1.xml'))
        assert ET.tostring(r7_edit_checks._find(after,'text'))==ET.tostring(r7_edit_checks._find(root,'text'))
        assert r7_edit_checks._find(after,'unrelated-empty-group') is not None
        assert archive.read('ppt/media/original.png')==parts['ppt/media/original.png']


@pytest.mark.parametrize('scenario,expected_status',[
    ('transparent_margin','REVIEW_REQUIRED'),('actual_foreground','FAIL'),
    ('missing_neighbor_proof','REVIEW_REQUIRED'),('short_missing_neighbor_proof','FAIL')])
def test_replacement_collision_requires_actual_neighbor_contribution(tmp_path,monkeypatch,scenario,expected_status):
    """Actual ZIP edits plus controlled renderer outputs test evidence semantics."""
    source=pptx(tmp_path/'original.pptx')
    with ZipFile(source) as archive:
        parts={name:archive.read(name) for name in archive.namelist()}
    root=ET.fromstring(parts['ppt/slides/slide1.xml'])
    card=r7_edit_checks._find(root,'surface-card')
    card.tag='{'+NS['p']+'}pic';card.find('p:nvSpPr',NS).tag='{'+NS['p']+'}nvPicPr'
    # The image has a large transparent margin around its small visible content.
    import io
    transparent=Image.new('RGBA',(30,30),(0,0,0,0))
    for y in range(20,23):
        for x in range(20,23):transparent.putpixel((x,y),(0,0,0,255))
    data=io.BytesIO();transparent.save(data,format='PNG')
    parts['ppt/media/original.png']=data.getvalue()
    parts['ppt/slides/slide1.xml']=ET.tostring(root)
    image_deck=tmp_path/'image-deck.pptx'
    with ZipFile(image_deck,'x') as archive:
        for name,data in parts.items():archive.writestr(name,data)
    original_hash=sha256(image_deck)
    ir=example();ir['nodes']=[ir['nodes'][0],ir['nodes'][-1]];ir['nodes'][0]['z']=10
    ir['nodes'][1].update(type='image',z=5)
    irfile=tmp_path/'ir.json';irfile.write_text(json.dumps(ir),encoding='utf-8')
    blank=np.full((100,100,3),255,np.uint8)
    before=blank.copy();before[12:15,12:15]=0;before[60:63,60:63]=0
    target_deleted=blank.copy();target_deleted[60:63,60:63]=0
    neighbors_deleted=blank.copy();neighbors_deleted[12:15,12:15]=0
    actual=target_deleted.copy()
    if scenario=='actual_foreground':actual[60:62,60:62]=[255,0,0]
    else:actual[45:47,45:47]=[255,0,0]
    for name,pixels_ in [('before',before),('target_deleted',target_deleted),('neighbors_deleted',neighbors_deleted),('actual',actual)]:
        Image.fromarray(pixels_).save(tmp_path/(name+'.png'))
    def expectation(delivery,changed,output,backend):
        output.mkdir()
        retained=any(n['id']=='text' for n in changed['nodes'])
        return {'preview':str(tmp_path/('actual.png' if retained else 'target_deleted.png'))}
    monkeypatch.setattr(r7_edit_checks,'_expected',expectation)
    class Backend:
        def render(self,path,output):
            if Path(path).name=='neighbors-deleted.pptx':
                if 'missing_neighbor_proof' in scenario:raise PermissionError('diagnostic evidence unavailable')
                return {'pages':[{'path':str(tmp_path/'neighbors_deleted.png')}]}
            return {'pages':[{'path':str(tmp_path/'actual.png')}]}
    action='replace_short' if scenario=='short_missing_neighbor_proof' else 'replace_long'
    report=r7_edit_checks.audit_page({'ir':str(irfile),'page':'01','pptx':str(image_deck),
                                     'preview':str(tmp_path/'before.png')},tmp_path/'audit',Backend(),
                                    expected_probes=[('text',action)])
    probe=report['probes'][0]
    assert report['status']==expected_status
    assert probe['potential_neighbor_region_pixels']>0
    if 'missing_neighbor_proof' in scenario:
        assert probe['neighbor_evidence']['status']=='REVIEW_REQUIRED'
        assert probe['protected_overlap_pixels'] is None
        if action=='replace_long':assert probe['capacity_overflow']['verified'] is False
    else:
        evidence=probe['neighbor_evidence']
        assert evidence['status']=='PASS' and evidence['visible_pixels']==9
        assert evidence['mask_sha256']==sha256(evidence['mask'])
        assert evidence['original_pptx_sha256']==original_hash
        assert evidence['original_ir_sha256']==sha256(irfile)
        assert probe['protected_overlap_pixels']==(4 if scenario=='actual_foreground' else 0)
        if scenario=='transparent_margin':assert probe['capacity_overflow']['verified'] is True
    assert sha256(image_deck)==original_hash
