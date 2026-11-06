import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from relation_validator import prepare_spec, check_relations, check_authorized_changes, check_edit_regions


REVIEW_ROOT = os.environ.get('IMAGE2PPT_R6_REVIEW_ROOT')
REPO = Path(__file__).resolve().parents[2]
REVIEW_DIRS = [root/'outputs/review-4090-20261106-r6/delivery' for root in [REPO, *REPO.parents]]
SAMPLES = (Path(REVIEW_ROOT) if REVIEW_ROOT else next(
    (path for path in REVIEW_DIRS if (path/'pages').is_dir()), REVIEW_DIRS[0]))/'pages'


@pytest.mark.parametrize('page,count', [('02', 1), ('08', 4)])
def test_frozen_real_source_passes_and_r6_missing_elements_fail(tmp_path, page, count):
    source, current = SAMPLES/page/'source.png', SAMPLES/page/'r6.png'
    if not source.exists():
        if REVIEW_ROOT: pytest.fail('Configured IMAGE2PPT_R6_REVIEW_ROOT lacks source: '+str(source))
        pytest.skip('Private source regression images not installed')
    spec = prepare_spec(page, source, tmp_path)
    assert len(spec['elements']) == count
    assert check_relations(source, source, spec)['status'] == 'PASS'
    result = check_relations(source, current, spec)
    assert result['status'] == 'FAIL'
    assert all(x['status'] == 'FAIL' for x in result['objects'])


def test_unauthorized_and_protected_change_is_rejected():
    before = np.zeros((20,20,3), np.uint8)
    allowed = np.zeros((20,20), bool); allowed[5:10,5:10] = True
    protected = np.zeros_like(allowed); protected[7,7] = True
    after = before.copy(); after[7,7] = 255
    assert check_authorized_changes(before, after, allowed, protected)['status'] == 'FAIL'
    after = before.copy(); after[0,0] = 255
    assert check_authorized_changes(before, after, allowed)['outside_changed_pixels'] == 1


def test_edit_gate_requires_independent_expectation_and_checks_edges():
    before = np.zeros((20,20,3), np.uint8); after = before.copy()
    old = np.zeros((20,20), bool); old[2:6,2:6] = True
    new = np.zeros_like(old); new[12:16,12:16] = True
    protected = np.zeros_like(old)
    args = dict(allowed_mask=old|new, old_mask=old, new_mask=new, protection_mask=protected)
    assert check_edit_regions(before,after,before,**args)['status'] == 'REVIEW_REQUIRED'
    after[2,2] = 255
    result = check_edit_regions(before,after,before,expected_plate_verified=True,**args)
    assert result['status'] == 'FAIL'
    assert result['regions']['edge_ring']['status'] == 'FAIL'


def configured_spec(tmp_path, pixels, elements, *, page='test'):
    source = tmp_path/'source.png'
    Image.fromarray(pixels).save(source)
    catalog = {'schema': 'RelationCatalog/1', 'pages': {page: {'elements': elements}}}
    return source, prepare_spec(page, source, tmp_path/'spec', config=catalog)


def test_dynamic_canvas_and_nonblue_configured_icon(tmp_path):
    source_pixels = np.full((47, 83, 3), 255, np.uint8)
    source_pixels[10:30, 20:50] = [180, 20, 20]
    source, spec = configured_spec(tmp_path, source_pixels, [{
        'id': 'red-icon', 'kind': 'ICON_PRESENCE', 'bbox_xyxy': [18, 8, 52, 32],
        'segmentation': 'color_distance', 'foreground_rgb': [180, 20, 20], 'color_tolerance': 10}])
    assert spec['canvas_px'] == [83, 47]
    assert check_relations(source, source, spec)['status'] == 'PASS'
    erased = source_pixels.copy(); erased[10:30, 20:50] = 255
    assert check_relations(source, erased, spec)['status'] == 'FAIL'


@pytest.mark.parametrize('remaining', [0, 1])
def test_label_text_all_deleted_or_one_white_pixel_fails(tmp_path, remaining):
    pixels = np.full((40, 80, 3), 255, np.uint8)
    pixels[5:35, 5:75] = [10, 80, 170]
    pixels[15:23, 15:21] = 255
    pixels[15:23, 32:38] = 255
    source, spec = configured_spec(tmp_path, pixels, [{
        'id': 'label', 'kind': 'LABEL_CONTRAST', 'bbox_xyxy': [5,5,75,35],
        'text_target_id': 'text-1', 'text_bbox_xyxy': [10,10,65,30]}])
    assert check_relations(source, source, spec)['status'] == 'PASS'
    candidate = pixels.copy(); candidate[5:35, 5:75] = [10, 80, 170]
    if remaining: candidate[18, 18] = 255
    result = check_relations(source, candidate, spec)
    assert result['status'] == 'FAIL'
    row = result['objects'][0]
    assert row['source_text_pixels'] == 96
    assert row['candidate_text_pixels'] == remaining
    assert 'label_text_contribution_missing' in row['reasons']


def test_unknown_kind_and_undefined_scope_remain_pending(tmp_path):
    pixels = np.full((20,30,3), 255, np.uint8)
    source, spec = configured_spec(tmp_path, pixels, [{'id': 'unknown', 'kind': 'not-supported'}])
    result = check_relations(source, source, spec)
    assert result['status'] == 'REVIEW_REQUIRED'
    assert result['coverage']['expected'] == 1
    assert result['coverage']['executed'] == 0
    assert result['coverage']['executed_pass_rate'] is None
    spec['elements'] = []; spec['expected_relation_ids'] = []
    empty = check_relations(source, source, spec)
    assert empty['status'] == 'REVIEW_REQUIRED'
    assert empty['coverage']['execution_coverage'] is None


def test_required_relation_does_not_disappear_with_object(tmp_path):
    source, spec = configured_spec(tmp_path, np.zeros((20,30,3), np.uint8), [])
    spec['expected_relation_ids'] = ['missing-icon']
    result = check_relations(source, source, spec)
    assert result['status'] == 'FAIL'
    assert result['coverage']['expected'] == 1
    assert result['coverage']['defined'] == 0
    assert result['coverage']['failed'] == 1


def test_not_applicable_requires_reason_and_never_counts_pass(tmp_path):
    source, spec = configured_spec(tmp_path, np.zeros((20,30,3), np.uint8), [
        {'id': 'no-arrow', 'kind': 'ARROW_CONNECTIVITY', 'applicable': False,
         'not_applicable_reason': 'page_has_no_flow_connection'}])
    result = check_relations(source, source, spec)
    assert result['status'] == 'NOT_APPLICABLE'
    assert result['coverage']['passed'] == 0
    assert result['coverage']['executed_pass_rate'] is None
    del spec['elements'][0]['not_applicable_reason']
    assert check_relations(source, source, spec)['status'] == 'FAIL'


def test_missing_plate_cannot_hide_protected_or_outside_changes():
    before = np.zeros((20,20,3), np.uint8)
    edited = before.copy(); edited[0,0] = 255
    empty = np.zeros((20,20), bool)
    protected = empty.copy(); protected[0,0] = True
    result = check_edit_regions(before, edited, None, allowed_mask=empty, old_mask=empty,
        new_mask=empty, protection_mask=protected, expected_plate_verified=False)
    assert result['status'] == 'FAIL'
    assert result['outside_changed_pixels'] == result['protected_changed_pixels'] == 1
    assert result['regions']['old_site']['status'] == 'REVIEW_REQUIRED'


def test_empty_inspection_regions_are_pending_not_hard_failure():
    pixels = np.zeros((20,20,3), np.uint8)
    empty = np.zeros((20,20), bool)
    result = check_edit_regions(pixels, pixels, pixels, allowed_mask=empty, old_mask=empty,
        new_mask=empty, protection_mask=empty, expected_plate_verified=True)
    assert result['status'] == 'REVIEW_REQUIRED'
    assert result['outside_changed_pixels'] == 0


def test_mask_origin_and_bbox_are_validated(tmp_path):
    pixels = np.zeros((40,80,3), np.uint8); pixels[10:30,20:50] = [0,80,170]
    source, spec = configured_spec(tmp_path, pixels, [
        {'id': 'icon', 'kind': 'ICON_PRESENCE', 'bbox_xyxy': [18,8,52,32]}])
    spec['elements'][0]['masks']['source_ink']['origin_px'][0] += 1
    with pytest.raises(ValueError, match='origin/extent'):
        check_relations(source, source, spec)
    with pytest.raises(ValueError, match='outside source canvas'):
        prepare_spec('test', source, tmp_path/'bad-spec', config={
            'schema': 'RelationCatalog/1', 'pages': {'test': {'elements': [
                {'id': 'bad', 'kind': 'ICON_PRESENCE', 'bbox_xyxy': [-1,0,10,10]}]}}})


def test_arrow_direction_supports_up_and_detects_reversal(tmp_path):
    import cv2
    pixels = np.full((60,60,3), 255, np.uint8)
    points = np.array([[30,10],[44,26],[35,26],[35,48],[25,48],[25,26],[16,26]])
    cv2.fillPoly(pixels, [points], [0,80,170])
    source, spec = configured_spec(tmp_path, pixels, [
        {'id': 'up', 'kind': 'ARROW_CONNECTIVITY', 'bbox_xyxy': [10,5,50,55], 'direction': 'up'}])
    assert check_relations(source, source, spec)['status'] == 'PASS'
    flipped = pixels[::-1].copy()
    result = check_relations(source, flipped, spec)
    assert result['status'] == 'FAIL'
    assert result['objects'][0]['direction'] == 'up'


@pytest.mark.parametrize('page, expected', [('03', 6), ('04', 7), ('10', 3)])
def test_real_expanded_inventory_and_missing_evidence_are_counted(tmp_path, page, expected):
    source, ir = SAMPLES/page/'source.png', SAMPLES/page/'revision/slideir.json'
    if not source.exists() or not ir.exists():
        pytest.skip('Private source regression scene not installed')
    spec = prepare_spec(page, source, tmp_path/'with-evidence', ir=ir)
    result = check_relations(source, source, spec)
    assert result['coverage']['expected'] == expected
    assert result['coverage']['executed'] == expected
    assert all(not row['repairable'] for row in result['objects'])
    # Surface scope is intentionally partial: no independent fill/occlusion proof.
    assert result['status'] == ('PASS' if page == '03' else 'REVIEW_REQUIRED')
    pending_spec = prepare_spec(page, source, tmp_path/'without-evidence')
    pending = check_relations(source, source, pending_spec)
    assert pending['coverage']['expected'] == expected
    assert pending['coverage']['review_required'] > 0


@pytest.mark.parametrize('identifier', ['icon-crane', 'icon-building', 'icon-laptop',
                                       'icon-organization', 'icon-analysis', 'icon-handshake'])
def test_each_of_six_real_icons_has_independent_deletion_gate(tmp_path, identifier):
    source, ir = SAMPLES/'03/source.png', SAMPLES/'03/revision/slideir.json'
    if not source.exists() or not ir.exists():
        pytest.skip('Private source regression scene not installed')
    spec = prepare_spec('03', source, tmp_path/'spec', ir=ir)
    erased = np.asarray(Image.open(source).convert('RGB')).copy()
    target = next(e for e in spec['elements'] if e['id'] == identifier)
    x, y, right, bottom = target['bbox_xyxy']
    erased[y:bottom, x:right] = 255
    result = check_relations(source, erased, spec)
    failed = [r['id'] for r in result['objects'] if r['status'] == 'FAIL']
    assert failed == [identifier]


def test_surface_missing_boundary_is_failure_despite_incomplete_precision_evidence(tmp_path):
    import cv2
    pixels = np.full((60,100,3), 255, np.uint8)
    cv2.rectangle(pixels, (12,12), (88,48), (0,80,170), 2)
    source, spec = configured_spec(tmp_path, pixels, [{
        'id': 'card', 'kind': 'SURFACE_GEOMETRY', 'bbox_xyxy': [9,9,92,52]}])
    assert check_relations(source, source, spec)['status'] == 'REVIEW_REQUIRED'
    result = check_relations(source, np.full_like(pixels,255), spec)
    assert result['status'] == 'FAIL'
    assert 'source_contribution_missing' in result['objects'][0]['reasons']


@pytest.mark.parametrize('page', ['02', '08'])
def test_existing_r7_fixed_elements_remain_pass(tmp_path, page):
    roots = [root/'outputs/reconstruction-r7-20261106/review-evidence/delivery-20261106T090914-e31b01f8/pages'
             for root in [REPO, *REPO.parents]]
    samples = next((root for root in roots if (root/page/'r7.png').is_file()), None)
    if samples is None:
        pytest.skip('Private r7 review images not installed')
    source = samples/page/'source.png'
    spec = prepare_spec(page, source, tmp_path/'spec')
    assert check_relations(source, samples/page/'r7.png', spec)['status'] == 'PASS'
