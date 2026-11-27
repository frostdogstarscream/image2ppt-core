"""Mechanism checks; synthetic images do not establish slide quality."""
import base64
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'services/vision/src'))
from image2ppt_vision.layered import plan_first as p


def payload(rows):
    return {'schema': 'LayeredPlan/1', 'source_sha256': 'test', 'image_size': [200, 100],
            'plan': {'schema': 'SegmentationPlan/1', 'summary': '', 'risks': [], 'regions': rows}}


def row(id='G1', kind='graphic', action='keep_pixels', box=None, parent=None):
    return {'id': id, 'kind': kind, 'action': action, 'bbox': box or [100, 100, 400, 700],
            'parent_id': parent, 'label': '', 'reason': ''}


def test_module_leaf_guides_before_sam_and_container_does_not_duplicate():
    rgb = np.full((100, 200, 3), 255, np.uint8); rgb[15:60, 25:70] = 0
    rows = [row('M1', 'module', box=[50, 50, 500, 800]), row(parent='M1')]
    work, mask, report = p.guide(rgb, [], [], payload(rows), 'test')
    assert [r['status'] for r in report['regions']] == ['container', 'guided']
    assert mask[30, 30] and (work[30, 30] == 255).all()
    assert (rgb[30, 30] == 0).all()
    leaf = payload([row('M1', 'module')])
    assert p.guide(rgb, [], [], leaf, 'test')[2]['regions'][0]['status'] == 'guided'


def test_reject_crossing_text_table_and_overlapping_plans():
    rgb = np.full((100, 200, 3), 255, np.uint8)
    lines = [{'bbox': [60, 25, 120, 12]}]
    assert p.guide(rgb, lines, [], payload([row()]), 'test')[2]['regions'][0]['status'] == 'fallback'
    assert p.guide(rgb, [], [{'bbox': [25, 20, 40, 20]}], payload([row()]), 'test')[2]['regions'][0]['status'] == 'fallback'
    report = p.guide(rgb, [], [], payload([row(), row('G2')]), 'test')[2]
    assert [r['status'] for r in report['regions']] == ['guided', 'fallback']


def test_planned_crop_has_no_duplicate_raster_pixels():
    from image2ppt_vision.layered.review import _hard_rect
    old = _hard_rect([0, 0, 140, 90], 100, 200); old['id'] = 'el-000'
    report = {'regions': [{'id': 'G1', 'bbox': [20, 10, 60, 60], 'status': 'guided'}]}
    elements, erase = p.install([old], report, 100, 200)
    assert len(elements) == 2
    assert not (elements[0]['mask'] & elements[1]['mask']).any()
    assert erase[30, 30] and elements[1]['plan_region_id'] == 'G1'


def test_synthetic_planning_fill_cannot_become_native_shape():
    claimed = np.zeros((100, 200), bool); claimed[20:60, 30:80] = True
    false_card = np.ones_like(claimed)
    genuine_line = np.zeros_like(claimed); genuine_line[:, 120] = True
    kept = p.unclaimed_candidates([(false_card, {'id': 'fake'}), (genuine_line, {'id': 'line'})], claimed)
    assert len(kept) == 1 and kept[0][1]['id'] == 'line'


def bundle(tmp_path):
    Image.new('RGB', (200, 100), 'white').save(tmp_path/'source.png')
    plan = payload([row()])['plan']
    request = {'body': {'messages': [{'content': [{'type': 'image_url', 'image_url': {
        'url': 'data:image/png;base64,'+base64.b64encode((tmp_path/'source.png').read_bytes()).decode()}}]}]}}
    p.write(tmp_path/'request.json', request); p.write(tmp_path/'response.json', {'ok': True, 'content': json.dumps(plan)})
    p.write(tmp_path/'plan.json', plan)
    p.write(tmp_path/'binding.json', {'source_sha256': p.sha(tmp_path/'source.png'), 'request_sha256': p.sha(tmp_path/'request.json')})
    p.write(tmp_path/'result.json', {'status': 'REVIEW_REQUIRED', 'response_sha256': p.sha(tmp_path/'response.json')})


def test_reject_modified_plan_and_response(tmp_path):
    bundle(tmp_path)
    assert p.load_bundle(tmp_path, tmp_path/'source.png')[0]['schema'] == 'SegmentationPlan/1'
    file = tmp_path/'plan.json'; plan = p.read(file); plan['regions'][0]['bbox'][0] = 0
    file.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='differs'): p.load_bundle(tmp_path, tmp_path/'source.png')
    (tmp_path/'response.json').write_text('{}')
    with pytest.raises(ValueError, match='response hash'): p.load_bundle(tmp_path, tmp_path/'source.png')


def test_fail_binding_before_segmentation():
    with pytest.raises(ValueError, match='binding'):
        p.guide(np.zeros((100, 200, 3), np.uint8), [], [], payload([row()]), 'wrong')


def test_worker_uses_plan_before_sam_and_review_preserves_it(tmp_path):
    from image2ppt_vision.layered.perceive import perceive
    from image2ppt_vision.layered.review import review
    rgb = np.full((100, 200, 3), 255, np.uint8)
    rgb[20:55, 30:60] = [45, 90, 170]
    source = tmp_path/'source.png'; Image.fromarray(rgb).save(source)
    value = payload([row()]); value['source_sha256'] = p.sha(source)
    observed = []
    def sam(image):
        observed.append(image.copy()); return []
    out = tmp_path/'perception'
    result = perceive(source, out, ocr=lambda _: [], sam=sam, plan=value)
    assert (observed[0][30, 40] == 255).all()
    assert any(e.get('source') == 'plan_first' for e in result['elements'])
    class NoCall:
        def request(self, *args, **kwargs): raise AssertionError('unexpected model call')
    revised, report = review(out, result, remote=NoCall(), model='unused', quality_loop=True)
    assert not report['model_used']
    assert any(e.get('plan_region_id') == 'G1' for e in revised['elements'])
    assert p.sha(source) == value['source_sha256']
