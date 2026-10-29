"""Mechanism tests. These synthetic cases do not claim real-slide acceptance."""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'services/vision/src')]
from image2ppt_vision.layered.loop_policy import LoopPolicy
from image2ppt_vision.layered.loop_review import CropReviewer, validate_decisions
from image2ppt_vision.layered.loop_scan import candidates, protect_masks, guarded_lines
from image2ppt_vision.layered.loop_text import compare_text, regional_text
from image2ppt_vision.layered.loop_checks import issue, better, graphic_losses, final_file, verify_inventory, inventory
from image2ppt_vision.layered.loop_regions import restore, safe_order, group_modules
from image2ppt_vision.layered import review


@pytest.mark.parametrize('value', [-1, 3, True, 1.5])
def test_budgets_reject_invalid(value):
    with pytest.raises(ValueError): LoopPolicy(value, 2)
    with pytest.raises(ValueError): LoopPolicy(2, value)


def test_failed_model_call_consumes_budget(tmp_path):
    class Remote:
        calls = 0
        def request(self, *a, **k):
            self.calls += 1
            raise TimeoutError()
    remote = Remote(); image = tmp_path/'source.png'; Image.new('RGB', (50, 50), 'white').save(image)
    row = {'id': 't0', 'bbox': [1, 1, 10, 10], 'context_bbox': [0, 0, 20, 20], 'text': '1'}
    reviewer = CropReviewer(tmp_path/'review', remote=remote, model='test', max_requests=1)
    a = reviewer.run([row], image, phase='pre'); b = reviewer.run([row], image, phase='post')
    assert remote.calls == 1 and a['requests'] == 1 and b['requests'] == 0
    assert 'visual_call_failed' in a['issues'] and 'visual_budget_exhausted' in b['issues']
    assert a['decisions'][0]['kind'] == 'uncertain'


def test_invalid_duplicate_review_decisions():
    row = {'id': 't0', 'kind': 'text', 'text': '12', 'reason': 'visible'}
    decisions, _ = validate_decisions({'regions': [row, row]}, {'t0'})
    assert decisions == []


def test_decoration_is_candidate_and_protected():
    scan = {'image_size': [200, 100], 'source_sha256': 'x',
            'ocr': [{'id': 't0', 'text': '1 Heading', 'bbox': [10, 10, 90, 20], 'score': .99}], 'text': []}
    assert 'possible_leading_decoration' in candidates(scan)[0]['reasons']
    text, icons = np.ones((100, 200), bool), np.zeros((100, 200), bool)
    protect_masks(text, icons, scan['ocr'], [], {'retained': {'t0': 'uncertain'}})
    assert not text[15, 15] and icons[15, 15]
    with pytest.raises(ValueError):
        guarded_lines(scan, {'source_sha256': 'x', 'corrections': {'t0': {'text': 'Heading', 'independent_text': 'Other'}}})
    assert scan['ocr'][0]['text'] == '1 Heading'


def test_text_order_numbers_units():
    assert compare_text('12人', '21人')['numbers_exact'] is False
    assert compare_text('10万元', '10元')['exact'] is False
    assert compare_text('ABCD', 'CDAB')['reordered'] is True
    assert compare_text('12.5%', '12.5')['numbers_exact'] is False


def test_low_contrast_graphic_loss_and_ocr_exclusion():
    source = np.full((200, 400, 3), 255, np.uint8); source[70:73, 80:190] = 220
    missing = np.full_like(source, 255)
    assert graphic_losses(source, missing)
    assert not graphic_losses(source, source)
    assert not graphic_losses(source, missing, [[70, 60, 130, 30]])
    assert graphic_losses(source, missing, [[70, 60, 130, 30]], protected=[[70, 60, 130, 30]])


def test_candidates_never_trade_new_confirmed_loss():
    previous = {'issues': [], 'region_error': 10, 'native_count': 5}
    new = {'issues': [issue('missing', target='a', confirmed=True)], 'region_error': 0, 'native_count': 6}
    assert not better(new, previous)
    assert better({**previous, 'region_error': 8}, previous)
    assert not better({**previous, 'native_count': 4}, previous)


def test_newly_exposed_duplicate_does_not_mask_reduced_blockers():
    old = {'issues': [issue('actual_duplicate', target=str(i), confirmed=True, severity='blocking') for i in range(4)],
           'region_error': 10, 'native_count': 5}
    new = {**old, 'issues': [issue('actual_duplicate', target='exposed', confirmed=True, severity='blocking')]}
    assert better(new, old)
    new['issues'].append(issue('output_text_changed', target='text', confirmed=True, severity='blocking'))
    assert not better(new, old)


def test_inventory_detects_modified_source(tmp_path):
    f = tmp_path/'source'; f.write_bytes(b'a'); frozen = inventory(tmp_path); f.write_bytes(b'b')
    with pytest.raises(ValueError): verify_inventory(tmp_path, frozen)


def fixture_ir(tmp_path):
    (tmp_path/'assets').mkdir(); Image.new('RGB', (400, 200), 'white').save(tmp_path/'source.png')
    Image.new('RGBA', (200, 100), (0, 0, 0, 255)).save(tmp_path/'assets'/'picture.png')
    return {'canvas': {'source_px': [400, 200]}, 'source_image': 'source.png', 'groups': [],
            'evidence': [{'id': 't', 'kind': 'text', 'bbox': [40, 40, 30, 20]}, {'id': 'pic', 'kind': 'image', 'bbox': [10, 10, 200, 100]}],
            'unresolved_evidence_ids': [],
            'assets': [{'id': 'picture', 'path': 'assets/picture.png', 'contains_evidence_ids': ['pic']}],
            'nodes': [{'id': 'picture', 'type': 'image', 'asset_id': 'picture', 'bbox': [10, 10, 200, 100], 'z': 0, 'evidence_ids': []},
                      {'id': 'text', 'type': 'text', 'text': '12', 'bbox': [40, 40, 30, 20], 'z': 1, 'evidence_ids': ['t']}]}


def test_retention_removes_native_and_punches_underlying_raster(tmp_path):
    ir = fixture_ir(tmp_path)
    report = restore(ir, tmp_path, [{'bbox': [45, 45, 10, 10]}])
    assert len(report['regions']) == 1 and 'text' not in {n['id'] for n in ir['nodes']}
    assert report['regions'][0]['editable'] == 'move_resize_only'
    asset = next(a for a in ir['assets'] if a['id'] == 'picture')
    with Image.open(tmp_path/asset['path']) as im: assert im.getpixel((40, 40))[3] == 0
    owners = [a['id'] for a in ir['assets'] if 't' in a['contains_evidence_ids']]
    assert len(owners) == 1


def test_whole_page_fallback_rejected(tmp_path):
    ir = fixture_ir(tmp_path); before = copy.deepcopy(ir)
    result = restore(ir, tmp_path, [{'bbox': [0, 0, 400, 200]}])
    assert result['rejected'] and ir['nodes'] == before['nodes']


def test_grouping_cannot_invert_overlap():
    nodes = [{'id': n, 'z': i, 'bbox': [0, 0, 40, 40]} for i, n in enumerate('abc')]
    assert safe_order(nodes, [{'id': 'g', 'member_ids': ['a', 'c']}]) is None


def test_shape_with_internal_artwork_falls_back():
    rgb = np.zeros((100, 100, 3), np.uint8); mask = np.ones((100, 100), bool)
    record = {'icons': [mask], 'kind': 'native_shape'}
    with patch.object(review, '_contour', return_value=mask), patch.object(review, 'measure', return_value=record):
        assert review._measured_shape(rgb, np.zeros((100, 100)), [0, 0, 100, 100], strict=True) is None


def test_actual_pptx_missing_node_is_blocking():
    ir = {'nodes': [{'id': 'text', 'type': 'text', 'text': '10', 'bbox': [1, 1, 20, 20]}], 'groups': []}
    with patch('image2ppt.quality.inspect_pptx', return_value={'slide_count': 1, 'slides': [{'objects': []}]}), \
         patch('image2ppt_vision.layered.loop_checks.digest', return_value='x'):
        assert final_file(ir, 'unused')['issues'][0]['severity'] == 'blocking'


def test_native_shape_is_cropped_out_from_under_retention(tmp_path):
    ir = fixture_ir(tmp_path)
    ir['nodes'].insert(0, {'id': 'card', 'type': 'shape', 'geometry': 'rect', 'fill': 'FFFFFF',
                           'bbox': [0, 0, 250, 120], 'z': -1, 'evidence_ids': []})
    report = restore(ir, tmp_path, [{'bbox': [40, 40, 30, 20]}])
    box = report['regions'][0]['bbox']
    from image2ppt_vision.layered.loop_review import intersect
    assert all(not intersect(n['bbox'], box) for n in ir['nodes'] if n['type'] == 'shape')


def test_icon_damage_gets_local_mask_repair():
    from image2ppt_vision.layered.loop_inpaint import repair_icons
    source = np.full((150, 300, 3), 255, np.uint8); source[40:70, 40:70] = 40
    cleaned = np.full_like(source, 255); calls = []
    def inpaint(rgb, mask, model):
        calls.append(mask.copy()); return rgb.copy(), 'synthetic'
    records = [{'ocr_bbox': [30, 30, 100, 50], 'icon_candidates': [[35, 35, 40, 40]]}]
    result, report = repair_icons(source, cleaned, np.ones((150, 300), bool), records, inpaint=inpaint)
    assert calls and report['actions'][0]['action'] == 'local_mask_repair'
    assert np.array_equal(result[35:75, 35:75], source[35:75, 35:75])


def test_disjoint_probe_batches_cover_every_candidate_once():
    from image2ppt_vision.layered.loop_probes import disjoint_batches
    from image2ppt_vision.layered.loop_review import intersect
    rows = [{'id': str(i), 'bbox': box} for i, box in enumerate([[0, 0, 20, 20], [10, 0, 20, 20], [40, 0, 20, 20]])]
    batches = disjoint_batches(rows)
    assert sorted(n['id'] for b in batches for n in b) == ['0', '1', '2']
    assert all(not intersect(a['bbox'], b['bbox']) for group in batches for i, a in enumerate(group) for b in group[i+1:])


def test_final_duplicate_requires_actual_hidden_copy(tmp_path):
    from image2ppt_vision.layered import loop_probes
    ir = fixture_ir(tmp_path)
    preview = tmp_path/'preview.png'; canvas = np.full((200, 400, 3), 255, np.uint8)
    canvas[30:55, 30:55] = 0; Image.fromarray(canvas).save(preview)
    pptx = tmp_path/'test.pptx'; pptx.write_bytes(b'original')
    node = {'id': 'picture', 'bbox': [10, 10, 100, 100]}
    class Backend:
        def render(self, path, out): return {'pages': [{'path': str(preview)}]}
    def hide(source, target, names): target.write_bytes(b'edited')
    with patch.object(loop_probes, 'duplicate_candidates', return_value=[node]), patch.object(loop_probes, 'hide_copy', side_effect=hide):
        result = loop_probes.run(ir, tmp_path, pptx, preview, tmp_path/'probes', Backend())
    assert result['issues'][0]['code'] == 'actual_duplicate' and result['issues'][0]['confirmed']
    assert pptx.read_bytes() == b'original'


def test_quality_scan_and_perception_preserve_original_evidence(tmp_path):
    import cv2
    from image2ppt_vision.layered.loop_scan import scan
    from image2ppt_vision.layered.perceive import perceive
    from image2ppt_vision.storage import digest
    rgb = np.full((180, 320, 3), 255, np.uint8)
    cv2.putText(rgb, 'REPORT', (40, 65), cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 40, 80), 2)
    source = tmp_path/'source.png'; Image.fromarray(rgb).save(source); original = digest(source)
    def ocr(image): return [{'text': 'REPORT', 'bbox': [38, 46, 94, 23], 'score': .99}]
    scanned = scan(source, tmp_path/'scan', ocr)
    guards = {'source_sha256': scanned['source_sha256'], 'corrections': {}, 'retained': {'t000': 'uncertain'}}
    result = perceive(source, tmp_path/'perception', ocr=ocr, sam=None, scan=scanned, guards=guards)
    assert result['original_ocr'] == scanned['ocr']
    assert result['protected_regions'] and result['local_mask_repairs'] is not None
    assert digest(source) == original


def test_early_worker_failure_saves_failed_job_without_overwriting(tmp_path, monkeypatch):
    from image2ppt_vision.layered import quality_loop
    monkeypatch.chdir(tmp_path)
    def fail(inputs, output, **options):
        root = Path(output)/options['job']; (root/'input').mkdir(parents=True)
        assert root.is_absolute()
        raise RuntimeError('synthetic worker unavailable')
    with patch.object(quality_loop, '_run', side_effect=fail):
        result = quality_loop.run(['source.png'], 'output', job='failure-test')
        assert result['status'] == 'FAILED' and result['pages']['source']['status'] == 'FAILED'
        assert (tmp_path/'output/failure-test/job.json').is_file()
        with pytest.raises(FileExistsError): quality_loop.run(['source.png'], 'output', job='failure-test')
