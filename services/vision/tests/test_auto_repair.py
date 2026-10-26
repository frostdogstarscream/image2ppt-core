"""Deterministic repair contracts. Synthetic fixtures; they do not establish slide fidelity."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import urllib.request

import pytest

from image2ppt_vision.auto_provider import fingerprint
from image2ppt_vision.auto_repair import (repair_layout, align_decision, normalize_draft_colors, derive_mapping,
                                          tagged_view)
from image2ppt_vision.auto_region_protocol import (layout_from_draft, layout_with_repair, decision_record,
                                                   expand_region, validate_region)
from image2ppt_vision.auto_workflow import checked_model_json
from image2ppt_vision.region_plan import validate_decision
from test_auto_regions import fixture, routing, bound, native_draft

spec = importlib.util.spec_from_file_location('auto_remote_repair_test', Path(__file__).parents[1]/'tools/auto_remote.py')
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def ops(record):
    return [o['op'] for o in record['operations']]


# ------------------------------------------------------------------ layout

def test_subregion_outside_parent_grows_parent_when_safe():
    inv, _, draft = fixture()
    draft['regions'][0]['bbox'] = [0, 0, 1000, 200]
    draft['regions'][0]['subregions'] = [{'id': 'title', 'kind': 'title', 'bbox': [0, 0, 1000, 240]}]
    raw = deepcopy(draft)
    repaired, record = repair_layout(draft, inv)
    assert draft == raw and record['raw_sha256'] == fingerprint(raw)
    assert repaired['regions'][0]['bbox'] == [0, 0, 1000, 240] and ops(record) == ['grow_region_for_subregion']
    assert layout_from_draft(repaired, inv)['regions'][0]['subregions'][0]['id'] == 'title'


def test_subregion_is_clipped_when_growing_would_overlap_a_neighbor():
    inv, _, draft = fixture()
    draft['regions'][2]['subregions'] = [{'id': 'footer', 'kind': 'text', 'bbox': [0, 700, 500, 1000]}]
    repaired, record = repair_layout(draft, inv)
    assert repaired['regions'][2]['bbox'] == [0, 800, 1000, 1000]
    assert repaired['regions'][2]['subregions'][0]['bbox'] == [0, 800, 500, 1000] and ops(record) == ['clip_subregion']
    layout_from_draft(repaired, inv)


def test_unowned_ocr_is_assigned_to_the_region_that_already_covers_most_of_it():
    inv, _, draft = fixture(); draft['regions'][0]['bbox'] = [0, 0, 1000, 150]
    with pytest.raises(ValueError, match='OCR must belong'):
        layout_from_draft(draft, inv)
    layout = layout_with_repair(draft, inv)
    assert layout['regions'][0]['text_ids'] == ['t1']
    assert layout['program_repair']['operations'][0]['op'] == 'grow_region_for_ocr'


def test_substantial_overlap_is_cut_without_orphaning_ocr():
    inv, _, draft = fixture(); draft['regions'][1]['bbox'] = [0, 200, 1000, 800]
    with pytest.raises(ValueError, match='overlap'):
        layout_from_draft(draft, inv)
    layout = layout_with_repair(draft, inv)
    assert [r['text_ids'] for r in layout['regions']] == [['t1'], ['t2'], ['t3']]
    assert ops(layout['program_repair']) == ['resolve_overlap']


def test_classification_alias_and_color_notation_are_normalized():
    inv, _, draft = fixture(); draft['regions'][0]['classification'] = 'Header'; draft['background_color'] = '#ffffff'
    layout = layout_with_repair(draft, inv)
    assert layout['regions'][0]['classification'] == 'decorative' and layout['background_color'] == 'FFFFFF'


def test_valid_layout_records_nothing_and_unrepairable_layout_still_fails():
    inv, _, draft = fixture()
    assert 'program_repair' not in layout_with_repair(draft, inv)
    draft['regions'][0]['bbox'] = [0, 0, 1000, 1000]
    with pytest.raises(ValueError, match='whole-page'):
        layout_with_repair(draft, inv)
    broken = {'regions': 'not a list'}
    assert repair_layout(broken, inv)[0] == broken


# ---------------------------------------------------------------- decisions

def decision(native, retained=(), kind='native', reasons=()):
    value = routing(kind)
    value.update(native_parts=list(native), retained_parts=list(retained), retain_reasons=list(reasons))
    value['can_rebuild']['native_parts'] = list(native)
    return value


def test_icon_artwork_is_downgraded_to_retained_pixels():
    aligned, operations = align_decision(decision(['section_title', 'title_icon']))
    checked = validate_decision(aligned)
    assert checked['decision'] == 'hybrid' and checked['native_parts'] == ['section_title']
    assert checked['retained_parts'] == ['icons'] and checked['retain_reasons'] == ['non_native_geometry']
    assert {'op': 'decision_kind_follows_scope', 'from': 'native', 'to': 'hybrid'} in operations
    only_icon, _ = align_decision(decision(['icon']))
    assert validate_decision(only_icon)['decision'] == 'retain_module'


def test_decision_record_binds_alignment_and_leaves_clean_decisions_untouched():
    inv, _, draft = fixture(); layout = layout_from_draft(draft, inv); region = layout['regions'][0]
    raw = decision(['labels', 'title_icon'])
    record = decision_record(raw, region=region, layout=layout, crop_sha256='b'*64, capabilities={}, response_sha256='c'*64)
    assert record['program_alignment']['model_decision_sha256'] == fingerprint(raw)
    assert 'program_alignment' not in bound(layout, region)


# ----------------------------------------------------------------- mapping

def hybrid_record(layout, region, native, retained=('module_body',)):
    raw = decision(native, retained, 'hybrid', ['ui_mockup'])
    return decision_record(raw, region=region, layout=layout, crop_sha256='b'*64, capabilities={}, response_sha256='c'*64)


def tagged_draft(region, record, objects, text_parts=None):
    draft = native_draft(region, record); draft.pop('mapping'); draft['objects'] = objects
    if text_parts is not None:
        draft['text_parts'] = text_parts
    return draft


def test_mapping_is_derived_from_part_tags():
    inv, defaults, draft = fixture(); layout = layout_from_draft(draft, inv); region = layout['regions'][1]
    record = hybrid_record(layout, region, ['labels'])
    value = tagged_draft(region, record, [
        {'id': 'r02-bg', 'kind': 'native_shape', 'part': 'labels', 'bbox': [0, 250, 1000, 800], 'z': 0, 'geometry': 'rect', 'fill': 'FFFFFF'},
        {'id': 'r02-icon', 'kind': 'raster_asset', 'part': 'module_body', 'source_crop_bbox': [800, 300, 900, 450],
         'dest_bbox': [800, 300, 900, 450], 'z': 1, 'contains_text_ids': []}])
    result = expand_region(value, defaults, inv, region, record, layout)
    assert result['mapping'] == {'native': {'labels': ['r02-bg', 't2']}, 'retained': {'module_body': ['r02-icon']}}
    assert 'scope_adjustment' not in result and all('part' not in o for o in result['objects'])
    view = tagged_view(result)
    assert 'mapping' not in view and view['objects'][1]['part'] == 'module_body' and view['text_parts'] == {'t2': 'labels'}


def test_raster_for_a_native_part_and_unrealized_parts_adjust_scope_with_a_record():
    inv, defaults, draft = fixture(); layout = layout_from_draft(draft, inv); region = layout['regions'][1]
    record = hybrid_record(layout, region, ['labels', 'panel', 'arrow'])
    value = tagged_draft(region, record, [
        {'id': 'r02-panel', 'kind': 'raster_asset', 'part': 'panel', 'source_crop_bbox': [800, 300, 900, 450],
         'dest_bbox': [800, 300, 900, 450], 'z': 1, 'contains_text_ids': []}])
    result = expand_region(value, defaults, inv, region, record, layout)
    scope = result['scope_adjustment']
    assert scope['native_parts'] == ['labels'] and scope['retained_parts'] == ['panel']
    assert {o['op'] for o in scope['operations']} == {'native_part_realized_as_raster', 'unrealized_part_dropped'}
    tampered = deepcopy(result); tampered['scope_adjustment']['operations'] = [scope['operations'][0]]
    with pytest.raises(ValueError, match='not fully recorded'):
        validate_region(tampered, inv, region, record, layout)


def test_native_shape_cannot_stand_in_for_retained_pixels():
    inv, defaults, draft = fixture(); layout = layout_from_draft(draft, inv); region = layout['regions'][1]
    record = hybrid_record(layout, region, ['labels'])
    value = tagged_draft(region, record, [
        {'id': 'r02-dot', 'kind': 'native_shape', 'part': 'module_body', 'bbox': [800, 300, 900, 450], 'z': 1,
         'geometry': 'ellipse', 'fill': '0000FF'}])
    with pytest.raises(ValueError, match='actual source pixels'):
        expand_region(value, defaults, inv, region, record, layout)


def test_unknown_text_part_ids_are_rejected():
    inv, defaults, draft = fixture(); layout = layout_from_draft(draft, inv); region = layout['regions'][0]
    record = bound(layout, region)
    with pytest.raises(ValueError, match='text_parts keys'):
        expand_region(tagged_draft(region, record, [], {'t9': 'labels'}), defaults, inv, region, record, layout)


# ------------------------------------------------------------------ colors

def test_color_notations_are_normalized_and_bad_fields_are_named():
    draft = {'objects': [{'id': 'a', 'kind': 'native_shape', 'fill': '#ff0000', 'line': {'color': 'none', 'width_pt': 1}},
                         {'id': 'b', 'kind': 'native_shape', 'fill': 'rgb(0, 128, 255)', 'line': {'color': '#abc', 'width_pt': 1}}],
             'text_overrides': [{'id': 't1', 'color': 'transparent'}]}
    value, operations, problems = normalize_draft_colors(draft)
    assert value['objects'][0]['fill'] == 'FF0000' and 'line' not in value['objects'][0]
    assert value['objects'][1]['fill'] == '0080FF' and value['objects'][1]['line']['color'] == 'AABBCC'
    assert problems == [{'field': 't1.color', 'value': 'transparent', 'allowed': 'six hex RRGGBB'}]
    assert draft['objects'][0]['fill'] == '#ff0000' and len(operations) == 4


def test_region_draft_with_unusable_text_color_names_the_field():
    inv, defaults, draft = fixture(); layout = layout_from_draft(draft, inv); region = layout['regions'][0]
    record = bound(layout, region); value = native_draft(region, record)
    value['text_overrides'] = [{'id': 't1', 'color': 'none'}]
    with pytest.raises(ValueError, match=r'invalid color at t1\.color'):
        expand_region(value, defaults, inv, region, record, layout)


def test_raster_edge_overlapping_native_text_is_trimmed_but_enclosed_icon_is_not():
    from image2ppt_vision.auto_repair import trim_rasters_off_native_text
    icon = {'id': 'r-icon', 'kind': 'raster_asset', 'source_crop_bbox': [0, 0, 40, 40], 'dest_bbox': [0, 0, 40, 40],
            'contains_text_ids': []}
    operations = trim_rasters_off_native_text([icon], {'t': [26, 5, 100, 30]}, {})
    assert icon['source_crop_bbox'] == [0, 0, 26, 40] == icon['dest_bbox'] and operations[0]['op'] == 'trim_raster_off_native_text'
    swallowed = {'id': 'r-icon2', 'kind': 'raster_asset', 'source_crop_bbox': [10, 0, 40, 40], 'dest_bbox': [10, 0, 40, 40],
                 'contains_text_ids': []}
    assert trim_rasters_off_native_text([swallowed], {'t': [5, 2, 200, 36]}, {}) == []
    assert swallowed['source_crop_bbox'] == [10, 0, 40, 40]


def test_raster_grows_to_the_full_ocr_box_of_text_it_declares():
    from image2ppt_vision.auto_repair import grow_rasters_to_declared_text
    icon = {'id': 'r-icon', 'kind': 'raster_asset', 'source_crop_bbox': [10, 10, 40, 30], 'dest_bbox': [10, 10, 40, 30],
            'contains_text_ids': ['t']}
    operations = grow_rasters_to_declared_text([icon], {'t': [20, 20, 20, 24]}, {}, [0, 0, 100, 100])
    assert icon['source_crop_bbox'] == [10, 10, 40, 34] == icon['dest_bbox'] and len(operations) == 1
    blocked = {'id': 'r-b', 'kind': 'raster_asset', 'source_crop_bbox': [10, 10, 40, 30], 'dest_bbox': [10, 10, 40, 30],
               'contains_text_ids': ['t']}
    assert grow_rasters_to_declared_text([blocked], {'t': [20, 20, 20, 24]}, {'n': [10, 40, 40, 10]}, [0, 0, 100, 100]) == []


def test_echoed_response_format_marker_is_dropped_and_schema_filled():
    from image2ppt_vision.auto_repair import normalize_draft_envelope
    value, operations = normalize_draft_envelope({'type': 'json_object', 'coordinate_space': 'x'}, 'S/1')
    assert value == {'coordinate_space': 'x', 'schema': 'S/1'} and [o['op'] for o in operations] == [
        'drop_response_format_marker', 'fill_schema_constant']
    assert normalize_draft_envelope({'type': 'other', 'schema': 'S/1'}, 'S/1')[0]['type'] == 'other'


def test_review_bookkeeping_is_repaired_without_softening_findings():
    from image2ppt_vision.auto_repair import normalize_visual_review
    raw = {'type': 'json_object', 'quality_passed': True, 'issues': [
        {'id': 'a', 'kind': 'missing', 'severity': 'minor', 'object_ids': ['17', 'r01-icon'], 'bbox': [0, 0, 1, 1],
         'description': 'icon absent', 'object_ids_note': ''}]}
    value, operations = normalize_visual_review(raw, {'r01-icon'})
    issue = value['issues'][0]
    assert 'type' not in value and value['quality_passed'] is False
    assert issue['object_ids'] == ['r01-icon'] and issue['severity'] == 'blocking' and 'object_ids_note' not in issue
    assert {o['op'] for o in operations} == {'drop_response_format_marker', 'drop_issue_fields', 'drop_unknown_issue_targets',
                                             'content_issue_is_blocking', 'blocking_finding_fails_review'}
    assert raw['issues'][0]['severity'] == 'minor'


def test_unfitted_text_size_uses_the_page_measured_ratio(monkeypatch, tmp_path):
    import time
    from PIL import Image
    from image2ppt_vision import auto_typography
    Image.new('RGB', (400, 300), 'white').save(tmp_path/'s.png')
    inventory = {'source_sha256': 'a'*64, 'image_size': [400, 300],
                 'texts': [{'id': f't{i}', 'text': 'ab', 'bbox': [10, 10+i*40, 80, 20]} for i in range(6)]}
    def estimate(pixels, original, isolate_ocr_window):
        if original['id'] == 't5':
            return {'native_eligible': False, 'diagnostics': {}}
        x, y, w, h = original['bbox']
        return {'native_eligible': True, 'font_family': 'F', 'font_size_px': h*.7, 'bbox': [x, y, w, h], 'bold': False,
                'color': '#000000', 'diagnostics': {'font': {'dimension_error': 0, 'shape_overlap': 1, 'render_origin': [x, y]}}}
    monkeypatch.setattr(auto_typography, 'estimate_text', estimate)
    monkeypatch.setattr(auto_typography, 'font_status', lambda: {'families': ['F']})
    monkeypatch.setattr(auto_typography, 'origin_correction', lambda *a, **k: {'dy': 0})
    result = auto_typography.measure_text_defaults(tmp_path/'s.png', inventory, renderer_version='x', deadline=time.time()+60)
    fallback = next(t for t in result['text_decisions'] if t['id'] == 't5')
    assert fallback['font_size_px'] == 14.0 and 'ratio 0.700' in fallback['font_evidence']


def test_patch_is_confined_to_reviewed_objects():
    from image2ppt_vision.auto_region_protocol import confine_patch
    previous = {'objects': [{'id': 'a', 'z': 1}, {'id': 'b', 'z': 2}], 'text_decisions': [{'id': 't', 'bold': False}],
                'text_audit': [{'id': 't', 'status': 'calibrated'}]}
    current = {'objects': [{'id': 'a', 'z': 5}, {'id': 'b', 'z': 9}], 'text_decisions': [{'id': 't', 'bold': True}],
               'text_audit': [{'id': 't', 'status': 'unresolved'}]}
    value = confine_patch(previous, current, {'issues': [{'object_ids': ['a']}]})
    assert value['objects'] == [{'id': 'a', 'z': 5}, {'id': 'b', 'z': 2}]
    assert value['text_decisions'] == previous['text_decisions'] and value['text_audit'] == previous['text_audit']
    assert [o['op'] for o in value['program_repair']['operations']] == [
        'revert_unreviewed_change', 'revert_unreviewed_change', 'revert_unreviewed_audit']
    assert current['objects'][1]['z'] == 9


def test_unit_gradient_offsets_are_scaled_and_bad_gradients_are_named():
    from image2ppt_vision.auto_repair import normalize_draft_structure, gradient_problem
    draft = {'objects': [{'id': 'g', 'kind': 'native_shape', 'geometry': 'rect', 'fill_gradient': {
        'angle_deg': 0, 'stops': [{'offset': 0, 'color': 'AAAAAA'}, {'offset': 1, 'color': 'BBBBBB'}]}}]}
    value, operations = normalize_draft_structure(draft)
    gradient = value['objects'][0]['fill_gradient']
    assert [s['offset'] for s in gradient['stops']] == [0, 100000] and operations[0]['factor'] == 100000
    assert gradient_problem(gradient) is None
    assert 'strictly increase' in gradient_problem({'angle_deg': 0, 'stops': [{'offset': 0, 'color': 'AAAAAA'},
                                                                              {'offset': 50, 'color': 'BBBBBB'}]})


def test_overrides_keyed_by_text_id_become_a_list():
    from image2ppt_vision.auto_repair import normalize_draft_structure
    value, operations = normalize_draft_structure({'objects': [], 'text_overrides': {'t1': {'bold': True}}})
    assert value['text_overrides'] == [{'id': 't1', 'bold': True}]
    assert operations == [{'op': 'overrides_keyed_by_id_to_list', 'texts': ['t1']}]


def test_audit_status_follows_crop_containment():
    from image2ppt_vision.auto_repair import reconcile_audit
    texts = {'a': {'action': 'retained_in_asset'}, 'b': {'action': 'native'}}
    audit = [{'id': 'a', 'status': 'confirmed', 'reason': 'xxxx', 'style_group': ''},
             {'id': 'b', 'status': 'retained', 'reason': 'xxxx', 'style_group': ''}]
    result, operations = reconcile_audit(audit, texts)
    assert [r['status'] for r in result] == ['retained', 'unresolved'] and len(operations) == 2


def test_overshooting_native_text_box_is_clipped_to_its_region():
    inv, defaults, draft = fixture(); layout = layout_from_draft(draft, inv); region = layout['regions'][0]
    record = bound(layout, region); value = native_draft(region, record)
    value['text_overrides'] = [{'id': 't1', 'bbox': [33.3, 50, 533.3, 270]}]
    result = expand_region(value, defaults, inv, region, record, layout)
    assert result['text_decisions'][0]['bbox'][1]+result['text_decisions'][0]['bbox'][3] <= 50+.01
    assert {'op': 'clip_text_box_to_region', 'text': 't1'} in result['program_repair']['operations']


# --------------------------------------------------------------- transport

class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses); self.phases = []; self.evidence = []
    def call(self, phase, instruction, evidence, images, max_tokens):
        self.phases.append(phase); self.evidence.append(deepcopy(evidence))
        return self.responses.pop(0)
    def remaining(self):
        return 10_000


def ok(content='{}'):
    return {'ok': True, 'content': content}


def test_throttling_backs_off_without_spending_validation_attempts():
    session = FakeSession([{'ok': False, 'error': 'model_http_429'}, {'ok': False, 'error': 'model_http_503'}, ok('{"a":1}')])
    sleeps = []
    value = checked_model_json(session, 'p', 'i', {}, [], lambda v: v, sleep=sleeps.append)
    assert value == {'a': 1} and session.phases == ['p-0', 'p-0-retry1', 'p-0-retry2'] and sleeps == [15, 45]


def test_same_rejection_twice_stops_early():
    session = FakeSession([ok(), ok(), ok()])
    def reject(_): raise ValueError('invalid color')
    with pytest.raises(ValueError, match='exhausted: invalid color'):
        checked_model_json(session, 'p', 'i', {}, [], reject, sleep=lambda _: None)
    assert session.phases == ['p-0', 'p-1']


def test_type_and_key_errors_become_actionable_feedback():
    session = FakeSession([ok(), ok(), ok('{"b":2}')])
    calls = []
    def validator(v):
        calls.append(v)
        if len(calls) == 1: raise TypeError("unhashable type: 'dict'")
        if len(calls) == 2: raise KeyError('z')
        return v
    assert checked_model_json(session, 'p', 'i', {}, [], validator, sleep=lambda _: None) == {'b': 2}
    assert 'wrong JSON value type' in session.evidence[1]['validation_feedback']['error']
    assert session.evidence[2]['validation_feedback']['error'] == "missing required field 'z'"


def test_bad_request_is_not_retried():
    session = FakeSession([{'ok': False, 'error': 'model_http_400'}])
    with pytest.raises(ValueError, match='model_http_400'):
        checked_model_json(session, 'p', 'i', {}, [], lambda v: v, sleep=lambda _: None)


# ------------------------------------------------------------ key-file worker

def test_key_file_mode_streams_directly_and_redacts_the_secret(monkeypatch, tmp_path):
    lines = [b'data: '+json.dumps({'choices': [{'delta': {'content': '{"k":"sk-fixture"}'}, 'finish_reason': 'stop'}],
                                    'usage': {'total_tokens': 5}}).encode()+b'\n', b'data: [DONE]\n']
    opened = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __iter__(self): return iter(lines)
    class Opener:
        def open(self, request, timeout): opened.append(request); return Response()
    monkeypatch.setattr(urllib.request, 'build_opener', lambda *_: Opener())
    monkeypatch.setattr(worker, 'read_key', lambda config: 'sk-fixture')
    config = {'credential_mode': 'key_file', 'api_key_file': str(tmp_path/'key'), 'base_url': 'https://fixture.invalid',
              'allowed_models': ['deepseek-flash'], 'body_adapter': {'drop': ['enable_thinking'], 'set': {'x': 1}}}
    body = {'model': 'deepseek-flash', 'messages': [], 'temperature': .1, 'enable_thinking': False}
    result = worker.model_call(config, {'body': body, 'timeout': 30})
    assert result['ok'] and result['content'] == '{"k":"[REDACTED]"}' and result['usage']['total_tokens'] == 5
    sent = json.loads(opened[0].data)
    assert opened[0].full_url == 'https://fixture.invalid/chat/completions'
    assert opened[0].get_header('Authorization') == 'Bearer sk-fixture'
    assert 'enable_thinking' not in sent and sent['x'] == 1 and sent['stream'] is True


def test_provider_rejection_keeps_a_short_redacted_reason(monkeypatch):
    import io, urllib.error
    message = b'{"error":{"message":"bad request for sk-fixture"}}'
    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 400, 'Bad Request', {}, io.BytesIO(message))
    monkeypatch.setattr(urllib.request, 'build_opener', lambda *_: Opener())
    result = worker.stream_chat('https://fixture.invalid', 'sk-fixture', {'model': 'm', 'messages': []}, 30)
    assert result['error'] == 'model_http_400' and 'sk-fixture' not in result['provider_message']
    assert '[REDACTED]' in result['provider_message']


@pytest.mark.skipif(os.name == 'nt', reason='POSIX permission bits')
def test_key_file_must_be_private(tmp_path):
    key = tmp_path/'key'; key.write_text('sk-fixture\n'); key.chmod(0o644)
    with pytest.raises(PermissionError):
        worker.read_key({'api_key_file': str(key)})
    key.chmod(0o600)
    assert worker.read_key({'api_key_file': str(key)}) == 'sk-fixture'
