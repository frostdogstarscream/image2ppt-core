"""Synthetic contract tests; these do not establish real-image fidelity."""
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from image2ppt.storage import read_json, write_json, digest
from image2ppt_vision.auto_artifacts import file_hashes, verify_hashes
from image2ppt_vision.auto_provider import fingerprint, save_new, parse_object, model_body
from image2ppt_vision.auto_protocol import validate_plan, validate_review, scoped_patch, transform_model_boxes, expand_model_draft
from image2ppt_vision.auto_workflow import (PageSession, BudgetExhausted, create_job, run_job,
    checked_model_json, POLICY)
from image2ppt_vision.element_service import validate_payload
from image2ppt_vision.auto_api import create_app
from image2ppt_vision.auto_typography import expand_draft, compact_evidence_texts, draft_feedback


def plan_for(sha='a'*64, size=None):
    return {'schema': 'ElementPlan/3', 'source_sha256': sha, 'image_size': size or [100, 100],
            'background_color': 'FFFFFF',
            'objects': [{'id': 'panel', 'kind': 'native_shape', 'bbox': [5, 5, 90, 90],
                         'z': 0, 'geometry': 'rect', 'fill': 'EEEEEE'}],
            'text_decisions': [{'id': 't1', 'action': 'native', 'text': '12', 'bbox': [20, 20, 25, 20],
                'font_size_px': 14, 'bold': False, 'color': '000000', 'z': 1, 'align': 'left',
                'font_family': 'Arial', 'font_latin': 'Arial', 'font_confidence': .8,
                'font_evidence': 'synthetic sans serif fixture'}], 'limitations': []}


def inventory_for(plan):
    return {'source_sha256': plan['source_sha256'], 'image_size': plan['image_size'], 'candidates': [],
            'texts': [{k: t[k] for k in ('id', 'text', 'bbox')} for t in plan['text_decisions']]}


def review_for(binding='b'*64, issues=None):
    issues = issues or []
    return {'schema': 'AutomaticReview/1', 'artifact_binding': binding,
            'quality_passed': not issues, 'edit_probe_clear': True, 'issues': issues,
            'summary': 'synthetic review fixture'}


def issue(kind='style', ids=None, bbox=None):
    return {'id': 'issue1', 'kind': kind, 'severity': 'blocking', 'object_ids': ids or ['t1'],
            'bbox': bbox or [10, 10, 60, 40], 'description': 'synthetic issue'}


def test_element3_http_admission_and_legacy_coordinates():
    plan = plan_for()
    plan['objects'].append({'id': 'icon', 'kind': 'raster_asset', 'source_crop_bbox': [70, 70, 10, 10],
                            'dest_bbox': [75, 75, 10, 10], 'z': 1, 'contains_text_ids': []})
    validate_payload(plan); validate_plan(plan, inventory_for(plan))
    old = deepcopy(plan); old['schema'] = 'ElementPlan/1'
    asset = old['objects'][1]; asset['bbox'] = asset.pop('source_crop_bbox'); asset.pop('dest_bbox')
    validate_payload(old)
    with pytest.raises(ValueError):
        old['schema'] = 'ElementPlan/3'; validate_payload(old)


@pytest.mark.parametrize('mutation', [
    lambda p: p['objects'][0].update(source_crop_bbox=[1, 1, 2, 2]),
    lambda p: p['objects'][0].pop('bbox'),
    lambda p: p['text_decisions'][0].update(font_confidence=float('nan')),
    lambda p: p.update(source_sha256='f'*64),
    lambda p: p['text_decisions'].append(deepcopy(p['text_decisions'][0])),
])
def test_invalid_automatic_plan(mutation):
    plan = plan_for(); inv = inventory_for(plan); mutation(plan)
    with pytest.raises(ValueError): validate_plan(plan, inv)


@pytest.mark.parametrize('content', ['{"x":NaN}', '{"x":1,"x":2}', '[]', '{not json'])
def test_strict_model_json(content):
    with pytest.raises(ValueError): parse_object(content)


def test_validation_feedback_is_an_explicit_followup_not_buried_in_source_data():
    evidence = {'source_sha256': 'a'*64, 'validation_feedback': {'error': 'invalid crop', 'previous_output': '{"draft":1}'}}
    before = deepcopy(evidence)
    body = model_body('fixture', 'system rules', evidence, [])
    assert [m['role'] for m in body['messages']] == ['system','user','assistant','user']
    assert 'validation_feedback' not in body['messages'][1]['content'][0]['text']
    assert body['messages'][2]['content'] == '{"draft":1}'
    assert 'invalid crop' in body['messages'][-1]['content'] and evidence == before


def test_compact_draft_preserves_defaults_and_checks_binding():
    plan = plan_for(); inventory = inventory_for(plan)
    defaults = {'source_sha256': plan['source_sha256'], 'text_decisions': plan['text_decisions']}
    draft = {k: deepcopy(v) for k, v in plan.items() if k != 'text_decisions'}
    draft.update(schema='ElementPlanDraft/1', text_overrides=[])
    assert expand_draft(draft, defaults, inventory) == plan
    draft['text_overrides'] = [{'id': 't1', 'bold': True}]
    result = expand_draft(draft, defaults, inventory)
    assert result['text_decisions'][0]['bold'] and not defaults['text_decisions'][0]['bold']
    draft['source_sha256'] = 'f'*64
    with pytest.raises(ValueError): expand_draft(draft, defaults, inventory)


@pytest.mark.parametrize('updates', [[{'id': 'unknown'}], [{'id': 't1'}, {'id': 't1'}],
                                    [{'id': 't1', 'bbox': [95, 95, 20, 20]}]])
def test_compact_draft_rejects_unknown_duplicate_or_invalid_override(updates):
    plan = plan_for(); inventory = inventory_for(plan)
    defaults = {'source_sha256': plan['source_sha256'], 'text_decisions': plan['text_decisions']}
    draft = {k: deepcopy(v) for k, v in plan.items() if k != 'text_decisions'}
    draft.update(schema='ElementPlanDraft/1', text_overrides=updates)
    with pytest.raises(ValueError): expand_draft(draft, defaults, inventory)


def test_explicit_normalized_transport_converts_without_guessing():
    plan = plan_for(size=[200, 100]); inventory = inventory_for(plan)
    defaults = {'source_sha256': plan['source_sha256'], 'text_decisions': plan['text_decisions']}
    evidence = compact_evidence_texts(defaults, plan['image_size'])
    assert evidence[0]['bbox'] == [100, 200, 225, 400]
    draft = {k: deepcopy(v) for k, v in plan.items() if k != 'text_decisions'}
    draft.update(schema='ElementPlanDraft/2', coordinate_space='normalized_1000_xyxy',
                 text_overrides=[{'id': 't1', 'bbox': evidence[0]['bbox']}])
    draft['objects'][0]['bbox'] = [25, 50, 475, 950]
    assert expand_draft(draft, defaults, inventory) == plan
    draft['coordinate_space'] = 'guessed'
    with pytest.raises(ValueError): expand_draft(draft, defaults, inventory)


@pytest.mark.parametrize('box', [[0,0,1001,900], [20,20,10,10], [0,0,float('nan'),50]])
def test_normalized_transport_rejects_invalid_boxes(box):
    plan = plan_for(); inventory = inventory_for(plan)
    defaults = {'source_sha256': plan['source_sha256'], 'text_decisions': plan['text_decisions']}
    draft = {k: deepcopy(v) for k, v in plan.items() if k != 'text_decisions'}
    draft.update(schema='ElementPlanDraft/2', coordinate_space='normalized_1000_xyxy', text_overrides=[])
    draft['objects'][0]['bbox'] = box
    with pytest.raises(ValueError, match='normalized box'): expand_draft(draft, defaults, inventory)


def test_constraint_feedback_identifies_crop_ownership_without_editing_draft():
    plan = plan_for(); inventory = inventory_for(plan)
    draft = {'schema': 'ElementPlanDraft/2', 'coordinate_space': 'normalized_1000_xyxy',
             'objects': [{'id': 'crop', 'kind': 'raster_asset', 'source_crop_bbox': [190,190,460,410],
                          'contains_text_ids': ['wrong']}]}
    before = deepcopy(draft)
    issues = draft_feedback(draft, inventory)
    assert issues[0]['fully_contained_text_ids'] == ['t1'] and not issues[0]['crossing_text_ids']
    assert draft == before
    draft['objects'][0]['source_crop_bbox'] = [190,190,300,410]
    issues = draft_feedback(draft, inventory)
    assert issues[0]['crossing_text_ids'] == ['t1'] and not issues[0]['fully_contained_text_ids']


def test_stale_review_and_unknown_objects_rejected():
    plan = plan_for(); review = review_for()
    with pytest.raises(ValueError): validate_review(review, 'c'*64, plan)
    review = review_for(issues=[issue(ids=['invented'])])
    with pytest.raises(ValueError): validate_review(review, 'b'*64, plan)


def test_normalized_review_and_patch_keep_binding_and_scope():
    plan = plan_for(size=[200,100]); inv = inventory_for(plan)
    review = review_for(issues=[issue()])
    draft = transform_model_boxes(review, plan['image_size'], to_model=True)
    draft.update(schema='AutomaticReviewDraft/1', coordinate_space='normalized_1000_xyxy')
    canonical = expand_model_draft(draft, 'AutomaticReviewDraft/1', 'AutomaticReview/1', plan['image_size'])
    assert validate_review(canonical, 'b'*64, plan) == review
    patch = {'schema':'ElementPatchDraft/1','coordinate_space':'normalized_1000_xyxy',
             'source_sha256': plan['source_sha256'], 'text_decisions':[{'id':'t1','bbox':[105,200,230,400]}]}
    normalized = expand_model_draft(patch, 'ElementPatchDraft/1', 'ElementPatch/1', plan['image_size'])
    assert scoped_patch(plan, normalized, review, inv, 'b'*64)['text_decisions'][0]['bbox'] == [21,20,25,20]
    patch['text_decisions'][0]['bbox'] = [105,200,1001,400]
    with pytest.raises(ValueError): expand_model_draft(patch, 'ElementPatchDraft/1', 'ElementPatch/1', plan['image_size'])
    review = review_for(issues=[issue('numeric')]); review['issues'][0]['severity'] = 'minor'
    with pytest.raises(ValueError): validate_review(review, 'b'*64, plan)


def test_scoped_patch_preserves_unmentioned_objects():
    plan = plan_for(); inv = inventory_for(plan); review = review_for(issues=[issue()])
    patch = {'schema': 'ElementPatch/1', 'source_sha256': plan['source_sha256'],
             'text_decisions': [{'id': 't1', 'bold': True, 'bbox': [21, 20, 25, 20]}]}
    corrected = scoped_patch(plan, patch, review, inv, 'b'*64)
    assert corrected['objects'] == plan['objects'] and corrected['text_decisions'][0]['bold']
    patch['objects'] = [{'id': 'panel', 'fill': '000000'}]
    with pytest.raises(ValueError, match='outside reviewed'): scoped_patch(plan, patch, review, inv, 'b'*64)


def test_patch_rejects_unreviewed_content_and_destination():
    plan = plan_for(); inv = inventory_for(plan); review = review_for(issues=[issue()])
    patch = {'schema': 'ElementPatch/1', 'source_sha256': plan['source_sha256'], 'text_decisions': [{'id': 't1', 'text': '99'}]}
    with pytest.raises(ValueError, match='content issue'): scoped_patch(plan, patch, review, inv, 'b'*64)
    patch['text_decisions'] = [{'id': 't1', 'bbox': [80, 80, 10, 10]}]
    with pytest.raises(ValueError, match='correction region'): scoped_patch(plan, patch, review, inv, 'b'*64)


def test_new_object_requires_reviewed_missing_region():
    plan = plan_for(); inv = inventory_for(plan)
    finding = issue('missing'); finding['object_ids'] = []; finding['bbox'] = [60, 60, 30, 30]
    review = review_for(issues=[finding])
    patch = {'schema': 'ElementPatch/1', 'source_sha256': plan['source_sha256'],
             'objects': [{'id': 'new', 'kind': 'native_shape', 'bbox': [65, 65, 10, 10], 'geometry': 'rect', 'fill': '222222', 'z': 1}]}
    assert len(scoped_patch(plan, patch, review, inv, 'b'*64)['objects']) == 2
    patch['objects'][0]['bbox'] = [5, 5, 10, 10]
    with pytest.raises(ValueError): scoped_patch(plan, patch, review, inv, 'b'*64)


class EchoRemote:
    def __init__(self): self.calls = []
    def request(self, request, timeout=420):
        self.calls.append(deepcopy(request))
        return {'ok': True, 'request_id': request['request_id'], 'content': '{"ok":true}', 'usage': {'total_tokens': 3}}


def test_budget_counts_reservations_and_reuses_exact_calls(tmp_path):
    remote = EchoRemote(); session = PageSession(tmp_path, remote, 'fixture')
    for i in range(8): session.call(str(i), 'test', {}, [], 10)
    assert len(remote.calls) == 8
    session.call('0', 'test', {}, [], 10)
    assert len(remote.calls) == 8
    with pytest.raises(BudgetExhausted): session.call('9', 'test', {}, [])
    assert session.summary()['usage']['total_tokens'] == 24


def test_call_response_tamper_is_not_reused(tmp_path):
    session = PageSession(tmp_path, EchoRemote(), 'fixture')
    session.call('x', 'test', {}, [])
    (tmp_path/'calls/c00/response.json').write_text('{}')
    with pytest.raises(ValueError, match='evidence changed'): session.call('x', 'test', {}, [])


def test_transport_recovery_uses_same_request_and_preserves_original_failure(tmp_path):
    class FlakyRemote(EchoRemote):
        def request(self, request, timeout=420):
            result = super().request(request, timeout)
            return {'ok': False, 'error': 'ssh_failed', 'request_id': request['request_id']} if len(self.calls) == 1 else result
    remote = FlakyRemote(); session = PageSession(tmp_path, remote, 'fixture')
    assert session.call('x', 'test', {}, [])['ok']
    assert len(remote.calls) == 2 and remote.calls[0] == remote.calls[1]
    assert read_json(tmp_path/'calls/c00/response.json')['error'] == 'ssh_failed'
    assert session.summary()['model_requests'] == 1
    assert session.call('x', 'test', {}, [])['ok'] and len(remote.calls) == 2


def test_page_timeout_stops_before_network(tmp_path):
    remote = EchoRemote(); clock = [10]
    session = PageSession(tmp_path, remote, 'fixture', clock=lambda: clock[0])
    clock[0] += 1801
    with pytest.raises(BudgetExhausted): session.call('x', 'test', {}, [])
    assert not remote.calls


def test_invalid_json_retry_has_bounded_budget(tmp_path):
    class BadRemote(EchoRemote):
        def request(self, request, timeout=420):
            result = super().request(request, timeout); result['content'] = '{'; return result
    remote = BadRemote(); session = PageSession(tmp_path, remote, 'fixture')
    with pytest.raises(ValueError, match='validation_exhausted'):
        checked_model_json(session, 'plan', 'test', {}, [], lambda x: x)
    assert len(remote.calls) == 3


@pytest.fixture
def automatic_case(tmp_path, monkeypatch):
    monkeypatch.setattr('image2ppt_vision.auto_workflow.get_text_defaults',
        lambda session, source, inventory, project_root: {'source_sha256': inventory['source_sha256'],
            'text_decisions': plan_for(inventory['source_sha256'])['text_decisions']})
    source = tmp_path/'input.png'; image = Image.new('RGB', (100, 100), 'white')
    ImageDraw.Draw(image).text((20, 20), '12', fill='black'); image.save(source)
    config = {'ssh_host': 'fixture', 'remote_worker': '/worker.py', 'remote_config': '/config.json', 'models': ['fixture-model']}
    proof = tmp_path/'preflight'; proof.mkdir(); (proof/'probe.txt').write_text('synthetic')
    save_new(proof/'result.json', {'schema': 'AutomaticPreflight/1', 'config_sha256': fingerprint(config),
                                  'passed': True, 'selected_model': 'fixture-model', 'files': file_hashes(proof)})
    return source, config, proof


class PipelineRemote:
    def __init__(self, review_pass=True): self.calls = []; self.review_pass = review_pass
    def request(self, request, timeout=420):
        self.calls.append(deepcopy(request))
        if request['operation'] == 'analyze':
            analysis = {'image_size': [100, 100], 'inputs': {'fixture': {'original_sha256': request['source_sha256']}},
                        'stages': [{'name': 'ocr', 'status': 'PASS'}],
                        'objects': [{'id': 't1', 'kind': 'text', 'text': '12', 'bbox': [20, 20, 25, 20]}]}
            return {'ok': True, 'analysis': analysis, 'job_id': 'f'*32}
        evidence = json.loads(request['body']['messages'][1]['content'][0]['text'])
        if 'ocr' in evidence:
            value = plan_for(evidence['source_sha256'], evidence['image_size'])
        elif 'artifact_binding' in evidence:
            value = review_for(evidence['artifact_binding'], [] if self.review_pass else [issue()])
        else:
            value = {'schema': 'ElementPatch/1', 'source_sha256': evidence['plan']['source_sha256'],
                     'text_decisions': [{'id': 't1', 'bold': True}]}
        return {'ok': True, 'content': json.dumps(value), 'request_id': request['request_id'], 'usage': {'total_tokens': 10}}


def fake_builder(source, inventory, plan, directory, **kwargs):
    # Files intentionally are not valid PPTX; this tests only orchestration.
    directory = Path(directory); packet = directory/'evidence'; packet.mkdir(parents=True)
    for name in ('result.pptx', 'preview.png', 'comparison.png', 'difference.png', 'delete-probe.png', 'delete-probe.pptx', 'source.png'):
        (packet/name).write_bytes(b'synthetic-not-a-real-deck')
    result = {'artifact_binding': fingerprint({'plan': plan, 'directory': str(directory)}),
              'plan_sha256': fingerprint(plan), 'files': file_hashes(directory),
              'probe_targets': ['t1'], 'structural_issues': [],
              'metrics': {'mean_absolute_pixel_difference': 1},
              'counts': {'native_text': 1, 'native_shapes': 1, 'raster_assets': 0}}
    save_new(directory/'result.json', result)
    return result


def test_complete_job_and_resume_do_not_repeat_model_calls(automatic_case, tmp_path, monkeypatch):
    # Model images are fake in builder; image_content reads bytes without trusting them.
    source, config, proof = automatic_case; remote = PipelineRemote()
    job = create_job([source], tmp_path/'jobs', config)
    first = run_job(job, project_root=tmp_path, preflight_directory=proof, remote=remote, builder=fake_builder)
    count = len(remote.calls)
    assert first['status'] == 'COMPLETED' and first['machine_quality_status'] == 'PASSED'
    again = run_job(job, project_root=tmp_path, remote=remote, builder=fake_builder)
    assert len(remote.calls) == count and again['pages']['P1']['human_acceptance'] is False


def test_gate_halts_batch_after_two_correction_rounds(automatic_case, tmp_path):
    source, config, proof = automatic_case; remote = PipelineRemote(review_pass=False)
    job = create_job([source]*4, tmp_path/'jobs', config, gate_pages=[1])
    result = run_job(job, project_root=tmp_path, preflight_directory=proof, remote=remote, builder=fake_builder)
    assert result['status'] == 'GATE_REVIEW_REQUIRED' and result['skipped_pages'] == ['P2', 'P3', 'P4']
    assert result['pages']['P1']['revision_count'] == 3
    assert result['pages']['P1']['model_requests'] == 6
    assert not (job/'P2/analysis.json').exists()


def test_interrupted_build_resumes_with_cached_plan(automatic_case, tmp_path):
    source, config, proof = automatic_case; remote = PipelineRemote()
    job = create_job([source], tmp_path/'jobs', config)
    def interrupted(*args, **kwargs): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_job(job, project_root=tmp_path, preflight_directory=proof, remote=remote, builder=interrupted)
    before = len(remote.calls)
    result = run_job(job, project_root=tmp_path, remote=remote, builder=fake_builder)
    assert result['status'] == 'COMPLETED'
    assert len(remote.calls) == before+1  # review only; no repeated OCR or planning


def test_failed_build_retry_preserves_history_and_original_budget(automatic_case, tmp_path):
    source, config, proof = automatic_case; remote = PipelineRemote()
    job = create_job([source], tmp_path/'jobs', config)
    def failed(*args, **kwargs): raise RuntimeError('fixture build failure')
    first = run_job(job, project_root=tmp_path, preflight_directory=proof, remote=remote, builder=failed)
    assert first['pages']['P1']['execution_status'] == 'FAILED'
    budget = (job/'P1/budget.json').read_bytes(); before = len(remote.calls)
    result = run_job(job, project_root=tmp_path, remote=remote, builder=fake_builder, retry_failed=True)
    assert result['status'] == 'COMPLETED' and len(remote.calls) == before+1
    assert (job/'P1/budget.json').read_bytes() == budget
    assert len(list((job/'P1/previous-results').glob('*.json'))) == 1


def test_valid_file_without_validated_model_review_is_partial(automatic_case, tmp_path):
    source, config, proof = automatic_case
    class InvalidReviewer(PipelineRemote):
        def request(self, request, timeout=420):
            result = super().request(request, timeout)
            if request['operation'] == 'model':
                evidence = json.loads(request['body']['messages'][1]['content'][0]['text'])
                if 'artifact_binding' in evidence:
                    result['content'] = '{invalid review'
            return result
    job = create_job([source], tmp_path/'jobs', config)
    result = run_job(job, project_root=tmp_path, preflight_directory=proof, remote=InvalidReviewer(), builder=fake_builder)
    page = result['pages']['P1']
    assert result['status'] == 'PARTIAL' and page['execution_status'] == 'PARTIAL'
    assert not page['pipeline_completed'] and page['review_status'] == 'INCOMPLETE'
    assert page['machine_quality_status'] == 'REVIEW_REQUIRED' and page['build_directory']


def test_source_and_final_evidence_tamper_rejected(automatic_case, tmp_path):
    source, config, proof = automatic_case; job = create_job([source], tmp_path/'jobs', config)
    result = run_job(job, project_root=tmp_path, preflight_directory=proof, remote=PipelineRemote(), builder=fake_builder)
    packet = job/'P1'/result['pages']['P1']['build_directory']/'evidence/result.pptx'
    packet.write_bytes(b'changed')
    with pytest.raises(ValueError, match='integrity mismatch'):
        run_job(job, project_root=tmp_path, remote=PipelineRemote(), builder=fake_builder)


def test_loopback_api_requires_session_and_same_origin(automatic_case, tmp_path):
    source, config, _ = automatic_case; settings = tmp_path/'connection.json'; save_new(settings, config)
    app = create_app(tmp_path/'api', settings, tmp_path, runner=lambda *a, **k: None)
    with TestClient(app, base_url='http://127.0.0.1') as client:
        assert client.get('/health').status_code == 200
        assert client.get('/v1/auto-jobs').status_code == 403
        html = client.get('/').text
        import re
        token = re.search("const token='([^']+)'", html).group(1)
        headers = {'X-Local-Token': token}
        assert client.get('/v1/auto-jobs', headers=headers).status_code == 200
        assert client.get('/v1/auto-jobs', headers={**headers, 'Origin': 'https://unrelated.example'}).status_code == 403
        response = client.post('/v1/auto-jobs', headers=headers, files={'files': ('page.png', source.read_bytes(), 'image/png')})
        assert response.status_code == 202
        job = response.json()['job_id']
        assert client.get('/v1/auto-jobs/'+job, headers=headers).status_code == 200
        assert client.get('/v1/auto-jobs/'+job+'/artifacts/connection.json', headers=headers).status_code == 404
    with TestClient(app, base_url='http://evil.example') as client:
        assert client.get('/').status_code == 403
