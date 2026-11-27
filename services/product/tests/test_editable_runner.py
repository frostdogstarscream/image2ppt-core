import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.materialize import write
from product.runtime.editable_runner import POLICY_FILE, _attachments, _candidate_progress, run
from product.runtime.service import RuntimeService


class Actor:
    model = 'test-fixture'

    def __init__(self, choose):
        self.choose = choose
        self.calls = []
        self.prompts = []

    def ask_receipted(self, prompt, context, images, contract, receipt):
        self.calls.append((context, images))
        self.prompts.append(prompt)
        result = self.choose(context, len(self.calls))
        write(receipt, {'content': json.dumps(result), 'finish_reason':'stop'})
        return result


@pytest.fixture
def job(tmp_path, monkeypatch):
    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT', str(tmp_path/'locks'))
    path = tmp_path/uuid4().hex
    (path/'inputs').mkdir(parents=True)
    Image.new('RGB', (30, 20), 'white').save(path/'inputs/01.png')
    write(path/'request.json', {'orchestrator':'editable_reconstruction','visual_review_backend':'moma',
                                'agent_steps':4})
    return path, RuntimeService(tmp_path)


def test_native_worker_stops_after_uncertain_office_response(job,monkeypatch):
    from product.runtime import editable_runner
    path,service=job
    request=json.loads((path/'request.json').read_text('utf8'))
    request['native_variant']='native_synthesis'
    actor=Actor(lambda context,turn:{'tool':'validate_office','reason':'check the actual delivery'})
    monkeypatch.setattr(editable_runner,'_dispatch',lambda *args:{
        'status':'UNKNOWN','failure':'OFFICE_WORKER_TIMEOUT'})
    result=editable_runner._run(service.state(path.name),service,request,actor)
    assert result['status']=='REVIEW_REQUIRED' and 'OFFICE_WORKER_TIMEOUT' in result['reason']
    assert len(actor.calls)==1


def test_direct_tool_crop_and_view_journal_resume(job):
    path, service = job

    def choose(ctx, n):
        if n == 1:
            return {'tool':'crop_image', 'reason':'inspect title', 'arguments':{
                'evidence_id':ctx['pages'][0]['id'], 'bbox':[1,1,12,7]}}
        if n == 2:
            return {'tool':'get_evidence', 'reason':'observe crop', 'arguments':{
                'evidence_ids':[ctx['history'][-1]['outcome']['image']['id']]}}
        return {'tool':'stop', 'reason':'review incomplete'}

    actor = Actor(choose)
    result = run(path, service=service, actor=actor)
    assert result['status'] == 'REVIEW_REQUIRED'
    assert [len(images) for _,images in actor.calls] == [1,0,1]
    crop = actor.calls[2][0]['observations'][0]['id']
    assert service.state(path.name).evidence(crop)[1]['provided_to_agent']
    assert run(path, service=service, actor=actor) == result
    assert len(actor.calls) == 3


def test_unknown_response_is_not_retried_or_marked_delivered(job):
    path, service = job
    def timeout(ctx, n):
        raise TimeoutError('transport closed')
    actor = Actor(timeout)
    assert run(path, service=service, actor=actor)['final_qa'] == 'UNKNOWN'
    assert run(path, service=service, actor=actor)['reason'] == 'MODEL_RESPONSE_UNKNOWN'
    assert len(actor.calls)==1
    assert not service.state(path.name).evidence(actor.calls[0][0]['pages'][0]['id'])[1]['provided_to_agent']


def test_missing_image_and_scope_cannot_approve_result(job):
    path, service = job
    def choose(ctx, n):
        if n==1:
            return {'tool':'get_evidence','arguments':{'evidence_ids':['missing']},'reason':'inspect'}
        if n==2:
            return {'tool':'finalize_job','arguments':{'job_id':'foreign'},'reason':'check'}
        return {'tool':'finalize_job','reason':'check'}
    actor=Actor(choose)
    result=run(path, service=service, actor=actor)
    assert result['reason']=='MODEL_CALL_BUDGET_EXHAUSTED'
    assert 'unknown evidence' in actor.calls[1][0]['history'][-1]['outcome']['detail']
    assert 'host-owned' in actor.calls[2][0]['history'][-1]['outcome']['detail']
    assert not result['automatic_publication']


def test_direct_dispatch_uses_shared_cost_budget(job):
    path, service = job
    request = json.loads((path/'request.json').read_text('utf8'))
    request['adaptive_budget'] = {'cost_units': 0}
    write(path/'request.json', request)
    actor = Actor(lambda ctx,n: {'tool':'stop','reason':'stop'})
    assert run(path, service=service, actor=actor)['reason'] == 'MODEL_OR_COST_BUDGET_EXHAUSTED'
    assert not actor.calls


def test_validated_candidate_visible_after_five_read_turns(job):
    path, service = job
    state = service.state(path.name)
    current = state.read()
    current['candidates']['c-example'] = {'id': 'c-example', 'page': 1, 'kind': 'batch',
        'object_ids': ['icon', 'label'], 'status': 'VALIDATED',
        'validation': {'status': 'PASS', 'office': {'status': 'PASS_APP_AUTOMATED'}}}
    write(state.root/'state.json', current)
    request = json.loads((path/'request.json').read_text('utf8'))
    write(path/'request.json', {**request, 'agent_steps': 7})

    def choose(ctx, turn):
        if turn < 6:
            return {'tool': 'list_evidence', 'reason': 'inspect available evidence'}
        assert ctx['candidates'] == [{'page': 1, 'kind': 'batch', 'status': 'VALIDATED',
            'candidate_id': 'c-example', 'validation_status': 'PASS',
            'office_status': 'PASS_APP_AUTOMATED', 'object_count': 2}]
        return {'tool': 'stop', 'reason': 'test complete'}

    actor = Actor(choose)
    assert run(path, service=service, actor=actor)['reason'] == 'test complete'
    admission = json.loads((state.root/'editable-turns/0006/request.json').read_text('utf8'))
    policy = POLICY_FILE.read_bytes()
    assert policy.decode('utf-8-sig') in actor.prompts[-1]
    assert admission['policy_sha256'] == hashlib.sha256(policy).hexdigest()
    assert admission['system_prompt_sha256'] == hashlib.sha256(actor.prompts[-1].encode('utf8')).hexdigest()
    assert len(actor.calls[-1][0]['history']) == 4


def test_candidate_visibility_marks_superseded_parent(tmp_path):
    ir, scene = tmp_path/'ir', tmp_path/'scene'
    ir.write_bytes(b'ir')
    scene.write_bytes(b'scene')
    candidate = {'page': 1, 'status': 'VALIDATED', 'parent_ir_sha256': hashlib.sha256(b'ir').hexdigest(),
                 'parent_scene_sha256': hashlib.sha256(b'scene').hexdigest()}
    current = {'pages': [{'ir': ir, 'scene': scene}], 'candidates': {'c-old': candidate}}
    assert _candidate_progress(current)[0]['matches_current_page'] is True
    scene.write_bytes(b'new evidence')
    assert _candidate_progress(current)[0]['matches_current_page'] is False


def test_large_office_receipt_retains_failures_and_hash(job):
    path, service = job
    state = service.state(path.name)
    report = path/'office-checks.json'
    write(report, {'status': 'REVIEW_REQUIRED', 'pptx_sha256': 'bound-hash', 'files': {'x': 'x' * 251000},
                   'office': {'status': 'PASS_APP_AUTOMATED', 'records': [{'id': 'a-1', 'action': 'move',
                             'saved_reopened': True, 'outside_pixels': 0}],
                             'original_sites': [{'page': 1, 'action': 'move-clear', 'members': ['a-1'], 'outside_pixels': 0}],
                             'unverified_original_site_groups': [{'backing':'card', 'reason':'MULTIPLE_CHILDREN'}]},
                   'replacements': {'status': 'UNKNOWN', 'records': []},
                   'residuals': [[{'id': 'a-1', 'residual_check_status': 'UNKNOWN', 'reason': 'occluded'}]]})
    ref = state.register(report, 'hard_checks')
    documents, images, refs = _attachments(state, [ref['id']])
    assert not images and refs[0]['sha256'] == hashlib.sha256(report.read_bytes()).hexdigest()
    observation = documents[0]
    assert observation['representation'] == 'office_receipt_summary'
    assert observation['json']['office']['actions'] == {'move': 1}
    assert observation['json']['residuals_by_page'][0]['exceptions'][0]['reason'] == 'occluded'
    assert observation['json']['replacements']['status'] == 'UNKNOWN'
    assert observation['json']['unverified_original_site_groups'][0]['reason'] == 'MULTIPLE_CHILDREN'
    assert observation['json']['original_sites'][0]['members'] == ['a-1']
