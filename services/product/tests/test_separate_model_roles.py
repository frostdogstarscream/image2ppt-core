import json
from types import SimpleNamespace

import pytest

from product.materialize import write
from product.runtime import independent_review, review_control
from product.runtime.moma_review import MomaReviewer
from product.runtime.provider import configure_model, model_identity, planning_actor, GO_BASE_URL
from product.runtime.runner import prepare_workspace


@pytest.fixture
def dual(tmp_path, monkeypatch):
    for name in ('IMAGE2PPT_AGENT_MODEL', 'PLANNER_MODEL', 'MOMA_MODEL', 'VISION_LLM_MODEL',
                 'MOMA_BASE_URL', 'PLANNER_BASE_URL', 'OPENCODE_GO_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('IMAGE2PPT_AGENT_PROVIDER', 'moma')
    monkeypatch.setenv('IMAGE2PPT_AGENT_API_KEY', 'fake-executor')
    monkeypatch.setenv('MOMA_API_KEY', 'fake-reviewer')
    monkeypatch.setenv('PLANNER_API_KEY', 'fake-legacy')
    monkeypatch.setenv('IMAGE2PPT_REVIEW_PROVIDER', 'moma')
    monkeypatch.setenv('IMAGE2PPT_REVIEW_MODEL', 'qwen/qwen3.8-max')
    write(tmp_path/'request.json', {'agent_model':'deepseek-v4.1-flash','agent_inference_provider':'moma'})
    write(tmp_path/'runtime/session.json', {'id':'ses_synthetic_builder'})
    return tmp_path


def test_execution_and_independent_roles_have_distinct_credentials(dual, monkeypatch):
    monkeypatch.setenv('PLANNER_BASE_URL', GO_BASE_URL)
    actor = planning_actor(dual)
    audit = MomaReviewer('qwen/qwen3.8-max')
    assert actor.cfg.api_key == 'fake-executor' and audit.settings.api_key == 'fake-reviewer'
    assert actor.cfg.model == 'deepseek-v4.1-flash' and actor.cfg.base_url == audit.settings.base_url
    encoded = (prepare_workspace(dual)/'opencode.json').read_text('utf8')
    provider = json.loads(encoded)['providers']['image2ppt-moma']
    assert provider['env'] == ['IMAGE2PPT_AGENT_API_KEY']
    assert provider['models']['vision']['modelID'] == actor.cfg.model
    assert all(value not in encoded for value in ('fake-executor', 'fake-reviewer', 'fake-legacy'))
    monkeypatch.setenv('IMAGE2PPT_AGENT_API_KEY','fake-reviewer')
    with pytest.raises(RuntimeError, match='distinct credentials'):
        configure_model(dual)
    monkeypatch.delenv('IMAGE2PPT_AGENT_API_KEY')
    with pytest.raises(RuntimeError, match='separate MoMA Agent credential'):
        configure_model(dual)


def test_legacy_moma_and_go_settings_do_not_leak_between_jobs(dual, monkeypatch):
    monkeypatch.delenv('IMAGE2PPT_AGENT_API_KEY')
    monkeypatch.delenv('IMAGE2PPT_REVIEW_PROVIDER')
    monkeypatch.setenv('OPENCODE_GO_API_KEY', 'fake-go')
    monkeypatch.setenv('PLANNER_BASE_URL','https://example.invalid/legacy')
    write(dual/'request.json', {'agent_inference_provider':'opencode_go','agent_model':'hunyuan4-flash-free'})
    _, cfg = configure_model(dual)
    assert (cfg.api_key, cfg.base_url, cfg.model) == ('fake-go', GO_BASE_URL, 'hy4-flash-free')
    write(dual/'request.json', {'agent_inference_provider':'moma','agent_model':'qwen/qwen3.8-max'})
    _, cfg = configure_model(dual)
    assert (cfg.api_key,cfg.base_url) == ('fake-legacy','https://example.invalid/legacy')
    assert model_identity('opencode-go/hunyuan4-flash-free') == model_identity('hy4-flash-free')


def run_audit(job,monkeypatch,reported='qwen3.8-max', changes=None,control_model='qwen3.8-max'):
    from pathlib import Path
    assets = [job/n for n in ('source.png','render.png','owner.png','edits.png','deck.pptx')]
    for path in assets:
        if not path.exists():
            path.write_bytes(path.name.encode())
    if changes:
        target,data = changes
        Path(job/target).write_bytes(data)
    calls = []
    def fake_ask(actor,*args):
        calls.append((actor.model, actor.session))
        actor.reported_model=reported
        actor.last_usage={'total_tokens':8}
        return {'passed':True, 'defects':[], 'summary':'ok'}
    monkeypatch.setattr(MomaReviewer,'ask',fake_ask)
    monkeypatch.setattr(review_control,'verify_reviewer',lambda *args: {'passed':True, 'reported_model':control_model})
    source,render,owner,edits,deck=assets
    result = independent_review._stateless_vlm_audit(SimpleNamespace(job=job),source,render,owner,{},deck_path=deck,edit_panels=[edits])
    return result,calls


def test_moma_cross_model_review_is_bound_to_deck_edits_and_builder_session(dual,monkeypatch):
    first,calls=run_audit(dual,monkeypatch)
    assert first['passed'] and first['reported_model']=='qwen3.8-max'
    assert first['builder_provider']=='moma' and first['builder_model']=='deepseek-v4.1-flash'
    assert calls == [('qwen/qwen3.8-max',first['session'])]
    assert first['session'] != first['builder_session'] == 'ses_synthetic_builder'
    cached,calls = run_audit(dual,monkeypatch)
    assert cached['status']=='CACHED' and not calls
    for name in ('edits.png','deck.pptx'):
        second,calls = run_audit(dual,monkeypatch,changes=(name,b'changed'))
        assert second['passed'] and second['receipt']!=first['receipt'] and calls
        first=second


def test_reviewer_cannot_use_requested_or_reported_builder_alias(dual,monkeypatch):
    write(dual/'request.json',{'agent_inference_provider':'moma','agent_model':'Qwen3.8-Max'})
    result,calls=run_audit(dual,monkeypatch)
    assert not result['passed'] and not calls
    write(dual/'request.json',{'agent_inference_provider':'moma','agent_model':'deepseek/deepseek-v4.1-flash'})
    result,calls=run_audit(dual,monkeypatch,reported='deepseek-v4.1-flash')
    assert calls and result['status']=='MODEL_IDENTITY_UNVERIFIED' and not result['passed']


def test_control_and_session_cannot_use_execution_identity(dual,monkeypatch):
    result,calls=run_audit(dual,monkeypatch,control_model='deepseek-v4.1-flash')
    assert result['status']=='CONTROL_UNVERIFIED' and not calls
    monkeypatch.setattr(MomaReviewer, '__init__',
                        lambda self,model: self.__dict__.update(model=model,session='ses_synthetic_builder'))
    result,calls=run_audit(dual,monkeypatch)
    assert not result['passed'] and not calls


def test_moma_can_still_review_go(dual,monkeypatch):
    write(dual/'request.json',{'agent_inference_provider':'opencode_go','agent_model':'hy4-flash-free'})
    result,calls=run_audit(dual,monkeypatch)
    assert result['passed'] and result['provider']=='moma' and calls


@pytest.mark.parametrize('thinking', [False, True])
def test_official_deepseek_workspace_and_direct_tool_use_same_wire_contract(dual, monkeypatch, thinking):
    import asyncio
    import httpx
    from image2ppt_planner.client import MomaClient

    monkeypatch.setenv('IMAGE2PPT_AGENT_PROVIDER', 'deepseek')
    monkeypatch.setenv('PLANNER_BASE_URL', 'https://wrong.invalid/v1')
    monkeypatch.setenv('MOMA_BASE_URL', 'https://wrong-review.invalid/v1')
    write(dual/'request.json', {'agent_inference_provider':'deepseek', 'agent_thinking':thinking})
    selected, cfg = configure_model(dual)
    assert (selected, cfg.base_url, cfg.model) == ('deepseek', 'https://api.deepseek.com/v1', 'deepseek-flash')
    assert planning_actor(dual).cfg == cfg
    workspace = (prepare_workspace(dual)/'opencode.json').read_text('utf8')
    assert not any(value in workspace for value in ('fake-executor','fake-reviewer','fake-legacy','wrong.invalid'))
    entry, = json.loads(workspace)['providers'].items()
    name, provider = entry
    assert name == 'image2ppt-deepseek' and provider['env'] == ['IMAGE2PPT_AGENT_API_KEY']
    assert provider['settings']['baseURL'] == cfg.base_url
    assert provider['models']['vision']['modelID'] == cfg.model
    assert provider['body'] == {'thinking':{'type':'enabled' if thinking else 'disabled'}}

    def capture(request):
        assert str(request.url) == 'https://api.deepseek.com/v1/chat/completions'
        assert request.headers['Authorization'] == 'Bearer fake-executor'
        payload = json.loads(request.content)
        assert payload['model'] == cfg.model and payload['thinking'] == provider['body']['thinking']
        assert 'enable_thinking' not in payload
        return httpx.Response(200,json={'model':cfg.model,'choices':[{'message':{'content':'{}'},'finish_reason':'stop'}]})
    reply = asyncio.run(MomaClient(cfg, httpx.MockTransport(capture)).chat([{'role':'user','content':'Return JSON.'}]))
    assert reply.model == cfg.model


def test_official_execution_key_cannot_be_used_by_another_provider(dual, monkeypatch):
    monkeypatch.setenv('IMAGE2PPT_AGENT_PROVIDER', 'deepseek')
    for provider in ('moma','opencode_go'):
        write(dual/'request.json', {'agent_inference_provider':provider})
        with pytest.raises(RuntimeError, match='bound'):
            prepare_workspace(dual)
    write(dual/'request.json', {'agent_inference_provider':'deepseek'})
    monkeypatch.setenv('IMAGE2PPT_AGENT_PROVIDER','moma')
    with pytest.raises(RuntimeError, match='bound'):
        configure_model(dual)
    monkeypatch.setenv('IMAGE2PPT_AGENT_PROVIDER','deepseek')
    monkeypatch.setenv('IMAGE2PPT_AGENT_API_KEY','fake-reviewer')
    with pytest.raises(RuntimeError, match='distinct credentials'):
        configure_model(dual)
    with pytest.raises(RuntimeError, match='own credential'):
        MomaReviewer('qwen/qwen3.8-max')
    monkeypatch.delenv('IMAGE2PPT_AGENT_API_KEY')
    with pytest.raises(RuntimeError, match='Agent credential absent'):
        configure_model(dual)


def test_moma_reviews_official_deepseek_and_checks_wire_identity(dual, monkeypatch):
    monkeypatch.setenv('IMAGE2PPT_AGENT_PROVIDER','deepseek')
    monkeypatch.delenv('IMAGE2PPT_REVIEW_PROVIDER')
    monkeypatch.delenv('IMAGE2PPT_REVIEW_MODEL')
    write(dual/'request.json', {'agent_inference_provider':'deepseek'})
    verified,calls = run_audit(dual,monkeypatch)
    assert verified['passed'] and calls == [('qwen/qwen3.8-max',verified['session'])]
    assert verified['provider']=='moma' and verified['builder_provider']=='deepseek'
    assert verified['builder_model']=='deepseek-flash' and verified['builder_session']!=verified['session']
    for model in ('deepseek/deepseek-v4-flash-vision-exp','deepseek-v4-flash',
                  'deepseek-v4.1-flash-1017',None,'unrelated-model'):
        result,calls = run_audit(dual,monkeypatch,reported=model,changes=('deck.pptx',str(model).encode()))
        assert calls and result['status']=='MODEL_IDENTITY_UNVERIFIED' and not result['passed']
    result,calls = run_audit(dual,monkeypatch,control_model='deepseek-flash',changes=('edits.png',b'other edit'))
    assert not calls and result['status']=='CONTROL_UNVERIFIED' and not result['passed']
