"""Provider transport fixtures; never connects to Dify, SSH or a model API."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import urllib.error
import urllib.request

import pytest

spec = importlib.util.spec_from_file_location('auto_remote_test', Path(__file__).parents[1]/'tools/auto_remote.py')
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def stream_event(value): return b'data: '+json.dumps(value).encode()+b'\n\n'


def invoke_dify(monkeypatch, lines, *, content_type='text/event-stream', error=None):
    class Query:
        def where(self, *a): return self
    row = SimpleNamespace(tenant_id='fixture', encrypted_config=json.dumps({'api_key': 'encrypted-fixture', 'endpoint_url': 'https://fixture.invalid/v1'}))
    session = SimpleNamespace(scalars=lambda _: SimpleNamespace(all=lambda: [row]))
    app = SimpleNamespace(app_context=contextlib.nullcontext)
    modules = {
        'app_factory': SimpleNamespace(create_app=lambda: (app, object())),
        'extensions.ext_database': SimpleNamespace(db=SimpleNamespace(session=session)),
        'models.provider': SimpleNamespace(ProviderModelCredential=SimpleNamespace(provider_name='provider', model_name='model')),
        'core.helper.encrypter': SimpleNamespace(decrypt_token=lambda *_: 'fixture-secret-value'),
        'sqlalchemy': SimpleNamespace(select=lambda _: Query()),
    }
    for name, module in modules.items(): monkeypatch.setitem(sys.modules, name, module)
    class Response:
        headers = {'Content-Type': content_type}
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __iter__(self): return iter(lines)
        def read(self, *_): return b''.join(lines)
    opened = []
    class Opener:
        def open(self, request, timeout):
            opened.append(request)
            if error: raise error
            return Response()
    monkeypatch.setattr(urllib.request, 'build_opener', lambda *_: Opener())
    payload = {'credential_provider': 'provider', 'credential_model': 'model', 'base_url': 'https://fixture.invalid/v1',
               'timeout': 60, 'body': {'model': 'fixture', 'messages': []}}
    stdout = io.StringIO(); monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(payload)))
    with contextlib.redirect_stdout(stdout): exec(compile(worker.DIFY_CALL, '<provider fixture>', 'exec'), {})
    result = json.loads(stdout.getvalue().split('I2P_RESULT=')[-1])
    return result, opened, stdout.getvalue()


def test_stream_collects_content_and_usage_without_secret(monkeypatch):
    lines = [stream_event({'choices': [{'delta': {'content': '{"answer":"'}, 'finish_reason': None}]}),
             stream_event({'choices': [{'delta': {'content': 'fixture-secret-value"}'}, 'finish_reason': 'stop'}]}),
             stream_event({'choices': [], 'usage': {'total_tokens': 12}}), b'data: [DONE]\n']
    result, requests, raw = invoke_dify(monkeypatch, lines)
    assert result['ok'] and result['content'] == '{"answer":"[REDACTED]"}'
    assert result['usage']['total_tokens'] == 12 and 'fixture-secret-value' not in raw
    assert json.loads(requests[0].data)['stream'] is True


@pytest.mark.parametrize('lines', [
    [stream_event({'choices': [{'delta': {'content': '{}'}, 'finish_reason': None}]}), b'data: [DONE]\n'],
    [stream_event({'choices': [{'delta': {'content': '{}'}, 'finish_reason': 'length'}]}), b'data: [DONE]\n'],
    [stream_event({'choices': [{'delta': {'content': '{}'}, 'finish_reason': 'stop'}]})],
])
def test_incomplete_stream_never_becomes_a_plan(monkeypatch, lines):
    result, _, _ = invoke_dify(monkeypatch, lines)
    assert not result['ok'] and result['error'] == 'model_response_incomplete'


def test_nonstream_compatible_response(monkeypatch):
    result, _, _ = invoke_dify(monkeypatch, [json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}).encode()], content_type='application/json')
    assert result['ok'] and result['content'] == '{}'


def test_http_error_does_not_return_body_or_url(monkeypatch):
    error = urllib.error.HTTPError('https://fixture.invalid/secret', 403, 'secret', {}, io.BytesIO(b'secret'))
    result, _, raw = invoke_dify(monkeypatch, [], error=error)
    assert result == {'ok': False, 'error': 'model_http_403', 'http_status': 403}
    assert 'secret' not in raw


def test_tool_call_stream_rejected(monkeypatch):
    lines = [stream_event({'choices': [{'delta': {'content': '{}', 'tool_calls': [{'id': 'bad'}]}, 'finish_reason': 'stop'}]}), b'data: [DONE]\n']
    result, _, _ = invoke_dify(monkeypatch, lines)
    assert not result['ok'] and result['error'] == 'invalid_model_message'
