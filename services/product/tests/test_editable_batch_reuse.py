"""Cross-turn replay of a validated batch; external staging and Office use fixtures."""
from pathlib import Path
import shutil
import sys
from uuid import uuid4

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.materialize import digest, write
from product.runtime.service import RuntimeService
from product.runtime.state import RuntimeConflict


def test_equivalent_batch_preserves_candidate_and_checks_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT', str(tmp_path/'locks'))
    monkeypatch.setenv('IMAGE2PPT_OFFICE_MODE', 'local')
    monkeypatch.setattr('product.fonts.font_manifest', lambda texts: {'fonts': []})
    monkeypatch.setattr('product.layout_evidence.available', lambda: False)
    service = RuntimeService(tmp_path)
    # Isolate the versioned inputs from concurrent edits in a developer's checkout.
    revision = tmp_path / 'fixture-implementation.py'
    revision.write_text('version = 1', encoding='utf8')
    monkeypatch.setattr(service, 'code_binding', lambda: digest(revision))
    monkeypatch.setattr('product.recovery.asset_cache.code_fingerprint',
                        lambda root, *, compiler: {str(revision): digest(revision)})
    toolchain = {'engine': 'fixture-node-A', 'rmbg': 'fixture-worker-A'}
    monkeypatch.setattr('product.recovery.asset_cache.compiler_runtime_fingerprint',
                        lambda *, local_office: {'compiler': toolchain['engine'], 'office': local_office})
    monkeypatch.setattr('product.matting.runtime_binding', lambda: {'worker': toolchain['rmbg']})
    job_id = uuid4().hex
    job = tmp_path / job_id
    (job/'inputs').mkdir(parents=True)
    Image.new('RGB', (80, 60), 'white').save(job/'inputs/01.png')
    write(job/'request.json', {'sources': ['01.png'], 'orchestrator': 'editable_reconstruction',
                               'visual_review_backend': 'moma'})
    service.call('create_scene', job_id, page=1, evidence={}, idempotency_key=uuid4().hex)
    invocations = []
    def stage(**kwargs):
        invocations.append('stage')
        staged = kwargs['directory'] / kwargs['tag']
        shutil.copytree(kwargs['initial_baseline'], staged)
        return staged, ['test-icon'], []
    monkeypatch.setattr('product.runtime.batch_backbone._chain_stage_from_spec', stage)
    actual_validate = service.validate_candidate
    def validate(state, current, directory, *, candidate_id):
        candidate = current['candidates'][candidate_id]
        if candidate['status'] == 'VALIDATED':
            return actual_validate(state, current, directory, candidate_id=candidate_id)
        invocations.append('office')
        receipt = directory / 'fixture-office.json'
        write(receipt, {'fixture': True})
        candidate['status'] = 'VALIDATED'
        candidate['validation'] = {'status': 'PASS', 'files': {str(receipt): digest(receipt)}}
        return candidate['validation']
    monkeypatch.setattr(service, 'validate_candidate', validate)

    state = service.state(job_id)
    def batch(**kwargs):
        return service.call('batch_reconstruct_page', job_id, expected_version=state.read()['version'],
                            idempotency_key=uuid4().hex, page=1, **kwargs)
    first = batch(include_region_crops=False)
    same = batch()
    assert first['candidate_id'] == same['candidate_id'] and same['cache_hit']
    assert len(state.read()['candidates']) == 1 and invocations == ['stage', 'office']
    assert same['version'] == first['version'] + 1
    assert f"expected_version={same['version']}" in same['next_step']
    assert same['committed_count'] == 0 and 'timings' not in same
    with pytest.raises(RuntimeConflict, match='version conflict'):
        service.call('batch_reconstruct_page', job_id, expected_version=first['version'],
                     idempotency_key=uuid4().hex, page=1)

    # Explicit object scopes use set semantics, while None and [] mean different work.
    scoped = batch(object_ids=['b', 'a', 'a'])
    assert batch(object_ids=['a', 'b'])['candidate_id'] == scoped['candidate_id']
    assert scoped['candidate_id'] != first['candidate_id']
    assert len(invocations) == 4
    revision.write_text('version = 2', encoding='utf8')
    upgraded = batch()
    assert upgraded['candidate_id'] != first['candidate_id'] and len(invocations) == 6
    # A changed background/evidence file is a different input even at the same IR version.
    backing = Path(state.read()['pages'][0]['ir']).parent / 'runtime-ocr-protection.json'
    write(backing, {'lines': [], 'changed': True})
    updated = batch()
    assert updated['candidate_id'] != upgraded['candidate_id'] and len(invocations) == 8
    toolchain['rmbg'] = 'fixture-worker-B'
    rematted = batch()
    toolchain['engine'] = 'fixture-node-B'
    recompiled = batch()
    assert len({x['candidate_id'] for x in (updated, rematted, recompiled)}) == 3
    updated = recompiled
    reuse_args = {'expected_version': updated['version'], 'idempotency_key': uuid4().hex, 'page': 1}
    hit = service.call('batch_reconstruct_page', job_id, **reuse_args)
    assert hit == service.call('batch_reconstruct_page', job_id, **reuse_args)
    candidate = state.read()['candidates'][updated['candidate_id']]
    validation_file = next(iter(candidate['validation']['files']))
    write(validation_file, {'fixture': 'changed'})
    with pytest.raises(RuntimeConflict, match='validation evidence changed'):
        batch()
    with pytest.raises(RuntimeConflict, match='completed operation artifact changed'):
        service.call('batch_reconstruct_page', job_id, **reuse_args)
    assert len(invocations) == 12
    def unavailable(*, local_office):
        raise FileNotFoundError('compiler version information unavailable')
    monkeypatch.setattr('product.recovery.asset_cache.compiler_runtime_fingerprint', unavailable)
    trial = batch()
    assert batch()['candidate_id'] != trial['candidate_id'] and len(invocations) == 16


@pytest.mark.parametrize('arguments', [
    {'page': True}, {'page': '1'}, {'page': 0}, {'page': 1, 'verify_office': 'false'},
    {'page': 1, 'include_assets': 1}, {'page': 1, 'object_ids': 'a-1'},
    {'page': 1, 'object_ids': ['']}, {'page': 1, 'object_ids': [2]}, {'page': 1, 'invented': True},
])
def test_invalid_batch_arguments_fail_before_reservation(tmp_path, arguments):
    job_id = uuid4().hex
    job = tmp_path / job_id
    job.mkdir()
    write(job/'request.json', {'orchestrator': 'editable_reconstruction', 'visual_review_backend': 'moma'})
    service = RuntimeService(tmp_path)
    with pytest.raises(ValueError):
        service.call('batch_reconstruct_page', job_id, idempotency_key=uuid4().hex, **arguments)
    state = service.state(job_id)
    with state.db() as db:
        assert db.execute('SELECT count(*) FROM operations').fetchone()[0] == 0
    assert state.read()['version'] == 0 and not (job/'state').exists()


def test_rmbg_binding_tracks_actual_weights_worker_and_model_code(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from product import matting

    model = tmp_path / 'model'
    model.mkdir()
    for name in ('model.safetensors', 'birefnet.py', 'BiRefNet_config.py'):
        (model / name).write_bytes(b'v1')
    worker = tmp_path / 'worker.py'
    worker.write_bytes(b'worker v1')
    monkeypatch.setattr(matting, 'WORKER', worker)
    monkeypatch.setattr(matting, '_config', lambda: {'model': str(model), 'sha256': None,
                                                      'python': None, 'python_path': None})
    monkeypatch.setattr(matting, '_can_run_in_process', lambda *args: True)
    versions = {'torch': '1', 'safetensors': '2'}
    monkeypatch.setattr(matting.importlib.metadata, 'distributions',
                        lambda *, name, path: [SimpleNamespace(version=versions[name])] if name in versions else [])
    original = matting.runtime_binding()
    assert original == matting.runtime_binding()
    worker.write_bytes(b'new worker')
    (model / 'model.safetensors').write_bytes(b'new weights')
    (model / 'BiRefNet_config.py').write_bytes(b'new config')
    new = matting.runtime_binding()
    assert all(original[k] != new[k] for k in ('worker_sha256', 'model_sha256', 'model_files'))
    del versions['safetensors']
    with pytest.raises(ValueError, match='engine version unavailable'):
        matting.runtime_binding()
