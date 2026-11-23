import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from product import matting
from product.materialize import digest, write
from product.object_contracts import ClosedObject
from product.parallel_fusion import fuse
from product.parallel_rebuild import build
from product.recovery.agent_asset_stage import stage_asset_candidate
from product.recovery.asset_retry import retry_asset_extraction
from test_object_extraction import fixture
from test_parallel import branches


def test_opencode_matte_timeout_is_bound_to_crop_model_and_job(tmp_path, monkeypatch):
    from product.runtime.state import RuntimeState

    model = tmp_path / 'weights.onnx'
    model.write_bytes(b'fixture weights only')
    monkeypatch.setenv('IMAGE2PPT_MATTING_MODEL', str(model))
    monkeypatch.setattr(matting, 'binding', lambda: {'model_sha256': digest(model)})
    calls = []

    def run_child(argv, **kwargs):
        assert argv[-2:] == ['--size', '16'] and 0 < kwargs['timeout'] <= 120
        with Image.open(argv[3]) as image:
            value = image.getpixel((0, 0))[0]
            calls.append(value)
            if value in (1, 3):
                raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
            Image.new('L', image.size, color=255).save(argv[4])

    monkeypatch.setattr(matting.subprocess, 'run', run_child)

    def predict(value):
        return matting.predict_alpha(np.full((10, 12, 3), value, np.uint8), size=16)

    state = RuntimeState(tmp_path)
    def operation(current, directory):
        for value in (1, 1):
            with pytest.raises(matting.MattingTimeout):
                predict(value)
        assert predict(2).shape == (10, 12)
        for value in (3, 4):
            with pytest.raises(matting.MattingTimeout):
                predict(value)
        return {'status': 'completed-with-other-candidates'}

    state.mutate('batch', {}, None, 'optional-matte-test', operation)
    assert calls == [1, 2, 3]
    history = json.loads((state.root/'rmbg-timeouts.json').read_text('utf8'))
    assert history['model']['model_sha256'] == digest(model)
    assert len(history['timeouts']) == 2
    with matting.job_matting(tmp_path/'another-job'):
        assert predict(4).min() == 1
    assert calls == [1, 2, 3, 4]


def test_interrupted_matte_does_not_cache_other_candidates_as_a_full_comparison(tmp_path, monkeypatch):
    from product import parallel_rebuild

    source, _ = _scene()
    obj = SimpleNamespace(id='glyph', bbox=[28, 18, 24, 24], layers=[], exclusions=[], parts=[])
    monkeypatch.setattr(parallel_rebuild, '_matting_binding', lambda strategy: {'model_sha256': 'fixture'})
    monkeypatch.setattr(parallel_rebuild, 'matte_from_evidence', lambda *a, **kw: ('sample', {'method': 'fitted'}))

    def interrupted(*args):
        raise matting.MattingTimeout('RMBG_TIMEOUT')

    monkeypatch.setattr(matting, 'matte_candidate', interrupted)
    kept = parallel_rebuild._asset_candidates(obj, source, None, np.zeros(source.shape[:2], bool),
                                               [], 'auto', cache=tmp_path)
    assert kept == [('sample', {'method': 'fitted'})]
    assert not list(tmp_path.rglob('manifest.json'))


def _scene():
    src = np.full((60, 80, 3), 240, np.uint8)
    yy, xx = np.indices((60, 80))
    disk = (yy - 30) ** 2 + (xx - 40) ** 2 <= 100
    src[disk] = [20, 90, 200]
    return src, disk


def test_matte_candidate_unmixes_against_fitted_backing():
    src, disk = _scene()
    obj = SimpleNamespace(bbox=[28, 18, 24, 24], exclusions=[], parts=[])
    predictor = lambda crop: (np.abs(crop.astype(float) - 240).max(2) > 30).astype(np.float32)
    rgba, meta = matting.matte_candidate(obj, src, np.zeros(src.shape[:2], bool), predictor=predictor)
    assert meta['method'] == 'rmbg2' and meta['ring_error'] <= 6
    assert (rgba[..., 3] > 0).sum() == disk.sum()
    assert np.array_equal(rgba[disk][:, :3], src[disk])


def test_matte_touching_crop_boundary_or_protected_pixels_is_rejected():
    src, disk = _scene()
    obj = SimpleNamespace(bbox=[28, 18, 24, 24], exclusions=[], parts=[])
    with pytest.raises(ValueError):
        matting.matte_candidate(obj, src, np.zeros(src.shape[:2], bool),
                                predictor=lambda crop: np.ones(crop.shape[:2], np.float32))
    protected = disk.copy()
    with pytest.raises(ValueError):
        matting.matte_candidate(obj, src, protected,
                                predictor=lambda crop: (np.abs(crop.astype(float) - 240).max(2) > 30).astype(np.float32))


def test_soft_alpha_unmix_recovers_foreground_colour():
    backing = np.full((4, 4, 3), 200.0)
    fg = np.array([10.0, 20.0, 30.0])
    alpha = np.full((4, 4), 0.5, np.float32)
    crop = (alpha[..., None] * fg + (1 - alpha[..., None]) * backing).astype(np.uint8)
    rgba = matting.unmix(crop, backing, alpha)
    assert np.abs(rgba[..., :3].astype(int) - fg).max() <= 2 and (rgba[..., 3] == 128).all()


def test_retry_asset_extraction_rmbg2_stages_through_shared_gates(tmp_path, monkeypatch):
    layers, _, _ = fixture(tmp_path)
    src_path = layers / 'source.png'
    snap, evidence = branches(src_path)
    evidence['asset'].objects.append(ClosedObject(
        id='icon-1', description='whole independent source icon',
        prompt_en='whole independent source icon', kind='icon',
        bbox=[8, 8, 16, 20], layers=['layer-01.png'], components=['whole'],
        text_policy='exclude_independent_text',
        parts=[{'name': 'whole', 'bbox': [8, 8, 16, 20], 'layers': ['layer-01.png']}],
    ))
    graph = fuse(snap, evidence, {'lines': []})
    baseline = tmp_path / 'baseline'
    build(graph, snap, {'fonts': []}, baseline, strategy='off', text_candidates=False)

    layer0 = np.array(Image.open(layers / 'layer-00.png').convert('RGB'))
    bg = np.median(layer0.reshape(-1, 3), axis=0)
    monkeypatch.setattr(matting, 'predict_alpha',
                        lambda crop: (np.abs(crop.astype(float) - bg).max(2) > 25).astype(np.float32))

    res = retry_asset_extraction(graph, 'icon-1', src_path, layers, baseline,
                                 tmp_path / 'retry-rmbg2', strategy='rmbg2')
    assert res['status'] == 'CANDIDATE_READY'
    ir = stage_asset_candidate(res['candidate_directory'], graph, baseline,
                               tmp_path / 'staged-rmbg2', layer_directory=layers)
    staged_graph = fuse(snap, evidence, {'lines': []}).model_validate_json((ir.parent / 'scene.json').read_text('utf8'))
    assert 'icon-1' in staged_graph.selected
    assert staged_graph.selected['icon-1']['owner_evidence']


def test_predict_alpha_uses_in_process_cached_session(tmp_path, monkeypatch):
    model = tmp_path / 'fake.onnx'
    model.write_bytes(b'fake')
    calls = []

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name='pixel_values', type='tensor(float)')]

        def run(self, _out, feed):
            calls.append(feed['pixel_values'].shape)
            return [np.full((1, 1, 16, 16), 2.0, np.float32)]

    monkeypatch.setenv('IMAGE2PPT_MATTING_MODEL', str(model))
    monkeypatch.setattr(matting, '_ort_session', lambda m, p: FakeSession())
    alpha = matting.predict_alpha(np.zeros((10, 12, 3), np.uint8), size=16)
    assert alpha.shape == (10, 12) and alpha.min() > 0.8 and len(calls) == 1


def test_predict_alpha_uses_cached_pytorch_directory_model(tmp_path, monkeypatch):
    import torch

    model_dir = tmp_path / 'rmbg-pt'
    model_dir.mkdir()
    for name in ('model.safetensors', 'birefnet.py', 'BiRefNet_config.py'):
        (model_dir / name).write_bytes(b'x')

    class FakeNet:
        def __call__(self, tensor):
            return [torch.full((1, 1, tensor.shape[2], tensor.shape[3]), 2.0, dtype=torch.float32)]

    monkeypatch.setenv('IMAGE2PPT_MATTING_MODEL', str(model_dir))
    monkeypatch.setattr(matting, '_torch_model', lambda _d: (FakeNet(), 'cpu'))
    alpha = matting.predict_alpha(np.zeros((10, 12, 3), np.uint8), size=16)
    assert alpha.shape == (10, 12) and alpha.min() > 0.8
