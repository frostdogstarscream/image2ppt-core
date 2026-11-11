"""Input adapters only. Mock transport does not establish service acceptance."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.recovery import layerd_adapter
from product.recovery.background_adapter import RecoveryBlocked
from product.recovery.lama_adapter import LamaClient
from product.object_extract import recompose
from layerd_loader import Layer, Page
import text_mask


def layer_page(tmp_path, monkeypatch, *, opaque=True):
    arrays = [np.full((32, 48, 4), [220, 230, 240, 255 if opaque else 128], np.uint8),
              np.full((32, 48, 4), [40, 60, 80, 96], np.uint8),
              np.zeros((32, 48, 4), np.uint8)]
    layers = []
    for index, array in enumerate(arrays):
        path = tmp_path / f'layer-{index}.png'
        Image.fromarray(array).save(path)
        layers.append(Layer(path.name, path, index, 'foreground', (48, 32),
                            'RGBA', [0, 0, 48, 32], 1, 0, str(index)))
    page = Page('synthetic', tmp_path, (48, 32), 'back-to-front',
                tmp_path / 'source.png', None, layers)
    source = recompose(page)
    Image.fromarray(source).save(page.source_path)
    (tmp_path / 'manifest.json').write_text('{}', encoding='utf8')
    monkeypatch.setattr(layerd_adapter, 'load_page', lambda _: page)
    mask = np.zeros(source.shape[:2], bool)
    mask[10:20, 10:20] = True
    return page, source, mask


def test_lower_prefix_order_alpha_and_source_binding(tmp_path, monkeypatch):
    page, source, mask = layer_page(tmp_path, monkeypatch)
    results = layerd_adapter.lower_backgrounds(source, tmp_path, mask)
    assert [r.provenance['below_z'] for r in results] == [1, 2]
    assert np.array_equal(results[1].image, recompose(replace(page, layers=page.layers[:2])))
    assert not results[0].image.flags.writeable
    assert results[1].provenance['source_file_sha256']
    changed = source.copy()
    changed[0, 0, 0] ^= 1
    with pytest.raises(RecoveryBlocked, match='source pixels'):
        layerd_adapter.lower_backgrounds(changed, tmp_path, mask)


def test_transparent_lower_does_not_treat_white_canvas_as_evidence(tmp_path, monkeypatch):
    _, source, mask = layer_page(tmp_path, monkeypatch, opaque=False)
    assert layerd_adapter.lower_backgrounds(source, tmp_path, mask) == []


def test_recomposition_mismatch_blocks_prefixes(tmp_path, monkeypatch):
    _, source, mask = layer_page(tmp_path, monkeypatch)
    monkeypatch.setattr(layerd_adapter, 'recompose', lambda _: np.zeros_like(source))
    with pytest.raises(RecoveryBlocked) as error:
        layerd_adapter.lower_backgrounds(source, tmp_path, mask)
    assert error.value.code == 'LAYER_RECOMPOSITION_MISMATCH'


def inputs():
    source = np.full((20, 30, 3), 230, np.uint8)
    mask = np.zeros((20, 30), bool)
    mask[5:15, 10:20] = True
    return source, mask


def test_lama_reservation_before_transport_and_bound_cache(tmp_path, monkeypatch):
    source, mask = inputs()
    calls = []
    def transport(image, selected, url, timeout):
        receipt = json.loads(next(tmp_path.glob('*/request.json')).read_text('utf8'))
        assert receipt['status'] == 'RESERVED'
        calls.append(url)
        image[selected] = 200
        return image
    monkeypatch.setattr(text_mask, 'remote_inpaint', transport)
    client = LamaClient('http://127.0.0.1:18848/inpaint', 'test-model', tmp_path)
    first = client(source, mask)
    assert np.array_equal(first, client(source, mask))
    assert len(calls) == 1
    assert np.all(source == 230)
    response = next(tmp_path.glob('*/response.png'))
    Image.fromarray(source).save(response)
    with pytest.raises(RecoveryBlocked) as error:
        client(source, mask)
    assert error.value.code == 'INPAINT_CACHE_MISMATCH'
    assert len(calls) == 1


@pytest.mark.parametrize('invalid', [False, True])
def test_unknown_lama_response_never_resubmitted(tmp_path, monkeypatch, invalid):
    source, mask = inputs()
    calls = []
    def transport(*args, **kwargs):
        calls.append(1)
        if invalid:
            return np.zeros((1, 1, 3), np.uint8)
        raise TimeoutError('simulated response loss')
    monkeypatch.setattr(text_mask, 'remote_inpaint', transport)
    client = LamaClient('http://127.0.0.1:18848/inpaint', 'test-model', tmp_path)
    for _ in range(2):
        with pytest.raises(RecoveryBlocked) as error:
            client(source, mask)
        assert error.value.code == 'INPAINT_RESPONSE_UNKNOWN'
    assert len(calls) == 1
    assert json.loads(next(tmp_path.glob('*/request.json')).read_text('utf8'))['status'] == 'UNKNOWN'


@pytest.mark.parametrize('url', ['ssh://host/inpaint', 'http://user:password@localhost/inpaint',
                                 'http://localhost/inpaint?token=secret'])
def test_lama_rejects_unsupported_transport_or_embedded_credentials(tmp_path, url):
    with pytest.raises(ValueError):
        LamaClient(url, 'test-model', tmp_path)


def test_all_layer_prefixes_are_tried_before_one_lama_call(monkeypatch):
    from product.recovery import candidates
    source, mask = inputs()
    source[7:13, 12:15] = 20
    protected = np.zeros_like(mask)
    bad = source.copy()
    calls = []
    monkeypatch.setattr(candidates, 'lower_backgrounds', lambda *args: [
        layerd_adapter.LowerBackground(bad, {'below_z': 1}),
        layerd_adapter.LowerBackground(bad, {'below_z': 2})])
    def inpaint(image, selected):
        calls.append(1)
        image[selected] = 230
        return image
    result, issues = candidates.generate_with_layerd(source, mask, protected, '.', inpaint=inpaint)
    assert len(calls) == 1
    assert result[0].method == 'big_lama'
    assert [r['provenance']['below_z'] for r in issues if r['code'] == 'LOWER_NOT_VERIFIED'] == [1, 2]


def test_good_layer_prefix_skips_lama_and_keeps_provenance(monkeypatch):
    from product.recovery import candidates
    source, mask = inputs()
    lower = source.copy()
    source[7:13, 12:15] = 20
    monkeypatch.setattr(candidates, 'lower_backgrounds', lambda *args: [
        layerd_adapter.LowerBackground(lower, {'below_z': 1})])
    def forbidden(*args):
        pytest.fail('valid lower should skip inpaint')
    result, _ = candidates.generate_with_layerd(source, mask, np.zeros_like(mask), '.', inpaint=forbidden)
    assert result[0].method == 'layerd_lower'
    assert result[0].diagnostics['layerd_provenance']['below_z'] == 1


def test_protected_pixels_block_before_loading_layers(monkeypatch):
    from product.recovery import candidates
    source, mask = inputs()
    def forbidden(*args):
        pytest.fail('protection conflict must block before tools')
    monkeypatch.setattr(candidates, 'lower_backgrounds', forbidden)
    with pytest.raises(RecoveryBlocked) as error:
        candidates.generate_with_layerd(source, mask, mask, '.', inpaint=forbidden)
    assert error.value.code == 'PROTECTED_NEIGHBOR'
