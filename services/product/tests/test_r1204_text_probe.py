"""Local candidate cache safety without network, compiler or Office calls."""
import pytest

from test_r12_product_pipeline import product_fixture
from product.parallel_rebuild import build
from product.recovery import text_probe


def prepare(fixture):
    fonts = {'fonts': []}
    ir, _ = build(fixture.graph, fixture.snap, fonts, fixture.job/'baseline',
                  strategy='off', text_candidates=False)
    return fonts, ir.parent


def test_repeated_local_probe_reuses_first_staged_candidate(product_fixture, monkeypatch):
    fixture = product_fixture
    fonts, baseline = prepare(fixture)
    original = text_probe.adaptive_candidates
    calls = []
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(text_probe, 'adaptive_candidates', count)
    args = (fixture.graph, 'a', fixture.snap['path'], None, baseline, fonts, fixture.job/'probe')
    first = text_probe.probe_one(*args)
    second = text_probe.probe_one(*args)
    assert first['state'] == second['state'] == 'STAGED_LOCAL'
    assert not first['cache_hit'] and second['cache_hit']
    assert calls == [1]
    assert first['office'] == first['visual'] == 'NOT_RUN'
    assert not first['automatic_publication']


def test_modified_cached_candidate_is_rejected(product_fixture):
    fixture = product_fixture
    fonts, baseline = prepare(fixture)
    args = (fixture.graph, 'a', fixture.snap['path'], None, baseline, fonts, fixture.job/'probe')
    result = text_probe.probe_one(*args)
    from pathlib import Path
    file = Path(result['candidate_directory'])/'written_back.png'
    file.write_bytes(file.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='cached artifact changed'):
        text_probe.probe_one(*args)


def test_modified_source_cannot_reuse_recovery_receipt(product_fixture):
    fixture = product_fixture
    fonts, baseline = prepare(fixture)
    args = (fixture.graph, 'a', fixture.snap['path'], None, baseline, fonts, fixture.job/'probe')
    text_probe.probe_one(*args)
    from pathlib import Path
    file = Path(fixture.snap['path'])
    file.write_bytes(file.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='source binding mismatch'):
        text_probe.probe_one(*args)
