"""Evidence guards for local mask retry and background switching tools."""
from pathlib import Path

import numpy as np
import pytest

from test_r12_product_pipeline import product_fixture
from product.parallel_rebuild import build
from product.parallel_store import key
from product.parallel_text_candidates import peer_protection
from product.recovery import text_actions, text_probe
from product.recovery.background_adapter import RecoveryBlocked
from product.recovery.diagnostics import save_trial
from product.recovery.gradient_backing import fit_gradient


def baseline(fixture):
    fonts = {'fonts': []}
    ir, _ = build(fixture.graph, fixture.snap, fonts, fixture.job/'baseline',
                  strategy='off', text_candidates=False)
    return fonts, ir.parent


def rejected_trial(fixture, directory, mask, raw):
    text = fixture.graph.text.texts[0]
    x, y, w, h = map(round, text.bbox)
    box = [x-3, y-3, w+6, h+6]
    protected = peer_protection(text, fixture.graph, fixture.source,
                                np.zeros(fixture.source.shape[:2], bool))
    reference, report = fit_gradient(fixture.source, box, protected)
    return save_trial(directory, fixture.source, mask, raw, protected, box, reference, report,
                      {'object_id': text.id, 'source_sha256': fixture.graph.source_sha256,
                       'scene_graph_sha256': key(fixture.graph.model_dump()),
                       'mask_variant': 'original', 'candidate': 'original--layerd-0', 'method': 'layerd-0'})


def test_mask_retry_reuses_unchanged_failure(product_fixture, monkeypatch):
    fixture = product_fixture
    fonts, parent = baseline(fixture)
    calls = []
    def unavailable(*args, **kwargs):
        calls.append(1)
        raise RecoveryBlocked('MASK_INCOMPLETE', 'injected unresolved glyph pixels')
        yield
    monkeypatch.setattr(text_probe, 'adaptive_candidates', unavailable)
    arguments = (fixture.graph, 'a', fixture.snap['path'], None, parent, fonts, fixture.job/'probe')
    previous = text_probe.probe_one(*arguments)
    result = text_actions.retry_text_mask_completion(*arguments, previous_receipt=previous['receipt_file'])
    assert calls == [1]
    assert result['state'] == 'BLOCKED' and result['cache_hit']
    assert not result['input_evidence_changed']


def test_partial_source_mask_can_retry_existing_completion(product_fixture):
    fixture = product_fixture
    fonts, parent = baseline(fixture)
    mask = np.zeros(fixture.source.shape[:2], bool)
    mask[25:40, 25:28] = True
    raw = np.full_like(fixture.source, 235)
    rejected = fixture.job/'incomplete'
    report = rejected_trial(fixture, rejected, mask, raw)
    assert report['visible_source_pixels_outside_mask'] > 0
    result = text_actions.retry_text_mask_completion(
        fixture.graph, 'a', fixture.snap['path'], None, parent, fonts, fixture.job/'retry',
        previous_receipt=rejected/'residual_report.json')
    assert result['state'] == 'STAGED_LOCAL'
    assert result['previous_missing_pixels'] == report['visible_source_pixels_outside_mask']
    assert result['office'] == result['visual'] == 'NOT_RUN'


def test_background_switch_requires_background_specific_evidence(product_fixture):
    fixture = product_fixture
    fonts, parent = baseline(fixture)
    mask = np.zeros(fixture.source.shape[:2], bool)
    mask[25:40, 25:30] = True
    rejected = fixture.job/'bad-background'
    report = rejected_trial(fixture, rejected, mask, np.full_like(fixture.source, 200))
    assert report['checks']['complete_visible_scope'] and report['status'] == 'REVIEW_REQUIRED'
    result = text_actions.switch_background_candidate(
        fixture.graph, 'a', fixture.snap['path'], None, parent, fonts, fixture.job/'switch',
        rejected_candidate_directory=rejected, rejection_report=rejected/'residual_report.json')
    assert result['state'] == 'STAGED_LOCAL'
    assert Path(result['candidate_directory']).name != report['candidate']
    assert not result['automatic_publication']


def test_incomplete_mask_cannot_be_waived_by_background_switch(product_fixture):
    fixture = product_fixture
    fonts, parent = baseline(fixture)
    mask = np.zeros(fixture.source.shape[:2], bool)
    mask[25:40, 25:28] = True
    rejected = fixture.job/'incomplete'
    rejected_trial(fixture, rejected, mask, np.full_like(fixture.source, 235))
    with pytest.raises(ValueError, match='complete legal mask'):
        text_actions.switch_background_candidate(
            fixture.graph, 'a', fixture.snap['path'], None, parent, fonts, fixture.job/'switch',
            rejected_candidate_directory=rejected, rejection_report=rejected/'residual_report.json')


def test_transparent_font_frame_conflict_is_not_source_owner_failure():
    row = {'state': 'BLOCKED', 'reason': 'native font frame allowance conflicts with canvas or protected peers'}
    assert text_actions.classify_failure(row) == 'FONT_LAYOUT_FAILED'
    row = {'state': 'BLOCKED', 'failure': 'PROTECTED_NEIGHBOR',
           'reason': 'confirmed text overlaps protected peer'}
    assert text_actions.classify_failure(row) == 'OWNER_CONFLICT'
