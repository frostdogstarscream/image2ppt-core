"""Run: python -m pytest services/product/tests/test_r12_background_adapter.py -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.recovery.background_adapter import RecoveryBlocked, generate_background_candidates


def scene():
    source = np.full((96, 160, 3), 230, np.uint8)
    lower = source.copy()
    # Synthetic dark text strokes embedded in the source screenshot.
    source[30:62, 60:64] = 20
    source[30:62, 72:76] = 20
    mask = np.zeros(source.shape[:2], bool)
    mask[28:64, 56:81] = True
    protected = np.zeros_like(mask)
    protected[28:64, 90:99] = True
    return source, lower, mask, protected


def test_verified_layerd_lower_is_first_and_original_unchanged():
    source, lower, mask, protected = scene()
    original = source.copy()
    results, issues = generate_background_candidates(source, mask, protected, lower_rgb=lower,
                                                       lower_provenance_verified=True)
    assert results and results[0].method == "layerd_lower"
    assert results[0].status == "PRECHECK_PASS"
    assert np.array_equal(source, original)
    assert np.array_equal(results[0].image[~mask], source[~mask])
    assert np.all(results[0].image[mask] == 230)
    assert not issues


def test_unverified_lower_is_not_trusted_and_lama_can_repair():
    source, lower, mask, protected = scene()
    calls = []
    def fake_lama(image, input_mask):
        calls.append((image.copy(), input_mask.copy()))
        image[input_mask] = 230
        image[0, 0] = 0  # A badly behaved service must never change outside mask.
        return image
    results, issues = generate_background_candidates(source, mask, protected, lower_rgb=lower,
                                                       lower_provenance_verified=False, inpaint=fake_lama)
    assert len(calls) == 1
    assert any(x['code'] == 'UNVERIFIED_LOWER_PROVENANCE' for x in issues)
    assert [r.method for r in results] == ['big_lama']
    assert np.array_equal(results[0].image[~mask], source[~mask])


def test_protected_collision_blocks_all_tools():
    source, lower, mask, protected = scene()
    protected[30, 60] = True
    called = []
    with pytest.raises(RecoveryBlocked) as err:
        generate_background_candidates(source, mask, protected, lower_rgb=lower,
                                       lower_provenance_verified=True,
                                       inpaint=lambda *_: called.append(1))
    assert err.value.code == 'PROTECTED_NEIGHBOR'
    assert not called


def test_no_context_is_unknown_not_pass():
    source, lower, mask, protected = scene()
    mask[:] = True
    protected[:] = False
    results, issues = generate_background_candidates(source, mask, protected, lower_rgb=lower,
                                                       lower_provenance_verified=True)
    assert not results
    assert any(x['code'] == 'LOWER_NOT_VERIFIED' for x in issues)


def test_lama_uncertain_keeps_trial_but_not_approved():
    source, lower, mask, protected = scene()
    def bad_lama(image, input_mask):
        return image
    results, issues = generate_background_candidates(source, mask, protected, inpaint=bad_lama)
    assert results[0].status == 'REVIEW_REQUIRED'
    assert any(x['code'] == 'INPAINT_NEEDS_REVIEW' for x in issues)


def test_lower_context_mismatch_blocks_false_pass():
    source, lower, mask, protected = scene()
    lower[24:27, 60:70] = 100  # near the target, outside authorized mask
    results, issues = generate_background_candidates(source, mask, protected, lower_rgb=lower,
                                                       lower_provenance_verified=True)
    assert not results
    assert issues[0]['code'] == 'LOWER_NOT_VERIFIED'
    assert issues[0]['metrics']['ring_mae'] > 3


def test_valid_lower_skips_optional_lama_by_default():
    source, lower, mask, protected = scene()
    called = []
    results, _ = generate_background_candidates(source, mask, protected, lower_rgb=lower,
                                                lower_provenance_verified=True,
                                                inpaint=lambda *_: called.append(1))
    assert len(results) == 1 and results[0].method == 'layerd_lower'
    assert not called
