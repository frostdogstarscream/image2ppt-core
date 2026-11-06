import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from relation_validator import prepare_spec, check_relations, check_authorized_changes, check_edit_regions


REVIEW_ROOT = os.environ.get('IMAGE2PPT_R6_REVIEW_ROOT')
REPO = Path(__file__).resolve().parents[2]
REVIEW_DIRS = [root/'outputs/review-4090-20261106-r6/delivery' for root in [REPO, *REPO.parents]]
SAMPLES = (Path(REVIEW_ROOT) if REVIEW_ROOT else next(
    (path for path in REVIEW_DIRS if (path/'pages').is_dir()), REVIEW_DIRS[0]))/'pages'


@pytest.mark.parametrize('page,count', [('02', 1), ('08', 4)])
def test_frozen_real_source_passes_and_r6_missing_elements_fail(tmp_path, page, count):
    source, current = SAMPLES/page/'source.png', SAMPLES/page/'r6.png'
    if not source.exists():
        if REVIEW_ROOT: pytest.fail('Configured IMAGE2PPT_R6_REVIEW_ROOT lacks source: '+str(source))
        pytest.skip('Private source regression images not installed')
    spec = prepare_spec(page, source, tmp_path)
    assert len(spec['elements']) == count
    assert check_relations(source, source, spec)['status'] == 'PASS'
    result = check_relations(source, current, spec)
    assert result['status'] == 'FAIL'
    assert all(x['status'] == 'FAIL' for x in result['objects'])


def test_unauthorized_and_protected_change_is_rejected():
    before = np.zeros((20,20,3), np.uint8)
    allowed = np.zeros((20,20), bool); allowed[5:10,5:10] = True
    protected = np.zeros_like(allowed); protected[7,7] = True
    after = before.copy(); after[7,7] = 255
    assert check_authorized_changes(before, after, allowed, protected)['status'] == 'FAIL'
    after = before.copy(); after[0,0] = 255
    assert check_authorized_changes(before, after, allowed)['outside_changed_pixels'] == 1


def test_edit_gate_requires_independent_expectation_and_checks_edges():
    before = np.zeros((20,20,3), np.uint8); after = before.copy()
    old = np.zeros((20,20), bool); old[2:6,2:6] = True
    new = np.zeros_like(old); new[12:16,12:16] = True
    protected = np.zeros_like(old)
    args = dict(allowed_mask=old|new, old_mask=old, new_mask=new, protection_mask=protected)
    assert check_edit_regions(before,after,before,**args)['status'] == 'REVIEW_REQUIRED'
    after[2,2] = 255
    result = check_edit_regions(before,after,before,expected_plate_verified=True,**args)
    assert result['status'] == 'FAIL'
    assert result['regions']['edge_ring']['status'] == 'FAIL'
