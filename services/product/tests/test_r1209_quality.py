"""Pure italic differences use the existing visual-minor schema."""
from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.recovery.adaptive_policy import QualityReview, quality_status


def review():
    return {'schema': 'VisualReview/1', 'source_sha256': 'source', 'plan_sha256': 'plan',
            'preview_sha256': 'preview', 'passed': True,
            'checked': ['wording', 'structure', 'font', 'emphasis', 'layout', 'graphics'],
            'findings': [], 'minor_deviations': [{'category': 'font', 'aspect': 'italic_only',
                'description': 'only font slant differs', 'node_ids': ['title'],
                'reason_to_accept': 'existing acceptable font difference'}]}


def test_italic_minor_is_retained_and_does_not_override_hard_gates():
    parsed = QualityReview.model_validate(review())
    assert parsed.minor_deviations[0].aspect == 'italic_only'
    assert quality_status(True, parsed.model_dump()) == 'ACCEPT_WITH_MINOR_DEVIATION'
    assert quality_status(False, parsed.model_dump()) == 'HARD_GATE_FAILED'
    assert quality_status(True, {'status': 'UNKNOWN'}) == 'UNKNOWN'


@pytest.mark.parametrize('category', ['wording', 'structure', 'graphics'])
def test_italic_tag_cannot_accept_content_or_graphic_defects(category):
    raw = review()
    raw['minor_deviations'][0]['category'] = category
    with pytest.raises(ValidationError):
        QualityReview.model_validate(raw)


def test_missing_response_or_incomplete_review_stays_invalid():
    with pytest.raises(ValidationError):
        QualityReview.model_validate({})
    raw = review()
    raw['checked'].remove('wording')
    with pytest.raises(ValidationError):
        QualityReview.model_validate(raw)
    raw = review()
    raw['findings'] = [{'category': 'wording', 'description': 'missing text'}]
    with pytest.raises(ValidationError):
        QualityReview.model_validate(raw)
