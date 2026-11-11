"""Durable local region receipts survive a later composition interruption."""
import json
from pathlib import Path

import pytest

from test_r12_product_pipeline import product_fixture


def test_restart_reuses_verified_regions_after_compile_failure(product_fixture,monkeypatch):
    from product.parallel_pipeline import run
    from product.recovery import product_pipeline as regional
    fixture=product_fixture
    original=regional.stage_and_verify
    calls=[]
    def verify(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(regional,'stage_and_verify',verify)
    def fail_compilation(*args):raise RuntimeError('injected composition interruption')
    with pytest.raises(RuntimeError,match='composition interruption'):
        run(fixture.job,provider=fixture.actor,review_provider=fixture.actor,
            compiler=fail_compilation,office=fixture.office,fonts={'fonts':[]})
    assert len(calls)==2
    result=run(fixture.job,provider=fixture.actor,review_provider=fixture.actor,
        compiler=fixture.compiler,office=fixture.office,fonts={'fonts':[]})
    assert result['locally_verified_text_regions']==2 and len(calls)==2
    assert all(r['resumed_from_verified_receipt'] for r in result['regions'][0])
    assert json.loads((fixture.job/'qa.json').read_text('utf8'))['regions']==result['regions']
