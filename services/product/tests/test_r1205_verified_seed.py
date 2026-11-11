"""Synthetic committed receipts test reuse rules, not Office acceptance."""
import json
from pathlib import Path

import pytest

from test_r12_product_pipeline import product_fixture
from product.materialize import digest
from product.parallel_contracts import SceneGraph
from product.parallel_pipeline import run
from product.recovery.verified_seed import load,materialize


@pytest.fixture
def seed(product_fixture):
    f=product_fixture
    font=f.job/'synthetic-font.bin';font.write_bytes(b'SYNTHETIC-FONT')
    fonts={'fonts':[{'family':'Arial','bold':False,'file':str(font),'sha256':digest(font),
                     'index':0,'cjk':False,'weight':400}]}
    run(f.job,provider=f.actor,review_provider=f.actor,compiler=f.compiler,office=f.office,fonts=fonts)
    graph=SceneGraph.model_validate_json((f.job/'page-01/barrier-1.json').read_text('utf8'))
    return f,graph,fonts,font


def test_verified_seed_copies_without_mutating_prior(seed):
    f,graph,fonts,_=seed
    receipt=load(f.job,graph,fonts)
    original=Path(receipt['source_ir']);old=digest(original.parent/'source-plate.png')
    copied=materialize(receipt,f.job/'expanded')
    assert digest(copied)==digest(original) and len(receipt['regions'])==2
    (copied.parent/'source-plate.png').write_bytes(b'changed copy')
    assert digest(original.parent/'source-plate.png')==old


def test_verified_seed_rejects_changed_content_or_owner(seed):
    f,graph,fonts,_=seed
    changed=graph.model_copy(deep=True);changed.text.texts[0].text+='x'
    with pytest.raises(ValueError,match='content|text|typography'):
        load(f.job,changed,fonts)
    changed=graph.model_copy(deep=True);changed.ownership['a']='AMBIGUOUS'
    with pytest.raises(ValueError,match='owner'):
        load(f.job,changed,fonts)


def test_verified_seed_rejects_changed_font_bytes(seed):
    f,graph,fonts,font=seed
    font.write_bytes(b'changed font')
    with pytest.raises(ValueError,match='font bytes'):
        load(f.job,graph,fonts)
