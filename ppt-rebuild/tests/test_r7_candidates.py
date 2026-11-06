import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from relation_validator import prepare_spec,digest
from r7_candidates import build_candidate


REVIEW_ROOT = os.environ.get('IMAGE2PPT_R6_REVIEW_ROOT')
REPO = Path(__file__).resolve().parents[2]
REVIEW_DIRS = [root/'outputs/review-4090-20261106-r6/delivery' for root in [REPO, *REPO.parents]]
SAMPLES = (Path(REVIEW_ROOT) if REVIEW_ROOT else next(
    (path for path in REVIEW_DIRS if (path/'pages').is_dir()), REVIEW_DIRS[0]))/'pages'


@pytest.mark.parametrize('page', ['02','08'])
@pytest.mark.parametrize('raster', [False, True])
def test_repair_keeps_baseline_text_assets_and_relative_order(tmp_path,page,raster):
    original=SAMPLES/page/'revision/slideir.json'
    if not original.exists():
        if REVIEW_ROOT: pytest.fail('Configured IMAGE2PPT_R6_REVIEW_ROOT lacks baseline: '+str(original))
        pytest.skip('Private r6 baseline is not installed')
    before_hash=digest(original)
    before=json.loads(original.read_text(encoding='utf-8'))
    spec=prepare_spec(page,original.parent/before['source_image'],tmp_path/'spec')
    result=build_candidate(original,spec,tmp_path/'candidate',raster_ids=[x['id'] for x in spec['elements']] if raster else [])
    after=json.loads(Path(result['ir']).read_text(encoding='utf-8'))
    assert not result['registry']['conflicts']
    assert result['registry']['schema']=='ObjectRegistry/2'
    assert before_hash==digest(original)
    removed=set(result['removed_ids']);added=set(result['added_ids'])
    assert [n['id'] for n in before['nodes'] if n['id'] not in removed]==[n['id'] for n in after['nodes'] if n['id'] not in added]
    after_map={n['id']:n for n in after['nodes']}
    for old in before['nodes']:
        if old['id'] in removed:continue
        a=copy.deepcopy(old);b=copy.deepcopy(after_map[a['id']]);a.pop('z');b.pop('z')
        assert a==b
    assert before['groups']==after['groups']
    for asset in before['assets']:
        assert digest(original.parent/asset['path'])==digest(Path(result['ir']).parent/asset['path'])
    if raster and page=='02':
        label=np.asarray(Image.open(tmp_path/'candidate/assets/s02-sidebar-label.png'))
        opaque=label[:,:,3]>0
        assert opaque.sum()>10000
        assert not (label[opaque,:3].min(axis=1)>220).any(), 'Label PNG must never contain white source letters'


def test_unknown_raster_id_rejected(tmp_path):
    original=SAMPLES/'02/revision/slideir.json'
    if not original.exists():
        if REVIEW_ROOT: pytest.fail('Configured IMAGE2PPT_R6_REVIEW_ROOT lacks baseline: '+str(original))
        pytest.skip('Private baseline not installed')
    before=json.loads(original.read_text(encoding='utf-8'))
    spec=prepare_spec('02',original.parent/before['source_image'],tmp_path/'spec')
    with pytest.raises(ValueError,match='Unknown raster'):
        build_candidate(original,spec,tmp_path/'candidate',raster_ids=['not-reviewed'])
