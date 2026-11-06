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
from object_resolver import RepairAuthorizationError, authorization_digest, mask_ref
from reconstruction_transaction import scene_fingerprint


REVIEW_ROOT = os.environ.get('IMAGE2PPT_R6_REVIEW_ROOT')
REPO = Path(__file__).resolve().parents[2]
REVIEW_DIRS = [root/'outputs/review-4090-20261106-r6/delivery' for root in [REPO, *REPO.parents]]
SAMPLES = (Path(REVIEW_ROOT) if REVIEW_ROOT else next(
    (path for path in REVIEW_DIRS if (path/'pages').is_dir()), REVIEW_DIRS[0]))/'pages'


@pytest.mark.parametrize('page', ['02','08'])
@pytest.mark.parametrize('raster', [False, True])
def test_private_unproven_neighbors_block_grant_without_changing_baseline(tmp_path,page,raster):
    original=SAMPLES/page/'revision/slideir.json'
    if not original.exists():
        if REVIEW_ROOT: pytest.fail('Configured IMAGE2PPT_R6_REVIEW_ROOT lacks baseline: '+str(original))
        pytest.skip('Private r6 baseline is not installed')
    before_hash=digest(original)
    before=json.loads(original.read_text(encoding='utf-8'))
    spec=prepare_spec(page,original.parent/before['source_image'],tmp_path/'spec')
    with pytest.raises(RepairAuthorizationError) as error:
        build_candidate(original,spec,tmp_path/'candidate',tx_id='test-private',
                        raster_ids=[x['id'] for x in spec['elements']] if raster else [])
    assert error.value.status=='REVIEW_REQUIRED'
    assert any(c['type']=='UNKNOWN_AFFECTED_PROTECTION' for c in error.value.conflicts)
    after=json.loads((tmp_path/'candidate/slideir.json').read_text(encoding='utf-8'))
    assert before_hash==digest(original)
    removed={n['id'] for n in before['nodes']}-{n['id'] for n in after['nodes']}
    added={n['id'] for n in after['nodes']}-{n['id'] for n in before['nodes']}
    assert [n['id'] for n in before['nodes'] if n['id'] not in removed]==[n['id'] for n in after['nodes'] if n['id'] not in added]
    after_map={n['id']:n for n in after['nodes']}
    for old in before['nodes']:
        if old['id'] in removed:continue
        a=copy.deepcopy(old);b=copy.deepcopy(after_map[a['id']]);a.pop('z');b.pop('z')
        assert a==b
    assert before['groups']==after['groups']
    for asset in before['assets']:
        assert digest(original.parent/asset['path'])==digest(tmp_path/'candidate'/asset['path'])
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


def isolated_fixture(tmp_path, *, neighbor=False):
    parent=tmp_path/'parent';parent.mkdir()
    source=np.full((60,100,3),255,np.uint8)
    source[10:35,10:50]=[20,80,140]
    source[18:24,20:26]=255
    Image.fromarray(source).save(parent/'source.png')
    ir={'canvas':{'source_px':[100,60]},'source_image':'source.png',
        'source_sha256':digest(parent/'source.png'),'background':{'color':'FFFFFF'},
        'assets':[],'evidence':[{'id':'ev-title','kind':'text','bbox':[65,42,25,10]}],
        'nodes':[{'id':'title','type':'text','bbox':[65,42,25,10],'z':1,'text':'unchanged',
                  'evidence_ids':['ev-title']}],'groups':[]}
    if neighbor:
        ir['nodes'].append({'id':'neighbor','type':'shape','bbox':[10,10,40,25],
                            'geometry':'rect','fill':'FFFFFF','z':0,'evidence_ids':[]})
    irfile=parent/'slideir.json';irfile.write_text(json.dumps(ir),encoding='utf-8')
    local=source[10:35,10:50]
    ink=local[:,:,0]<100;shape=np.ones_like(ink)
    references={}
    for role,mask in [('source_ink',ink),('silhouette',shape)]:
        path=tmp_path/(role+'.png');Image.fromarray(mask.astype('uint8')*255).save(path)
        references[role]={'path':str(path),'sha256':digest(path),'size_px':[40,25],'origin_px':[10,10]}
    spec={'page':'custom','source_sha256':digest(parent/'source.png'),'canvas_px':[100,60],
          'elements':[{'id':'label','kind':'label','bbox_xyxy':[10,10,50,35],
                       'repairable':True,'masks':references},
                      {'id':'unselected-unsupported','kind':'ICON_PRESENCE','repairable':False}]}
    return irfile,spec


@pytest.mark.parametrize('raster',[False,True])
def test_subset_repairs_bind_actual_transaction_without_processing_unrelated_spec(tmp_path,raster):
    original,spec=isolated_fixture(tmp_path)
    before=scene_fingerprint(original)
    result=build_candidate(original,spec,tmp_path/'candidate',object_ids=['label'],
                           tx_id='repair-one',parent_scene_sha256=before['sha256'],
                           raster_ids=['label'] if raster else [])
    assert scene_fingerprint(original)==before
    assert result['added_ids']==['label'] and result['removed_ids']==[]
    grant=result['authorization']
    assert grant['tx_id']=='repair-one' and grant['object_ids']==['label']
    assert grant['parent_scene_sha256']==before['sha256']
    assert grant['authorization_sha256']==authorization_digest(grant)
    assert grant['evidence_files'][result['authorized_mask']]==digest(result['authorized_mask'])
    assert json.loads(Path(result['authorization_path']).read_text(encoding='utf-8'))==grant
    if raster:
        pixels=np.asarray(Image.open(tmp_path/'candidate/assets/label.png'))
        assert not (pixels[pixels[:,:,3]>0,:3].min(axis=1)>220).any()


def test_empty_selection_is_noop_without_creating_output(tmp_path):
    original,spec=isolated_fixture(tmp_path)
    result=build_candidate(original,spec,tmp_path/'not-created',object_ids=[])
    assert result['action']=='NO_OP' and result['ir']==str(original.resolve())
    assert not (tmp_path/'not-created').exists()


def test_existing_stable_target_is_retained_without_duplicates(tmp_path):
    original,spec=isolated_fixture(tmp_path)
    result=build_candidate(original,spec,tmp_path/'candidate',object_ids=['label'],tx_id='first')
    before=scene_fingerprint(result['ir'])
    with pytest.raises(RepairAuthorizationError,match='unsupported_existing_object_replacement') as error:
        build_candidate(result['ir'],spec,tmp_path/'second',object_ids=['label'],tx_id='second')
    assert error.value.status=='REVIEW_REQUIRED'
    assert scene_fingerprint(result['ir'])==before
    assert not (tmp_path/'second').exists()


def test_unknown_neighbor_remains_review_required(tmp_path):
    original,spec=isolated_fixture(tmp_path,neighbor=True)
    with pytest.raises(RepairAuthorizationError) as error:
        build_candidate(original,spec,tmp_path/'candidate',object_ids=['label'],tx_id='test')
    assert error.value.status=='REVIEW_REQUIRED'
    assert any(c['type']=='UNKNOWN_AFFECTED_PROTECTION' for c in error.value.conflicts)


def test_supplied_local_protection_evidence_is_checked_and_bound(tmp_path):
    original,spec=isolated_fixture(tmp_path,neighbor=True)
    # Synthetic independent evidence says this background-shaped neighbor has
    # no protected contribution in the affected region. Never infer this in production.
    scope=np.zeros((60,100),np.uint8);scope[10:35,10:50]=255
    scopepath=tmp_path/'scope.png';Image.fromarray(scope).save(scopepath)
    protectpath=tmp_path/'protect.png';Image.fromarray(np.zeros_like(scope)).save(protectpath)
    spec['local_protection']={'neighbor':{'method':'synthetic_independent_local_plate',
       'source_sha256':spec['source_sha256'],'scope':mask_ref(scopepath,[100,60]),
       'protection':mask_ref(protectpath,[100,60])}}
    result=build_candidate(original,spec,tmp_path/'candidate',object_ids=['label'],tx_id='test')
    assert str(scopepath.resolve()) in result['authorization']['evidence_files']


def test_actual_transaction_and_parent_are_required_before_mutation(tmp_path):
    original,spec=isolated_fixture(tmp_path)
    with pytest.raises(ValueError,match='Actual transaction'):
        build_candidate(original,spec,tmp_path/'candidate',object_ids=['label'])
    with pytest.raises(ValueError,match='parent scene'):
        build_candidate(original,spec,tmp_path/'candidate',object_ids=['label'],tx_id='test',
                        parent_scene_sha256='wrong')
    assert not (tmp_path/'candidate').exists()
