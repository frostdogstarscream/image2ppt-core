"""Synthetic evidence-binding fixtures, not actual Office acceptance."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from product.materialize import digest,write
from product.object_contracts import ClosedObject
from product.recovery.stable_assets import verified_candidate
from test_parallel import branches
from product.parallel_fusion import fuse


@pytest.fixture
def stable(tmp_path):
    source=np.full((50,80,3),245,np.uint8);source[10:25,10:25]=[0,50,180]
    image=tmp_path/'source.png';Image.fromarray(source).save(image)
    snap,b=branches(image)
    obj=ClosedObject(id='icon',description='icon',prompt_en='icon',bbox=[12,12,10,10],layers=['layer-01.png'],
        text_policy='exclude_independent_text',kind='icon',components=['icon'],
        parts=[{'name':'icon','bbox':[12,12,10,10],'layers':['layer-01.png']}])
    b['asset'].objects=[obj];graph=fuse(snap,b,{'lines':[]})
    rgba=np.zeros((18,18,4),np.uint8);rgba[1:16,1:16]=[0,50,180,255]
    png=tmp_path/'icon.png';Image.fromarray(rgba).save(png)
    mask=np.zeros((50,80),bool);mask[10:25,10:25]=True
    owner=tmp_path/'icon-owner.npz'
    np.savez_compressed(owner,mask=mask,restore_bbox=[9,9,18,18],backing_patch=np.full((18,18,3),245,np.uint8))
    graph.selected['icon']={'id':'icon','kind':'asset','bbox':[9,9,18,18],'source_sha256':snap['source_sha256'],
        'complete_owner_scope':True,'payload':{'method':'alpha'},
        'mask_sha256':hashlib.sha256(mask.astype('uint8').tobytes()).hexdigest(),
        'owner_evidence':{'file':owner.name,'sha256':digest(owner)},
        'asset_reference':{'path':png.name,'sha256':digest(png)}}
    scene=tmp_path/'scene.json';write(scene,graph.model_dump())
    pptx=tmp_path/'synthetic.pptx';pptx.write_bytes(b'SYNTHETIC FIXTURE ONLY')
    acceptance=tmp_path/'synthetic-acceptance.json'
    write(acceptance,{'synthetic_fixture':True,'source_sha256':snap['source_sha256'],
        'artifacts':{'pptx':str(pptx),'pptx_sha256':digest(pptx)},
        'office':{'status':'PASS_APP_AUTOMATED','pptx_sha256':digest(pptx),'records':[
            {'id':'icon','page':1,'action':action,'saved_reopened':True,'outside_pixels':0} for action in ('move-scale','delete')]},
        'owner_residual':[{'id':'icon','residual_check_status':'PASS','residual_pixels_confirmed':0,
                          'unexplained_pixels':0,'reference_sha256':digest(owner)}]})
    record={'id':'icon','path':str(png),'sha256':digest(png),'bbox':[9,9,18,18],
        'verified_replacement':{name:{'path':str(path),'sha256':digest(path)} for name,path in
            [('scene',scene),('owner',owner),('acceptance',acceptance)]}}
    return obj,source,record


def test_verified_crop_can_differ_from_unchanged_logical_search_box(stable):
    obj,source,record=stable
    result=verified_candidate(obj,source,np.zeros(source.shape[:2],bool),[record])
    assert result['bbox']==[9,9,18,18] and result['mask'].sum()==225
    assert result['payload']['original_sha256']==record['sha256']


def test_current_protected_peer_still_vetoes_stable_owner(stable):
    obj,source,record=stable;protected=np.zeros(source.shape[:2],bool);protected[12,12]=True
    with pytest.raises(ValueError,match='protected peer'):
        verified_candidate(obj,source,protected,[record])


def test_source_change_and_semantic_change_require_revalidation(stable):
    obj,source,record=stable
    assert verified_candidate(obj.model_copy(update={'bbox':[13,12,10,10]}),source,
                              np.zeros(source.shape[:2],bool),[record]) is None
    changed=source.copy();changed[0,0,0]^=1
    with pytest.raises(ValueError,match='source changed'):
        verified_candidate(obj,changed,np.zeros(source.shape[:2],bool),[record])


def test_tampered_office_evidence_cannot_reuse_stable_owner(stable):
    obj,source,record=stable
    path=record['verified_replacement']['acceptance']['path']
    with open(path,'a',encoding='utf8') as handle:handle.write(' ')
    with pytest.raises(ValueError,match='binding mismatch'):
        verified_candidate(obj,source,np.zeros(source.shape[:2],bool),[record])
