"""Source component completion retains the original owner and neighbor gates."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from product.object_contracts import ClosedObject
from product.parallel_rebuild import source_owner
from product.recovery.asset_retry import complete_source_halo


def evidence(pixel_x=46, value=(242,243,253)):
    source=np.full((80,80,3),(235,239,249),np.uint8)
    source[25:40,25:45]=[60,90,170]
    source[28,pixel_x]=value
    seed=np.zeros((80,80,4),np.uint8)
    seed[25:40,25:45,:3]=source[25:40,25:45]
    seed[25:40,25:45,3]=255
    obj=ClosedObject(id='icon',description='whole source icon',prompt_en='whole source icon',
        kind='icon',bbox=[20,20,35,25],layers=['layer.png'],components=['whole'],
        text_policy='exclude_independent_text',parts=[{'name':'whole','bbox':[20,20,35,25],'layers':['layer.png']}])
    return source,seed,obj,np.zeros((80,80),bool)


def test_source_halo_supplies_missing_segmentation_without_relaxing_owner():
    source,seed,obj,protected=evidence()
    with pytest.raises(ValueError,match='no complete candidate owner'):
        source_owner(obj,seed,source,protected,None)
    result,report=complete_source_halo(obj,seed,source,protected,None)
    _,_,mask,_=source_owner(obj,result,source,protected,None)
    assert report['added_pixels']==1 and mask[28,46]
    assert np.array_equal(result[seed[...,3]>0],seed[seed[...,3]>0])
    assert report['source_owner_gate_relaxed'] is False


@pytest.mark.parametrize('pixel_x,value',[(52,(242,243,253)),(46,(100,100,100))])
def test_distant_or_strong_unowned_source_signal_remains_blocked(pixel_x,value):
    source,seed,obj,protected=evidence(pixel_x,value)
    result,report=complete_source_halo(obj,seed,source,protected,None)
    assert report['added_pixels']==0
    with pytest.raises(ValueError,match='no complete candidate owner'):
        source_owner(obj,result,source,protected,None)


def test_protected_source_component_is_never_added():
    source,seed,obj,protected=evidence()
    protected[28,46]=True
    result,report=complete_source_halo(obj,seed,source,protected,None)
    assert report['added_pixels']==0 and result[28,46,3]==0
