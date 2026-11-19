import numpy as np
import pytest
from PIL import Image
from types import SimpleNamespace

from product.runtime.text_reading import authorize_readings, quantity_changed
from product.recovery.layered_card import contained_peers, frame_layers
from product.materialize import digest


def scene(text):
    return SimpleNamespace(source_sha256='source',text=SimpleNamespace(texts=[SimpleNamespace(id='t',text=text)]))


def test_card_keeps_verified_photo_hole_and_rejects_crossing_photo(tmp_path):
    box=[10,10,120,90]
    photo=[22,40,36,25]
    mask=np.zeros((130,160),bool)
    mask[40:65,22:58]=True
    f=tmp_path/'owner.npz'
    np.savez_compressed(f,mask=mask)
    child={'kind':'asset','bbox':photo,'move_safety':'MOVABLE','owner_evidence':{'file':f.name,'sha256':digest(f)}}
    assert contained_peers(box,[child])
    assert not contained_peers(box,[{**child,'bbox':[4,40,36,25]}])
    assert not contained_peers(box,[{**child,'move_safety':'MOVE_UNSAFE'}])
    patch=np.full((90,120,3),245,np.uint8)
    patch[60:75,15:100]=[20,60,140]
    result=frame_layers(patch, np.full_like(patch,248),box,[child],tmp_path,[])
    assert result is not None
    rgba=result[-1]
    assert not rgba[30:55,12:48,3].any()
    assert rgba[65,20,3]==255


@pytest.mark.parametrize('old,new',[('26项','26套'),('30%','30‰'),('8万元','8元'),
                                    ('1.5 kg','15 kg'),('二十公里','二十公斤')])
def test_conflicting_quantities_are_never_written_as_guesses(old,new):
    assert quantity_changed(old,new)


def test_tentative_reading_requires_two_crops_and_a_ranked_choice():
    data={'object_id':'t','before':'资料整里','after':'资料整理','status':'pending_confirmation',
          'alternatives':['资料整里','资料整理'],'context_sha256':'b','crop_sha256':'a','choice_receipt':'choice'}
    authorize_readings(scene(data['before']),scene(data['after']),{'source_sha256':'source','changes':[data]})
    with pytest.raises(ValueError):
        authorize_readings(scene(data['before']),scene(data['after']),{'source_sha256':'source','changes':[{**data,'choice_receipt':None}]})
