import json
import numpy as np
from PIL import Image
from product.materialize import write,digest
from product.parallel_residual_gate import inspect_residuals


def test_actual_delete_residual_is_separate_from_outside_gate(tmp_path):
    source=np.full((20,30,3),255,np.uint8);source[5:10,8:13]=0
    Image.fromarray(source).save(tmp_path/'source.png');mask=np.zeros((20,30),bool);mask[5:10,8:13]=True
    np.savez_compressed(tmp_path/'owner.npz',mask=mask,restore_bbox=[8,5,5,5],backing_patch=np.full((5,5,3),255,np.uint8))
    write(tmp_path/'scene.json',{'source_sha256':digest(tmp_path/'source.png'),'selected':{'t':{'id':'t','owner_evidence':{'file':'owner.npz','sha256':digest(tmp_path/'owner.npz')}}}})
    Image.fromarray(source).save(tmp_path/'01-t-delete.png')
    failed=inspect_residuals(tmp_path/'slideir.json',tmp_path,1,tmp_path/'fail.json')[0]
    assert failed['residual_check_status']=='FAIL' and failed['residual_pixels_confirmed']==25
    Image.new('RGB',(30,20),'white').save(tmp_path/'01-t-delete.png')
    passed=inspect_residuals(tmp_path/'slideir.json',tmp_path,1,tmp_path/'pass.json')[0]
    assert passed['residual_check_status']=='PASS' and passed['residual_pixels_confirmed']==0


def test_missing_owner_evidence_is_unknown(tmp_path):
    Image.new('RGB',(30,20),'white').save(tmp_path/'source.png')
    write(tmp_path/'scene.json',{'source_sha256':digest(tmp_path/'source.png'),'selected':{'t':{'id':'t'}}})
    r=inspect_residuals(tmp_path/'slideir.json',tmp_path,1,tmp_path/'report.json')[0]
    assert r['residual_check_status']=='UNKNOWN' and r['residual_pixels_confirmed'] is None
