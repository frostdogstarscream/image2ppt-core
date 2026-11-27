"""Synthetic contract checks only; no weights, GPU or paid provider calls."""
import copy
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'services/vision/src'))
from image2ppt_vision.layered.sam3_worker import prompt_box, pixel_box, restore_crop
from image2ppt_vision.layered.sam3_masks import select, guide
from image2ppt_vision.layered.plan_schema import validate_plan


def test_prompt_coordinates_are_normalized_center_not_xyxy():
    assert prompt_box([100,200,500,800]) == [.3,.5,.4,.6]


def test_crop_predictions_restore_to_source_coordinates_without_any_spill():
    box = pixel_box([101, 201, 499, 799], (201, 101))
    assert box == [20, 20, 101, 81]
    local = np.ones((2, 61, 81), bool)
    masks, boxes = restore_crop(local, np.array([[-2., -3., 90., 70.], [1., 2., 3., 4.]]), box, (201, 101))
    assert masks.shape == (2, 101, 201) and masks.sum() == local.sum()
    assert not masks[:, :20].any() and not masks[:, 81:].any()
    assert not masks[:, :, :20].any() and not masks[:, :, 101:].any()
    assert boxes.tolist() == [[20., 20., 101., 81.], [21., 22., 23., 24.]]
    empty, _ = restore_crop(np.zeros((0,61,81),bool),np.zeros((0,4)),box,(201,101))
    assert empty.shape == (0,101,201)


def test_hard_box_is_not_expanded_to_include_crossing_text():
    from image2ppt_vision.layered import plan_first
    rgb=np.full((100,200,3),255,np.uint8)
    payload={'schema':'LayeredPlan/1','source_sha256':'source','image_size':[200,100],
      'plan':{'schema':'SegmentationPlan/2','summary':'','risks':[],'regions':[
        {'id':'G1','kind':'graphic','action':'keep_pixels','bbox':[100,200,400,800],
         'label':'','reason':'','parent_id':None,'sam_prompt_en':'blue icon','segmentation_mode':'object'}]}}
    line={'bbox':[75,30,12,10]}
    _,_,soft=plan_first.guide(rgb,[line],[],payload,'source')
    _,_,hard=plan_first.guide(rgb,[line],[],payload,'source',hard_boxes=True)
    assert soft['regions'][0]['bbox'] == [20,20,67,60]
    assert hard['regions'][0]['bbox'] == [20,20,60,60]
    assert hard['regions'][0]['status'] == 'fallback'
    assert 'text_crosses_boundary' in hard['regions'][0]['reasons']


def test_hard_box_consumer_rejects_a_hash_bound_mask_outside_original_roi(tmp_path):
    import json,hashlib
    from image2ppt_vision.layered import plan_first
    payload={'schema':'LayeredPlan/1','source_sha256':'source','image_size':[200,100],
      'plan':{'schema':'SegmentationPlan/2','summary':'','risks':[],'regions':[
        {'id':'G1','kind':'graphic','action':'keep_pixels','bbox':[100,200,400,800],
         'label':'','reason':'','parent_id':None,'sam_prompt_en':'blue icon','segmentation_mode':'object'}]}}
    folder=tmp_path/'fp16';folder.mkdir()
    raw=np.zeros((1,100,200),bool);raw[0,20:80,20:80]=True;raw[0,50,80]=True
    np.savez_compressed(folder/'G1.npz',masks=raw,scores=np.array([.9]))
    result={'source_sha256':'source','image_size':[200,100],
      'plan_payload_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
      'box_policy':'crop-hard','precision':'fp16','checkpoint_sha256':'test',
      'regions':[{'id':'G1','mask_file':'G1.npz','sha256':plan_first.sha(folder/'G1.npz'),
                  'segmentation_mode':'object','crop_bbox_px':[20,20,80,80]}]}
    (tmp_path/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='hard box violation'):
        guide(np.full((100,200,3),255,np.uint8),[],[],payload,'source',tmp_path)


def test_identical_remote_icon_is_rejected_and_composite_unions_local_parts():
    masks=np.zeros((3,100,200),bool)
    masks[0,20:40,20:40]=True;masks[1,20:40,120:140]=True;masks[2,50:65,25:40]=True
    scores=np.array([.8,.99,.9])
    mask,indices=select(masks,scores,[10,10,60,70],'object')
    assert indices==[0] and not mask[:,100:].any()
    mask,indices=select(masks,scores,[10,10,60,70],'composite')
    assert set(indices)=={0,2} and mask.sum()==625


def test_small_mask_hole_does_not_leave_duplicate_native_object(tmp_path):
    from image2ppt_vision.layered.sam3_masks import install
    from image2ppt_vision.layered.review import _hard_rect
    (tmp_path/'masks').mkdir()
    old=_hard_rect([20,20,40,40],100,100);old.update(id='el-000',kind='native_shape')
    mask=old['mask'].copy();mask[30:32,30:32]=False
    report={'regions':[{'id':'G1','status':'guided','sam3_status':'accepted_mask'}]}
    elements,_=install([old],report,{'G1':mask},100,100,tmp_path)
    assert len(elements)==1 and elements[0]['kind']=='raster'
    assert report['regions'][0]['replaced_native_ids']==['el-000']


def test_v1_remains_valid_but_v2_requires_prompt_and_mode():
    plan={'schema':'SegmentationPlan/1','summary':'','risks':[],'regions':[
        {'id':'G1','kind':'graphic','action':'keep_pixels','bbox':[100,100,400,500],
         'label':'','reason':'','parent_id':None}]}
    validate_plan(plan);plan['schema']='SegmentationPlan/2'
    with pytest.raises(ValueError,match='English'):validate_plan(plan)
    plan['regions'][0]['sam_prompt_en']='blue shield'
    with pytest.raises(ValueError,match='mode'):validate_plan(plan)
    plan['regions'][0]['segmentation_mode']='object';validate_plan(plan)


def test_result_binding_is_checked_before_reading_masks(tmp_path):
    import json
    plan={'schema':'LayeredPlan/1','source_sha256':'source','image_size':[200,100],
      'plan':{'schema':'SegmentationPlan/2','summary':'','risks':[],'regions':[
        {'id':'T1','kind':'text','action':'ocr','bbox':[100,100,400,500],'label':'','reason':'','parent_id':None}]}}
    (tmp_path/'result.json').write_text(json.dumps({'source_sha256':'wrong','image_size':[200,100],'plan_payload_sha256':'wrong'}))
    with pytest.raises(ValueError,match='binding'):guide(np.full((100,200,3),255,np.uint8),[],[],plan,'source',tmp_path)


@pytest.mark.parametrize('provider_ok,expected_calls', [(False,1),(True,2)])
def test_batch_reserves_before_call_and_only_retries_schema(tmp_path,monkeypatch,provider_ok,expected_calls):
    import json
    import importlib.util
    scripts=Path(__file__).resolve().parents[1]/'scripts';monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('sam3_batch_test',scripts/'plan_sam3_batch.py')
    batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)
    source=tmp_path/'page.png';source.write_bytes(b'test');out=tmp_path/'batch'
    calls=[]
    def fake(args):
        ledger=json.loads((out/'batch.json').read_text())
        assert len(ledger['requests'])==len(calls)+1
        calls.append(args);p=Path(args.output);p.mkdir(parents=True)
        (p/'job.json').write_text(json.dumps({'status':'FAILED'}))
        (p/'response.json').write_text(json.dumps({'ok':provider_ok}))
    monkeypatch.setattr(batch,'run',fake)
    monkeypatch.setattr(sys,'argv',['batch','--inputs',str(source),'--output',str(out),'--model-config','unused','--model','qwen/qwen3.8-max'])
    assert batch.main()==2 and len(calls)==expected_calls
    with pytest.raises(FileExistsError):batch.main()
