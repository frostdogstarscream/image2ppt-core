import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from product.object_extract import component_support, visible_rgba, extract, load_sam
from product.object_contracts import ObjectPlan
from product.materialize import digest, write


def fixture(tmp_path,*,text=False,ambiguous=False):
    d=tmp_path/'layers';d.mkdir();base=tmp_path/'base';base.mkdir()
    bg=np.full((40,60,4),255,np.uint8);fg=np.zeros_like(bg)
    fg[10:20,10:20]=[20,50,180,255];fg[10:20,19]=[20,50,180,90]
    fg[23:25,12:17]=[20,50,180,255]
    if ambiguous:fg[0:40,19]=[20,50,180,255]
    Image.fromarray(bg).save(d/'layer-00.png');Image.fromarray(fg).save(d/'layer-01.png')
    source=Image.alpha_composite(Image.fromarray(bg),Image.fromarray(fg)).convert('RGB');source.save(d/'source.png');source.save(base/'source.png');source.crop((8,8,23,28)).save(base/'crop.png')
    write(d/'manifest.json',{'layer_order':'back_to_front','image_size':[60,40],'source_sha256':digest(d/'source.png'),'layers':[{'file':f'layer-0{i}.png','z':i,'size':[60,40],'sha256':digest(d/f'layer-0{i}.png')} for i in range(2)]})
    nodes=[{'id':'old','type':'image','bbox':[8,8,15,20],'z':0,'asset_id':'oldasset','evidence_ids':[]}]
    if text:nodes.append({'id':'t','type':'text','text':'X','bbox':[10,10,6,6],'z':1})
    write(base/'slideir.json',{'canvas':{'source_px':[60,40]},'source_image':'source.png','nodes':nodes,'assets':[{'id':'oldasset','path':'crop.png','contains_evidence_ids':['old-e']}],'evidence':[{'id':'old-e'}],'groups':[]})
    p={'schema':'LogicalObjects/1','source_sha256':digest(d/'source.png'),'image_size':[60,40],'objects':[{'id':'icon','description':'blue icon','prompt_en':'blue icon','bbox':[8,8,15,20],'layers':['layer-01.png'],'text_policy':'exclude_independent_text','components':['square','detached underline'],'replaces':['old']}],'limitations':[]}
    write(tmp_path/'plan.json',p)
    return d,base,tmp_path/'plan.json'


def test_components_whole_not_rectangle():
    a=np.zeros((20,30),np.uint8);a[4:9,4:12]=255;a[13:15,6:9]=255
    mask,bad=component_support(a,[3,3,8,13],(30,20))
    assert mask[4,11] and mask[13,6] and not mask[11,7] and not bad


def test_oversize_component_is_not_clipped():
    a=np.full((20,30),255,np.uint8);mask,bad=component_support(a,[3,3,8,13],(30,20))
    assert not mask.any() and bad


def test_visible_rgba_preserves_alpha_and_occlusion():
    bottom=np.array([[[100,0,0,128],[0,100,0,255]]],np.uint8)
    top=np.array([[[0,0,100,128],[0,0,100,255]]],np.uint8)
    result=visible_rgba([bottom,top],[np.ones((1,2),bool),np.zeros((1,2),bool)])
    assert result[0,0,3]==64 and result[0,1,3]==0
    assert tuple(result[0,0,:3])==(100,0,0)


def test_atomic_replacement_and_transparency(tmp_path):
    d,base,plan=fixture(tmp_path);out,report=extract(plan,d,base/'slideir.json',tmp_path/'out')
    ir=json.loads(out.read_text());assert report['extracted']==1
    assert [n['id'] for n in ir['nodes']]==['extracted-icon']
    assert ir['assets'][0]['id']=='extracted-icon'
    image=np.array(Image.open(out.parent/'object-icon.png'));assert 90 in image[...,3] and 0 in image[...,3]


def test_native_text_conflict_rolls_back_whole_object(tmp_path):
    d,base,plan=fixture(tmp_path,text=True);out,report=extract(plan,d,base/'slideir.json',tmp_path/'out')
    assert report['fallback']==1 and report['objects'][0]['reasons']==['protected_native_text_overlap']
    original=json.loads((base/'slideir.json').read_text());actual=json.loads(out.read_text())
    assert actual['nodes']==original['nodes'] and actual['assets']==original['assets']


def test_unavailable_sam_never_claims_success(tmp_path):
    d,base,plan=fixture(tmp_path,ambiguous=True);out,report=extract(plan,d,base/'slideir.json',tmp_path/'out',route='sam3.1')
    assert report['extracted']==0 and report['objects'][0]['reasons']==['sam_not_configured']


def test_source_hash_fails_before_replacement(tmp_path):
    d,base,plan=fixture(tmp_path);p=json.loads(plan.read_text());p['source_sha256']='0'*64;write(plan,p)
    with pytest.raises(ValueError,match='source mismatch'):extract(plan,d,base/'slideir.json',tmp_path/'out')


def test_duplicate_ownership_rejected(tmp_path):
    d,base,plan=fixture(tmp_path);p=json.loads(plan.read_text());p['objects'].append({**p['objects'][0],'id':'second'})
    with pytest.raises(ValueError,match='duplicate ownership'):ObjectPlan.model_validate(p)


def test_replay_does_not_mutate_original(tmp_path):
    d,base,plan=fixture(tmp_path);before=digest(base/'slideir.json')
    a,ra=extract(plan,d,base/'slideir.json',tmp_path/'a');b,rb=extract(plan,d,base/'slideir.json',tmp_path/'b')
    assert a.read_bytes()==b.read_bytes() and ra==rb and digest(base/'slideir.json')==before


def test_mask_metrics_do_not_claim_semantic_acceptance():
    from product.object_metrics import measure
    reference=np.zeros((12,12),bool);reference[2:10,5]=True
    actual=reference.copy();actual[7:10]=False
    result=measure(actual,reference,np.zeros_like(reference))
    assert result['thin_line_skeleton_recall']==5/8
    assert result['missing_pixels']==3 and not result['reference_verified'] and not result['semantic_acceptance']


def test_sam_mask_cache_rejects_wrong_plan_and_nonfinite(tmp_path):
    d,base,planfile=fixture(tmp_path);plan=ObjectPlan.model_validate(json.loads(planfile.read_text()))
    masks=tmp_path/'sam';masks.mkdir();file=masks/'raw.npz'
    np.savez_compressed(file,masks=np.ones((1,40,60)),scores=np.array([float('nan')]))
    meta={'status':'SUCCEEDED','source_sha256':plan.source_sha256,'plan_sha256':digest(planfile),'version':'sam3','image_size':[60,40],
          'mask_file':'raw.npz','mask_sha256':digest(file),'checkpoint_sha256':'a'*64,'code_revision':'b'*40}
    write(masks/'icon.json',meta)
    assert load_sam(masks,plan.objects[0],plan.source_sha256,'0'*64,(60,40),'sam3')[1]=='sam_binding_mismatch'
    assert load_sam(masks,plan.objects[0],plan.source_sha256,digest(planfile),(60,40),'sam3')[1]=='sam_invalid_mask'


def test_single_frame_box_uses_xywh_not_center_coordinates():
    import importlib.util
    file=Path(__file__).resolve().parents[2]/'segmentation/single_frame.py'
    spec=importlib.util.spec_from_file_location('single_frame',file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.normalize_box([20,10,40,30],(100,50))==[.2,.2,.4,.6]
    with pytest.raises(ValueError):module.normalize_box([90,10,40,30],(100,50))


def test_sam31_bridge_only_omits_disabled_unsupported_state_offload():
    import importlib.util
    file=Path(__file__).resolve().parents[2]/'segmentation/single_frame.py'
    spec=importlib.util.spec_from_file_location('sam_bridge',file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Model:
        def init_state(self,resource_path,offload_video_to_cpu=False):return resource_path,offload_video_to_cpu
    model=Model();assert module.bridge_disabled_state_offload(model)
    assert model.init_state(resource_path='fixture',offload_video_to_cpu=True,offload_state_to_cpu=False)==('fixture',True)
    with pytest.raises(ValueError):model.init_state(resource_path='fixture',offload_state_to_cpu=True)
    class Supported:
        def init_state(self,offload_state_to_cpu=False):return offload_state_to_cpu
    supported=Supported();assert not module.bridge_disabled_state_offload(supported)
    assert supported.init_state(offload_state_to_cpu=True)


def test_blank_crop_padding_is_not_another_object(tmp_path):
    d,base,plan=fixture(tmp_path)
    ir=json.loads((base/'slideir.json').read_text());ir['nodes'][0]['bbox']=[5,5,22,27];write(base/'slideir.json',ir)
    output,report=extract(plan,d,base/'slideir.json',tmp_path/'out')
    assert report['extracted']==1


def test_padding_with_unclaimed_foreground_rolls_back(tmp_path):
    d,base,plan=fixture(tmp_path)
    fg=np.array(Image.open(d/'layer-01.png'));fg[10:13,24:26]=[200,20,20,255];Image.fromarray(fg).save(d/'layer-01.png')
    bg=Image.open(d/'layer-00.png');src=Image.alpha_composite(bg,Image.fromarray(fg)).convert('RGB');src.save(d/'source.png');src.save(base/'source.png')
    meta=json.loads((d/'manifest.json').read_text());meta['source_sha256']=digest(d/'source.png');meta['layers'][1]['sha256']=digest(d/'layer-01.png');write(d/'manifest.json',meta)
    p=json.loads(plan.read_text());p['source_sha256']=digest(d/'source.png');write(plan,p)
    ir=json.loads((base/'slideir.json').read_text());ir['nodes'][0]['bbox']=[5,5,22,27];write(base/'slideir.json',ir)
    output,report=extract(plan,d,base/'slideir.json',tmp_path/'out')
    assert report['fallback']==1 and report['objects'][0]['reasons']==['replacement_contains_other_content']


def test_visual_rejection_rolls_back_entire_logical_object(tmp_path):
    d,base,plan=fixture(tmp_path)
    out,report=extract(plan,d,base/'slideir.json',tmp_path/'out',rejected={'icon':['thin line missing']})
    assert report['extracted']==0 and report['objects'][0]['reasons']==['independent_visual_review']
    assert json.loads(out.read_text())['nodes']==json.loads((base/'slideir.json').read_text())['nodes']


def test_visual_rejection_survives_an_earlier_extracted_member(tmp_path):
    d,base,planfile=fixture(tmp_path)
    plan=json.loads(planfile.read_text())
    plan['objects'].append({**plan['objects'][0],'id':'later-icon','bbox':[35,8,15,20],'replaces':[]})
    write(planfile,plan)
    out,report=extract(planfile,d,base/'slideir.json',tmp_path/'out',rejected={'later-icon':['review rejected this later member']})
    assert report['objects'][0]['status']=='EXTRACTED'
    assert report['objects'][1]['reasons']==['independent_visual_review']
    assert report['objects'][1]['review_findings']==['review rejected this later member']


def test_object_route_default_is_disabled_and_rejects_unknown(monkeypatch):
    from product.object_stage import route_setting
    monkeypatch.delenv('IMAGE2PPT_OBJECT_EXTRACTION',raising=False)
    assert route_setting()=='off'
    monkeypatch.setenv('IMAGE2PPT_OBJECT_EXTRACTION','sam3.1')
    assert route_setting()=='sam3.1' and route_setting('off')=='off'
    with pytest.raises(ValueError):route_setting('../other')
