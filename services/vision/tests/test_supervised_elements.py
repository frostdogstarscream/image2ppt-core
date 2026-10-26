from copy import deepcopy
import json
import pytest
from PIL import Image
from image2ppt_vision.supervised_elements import validate_element_plan,materialize_elements,parse_response,apply_element_patch,raster_text_conflicts
from image2ppt_vision.storage import digest
from image2ppt_vision.supervised_elements import normalize_element_coordinates

def fixture(tmp_path):
    source=tmp_path/'s.png';Image.new('RGB',(400,200),'white').save(source)
    inv={'source_sha256':digest(source),'image_size':[400,200],'texts':[{'id':'t1','text':'OCR','bbox':[100,50,100,25]}],'candidates':[]}
    plan={'schema':'ElementPlan/1','source_sha256':digest(source),'image_size':[400,200],'background_color':'FFFFFF','objects':[{'id':'card','kind':'native_shape','bbox':[20,20,300,130],'geometry':'roundRect','fill':'EEEEFF','z':0},{'id':'icon','kind':'raster_asset','bbox':[25,35,50,50],'mask':'ellipse','contains_text_ids':[],'z':1}], 'text_decisions':[{'id':'t1','text':'Corrected','action':'native','bbox':[100,50,100,25],'font_size_px':24,'color':'000000','bold':False,'z':2}], 'limitations':['synthetic fixture only']}
    return source,inv,plan

def test_reject_page_raster_and_duplicate_ocr(tmp_path):
    _,inv,p=fixture(tmp_path);p['objects'][1]['bbox']=[0,0,400,200]
    with pytest.raises(ValueError,match='large raster'):validate_element_plan(p,inv)
    _,inv,p=fixture(tmp_path);p['text_decisions'].append(deepcopy(p['text_decisions'][0]))
    with pytest.raises(ValueError,match='exactly once'):validate_element_plan(p,inv)

def test_reject_hidden_original_text_under_native(tmp_path):
    _,inv,p=fixture(tmp_path);p['objects'][1]['bbox']=[95,45,110,40]
    with pytest.raises(ValueError,match='duplicates raster'):validate_element_plan(p,inv)

def test_retained_text_requires_actual_owner(tmp_path):
    _,inv,p=fixture(tmp_path);p['text_decisions'][0]['action']='retained_in_asset'
    with pytest.raises(ValueError,match='retention mismatch'):validate_element_plan(p,inv)
    p['objects'][1].update(bbox=[95,45,110,40],contains_text_ids=['t1'],mask='rect')
    validate_element_plan(p,inv)

def test_materialization_preserves_source_and_masks_white_interior(tmp_path):
    source,inv,p=fixture(tmp_path);out=tmp_path/'out'
    r=materialize_elements(source,inv,p,out,model_id='fixture',response_sha256='0'*64)
    assert digest(source)==inv['source_sha256'] and r['native_shapes']==1 and not r['human_acceptance']
    with Image.open(out/'assets/icon.png') as im:
        assert im.getpixel((25,25))==(255,255,255,255) and im.getpixel((0,0))[3]==0
    ir=json.loads((out/'slideir.json').read_text(encoding='utf8'))
    assert ir['nodes'][-1]['text_override_reason'] and len(ir['evidence'])==3
    with pytest.raises(FileExistsError):materialize_elements(source,inv,p,out,model_id='fixture',response_sha256='0'*64)

def test_truncated_model_plan_never_materialized(tmp_path):
    f=tmp_path/'response.body';f.write_text(json.dumps({'choices':[{'finish_reason':'length','message':{'content':'{}'}}]}))
    with pytest.raises(ValueError,match='incomplete'):parse_response(f)

def test_patch_preserves_original_and_cannot_change_source(tmp_path):
    _,inv,p=fixture(tmp_path);before=deepcopy(p)
    patch={'schema':'ElementPatch/1','source_sha256':p['source_sha256'],'objects':[{'id':'card','fill':'FFFFFF'}]}
    result=apply_element_patch(p,patch)
    assert p==before and result['objects'][0]['fill']=='FFFFFF'
    validate_element_plan(result,inv)
    patch['source_sha256']='bad'
    with pytest.raises(ValueError,match='source mismatch'):apply_element_patch(p,patch)

def test_collects_all_false_ocr_icon_conflicts(tmp_path):
    _,inv,p=fixture(tmp_path);p['objects'][1]['bbox']=[95,45,110,40]
    conflicts=raster_text_conflicts(p,inv)
    assert len(conflicts)==1 and conflicts[0]['text_id']=='t1' and conflicts[0]['asset_id']=='icon'

def test_gradient_without_fallback_fill_is_canonicalized_without_mutating_response(tmp_path):
    _,inv,p=fixture(tmp_path);p['objects'][0].pop('fill');p['objects'][0]['fill_gradient']={'angle_deg':0,'stops':[{'offset':0,'color':'FFFFFF'},{'offset':100000,'color':'AABBCC'}]};p['objects'][0]['line']=None
    result=validate_element_plan(p,inv)
    assert result['objects'][0]['fill']=='FFFFFF' and 'line' not in result['objects'][0] and 'fill' not in p['objects'][0]


def test_move_legacy_raster_freezes_source_pixels_and_other_objects(tmp_path):
    source, inv, plan = fixture(tmp_path)
    with Image.open(source) as image:
        image.paste((255, 0, 0), (25, 35, 50, 60))
        image.save(source)
    inv['source_sha256'] = plan['source_sha256'] = digest(source)
    original = deepcopy(plan)
    patch = {'schema': 'ElementPatch/1', 'source_sha256': plan['source_sha256'],
             'objects': [{'id': 'icon', 'bbox': [200, 100, 60, 60]}]}
    moved = apply_element_patch(plan, patch)
    assert moved['objects'][1]['source_crop_bbox'] == [25, 35, 50, 50]
    assert moved['objects'][1]['dest_bbox'] == [200, 100, 60, 60]
    assert moved['objects'][0] == original['objects'][0] and plan == original
    for name, value in [('before', plan), ('after', moved)]:
        materialize_elements(source, inv, value, tmp_path / name, model_id='fixture', response_sha256='0'*64)
    assert digest(tmp_path/'before/assets/icon.png') == digest(tmp_path/'after/assets/icon.png')
    saved = json.loads((tmp_path/'after/element-plan.json').read_text(encoding='utf8'))
    assert saved['objects'][1]['source_crop_bbox'] == [25, 35, 50, 50]
    ir = json.loads((tmp_path/'after/slideir.json').read_text(encoding='utf8'))
    assert next(n for n in ir['nodes'] if n['id']=='icon')['bbox'] == [200, 100, 60, 60]
    cropped = apply_element_patch(moved, {'schema':'ElementPatch/1','source_sha256':plan['source_sha256'],
                                       'objects':[{'id':'icon','source_crop_bbox':[30,35,50,50]}]})
    materialize_elements(source,inv,cropped,tmp_path/'recrop',model_id='fixture',response_sha256='0'*64)
    assert digest(tmp_path/'recrop/assets/icon.png') != digest(tmp_path/'after/assets/icon.png')
    panel = apply_element_patch(moved, {'schema':'ElementPatch/1','source_sha256':plan['source_sha256'],
                                     'objects':[{'id':'card','bbox':[10,10,310,140]}]})
    assert panel['objects'][1] == moved['objects'][1]


def test_v3_boxes_font_contract_and_source_ownership(tmp_path):
    _, inv, plan = fixture(tmp_path)
    plan['schema'] = 'ElementPlan/3'
    with pytest.raises(ValueError, match='requires source_crop_bbox'):
        validate_element_plan(plan, inv)
    raster = plan['objects'][1]
    raster['source_crop_bbox'] = raster.pop('bbox')
    raster['dest_bbox'] = [200, 100, 50, 50]
    with pytest.raises(ValueError, match='font'):
        validate_element_plan(plan, inv)
    plan['text_decisions'][0].update(font_family='Microsoft YaHei', font_confidence=0.5,
                                   font_evidence='synthetic font contract test')
    validate_element_plan(plan, inv)
    raster['source_crop_bbox'] = [95,45,110,40]
    # Moving the destination away must not conceal duplicate source text.
    with pytest.raises(ValueError, match='duplicates raster'):
        validate_element_plan(plan, inv)
    plan['text_decisions'][0]['action']='retained_in_asset';raster['contains_text_ids']=['t1']
    validate_element_plan(plan, inv)
    raster['bbox'] = [1,1,10,10]
    with pytest.raises(ValueError, match='ambiguous'):
        normalize_element_coordinates(plan)
