"""Boundary tests for model-authored segmentation plans; no model calls."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('segmentation_planner',Path(__file__).resolve().parents[1]/'scripts/plan_segmentation.py')
planner=importlib.util.module_from_spec(spec);spec.loader.exec_module(planner)

def plan():
    return {'schema':planner.SCHEMA,'summary':'test','risks':[], 'regions':[
        {'id':'M01','parent_id':None,'label':'module','kind':'module','bbox':[0,0,500,500],'action':'split_children','reason':'visible'},
        {'id':'T01','parent_id':'M01','label':'text','kind':'text','bbox':[20,20,400,100],'action':'ocr','reason':'visible'}]}

@pytest.mark.parametrize('box', [[0,0,1001,100],[100,0,50,100],[0,0,float('nan'),100],[False,0,50,100]])
def test_reject_invalid_model_coordinates(box):
    value=plan();value['regions'][1]['bbox']=box
    with pytest.raises(ValueError):planner.validate_plan(value)

def test_duplicate_and_cyclic_model_ids_rejected():
    value=plan();value['regions'].append(copy.deepcopy(value['regions'][1]))
    with pytest.raises(ValueError):planner.validate_plan(value)
    value=plan();value['regions'][0]['parent_id']='M01'
    with pytest.raises(ValueError):planner.validate_plan(value)

def test_outside_child_disclosed_without_adjusting_model_edges():
    value=plan();value['regions'][1]['bbox']=[20,20,600,100];before=copy.deepcopy(value)
    assert planner.validate_plan(value)==[{'code':'child_outside_parent','id':'T01','parent':'M01'}]
    assert value==before

def test_renderer_preserves_source_and_labels_views(tmp_path):
    source=tmp_path/'source.png';planner.Image.new('RGB',(500,300),'white').save(source)
    original=source.read_bytes();planner.save_new(tmp_path/'binding.json',{'source_sha256':planner.digest(source)})
    planner.save_new(tmp_path/'plan.json',plan());files,warnings=planner.render(tmp_path)
    assert source.read_bytes()==original and not warnings
    assert set(files)=={'modules','objects','all'}
    with planner.Image.open(tmp_path/'modules.png') as im:
        assert im.size==(500,300) and im.getpixel((250,100))!=(255,255,255)
    with pytest.raises(FileExistsError):planner.render(tmp_path)
