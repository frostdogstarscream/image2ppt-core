"""Coordinate adaptation does not grant source ownership or edit acceptance."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from product.asset_geometry import normalize
from product.parallel_contracts import AssetEvidence
from product.parallel_store import Store


def inventory(boxes):
    return AssetEvidence.model_validate({'schema':'AssetEvidence/1','source_sha256':'a'*64,
        'image_size':[400,300],'method_version':'synthetic/1','objects':[
            {'id':'a-'+str(i),'description':'source icon','prompt_en':'source icon',
             'kind':'icon','bbox':box,'layers':['layer-01.png'],'components':['whole'],
             'text_policy':'exclude_independent_text','parts':[{'name':'whole','bbox':box,'layers':['layer-01.png']}]}
            for i,box in enumerate(boxes)]})


def test_consistent_invalid_xywh_inventory_is_explicitly_adapted(tmp_path):
    original=inventory([[280,150,340,200],[300,220,360,270],[20,20,50,60]])
    raw=original.model_dump()
    adapted,report=normalize(original,Store(tmp_path/'state'))
    assert adapted.objects[0].bbox==[280,150,60,50]
    assert adapted.objects[0].parts[0].bbox==[280,150,60,50]
    assert adapted.objects[2].bbox==[20,20,30,40]
    assert original.model_dump()==raw
    assert report['status']=='ADAPTED_XYXY_TO_XYWH'
    assert report['pixel_ownership_granted'] is False
    assert len(report['invalid_xywh_objects'])==2
    assert Path(adapted.references[-1]).is_file()


def test_valid_xywh_inventory_stays_unchanged_when_dialect_is_ambiguous(tmp_path):
    original=inventory([[20,20,50,60],[60,60,100,100]])
    adapted,report=normalize(original,Store(tmp_path/'state'))
    assert adapted is original
    assert report['status']=='UNCHANGED'


def test_mixed_coordinate_evidence_is_not_reinterpreted(tmp_path):
    original=inventory([[280,150,340,200],[300,220,360,270],[100,20,30,40]])
    adapted,report=normalize(original,Store(tmp_path/'state'))
    assert adapted is original
    assert report['all_boxes_valid_xyxy'] is False


def test_single_invalid_proposal_cannot_change_the_whole_branch(tmp_path):
    original=inventory([[280,150,340,200],[20,20,50,60]])
    adapted,report=normalize(original,Store(tmp_path/'state'))
    assert adapted is original and report['status']=='UNCHANGED'
