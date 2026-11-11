import json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_r1205_agent_asset_stage import scene
from product.recovery.single_asset import probe
from product.recovery.asset_retry import sam_source_candidates
from product.recovery.failure_agent import route


def test_cached_single_object_does_not_reexecute_or_accept_changed_png(tmp_path):
    graph,layers,base=scene(tmp_path)
    one=probe(graph,'first',layers/'source.png',layers,base,tmp_path/'probe',strategy='source-halo')
    assert one['status']=='CANDIDATE_READY' and one['office']=='NOT_RUN'
    two=probe(graph,'first',layers/'source.png',layers,base,tmp_path/'probe',strategy='source-halo')
    assert two['cache_hit'] is True and two['candidate_hashes']==one['candidate_hashes']
    file=Path(one['candidate_directory'])/'asset.png';file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='cache evidence changed'):
        probe(graph,'first',layers/'source.png',layers,base,tmp_path/'probe',strategy='source-halo')


def test_unresolved_owner_does_not_dispatch_sam(tmp_path,monkeypatch):
    graph,layers,base=scene(tmp_path);graph.ownership['first']='AMBIGUOUS'
    monkeypatch.setattr('product.object_stage.infer_sam',lambda *args:pytest.fail('unresolved owner must not call SAM'))
    result=probe(graph,'first',layers/'source.png',layers,base,tmp_path/'probe',strategy='sam')
    assert result['status']=='BLOCKED' and result['model_calls']==0


def test_sam_selects_whole_source_components_without_mask_clipping(tmp_path):
    graph,_,_=scene(tmp_path);obj=graph.asset.objects[0]
    source=np.full((50,70,3),240,np.uint8);rgba=np.zeros((50,70,4),np.uint8)
    rgba[10:20,10:20]=[20,40,100,255];rgba[30:35,40:45]=[20,40,100,255]
    mask=np.zeros((50,70),bool);mask[12,12]=True
    result=sam_source_candidates(obj,source,np.zeros((50,70),bool),None,[{'mask':mask,'index':0,'score':.9}],[(rgba,{'method':'source-unmix'})])
    assert len(result)==1 and int((result[0][0][...,3]>0).sum())==100
    assert np.array_equal(result[0][0][10:20,10:20],rgba[10:20,10:20])
    assert not result[0][0][30:35,40:45,3].any()


def test_new_sam_evidence_upgrades_asset_route():
    old=route('ASSET_EXTRACTION_UNCERTAIN','asset',[],already_attempted=['source-halo'])
    new=route('ASSET_EXTRACTION_UNCERTAIN','asset',[],already_attempted=['source-halo'],new_sam_evidence=True)
    assert old['status']=='NO_CHANGED_EVIDENCE' and new['strategy']=='sam'


def test_sam_assembly_supplies_extent_to_existing_unmix_without_granting_ownership(tmp_path, monkeypatch):
    graph, _, _ = scene(tmp_path)
    source = np.full((50, 70, 3), 240, np.uint8)
    prior = np.zeros((50, 70, 4), np.uint8)
    prior[10:20, 10:20] = [20, 40, 100, 255]
    mask = prior[..., 3] > 0
    calls = []
    def unmix(obj, pixels, protected, *, original_alpha, surface):
        calls.append(original_alpha)
        assert pixels is source and np.array_equal(original_alpha, prior)
        return prior.copy(), {'method': 'source-unmix', 'scope': [6, 6, 18, 18]}
    monkeypatch.setattr('product.object_unmix.candidate_from_evidence', unmix)
    candidates = sam_source_candidates(graph.asset.objects[0], source, np.zeros((50, 70), bool),
        None, [{'mask': mask, 'index': 0, 'score': .9}], [(prior, {'method': 'sam3', 'mask_index': 0})])
    assert len(calls) == 1 and np.array_equal(candidates[0][0], prior)
    assert candidates[0][1]['method'] == 'sam-source-unmix'
