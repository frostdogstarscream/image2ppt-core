"""Synthetic asset transaction tests; actual Office evidence is separate."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from product.materialize import digest,write
from product.object_contracts import ClosedObject
from product.parallel_contracts import SceneGraph
from product.parallel_fusion import fuse
from product.parallel_rebuild import build
from product.recovery.asset_retry import retry_asset_extraction
from product.recovery.agent_asset_stage import stage_asset_candidate
from test_parallel import branches
from test_object_extraction import fixture


def scene(tmp_path):
    layers,_,_=fixture(tmp_path)
    fg=np.array(Image.open(layers/'layer-01.png').convert('RGBA'))
    fg[12:22,42:50]=[30,80,180,255]
    Image.fromarray(fg).save(layers/'layer-01.png')
    bg=Image.open(layers/'layer-00.png').convert('RGBA')
    Image.alpha_composite(bg,Image.fromarray(fg)).convert('RGB').save(layers/'source.png')
    manifest=json.loads((layers/'manifest.json').read_text('utf8'))
    manifest['source_sha256']=digest(layers/'source.png')
    manifest['layers'][1]['sha256']=digest(layers/'layer-01.png');write(layers/'manifest.json',manifest)
    snap,evidence=branches(layers/'source.png')
    for identifier,box in [('first',[8,8,16,20]),('second',[40,10,12,16])]:
        evidence['asset'].objects.append(ClosedObject(id=identifier,description='whole independent source icon',
            prompt_en='whole independent source icon',kind='icon',bbox=box,layers=['layer-01.png'],
            components=['whole'],text_policy='exclude_independent_text',
            parts=[{'name':'whole','bbox':box,'layers':['layer-01.png']}]))
    graph=fuse(snap,evidence,{'lines':[]})
    baseline=tmp_path/'baseline'
    build(graph,snap,{'fonts':[]},baseline,strategy='off',text_candidates=False)
    return graph,layers,baseline


def test_sequential_asset_staging_retains_prior_png_and_graphic_ownership(tmp_path):
    graph,layers,baseline=scene(tmp_path)
    parent={str(p.relative_to(baseline)):digest(p) for p in baseline.rglob('*') if p.is_file()}
    first=retry_asset_extraction(graph,'first',layers/'source.png',layers,baseline,tmp_path/'retry1')
    assert first['status']=='CANDIDATE_READY'
    ir1=stage_asset_candidate(first['candidate_directory'],graph,baseline,tmp_path/'stage1',layer_directory=layers)
    first_bytes=digest(ir1.parent/'first.png')
    updated=SceneGraph.model_validate_json((ir1.parent/'scene.json').read_text('utf8'))
    second=retry_asset_extraction(updated,'second',layers/'source.png',layers,ir1.parent,tmp_path/'retry2')
    assert second['status']=='CANDIDATE_READY'
    ir2=stage_asset_candidate(second['candidate_directory'],updated,ir1.parent,tmp_path/'stage2',layer_directory=layers)
    final=SceneGraph.model_validate_json((ir2.parent/'scene.json').read_text('utf8'))
    assert set(final.selected)=={'first','second'}
    assert digest(ir2.parent/'first.png')==first_bytes
    assert final.selected['second']['asset_reference']['path']=='second.png'
    with np.load(ir2.parent/'ownership.npz',allow_pickle=False) as data:ownership=data['ownership']
    for record in final.selected.values():
        with np.load(ir2.parent/record['owner_evidence']['file'],allow_pickle=False) as data:
            assert np.all(ownership[data['mask']]==0)
    assert parent=={str(p.relative_to(baseline)):digest(p) for p in baseline.rglob('*') if p.is_file()}


def test_false_preflight_cannot_hide_changed_source_owner_mask(tmp_path):
    graph,layers,baseline=scene(tmp_path)
    result=retry_asset_extraction(graph,'first',layers/'source.png',layers,baseline,tmp_path/'retry')
    directory=Path(result['candidate_directory'])
    with np.load(directory/'pixels.npz',allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    yy,xx=np.nonzero(arrays['mask']);arrays['mask'][yy[0],xx[0]]=False
    np.savez_compressed(directory/'pixels.npz',**arrays)
    candidate=json.loads((directory/'candidate.json').read_text('utf8'))
    candidate['references']['pixels.npz']=digest(directory/'pixels.npz');write(directory/'candidate.json',candidate)
    with pytest.raises(ValueError,match='owner closure'):
        stage_asset_candidate(directory,graph,baseline,tmp_path/'invalid-stage',layer_directory=layers)
    assert not (tmp_path/'invalid-stage').exists()
