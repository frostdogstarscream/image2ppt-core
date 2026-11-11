"""Synthetic orchestration faults do not establish actual Office acceptance."""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from product.materialize import digest, write
from product.object_contracts import ClosedObject
from product.parallel_contracts import SceneGraph
from product.parallel_fusion import fuse
from product.parallel_office import OfficeValidationError
from product.parallel_rebuild import adapt, build
from product.recovery import product_pipeline as regional
from test_parallel import branches


@pytest.fixture
def expansion(tmp_path,monkeypatch):
    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT',str(tmp_path/'locks'))
    source=np.full((100,180,3),235,np.uint8)
    positions={'stable':[20,70,8,8],'icon-a':[35,25,8,8],
               'icon-b':[85,25,8,8],'icon-c':[135,25,8,8]}
    for x,y,w,h in positions.values():source[y:y+h,x:x+w]=[40,80,150]
    file=tmp_path/'source.png';Image.fromarray(source).save(file)
    snap,evidence=branches(file)
    evidence['asset'].objects=[ClosedObject(id=name,description='whole independent icon',prompt_en='icon',
        bbox=box,layers=['layer.png'],text_policy='exclude_independent_text',components=['whole'],
        kind='icon',parts=[{'name':'whole','bbox':box,'layers':['layer.png']}]) for name,box in positions.items()]
    graph=fuse(snap,evidence,{'lines':[]})
    initial,_=build(graph,snap,{'fonts':[]},tmp_path/'baseline',strategy='off',text_candidates=False)

    def append(identifier,parent,output):
        shutil.copytree(parent,output)
        scene=SceneGraph.model_validate_json((output/'scene.json').read_text('utf8'))
        x,y,w,h=positions[identifier];mask=np.zeros(source.shape[:2],bool);mask[y:y+h,x:x+w]=True
        patch=np.full((h,w,3),235,np.uint8)
        plate=np.array(Image.open(output/'source-plate.png').convert('RGBA'))
        plate[mask,:3]=235;Image.fromarray(plate).save(output/'source-plate.png')
        png=output/(identifier+'.png');Image.fromarray(source[y:y+h,x:x+w]).save(png)
        owner=output/(identifier+'-owner.npz')
        np.savez_compressed(owner,mask=mask,restore_bbox=positions[identifier],backing_patch=patch)
        row={'id':identifier,'kind':'asset','bbox':positions[identifier],
             'payload':{'method':'synthetic-fault-fixture'},'state':'STAGED',
             'source_sha256':graph.source_sha256,
             'owner_evidence':{'file':owner.name,'sha256':digest(owner)},
             'asset_reference':{'path':png.name,'sha256':digest(png),
                 'source_sha256':graph.source_sha256,'bbox':positions[identifier]}}
        scene.selected[identifier]=row;scene.transactions.append(row)
        scene.raster_backing['sha256']=digest(output/'source-plate.png')
        write(output/'scene.json',scene.model_dump())
        write(output/'transactions.json',list(scene.selected.values()))
        write(output/'slideir.json',adapt(scene,output))
        with np.load(output/'editable-mask.npz') as data:editable=data['mask']|mask
        np.savez_compressed(output/'editable-mask.npz',mask=editable)
        with np.load(output/'ownership.npz') as data:maps={name:data[name].copy() for name in data.files}
        maps['ownership'][mask]=0;np.savez_compressed(output/'ownership.npz',**maps)
        ownership=json.loads((output/'ownership.json').read_text('utf8'))
        ownership['map_sha256']=digest(output/'ownership.npz');write(output/'ownership.json',ownership)
        return output/'slideir.json'

    current=append('stable',initial.parent,tmp_path/'seed')
    graph=SceneGraph.model_validate_json((current.parent/'scene.json').read_text('utf8'))
    original_stable=digest(current.parent/'stable.png')
    state=SimpleNamespace(candidate_failure=None,residual_unknown=None,residual_missing=None,compiler_failure=False)
    def retry(latest,identifier,source_file,layers,baseline,output):
        if identifier==state.candidate_failure:raise ValueError('injected bad candidate')
        output.mkdir(parents=True)
        write(output/'identity.json',{'id':identifier})
        return {'status':'CANDIDATE_READY','candidate_directory':str(output)}
    def stage(candidate,latest,baseline,output,**kwargs):
        return append(json.loads((Path(candidate)/'identity.json').read_text('utf8'))['id'],baseline,output)
    def compiler(paths,output,texts):
        if state.compiler_failure:raise OSError('injected compiler unavailable')
        output.mkdir();file=output/'synthetic.pptx';file.write_bytes(b'SYNTHETIC')
        return {'pptx':str(file)}
    def residuals(ir,office,page,output):
        selected=SceneGraph.model_validate_json((Path(ir).parent/'scene.json').read_text('utf8')).selected
        return [{'id':identifier,'residual_check_status':'UNKNOWN' if identifier==state.residual_unknown else 'PASS'}
                for identifier in selected if identifier!='stable' and identifier!=state.residual_missing]
    monkeypatch.setattr('product.recovery.asset_retry.retry_asset_extraction',retry)
    monkeypatch.setattr('product.recovery.agent_asset_stage.stage_asset_candidate',stage)
    monkeypatch.setattr(regional,'inspect_residuals',residuals)
    def execute(office):
        folder=tmp_path/'run';folder.mkdir()
        path,rows=regional.expand_assets(graph,current,file,None,folder,{},compiler,office)
        selected=SceneGraph.model_validate_json((path.parent/'scene.json').read_text('utf8')).selected
        assert digest(path.parent/'stable.png')==original_stable
        assert digest(current.parent/'stable.png')==original_stable
        return path,{row['object_id']:row for row in rows},set(selected)
    return SimpleNamespace(run=execute,state=state)


def pass_office(pptx,paths,output,**kwargs):
    assert set(kwargs['target_ids'])<= {'icon-a','icon-b','icon-c'}
    return {'status':'PASS_APP_AUTOMATED'}


def test_one_failed_asset_candidate_keeps_other_staged_assets(expansion):
    expansion.state.candidate_failure='icon-b'
    _,rows,selected=expansion.run(pass_office)
    assert selected=={'stable','icon-a','icon-c'}
    assert rows['icon-a']['state']==rows['icon-c']['state']=='VERIFIED_LOCAL'
    assert rows['icon-b']['state']=='LOCAL_FALLBACK'


def test_office_object_failure_rolls_back_only_affected_asset(expansion):
    def failing_office(*args,**kwargs):
        raise OfficeValidationError('icon-b','injected failed edit')
    path,rows,selected=expansion.run(failing_office)
    assert selected=={'stable','icon-a','icon-c'}
    assert rows['icon-b']['state']=='LOCAL_FALLBACK'
    assert rows['icon-a']['state']==rows['icon-c']['state']=='STAGED_LOCAL'
    rollback=json.loads((path.parent/'rollback.json').read_text('utf8'))
    assert rollback['removed']==['icon-b']


@pytest.mark.parametrize('mode',['exception','unknown-receipt'])
def test_office_unavailable_retains_candidates_without_pass_claim(expansion,mode):
    def unavailable(*args,**kwargs):
        if mode=='exception':raise TimeoutError('injected Office unavailable')
        return {'status':'UNKNOWN','reason':'actual Office unavailable'}
    _,rows,selected=expansion.run(unavailable)
    assert selected=={'stable','icon-a','icon-b','icon-c'}
    assert all(row['state']=='STAGED_LOCAL' and row['office']=='UNKNOWN' for row in rows.values())
    assert all(row['failure']=='QA_INFRASTRUCTURE_UNAVAILABLE' for row in rows.values())


def test_missing_delete_evidence_keeps_asset_in_review(expansion):
    expansion.state.residual_unknown='icon-b'
    _,rows,selected=expansion.run(pass_office)
    assert selected=={'stable','icon-a','icon-b','icon-c'}
    assert rows['icon-b']['state']=='STAGED_LOCAL' and rows['icon-b']['office']=='UNKNOWN'
    assert rows['icon-a']['state']==rows['icon-c']['state']=='VERIFIED_LOCAL'


def test_missing_residual_receipt_never_grants_verified_status(expansion):
    expansion.state.residual_missing='icon-b'
    _,rows,selected=expansion.run(pass_office)
    assert selected=={'stable','icon-a','icon-b','icon-c'}
    assert rows['icon-b']['state']=='STAGED_LOCAL' and rows['icon-b']['office']=='UNKNOWN'
    assert rows['icon-a']['state']==rows['icon-c']['state']=='VERIFIED_LOCAL'


def test_compiler_unavailable_keeps_staged_assets_and_receipt(expansion):
    expansion.state.compiler_failure=True
    path,rows,selected=expansion.run(pass_office)
    assert selected=={'stable','icon-a','icon-b','icon-c'}
    assert all(row['state']=='STAGED_LOCAL' and row['office']=='UNKNOWN' for row in rows.values())
    assert all(row['failure']=='QA_INFRASTRUCTURE_UNAVAILABLE' for row in rows.values())
