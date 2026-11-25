"""Regression coverage for host-numbered card ownership and actual emitted native groups."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from product.parallel_contracts import SceneGraph
from product.parallel_fusion import fuse
from product.parallel_rebuild import build
from product.recovery.badge_stage import stage_badge_symbols
from product.recovery.container_stage import stage_containers_and_surfaces
from product.runtime.host_evidence import deterministic_plan
from product.runtime.marked_routes import (
    apply_move_groups, measured_label_face, plan_move_groups, stage_planned_groups,
)
from test_badge_panel_layers import flat_card_scene
from test_parallel import branches


def mark(oid, label, bbox, **extra):
    return {'id':oid,'label':label,'bbox':bbox,**extra}


def test_host_plans_one_inner_card_with_distinct_text_and_graphic():
    rows = [mark('outer','card_frame',[0,0,180,120]),
            mark('face','icon',[10,8,95,92],sources=['ui'],intersecting_ocr_ids=['l']),
            mark('text','text',[45,20,33,18],ocr_ids=['l']),
            mark('a-symbol','icon',[15,20,20,18]),mark('detector','image',[15,20,20,18]),
            mark('cross','image',[89,42,28,32])]
    old = deterministic_plan(rows,{})
    assert all(not r['move_with'] for r in old)
    fields = {r['id']:r for r in deterministic_plan(rows,{},card_edits=True)}
    assert fields['face']['route']=='surface' and fields['face']['move_with']==['text','a-symbol']
    assert not fields['outer']['move_with']


def test_detached_surface_and_badge_alias_resolve_and_trace_missing_member():
    route = dict(id='face',bbox=[10,10,75,55],route='surface',move_with=['word','seed'])
    scene = SimpleNamespace(selected={'card':dict(kind='surface',bbox=route['bbox']),
                                      'icon':dict(kind='asset',bbox=[20,20,6,6],badge_evidence_ids=['layout:seed']),
                                      'word':dict(kind='text',bbox=[30,20,20,10])})
    assert plan_move_groups([route],scene)==[['card','icon','word']]
    issues = []
    assert not plan_move_groups([{**route,'move_with':['lost']}],scene,issues=issues)
    assert issues[0]['conflict_ids']==['lost']
    ir = {'groups':[], 'nodes':[dict(id=k,z=i,bbox=b) for i,(k,b) in enumerate([
        ('card',route['bbox']),('neighbor',[15,18,50,10]),('icon',[20,20,6,6]),('word',[30,20,20,10])])]}
    assert apply_move_groups(ir,[['card','icon','word']])[0]['conflict_ids']==['neighbor']
    assert not ir['groups']


def test_local_fill_requires_measured_boundary_and_survives_scoped_stage(tmp_path):
    from product.runtime.batch_backbone import _chain_stage_from_spec
    image = np.full((210,340,3),250,np.uint8)
    image[45:140,35:130] = [40,85,175]
    hint = [33,42,100,102]
    face = measured_label_face(image,hint)
    assert face == [35,45,95,95]
    assert measured_label_face(image,[160,45,95,95]) is None
    path=tmp_path/'source.png'
    Image.fromarray(image).save(path)
    marks=[mark('face','icon',hint,sources=['ui'],intersecting_ocr_ids=['l']),
           mark('title','text',[52,63,38,18],ocr_ids=['l']),
           mark('empty-window','icon',[160,45,95,95],sources=['ui'],intersecting_ocr_ids=['other']),
           mark('unboxed','text',[175,55,43,15],ocr_ids=['other'])]
    plans={p['id']:p for p in deterministic_plan(marks,{},card_edits=True,source_path=path)}
    assert plans['face']['move_with']==['title'] and 'empty-window' not in plans
    snap,ev=branches(path)
    graph=fuse(snap,ev,{'lines':[]})
    base=tmp_path/'base'
    build(graph,snap,{'fonts':[]},base,strategy='off',text_candidates=False)
    route = dict(id='square',bbox=hint,route='surface',target_ids=[],measured_label_face=face)
    out,ids,_=_chain_stage_from_spec(initial_graph=graph,initial_baseline=base,fonts={'fonts':[]},
        source_path=base/'source.png',layers_dir=None,deferred_path=None,page_cache_dir=tmp_path,
        directory=tmp_path, tag='scoped',trials_spec=[],frame_overrides={},skip_ids=set(),include_assets=False,
        include_containers=True,wanted_filter=None,sam_enabled=False,single_object_only=True,panel_regions=[hint],
        marked_routes=[route])
    selected=json.loads((out/'scene.json').read_bytes())['selected']
    assert any(selected[k].get('mark_id')=='square' and selected[k]['bbox']==face for k in ids)


def test_scene_adapter_writes_a_single_group_and_keeps_the_input_immutable(tmp_path):
    base,box=flat_card_scene(tmp_path)
    peeled,ids=stage_badge_symbols(base,tmp_path/'badge',None)
    deck,_=stage_containers_and_surfaces(peeled,tmp_path/'frame',stable_ids=True,native_only=True)
    before=(deck.parent/'scene.json').read_bytes()
    graph=SceneGraph.model_validate_json(before)
    face=next(k for k,v in graph.selected.items() if v['kind']=='surface' and v['bbox']==box)
    route=dict(id='frame',bbox=box,route='surface',move_with=['cloud'])
    out,changed,issues=stage_planned_groups([route],graph,deck.parent,tmp_path/'composed',set())
    assert not issues and set(changed)=={face,ids[0]}
    ir=json.loads((out/'slideir.json').read_bytes())
    assert any(set(g['member_ids'])=={face,ids[0]} for g in ir['groups'])
    assert (deck.parent/'scene.json').read_bytes()==before
    same,changed,issues=stage_planned_groups([route],graph,out,tmp_path/'again',set())
    assert same==out and not changed and not issues


def test_incremental_group_can_gain_an_existing_picture_and_roll_back_on_conflict(tmp_path,monkeypatch):
    from product import parallel_rebuild
    base=tmp_path/'base'
    base.mkdir()
    face_box=[0,0,120,90]
    graph=SimpleNamespace(selected={
        'face':dict(kind='surface',bbox=face_box,move_members=['face','title']),
        'title':dict(kind='text',bbox=[10,10,70,15]),'photo':dict(kind='asset',bbox=[10,35,75,48]),
        'neighbor':dict(kind='asset',bbox=[190,35,75,48])})
    graph.model_dump=lambda: {'selected':graph.selected}
    route=dict(id='frame',bbox=face_box,route='surface',move_with=['title','photo'])

    def adapt(g,path,*,group_issues=None):
        ir={'nodes':[{'id':oid,'z':z,'bbox':g.selected[oid]['bbox']}
                     for z,oid in enumerate(['face','title','neighbor','photo'])],'groups':[]}
        problems=apply_move_groups(ir,[r['move_members'] for r in g.selected.values() if r.get('move_members')])
        if group_issues is not None:
            group_issues.extend(problems)
        return ir

    monkeypatch.setattr(parallel_rebuild,'adapt',adapt)
    graph.selected['neighbor']['bbox'][0]=30
    unchanged,_,blocked=stage_planned_groups([route],graph,base,tmp_path/'blocked',set())
    assert unchanged==base and blocked[0]['conflict_ids']==['neighbor']
    assert graph.selected['face']['move_members']==['face','title']
    graph.selected['neighbor']['bbox'][0]=190
    out,changed,issues=stage_planned_groups([route],graph,base,tmp_path/'allowed',set())
    assert not issues and out!=base and set(changed)=={'face','title','photo'}
    assert len(json.loads((out/'slideir.json').read_bytes())['groups'])==1


def test_shared_title_cannot_be_claimed_by_two_overlapping_card_groups(tmp_path):
    graph=SimpleNamespace(selected={'left':dict(kind='surface',bbox=[0,0,60,50]),
                                    'right':dict(kind='surface',bbox=[44,0,60,50]),
                                    'shared':dict(kind='text',bbox=[46,10,8,20])})
    routes=[dict(id=oid,bbox=graph.selected[oid]['bbox'],route='surface',move_with=['shared'])
            for oid in ('left','right')]
    _,changed,issues=stage_planned_groups(routes,graph,tmp_path,tmp_path/'unused',set())
    assert not changed and {r['object_id'] for r in issues}=={'left','right'}
    assert all(r['failure']=='GROUP_AMBIGUOUS_OWNER' and r['conflict_ids']==['shared'] for r in issues)


def test_verified_photo_frame_resolves_an_outer_card_after_child_frames_settle():
    box=[30,50,300,210]
    graph=SimpleNamespace(selected={'native-face':dict(kind='surface',bbox=box,mark_id='outer'),
                                    'photo':dict(kind='asset',bbox=[48,98,145,92],
                                                 payload={'background_parent_id':'native-face'}),
                                    'title':dict(kind='text',bbox=[55,55,90,23]),
                                    'adjacent':dict(kind='text',bbox=[349,55,90,23])},
                          relations=[SimpleNamespace(source='native-face',target=k,kind='contains',status=s)
                                     for k,s in [('title','RESOLVED'),('adjacent','AMBIGUOUS')]])
    route=dict(id='outer',route='surface',bbox=box,move_with=[],group_owned_children=True)
    assert plan_move_groups([route],graph)==[['native-face','photo','title']]
    assert not plan_move_groups([{**route,'group_owned_children':False}],graph)
