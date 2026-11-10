"""Safety behavior, not slide-quality or human-acceptance fixtures."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from test_parallel import branches
from product.contracts import Style,Text
from product.parallel_contracts import Relation,mode
from product.parallel_fusion import fuse
from product.parallel_store import Store,key
from product.parallel_candidate_engine import discover,CHECKS
from product.parallel_candidate_engine import engine_mode,shadow_discover
from product.parallel_candidate_contracts import ReconstructionCandidate,PreflightResult,transition,candidate_path
from product.parallel_candidate_stage import stage
from product.parallel_text_candidates import generate,perimeter_background,TextUncertainty
from product.parallel_edit_probe import replacement,layout_result
from product.parallel_rebuild import build
from product.materialize import digest,write


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT',str(tmp_path/'leases'))
    # Font installation is tested separately by the product font gate; this
    # fixture isolates source ownership and protocol behavior.
    monkeypatch.setattr('product.materialize.font_gate',lambda *args:None)
    monkeypatch.setattr('product.parallel_rebuild.font_gate',lambda *args:None)
    rgb=np.full((100,180,3),254,np.uint8);rgb[40:46,50:60]=[37,83,179]
    source=tmp_path/'source.png';Image.fromarray(rgb).save(source)
    snap,b=branches(source)
    b['text'].styles=[Style(id='s',role='body',ea='Arial',latin='Arial',size_px=10.,bold=False,color='2553B3')]
    b['text'].texts=[Text(id='t',text='测试',bbox=[40.,30.,40.,30.],z=1,style='s')];b['text'].members={'t':['l']}
    graph=fuse(snap,b,{'lines':[{'id':'l','text':'测试','score':.99}]})
    return graph,snap,rgb,tmp_path


def collect(fixture,name='candidates'):
    graph,snap,rgb,path=fixture
    rows=discover(graph,snap,{'fonts':[]},path/name,Store(path/'state'))
    return rows,path/name


def test_pure_candidates_leave_source_and_scene_unchanged(fixture):
    graph,snap,rgb,path=fixture;before=key(graph.model_dump());sha=digest(snap['path'])
    rows,directory=collect(fixture)
    assert key(graph.model_dump())==before and digest(snap['path'])==sha
    assert len(rows)==4 and not list(directory.rglob('source-plate.png'))
    assert len({r['candidate_id'] for r in rows})==4


def test_upload_shadow_resumes_without_mutating_source(fixture):
    graph,snap,_,path=fixture;store=Store(path/'state');before=key(graph.model_dump())
    first=shadow_discover(graph,snap,{'fonts':[]},path/'upload',store)
    second=shadow_discover(graph,snap,{'fonts':[]},path/'upload',store)
    assert first==second and key(graph.model_dump())==before
    assert len(list((path/'upload').iterdir()))==1
    assert engine_mode()=='off' and engine_mode('r11-shadow')=='r11-shadow'
    with pytest.raises(ValueError):engine_mode('r11-auto')


def test_legacy_build_does_not_mutate_parent(fixture):
    graph,snap,_,path=fixture;before=key(graph.model_dump())
    build(graph,snap,{'fonts':[]},path/'legacy')
    assert key(graph.model_dump())==before


def test_t1_failure_does_not_skip_t2(fixture,monkeypatch):
    def fail(*args):raise ValueError('text source glyph ownership uncertain')
    monkeypatch.setattr('product.parallel_rebuild.text_candidate',fail)
    rows,_=collect(fixture)
    assert rows[0]['hard_gate']=='UNKNOWN' and rows[1]['hard_gate']=='PASS'
    assert rows[2]['hard_gate']=='UNKNOWN' and rows[3]['representation']=='RASTER_FALLBACK'


def test_content_conflict_is_hard_failure(fixture):
    fixture[0].ownership['t']='AMBIGUOUS'
    rows,_=collect(fixture)
    assert all(r['hard_gate']=='FAIL' for r in rows[:3])


def test_source_binding_rejected(fixture):
    fixture[0].source_sha256='a'*64
    with pytest.raises(ValueError,match='source binding'):collect(fixture)


def test_missing_font_keeps_receipt_unknown(fixture,monkeypatch):
    def fail(*args):raise ValueError('font/weight unavailable')
    monkeypatch.setattr('product.materialize.font_gate',fail)
    rows,_=collect(fixture)
    assert rows[1]['hard_gate']=='UNKNOWN'


def test_protected_peer_not_erased(fixture):
    graph,snap,rgb,path=fixture
    blocked=np.zeros(rgb.shape[:2],bool);blocked[42,53]=True
    rows=discover(graph,snap,{'fonts':[]},path/'blocked',Store(path/'state'),blocked=blocked)
    assert all(r['hard_gate']!='PASS' for r in rows)


def test_unknown_red_pixel_is_not_absorbed(fixture):
    graph,snap,rgb,path=fixture;rgb[42,53]=[255,0,0]
    with pytest.raises(TextUncertainty,match='unexplained'):
        generate(graph.text.texts[0],graph,rgb,{'fonts':[]},np.zeros(rgb.shape[:2],bool),'T2-soft-alpha-perimeter')


def test_enclosed_source_overshoot_is_not_boundary_growth(fixture):
    graph,_,rgb,_=fixture;rgb[42,53]=[3,71,167]
    result=generate(graph.text.texts[0],graph,rgb,{'fonts':[]},np.zeros(rgb.shape[:2],bool),'T2-soft-alpha-perimeter')
    assert result['background_evidence']['enclosed_overshoot_pixels']==1
    assert result['mask'].sum()==60


def test_overshoot_at_glyph_boundary_stays_unknown(fixture):
    graph,_,rgb,_=fixture;rgb[40,50]=[3,71,167]
    with pytest.raises(TextUncertainty):generate(graph.text.texts[0],graph,rgb,{'fonts':[]},np.zeros(rgb.shape[:2],bool),'T2-soft-alpha-perimeter')


def test_nonuniform_backing_does_not_get_flattened(fixture):
    _,_,rgb,_=fixture;rgb[19:22,32:88]=[200,50,0]
    with pytest.raises(TextUncertainty):perimeter_background(rgb,[32,22,56,46],np.zeros(rgb.shape[:2],bool))


def test_joint_region_cannot_partially_stage(fixture):
    graph,_,_,_=fixture
    graph.text.texts.append(graph.text.texts[0].model_copy(update={'id':'peer','bbox':[100.,30.,30.,20.]}))
    graph.ownership['peer']='RESOLVED';graph.relations.append(Relation(source='t',target='peer',kind='part_of',status='RESOLVED',evidence=['source']))
    rows,_=collect(fixture)
    assert all(r['hard_gate']!='PASS' for r in rows)


def test_preflight_unknown_cannot_claim_pass():
    checks={c:'PASS' for c in CHECKS};checks['owner_coverage']='UNKNOWN'
    with pytest.raises(ValueError):PreflightResult(candidate_id='x',status='PASS',checks=checks,unknown_pixels=0)


def test_preflight_requires_all_safety_checks():
    with pytest.raises(ValueError):PreflightResult(candidate_id='x',status='PASS',checks={},unknown_pixels=0)


def test_cannot_jump_to_verified_or_publish(fixture):
    rows,_=collect(fixture);candidate=ReconstructionCandidate.model_validate(rows[1])
    with pytest.raises(ValueError):transition(candidate,'VERIFIED')
    assert transition(candidate,'STAGED').status=='STAGED'
    with pytest.raises(ValueError):mode('parallel_auto')


def trial_fixture(fixture):
    graph,snap,_,path=fixture;rows,directory=collect(fixture)
    safe=next(r for r in rows if r['strategy']=='T2-soft-alpha-perimeter')
    fallback=graph.model_copy(deep=True);fallback.ownership['t']='AMBIGUOUS'
    ir,_=build(fallback,snap,{'fonts':[]},path/'baseline')
    return candidate_path(directory,safe['candidate_id']),ir.parent


def test_stage_isolated_baseline_and_stable_bytes(fixture):
    candidate,base=trial_fixture(fixture);graph,_,_,path=fixture
    before={p.name:digest(p) for p in base.iterdir() if p.is_file()}
    ir=stage(candidate,graph,base,path/'staged');value=json.loads(ir.read_text(encoding='utf-8'))
    assert any(n['id']=='t' and n['type']=='text' and n['text']=='测试' for n in value['nodes'])
    assert {p.name:digest(p) for p in base.iterdir() if p.is_file()}==before
    assert not graph.selected
    with pytest.raises(FileExistsError):stage(candidate,graph,base,path/'staged')


def test_tampered_candidate_cannot_stage(fixture):
    candidate,base=trial_fixture(fixture)
    (candidate/'payload.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='tampered'):stage(candidate,fixture[0],base,fixture[3]/'staged')


def test_stale_scene_cannot_stage(fixture):
    candidate,base=trial_fixture(fixture);fixture[0].method_version='changed'
    with pytest.raises(ValueError,match='stale'):stage(candidate,fixture[0],base,fixture[3]/'staged')


def test_equal_and_long_replacement_preserve_count_and_newlines():
    value='中文123\n标题'
    assert len(replacement(value))==len(value) and replacement(value).count('\n')==1
    assert len(replacement(value,True))==4*len(value)
    assert len(replacement(value,shorter=True))<len(value)


@pytest.mark.parametrize('bounds,lines,overflow,wrap',[
    ((0,0,80,20),1,False,False),((0,0,110,20),1,True,False),
    ((0,0,80,40),2,True,True),((-3,0,80,20),1,True,False)])
def test_overflow_and_wrap_detection(bounds,lines,overflow,wrap):
    result=layout_result((0,0,100,25),bounds,lines,1)
    assert result['overflow']==overflow and result['unexpected_wrap']==wrap
