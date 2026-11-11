"""Fault isolation fixtures. Synthetic probes do not establish Office acceptance."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from product.materialize import write,digest
from product.contracts import Style,Text
from product.parallel_branches import snapshot
from product.parallel_fusion import fuse
from product.parallel_contracts import SceneGraph,Relation
from product.parallel_store import Store
from product.recovery.background_adapter import RecoveryBlocked
from product.recovery import product_pipeline as regional
from product.recovery.region_candidates import generate
from product.recovery.region_state import rollback,closure
from product.recovery.candidate_stage import stage_background
from test_parallel import branches


@pytest.fixture
def product_fixture(tmp_path,monkeypatch):
    from product import parallel_pipeline as pipeline
    job=tmp_path/'job';(job/'inputs').mkdir(parents=True)
    source=np.full((120,180,3),235,np.uint8)
    source[25:40,25:30]=20;source[75:90,115:120]=20
    file=job/'inputs/01.png';Image.fromarray(source).save(file)
    snap,b=branches(file)
    b['text'].styles=[Style(id='s',role='body',ea='Arial',latin='Arial',size_px=10.,bold=False,color='141414')]
    b['text'].texts=[Text(id='a',text='正文甲',bbox=[15.,15.,40.,35.],z=1,style='s'),
                     Text(id='b',text='正文乙',bbox=[105.,65.,40.,35.],z=2,style='s')]
    b['text'].members={'a':['l1'],'b':['l2']}
    ocr={'lines':[{'id':'l1','text':'正文甲','score':.99,'bbox':[15,15,40,35]},
                  {'id':'l2','text':'正文乙','score':.99,'bbox':[105,65,40,35]}]}
    graph=fuse(snap,b,ocr)
    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT',str(tmp_path/'locks'))
    monkeypatch.setenv('IMAGE2PPT_PARALLEL_CACHE',str(tmp_path/'cache'))
    monkeypatch.setattr(pipeline,'perceive',lambda *args,**kwargs:(b,{}))
    monkeypatch.setattr(pipeline,'local_ocr',lambda *args:ocr)
    monkeypatch.setattr('product.recovery.product_pipeline.compare',lambda *args:{})
    monkeypatch.setattr('product.recovery.candidate_stage.font_gate',lambda *args:None)
    monkeypatch.setattr('product.recovery.candidate_stage.native_frame',lambda text,*args:text.model_dump())
    monkeypatch.setattr('product.recovery.product_pipeline.inspect_text',lambda *args,**kwargs:{'status':'PASS','synthetic':True})
    class Actor:
        model='synthetic'
        def ask(self,system,ctx,images,contract):
            return {'schema':'VisualReview/1',**{k:ctx[k] for k in ('source_sha256','plan_sha256','preview_sha256')},
                    'passed':True,'findings':[],'checked':['wording','structure','font','emphasis','layout','graphics']}
    def compiler(irs,output,pages):
        output.mkdir();(output/'preview').mkdir();(output/'fixture.pptx').write_bytes(b'SYNTHETIC')
        for i,ir in enumerate(irs,1):
            Image.open(Path(ir).parent/'source-plate.png').convert('RGB').save(output/'preview'/f'{i:02d}.png')
        return {'pptx':str(output/'fixture.pptx'),'preview':str(output/'preview')}
    def office(pptx,irs,output,**kwargs):
        output.mkdir()
        for i,ir in enumerate(irs,1):
            plate=Image.open(Path(ir).parent/'source-plate.png').convert('RGB')
            plate.save(output/f'page-{i:02d}-before.png')
            for node in json.loads(Path(ir).read_text('utf8'))['nodes']:
                if node['id']!='source-plate':plate.save(output/f'{i:02d}-{node["id"]}-delete.png')
        return {'status':'PASS_APP_AUTOMATED','synthetic':True}
    def verify(trial,g,base,fonts,output,**kwargs):
        output.mkdir()
        ir=stage_background(trial,g,base,fonts,output/'region')
        office(None,[ir],output/'office')
        result={'regional_automatic_checks':'PASS','synthetic':True}
        write(output/'acceptance.json',result)
        return result
    monkeypatch.setattr(regional,'stage_and_verify',verify)
    write(job/'request.json',{'sources':['01.png'],'pipeline_mode':'parallel_review','candidate_engine':'r12-regional',
          'regional_object_ids':['a','b'],'object_extraction':'alpha'})
    return SimpleNamespace(job=job,graph=graph,snap=snap,source=source,actor=Actor(),compiler=compiler,office=office)


@pytest.mark.parametrize('scenario',['normal','region-failure','go-unavailable','go-unconfigured','office-failure'])
def test_formal_entry_region_isolation(product_fixture,monkeypatch,scenario):
    from product.parallel_pipeline import run
    f=product_fixture
    if scenario=='region-failure':
        original=regional.adaptive_candidates
        def fail_region(graph,identifier,*args,**kwargs):
            if identifier=='b':raise RecoveryBlocked('MASK_INCOMPLETE','injected incomplete mask')
            return original(graph,identifier,*args,**kwargs)
        monkeypatch.setattr(regional,'adaptive_candidates',fail_region)
    if scenario=='go-unavailable':
        monkeypatch.setattr(f.actor,'ask',lambda *args:(_ for _ in ()).throw(TimeoutError('injected Go unavailable')))
    office=f.office
    if scenario=='office-failure':
        from product.parallel_office import OfficeValidationError
        def office(pptx,irs,output,**kwargs):
            ids={n['id'] for n in json.loads(Path(irs[0]).read_text('utf8'))['nodes']}
            if {'a','b'}<=ids:raise OfficeValidationError('b','injected failing combined object')
            return f.office(pptx,irs,output,**kwargs)
    if scenario=='go-unconfigured':monkeypatch.delenv('OPENCODE_GO_API_KEY',raising=False)
    result=run(f.job,provider=f.actor,review_provider=None if scenario=='go-unconfigured' else f.actor,
               compiler=f.compiler,office=office,fonts={'fonts':[]})
    isolated=scenario in ('region-failure','office-failure')
    assert result['locally_verified_text_regions']==(1 if isolated else 2)
    selected=SceneGraph.model_validate_json((Path(result['slide_irs'][0]).parent/'scene.json').read_text('utf8')).selected
    assert set(selected)==({'a'} if isolated else {'a','b'})
    assert not result['automatic_publication']
    assert Store(f.job/'state').committed()['result']==result
    if scenario in ('go-unavailable','go-unconfigured'):assert result['independent_review'][0]['failure']=='QA_INFRASTRUCTURE_UNAVAILABLE'
    if scenario=='region-failure':assert result['regions'][0][1]['failure']=='MASK_INCOMPLETE'


def test_lama_timeout_preserves_local_background_candidates(product_fixture):
    f=product_fixture
    def timeout(*args):raise TimeoutError('injected LaMa timeout')
    passed,failed=generate(f.graph,'a',f.snap['path'],None,f.job/'lama-fault',inpaint=timeout,force_inpaint=True)
    assert passed and passed[0]['method']=='gradient'
    assert failed[0]['error_type']=='TimeoutError' and failed[0]['other_candidates_retained']


def test_local_rollback_preserves_other_staged_text(product_fixture):
    from product.parallel_pipeline import run
    f=product_fixture
    result=run(f.job,provider=f.actor,review_provider=f.actor,compiler=f.compiler,office=f.office,fonts={'fonts':[]})
    base=Path(result['slide_irs'][0]).parent
    before=digest(base/'b-owner.npz')
    ir=rollback(f.graph,base,{'a'},f.job/'rolled-back','INJECTED_REGION_FAILURE')
    graph=SceneGraph.model_validate_json((ir.parent/'scene.json').read_text('utf8'))
    assert set(graph.selected)=={'b'} and digest(ir.parent/'b-owner.npz')==before
    source=np.array(Image.open(ir.parent/'source.png').convert('RGB'))
    plate=np.array(Image.open(ir.parent/'source-plate.png').convert('RGB'))
    assert np.array_equal(source[25:40,25:30],plate[25:40,25:30])
    assert np.all(plate[75:90,115:120]==235)


def test_indivisible_dependency_closure_ignores_shared_font(product_fixture):
    graph=product_fixture.graph
    assert closure(graph,{'a'})=={'a'}
    graph.relations.append(Relation(source='b',target='a',kind='part_of',status='RESOLVED',evidence=['fixture']))
    assert closure(graph,{'a'})=={'a','b'}


@pytest.mark.parametrize('code,expected',[('MASK_INCOMPLETE','MASK_INCOMPLETE'),('PROTECTED_NEIGHBOR','OWNER_CONFLICT'),
    ('OFFICE_VALIDATION_FAILED','OFFICE_VALIDATION_FAILED')])
def test_failure_attribution_preserves_stage(code,expected):
    assert regional.failure(RecoveryBlocked(code,'injected'),'font')==expected


def test_explicit_unresolved_region_is_reported_as_owner_conflict(product_fixture,monkeypatch):
    from product.parallel_pipeline import run
    from product.parallel_contracts import SceneGraph
    original=SceneGraph.model_validate
    def ambiguous(value,*args,**kwargs):
        graph=original(value,*args,**kwargs)
        graph.ownership['b']='AMBIGUOUS'
        return graph
    monkeypatch.setattr('product.parallel_fusion.SceneGraph.model_validate',ambiguous)
    f=product_fixture
    result=run(f.job,provider=f.actor,review_provider=f.actor,compiler=f.compiler,office=f.office,fonts={'fonts':[]})
    assert result['locally_verified_text_regions']==1
    assert result['regions'][0][1]['failure']=='OWNER_CONFLICT'


def test_indivisible_region_failure_is_not_misclassified_as_font():
    assert regional.failure(ValueError('indivisible region cannot be split'),'font')=='OWNER_CONFLICT'


def test_owner_failure_during_staging_retains_specific_cause(product_fixture,monkeypatch):
    from product.parallel_pipeline import run
    original=regional.stage_and_verify
    def fail(trial,*args,**kwargs):
        if Path(trial).parent.name=='b':raise ValueError('candidate conflicts with stable owner')
        return original(trial,*args,**kwargs)
    monkeypatch.setattr(regional,'stage_and_verify',fail)
    f=product_fixture
    result=run(f.job,provider=f.actor,review_provider=f.actor,compiler=f.compiler,office=f.office,fonts={'fonts':[]})
    assert result['locally_verified_text_regions']==1
    assert result['regions'][0][1]['failure']=='OWNER_CONFLICT'
