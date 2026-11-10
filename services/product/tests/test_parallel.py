import copy,json,sys
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from product.parallel_store import Store,UnknownCall,Conflict
from product.parallel_contracts import TextEvidence,AssetEvidence,SurfaceEvidence,SceneGraph,mode,strategy
from product.parallel_branches import snapshot
from product.parallel_fusion import fuse,glyph_evidence
from product.parallel_rebuild import build
from product.materialize import digest,write


@pytest.fixture(autouse=True)
def isolated_resources(tmp_path,monkeypatch):
    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT',str(tmp_path/'test-resource-leases'))


def branches(source):
    snap=snapshot(source,source.parent/'snapshots');base={k:snap[k] for k in ('source_sha256','image_size','transform')}
    base['method_version']='test/1'
    b={'text':TextEvidence.model_validate({'schema':'TextEvidence/1',**base}),
       'asset':AssetEvidence.model_validate({'schema':'AssetEvidence/1',**base}),
       'surface':SurfaceEvidence.model_validate({'schema':'SurfaceEvidence/1',**base,'background':'FFFFFF'})}
    return snap,b


def test_single_generation_cas_and_tamper(tmp_path):
    store=Store(tmp_path/'db');file=tmp_path/'asset';file.write_text('a')
    manifest={'files':{str(file):digest(file)},'result':{'pages':12}}
    assert store.publish(manifest,0)==1
    with pytest.raises(Conflict):store.publish(manifest,0)
    assert store.committed()['result']['pages']==12
    file.write_text('b')
    with pytest.raises(ValueError):store.committed()


def test_call_unknown_never_repeated_and_raw_schema_replay(tmp_path):
    class Actor:
        model='test';calls=0
        def ask(self,*args):self.calls+=1;raise TimeoutError()
    store=Store(tmp_path);a=Actor()
    with pytest.raises(TimeoutError):store.call('x',a,'prompt',{},[],TextEvidence,nullcontext())
    with pytest.raises(UnknownCall):store.call('x',a,'prompt',{},[],TextEvidence,nullcontext())
    assert a.calls==1


def test_same_call_reuses_raw_and_contract_failure_is_not_recharged(tmp_path):
    class Actor:
        model='test';calls=0
        def ask(self,*args):self.calls+=1;return {'schema':'wrong'}
    store=Store(tmp_path);a=Actor()
    for i in range(2):
        with pytest.raises(ValueError):store.call('x',a,'prompt',{},[],TextEvidence,nullcontext())
    assert a.calls==1


def test_dag_invalidation_is_local(tmp_path):
    s=Store(tmp_path)
    s.task('text','text','COMPLETED',{})
    s.task('asset','asset','COMPLETED',{})
    s.task('relation','fusion','COMPLETED',{},['text'])
    assert s.dependents('text')=={'text','relation'}


def test_no_serial_execution_mode_and_default_strategy():
    assert mode()=='parallel_review' and strategy()=='auto'
    with pytest.raises(ValueError):mode('serial')
    with pytest.raises(ValueError):mode('parallel_auto')
    assert strategy('off')=='off'


def test_source_fallback_is_complete_and_immutable(tmp_path):
    p=tmp_path/'source.png';Image.new('RGB',(40,30),'#AABBCC').save(p);original=digest(p)
    snap,b=branches(p);g=fuse(snap,b,{'lines':[]})
    ir,report=build(g,snap,{'fonts':[]},tmp_path/'scene',strategy='off')
    r=json.loads(ir.read_text(encoding='utf-8'))
    assert r['nodes'][0]['id']=='source-plate' and len(r['nodes'])==1
    plate=np.array(Image.open(ir.parent/'source-plate.png').convert('RGB'))
    assert np.array_equal(plate,np.array(Image.open(p))) and digest(p)==original


def test_cross_branch_sha_mismatch_rejected(tmp_path):
    p=tmp_path/'source.png';Image.new('RGB',(40,30),'white').save(p);snap,b=branches(p)
    b['asset'].source_sha256='f'*64
    with pytest.raises(ValueError,match='snapshot'):fuse(snap,b,{'lines':[]})


def test_glyph_mask_is_ink_not_box_and_white_on_blue():
    bg=np.full((20,30,3),[20,40,80],dtype=np.uint8);src=bg.copy();src[4:15,7:9]=255
    ink,unknown=glyph_evidence(src,[0,0,30,20],'FFFFFF',bg)
    assert ink.sum()==22 and not unknown.any() and not ink[0,0]
    src[5,20]=[255,0,0]
    ink,unknown=glyph_evidence(src,[0,0,30,20],'FFFFFF',bg)
    assert unknown[5,20] and not ink[5,20]


def test_two_number_and_title_members_can_cover_one_ocr_line(tmp_path):
    p=tmp_path/'source.png';Image.new('RGB',(100,30),'white').save(p);snap,b=branches(p)
    b['text'].styles=[]
    from product.contracts import Style,Text
    b['text'].styles=[Style(id='s',role='title',ea='Arial',latin='Arial',size_px=12.,bold=True,color='000000')]
    b['text'].texts=[Text(id='t-number',text='02',bbox=[0.,0.,20.,20.],z=1,style='s'),Text(id='t-title',text='标题',bbox=[25.,0.,50.,20.],z=1,style='s')]
    b['text'].members={'t-number':['l0'],'t-title':['l0']}
    g=fuse(snap,b,{'lines':[{'id':'l0','text':'02标题','score':.99}]})
    assert set(g.ownership.values())=={'RESOLVED'}


def test_worker_never_calls_legacy_serial_entry():
    import ast
    from product import parallel_pipeline,parallel_branches,parallel_rebuild
    for module in (parallel_pipeline,parallel_branches,parallel_rebuild):
        tree=ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module=='pipeline':
                assert not {'run','Page','materialize'} & {x.name for x in node.names}


def test_failed_generation_cannot_publish_half_artifacts(tmp_path):
    s=Store(tmp_path/'db');present=tmp_path/'exists';present.write_text('x')
    with pytest.raises(FileNotFoundError):s.publish({'files':{str(present):digest(present),str(tmp_path/'missing'):'a'},'result':{}},0)
    assert s.head()==0 and s.committed() is None


def test_perception_is_independent_and_branch_failure_isolated(tmp_path,monkeypatch):
    from product import parallel_branches as mod
    p=tmp_path/'source.png';Image.new('RGB',(40,30),'white').save(p);snap,_=branches(p)
    layerdir=tmp_path/'layers';layerdir.mkdir()
    Image.new('RGBA',(40,30),(0,0,255,0)).save(layerdir/'layer-00.png')
    monkeypatch.setattr(mod,'local_ocr',lambda *a:{'schema':'OCRLines/1','source_sha256':snap['source_sha256'],'lines':[]})
    monkeypatch.setattr(mod,'layer_evidence',lambda *a:(layerdir,[{'file':'layer-00.png','sha256':digest(layerdir/'layer-00.png'),'z':0}]))
    class Actor:
        model='fixture'
        def __init__(self):self.calls=[]
        def ask(self,system,ctx,images,contract):
            self.calls.append(contract.__name__)
            if contract is SurfaceEvidence:raise TimeoutError('synthetic unavailable')
            base={k:ctx[k] for k in ('source_sha256','image_size','transform')}
            return {'schema':contract.__name__+'/1',**base,'method_version':'test/1'}
    a=Actor();result,failed=mod.perceive(snap,Store(tmp_path/'db'),a,{'fonts':[]},tmp_path/'cache','unused')
    assert set(a.calls)=={'TextEvidence','AssetEvidence','SurfaceEvidence'}
    assert set(failed)=={'surface'} and result['text'].method_version=='test/1'
    assert result['surface'].method_version=='r10-fallback/1'


def test_repair_budget_is_durable_and_deduplicated(tmp_path):
    from product.parallel_store import BudgetExceeded
    s=Store(tmp_path)
    assert s.consume('object','sam','prompt1',2)
    assert not s.consume('object','sam','prompt1',2)
    assert Store(tmp_path).consume('object','sam','prompt2',2)
    with pytest.raises(BudgetExceeded):Store(tmp_path).consume('object','sam','prompt3',2)


def test_font_alias_adapter_requires_an_actual_installed_face(tmp_path):
    from product.parallel_branches import canonical_fonts
    from product.contracts import Style
    p=tmp_path/'s.png';Image.new('RGB',(30,20),'white').save(p);snap,b=branches(p)
    b['text'].styles=[Style(id='s',role='body',ea='Microsoft YaHei',latin='Arial',size_px=12.,bold=True,color='000000')]
    v=canonical_fonts(b['text'],{'fonts':[{'family':'微软雅黑','bold':True}]},Store(tmp_path/'db'))
    assert v.styles[0].ea=='微软雅黑' and b['text'].styles[0].ea=='Microsoft YaHei'
    assert v.references and v.styles[0].bold


def test_pixel_cache_ignores_names_and_preserves_soft_alpha(tmp_path):
    from product.parallel_rebuild import asset_candidates
    from product.object_contracts import ClosedObject
    source=np.full((30,40,3),255,np.uint8);png=np.zeros((4,5,4),np.uint8);png[1:3,1:4]=[255,255,255,90]
    file=tmp_path/'stable.png';Image.fromarray(png).save(file)
    r={'id':'known','bbox':[4,4,5,4],'path':str(file),'sha256':digest(file)}
    o=ClosedObject(id='known',description='icon',prompt_en='icon',bbox=[4,4,5,4],layers=['layer-01.png'],
        text_policy='exclude_independent_text',components=['icon'],kind='icon',parts=[{'name':'part','bbox':[4,4,5,4],'layers':['layer-01.png']}])
    a=asset_candidates(o,source,None,np.zeros((30,40),bool),[r],'auto')
    o.description='renamed';o.parts[0].name='renamed-part'
    b=asset_candidates(o,source,None,np.zeros((30,40),bool),[r],'auto')
    assert np.array_equal(a[0][0],b[0][0]) and a[0][0][5,5].tolist()==[255,255,255,90]


@pytest.mark.parametrize('candidate_engine',['off','r11-shadow'])
def test_complete_worker_keeps_source_when_branch_crashes(tmp_path,monkeypatch,candidate_engine):
    from product import parallel_pipeline as mod
    from product.contracts import Review
    job=tmp_path/'job';(job/'inputs').mkdir(parents=True);Image.new('RGB',(40,30),'white').save(job/'inputs/01.png')
    write(job/'request.json',{'sources':['01.png'],'pipeline_mode':'parallel_review','object_extraction':'off'})
    monkeypatch.setenv('IMAGE2PPT_PARALLEL_CACHE',str(tmp_path/'cache'))
    monkeypatch.setenv('IMAGE2PPT_CANDIDATE_ENGINE',candidate_engine)
    monkeypatch.setattr(mod,'perceive',lambda *a,**k:(_ for _ in ()).throw(ValueError('wrong source SHA')))
    class Actor:
        model='synthetic'
        def ask(self,system,ctx,images,contract):
            assert contract is Review
            return {'schema':'VisualReview/1',**{k:ctx[k] for k in ('source_sha256','plan_sha256','preview_sha256')},
                'passed':True,'findings':[],'checked':['wording','structure','font','emphasis','layout','graphics']}
    def compiler(irs,output,pages):
        output.mkdir();(output/'preview').mkdir();(output/'fixture.pptx').write_bytes(b'synthetic-not-an-office-acceptance')
        for i in range(len(irs)):Image.new('RGB',(40,30),'white').save(output/'preview'/f'{i+1:02d}.png')
        return {'pptx':str(output/'fixture.pptx'),'preview':str(output/'preview')}
    def office(pptx,irs,output):
        output.mkdir()
        for i in range(len(irs)):Image.new('RGB',(40,30),'white').save(output/f'page-{i+1:02d}-before.png')
        return {'status':'PASS_APP_AUTOMATED','synthetic':True}
    r=mod.run(job,provider=Actor(),review_provider=Actor(),compiler=compiler,office=office,fonts={'fonts':[]})
    assert r['status']=='REVIEW_REQUIRED' and r['reconstructed_pages']==0 and r['generation']==1
    assert len(json.loads(Path(r['slide_irs'][0]).read_text())['nodes'])==1
    assert Store(job/'state').committed()['result']==r
    assert bool(list(job.glob('page-01/r11-shadow-*/candidate-index.json')))==(candidate_engine=='r11-shadow')


def test_shared_gpu_lease_blocks_another_process(tmp_path):
    import os,subprocess
    from product.resource_lease import lease
    services=Path(__file__).resolve().parents[2]
    code='from resource_lease import lease\nwith lease("gpu-fixture",timeout=.2):print("unexpected")'
    env={**os.environ,'PYTHONPATH':str(services),'IMAGE2PPT_RESOURCE_ROOT':str(tmp_path/'locks')}
    with lease('gpu-fixture',root=tmp_path/'locks',timeout=1):
        r=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,timeout=10)
    assert r.returncode!=0 and b'resource wait exceeded' in r.stderr


def test_scene_graph_alone_reproduces_ir_and_checks_backing_hash(tmp_path):
    from product.parallel_rebuild import adapt
    p=tmp_path/'s.png';Image.new('RGB',(40,30),'white').save(p);snap,b=branches(p);g=fuse(snap,b,{'lines':[]})
    ir,_=build(g,snap,{'fonts':[]},tmp_path/'scene')
    stored=SceneGraph.model_validate(json.loads((ir.parent/'scene.json').read_text(encoding='utf-8')))
    assert adapt(stored,ir.parent)==json.loads(ir.read_text(encoding='utf-8'))
    Image.new('RGBA',(40,30),'black').save(ir.parent/'source-plate.png')
    with pytest.raises(ValueError,match='reference binding'):adapt(stored,ir.parent)


def test_explicit_indivisible_title_reverts_when_one_paragraph_fails(tmp_path,monkeypatch):
    import product.parallel_rebuild as mod
    from product.contracts import Text,Style
    p=tmp_path/'s.png';Image.new('RGB',(80,60),'white').save(p);snap,b=branches(p)
    b['text'].styles=[Style(id='title',role='wrapped-title',ea='Arial',latin='Arial',size_px=16.,bold=True,color='000000')]
    b['text'].texts=[Text(id='t-business',text='业务',bbox=[2.,2.,30.,20.],z=1,style='title'),Text(id='t-application',text='应用',bbox=[2.,24.,30.,20.],z=1,style='title')]
    b['text'].members={'t-business':['l1'],'t-application':['l2']}
    g=fuse(snap,b,{'lines':[{'id':'l1','text':'业务','score':.99},{'id':'l2','text':'应用','score':.99}]})
    from product.parallel_contracts import Relation
    g.relations.append(Relation(source='t-application',target='t-business',kind='part_of',status='RESOLVED',evidence=['indivisible-title']))
    def candidate(t,graph,source,fonts,blocked):
        if t.id=='t-application':raise ValueError('synthetic member failed')
        mask=np.zeros(source.shape[:2],bool);mask[3,3]=True
        return {'id':t.id,'kind':'text','bbox':[2,2,30,20],'mask':mask,'patch':np.full((20,30,3),255,np.uint8),'payload':t.model_dump()}
    monkeypatch.setattr(mod,'text_candidate',candidate)
    ir,report=build(g,snap,{'fonts':[]},tmp_path/'scene')
    assert len(json.loads(ir.read_text(encoding='utf-8'))['nodes'])==1
    assert any(r.get('reason_code')=='REGION_MEMBER_FAILURE' for r in report)


def test_source_owner_uses_source_components_not_its_own_mask():
    from product.parallel_rebuild import source_owner
    from product.object_contracts import ClosedObject
    src=np.full((40,40,3),255,np.uint8);src[10:20,10:20]=225;src[11:19,11:19]=[0,0,200]
    rgba=np.zeros((40,40,4),np.uint8);rgba[11:19,11:19]=[0,0,200,255]
    obj=ClosedObject(id='cloud',description='cloud',prompt_en='cloud',bbox=[10,10,10,10],layers=['layer-01.png'],
        text_policy='exclude_independent_text',components=['cloud'],kind='icon',parts=[{'name':'cloud','bbox':[10,10,10,10],'layers':['layer-01.png']}])
    surface=SurfaceEvidence.model_validate({'schema':'SurfaceEvidence/1','source_sha256':'a'*64,'image_size':[40,40],'method_version':'test','background':'FFFFFF'})
    box,bg,mask,proof=source_owner(obj,rgba,src,np.zeros((40,40),bool),surface)
    assert int(mask.sum())==100 and int((rgba[...,3]>0).sum())==64 and proof['unowned_source_ink_pixels']==0
    src[10,10]=[0,0,200]
    with pytest.raises(ValueError,match='strong strokes'):source_owner(obj,rgba,src,np.zeros((40,40),bool),surface)


def test_pure_unmix_refuses_half_old_cloud_collection(tmp_path):
    from product.object_unmix import candidate_from_evidence
    from product.object_contracts import ClosedObject
    obj=ClosedObject(id='cloud',description='cloud',prompt_en='cloud',bbox=[10,10,10,10],layers=['layer-01.png'],
        text_policy='exclude_independent_text',components=['cloud'],kind='icon',parts=[{'name':'cloud','bbox':[10,10,10,10],'layers':['layer-01.png']}],
        replaces=['cloudleft','cloudright'],required_owners=['cloudleft','cloudright'])
    with pytest.raises(ValueError,match='incomplete old owner'):
        candidate_from_evidence(obj,np.full((40,40,3),255,np.uint8),np.zeros((40,40),bool),owners=[{'id':'cloudleft'}])


def test_provider_references_are_never_filesystem_authority(tmp_path):
    class Actor:
        model='fixture'
        def ask(self,*args):return {'schema':'TextEvidence/1','source_sha256':'a'*64,'image_size':[40,30],
            'method_version':'test','references':[str(tmp_path/'private.json')]}
    s=Store(tmp_path/'db');result=s.call('text',Actor(),'JSON',{},[],TextEvidence,nullcontext())
    assert str(tmp_path/'private.json') not in result.references
    adapter=next((s.root/'reference-adapters').glob('*.json'))
    assert json.loads(adapter.read_text())['filesystem_authority'] is False


def test_network_receipt_crash_recovers_locally_without_paid_resend(tmp_path):
    class Actor:
        model='receipt-fixture';calls=0
        def ask_receipted(self,system,context,images,contract,receipt):
            if not receipt.exists():
                self.calls+=1
                write(receipt,{'schema':'TextEvidence/1','source_sha256':'a'*64,'image_size':[40,30],'method_version':'test'})
                raise InterruptedError('crashed after durable receipt')
            return json.loads(receipt.read_text())
    s=Store(tmp_path/'db');actor=Actor()
    with pytest.raises(InterruptedError):s.call('x',actor,'JSON',{},[],TextEvidence,nullcontext())
    assert s.call('x',actor,'JSON',{},[],TextEvidence,nullcontext()).source_sha256=='a'*64
    assert actor.calls==1


def test_graphic_protection_uses_actual_alpha_not_logo_rectangle(tmp_path):
    from product.parallel_pixels import graphic_ink
    rgba=np.zeros((20,30,4),np.uint8);rgba[3:7,4:10]=[255,255,255,128]
    file=tmp_path/'logo.png';Image.fromarray(rgba).save(file)
    mask=graphic_ink([{'id':'logo','path':str(file),'sha256':digest(file),'source_sha256':'a'*64,'bbox':[5,6,30,20]}],(50,40),'a'*64)
    assert mask.sum()==24 and not mask[6,5] and mask[9,9]


def test_boundary_change_is_repaired_even_with_faithful_source_fallback(tmp_path):
    from product.parallel_repair import local_findings,route
    from product.object_contracts import ClosedObject
    p=tmp_path/'s.png';Image.new('RGB',(80,60),'white').save(p);snap,b=branches(p)
    b['asset'].objects=[ClosedObject(id='cloud',description='cloud with label',prompt_en='cloud',bbox=[0,0,70,50],layers=['layer-01.png'],
        text_policy='intrinsic_label',intrinsic_text='中心算力',components=['cloud'],kind='icon',
        parts=[{'name':'cloud','bbox':[0,0,70,50],'layers':['layer-01.png']}])]
    g=fuse(snap,b,{'lines':[]})
    f=local_findings(g,[],[{'id':'cloud','bbox':[10,10,30,20],'definition':{'description':'cloud, excluding label and base','text_policy':'exclude_independent_text'}}])
    assert len(f)==1 and route(g,f)=={'asset'} and 'boundary' in f[0]['description']


def test_final_office_failure_reverts_all_pages_and_publishes_complete_source(tmp_path,monkeypatch):
    from product import parallel_pipeline as mod,parallel_repair
    job=tmp_path/'job';(job/'inputs').mkdir(parents=True)
    for name,color in [('01.png','white'),('02.png','#EEEEEE')]:Image.new('RGB',(40,30),color).save(job/'inputs'/name)
    write(job/'request.json',{'sources':['01.png','02.png'],'pipeline_mode':'parallel_review','object_extraction':'off'})
    monkeypatch.setenv('IMAGE2PPT_PARALLEL_CACHE',str(tmp_path/'cache'))
    monkeypatch.setattr(mod,'perceive',lambda *a,**k:(_ for _ in ()).throw(ValueError('injected branch failure')))
    monkeypatch.setattr(parallel_repair,'repair',lambda *a,**k:(_ for _ in ()).throw(ValueError('injected repair failure')))
    class Actor:
        model='failure-fixture';calls=0
        def ask(self,system,ctx,images,contract):
            self.calls+=1;fail=self.calls==2
            return {'schema':'VisualReview/1',**{k:ctx[k] for k in ('source_sha256','plan_sha256','preview_sha256')},
                'passed':not fail,'findings':[{'category':'graphics','node_ids':[],
                    'description':'injected visual failure','suggested_fix':'keep source'}] if fail else [],
                'checked':['wording','structure','font','emphasis','layout','graphics']}
    def compiler(irs,output,pages):
        output.mkdir();(output/'preview').mkdir();(output/'fixture.pptx').write_bytes(b'fixture')
        for i in range(2):Image.new('RGB',(40,30),'white').save(output/'preview'/f'{i+1:02d}.png')
        return {'pptx':str(output/'fixture.pptx'),'preview':str(output/'preview')}
    def office(pptx,irs,output):
        if output.name=='office-final':raise RuntimeError('injected final Office loss')
        output.mkdir()
        for i in range(2):Image.new('RGB',(40,30),'white').save(output/f'page-{i+1:02d}-before.png')
        return {'status':'PASS_APP_AUTOMATED','synthetic':True}
    r=mod.run(job,provider=Actor(),review_provider=Actor(),compiler=compiler,office=office,fonts={'fonts':[]})
    assert r['reconstructed_pages']==0 and r['raster_fallback_pages']==[1,2]
    assert all('final-safe-source' in p for p in r['slide_irs'])
    commit=Store(job/'state').committed()
    assert any(p.endswith('scene.json') for p in commit['files'])


@pytest.mark.parametrize('target',['../media/icon.png','/ppt/media/icon.png'])
def test_office_picture_verifier_handles_relative_and_absolute_media(tmp_path,target):
    import zipfile
    from product.parallel_office import verify_pictures
    rgba=np.zeros((6,8,4),np.uint8);rgba[1:5,2:7]=[255,255,255,128]
    asset=tmp_path/'icon.png';Image.fromarray(rgba).save(asset)
    ir=tmp_path/'ir.json';write(ir,{'assets':[{'id':'icon','path':'icon.png','sha256':digest(asset)}],
        'nodes':[{'id':'logo','type':'image','asset_id':'icon'}]})
    pptx=tmp_path/'fixture.pptx'
    xml='<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:pic><p:nvPicPr><p:cNvPr name="logo"/></p:nvPicPr><p:blipFill><a:blip r:embed="rId1"/></p:blipFill></p:pic></p:sld>'
    with zipfile.ZipFile(pptx,'w') as z:
        z.writestr('ppt/slides/slide1.xml',xml)
        z.writestr('ppt/slides/_rels/slide1.xml.rels',f'<Relationships><Relationship Id="rId1" Target="{target}"/></Relationships>')
        z.write(asset,'ppt/media/icon.png')
    verify_pictures(pptx,[ir])
    rgba[1,2,3]=255;Image.fromarray(rgba).save(asset)
    with pytest.raises(ValueError):verify_pictures(pptx,[ir])


def test_review_cache_binds_actual_request_parameters(tmp_path):
    class Actor:
        model='fixture';request_binding={'temperature':0};calls=0
        def ask(self,*args):
            self.calls+=1
            return {'schema':'TextEvidence/1','source_sha256':'a'*64,'image_size':[40,30],'method_version':'test'}
    s=Store(tmp_path/'db');actor=Actor()
    s.call('review',actor,'JSON',{},[],TextEvidence,nullcontext())
    actor.request_binding={'temperature':.1}
    s.call('review',actor,'JSON',{},[],TextEvidence,nullcontext())
    s.call('review',actor,'JSON',{},[],TextEvidence,nullcontext())
    assert actor.calls==2


def test_substantive_repair_budget_ignores_receipt_paths_but_counts_changed_evidence(tmp_path,monkeypatch):
    from product import parallel_repair as mod
    from product.parallel_store import BudgetExceeded
    p=tmp_path/'source.png';Image.new('RGB',(40,30),'white').save(p);snap,b=branches(p);g=fuse(snap,b,{'lines':[]})
    s=Store(tmp_path/'db');observed=[]
    def perceive(*args,**kwargs):
        observed.append(kwargs['repair']['text'])
        assert not observed[-1]['previous_evidence']['references']
        raise RuntimeError('fixture stopping before any model or reconstruction')
    monkeypatch.setattr(mod,'perceive',perceive)
    def invoke():mod.repair(g,snap,[{'category':'font','node_ids':[]}],s,None,{},tmp_path,'unused',tmp_path/'stage',{'pages':[],'stable_assets':[]},'auto')
    g.text.references=['receipt-A']
    with pytest.raises(RuntimeError):invoke()
    g.text.references=['receipt-B']
    with pytest.raises(RuntimeError):invoke()  # Same meaningful input: no new budget.
    g.text.limitations=['changed semantic evidence']
    with pytest.raises(RuntimeError):invoke()
    g.text.limitations=['another substantive change']
    with pytest.raises(BudgetExceeded):invoke()
    assert len(observed)==3


def test_identical_repair_ir_stops_recompiling_and_retains_parent(tmp_path,monkeypatch):
    from product import parallel_pipeline as mod,parallel_repair
    job=tmp_path/'job';(job/'inputs').mkdir(parents=True);Image.new('RGB',(40,30),'white').save(job/'inputs/01.png')
    write(job/'request.json',{'sources':['01.png'],'pipeline_mode':'parallel_review','object_extraction':'off'})
    monkeypatch.setenv('IMAGE2PPT_PARALLEL_CACHE',str(tmp_path/'cache'))
    monkeypatch.setattr(mod,'perceive',lambda *a,**k:(_ for _ in ()).throw(ValueError('fixture')))
    calls=[];captured=[]
    class Actor:
        model='no-op-fixture'
        def ask(self,system,ctx,images,contract):
            return {'schema':'VisualReview/1',**{k:ctx[k] for k in ('source_sha256','plan_sha256','preview_sha256')},'passed':False,
                'findings':[{'category':'graphics','node_ids':[],'description':'fixture defect','suggested_fix':'retain source'}],
                'checked':['wording','structure','font','emphasis','layout','graphics']}
    def compiler(irs,output,pages):
        calls.append('compile');captured[:]=irs;output.mkdir();(output/'preview').mkdir();(output/'fixture.pptx').write_bytes(b'fixture')
        Image.new('RGB',(40,30),'white').save(output/'preview/01.png')
        return {'pptx':str(output/'fixture.pptx'),'preview':str(output/'preview')}
    def office(pptx,irs,output):
        output.mkdir();Image.new('RGB',(40,30),'white').save(output/'page-01-before.png');return {'status':'PASS_APP_AUTOMATED'}
    def repair(graph,*args,**kwargs):calls.append('repair');return graph,captured[0],[]
    monkeypatch.setattr(parallel_repair,'repair',repair)
    mod.run(job,provider=Actor(),review_provider=Actor(),compiler=compiler,office=office,fonts={'fonts':[]})
    assert calls==['compile','repair','compile']
    assert json.loads((job/'repair-no-change-0-01.json').read_text())['parent_retained'] is True
