import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from test_object_closure import prepared
from product.object_contracts import read_plan
from product.object_unmix import candidate
from product.object_closure import extract_closed
from product.object_finalize import object_pass
from product.materialize import write,digest


def test_running_worker_keeps_its_initial_algorithm_binding(monkeypatch):
    from product import object_unmix
    initial=object_unmix.binding()
    def changed_file(*args):raise AssertionError('re-read changing source during a job')
    monkeypatch.setattr(object_unmix,'digest',changed_file)
    assert object_unmix.binding()==initial


def test_shared_semi_automatic_unmix_recomposes_original_and_keeps_holes(tmp_path):
    layers,base,file=prepared(tmp_path)
    plan,_=read_plan(json.loads(file.read_text()));ir=json.loads((base/'slideir.json').read_text())
    source=np.array(Image.open(base/'source.png').convert('RGB'))
    matte,meta,backing,_,_=candidate(plan.objects[0],source,ir,base,np.zeros(source.shape[:2],bool))
    bg=np.full_like(source,255);alpha=matte[...,3:4]/255
    actual=matte[...,:3]*alpha+bg*(1-alpha)
    assert np.abs(actual-source).max()<2 and matte[15,15,3]>0
    assert matte[21,15,3]==0 and meta['estimated_alpha']
    assert meta['reconstruction_p95']<2 and backing[0]['source_unmix_scope_hypothesis']
    assert np.array_equal(np.array(Image.open(base/'source.png')),source)


def test_unmix_protected_letter_is_rejected_without_erasing_a_hole(tmp_path):
    layers,base,file=prepared(tmp_path);plan,_=read_plan(json.loads(file.read_text()))
    ir=json.loads((base/'slideir.json').read_text());source=np.array(Image.open(base/'source.png').convert('RGB'))
    blocked=np.zeros(source.shape[:2],bool);blocked[15,15]=True
    with pytest.raises(ValueError,match='protected_content'):candidate(plan.objects[0],source,ir,base,blocked)


def test_white_logo_ink_on_coloured_backing_is_preserved(tmp_path):
    layers,base,file=prepared(tmp_path);plan,_=read_plan(json.loads(file.read_text()));ir=json.loads((base/'slideir.json').read_text())
    source=np.empty((40,60,3),np.uint8);source[:]=[30,80,120]
    source[10:20,10:20]=255;source[13:17,13:17]=[30,80,120];source[23:25,12:17]=255
    Image.fromarray(source[8:28,8:23]).save(base/'crop.png');ir['background']={'color':'1E5078'}
    matte,_,_,_,_=candidate(plan.objects[0],source,ir,base,np.zeros(source.shape[:2],bool))
    assert tuple(matte[11,11])==(255,255,255,255) and matte[15,15,3]==0


def test_source_unmix_requires_its_own_independent_verdict(tmp_path):
    layers,base,file=prepared(tmp_path)
    ir,report=extract_closed(file,layers,base/'slideir.json',tmp_path/'out',source_unmix_only=True)
    row=report['objects'][0];assert row['status']=='STAGED' and row['method']=='source-unmix'
    assert row['source_unmix']['estimated_alpha']
    verdict={k:'PASS' for k in ('pixels','complete_object','replacement','background_restoration')}
    assert not object_pass(row,verdict)
    verdict['source_unmix']='UNKNOWN';assert not object_pass(row,verdict)
    verdict['source_unmix']='PASS';assert object_pass(row,verdict)


def test_code_provenance_alone_does_not_repeat_identical_visual_review(tmp_path):
    from product.object_finalize import review_objects
    layers,base,file=prepared(tmp_path)
    ir,report=extract_closed(file,layers,base/'slideir.json',tmp_path/'out',source_unmix_only=True)
    report['state_directory']=str(tmp_path/'state');calls=[]
    def ask(context,images,contract):
        calls.append(context)
        assert 'method_binding' not in context['objects'][0]['source_unmix']
        return {'schema':'ObjectReview/1','source_sha256':context['source_sha256'],'scene_sha256':context['scene_sha256'],
            'objects':[{'id':'icon','candidate_sha256':report['objects'][0]['asset_sha256'],
                'pixels':'PASS','complete_object':'PASS','replacement':'PASS','background_restoration':'PASS','source_unmix':'PASS','reasons':[]}]}
    review_objects(layers/'source.png',layers/'source.png',ir,report,tmp_path/'review-a',ask,review_model='synthetic-model')
    report['objects'][0]['source_unmix']=json.loads(json.dumps(report['objects'][0]['source_unmix']))
    report['objects'][0]['source_unmix']['method_binding']['adapter_sha256']='different-private-code-provenance'
    review_objects(layers/'source.png',layers/'source.png',ir,report,tmp_path/'review-b',ask,review_model='synthetic-model')
    assert len(calls)==1


def test_prior_alpha_distinguishes_real_white_fill_from_a_real_hole(tmp_path):
    layers,base,file=prepared(tmp_path);plan,_=read_plan(json.loads(file.read_text()))
    ir=json.loads((base/'slideir.json').read_text());source=np.array(Image.open(base/'source.png').convert('RGB'))
    source[14,14]=255;source[16,16]=255
    old=np.array(Image.open(layers/'layer-01.png'))[8:28,8:23].copy()
    old[6,6]=[255,255,255,255];old[8,8]=0
    Image.fromarray(old).save(base/'crop.png');before=digest(base/'crop.png')
    matte,meta,_,_,_=candidate(plan.objects[0],source,ir,base,np.zeros(source.shape[:2],bool))
    assert tuple(matte[14,14])==(255,255,255,255)
    assert matte[16,16,3]==0 and meta['retained_old_alpha_pixels']>0
    assert digest(base/'crop.png')==before


def test_grouped_old_representation_cannot_be_silently_flattened(tmp_path):
    layers,base,file=prepared(tmp_path);plan,_=read_plan(json.loads(file.read_text()))
    ir=json.loads((base/'slideir.json').read_text());ir['groups']=[{'id':'g','members':['old']}]
    source=np.array(Image.open(base/'source.png').convert('RGB'))
    with pytest.raises(ValueError,match='group_transaction'):
        candidate(plan.objects[0],source,ir,base,np.zeros(source.shape[:2],bool))


def test_original_source_can_recover_a_logo_missing_from_layerd(tmp_path):
    layers,base,file=prepared(tmp_path)
    Image.fromarray(np.zeros((40,60,4),np.uint8)).save(layers/'layer-01.png')
    manifest=json.loads((layers/'manifest.json').read_text());manifest['layers'][1]['sha256']=digest(layers/'layer-01.png')
    write(layers/'manifest.json',manifest)
    _,report=extract_closed(file,layers,base/'slideir.json',tmp_path/'out',source_unmix_only=True)
    row=report['objects'][0]
    assert row['status']=='STAGED' and row['method']=='source-unmix'
    assert row['source_unmix']['reconstruction_mae']<1


def test_part_evidence_expands_a_small_localization_box_without_cutting_strokes(tmp_path):
    layers,base,file=prepared(tmp_path);plan,_=read_plan(json.loads(file.read_text()))
    obj=plan.objects[0].model_copy(update={'bbox':[12,12,3,3],'replaces':[],'required_owners':[]})
    ir=json.loads((base/'slideir.json').read_text());ir['nodes']=[];ir['assets']=[]
    source=np.array(Image.open(base/'source.png').convert('RGB'))
    matte,_,_,_,_=candidate(obj,source,ir,base,np.zeros(source.shape[:2],bool))
    assert matte[10,10,3]>0 and matte[24,14,3]>0


def test_upload_auto_strategy_is_opt_in_and_preserves_existing_default(monkeypatch):
    import io,uuid
    from fastapi.testclient import TestClient
    from product import pipeline
    from product.app import create_app,ROOT
    monkeypatch.setattr(pipeline,'settings',lambda:object())
    monkeypatch.setenv('OPENCODE_GO_API_KEY','synthetic-test-key')
    monkeypatch.setenv('LAYERD_TOKEN_FILE','synthetic-token-file-no-network')
    monkeypatch.setenv('IMAGE2PPT_OBJECT_EXTRACTION','off')
    root=ROOT/'outputs'/('unmix-api-test-'+uuid.uuid4().hex)
    app=create_app(root,'x'*32,run_worker=False)
    blob=io.BytesIO();Image.new('RGB',(800,450),'white').save(blob,format='PNG')
    with TestClient(app) as client:
        for query,expected in (('', 'off'),('?object_extraction=auto','auto')):
            response=client.post('/api/jobs'+query,headers={'Authorization':'Bearer '+'x'*32},
                files=[('files',('page.png',blob.getvalue(),'image/png'))])
            assert response.status_code==202
            request=json.loads((root/response.json()['id']/'request.json').read_text())
            assert request['object_extraction']==expected


def test_parent_render_gate_detects_a_real_non_target_change(tmp_path):
    from product.object_finalize import parent_region_check
    _,base,_=prepared(tmp_path);parent=base/'slideir.json'
    before=tmp_path/'before.png';after=tmp_path/'after.png'
    Image.new('RGB',(60,40),'white').save(before)
    pixels=np.array(Image.open(before));pixels[12,12]=0;Image.fromarray(pixels).save(after)
    report={'parent_ir_sha256':digest(parent),'objects':[{'status':'STAGED','bbox':[10,10,10,10],'replaces':['old']}]}
    assert parent_region_check(parent,report,before,after)['status']=='PASS'
    pixels[35,50]=0;Image.fromarray(pixels).save(after)
    result=parent_region_check(parent,report,before,after)
    assert result['status']=='FAIL' and result['non_target_changed_pixels']==1
    report['parent_ir_sha256']='wrong'
    with pytest.raises(ValueError,match='binding'):parent_region_check(parent,report,before,after)


def test_product_prefers_bundled_node_over_ambient_node(tmp_path,monkeypatch):
    import os
    from product import pipeline
    profile=tmp_path/'profile';name='node.exe' if os.name=='nt' else 'node'
    bundled=profile/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin'/name
    bundled.parent.mkdir(parents=True);bundled.write_bytes(b'synthetic-executable-not-launched')
    monkeypatch.setattr(Path,'home',lambda:profile)
    monkeypatch.delenv('VISION_NODE',raising=False);monkeypatch.delenv('IMAGE2PPT_NODE',raising=False)
    monkeypatch.setattr(pipeline,'_compiler',lambda root:(root,Path('ambient-node')))
    assert pipeline.compiler_runtime()[1]==bundled


def test_source_matte_does_not_crowd_out_sam_white_content_hypothesis(tmp_path,monkeypatch):
    layers,base,file=prepared(tmp_path)
    fg=np.array(Image.open(layers/'layer-01.png'));fg[13:17,13:17]=0
    Image.fromarray(fg).save(layers/'layer-01.png')
    source=Image.alpha_composite(Image.open(layers/'layer-00.png'),Image.fromarray(fg)).convert('RGB')
    source.save(layers/'source.png');source.save(base/'source.png');source.crop((8,8,23,28)).save(base/'crop.png')
    manifest=json.loads((layers/'manifest.json').read_text());manifest['source_sha256']=digest(layers/'source.png')
    manifest['layers'][1]['sha256']=digest(layers/'layer-01.png');write(layers/'manifest.json',manifest)
    raw=json.loads(file.read_text());raw['source_sha256']=digest(layers/'source.png')
    raw['objects'][0]['layers']=['layer-00.png','layer-01.png'];write(file,raw)
    monkeypatch.setattr('product.object_closure.load_sam',lambda *a,**k:([{'index':0,'mask':fg[...,3]>0}],None))
    _,report=extract_closed(file,layers,base/'slideir.json',tmp_path/'out',sam_dir=tmp_path,candidate_offsets={'icon':99})
    methods=[r['method'] for r in report['objects'][0]['candidates']]
    assert len(methods)<=6 and 'source-unmix' in methods
    assert 'sam3-mask-0-source-alpha-holes-native-owner-bg' in methods


@pytest.mark.parametrize('transient',[True,False])
def test_only_first_party_import_timeouts_receive_one_local_retry(tmp_path,monkeypatch,transient):
    from types import SimpleNamespace
    from product import pipeline
    monkeypatch.setattr(pipeline,'compiler_runtime',lambda:(pipeline.ROOT,Path('node.exe')))
    calls=[]
    def invoke(command,**kwargs):
        calls.append(command)
        if len(calls)==1:
            kwargs['stderr'].write('verifyFirstPartyImport spawnSync ETIMEDOUT' if transient else 'invalid source IR')
            return SimpleNamespace(returncode=1)
        kwargs['stdout'].write('{}');return SimpleNamespace(returncode=0)
    monkeypatch.setattr(pipeline.subprocess,'run',invoke)
    monkeypatch.setattr(pipeline,'_compiled_artifacts',lambda *args:{'pptx':'synthetic.pptx','preview':'synthetic-preview'})
    monkeypatch.setattr(pipeline,'verify_package',lambda *args:None)
    output=tmp_path/'compile'
    if transient:
        assert pipeline.compile_pages([],output,[])['pptx']=='synthetic.pptx'
        assert len(calls)==2 and (output/'compiler-attempt-0.stderr').exists()
    else:
        with pytest.raises(pipeline.PipelineError,match='fixed_compiler_failed'):pipeline.compile_pages([],output,[])
        assert len(calls)==1 and not (output/'compile-retry-0.json').exists()


def test_source_unmix_does_not_regenerate_retained_asset(tmp_path,monkeypatch):
    layers,base,file=prepared(tmp_path);ir,report=extract_closed(file,layers,base/'slideir.json',tmp_path/'first')
    record=report['objects'][0];original=digest(ir.parent/record['asset'])
    def forbidden(*args,**kwargs):raise AssertionError('stable matte was regenerated')
    monkeypatch.setattr('product.object_unmix.candidate',forbidden)
    fresh,updated=extract_closed(file,layers,ir,tmp_path/'next',source_unmix_only=True,retained={'icon':record})
    assert updated['objects'][0]['status']=='STABLE_REUSED'
    assert digest(fresh.parent/record['asset'])==original


def test_new_parent_adapter_requires_all_halves_of_an_already_closed_object(tmp_path):
    from product.object_parent import adapt_owners
    layers,base,file=prepared(tmp_path);plan,_=read_plan(json.loads(file.read_text()))
    obj=plan.objects[0].model_copy(update={'id':'cloud','replaces':['left','right'],'required_owners':['left','right']})
    closure={'objects':[{'id':'cloud','status':'STAGED','replaces':['left','right'],'definition':{'replaces':['left','right']}}]}
    ir={'nodes':[{'id':'extracted-cloud'}]};history={'objects':[]}
    adapted,_=adapt_owners(plan.model_copy(update={'objects':[obj]}),ir,history,closure=closure)
    assert adapted.objects[0].replaces==['extracted-cloud']
    half=obj.model_copy(update={'replaces':['left'],'required_owners':['left']})
    with pytest.raises(ValueError,match='split historical'):adapt_owners(plan.model_copy(update={'objects':[half]}),ir,history,closure=closure)
