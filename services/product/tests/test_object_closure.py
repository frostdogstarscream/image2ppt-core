import copy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_object_extraction import fixture
from product.materialize import digest,write
from product.object_contracts import read_plan
from product.object_closure import extract_closed,components
from product.object_registry import Registry,pixel_key,scene_key
from product.object_scheduler import ObjectSchedule,repair_route
from product.object_ownership import image_pixels,text_evidence


def prepared(tmp_path):
    layers,base,plan=fixture(tmp_path)
    ir=json.loads((base/'slideir.json').read_text());ir['background']={'color':'FFFFFF'}
    write(base/'slideir.json',ir)
    return layers,base,plan


def test_closed_stage_preserves_soft_alpha_and_never_self_accepts(tmp_path):
    layers,base,plan=prepared(tmp_path)
    file,report=extract_closed(plan,layers,base/'slideir.json',tmp_path/'out')
    assert report['staged']==1 and report['automatic_quality']=='REVIEW_REQUIRED'
    row=report['objects'][0];assert row['status']=='STAGED' and not row['human_acceptance']
    alpha=np.array(Image.open(file.parent/row['asset']))[...,3]
    assert 90 in alpha and 0 in alpha
    assert not (tmp_path/'registry/stable.json').exists()


def test_final_page_review_reuses_exact_binding_and_keeps_uncertain_reservation(tmp_path):
    from object_closure_deliver import cached_page_review
    image=tmp_path/'image.png';Image.new('RGB',(10,10),'white').save(image)
    context={'source_sha256':'source','plan_sha256':'plan','preview_sha256':digest(image),'plan':{}}
    previous=tmp_path/'previous.json';write(previous,context);calls=[]
    def ask(*args):calls.append(args);return context
    cached_page_review(context,[image],tmp_path/'cache','go',ask,previous_raw=previous,previous_model='go')
    assert not calls
    cached_page_review(context,[image],tmp_path/'cache','other-model',ask,previous_raw=previous,previous_model='go')
    cached_page_review(context,[image],tmp_path/'cache','other-model',ask)
    assert len(calls)==1
    def interrupted(*args):calls.append(args);raise RuntimeError('interrupted after reservation')
    with pytest.raises(RuntimeError):cached_page_review(context,[image],tmp_path/'failed','go',interrupted)
    with pytest.raises(RuntimeError,match='already reserved'):cached_page_review(context,[image],tmp_path/'failed','go',interrupted)
    assert len(calls)==2


def test_transparent_white_is_owned_not_background(tmp_path):
    layers,base,plan=prepared(tmp_path);ir=json.loads((base/'slideir.json').read_text())
    rgba=np.zeros((20,15,4),np.uint8);rgba[2:12,2:12]=255
    Image.fromarray(rgba).save(base/'crop.png')
    mask,known=image_pixels(ir,ir['nodes'][0],base,(60,40),np.array(Image.open(base/'source.png')))
    assert known and mask[11,11] and not mask[8,8]


def test_text_rectangle_blank_is_not_ink(tmp_path):
    layers,base,plan=prepared(tmp_path);ir=json.loads((base/'slideir.json').read_text())
    ir['nodes'].append({'id':'body','type':'text','text':'text','bbox':[9,19,20,10],'z':2,'color':'000000'})
    write(base/'slideir.json',ir)
    hard,uncertain,_=text_evidence(ir,np.array(Image.open(base/'source.png')),(60,40))
    assert not hard.any() and not uncertain.any()
    _,report=extract_closed(plan,layers,base/'slideir.json',tmp_path/'out')
    assert report['staged']==1


def test_real_text_colour_leak_rejects_without_punching_holes(tmp_path):
    layers,base,plan=prepared(tmp_path);ir=json.loads((base/'slideir.json').read_text())
    ir['nodes'].append({'id':'caption','type':'text','text':'X','bbox':[10,10,10,10],'z':2,'color':'1432B4'})
    write(base/'slideir.json',ir)
    file,report=extract_closed(plan,layers,base/'slideir.json',tmp_path/'out')
    assert report['fallback']==1 and 'uncertain_text_ownership' in report['objects'][0]['reasons']
    assert json.loads(file.read_text())['nodes']==ir['nodes']


def test_partial_cloud_owner_set_rejects_all(tmp_path):
    layers,base,plan=prepared(tmp_path);ir=json.loads((base/'slideir.json').read_text())
    # Two transparent fragments of the same source object.
    source=Image.open(base/'source.png');source.crop((8,8,15,28)).save(base/'left.png');source.crop((15,8,23,28)).save(base/'right.png')
    ir['assets']=[{'id':'a','path':'left.png'},{'id':'b','path':'right.png'}]
    ir['nodes']=[{'id':'cloudleft','type':'image','bbox':[8,8,7,20],'z':0,'asset_id':'a'},
        {'id':'cloudright','type':'image','bbox':[15,8,8,20],'z':0,'asset_id':'b'}]
    write(base/'slideir.json',ir)
    p=json.loads(plan.read_text());p['objects'][0]['replaces']=['cloudleft'];write(plan,p)
    file,report=extract_closed(plan,layers,base/'slideir.json',tmp_path/'out')
    assert report['staged']==0 and 'incomplete_owner_set' in report['objects'][0]['reasons']
    assert json.loads(file.read_text())['nodes']==ir['nodes']
    p['objects'][0]['replaces'].append('cloudright');write(plan,p)
    file,report=extract_closed(plan,layers,base/'slideir.json',tmp_path/'complete')
    assert report['staged']==1 and len(json.loads(file.read_text())['nodes'])==1


def test_missing_evidence_registry_never_publishes(tmp_path):
    layers,base,plan=prepared(tmp_path);file,report=extract_closed(plan,layers,base/'slideir.json',tmp_path/'out')
    registry=Registry(tmp_path/'registry')
    with pytest.raises(ValueError,match='incomplete object gates'):registry.publish(file,report['objects'],{},expected_generation=0)
    assert registry.current()=={'generation':0,'objects':{}}


def test_pixel_cache_ignores_object_rename_but_not_layer_or_mask(tmp_path):
    layers,base,planfile=prepared(tmp_path);plan,_=read_plan(json.loads(planfile.read_text()));obj=plan.objects[0]
    k=pixel_key(plan.source_sha256,{'layer':'hash'},obj)
    renamed=obj.model_copy(update={'id':'renamed','description':'new words','replaces':['newowner']})
    assert pixel_key(plan.source_sha256,{'layer':'hash'},renamed)==k
    assert pixel_key(plan.source_sha256,{'layer':'newhash'},renamed)!=k
    changed=obj.model_copy(update={'bbox':[8,8,16,20]})
    assert pixel_key(plan.source_sha256,{'layer':'hash'},changed)!=k


def test_budget_is_per_object_and_deduplicates_identical_prompt(tmp_path):
    schedule=ObjectSchedule(tmp_path/'schedule.json')
    call,fresh=schedule.reserve('object','sam_prompt',{'prompt':'one'});assert fresh
    schedule.finish(call,'SUCCEEDED','cached')
    assert not ObjectSchedule(schedule.path).reserve('object','sam_prompt',{'prompt':'one'})[1]
    assert schedule.reserve('object','sam_prompt',{'prompt':'two'})[1]
    assert schedule.reserve('object','sam_prompt',{'prompt':'three'})==(None,False)
    assert repair_route(['protected_text_ink'])=='local_text_evidence'
    assert repair_route(['incomplete_owner_set'])=='replan_owners'
    assert repair_route(['wrong_layer_selected'])=='replan_layers'


def test_parts_do_not_clip_components_or_union_neighbors(tmp_path):
    _,_,file=prepared(tmp_path);plan,_=read_plan(json.loads(file.read_text()));part=plan.objects[0].parts[0]
    alpha=np.zeros((40,60),np.uint8);alpha[10:12,10:30]=255
    selected,bad=components(alpha,[part],[],(60,40))
    assert selected[10,29] and not bad
    negative=part.model_copy(update={'bbox':[25,10,5,2]})
    selected,bad=components(alpha,[part],[negative],(60,40))
    assert not selected.any() and bad


def test_sam_remote_cache_ignores_names_but_binds_model():
    file=Path(__file__).resolve().parents[2]/'segmentation/service.py'
    spec=importlib.util.spec_from_file_location('sam_service',file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p={'source_sha256':'a'*64,'image_size':[60,40],'objects':[{'id':'cloud','bbox':[5,5,20,20],'prompt_en':'a cloud','description':'old'}]}
    key,_,mapping=module.canonical(p,'a'*64,{'sha256':'model'})
    p['objects'][0].update(id='renamed',description='new')
    assert module.canonical(p,'a'*64,{'sha256':'model'})[0]==key
    assert module.canonical(p,'a'*64,{'sha256':'other'})[0]!=key
    assert mapping['cloud']=='obj-000'


def test_multimask_sam_never_automatically_unions(tmp_path):
    from product.object_extract import load_sam
    layers,base,file=prepared(tmp_path);plan,_=read_plan(json.loads(file.read_text()));obj=plan.objects[0]
    root=tmp_path/'sam';root.mkdir();masks=np.zeros((2,40,60),bool)
    masks[0,10:12,10:12]=True;masks[1,15:17,15:17]=True
    np.savez_compressed(root/'mask.npz',masks=masks,scores=np.array([.9,.8]))
    write(root/'icon.json',{'status':'SUCCEEDED','source_sha256':plan.source_sha256,'plan_sha256':digest(file),
        'version':'sam3','image_size':[60,40],'mask_file':'mask.npz','mask_sha256':digest(root/'mask.npz'),
        'checkpoint_sha256':'a'*64,'code_revision':'b'*40})
    mask,error=load_sam(root,obj,plan.source_sha256,digest(file),(60,40),'sam3')
    assert error is None and mask.sum()==4
    candidates,error=load_sam(root,obj,plan.source_sha256,digest(file),(60,40),'sam3',individual=True)
    assert len(candidates)==2 and all(c['mask'].sum()==4 for c in candidates)


def test_targeted_repairs_cannot_touch_neighbors(tmp_path):
    from product.object_contracts import apply_repairs
    _,_,file=prepared(tmp_path);plan,_=read_plan(json.loads(file.read_text()))
    raw={'schema':'ObjectRepairs/1','source_sha256':plan.source_sha256,'image_size':plan.image_size,
        'objects':[plan.objects[0].model_dump()]}
    assert apply_repairs(plan,raw,['icon'])==plan
    with pytest.raises(ValueError,match='non-target'):apply_repairs(plan,raw,['other'])


def test_compile_failure_cannot_publish_or_mutate_parent(tmp_path,monkeypatch):
    from product.object_finalize import finalize
    layers,base,file=prepared(tmp_path);ir,report=extract_closed(file,layers,base/'slideir.json',tmp_path/'stage')
    before=digest(base/'slideir.json');report['state_directory']=str(tmp_path/'state')
    monkeypatch.setattr('product.object_finalize.effective_page',lambda page,file:page)
    def broken(*args):raise RuntimeError('compiler failed')
    with pytest.raises(RuntimeError,match='compiler failed'):
        finalize([layers/'source.png'],[ir],[base/'slideir.json'],[report],[None],tmp_path/'qa',broken,lambda *args:None)
    assert digest(base/'slideir.json')==before and not (tmp_path/'state/registry/stable.json').exists()


def test_wrong_group_transaction_keeps_native_group(tmp_path):
    layers,base,file=prepared(tmp_path);ir=json.loads((base/'slideir.json').read_text())
    ir['groups']=[{'id':'real-group','members':['old']}];write(base/'slideir.json',ir)
    result,report=extract_closed(file,layers,base/'slideir.json',tmp_path/'out')
    assert report['fallback']==1 and json.loads(result.read_text())['groups']==ir['groups']


def test_corrupt_stable_asset_is_not_recropped(tmp_path):
    registry=Registry(tmp_path/'registry')
    write(registry.pointer,{'generation':1,'objects':{'key':{'sha256':'a'*64}}})
    directory=registry.root/'assets';directory.mkdir();(directory/('a'*64+'.png')).write_bytes(b'wrong')
    with pytest.raises(ValueError,match='hash mismatch'):registry.lookup('key')


def test_history_adapter_refuses_splitting_an_existing_whole(tmp_path):
    from product.object_parent import adapt_owners
    _,_,file=prepared(tmp_path);plan,_=read_plan(json.loads(file.read_text()))
    history={'objects':[{'id':'whole','status':'EXTRACTED','replaces':['old','second-half']}]}
    ir={'nodes':[{'id':'extracted-whole'}]}
    with pytest.raises(ValueError,match='split historical'):adapt_owners(plan,ir,history)
    obj=plan.objects[0].model_copy(update={'replaces':['old','second-half'],'required_owners':['old','second-half']})
    result,changes=adapt_owners(plan.model_copy(update={'objects':[obj]}),ir,history)
    assert result.objects[0].replaces==['extracted-whole'] and changes


@pytest.mark.parametrize('failed_reviews',[0,99])
def test_auto_service_failure_delivers_review_draft_not_success(tmp_path,monkeypatch,failed_reviews):
    from test_product import FakeProvider,page
    from product import pipeline
    source=tmp_path/'source.png';Image.new('RGB',(800,450),'white').save(source)
    provider=FakeProvider(page(source),failed_reviews=failed_reviews)
    monkeypatch.setattr(pipeline,'materialize',lambda plan,source,directory,*args:source)
    monkeypatch.setattr('product.object_stage.obtain_layers',lambda *args:tmp_path)
    def unavailable(*args,**kwargs):raise RuntimeError('object service unavailable')
    monkeypatch.setattr('product.object_stage.process',unavailable)
    def app_unavailable(*args,**kwargs):raise OSError('synthetic fixture does not launch Office')
    monkeypatch.setattr('product.object_finalize.subprocess.run',app_unavailable)
    def compiler(irs,out,plans):
        out.mkdir(parents=True);preview=out/'preview';preview.mkdir();Image.new('RGB',(800,450),'white').save(preview/'01.png')
        pptx=out/'deck.pptx';pptx.write_bytes(b'synthetic-compiler-fixture');return {'pptx':str(pptx),'preview':str(preview)}
    job=tmp_path/'job';(job/'inputs').mkdir(parents=True);(job/'inputs/01.png').write_bytes(source.read_bytes())
    write(job/'request.json',{'sources':['01.png'],'object_extraction':'auto'})
    result=pipeline.run(job,provider=provider,review_provider=provider,compiler=compiler,manifest={'fonts':[]})
    assert result['status']=='REVIEW_REQUIRED' and result['human_acceptance'] is False and (job/'delivery/result.pptx').exists()
    assert result['unresolved_pages']==[1]
    assert len(provider.calls)==2


def test_background_restoration_is_staged_not_assumed_verified(tmp_path):
    layers,base,file=prepared(tmp_path);ir=json.loads((base/'slideir.json').read_text())
    ir['background']={'color':'F0F0F0'};write(base/'slideir.json',ir)
    result,report=extract_closed(file,layers,base/'slideir.json',tmp_path/'out')
    row=report['objects'][0]
    assert row['status']=='STAGED' and row['background_assets'][0]['status']=='PENDING_BACKGROUND_REVIEW'
    bg=row['background_assets'][0];patch=np.array(Image.open(result.parent/bg['file']))
    assert tuple(patch[2,2,:3])==(255,255,255) and patch[2,2,3]==255
    assert row['candidates'][0]['pending_background_review'] and report['automatic_quality']=='REVIEW_REQUIRED'
    assert {n['id'] for n in json.loads(result.read_text())['nodes']}=={'extracted-icon',bg['id']}


def test_white_and_background_unknown_never_pass_by_default():
    from product.object_finalize import object_pass
    verdict={'pixels':'PASS','complete_object':'PASS','replacement':'PASS'}
    assert object_pass({},verdict)
    row={'background_assets':[{}],'source_alpha_hole_hypothesis':True}
    assert not object_pass(row,verdict)
    verdict['background_restoration']='PASS';assert not object_pass(row,verdict)
    verdict['white_content_assembly']='PASS';assert object_pass(row,verdict)


def test_native_background_candidate_keeps_non_owned_pixels_and_soft_foreground(tmp_path):
    layers,base,file=prepared(tmp_path);ir=json.loads((base/'slideir.json').read_text())
    ir['background']={'color':'F0F0F0'};write(base/'slideir.json',ir)
    result,report=extract_closed(file,layers,base/'slideir.json',tmp_path/'out',candidate_offsets={'icon':1})
    row=report['objects'][0];assert row['status']=='STAGED' and row['method']=='alpha-native-bg'
    bg=row['background_assets'][0];assert bg['backing_method']=='verified_native_fill'
    patch=np.array(Image.open(result.parent/bg['file']))
    original=np.array(Image.open(base/'crop.png').convert('RGBA'))
    fg=np.array(Image.open(layers/'layer-01.png'))[8:28,8:23]
    inside=fg[...,3]>0
    assert np.array_equal(patch[~inside],original[~inside])
    assert np.all(patch[inside,:3]==240)
    output=np.array(Image.open(result.parent/row['asset']))
    assert 90 in output[...,3] and bg['status']=='PENDING_BACKGROUND_REVIEW'


def test_whole_old_crop_scope_is_separate_unverified_hypothesis(tmp_path):
    from product.object_finalize import object_pass
    layers,base,file=prepared(tmp_path);ir=json.loads((base/'slideir.json').read_text())
    ir['background']={'color':'F0F0F0'};write(base/'slideir.json',ir)
    result,report=extract_closed(file,layers,base/'slideir.json',tmp_path/'out',candidate_offsets={'icon':2})
    row=report['objects'][0];assert row['method']=='alpha-native-owner-bg'
    bg=row['background_assets'][0];assert bg['native_owner_scope_hypothesis']
    assert not bg['non_owned_pixels_unchanged'] and bg['unverified_old_scope_pixels']>0
    patch=np.array(Image.open(result.parent/bg['file']));assert np.all(patch[...,:3]==240)
    verdict={k:'PASS' for k in ('pixels','complete_object','replacement','background_restoration')}
    assert not object_pass(row,verdict)
    verdict['native_owner_scope']='UNKNOWN';assert not object_pass(row,verdict)
    verdict['native_owner_scope']='PASS';assert object_pass(row,verdict)


def test_planning_budget_counts_substantive_repairs_but_preserves_failed_reservations(tmp_path):
    schedule=ObjectSchedule(tmp_path/'schedule.json')
    call,_=schedule.reserve('logical','planning_revision',{'protocol':'old-invalid'})
    schedule.finish(call,'FAILED')
    for i in range(2):
        call,fresh=schedule.reserve('logical','planning_revision',{'protocol':'corrected','revision':i})
        assert fresh;schedule.finish(call,'SUCCEEDED',{'substantive_change':True})
    assert schedule.reserve('logical','planning_revision',{'protocol':'corrected','revision':2})==(None,False)
    assert not schedule.reserve('logical','planning_revision',{'protocol':'old-invalid'})[1]


def test_identical_bound_go_review_is_reused_without_second_call(tmp_path):
    from product.object_finalize import review_objects
    layers,base,plan=prepared(tmp_path);ir,report=extract_closed(plan,layers,base/'slideir.json',tmp_path/'stage')
    report['state_directory']=str(tmp_path/'state');calls=[]
    def ask(context,images,contract):
        calls.append(context)
        return {'schema':'ObjectReview/1','source_sha256':context['source_sha256'],'scene_sha256':context['scene_sha256'],
            'objects':[{'id':r['id'],'candidate_sha256':r['candidate_sha256'],'pixels':'PASS','complete_object':'PASS','replacement':'PASS','reasons':[]} for r in context['objects']]}
    a=review_objects(layers/'source.png',layers/'source.png',ir,report,tmp_path/'review-a',ask,review_model='synthetic-review-fixture')
    b=review_objects(layers/'source.png',layers/'source.png',ir,report,tmp_path/'review-b',ask,review_model='synthetic-review-fixture')
    assert a==b and len(calls)==1


def test_reference_only_go_repair_preserves_raw_and_decisions(tmp_path):
    from product.object_finalize import review_objects
    layers,base,plan=prepared(tmp_path);ir,report=extract_closed(plan,layers,base/'slideir.json',tmp_path/'stage')
    report['state_directory']=str(tmp_path/'state');calls=[]
    def ask(context,images,contract):
        calls.append(context)
        if 'original_response' in context:
            assert images==[]
            fixed=copy.deepcopy(context['original_response'])
            for item in fixed['objects']:item['candidate_sha256']=context['png_bindings'][item['id']]
            return fixed
        return {'schema':'ObjectReview/1','source_sha256':context['source_sha256'],'scene_sha256':context['scene_sha256'],
            'objects':[{'id':'icon','candidate_sha256':report['objects'][0]['candidates'][0]['raw_sha256'],
                'pixels':'PASS','complete_object':'PASS','replacement':'PASS','reasons':['synthetic contract fixture']}]}
    values=review_objects(layers/'source.png',layers/'source.png',ir,report,tmp_path/'review',ask,review_model='synthetic-review-fixture')
    assert len(calls)==2 and values['icon']['candidate_sha256']==report['objects'][0]['asset_sha256']
    raw=json.loads((tmp_path/'review/review-000-raw.json').read_text())
    assert raw['objects'][0]['candidate_sha256']==report['objects'][0]['candidates'][0]['raw_sha256']
    assert (tmp_path/'review/binding-repair-000.json').exists()
