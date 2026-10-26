"""Synthetic v2 contracts. These fixtures do not establish business-slide fidelity."""
from copy import deepcopy
import json
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from image2ppt.storage import digest, read_json
from image2ppt.quality import inspect_pptx
from image2ppt_vision import auto_v2 as workflow
from image2ppt_vision.auto_provider import fingerprint, save_new
from image2ppt_vision.auto_workflow import PageSession, BudgetExhausted
from image2ppt_vision.auto_region_protocol import (POLICY, layout_from_draft, decision_record,
    expand_region, retained_region, assemble, validate_region, validate_correction, text_candidates)
from image2ppt_vision.auto_region_evidence import validate_visual, validate_probe, verify_deletion, check_transitions
from image2ppt_vision.auto_protocol import transform_model_boxes
from image2ppt_vision.auto_artifacts import file_hashes
from image2ppt_vision.supervised_elements import validate_element_plan


def fixture():
    texts=[{'id':'t1','text':'Heading','bbox':[10,10,150,22]},
           {'id':'t2','text':'Complex source UI','bbox':[10,70,180,22]},
           {'id':'t3','text':'Page 09','bbox':[10,170,100,22]}]
    inventory={'source_sha256':'a'*64,'image_size':[300,200],'texts':texts,'candidates':[]}
    decisions=[dict(t,action='native',font_size_px=16,font_family='Arial',font_latin='Arial',
                    font_confidence=.25,font_evidence='Synthetic fallback candidate',bold=False,
                    color='000000',align='left',z=10000+i) for i,t in enumerate(deepcopy(texts))]
    defaults={'text_decisions':decisions,'diagnostics':[{'id':t['id'],'applied':False} for t in texts]}
    draft={'schema':'RegionLayoutDraft/1','coordinate_space':'normalized_1000_xyxy',
           'source_sha256':inventory['source_sha256'],'image_size':inventory['image_size'],'background_color':'FFFFFF',
           'regions':[{'id':'r01','bbox':[0,0,1000,250],'classification':'step_cards','description':'Header title','subregions':[]},
                      {'id':'r02','bbox':[0,250,1000,800],'classification':'ui_mockup','description':'Complex UI module','subregions':[]},
                      {'id':'r03','bbox':[0,800,1000,1000],'classification':'flow','description':'Footer label','subregions':[]}]}
    return inventory,defaults,draft


def test_layout_diagnostics_reports_all_geometry_failures_without_rewriting():
    from image2ppt_vision.auto_region_protocol import layout_diagnostics
    inv,_,draft=fixture(); draft['regions'][0]['bbox']=[0,0,1000,100]
    draft['regions'][0]['subregions']=[{'id':'table','kind':'table','bbox':[0,0,1000,200]}]
    before=deepcopy(draft); report=layout_diagnostics(draft,inv)
    assert draft==before
    assert any(i.get('child_outside_parent')=='table' for i in report['violations'])
    assert any(i.get('ocr_id')=='t1' for i in report['violations'])
    assert report['violations'][0]['minimum_containing']['bbox']==[0,0,1000,200]
    fixed=fixture()[2];fixed['regions'][0]['subregions']=[{'id':'table','kind':'table','bbox':[0,0,1000,200]}]
    assert layout_from_draft(fixed,inv)['regions'][0]['subregions'][0]['kind']=='table'


def routing(kind='native'):
    native=['labels'] if kind!='retain_module' else []
    retained=['module_body'] if kind!='native' else []
    return {'decision':kind,'confidence':.8,'can_rebuild':{'feasible':kind!='retain_module','native_parts':native,'blockers':[]},
            'should_rebuild':{'editability_value':'high','fidelity_risk':'medium','cost':'medium'},
            'native_parts':native,'retained_parts':retained,'retain_reasons':['ui_mockup'] if retained else [],
            'evidence':['Synthetic source shows distinct text and a UI module.']}


def bound(layout,region,kind='native'):
    return decision_record(routing(kind),region=region,layout=layout,crop_sha256='b'*64,capabilities={},response_sha256='c'*64)


def native_draft(region,decision):
    return {'schema':'RegionElementsDraft/1','coordinate_space':'normalized_1000_xyxy',
            'region_id':region['id'],'decision_binding':decision['binding'],'objects':[],'text_overrides':[],
            'mapping':{'native':{'labels':region['text_ids']},'retained':{}},
            'text_audit':[{'id':tid,'status':'confirmed','reason':'Compared with synthetic source crop.','style_group':''} for tid in region['text_ids']],
            'limitations':[]}


def test_author_diagnostics_exposes_duplicate_text_kind_and_invalid_overrides():
    from image2ppt_vision.auto_region_protocol import element_diagnostics
    inv,defaults,draft=fixture(); layout=layout_from_draft(draft,inv); region=layout['regions'][0]
    decision=bound(layout,region); proposal=native_draft(region,decision)
    proposal['objects']=[{'id':'r01-copy','kind':'native_text','bbox':[0,0,100,100],'z':10000}]
    proposal['text_overrides']=[{'id':'t1','status':'retained'}]
    proposal['mapping']['native']['labels']=[]; before=deepcopy(proposal)
    report=element_diagnostics(proposal,defaults,inv,region,decision)
    assert proposal==before and len(report['violations'])>=4
    assert any(i.get('unsupported_override_fields')==['status'] for i in report['violations'])
    with pytest.raises(ValueError,match='text_overrides cannot contain'):
        expand_region(proposal,defaults,inv,region,decision,layout)


def test_nested_geometry_gets_actionable_feedback_instead_of_discarding_all_diagnostics():
    from image2ppt_vision.auto_typography import draft_feedback
    from image2ppt_vision.auto_region_protocol import element_diagnostics
    inv,defaults,draft=fixture();layout=layout_from_draft(draft,inv);region=layout['regions'][0]
    decision=bound(layout,region);proposal=native_draft(region,decision)
    proposal['objects']=[{'id':'r01-panel','kind':'native_shape','bbox':[0,0,1000,250],
                          'z':1,'geometry':{'kind':'rect'},'fill':'FFFFFF'}]
    report=element_diagnostics(proposal,defaults,inv,region,decision)
    assert any(i.get('unmapped_ids')==['r01-panel'] for i in report['violations'])
    assert any('allowed_geometries' in i for i in report['violations'])
    assert draft_feedback(proposal,inv)[0]['error']=='unsupported_geometry'
    with pytest.raises(ValueError,match='geometry must be a STRING'):
        expand_region(proposal,defaults,inv,region,decision,layout)


def test_prompt_only_improvement_reuses_valid_author_stage_but_rejects_changed_parent(tmp_path):
    from types import SimpleNamespace
    session=SimpleNamespace(root=tmp_path)
    call=tmp_path/'calls/c00/request.json';call.parent.mkdir(parents=True)
    save_new(call,{'request':{'body':{'messages':[{'role':'system','content':'original prompt'}]}}})
    path=tmp_path/'elements.json';inputs={'prompt':'original prompt','evidence':{'source':'same'},'model':'fixture'}
    workflow.step(path,inputs,lambda:{'fixture':'validated'})
    changed={**inputs,'prompt':'clearer new prompt'}
    assert workflow.step(path,changed,lambda:pytest.fail('must reuse'),session=session,
                         allow_prompt_history=True)=={'fixture':'validated'}
    assert list(tmp_path.glob('elements-prompt-reuse-*.json'))
    with pytest.raises(ValueError,match='saved stage evidence changed'):
        workflow.step(path,{**changed,'evidence':{'source':'changed'}},lambda:None,
                      session=session,allow_prompt_history=True)
    with pytest.raises(ValueError,match='saved stage evidence changed'):
        workflow.step(path,changed,lambda:None,session=session)


def test_initial_incompatible_scope_requires_new_agent_decision_before_reauthor(tmp_path,monkeypatch):
    from types import SimpleNamespace
    inv,defaults,draft=fixture();layout=layout_from_draft(draft,inv);region=layout['regions'][1]
    original=bound(layout,region,'native'); revised=bound(layout,region,'retain_module')
    session=SimpleNamespace(root=tmp_path,last_response_sha256='d'*64); phases=[]
    def decide(*args,**kwargs):
        phases.append(kwargs['phase'])
        if 'previous' in kwargs:
            assert kwargs['previous']==original and kwargs['review']['decision_binding']==original['binding']
            return revised
        return original
    def author(*args,**kwargs):
        phases.append(kwargs['phase'])
        if kwargs['phase'].startswith('initial/author/'):
            raise ValueError('model_output_validation_exhausted: geometry must be a STRING from allowed_geometries')
        assert args[6]==revised
        return {'synthetic':'valid new scope'}
    monkeypatch.setattr(workflow,'decide',decide);monkeypatch.setattr(workflow,'author',author)
    decision,elements=workflow.initial_region(session,tmp_path/'source.png',inv,defaults,layout,region,{},tmp_path/'initial/r02')
    assert decision==revised and elements=={'synthetic':'valid new scope'}
    assert phases==['initial/decision/r02','initial/author/r02','initial/redecision/r02','initial/reauthor/r02']
    confidence_only=deepcopy(original);confidence_only['decision']['confidence']=.1
    assert workflow.routing_scope(confidence_only)==workflow.routing_scope(original)


def test_initial_geometry_retry_extends_attempts_without_rewriting_the_proposal(tmp_path,monkeypatch):
    from types import SimpleNamespace
    inv,defaults,draft=fixture();layout=layout_from_draft(draft,inv);region=layout['regions'][0]
    decision=bound(layout,region);source=tmp_path/'source.png';Image.new('RGB',(300,200),'white').save(source)
    session=SimpleNamespace(root=tmp_path,model='fixture',last_response_sha256='e'*64);attempts=[]
    def checked(session,phase,prompt,evidence,images,validator,**kwargs):
        attempts.append(kwargs.get('attempts',3))
        if len(attempts)==1: raise ValueError('model_output_validation_exhausted: native text box leaves region: t1')
        return validator(native_draft(region,decision))
    monkeypatch.setattr(workflow,'checked',checked);monkeypatch.setattr(workflow,'font_status',lambda:{'families':['Arial']})
    result=workflow.author(session,source,inv,defaults,layout,region,decision,tmp_path/'author',phase='initial/author/r01')
    assert attempts==[3,6] and result['text_decisions'][0]['text']=='Heading'


def page(kind='retain_module'):
    inv,defaults,draft=fixture(); layout=layout_from_draft(draft,inv); ds={}; es={}
    for region in layout['regions']:
        rid=region['id']; ds[rid]=bound(layout,region,kind if rid=='r02' else 'native')
        es[rid]=(retained_region(defaults,inv,region,ds[rid],layout) if ds[rid]['decision']['decision']=='retain_module'
                 else expand_region(native_draft(region,ds[rid]),defaults,inv,region,ds[rid],layout))
    return inv,defaults,layout,ds,es


def test_region_native_and_authorized_large_retention_assemble_without_weakening_legacy():
    inv,_,layout,ds,es=page(); plan,scopes,retained=assemble(layout,inv,ds,es)
    assert retained==['t2'] and scopes['r02-retained-module']==[0,50,300,110]
    assert es['r02']['objects'][0]['source_crop_bbox']==es['r02']['objects'][0]['dest_bbox']
    with pytest.raises(ValueError,match='large raster'):
        validate_element_plan(plan,inv)  # legacy admission still forbids unscoped large crops
    validate_element_plan(plan,inv,raster_scopes=scopes)


def test_hybrid_retains_source_pixels_and_owns_text_exactly_once():
    inv,defaults,draft=fixture(); layout=layout_from_draft(draft,inv); region=layout['regions'][1]
    decision=bound(layout,region,'hybrid'); value=native_draft(region,decision)
    value['objects']=[{'id':'r02-background','kind':'native_shape','bbox':[0,250,1000,800],'z':0,'geometry':'rect','fill':'FFFFFF'},
                      {'id':'r02-icon','kind':'raster_asset','source_crop_bbox':[800,300,900,450],
                       'dest_bbox':[800,300,900,450],'z':1,'contains_text_ids':[]}]
    value['mapping']={'native':{'labels':['t2','r02-background']},'retained':{'module_body':['r02-icon']}}
    result=expand_region(value,defaults,inv,region,decision,layout)
    assert result['text_decisions'][0]['action']=='native'
    value['objects'][1]={'id':'r02-icon','kind':'native_shape','bbox':[800,300,900,450],'geometry':'ellipse','fill':'0000FF','z':1}
    with pytest.raises(ValueError,match='actual source pixels'):
        expand_region(value,defaults,inv,region,decision,layout)


def test_all_raster_tiles_and_whole_page_layout_rejected():
    inv,defaults,draft=fixture(); layout=layout_from_draft(draft,inv); ds={}; es={}
    for region in layout['regions']:
        ds[region['id']]=bound(layout,region,'retain_module')
        es[region['id']]=retained_region(defaults,inv,region,ds[region['id']],layout)
    with pytest.raises(ValueError,match='all-raster'):
        assemble(layout,inv,ds,es)
    draft['regions'][0]['bbox']=[0,0,1000,1000]
    with pytest.raises(ValueError,match='whole-page'):
        layout_from_draft(draft,inv)


@pytest.mark.parametrize('change',[
    lambda d:d['regions'][0].update(bbox=[0,0,1000,80]),
    lambda d:d['regions'][1].update(bbox=[0,100,1000,800]),
    lambda d:d.update(source_sha256='f'*64),
])
def test_layout_rejects_unowned_crossing_or_wrong_source(change):
    inv,_,draft=fixture(); change(draft)
    with pytest.raises(ValueError): layout_from_draft(draft,inv)


def test_typography_confidence_is_preserved_and_every_text_requires_audit():
    inv,defaults,layout,ds,es=page(); region=layout['regions'][0]
    candidate=text_candidates(defaults,inv,region)[0]
    assert candidate['fit_applied'] is False and candidate['font_confidence']==.25
    assert candidate['original_bbox']==[10*1000/300,50,160*1000/300,160]
    draft=native_draft(region,ds['r01']); draft['text_audit']=[]
    with pytest.raises(ValueError,match='typography audit'):
        expand_region(draft,defaults,inv,region,ds['r01'],layout)


def test_bound_routing_cannot_be_swapped_or_claim_human_approval():
    inv,_,layout,ds,es=page(); region=layout['regions'][0]
    for change in [lambda d:d.update(human_acceptance=True),lambda d:d.update(layout_binding='e'*64),
                   lambda d:d['decision'].update(confidence=.1)]:
        decision=deepcopy(ds['r01']); change(decision)
        with pytest.raises(ValueError): validate_region(es['r01'],inv,region,decision,layout)


def review_issue(kind='style',targets=None):
    return {'id':'style1','kind':kind,'severity':'blocking','object_ids':targets or ['t1'],
            'bbox':[0,0,300,50],'description':'Source title is a different size.'}


def test_no_effect_unreviewed_text_and_source_recrop_are_rejected():
    _,_,layout,_,es=page(); old=es['r01']; region=layout['regions'][0]; review={'issues':[review_issue()]}
    with pytest.raises(ValueError,match='no_effect'):
        validate_correction(old,deepcopy(old),review,region)
    metadata=deepcopy(old); metadata['text_audit'][0]['reason']='New prose without changed content.'
    with pytest.raises(ValueError,match='no_effect'):
        validate_correction(old,metadata,review,region)
    changed=deepcopy(old); changed['text_decisions'][0]['font_size_px']=18
    assert validate_correction(old,changed,review,region)==['t1']
    changed['text_decisions'][0]['text']='Invented heading'
    with pytest.raises(ValueError,match='content requires'):
        validate_correction(old,changed,review,region)
    changed=deepcopy(old); changed['text_decisions'][0]['font_size_px']=18
    with pytest.raises(ValueError,match='unreviewed'):
        validate_correction(old,changed,{'issues':[]},region)
    old=es['r02']; changed=deepcopy(old); changed['objects'][0]['source_crop_bbox'][0]=1
    with pytest.raises(ValueError,match='recrop requires'):
        validate_correction(old,changed,{'issues':[review_issue(targets=['r02-retained-module'])]},layout['regions'][1])


def visual(region,decision,binding='d'*64):
    return {'schema':'RegionVisualReviewDraft/1','coordinate_space':'normalized_1000_xyxy','region_id':region['id'],
            'artifact_binding':binding,'decision_binding':decision['binding'],'quality_passed':True,'issues':[],
            'redecision_required':False,'redecision_reason':'','summary':'Synthetic visual fixture.'}


def test_normal_visual_review_rejects_probe_editability_and_stale_findings():
    inv,_,layout,ds,es=page(); region=layout['regions'][0]
    raw=visual(region,ds['r01']); kwargs=dict(region=region,elements=es['r01'],binding='d'*64,decision_binding=ds['r01']['binding'],size=inv['image_size'])
    validate_visual(raw,**kwargs)
    for kind in ['probe','editable']:
        bad=deepcopy(raw); bad['quality_passed']=False
        bad['issues']=[{**review_issue(kind),'bbox':[0,0,1000,250]}]
        with pytest.raises(ValueError,match='cannot infer'): validate_visual(bad,**kwargs)
    raw['artifact_binding']='e'*64
    with pytest.raises(ValueError,match='stale'): validate_visual(raw,**kwargs)


def test_probe_ghost_targets_are_bound_and_verdict_consistent():
    raw={'schema':'RegionProbeReview/1','artifact_binding':'b','region_id':'r01','ghost_free':True,'findings':[]}
    validate_probe(raw,region_id='r01',binding='b',targets=['t1'])
    raw['findings']=[{'target_id':'t1','description':'Remaining duplicate glyphs.'}]
    with pytest.raises(ValueError,match='contradicts'): validate_probe(raw,region_id='r01',binding='b',targets=['t1'])


class CounterRemote:
    def __init__(self): self.calls=[]
    def request(self,request,timeout=420):
        self.calls.append(deepcopy(request)); return {'ok':True,'request_id':request['request_id'],'content':'{}'}


def test_32_initial_plus_16_reserved_budget_persists_across_resume(tmp_path):
    remote=CounterRemote(); session=PageSession(tmp_path,remote,'fixture',policy=POLICY)
    for i in range(32): session.call(f'initial/{i}','fixture',{'i':i},[])
    with pytest.raises(BudgetExhausted,match='review_budget_reserved'):
        session.call('initial/overflow','fixture',{},[])
    saved=(tmp_path/'budget.json').read_bytes(); again=PageSession(tmp_path,remote,'fixture',policy=POLICY)
    again.call('initial/0','fixture',{'i':0},[])  # cached receipt consumes no new call
    for i in range(16): again.call(f'review/{i}','fixture',{'i':i},[])
    with pytest.raises(BudgetExhausted,match='request_budget'): again.call('patch/overflow','fixture',{},[])
    assert len(remote.calls)==48 and (tmp_path/'budget.json').read_bytes()==saved
    with pytest.raises(ValueError,match='saved page budget'):
        PageSession(tmp_path,remote,'fixture',policy={**POLICY,'max_model_requests_per_page':49})


def write_pptx(path,records):
    ns='http://schemas.openxmlformats.org'; p=ns+'/presentationml/2006/main'; a=ns+'/drawingml/2006/main'; r=ns+'/officeDocument/2006/relationships'
    root=ET.Element('{'+p+'}sld'); tree=ET.SubElement(ET.SubElement(root,'{'+p+'}cSld'),'{'+p+'}spTree')
    for idx,item in enumerate(records,1):
        shape=ET.SubElement(tree,'{'+p+'}sp'); nv=ET.SubElement(shape,'{'+p+'}nvSpPr'); ET.SubElement(nv,'{'+p+'}cNvPr',id=str(idx),name=item['id'])
        props=ET.SubElement(shape,'{'+p+'}spPr'); xfrm=ET.SubElement(props,'{'+a+'}xfrm'); ET.SubElement(xfrm,'{'+a+'}off',x='0',y='0'); ET.SubElement(xfrm,'{'+a+'}ext',cx='1000',cy='1000')
        if 'text' in item:
            para=ET.SubElement(ET.SubElement(shape,'{'+p+'}txBody'),'{'+a+'}p'); run=ET.SubElement(para,'{'+a+'}r'); ET.SubElement(run,'{'+a+'}t').text=item['text']
    presentation=f'<p:presentation xmlns:p="{p}" xmlns:r="{r}"><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst><p:sldSz cx="2857500" cy="1905000"/></p:presentation>'
    rel=f'<Relationships xmlns="{ns}/package/2006/relationships"><Relationship Id="rId1" Target="slides/slide1.xml" Type="{r}/slide"/></Relationships>'
    with ZipFile(path,'w') as z:
        z.writestr('ppt/presentation.xml',presentation); z.writestr('ppt/_rels/presentation.xml.rels',rel); z.writestr('ppt/slides/slide1.xml',ET.tostring(root))


def test_actual_deletion_checks_other_objects_not_only_absent_target(tmp_path):
    from image2ppt.quality import edit_copy
    from image2ppt.contracts import ProbeCase
    a=tmp_path/'a.pptx'; b=tmp_path/'b.pptx'
    write_pptx(a,[{'id':'t1','text':'First'},{'id':'t2','text':'Second'}])
    edit_copy(a,b,ProbeCase(id='test',action='delete_text',targets=['t1']))
    assert verify_deletion(a,b,['t1'])['passed']
    write_pptx(b,[{'id':'t2','text':'Changed'}])
    assert not verify_deletion(a,b,['t1'])['passed']


def test_issue_transitions_do_not_treat_renamed_same_problem_as_fixed():
    old={'issues':[review_issue()]}; new={'issues':[{**review_issue(),'id':'new-name'}]}
    result=check_transitions(old,new,['t1'])
    assert len(result['unresolved'])==1 and result['resolved']==[]


class RoutedRemote:
    def __init__(self): self.calls=[]
    def request(self,request,timeout=420):
        self.calls.append(deepcopy(request)); body=request['body']; prompt=body['messages'][0]['content']; evidence=json.loads(body['messages'][1]['content'][0]['text'])
        if 'Inventory this slide' in prompt:
            _,_,value=fixture(); value['source_sha256']=evidence['source_sha256']
        elif 'conservative reconstruction planner' in prompt:
            value=routing('retain_module' if evidence['classification']=='ui_mockup' else 'native')
        elif 'Reconstruct ONLY' in prompt:
            value=native_draft(evidence['region'],{'binding':evidence['decision_binding']})
        elif 'Inspect only the deletion' in prompt:
            value={'schema':'RegionProbeReview/1','artifact_binding':evidence['artifact_binding'],'region_id':evidence['region_id'],'ghost_free':True,'findings':[]}
        else:
            value=visual(evidence['region'],{'binding':evidence['decision_binding']},evidence['artifact_binding'])
        return {'ok':True,'request_id':request['request_id'],'content':json.dumps(value)}


def fake_build(source,inventory,plan,directory,**kwargs):
    from image2ppt.quality import edit_copy
    from image2ppt.contracts import ProbeCase
    directory=Path(directory); packet=directory/'evidence'; packet.mkdir(parents=True)
    from shutil import copyfile
    copyfile(source,packet/'source.png'); copyfile(source,packet/'preview.png'); copyfile(source,packet/'delete-probe.png')
    copyfile(source,packet/'comparison.png'); copyfile(source,packet/'difference.png')
    records=plan['objects']+[t for t in plan['text_decisions'] if t['action']=='native']
    write_pptx(packet/'result.pptx',records); probes=[t['id'] for t in plan['text_decisions'] if t['action']=='native']
    edit_copy(packet/'result.pptx',packet/'delete-probe.pptx',ProbeCase(id='test',action='delete_text',targets=probes))
    save_new(packet/'objects.json',inspect_pptx(packet/'result.pptx'))
    result={'artifact_binding':fingerprint(plan),'plan_sha256':fingerprint(plan),'files':file_hashes(directory),
            'probe_targets':probes,'structural_issues':[],'counts':{'native_text':len(probes),'native_shapes':1,'raster_assets':1,'retained_text':1},
            'metrics':{'mean_absolute_pixel_difference':0}}
    save_new(directory/'result.json',result); return result


def test_v2_runs_routing_authoring_separate_reviews_and_stops_at_gate(tmp_path,monkeypatch):
    source=tmp_path/'source.png'; Image.new('RGB',(300,200),'white').save(source)
    inv,defaults,_=fixture()
    monkeypatch.setattr(workflow.legacy,'get_analysis',lambda *a:{})
    monkeypatch.setattr(workflow.legacy,'get_inventory',lambda session,source,analysis:{**inv,'source_sha256':digest(source)})
    monkeypatch.setattr(workflow.legacy,'get_text_defaults',lambda *a:defaults)
    monkeypatch.setattr(workflow,'capabilities',lambda *a:{})
    monkeypatch.setattr(workflow,'font_status',lambda:{'families':['Arial']})
    # Typography diagnostics are tested separately; the synthetic test image is blank.
    monkeypatch.setattr(workflow,'typography_checks',lambda *a:[])
    config={'ssh_host':'fixture','remote_worker':'/w','remote_config':'/c','models':['fixture']}
    proof=tmp_path/'preflight'; proof.mkdir(); save_new(proof/'evidence.json',{'synthetic':True})
    save_new(proof/'result.json',{'schema':'AutomaticPreflight/1','config_sha256':fingerprint(config),'passed':True,'selected_model':'fixture','files':file_hashes(proof)})
    job=workflow.create_job([source]*4,tmp_path/'jobs',config,gate_pages=[1]); remote=RoutedRemote()
    result=workflow.run_job(job,project_root=tmp_path,preflight_directory=proof,remote=remote,builder=fake_build)
    assert result['status']=='GATE_PASSED',result
    assert result['skipped_pages']==['P2','P3','P4'] and not (job/'P2/analysis.json').exists()
    assert result['pages']['P1']['regions'][1]['retained_text_ids']==['t2']
    assert len(remote.calls)==12
    for request in remote.calls:
        prompt=request['body']['messages'][0]['content']; images=[c for c in request['body']['messages'][1]['content'] if c['type']=='image_url']
        if 'Review source fidelity' in prompt or 'Inspect only the deletion' in prompt: assert len(images)==2
    count=len(remote.calls); workflow.run_job(job,project_root=tmp_path,remote=remote,builder=fake_build)
    assert len(remote.calls)==count
    manifest=read_json(job/'job.json'); manifest['schema']='AutomaticJob/1'; (job/'job.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='legacy job'): workflow.run_job(job,project_root=tmp_path,remote=remote)
