from copy import deepcopy
from pathlib import Path
import json

import pytest
from PIL import Image

from image2ppt.storage import digest, read_json, write_json
from image2ppt_vision import candidate_workflow as workflow


class FakeCompiler:
    def __init__(self, fail=False):
        self.fail=fail
        self.calls=0

    def build(self, ir_file, output):
        self.calls+=1
        if self.fail:raise RuntimeError('synthetic compiler failure')
        output.mkdir(parents=True)
        ir=read_json(ir_file)
        source=Path(ir_file).parent/ir['source_image']
        # This is deliberately only a workflow fixture, never a real PPTX claim.
        (output/'result.pptx').write_bytes(b'synthetic fixture')
        with Image.open(source) as image:image.save(output/'preview.png')
        write_json(output/'build.json',{'fixture':True})
        write_json(output/'objects.json',{'fixture':True,'objects':ir['nodes']})
        return {k:str(output/v) for k,v in {'pptx':'result.pptx','preview':'preview.png',
                                          'build':'build.json','objects':'objects.json'}.items()}


def author(file):
    write_json(file.parent/'authoring.json',{'format_version':'1','tool':'fixture','model_id':'fixture',
               'created_at':'2026-01-01T00:00:00Z','ir_sha256':digest(file)})


def setup(tmp_path):
    baseline=tmp_path/'baseline';baseline.mkdir()
    Image.new('RGB',(200,100),'white').save(baseline/'source.png')
    ir={'schema_version':'0.1','source_image':'source.png','source_sha256':digest(baseline/'source.png'),
        'canvas':{'source_px':[200,100],'target_px':[200,100]},'background':{'color':'FFFFFF'},
        'nodes':[{'id':'title','type':'text','bbox':[10,10,80,20],'z':0,'text':'title','evidence_ids':['e1']},
                 {'id':'frozen','type':'shape','bbox':[150,60,20,20],'z':1,'fill':'FF0000','evidence_ids':['e2']}],
        'assets':[],'evidence':[{'id':'e1','bbox':[10,10,80,20]},{'id':'e2','bbox':[150,60,20,20]}],'groups':[]}
    write_json(baseline/'slideir.json',ir)
    plans=tmp_path/'plans.json'
    write_json(plans,{'regions':[{'region_id':'P5-test','page':'P5','bbox':[0,0,100,50]}]})
    human=tmp_path/'human.json'
    write_json(human,{'decisions':{'P5-test':{'decision':'native','source':'fixture','note':'synthetic',
        'annotation_level':'explicit','retain_scope':'none','fidelity_target':'pixel',
        'required_edit_targets':['section_title'],'allowed_native_parts':['section_title'],'must_retain_parts':[]}}})
    candidate=Path(workflow.prepare(plans,human,baseline/'slideir.json','P5-test',tmp_path/'out')['candidate'])
    proposed=tmp_path/'proposed';proposed.mkdir()
    (proposed/'source.png').write_bytes((baseline/'source.png').read_bytes())
    ir['nodes'][0]['text']='fixed title'
    write_json(proposed/'slideir.json',ir);author(proposed/'slideir.json')
    mapping=tmp_path/'mapping.json';write_json(mapping,{'native':{'section_title':['title']},'retained':{}})
    return candidate,proposed/'slideir.json',mapping,baseline,plans


def record(tmp_path,result,issues=False):
    data={'schema':'CandidateReview/1','artifact_binding':result['artifact_binding'],
          'reviewer':'fixture','issues':[],'unresolved':[],'summary':'synthetic review','human_acceptance':False}
    if issues:data['issues']=[{'object_ids':['title'],'kind':'style',
        'source_evidence':'evidence/candidate-source.png','rendered_evidence':'evidence/candidate-rendered.png',
        'suggestion':'Match the source title weight.'}]
    file=tmp_path/('review-'+Path(result['revision']).name+'.json');write_json(file,data)
    return file


def test_build_review_immutable_revisions_and_budget(tmp_path):
    candidate,ir,mapping,baseline,_=setup(tmp_path)
    original=digest(baseline/'slideir.json')
    compiler=FakeCompiler()
    first=workflow.build(candidate,ir,mapping,compiler=compiler)
    assert first['state']=='TRIAL_BUILT' and compiler.calls==2
    assert first['metrics']['candidate']['region_mean_pixel_difference']==0
    reviewed=workflow.review(candidate,'r000',record(tmp_path,first,True))
    assert reviewed['state']=='REVIEW_REQUIRED' and not reviewed['human_acceptance']
    before=digest(candidate/'revisions/r000/result.json')
    second=workflow.build(candidate,ir,mapping,compiler=compiler)
    with pytest.raises(workflow.WorkflowError,match='stale review'):
        workflow.review(candidate,'r001',record(tmp_path,first))
    workflow.review(candidate,'r001',record(tmp_path,second,True))
    third=workflow.build(candidate,ir,mapping,compiler=compiler)
    review=workflow.review(candidate,'r002',record(tmp_path,third,True))
    assert review['correction_budget_exhausted'] and review['state']=='REVIEW_REQUIRED'
    with pytest.raises(workflow.WorkflowError,match='exhausted'):
        workflow.build(candidate,ir,mapping,compiler=compiler)
    assert digest(candidate/'revisions/r000/result.json')==before
    assert digest(baseline/'slideir.json')==original
    assert len(read_json(candidate/'records.json')['candidates'])==3


@pytest.mark.parametrize('failure',['frozen','scope','missing_target','unknown_id','authoring','input','compiler'])
def test_stops_before_invalid_build_and_preserves_failure(tmp_path,failure):
    candidate,ir,mapping,baseline,plans=setup(tmp_path)
    data=read_json(ir);mapped=read_json(mapping)
    if failure=='frozen':data['nodes'][1]['fill']='00FF00'
    if failure=='scope':mapped['native']={'kpi':['title']}
    if failure=='missing_target':mapped['native']={}
    if failure=='unknown_id':mapped['native']={'section_title':['unknown']}
    if failure=='input':write_json(plans,{'regions':[]})
    write_json(ir,data);write_json(mapping,mapped)
    if failure!='authoring':author(ir)
    else:(ir.parent/'authoring.json').unlink()
    compiler=FakeCompiler(fail=failure=='compiler')
    result=workflow.build(candidate,ir,mapping,compiler=compiler)
    assert result['state']=='FAILED'
    assert compiler.calls==(1 if failure=='compiler' else 0)
    assert (Path(result['revision'])/'result.json').exists()


def test_review_rejects_tamper_and_acceptance_claim(tmp_path):
    candidate,ir,mapping,_,_=setup(tmp_path)
    result=workflow.build(candidate,ir,mapping,compiler=FakeCompiler())
    file=record(tmp_path,result)
    data=read_json(file);data['human_acceptance']=True;write_json(file,data)
    with pytest.raises(workflow.WorkflowError,match='cannot grant'):
        workflow.review(candidate,'r000',file)
    data['human_acceptance']=False;write_json(file,data)
    (Path(result['revision'])/'evidence/candidate.pptx').write_bytes(b'changed')
    with pytest.raises(workflow.WorkflowError,match='hash changed'):
        workflow.review(candidate,'r000',file)


def test_clean_review_is_technical_verification_only(tmp_path):
    candidate,ir,mapping,_,_=setup(tmp_path)
    result=workflow.build(candidate,ir,mapping,compiler=FakeCompiler())
    review=workflow.review(candidate,'r000',record(tmp_path,result))
    assert review['state']=='VERIFIED' and review['human_acceptance'] is False
    with pytest.raises(FileExistsError):
        workflow.review(candidate,'r000',record(tmp_path,result))


def test_failed_attempt_does_not_spend_visual_correction_budget(tmp_path):
    candidate,ir,mapping,_,_=setup(tmp_path)
    failed=workflow.build(candidate,ir,mapping,compiler=FakeCompiler(fail=True))
    assert failed['state']=='FAILED'
    built=workflow.build(candidate,ir,mapping,compiler=FakeCompiler())
    assert built['correction_index']==0
    workflow.review(candidate,'r001',record(tmp_path,built,True))
    original=read_json(ir);original['nodes'][0]['text']='review-directed correction'
    write_json(ir,original);author(ir)
    corrected=workflow.build(candidate,ir,mapping,compiler=FakeCompiler())
    assert corrected['correction_index']==1 and corrected['state']=='TRIAL_BUILT'
    saved=read_json(Path(corrected['revision'])/'input/slideir.json')
    assert saved['nodes'][0]['text']=='review-directed correction'


def test_scope_manifest_cannot_silently_override_prepared_authority(tmp_path):
    candidate,ir,mapping,_,_=setup(tmp_path)
    manifest=read_json(candidate/'candidate.json')
    manifest['human']['allowed_native_parts'].append('kpi')
    write_json(candidate/'candidate.json',manifest)
    result=workflow.build(candidate,ir,mapping,compiler=FakeCompiler())
    assert result['state']=='FAILED' and 'authoritative snapshot' in result['error']['message']


def test_correction_cannot_edit_unmentioned_region_object(tmp_path):
    candidate,ir,mapping,_,_=setup(tmp_path)
    first=workflow.build(candidate,ir,mapping,compiler=FakeCompiler())
    workflow.review(candidate,'r000',record(tmp_path,first))
    proposed=read_json(ir);proposed['nodes'][0]['text']='unrequested change'
    write_json(ir,proposed);author(ir)
    result=workflow.build(candidate,ir,mapping,compiler=FakeCompiler())
    assert result['state']=='FAILED' and 'frozen node changed' in result['error']['message']
