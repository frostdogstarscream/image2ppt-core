"""Synthetic contract/failure checks. Real Agent and Office acceptance is separate."""
import asyncio
import json
from pathlib import Path
import sys
from uuid import uuid4

import pytest
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.materialize import write, digest
from product.runtime.service import RuntimeService, TOOLS
from product.runtime.state import RuntimeConflict
from product.runtime.mcp_server import schemas
from product.runtime.office_bridge import pack, unpack, OfficeQueue


@pytest.fixture
def job(tmp_path, monkeypatch):
    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT', str(tmp_path/'leases'))
    root = tmp_path/'jobs'
    identifier = uuid4().hex
    directory = root/identifier
    (directory/'inputs').mkdir(parents=True)
    Image.new('RGB', (80, 60), '#d7e8fa').save(directory/'inputs/01.png')
    write(directory/'request.json', {'sources': ['01.png'], 'orchestrator': 'opencode',
        'visual_review_backend': 'opencode', 'goal': 'editable test'})
    return RuntimeService(root), identifier, directory


def call(service, identifier, name, **arguments):
    return service.call(name, identifier, idempotency_key=uuid4().hex, **arguments)


def test_independent_image_tools_and_stale_version(job):
    service, identifier, directory = job
    context = service.call('get_job_context', identifier)
    source = context['sources'][0]['id']
    first = call(service, identifier, 'crop_image', evidence_id=source, bbox=[0, 0, 30, 20])
    second = call(service, identifier, 'compare_images', first=first['image']['id'], second=first['image']['id'])
    assert second['changed_pixels'] == 0
    assert not service.state(identifier).read()['pages']
    with pytest.raises(RuntimeConflict, match='version conflict'):
        call(service, identifier, 'inspect_image', expected_version=0, evidence_id=source)


def test_idempotency_and_changed_evidence(job):
    service, identifier, directory = job
    source = service.call('get_job_context', identifier)['sources'][0]['id']
    arguments = dict(idempotency_key='crop-request-0001', evidence_id=source, bbox=[0, 0, 20, 20])
    first = service.call('crop_image', identifier, **arguments)
    assert service.call('crop_image', identifier, **arguments) == first
    with pytest.raises(RuntimeConflict, match='different arguments'):
        service.call('crop_image', identifier, **{**arguments, 'bbox': [1, 0, 20, 20]})
    (directory/'inputs/01.png').write_bytes(b'changed')
    with pytest.raises(RuntimeConflict):
        service.state(identifier).evidence(source)


def test_completed_receipt_recovers_after_lost_database_update(job):
    service, identifier, _ = job
    source = service.call('get_job_context', identifier)['sources'][0]['id']
    args = dict(idempotency_key='recover-request-01', evidence_id=source)
    first = service.call('inspect_image', identifier, **args)
    state = service.state(identifier)
    with state.db() as db:
        db.execute("UPDATE operations SET state='UNKNOWN',result=NULL")
    assert service.call('inspect_image', identifier, **args) == first
    assert state.read()['version'] == 1


def test_unknown_operation_is_not_reexecuted(job):
    service, identifier, _ = job
    with pytest.raises(ValueError):
        service.call('crop_image', identifier, idempotency_key='unknown-0001', evidence_id='missing', bbox=[0, 0, 1, 1])
    with pytest.raises(RuntimeConflict, match='UNKNOWN'):
        service.call('crop_image', identifier, idempotency_key='unknown-0001', evidence_id='missing', bbox=[0, 0, 1, 1])


def test_lookup_failure_is_a_response_not_an_asyncio_future_deadlock(job):
    service, identifier, _ = job
    def missing(current,directory):
        return next(iter([]))
    with pytest.raises(ValueError,match='object reference'):
        service.state(identifier).mutate('lookup',{},None,'lookup-failure-01',missing)


def test_cross_job_references_and_finalization_are_blocked(job):
    service, identifier, directory = job
    other = service.root/uuid4().hex
    (other/'inputs').mkdir(parents=True)
    write(other/'request.json', json.loads((directory/'request.json').read_text('utf8')))
    Image.new('RGB', (10, 10)).save(other/'inputs/01.png')
    source = service.call('get_job_context', identifier)['sources'][0]['id']
    with pytest.raises(ValueError, match='unknown evidence'):
        service.call('get_evidence', other.name, evidence_ids=[source])
    with pytest.raises(ValueError, match='compose'):
        call(service, identifier, 'finalize_job')


def test_sandbox_unavailable_preserves_task(job, monkeypatch):
    service, identifier, _ = job
    monkeypatch.delenv('IMAGE2PPT_SANDBOX_IMAGE', raising=False)
    result = call(service, identifier, 'run_image_script', script='print(1)', inputs=[])
    assert result['status'] == 'UNKNOWN'
    assert service.state(identifier).read()['status'] == 'RUNNING'


def test_catalog_has_composable_tools_and_no_go():
    import jsonschema
    catalog = schemas()
    assert {t.name for t in catalog} == set(TOOLS)
    for tool in catalog:
        jsonschema.Draft202012Validator.check_schema(tool.inputSchema)
    assert not any('go' in tool.name for tool in catalog)
    crop = next(t for t in catalog if t.name == 'crop_image')
    assert 'expected_version' not in crop.inputSchema['required']
    commit = next(t for t in catalog if t.name == 'commit_candidate')
    assert 'expected_version' in commit.inputSchema['required']
    layout=next(t for t in catalog if t.name=='fit_text_layout')
    assert set(layout.inputSchema['required'])=={'job_id','page','object_id','idempotency_key'}
    assert 'background_id' not in layout.inputSchema['properties']
    segment=next(t for t in catalog if t.name=='segment_object')
    assert set(segment.inputSchema['required'])=={'job_id','page','idempotency_key'}
    assert 'glyph_height' not in segment.inputSchema['properties']


def test_scene_creation_does_not_require_full_perception_or_accept_owner_injection(job, monkeypatch):
    service, identifier, _ = job
    from product import fonts
    monkeypatch.setattr(fonts, 'font_manifest', lambda texts: {'fonts': []})
    result = call(service, identifier, 'create_scene', page=1, evidence={})
    assert result['scene']['selected'] == {}
    assert not service.state(identifier).read().get('perception_complete')
    with pytest.raises(ValueError, match='ownership is derived'):
        call(service, identifier, 'update_scene', page=1, evidence={'ownership': {'x': 'RESOLVED'}})


def test_basic_mask_background_composition_with_source_reference(job,monkeypatch):
    """Synthetic source/scene only; this is not Agent or Office acceptance."""
    service,identifier,directory=job
    from PIL import ImageDraw,ImageFont
    from product import fonts
    manifest=fonts.font_manifest([])
    record=next(r for r in manifest['fonts'] if r['family']=='Arial' and not r['bold'])
    monkeypatch.setattr(fonts,'font_manifest',lambda texts:{'fonts':[record]})
    image=Image.new('RGB',(400,120),'white')
    font=ImageFont.truetype(record['file'],24,index=record['index'])
    draw=ImageDraw.Draw(image);draw.text((40,30),'Test',font=font,fill='black')
    box=draw.textbbox((40,30),'Test',font=font)
    bbox=[box[0],box[1],box[2]-box[0],box[3]-box[1]]
    image.save(directory/'inputs/01.png')
    ocr={'lines':[{'id':'line-1','text':'Test','score':1.0,'bbox':bbox}]}
    write(directory/'ocr.json',ocr)
    state=service.state(identifier);current=state.read();current['ocr']={'1':str(directory/'ocr.json')}
    write(state.root/'state.json',current)
    evidence={'text':{'styles':[{'id':'s','role':'body','ea':'Arial','latin':'Arial','size_px':24,'bold':False,'color':'000000'}],
                     'texts':[{'id':'text','text':'Test','bbox':bbox,'style':'s','z':2,'align':'left'}]}}
    scene=call(service,identifier,'create_scene',page=1,evidence=evidence)
    assert scene['scene']['ownership']['text']=='RESOLVED'
    assert scene['scene']['text']['members']['text']==['line-1']
    source=service.call('get_job_context',identifier)['sources'][0]['id']
    backing=call(service,identifier,'fit_background',page=1,object_id='text',evidence_id=source)
    masks=call(service,identifier,'complete_text_mask',page=1,object_id='text',background_id=backing['background']['id'])
    candidate=call(service,identifier,'create_candidate',page=1,object_id='text',
                   mask_id=masks['masks']['semantic_completion']['id'],background_id=backing['background']['id'])
    assert candidate['status']=='EXPLORATORY'
    assert candidate['ownership_authority'] is False
    assert service.state(identifier).read()['pages'][0]['ir'] != candidate['ir']
    bad_mask=directory/'unproven-mask.png'
    Image.new('L',image.size,255).save(bad_mask)
    mask_ref=state.register(bad_mask,'mask_candidate')
    draft=call(service,identifier,'create_candidate',page=1,object_id='text',
               mask_id=mask_ref['id'],background_id=backing['background']['id'])
    assert draft['status']=='EXPLORATORY'
    submitted=call(service,identifier,'commit_candidate',expected_version=draft['version'],candidate_id=draft['id'])
    assert submitted['status']=='SUBMISSION_BLOCKED'
    assert not service.graph(service.state(identifier).read()['pages'][0]).selected
    import product.runtime.compiler as compiler
    def compile_preview(irs,output,texts):
        assert irs == [Path(draft['ir'])]
        output.mkdir(parents=True)
        (output/'preview.pptx').write_bytes(b'synthetic compiler response')
        return {'pptx':str(output/'preview.pptx')}
    monkeypatch.setattr(compiler,'compile_local',compile_preview)
    preview=call(service,identifier,'compile_pptx',candidate_id=draft['id'])
    assert preview['status']=='COMPILED' and not preview['delivery_approval']
    assert not service.state(identifier).read().get('composition')


def test_visual_self_review_requires_provided_images_and_current_hashes(job):
    service, identifier, directory = job
    state = service.state(identifier)
    fake = directory/'package.pptx'
    fake.write_bytes(b'synthetic package - no Office claim')
    source = state.register(directory/'inputs/01.png', 'source')
    preview = state.register(directory/'inputs/01.png', 'office_preview')
    current = state.read()
    current['composition'] = {'files': {str(fake): digest(fake)}, 'office': {'status': 'PASS'},
                              'required_evidence': [source, preview]}
    write(state.root/'state.json', current)
    review = {'passed': True, 'findings': [], 'minor_deviations': [],
              'evidence_ids': [source['id'], preview['id']],
              'checked': ['content','completeness','layout','residual','background','editing'], 'reason': 'synthetic'}
    with pytest.raises(ValueError, match='provided'):
        call(service, identifier, 'record_visual_review', review=review)
    for reference in (source, preview):
        state.evidence(reference['id'], delivered=True)
    assert call(service, identifier, 'record_visual_review', review=review)['status'] == 'PASS'
    fake.write_bytes(b'changed package')
    with pytest.raises(RuntimeConflict, match='changed'):
        call(service, identifier, 'record_visual_review', review=review)


def test_packet_hashes_paths_and_office_queue(tmp_path, monkeypatch):
    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT', str(tmp_path/'leases'))
    source = tmp_path/'source'
    source.mkdir()
    (source/'file.txt').write_text('evidence')
    pack(source, tmp_path/'packet.zip')
    unpack((tmp_path/'packet.zip').read_bytes(), tmp_path/'restored')
    assert digest(source/'file.txt') == digest(tmp_path/'restored/file.txt')
    import io, zipfile
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as archive:
        archive.writestr('packet-manifest.json', json.dumps({'../escape': 'x'}))
        archive.writestr('../escape', 'x')
    with pytest.raises(ValueError, match='unsafe'):
        unpack(data.getvalue(), tmp_path/'unsafe')
    root = tmp_path/'jobs'
    directory = root/uuid4().hex/'runtime/operations/op/check'
    directory.mkdir(parents=True)
    write(directory/'worker-state.json', {'status': 'QUEUED'})
    queue = OfficeQueue(root)
    ticket = queue.claim()
    assert queue.claim() is None
    assert queue.directory(**ticket) == directory
    with pytest.raises(ValueError):
        queue.directory(ticket['ticket'], 'wrong')


def test_parallel_job_claims_are_unique(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from product.app import Queue
    queue=Queue(tmp_path/'jobs')
    with queue.connect() as db:
        db.executemany('INSERT INTO jobs VALUES(?,?,?)',[('a'*32,'QUEUED',1),('b'*32,'QUEUED',2)])
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims=list(pool.map(lambda _:queue.take(),range(2)))
    assert set(claims)=={'a'*32,'b'*32}
    assert queue.take() is None


def test_opencode_api_admits_without_go_and_cancel(tmp_path, monkeypatch):
    import product.app as app_module
    import product.pipeline as pipeline
    monkeypatch.setattr(app_module, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'settings', lambda: object())
    monkeypatch.delenv('OPENCODE_GO_API_KEY', raising=False)
    client = TestClient(app_module.create_app(tmp_path/'jobs', token='t'*32, run_worker=False))
    import io
    picture = io.BytesIO()
    Image.new('RGB', (80, 60)).save(picture, 'PNG')
    auth = {'Authorization': 'Bearer '+'t'*32}
    response = client.post('/api/v1/jobs', files=[('files', ('x.png', picture.getvalue(), 'image/png'))], headers=auth)
    assert response.status_code == 202
    identifier = response.json()['id']
    request = json.loads((tmp_path/'jobs'/identifier/'request.json').read_text('utf8'))
    assert request['visual_review_backend'] == 'opencode'
    assert client.post(f'/api/v1/jobs/{identifier}/cancel', headers=auth).status_code == 200
    assert client.get(f'/api/v1/jobs/{identifier}', headers=auth).json()['status'] == 'CANCELLED'


def test_runner_preserves_completed_result_when_session_becomes_idle(job,monkeypatch):
    import httpx
    from product.runtime import runner
    service,identifier,directory=job
    def workspace(job):
        target=directory/'agent';target.mkdir()
        write(target/'opencode.json',{'providers':{'fixture-provider':{}}})
        return target
    monkeypatch.setattr(runner,'prepare_workspace',workspace)
    class Client:
        prompts=0
        def verify(self):return {'version':runner.PINNED_VERSION}
        def request(self,method,route,**options):
            if method=='GET' and route.endswith('/message'):return {'data':[],'cursor':{}}
            if method=='GET' and route=='/api/mcp':return {'data':[{'name':'image2ppt','status':{'status':'connected'}}]}
            if method=='GET':
                response=httpx.Response(404,request=httpx.Request(method,'http://test'+route))
                raise httpx.HTTPStatusError('missing',request=response.request,response=response)
            if route.endswith('/prompt'):
                self.prompts+=1
                return {}
            if route.endswith('/wait'):
                current=service.state(identifier).read();current['status']='SUCCEEDED'
                write(service.state(identifier).root/'state.json',current)
                write(directory/'result.json',{'status':'SUCCEEDED','fixture':'no real delivery'})
            return {}
    client=Client()
    result=runner.run(directory,client)
    assert result['status']=='SUCCEEDED'
    assert json.loads((directory/'result.json').read_text('utf8'))==result
    assert client.prompts==1


def test_exact_ocr_binding_preserves_disagreement_and_ambiguous_duplicates():
    from product.runtime.primitives import bind_exact_ocr_members
    from product.parallel_contracts import TextEvidence
    value=TextEvidence.model_validate({'schema':'TextEvidence/1','source_sha256':'a'*64,
        'image_size':[100,100],'method_version':'synthetic','styles':[
            {'id':'s','role':'body','ea':'Arial','latin':'Arial','size_px':12,'bold':False,'color':'000000'}],
        'texts':[{'id':'t','text':'Exact','bbox':[10,10,40,15],'style':'s','z':1,'align':'left'}]})
    ocr={'lines':[{'id':'l1','text':'Exact','score':1.,'bbox':[10,10,40,15]}]}
    assert bind_exact_ocr_members(value,ocr).members=={'t':['l1']}
    assert not value.members
    ocr['lines'].append({'id':'l2','text':'Exact','score':1.,'bbox':[10,10,40,15]})
    assert not bind_exact_ocr_members(value,ocr).members
    ocr['lines']=[{'id':'l1','text':'Different','score':1.,'bbox':[10,10,40,15]}]
    assert not bind_exact_ocr_members(value,ocr).members


def test_text_geometry_does_not_authorize_a_large_or_truncated_rectangle():
    from product.runtime.primitives import text_geometry_conflicts
    from product.parallel_contracts import TextEvidence
    value=TextEvidence.model_validate({'schema':'TextEvidence/1','source_sha256':'a'*64,
        'image_size':[200,100],'method_version':'synthetic','styles':[
            {'id':'s','role':'body','ea':'Arial','latin':'Arial','size_px':12,'bold':False,'color':'000000'}],
        'members':{'t':['l1']},'texts':[{'id':'t','text':'Exact','bbox':[10,10,40,15],'style':'s','z':1,'align':'left'}]})
    ocr={'lines':[{'id':'l1','text':'Exact','score':1.,'bbox':[10,10,40,15]}]}
    assert not text_geometry_conflicts(value,ocr)
    value.texts[0].bbox=[0,0,200,100]
    assert text_geometry_conflicts(value,ocr)=={'t'}
    value.texts[0].bbox=[10,10,15,15]
    assert text_geometry_conflicts(value,ocr)=={'t'}


def test_generated_service_profile_roundtrip(tmp_path):
    import os
    if os.name!='nt':
        pytest.skip('Windows DPAPI profile')
    from product.runtime.client import save_profile,load_profile
    profile={'url':'http://127.0.0.1:8772','business_token':'generated-test-token'}
    target=tmp_path/'profile.dpapi'
    save_profile(target,profile)
    assert load_profile(target)==profile
    assert profile['business_token'].encode() not in target.read_bytes()


def test_office_cache_reuses_exact_inputs_and_invalidates_font_configuration(job,monkeypatch):
    from product import fonts
    from product.runtime import office
    service,identifier,directory=job
    state=service.state(identifier)
    monkeypatch.setenv('IMAGE2PPT_OFFICE_MODE','local')
    faces={'fonts':[{'family':'Synthetic','sha256':'a'*64}]}
    monkeypatch.setattr(fonts,'font_manifest',lambda texts:faces)
    invocations=[]
    def check(pptx,irs,output):
        invocations.append(str(output))
        return {'status':'PASS','files':{}}
    monkeypatch.setattr(office,'check_local',check)
    pptx=directory/'test.pptx';pptx.write_bytes(b'synthetic package')
    ir=directory/'slideir.json';ir.write_text('{}')
    first=service.office_check(state,pptx,[ir],directory/'first')
    second=service.office_check(state,pptx,[ir],directory/'second')
    assert first['status']==second['status']=='PASS'
    assert second['cache_hit'] and len(invocations)==1
    faces['fonts'][0]['sha256']='b'*64
    service.office_check(state,pptx,[ir],directory/'third')
    assert len(invocations)==2


def test_draft_scene_additions_preserve_accepted_definitions(job,monkeypatch):
    from product import fonts
    from product.runtime.primitives import merge_scene_evidence
    service,identifier,_=job
    monkeypatch.setattr(fonts,'font_manifest',lambda texts:{'fonts':[]})
    scene=call(service,identifier,'create_scene',page=1,evidence={})
    from product.parallel_contracts import SceneGraph
    graph=SceneGraph.model_validate(scene['scene'])
    from product.contracts import Style,Text
    graph.text.styles=[Style(id='style',role='body',ea='Arial',latin='Arial',size_px=12.,bold=False,color='000000')]
    graph.text.texts=[Text(id='accepted',text='Exact',bbox=[10.,10.,20.,15.],style='style',z=1,align='left')]
    graph.selected={'accepted':{'kind':'text'}}
    new=graph.text.texts[0].model_copy(update={'id':'draft','bbox':[40.,10.,20.,15.]}).model_dump()
    merged=merge_scene_evidence(graph,{'text':{'texts':[new]}})
    assert [t['id'] for t in merged['text']['texts']]==['accepted','draft']
    assert merged['text']['texts'][0]==graph.text.texts[0].model_dump()
    with pytest.raises(ValueError,match='rollback'):
        merge_scene_evidence(graph,{'text':{'styles':[{**graph.text.styles[0].model_dump(),'bold':True}]}})


def test_windows_runtime_child_stops_when_ownership_handle_closes():
    import os,subprocess
    if os.name!='nt':pytest.skip('Windows Job Object')
    from product.runtime.windows_host import contain_child
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        ownership=contain_child(process)
        ownership.Close()
        process.wait(timeout=5)
        assert process.poll() is not None
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=5)


def test_transcript_pagination_preserves_full_decision_trace():
    from product.runtime.runner import read_transcript
    class Client:
        def request(self,method,route,**options):
            if options['params'].get('cursor')=='next':
                return {'data':[{'id':'b'}],'cursor':{}}
            return {'data':[{'id':'a'}],'cursor':{'next':'next'}}
    result=read_transcript(Client(),'synthetic-session')
    assert result['complete'] and result['pages_read']==2
    assert [m['id'] for m in result['data']]==['a','b']


def test_go_agent_configuration_and_tool_planner_use_the_selected_provider(job,monkeypatch):
    from product.runtime.runner import prepare_workspace
    from product.runtime.provider import planning_actor,GO_BASE_URL
    _,_,directory=job
    for name in ('PLANNER_API_KEY','PLANNER_BASE_URL','PLANNER_MODEL'):
        monkeypatch.setenv(name,'initial')
    monkeypatch.setenv('IMAGE2PPT_AGENT_PROVIDER','opencode_go')
    monkeypatch.setenv('IMAGE2PPT_AGENT_API_KEY','synthetic-agent-secret')
    monkeypatch.setenv('IMAGE2PPT_AGENT_MODEL','qwen3.8-flash')
    workspace=prepare_workspace(directory)
    encoded=(workspace/'opencode.json').read_text('utf8')
    configuration=json.loads(encoded)
    provider=configuration['providers']['image2ppt-go']
    assert provider['settings']['baseURL']==GO_BASE_URL
    assert provider['models']['vision']['modelID']=='qwen3.8-flash'
    assert provider['models']['vision']['capabilities']['tools'] is True
    assert 'synthetic-agent-secret' not in encoded
    actor=planning_actor(directory)
    assert actor.model=='qwen3.8-flash' and actor.base_url==GO_BASE_URL
    assert actor.session==directory.name


def test_agent_scene_reuses_installed_windows_font_aliases(job,monkeypatch):
    from product import fonts
    service,identifier,_=job
    monkeypatch.setattr(fonts,'font_manifest',lambda texts:{'fonts':[
        {'family':'微软雅黑','bold':False,'weight':400,'cjk':True}]})
    result=call(service,identifier,'create_scene',page=1,evidence={'text':{
        'styles':[{'id':'s','role':'body','ea':'Microsoft YaHei','latin':'Microsoft YaHei',
                   'size_px':12.,'bold':False,'color':'000000'}],
        'texts':[{'id':'t','text':'中文','bbox':[10.,10.,20.,15.],'style':'s','z':1,'align':'left'}]}})
    style=result['scene']['text']['styles'][0]
    assert style['ea']==style['latin']=='微软雅黑'


def test_remaining_budget_updates_on_idempotent_readback(job,monkeypatch):
    import product.runtime.state as state_module
    service,identifier,_=job
    state=service.state(identifier)
    write(state.root/'session.json',{'deadline':200})
    monkeypatch.setattr(state_module.time,'time',lambda:100)
    context=service.call('get_job_context',identifier)
    assert context['budget']['remaining_seconds']==100
    arguments={'evidence_id':context['sources'][0]['id'],'idempotency_key':'inspect-budget-01'}
    first=service.call('inspect_image',identifier,**arguments)
    monkeypatch.setattr(state_module.time,'time',lambda:110)
    second=service.call('inspect_image',identifier,**arguments)
    assert first['remaining_seconds']==100 and second['remaining_seconds']==90
    assert first['version']==second['version']==1


def test_cancelled_job_does_not_dispatch_queued_office_work(tmp_path,monkeypatch):
    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT',str(tmp_path/'leases'))
    root=tmp_path/'jobs'
    for identifier in ('a'*32,'b'*32):
        write(root/identifier/'runtime/o/check/worker-state.json',{'status':'QUEUED'})
    write(root/('a'*32)/'cancel.requested',{'requested':True})
    ticket=OfficeQueue(root).claim()
    assert ticket['ticket'].startswith('b'*32+'/')
    first=json.loads((root/('a'*32)/'runtime/o/check/worker-state.json').read_text('utf8'))
    assert first['status']=='CANCELLED'


def test_mcp_readiness_is_checked_before_agent_execution(job,monkeypatch):
    from product.runtime.runner import wait_for_mcp
    import product.runtime.runner as runner
    service,identifier,directory=job
    monkeypatch.setattr(runner.time,'sleep',lambda seconds:None)
    class Client:
        calls=0
        def request(self,method,route,**options):
            self.calls+=1
            return {'data':[] if self.calls==1 else [{'name':'image2ppt','status':{'status':'connected'}}]}
    client=Client()
    assert wait_for_mcp(client,directory,service.state(identifier))
    assert client.calls==2


def test_office_overflow_measurements_remain_unverified_repair_proposals():
    from types import SimpleNamespace
    from product.runtime.office import frame_feedback
    layout={'overflow':True,'unexpected_wrap':False,'frame':[139.5,29.25,212.25,48.],
            'text_bounds':[139.5,29.25,227.625,42.3]}
    result={'status':'REVIEW_REQUIRED','replacements':{'records':[
        {'id':'title','content_matches':True,'siblings_unchanged':True,'frame_unchanged':True,
         'outside_pixels':23,'layout':layout}]}}
    graph=SimpleNamespace(text=SimpleNamespace(texts=[SimpleNamespace(id='title',bbox=[186,39,283,64])]))
    feedback=frame_feedback(result,graph,'title')
    assert feedback['status']=='UNVERIFIED_LAYOUT_PROPOSAL'
    assert feedback['measured_frame_minimum_px']==[305,58]
    assert feedback['previous_edit_outside_pixels']==[23]
    assert result['status']=='REVIEW_REQUIRED'


def test_mask_measurements_do_not_grant_or_require_ownership(job):
    service, identifier, directory = job
    state = service.state(identifier)
    source = service.call('get_job_context', identifier)['sources'][0]['id']
    for name, color in [('wanted',255),('protected',255),('reference',0)]:
        Image.new('L',(80,60),color).save(directory/(name+'.png'))
    refs={name:state.register(directory/(name+'.png'),'mask')['id'] for name in ('wanted','protected','reference')}
    result=call(service,identifier,'validate_mask',evidence_id=source,mask_id=refs['wanted'],
                protected_id=refs['protected'],reference_mask_id=refs['reference'])
    assert result['status']=='MEASURED'
    assert result['measurements']['protected_overlap_pixels']==4800
    assert result['measurements']['outside_reference_pixels']==4800
    assert not result['ownership_authority'] and not state.read()['pages']


def test_unmix_and_rgba_measurements_work_without_staged_ir(job):
    service, identifier, directory = job
    state=service.state(identifier)
    source=service.call('get_job_context',identifier)['sources'][0]['id']
    Image.new('RGB',(80,60),'white').save(directory/'backing.png')
    Image.new('L',(80,60),255).save(directory/'mask.png')
    backing=state.register(directory/'backing.png','background')['id']
    mask=state.register(directory/'mask.png','mask')['id']
    result=call(service,identifier,'unmix_foreground',evidence_id=source,background_id=backing,mask_id=mask)
    assert result['status']=='COMPUTED'
    measured=call(service,identifier,'validate_rgba',evidence_id=result['image']['id'],
                  source_id=source,background_id=backing,protected_id=mask)
    assert measured['status']=='MEASURED'
    assert measured['measurements']['protected_overlap_pixels']>0
    assert 'recomposition_mae' in measured['measurements']
    assert not state.read()['pages']


def test_agent_can_record_failed_candidate_observation_before_office(job):
    service,identifier,_=job
    source=service.call('get_job_context',identifier)['sources'][0]['id']
    result=call(service,identifier,'record_observation',evidence_ids=[source],
                findings=[{'issue':'background uncertain'}],next_action='compare another background')
    assert result['status']=='RECORDED'
    assert not service.state(identifier).read().get('visual_review')
