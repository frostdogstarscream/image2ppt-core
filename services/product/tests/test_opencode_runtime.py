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
    with pytest.raises(ValueError,match='binary grayscale Mask'):
        call(service,identifier,'complete_text_mask',page=1,object_id='text',background_id=backing['background']['id'],original_mask_id=backing['background']['id'])
    masks=call(service,identifier,'complete_text_mask',page=1,object_id='text',background_id=backing['background']['id'])
    protection_file=Path(state.read()['pages'][0]['ir']).parent/'runtime-ocr-protection.json'
    retained=json.loads(protection_file.read_text('utf8'))
    write(protection_file,{**retained,'lines':retained['lines']+[{'id':'neighbor','bbox':bbox}]})
    protected_masks=call(service,identifier,'complete_text_mask',page=1,object_id='text',background_id=backing['background']['id'])
    assert protected_masks['report']['derived_seed_protected_pixels_excluded']>0
    assert protected_masks['status']=='NO_UNPROTECTED_SEED' and protected_masks['masks']=={}
    write(protection_file,retained)
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
    assert {'validation_reference_mask','unauthorized_mask','protected_mask'} <= submitted['checks']['evidence'].keys()
    for ref in submitted['checks']['evidence'].values():assert state.evidence(ref['id'])[0].is_file()
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
    # Retained formal-check receipts must not consume additional exploration routes.
    current=state.read()
    for index in range(3):
        receipt={**draft,'id':f'formal-{index}','fingerprint':f'formal-{index}',
                 'exploratory':False,'submission_of':draft['id']}
        current['candidates'][receipt['id']]=receipt
    write(state.root/'state.json',current)
    another=call(service,identifier,'create_candidate',page=1,object_id='text',
        mask_id=masks['masks']['semantic_completion']['id'],background_id=backing['background']['id'],
        frame_minimum=[100,30])
    assert another['status']=='EXPLORATORY'


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
    review = {'passed': True, 'findings': [], 'observations':[{'item':'synthetic descriptive observation'}], 'minor_deviations': [],
              'evidence_ids': [source['id'], preview['id']],
              'checked': ['content','completeness','layout','residual','background','editing'], 'reason': 'synthetic'}
    with pytest.raises(ValueError, match='provided'):
        call(service, identifier, 'record_visual_review', review=review)
    for reference in (source, preview):
        state.evidence(reference['id'], delivered=True)
    assert call(service, identifier, 'record_visual_review', review=review)['status'] == 'PASS'
    assert call(service,identifier,'record_visual_review',review={**review,'passed':False,'findings':[{'item':'unresolved defect'}]})['status']=='REVIEW_REQUIRED'
    with pytest.raises(ValueError,match='unresolved defects'):
        call(service,identifier,'record_visual_review',review={**review,'findings':[{'item':'unresolved defect'}]})
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


def test_incremental_scene_keeps_uncommitted_text_and_shared_styles(job,monkeypatch):
    service,identifier,_=job
    from product import fonts
    monkeypatch.setattr(fonts,'font_manifest',lambda texts:{'fonts':[
        {'family':'Arial','bold':False,'weight':400,'cjk':False}]})
    call(service,identifier,'create_scene',page=1,evidence={'text':{
        'styles':[{'id':'body','role':'body','ea':'Arial','latin':'Arial','size_px':12,'bold':False,'color':'000000'}],
        'texts':[{'id':'first','text':'First','bbox':[2,2,30,15],'style':'body','z':1}]}})
    result=call(service,identifier,'update_scene',page=1,evidence={'text':{
        'texts':[{'id':'second','text':'Second','bbox':[2,22,45,15],'style':'body','z':2}]}})
    assert {t['id'] for t in result['scene']['text']['texts']}=={'first','second'}
    assert [s['id'] for s in result['scene']['text']['styles']]==['body']
    assert result['scene']['selected']=={}
    added=call(service,identifier,'update_scene',page=1,evidence={'text':{
        'texts':[{'id':'third','text':'Third','bbox':[2,42,35,15],'style':'body','z':3}]}})
    assert {t['id'] for t in added['scene']['text']['texts']}=={'first','second','third'}

    before=service.state(identifier).read()['pages'][0]['scene']
    exists=call(service,identifier,'create_scene',page=1,evidence={})
    assert exists['status']=='SCENE_EXISTS'
    duplicate=call(service,identifier,'update_scene',page=1,evidence={'text':{'texts':[
        {'id':'alias','text':'First','bbox':[2,2,30,15],'style':'body','z':1}]}})
    assert duplicate['status']=='IDENTITY_CONFLICT'
    assert duplicate['duplicates'][0]['existing_id']=='first'
    assert service.state(identifier).read()['pages'][0]['scene']==before
    removed=call(service,identifier,'update_scene',page=1,evidence={},remove_object_ids=['second'])
    assert {t['id'] for t in removed['scene']['text']['texts']}=={'first','third'}
    assert service.call('get_progress',identifier)['reconstruction']=='NO_EDITABLE_RECONSTRUCTION'
    refs=service.call('list_evidence',identifier,role='agent_scene_proposal')['evidence']
    assert refs and all(r['role']=='agent_scene_proposal' for r in refs)


def test_catalog_exposes_real_asset_contract_and_unavailable_sandbox(job,monkeypatch):
    from jsonschema import Draft202012Validator
    monkeypatch.delenv('IMAGE2PPT_SANDBOX_IMAGE',raising=False)
    service,identifier,_=job
    assert service.call('get_job_context',identifier)['tool_availability']['run_image_script']['status']=='UNAVAILABLE'
    catalog={tool.name:tool.inputSchema for tool in schemas()}
    for schema in catalog.values():Draft202012Validator.check_schema(schema)
    obj=catalog['segment_object']['properties']['object_evidence']
    assert {'id','description','prompt_en','parts'} <= set(obj['required'])
    assert 'text' not in obj['properties']['kind']['enum']
    assert 'original_mask_id' not in catalog['complete_text_mask']['required']


def test_removed_draft_cannot_remove_independent_ocr_protection(tmp_path):
    from types import SimpleNamespace
    from product.parallel_text_candidates import peer_protection
    import product.peer_evidence as peers
    import numpy as np
    from unittest.mock import patch
    graph=SimpleNamespace(source_sha256='source',text=SimpleNamespace(members={'target':['own']}),ownership={'target':'RESOLVED'})
    write(tmp_path/'runtime-ocr-protection.json',{'source_sha256':'source','lines':[
        {'id':'own','bbox':[2,2,8,8]},{'id':'removed-peer','bbox':[2,12,8,8]}]})
    def observed(text,graph,source,blocked,baseline_directory,**kw):return blocked
    with patch.object(peers,'protection',observed):
        mask=peer_protection(SimpleNamespace(id='target'),graph,np.zeros((30,30,3),np.uint8),np.zeros((30,30),bool),baseline_directory=tmp_path)
    assert mask[15,5] and not mask[5,5]
    graph.source_sha256='changed'
    with pytest.raises(ValueError,match='source hash mismatch'):
        peer_protection(SimpleNamespace(id='target'),graph,np.zeros((30,30,3),np.uint8),np.zeros((30,30),bool),baseline_directory=tmp_path)


def test_missing_background_inputs_do_not_reserve_idempotency_key(job):
    from jsonschema import validate,ValidationError
    service,identifier,_=job
    args={'job_id':identifier,'idempotency_key':'background-missing-001'}
    schema=next(t.inputSchema for t in schemas() if t.name=='fit_background')
    with pytest.raises(ValidationError):validate(args,schema)
    with pytest.raises(ValueError,match='no operation was reserved'):
        service.call('fit_background',**args)
    with service.state(identifier).db() as db:
        assert db.execute('SELECT count(*) FROM operations WHERE id=?',(args['idempotency_key'],)).fetchone()[0]==0
    validate({**args,'page':1,'object_id':'draft'},schema)


def test_region_reference_is_not_reserved_as_candidate_submission(job):
    service,identifier,_=job
    with pytest.raises(ValueError,match='region IDs and object IDs'):
        service.call('commit_region',identifier,expected_version=0,idempotency_key='invalid-region-submit',candidate_id='region-not-a-candidate')
    with service.state(identifier).db() as db:
        assert db.execute('SELECT count(*) FROM operations').fetchone()[0]==0


def test_lateral_icon_intrusion_does_not_contaminate_gradient_validation():
    import numpy as np
    from product.recovery.gradient_backing import fit_gradient
    # White card with an unowned blue icon 8px to the left of the text bbox.
    source = np.full((80, 220, 3), 252, np.uint8)
    source[28:50, 48:58] = [25, 118, 255]
    backing, report = fit_gradient(source, [66, 25, 90, 30], np.zeros((80, 220), bool))
    assert report['excluded_foreground_context_pixels'] > 0
    assert report['validation_p95_rgb'] <= 1.0
    assert np.abs(backing[25:55, 66:156].astype(int) - 252).max() <= 1
    # Genuine two-tone step edge across the text bbox must NOT be masked out as a small icon.
    step = np.full((80, 220, 3), 252, np.uint8)
    step[:, :110] = [40, 80, 160]
    _, step_report = fit_gradient(step, [66, 25, 90, 30], np.zeros((80, 220), bool))
    assert step_report['excluded_foreground_context_pixels'] == 0
    assert step_report['validation_p95_rgb'] > 10.0


def test_nonplanar_inpaint_ring_and_stroke_ablation_proof():
    import numpy as np
    from product.recovery.diagnostics import assess_trial
    y, x = np.mgrid[:90, :180]
    clean_bg = np.clip(np.rint(180 + 35 * np.sin(x / 9.0) + 15 * np.cos(y / 11.0))[..., None], 0, 255).astype(np.uint8)
    clean_bg = np.repeat(clean_bg, 3, axis=2)
    source = clean_bg.copy()
    source[34:54, 72:76] = 15
    source[34:54, 84:88] = 15
    mask = np.zeros((90, 180), bool)
    mask[32:56, 70:90] = True
    protected = np.zeros_like(mask)
    box = [65, 28, 50, 34]
    fake_ref = np.full_like(source, 180)
    unreliable_fit = {'validation_p95_rgb': 28.0}
    # Without verified nonplanar provenance, unreliable planar reference stays REVIEW_REQUIRED.
    plain, _, _ = assess_trial(source, mask, clean_bg, protected, box, fake_ref, unreliable_fit, {})
    assert plain['status'] == 'REVIEW_REQUIRED'
    # Bad inpaint that leaves original dark strokes fails ablation check.
    bad, _, _ = assess_trial(source, mask, source, protected, box, fake_ref, unreliable_fit, {'method': 'big-lama'})
    assert bad['status'] == 'REVIEW_REQUIRED'
    # Clean inpaint erasing strokes smoothly on curved background passes via ring+ablation proof.
    good, _, _ = assess_trial(source, mask, clean_bg, protected, box, fake_ref, unreliable_fit, {'method': 'big-lama'})
    assert good['status'] == 'PRECHECK_PASS'
    assert good['measurement_interpretation']['reference_dependent_quality'] == 'MEASURED_RING_ABLATION'


def test_decorative_subspan_ocr_binding_and_remainder_protection(job, monkeypatch):
    import numpy as np
    from PIL import ImageDraw, ImageFont
    from product import fonts
    service, identifier, directory = job
    manifest = fonts.font_manifest([])
    record = next(r for r in manifest['fonts'] if r['family'] == 'Arial' and not r['bold'])
    monkeypatch.setattr(fonts, 'font_manifest', lambda texts: {'fonts': [record]})
    image = Image.new('RGB', (320, 100), 'white')
    font = ImageFont.truetype(record['file'], 22, index=record['index'])
    draw = ImageDraw.Draw(image)
    draw.text((30, 30), 'Focus >>', font=font, fill='black')
    core_box = draw.textbbox((30, 30), 'Focus', font=font)
    full_box = draw.textbbox((30, 30), 'Focus >>', font=font)
    core_bbox = [core_box[0], core_box[1], core_box[2] - core_box[0], core_box[3] - core_box[1]]
    full_bbox = [full_box[0], full_box[1], full_box[2] - full_box[0], full_box[3] - full_box[1]]
    image.save(directory / 'inputs/01.png')
    ocr = {'lines': [{'id': 'line-dec', 'text': 'Focus >>', 'score': 0.98, 'bbox': full_bbox}]}
    write(directory / 'ocr.json', ocr)
    state = service.state(identifier)
    current = state.read()
    current['ocr'] = {'1': str(directory / 'ocr.json')}
    write(state.root / 'state.json', current)
    scene = call(service, identifier, 'create_scene', page=1, evidence={'text': {
        'styles': [{'id': 's', 'role': 'body', 'ea': 'Arial', 'latin': 'Arial', 'size_px': 22, 'bold': False, 'color': '000000'}],
        'texts': [
            {'id': 'sub', 'text': 'Focus', 'bbox': core_bbox, 'style': 's', 'z': 1, 'align': 'left'},
            {'id': 'trunc', 'text': 'Foc', 'bbox': [core_bbox[0], core_bbox[1], 20, core_bbox[3]], 'style': 's', 'z': 2, 'align': 'left'},
        ]}})
    assert scene['scene']['ownership']['sub'] == 'RESOLVED'
    assert scene['scene']['text']['members']['sub'] == ['line-dec']
    # Truncating real alphanumeric letters ('us' from 'Focus') must remain AMBIGUOUS!
    assert scene['scene']['ownership']['trunc'] == 'AMBIGUOUS'
    call(service, identifier, 'update_scene', page=1, evidence={}, remove_object_ids=['trunc'])
    backing = call(service, identifier, 'fit_background', page=1, object_id='sub')
    masks = call(service, identifier, 'complete_text_mask', page=1, object_id='sub', background_id=backing['background']['id'])
    assert masks['status'] == 'COMPUTED'
    mask_arr = np.array(Image.open(state.evidence(masks['masks']['semantic_completion']['id'])[0]).convert('L')) > 0
    # Target word has stroke mask, while decorative '>>' on the right remains untouched (protected).
    assert mask_arr[:, core_bbox[0]:core_bbox[0] + core_bbox[2]].any()
    assert not mask_arr[:, full_bbox[0] + full_bbox[2] - 8:full_bbox[0] + full_bbox[2]].any()


def test_pixel_ink_calibration_overrides_wrong_model_color_and_splits_mixed_style(tmp_path):
    import numpy as np
    from PIL import ImageDraw, ImageFont
    from product import fonts
    from product.parallel_contracts import SceneGraph
    from product.parallel_store import Store
    from product.parallel_typography import calibrate
    manifest = fonts.font_manifest([])
    record = next(r for r in manifest['fonts'] if r['family'] == 'Arial' and not r['bold'])
    image = Image.new('RGB', (320, 140), 'white')
    font = ImageFont.truetype(record['file'], 24, index=record['index'])
    draw = ImageDraw.Draw(image)
    draw.text((25, 20), 'BlueTitle', font=font, fill=(22, 93, 255))
    draw.text((25, 75), 'DarkBody', font=font, fill=(30, 30, 30))
    b1 = draw.textbbox((25, 20), 'BlueTitle', font=font)
    b2 = draw.textbbox((25, 75), 'DarkBody', font=font)
    bbox1 = [float(b1[0]), float(b1[1]), float(b1[2] - b1[0]), float(b1[3] - b1[1])]
    bbox2 = [float(b2[0]), float(b2[1]), float(b2[2] - b2[0]), float(b2[3] - b2[1])]
    graph = SceneGraph.model_validate({
        'schema': 'SceneGraph/1', 'source_sha256': 'a' * 64, 'image_size': [320, 140], 'transform': [1, 0, 0, 1, 0, 0],
        'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': 'a' * 64, 'image_size': [320, 140], 'method_version': 'test',
                 'styles': [{'id': 'shared', 'role': 'body', 'ea': 'Arial', 'latin': 'Arial', 'size_px': 24.0, 'bold': False, 'color': '000000'}],
                 'texts': [{'id': 't1', 'text': 'BlueTitle', 'bbox': bbox1, 'style': 'shared', 'z': 1},
                           {'id': 't2', 'text': 'DarkBody', 'bbox': bbox2, 'style': 'shared', 'z': 2}],
                 'members': {'t1': ['l1'], 't2': ['l2']}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': 'a' * 64, 'image_size': [320, 140], 'method_version': 'test', 'objects': []},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': 'a' * 64, 'image_size': [320, 140], 'method_version': 'test', 'background': 'FFFFFF', 'shapes': []},
        'ownership': {'t1': 'RESOLVED', 't2': 'RESOLVED'}, 'relations': []})
    calibrated = calibrate(graph, np.array(image), {'fonts': [record]}, Store(tmp_path / 'store'), adaptive=True)
    styles = {s.id: s for s in calibrated.text.styles}
    texts = {t.id: t for t in calibrated.text.texts}
    assert texts['t1'].style != texts['t2'].style
    c_blue = [int(styles[texts['t1'].style].color[i:i + 2], 16) for i in (0, 2, 4)]
    assert c_blue[2] > 200 and c_blue[0] < 60


def test_prepare_page_evidence_and_batch_reconstruct_page_with_auto_frame_repair(job, monkeypatch):
    from PIL import ImageDraw, ImageFont
    from product import fonts
    import product.runtime.compiler as compiler
    service, identifier, directory = job
    manifest = fonts.font_manifest([])
    record = next(r for r in manifest['fonts'] if r['family'] == 'Arial' and not r['bold'])
    monkeypatch.setattr(fonts, 'font_manifest', lambda texts: {'fonts': [record]})
    image = Image.new('RGB', (420, 180), (246, 249, 255))
    font = ImageFont.truetype(record['file'], 22, index=record['index'])
    draw = ImageDraw.Draw(image)
    draw.text((35, 30), 'Header >>', font=font, fill=(20, 90, 220))
    draw.text((35, 95), 'Metric 98%', font=font, fill=(40, 40, 40))
    b1 = draw.textbbox((35, 30), 'Header >>', font=font)
    b2 = draw.textbbox((35, 95), 'Metric 98%', font=font)
    ocr_data = {'lines': [
        {'id': 'o1', 'text': 'Header >>', 'score': 0.99, 'bbox': [b1[0], b1[1], b1[2] - b1[0], b1[3] - b1[1]]},
        {'id': 'o2', 'text': 'Metric 98%', 'score': 0.99, 'bbox': [b2[0], b2[1], b2[2] - b2[0], b2[3] - b2[1]]},
    ]}
    image.save(directory / 'inputs/01.png')
    monkeypatch.setattr(service, 'run_ocr', lambda state, current, d, *, page: (
        write(d / 'ocr.json', ocr_data),
        current.setdefault('ocr', {}).update({str(page): str(d / 'ocr.json')}),
        {'result': ocr_data})[2])
    prep = call(service, identifier, 'prepare_page_evidence', page=1, run_layers=False)
    assert prep['status'] == 'PREPARED' and prep['resolved_text_count'] == 2
    assert prep['items'][0]['text'] == 'Header' and prep['items'][0]['stripped_affix'] == '>>'
    # First Office check reports overflow on t-001; second check passes after auto frame repair.
    def fake_compile(irs, out, texts):
        out.mkdir(parents=True, exist_ok=True)
        (out / 'deck.pptx').write_bytes(b'pptx:' + Path(irs[0]).read_bytes())
        return {'pptx': str(out / 'deck.pptx')}
    office_calls = []
    def fake_office(state, pptx, irs, out):
        office_calls.append(len(office_calls) + 1)
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        if len(office_calls) == 1:
            res = {'status': 'REVIEW_REQUIRED', 'files': {}, 'residuals': [[{'id': 't-001', 'residual_check_status': 'PASS'}, {'id': 't-002', 'residual_check_status': 'PASS'}]],
                   'replacements': {'records': [{'id': 't-002', 'content_matches': True, 'siblings_unchanged': True, 'frame_unchanged': True, 'outside_pixels': 5,
                                                 'layout': {'overflow': True, 'unexpected_wrap': False, 'frame': [26.0, 72.0, 60.0, 24.0], 'text_bounds': [26.0, 72.0, 72.0, 24.0]}}]}}
        else:
            res = {'status': 'PASS', 'files': {}, 'residuals': [[]], 'replacements': {'records': []}}
        write(out / 'checks.json', res)
        return res
    monkeypatch.setattr(compiler, 'compile_local', fake_compile)
    monkeypatch.setattr(service, 'office_check', fake_office)
    batch = call(service, identifier, 'batch_reconstruct_page', page=1, use_inpaint=False)
    assert batch['status'] == 'COMMITTED'
    assert batch['committed_count'] == 2
    assert batch['auto_repaired_frame_ids'] == ['t-002']
    assert len(office_calls) == 2
    progress = service.call('get_progress', identifier)
    assert progress['reconstruction'] == 'NATIVE_OBJECTS_COMMITTED'
    assert progress['pages'][0]['committed_objects'] == 2


def test_mild_interior_nonplanar_routes_to_layerd_while_blocking_foreign_stroke(tmp_path):
    import numpy as np
    from PIL import ImageDraw, ImageFont
    from product import fonts
    from product.parallel_branches import snapshot
    from product.parallel_contracts import SceneGraph
    from product.parallel_rebuild import build
    from product.recovery.region_candidates import adaptive_candidates
    from product.recovery.background_adapter import RecoveryBlocked

    manifest = fonts.font_manifest([])
    record = next(r for r in manifest['fonts'] if r['family'] == 'Arial' and not r['bold'])
    font = ImageFont.truetype(record['file'], 20, index=record['index'])
    w, h = 360, 120

    # Two-stage progress bar whose perimeter around text is uniform [254, 254, 254],
    # while interior left half has mild fill [245, 250, 254] (delta=9 <= 22).
    text_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(text_layer)
    tdraw.text((55, 46), 'Progress Step Label', font=font, fill=(25, 35, 50, 255))
    tb = tdraw.textbbox((55, 46), 'Progress Step Label', font=font)
    bbox = [float(tb[0]), float(tb[1]), float(tb[2] - tb[0]), float(tb[3] - tb[1])]

    layer0 = np.full((h, w, 4), [254, 254, 254, 255], dtype=np.uint8)
    layer1 = np.zeros((h, w, 4), dtype=np.uint8)
    layer1[int(bbox[1]) + 2:int(bbox[1] + bbox[3]) - 2, int(bbox[0]) + 8:int(bbox[0]) + 80] = [245, 250, 254, 255]
    clean_rgb = Image.alpha_composite(Image.fromarray(layer0), Image.fromarray(layer1)).convert('RGB')

    src_img = Image.alpha_composite(clean_rgb.convert('RGBA'), text_layer).convert('RGB')
    layers_dir = tmp_path / 'layers'
    layers_dir.mkdir()
    Image.fromarray(layer0).save(layers_dir / 'layer-00.png')
    Image.fromarray(layer1).save(layers_dir / 'layer-01.png')
    text_layer.save(layers_dir / 'layer-02.png')
    src_img.save(layers_dir / 'source.png')
    write(layers_dir / 'manifest.json', {
        'layer_order': 'back_to_front', 'image_size': [w, h], 'source_sha256': digest(layers_dir / 'source.png'),
        'layers': [{'file': f'layer-0{i}.png', 'z': i, 'size': [w, h], 'sha256': digest(layers_dir / f'layer-0{i}.png')} for i in range(3)],
    })
    snap = snapshot(layers_dir / 'source.png', tmp_path / 'snap')
    graph = SceneGraph.model_validate({
        'schema': 'SceneGraph/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'transform': [1, 0, 0, 1, 0, 0],
        'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'method_version': 'test',
                 'styles': [{'id': 's1', 'role': 'body', 'ea': 'Arial', 'latin': 'Arial', 'size_px': 20.0, 'bold': False, 'color': '192332'}],
                 'texts': [{'id': 't1', 'text': 'Progress Step Label', 'bbox': bbox, 'style': 's1', 'z': 1}],
                 'members': {'t1': ['l1']}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'method_version': 'test', 'objects': []},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'method_version': 'test', 'background': 'FEFEFE', 'shapes': []},
        'ownership': {'t1': 'RESOLVED'}, 'relations': [],
    })
    base_dir = tmp_path / 'base'
    build(graph, snap, {'fonts': [record]}, base_dir, strategy='off', text_candidates=False, surface_candidates=False)

    rows = list(adaptive_candidates(graph, 't1', layers_dir / 'source.png', layers_dir, tmp_path / 'trials', baseline_directory=base_dir))
    assert any(r['status'] == 'PRECHECK_PASS' and r['method'].startswith('layerd-') for r in rows)

    # Injecting a high-contrast unseeded foreign stroke (delta > 22) even WITH LayerD must still raise MASK_INCOMPLETE.
    bad_arr = np.array(src_img)
    bad_arr[int(bbox[1]) - 2:int(bbox[1]), int(bbox[0]) - 2:int(bbox[0])] = [220, 40, 40]
    bad_layers = tmp_path / 'bad_layers'
    bad_layers.mkdir()
    Image.fromarray(layer0).save(bad_layers / 'layer-00.png')
    Image.fromarray(layer1).save(bad_layers / 'layer-01.png')
    text_layer.save(bad_layers / 'layer-02.png')
    Image.fromarray(bad_arr).save(bad_layers / 'source.png')
    write(bad_layers / 'manifest.json', {
        'layer_order': 'back_to_front', 'image_size': [w, h], 'source_sha256': digest(bad_layers / 'source.png'),
        'layers': [{'file': f'layer-0{i}.png', 'z': i, 'size': [w, h], 'sha256': digest(bad_layers / f'layer-0{i}.png')} for i in range(3)],
    })
    bad_snap = snapshot(bad_layers / 'source.png', tmp_path / 'bad_snap')
    bad_graph = graph.model_copy(update={'source_sha256': bad_snap['source_sha256']})
    bad_graph.text.source_sha256 = bad_snap['source_sha256']
    bad_graph.asset.source_sha256 = bad_snap['source_sha256']
    bad_graph.surface.source_sha256 = bad_snap['source_sha256']
    bad_base = tmp_path / 'bad_base'
    build(bad_graph, bad_snap, {'fonts': [record]}, bad_base, strategy='off', text_candidates=False, surface_candidates=False)
    with pytest.raises(RecoveryBlocked) as exc_info:
        list(adaptive_candidates(bad_graph, 't1', bad_layers / 'source.png', bad_layers, tmp_path / 'bad_trials', baseline_directory=bad_base))
    assert exc_info.value.code == 'MASK_INCOMPLETE'


def test_whole_badge_and_illustration_discovery_and_deferred_caption_staging(job, monkeypatch, tmp_path):
    import numpy as np
    from PIL import ImageDraw, ImageFont
    from product import fonts
    from product.parallel_residual_gate import inspect_residuals

    service, identifier, directory = job
    manifest = fonts.font_manifest([])
    record = next(r for r in manifest['fonts'] if r['family'] == 'Arial' and not r['bold'])
    monkeypatch.setattr(fonts, 'font_manifest', lambda texts: {'fonts': [record]})

    w, h = 520, 260
    card_bg = (252, 253, 255, 255)
    # Layer 0 has card background + baked pale-blue circular badge [40..124, 40..124]
    l0 = Image.new('RGBA', (w, h), card_bg)
    d0 = ImageDraw.Draw(l0)
    d0.ellipse((40, 40, 124, 124), fill=(216, 234, 253, 255))

    # Layer 1 has foreground gold icon inside badge + large right-side multi-part illustration
    l1 = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d1 = ImageDraw.Draw(l1)
    d1.rectangle((68, 62, 96, 92), fill=(245, 175, 35, 255))
    d1.rectangle((280, 40, 470, 215), fill=(45, 115, 235, 255))
    d1.ellipse((310, 70, 440, 185), fill=(120, 195, 255, 255))

    # Layer 2 has caption text overlapping the bottom rim of the pale-blue circle badge!
    l2 = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    font = ImageFont.truetype(record['file'], 18, index=record['index'])
    d2 = ImageDraw.Draw(l2)
    d2.text((46, 110), 'BadgeLabel', font=font, fill=(20, 45, 95, 255))
    tb = d2.textbbox((46, 110), 'BadgeLabel', font=font)
    tb_box = [float(tb[0]), float(tb[1]), float(tb[2] - tb[0]), float(tb[3] - tb[1])]

    comp01 = Image.alpha_composite(l0, l1)
    full_src = Image.alpha_composite(comp01, l2).convert('RGB')
    full_src.save(directory / 'inputs/01.png')

    layers_dir = tmp_path / 'layers_fixture'
    layers_dir.mkdir()
    l0.save(layers_dir / 'layer-00.png')
    l1.save(layers_dir / 'layer-01.png')
    l2.save(layers_dir / 'layer-02.png')
    full_src.save(layers_dir / 'source.png')
    write(layers_dir / 'manifest.json', {
        'layer_order': 'back_to_front', 'image_size': [w, h], 'source_sha256': digest(layers_dir / 'source.png'),
        'layers': [{'file': f'layer-0{i}.png', 'z': i, 'size': [w, h], 'sha256': digest(layers_dir / f'layer-0{i}.png')} for i in range(3)],
    })

    ocr_data = {'lines': [{'id': 'o1', 'text': 'BadgeLabel', 'score': 0.99,
                           'bbox': tb_box, 'polygon': [[tb[0], tb[1]], [tb[2], tb[1]], [tb[2], tb[3]], [tb[0], tb[3]]]}]}
    monkeypatch.setattr(service, 'run_ocr', lambda state, current, d, *, page: (
        write(d / 'ocr.json', ocr_data),
        current.setdefault('ocr', {}).update({str(page): str(d / 'ocr.json')}),
        {'result': ocr_data})[2])
    monkeypatch.setattr(service, 'decompose_layers', lambda state, current, d, *, page: (
        current.setdefault('layers', {}).update({str(page): str(layers_dir)}),
        {'layers': [], 'images': []})[1])

    prep = call(service, identifier, 'prepare_page_evidence', page=1, run_layers=True)
    assert prep['status'] == 'PREPARED'
    assert prep['asset_count'] == 2
    assert prep['immediate_asset_count'] == 1
    assert prep['deferred_asset_count'] == 1

    batch = call(service, identifier, 'batch_reconstruct_page', page=1,
                 include_assets=True, use_inpaint=False, verify_office=False)
    assert batch['status'] == 'STAGED_LOCAL'
    assert batch['staged_count'] == 3
    assert batch['failed_count'] == 0

    cand_ir = Path(service.state(identifier).read()['candidates'][batch['candidate_id']]['ir'])
    ir_json = json.loads(cand_ir.read_text('utf8'))
    asset_nodes = [n for n in ir_json['nodes'] if n['type'] == 'image' and n['id'] != 'source-plate']
    text_nodes = [n for n in ir_json['nodes'] if n['type'] == 'text']
    assert len(asset_nodes) == 2 and len(text_nodes) == 1
    assert max(n['z'] for n in asset_nodes) < min(n['z'] for n in text_nodes)

    # Badge PNG must encompass the entire pale-blue circle (> 80x80), and source-plate must be cleanly erased to card_bg!
    badge_node = min(asset_nodes, key=lambda n: n['bbox'][0])
    assert badge_node['bbox'][2] >= 80 and badge_node['bbox'][3] >= 80
    plate_arr = np.array(Image.open(cand_ir.parent / 'source-plate.png').convert('RGB'))
    assert np.abs(plate_arr[50:115, 50:115].astype(float) - np.array(card_bg[:3])).max() <= 2.0

    # Residual gate on badge deletion passes even when higher-z native caption text remains rendered in delete preview!
    badge_id = badge_node['id'].removeprefix('extracted-')
    delete_preview = Image.new('RGBA', (w, h), card_bg)
    delete_preview = Image.alpha_composite(delete_preview, l2).convert('RGB')
    del_dir = tmp_path / 'office_del'
    del_dir.mkdir()
    delete_preview.save(del_dir / f'01-{badge_id}-delete.png')
    residuals = inspect_residuals(cand_ir, del_dir, 1, tmp_path / 'residuals.json')
    badge_res = next(r for r in residuals if r['id'] == badge_id)
    assert badge_res['residual_check_status'] == 'PASS'


def test_unmix_prior_rejects_stale_erased_text_and_illustration_halo_completes():
    import numpy as np
    from product.object_contracts import ClosedObject
    from product.object_unmix import candidate_from_evidence
    from product.parallel_rebuild import source_owner
    from product.recovery.asset_retry import complete_source_halo

    # 1. Stale dark erased-text ink in original_alpha must NOT leak into work_source white card,
    #    while genuine white-on-white interior pixels in original_alpha ARE preserved.
    work_source = np.full((120, 120, 3), 254, dtype=np.uint8)
    work_source[25:75, 25:75] = [218, 234, 252]  # pale-blue badge
    work_source[42:58, 42:58] = [254, 254, 254]  # genuine white-on-white highlight inside badge
    prior = np.zeros((120, 120, 4), dtype=np.uint8)
    prior[42:58, 42:58] = [254, 254, 254, 255]   # white highlight
    prior[80:92, 35:65] = [60, 60, 60, 255]      # stale dark caption ink already erased from work_source!
    badge_obj = ClosedObject(
        id='a-badge', description='badge', prompt_en='badge', kind='icon',
        bbox=[20, 20, 65, 78], layers=['l.png'], components=['whole'],
        text_policy='exclude_independent_text',
        parts=[{'name': 'whole', 'bbox': [20, 20, 65, 78], 'layers': ['l.png']}],
    )
    rgba, _ = candidate_from_evidence(badge_obj, work_source, np.zeros((120, 120), bool), original_alpha=prior)
    assert rgba[50, 50, 3] == 255                # white-on-white interior preserved
    assert int(rgba[80:92, 35:65, 3].max()) == 0 # stale dark text ink rejected

    # 2. Clean illustration completes soft interior glow/particles (dist ~ 12, peak ~ 20) while blocking strong strokes.
    ill_src = np.full((220, 220, 3), 250, dtype=np.uint8)
    ill_src[30:180, 30:160] = [45, 115, 235]
    ill_src[90:96, 172:180] = [232, 238, 250]    # 48px soft particle 12px away (peak=18 <= 28)
    seed = np.zeros((220, 220, 4), dtype=np.uint8)
    seed[30:180, 30:160, :3] = ill_src[30:180, 30:160]
    seed[30:180, 30:160, 3] = 255
    ill_obj = ClosedObject(
        id='a-ill', description='ai illustration', prompt_en='ai illustration', kind='illustration',
        bbox=[25, 25, 165, 165], layers=['l.png'], components=['whole'],
        text_policy='exclude_independent_text',
        parts=[{'name': 'whole', 'bbox': [25, 25, 165, 165], 'layers': ['l.png']}],
    )
    completed, report = complete_source_halo(ill_obj, seed, ill_src, np.zeros((220, 220), bool), None)
    assert report['added_pixels'] == 48
    _, _, owner_mask, _ = source_owner(ill_obj, completed, ill_src, np.zeros((220, 220), bool), None)
    assert owner_mask[92, 175]

    # Strong unowned stroke (peak=150 > 28) inside illustration still blocks!
    ill_src_bad = ill_src.copy()
    ill_src_bad[120:124, 172:176] = [100, 100, 100]
    completed_bad, _ = complete_source_halo(ill_obj, seed, ill_src_bad, np.zeros((220, 220), bool), None)
    with pytest.raises(ValueError, match='no complete candidate owner'):
        source_owner(ill_obj, completed_bad, ill_src_bad, np.zeros((220, 220), bool), None)


def test_hierarchical_container_and_surface_staging(job, monkeypatch, tmp_path):
    import numpy as np
    from PIL import ImageDraw, ImageFont
    from product import fonts
    from product.parallel_residual_gate import inspect_residuals

    service, identifier, directory = job
    manifest = fonts.font_manifest([])
    record = next(r for r in manifest['fonts'] if r['family'] == 'Arial' and not r['bold'])
    monkeypatch.setattr(fonts, 'font_manifest', lambda texts: {'fonts': [record]})

    w, h = 560, 320
    canvas_rgb = (254, 254, 254, 255)
    l0 = Image.new('RGBA', (w, h), canvas_rgb)
    d0 = ImageDraw.Draw(l0)
    # 1) Top chamfered banner with internal vertical dashed divider -> should become container asset PNG (a-c*)
    d0.polygon([(30, 25), (510, 25), (528, 43), (528, 105), (48, 105), (30, 87)],
               fill=(248, 251, 254, 255), outline=(25, 100, 200, 255))
    for yy in range(38, 95, 8):
        d0.line([(160, yy), (160, yy + 4)], fill=(40, 110, 205, 255), width=2)
    # 2) Outer white bordered rounded card [30..530, 130..295] -> should become outer native roundRect (s-*)
    d0.rounded_rectangle((30, 130, 530, 295), radius=8, fill=(254, 254, 254, 255), outline=(205, 220, 240, 255), width=1)
    # 3) Inner tinted pill bar [60..500, 165..220] -> should become inner native roundRect (s-*) above outer card
    d0.rounded_rectangle((60, 165, 500, 220), radius=7, fill=(238, 243, 249, 255))

    l1 = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    font = ImageFont.truetype(record['file'], 18, index=record['index'])
    d1 = ImageDraw.Draw(l1)
    d1.text((190, 54), 'TopBannerText', font=font, fill=(20, 60, 140, 255))
    d1.text((85, 182), 'InnerBarText', font=font, fill=(35, 40, 50, 255))
    b1 = d1.textbbox((190, 54), 'TopBannerText', font=font)
    b2 = d1.textbbox((85, 182), 'InnerBarText', font=font)

    full_src = Image.alpha_composite(l0, l1).convert('RGB')
    full_src.save(directory / 'inputs/01.png')

    layers_dir = tmp_path / 'cont_layers'
    layers_dir.mkdir()
    l0.save(layers_dir / 'layer-00.png')
    l1.save(layers_dir / 'layer-01.png')
    full_src.save(layers_dir / 'source.png')
    write(layers_dir / 'manifest.json', {
        'layer_order': 'back_to_front', 'image_size': [w, h], 'source_sha256': digest(layers_dir / 'source.png'),
        'layers': [{'file': f'layer-0{i}.png', 'z': i, 'size': [w, h], 'sha256': digest(layers_dir / f'layer-0{i}.png')} for i in range(2)],
    })
    ocr_data = {'lines': [
        {'id': 'o1', 'text': 'TopBannerText', 'score': 0.99, 'bbox': [float(b1[0]), float(b1[1]), float(b1[2] - b1[0]), float(b1[3] - b1[1])]},
        {'id': 'o2', 'text': 'InnerBarText', 'score': 0.99, 'bbox': [float(b2[0]), float(b2[1]), float(b2[2] - b2[0]), float(b2[3] - b2[1])]},
    ]}
    monkeypatch.setattr(service, 'run_ocr', lambda state, current, d, *, page: (
        write(d / 'ocr.json', ocr_data),
        current.setdefault('ocr', {}).update({str(page): str(d / 'ocr.json')}),
        {'result': ocr_data})[2])
    monkeypatch.setattr(service, 'decompose_layers', lambda state, current, d, *, page: (
        current.setdefault('layers', {}).update({str(page): str(layers_dir)}),
        {'layers': [], 'images': []})[1])

    prep = call(service, identifier, 'prepare_page_evidence', page=1, run_layers=True)
    assert prep['status'] == 'PREPARED'
    batch = call(service, identifier, 'batch_reconstruct_page', page=1,
                 include_assets=True, include_containers=True, use_inpaint=False, verify_office=False)
    assert batch['status'] == 'STAGED_LOCAL' and batch['failed_count'] == 0

    cand_ir = Path(service.state(identifier).read()['candidates'][batch['candidate_id']]['ir'])
    ir_json = json.loads(cand_ir.read_text('utf8'))
    shapes = [n for n in ir_json['nodes'] if n['type'] == 'shape']
    cont_imgs = [n for n in ir_json['nodes'] if n['type'] == 'image' and n['id'].startswith('a-c')]
    texts = [n for n in ir_json['nodes'] if n['type'] == 'text']
    assert len(shapes) == 2 and len(cont_imgs) == 1 and len(texts) == 2

    # Outer card has lower z than inner pill bar, and all containers sit below texts!
    outer_s = max(shapes, key=lambda n: n['bbox'][2] * n['bbox'][3])
    inner_s = min(shapes, key=lambda n: n['bbox'][2] * n['bbox'][3])
    assert outer_s['z'] < inner_s['z'] < min(n['z'] for n in texts)
    assert cont_imgs[0]['z'] < min(n['z'] for n in texts)

    # Base plate is now 100% blank canvas!
    plate_arr = np.array(Image.open(cand_ir.parent / 'source-plate.png').convert('RGB'))
    assert int((np.abs(plate_arr.astype(float) - 254.0).max(2) > 4).sum()) == 0


def test_odd_tenth_size_px_snaps_cleanly_for_ooxml_compile_and_office_packet_prunes_probes(job, tmp_path):
    from product.parallel_typography import snap_ooxml_size_px
    from product.parallel_pipeline import effective_texts
    from product.runtime.compiler import compile_local

    # 18.3px and 17.1px previously hit .5 hundredth-point boundary (1372.5 / 1282.5) and failed verify_package
    for raw in (18.3, 17.1, 15.5, 13.9, 21.7):
        snapped = snap_ooxml_size_px(raw)
        exact_hundredths = snapped * 0.75 * 100.0
        assert abs(exact_hundredths - round(exact_hundredths)) < 1e-9

    service, identifier, directory = job
    call(service, identifier, 'list_fonts')
    ocr = {'lines': [{'id': 'l-odd', 'text': 'Hello', 'score': 0.99, 'bbox': [18.0, 20.0, 56.0, 24.0]}]}
    write(directory / 'ocr.json', ocr)
    st = service.state(identifier)
    cur = st.read()
    cur['ocr'] = {'1': str(directory / 'ocr.json')}
    write(st.root / 'state.json', cur)
    scene_res = call(service, identifier, 'create_scene', page=1, evidence={'text': {
        'styles': [{'id': 's-odd', 'role': 'body', 'ea': 'Arial', 'latin': 'Arial',
                    'size_px': 18.3, 'bold': False, 'color': '112233'}],
        'texts': [{'id': 't-odd', 'text': 'Hello', 'bbox': [18.0, 20.0, 56.0, 24.0],
                   'style': 's-odd', 'z': 1, 'align': 'left'}],
    }})
    assert scene_res['scene']['text']['styles'][0]['size_px'] == 18.4

    # Verify compile_local + verify_package succeeds cleanly on the normalized style
    state = service.state(identifier)
    current = state.read()
    row = service.page(current, 1)
    graph = service.graph(row)
    # Stage t-odd into SlideIR via adapt and compile
    from product.parallel_rebuild import adapt
    stage_dir = tmp_path / 'odd_stage'
    import shutil
    shutil.copytree(Path(row['ir']).parent, stage_dir)
    graph.selected['t-odd'] = {'id': 't-odd', 'kind': 'text', 'bbox': [18.0, 20.0, 56.0, 24.0], 'payload': graph.text.texts[0].model_dump()}
    ir_dict = adapt(graph, stage_dir)
    write(stage_dir / 'slideir.json', ir_dict)
    compiled = compile_local([stage_dir / 'slideir.json'], tmp_path / 'odd_compiled', [effective_texts(graph, stage_dir / 'slideir.json')])
    assert Path(compiled['pptx']).is_file()


def test_submit_accepts_vendor_prefixed_moma_agent_model_and_rejects_traversal(job, tmp_path, monkeypatch):
    import io
    import product.app as app_module
    import product.pipeline as pipeline
    from product.runtime.runner import prepare_workspace

    _, _, directory = job
    for name in ('PLANNER_API_KEY', 'PLANNER_BASE_URL', 'PLANNER_MODEL'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('IMAGE2PPT_AGENT_PROVIDER', 'moma')
    monkeypatch.setenv('MOMA_API_KEY', 'synthetic-moma-secret')
    monkeypatch.setenv('IMAGE2PPT_AGENT_MODEL', 'qwen/qwen3.5-plus')
    req_path = directory / 'request.json'
    req = json.loads(req_path.read_text('utf8'))
    req.update(agent_inference_provider='moma', agent_model='qwen/qwen3.5-plus')
    write(req_path, req)
    workspace = prepare_workspace(directory)
    encoded = (workspace / 'opencode.json').read_text('utf8')
    cfg = json.loads(encoded)
    assert cfg['providers']['image2ppt-moma']['models']['vision']['modelID'] == 'qwen/qwen3.5-plus'
    assert 'synthetic-moma-secret' not in encoded

    app_root = tmp_path / 'api_root'
    app_root.mkdir()
    monkeypatch.setattr(app_module, 'ROOT', app_root)
    monkeypatch.setattr(pipeline, 'settings', lambda: object())
    client = TestClient(app_module.create_app(app_root / 'jobs', token='t' * 32, run_worker=False))
    picture = io.BytesIO()
    Image.new('RGB', (80, 60)).save(picture, 'PNG')
    auth = {'Authorization': 'Bearer ' + 't' * 32}

    ok = client.post(
        '/api/v1/jobs',
        params={'agent_model': 'qwen/qwen3.5-plus'},
        files=[('files', ('x.png', picture.getvalue(), 'image/png'))],
        headers=auth,
    )
    assert ok.status_code == 202
    saved_req = json.loads((app_root / 'jobs' / ok.json()['id'] / 'request.json').read_text('utf8'))
    assert saved_req['agent_model'] == 'qwen/qwen3.5-plus'

    for bad_model in ('../bad', 'a/b/c'):
        bad = client.post(
            '/api/v1/jobs',
            params={'agent_model': bad_model},
            files=[('files', ('x.png', picture.getvalue(), 'image/png'))],
            headers=auth,
        )
        assert bad.status_code == 422





