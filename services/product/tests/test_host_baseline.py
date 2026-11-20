"""Synthetic host lifecycle tests. OCR and Office are controlled evidence providers."""
import json
import shutil
from pathlib import Path
from threading import Barrier
from uuid import uuid4
from zipfile import ZipFile

import httpx
import pytest
from PIL import Image

from product.materialize import digest, write
from product.runtime import runner, host_baseline as host
from product.runtime.host_evidence import read, normalize_arguments
from product.runtime.service import RuntimeService
from product.runtime.state import RuntimeState


def test_office_cache_uses_worker_that_actually_checked(job, monkeypatch):
    state = RuntimeState(job)
    deck, ir = job/'sample.pptx', job/'sample.json'
    deck.write_bytes(b'test packet')
    ir.write_text('{}')
    monkeypatch.setenv('IMAGE2PPT_OFFICE_MODE','remote')
    calls = []
    def submit(*args):
        calls.append(True)
        return {'status':'PASS','worker_binding':'worker-b'}
    monkeypatch.setattr('product.runtime.office_bridge.submit',submit)
    binding = job.parent/'office-worker-binding.json'
    write(binding, {'binding':'worker-a'})
    RuntimeService.office_check(state,deck,[ir],job/'one')
    write(binding, {'binding':'worker-b'})
    assert RuntimeService.office_check(state,deck,[ir],job/'two')['cache_hit']
    assert len(calls) == 1


@pytest.fixture
def job(tmp_path,monkeypatch):
    job = tmp_path/uuid4().hex
    (job/'inputs').mkdir(parents=True)
    for i,color in enumerate(('white','blue'),1):
        Image.new('RGB',(128,80),color).save(job/'inputs'/f'{i}.png')
    write(job/'request.json',{'sources':['1.png','2.png'],'host_stage0':True,'orchestrator':'opencode'})
    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT',str(tmp_path/'leases'))
    monkeypatch.delenv('IMAGE2PPT_EXPERIMENT_CONFIG',raising=False)
    monkeypatch.setattr('product.layout_evidence.available',lambda:False)
    barrier = Barrier(2,timeout=20)
    def ocr(self,state,current,directory,*,page):
        barrier.wait()
        file = directory/'ocr.json'
        write(file,{'lines':[]})
        current.setdefault('ocr',{})[str(page)] = str(file)
    def layers(*args,**kwargs):
        raise RuntimeError('fixture has no layers')
    def office(self,state,current,directory):
        deck = current['composition']['artifacts']['pptx']
        directory.mkdir(parents=True,exist_ok=True)
        for i in (1,2):
            shutil.copyfile(job/f'inputs/{i}.png',directory/f'page-{i:02d}-before.png')
        result = {'status':'PASS','pptx_sha256':digest(deck),'office_directory':str(directory),
                  'office':{'pptx_sha256':digest(deck),'status':'PASS_APP_AUTOMATED','records':[]}}
        current['composition']['office'] = result
        return result
    monkeypatch.setattr(RuntimeService,'run_ocr',ocr)
    monkeypatch.setattr(RuntimeService,'decompose_layers',layers)
    monkeypatch.setattr(RuntimeService,'validate_office',office)
    return job


def test_first_inference_429_delivers_frozen_control(job,monkeypatch):
    workspace = job/'agent'
    write(workspace/'opencode.json',{'providers':{'fixture':{}}})
    monkeypatch.setattr(runner,'prepare_workspace',lambda _:workspace)
    class Limited:
        def verify(self):
            assert (job/'host-baseline/manifest.json').is_file()
        def request(self,method,path,**kwargs):
            if path.endswith('/prompt'):
                request = httpx.Request(method,'https://example.invalid/model')
                response = httpx.Response(429,request=request)
                raise httpx.HTTPStatusError('quota',request=request,response=response)
            if path=='/api/mcp':
                return {'data':[{'name':'image2ppt','status':{'status':'connected'}}]}
            return {'data':[]}
    result = runner.run(job,Limited())
    assert result['origin']==host.ORIGIN and result['status']=='REVIEW_REQUIRED'
    assert '429' in result['reason'] and not result['agent']['participated']
    assert digest(job/'delivery/result.pptx')==digest(job/'host-baseline/baseline.pptx')
    with ZipFile(job/'delivery/result.pptx') as deck:
        assert 'ppt/slides/slide2.xml' in deck.namelist()
    base = host.ensure_baseline(job)
    assert base['sources']==[digest(job/f'inputs/{i}.png') for i in (1,2)]
    Image.new('RGB',(128,80),'green').save(job/'inputs/1.png')
    with pytest.raises(ValueError,match='changed'):
        host.ensure_baseline(job)


def test_locally_worse_agent_result_rolls_back_file_and_state(job,monkeypatch):
    def worse(directory,client):
        state = RuntimeState(directory)
        current = state.read()
        fallback = current['composition']
        other = directory/'challenger'
        other.mkdir()
        candidate = other/'worse.pptx'
        shutil.copyfile(fallback['artifacts']['pptx'],candidate)
        with ZipFile(candidate,'a') as deck:
            deck.comment=b'synthetic contender'
        shutil.copytree(fallback['office']['office_directory'],other/'office')
        Image.new('RGB',(128,80),'black').save(other/'office/page-01-before.png')
        sha = digest(candidate)
        fallback.update(artifacts={'pptx':str(candidate)},files={str(candidate):sha})
        fallback['office'].update(pptx_sha256=sha,office_directory=str(other/'office'))
        fallback['office']['office']['pptx_sha256']=sha
        current.update(status='SUCCEEDED',visual_review={'status':'PASS'})
        write(state.root/'state.json',current)
        return {'status':'SUCCEEDED'}
    monkeypatch.setattr(runner,'_agent_run',worse)
    result = runner.run(job)
    assert result['origin']==host.ORIGIN and result['status']=='REVIEW_REQUIRED'
    assert any('p1:LOCAL_VISUAL' in r for r in result['agent']['fallback_reasons'])
    assert digest(job/'delivery/result.pptx')==result['baseline_sha256']
    assert RuntimeState(job).read()['pages']==read(job/'host-baseline/state.json')['pages']
    base = read(job/'host-baseline/scorecard.json')
    assert not host.regressions(base,base)
    degraded = json.loads(json.dumps(base))
    base['pages'][0].update(ocr_native_hits=['L1'],editable_count=1)
    assert {'p1:NATIVE_FIELDS_LOST','p1:EDITABLE_COUNT_DECLINED'}<=set(host.regressions(base,degraded))


def test_small_visual_differences_are_allowed_in_editable_regions(tmp_path):
    src, preview = tmp_path/'source.png', tmp_path/'preview.png'
    Image.new('RGB', (96, 50), '#fcfcfc').save(src)
    Image.new('RGB', (96, 50), '#f6f6f6').save(preview)
    marks = [{'id':'card', 'bbox':[8, 9, 82, 28]}]
    row = dict(page=1, source_sha256='fixed', editable_count=1, native_field_count=1, ocr_native_hits=['t1'])
    base = dict(verified=True, pages=[dict(row, local_fidelity=host.local_fidelity(src, src, marks))])
    candidate = dict(verified=True, pages=[dict(row, local_fidelity=host.local_fidelity(src, preview, marks))])
    assert set(candidate['pages'][0]['local_fidelity']) == {'whole-page', 'card'}
    assert candidate['pages'][0]['local_fidelity']['card']['mae_rgb'] > 2
    assert host.regressions(base, candidate) == []


def test_only_reviewed_transcription_gets_small_ink_tolerance():
    marks = {'caption': {'ssim': .20, 'delta_e2000_p95': 40.0},
             'neighbor': {'ssim': .20, 'delta_e2000_p95': 40.0}}
    row = dict(page=1, source_sha256='fixed', editable_count=2, native_field_count=2,
               ocr_native_hits=[], local_fidelity=marks)
    base = dict(verified=True, pages=[row])
    contender = json.loads(json.dumps(base))
    page = contender['pages'][0]
    page['pending_readings'] = {'caption': {'before':'旧读法', 'after':'新读法'}}
    page['local_fidelity']['caption'] = {'ssim': .195, 'delta_e2000_p95': 40.5}
    assert host.regressions(base,contender) == ['p1:LOCAL_VISUAL_REGRESSION:caption']
    assert not host.regressions(base,contender,reviewed_reading_draft=True)
    page['local_fidelity']['neighbor']['ssim'] = .195
    assert host.regressions(base,contender,reviewed_reading_draft=True) == ['p1:LOCAL_VISUAL_REGRESSION:neighbor']
    page['local_fidelity']['caption']['ssim'] = .15
    assert 'caption' in host.regressions(base,contender,reviewed_reading_draft=True)[0]


def test_reviewed_unchanged_candidate_and_retries_use_one_selected_deck(job,monkeypatch):
    calls = []
    def finalize(directory,client):
        calls.append('agent')
        runtime = RuntimeState(directory)
        cur = runtime.read()
        candidate = runtime.root/'agent-delivery/result.pptx'
        candidate.parent.mkdir()
        shutil.copyfile(cur['composition']['artifacts']['pptx'],candidate)
        cur.update(status='SUCCEEDED',visual_review={'status':'PASS'},independent_audit={'passed':True})
        write(runtime.root/'state.json',cur)
        return {'status':'SUCCEEDED','artifacts':{'result.pptx':digest(candidate)}}
    monkeypatch.setattr(runner,'_agent_run',finalize)
    first = runner.run(job)
    assert first['origin']==host.ORIGIN and not first['agent']['fallback_reasons']
    assert first['artifacts']['result.pptx']==digest(job/'host-baseline/baseline.pptx')
    assert first['artifacts']['agent-candidate.pptx']==digest(job/'runtime/agent-delivery/result.pptx')
    assert runner.run(job)==first and calls==['agent']


def test_published_reading_remains_downloadable_and_requires_content_confirmation(job):
    host.ensure_baseline(job)
    control = job/'host-baseline/baseline.pptx'
    runtime = RuntimeState(job)
    selected = runtime.root/'agent-delivery/result.pptx'
    selected.parent.mkdir()
    shutil.copyfile(control, selected)
    with ZipFile(selected,'a') as deck:
        deck.comment = b'reviewed sample'
    candidate = read(job/'host-baseline/scorecard.json')
    candidate['pptx_sha256'] = digest(selected)
    candidate['pages'][0]['pending_readings'] = {'caption': {'before':'旧读法','after':'新读法'}}
    result = host.publish(job, {'status':'SUCCEEDED','artifacts':{'result.pptx':digest(selected)}},
                          candidate=candidate)
    assert result['status'] == runtime.read()['status'] == 'REVIEW_REQUIRED'
    assert result['pending_text_objects'] == {'1':['caption']}
    assert result['artifacts']['result.pptx'] == digest(selected) != digest(control)
    assert result['workflow_states']['final_qa'] == 'REVIEW_REQUIRED'


def test_numbered_actions_reject_whole_page_and_unowned_marks(tmp_path):
    job=tmp_path/uuid4().hex
    job.mkdir()
    state=RuntimeState(job)
    write(job/'marks.json',{'rows':[{'mark':1,'id':'a'},{'mark':2,'id':'b'}]})
    write(job/'issues.json',{'items':[{'id':'a'}]})
    ref=state.register(job/'issues.json','numbered_questions')['id']
    c={'experiment_marks':{'1':str(job/'marks.json')},'questions':{'1':[{'block':1,'questions':ref}]}}
    batch=lambda **kw:normalize_arguments(state,c,'batch_reconstruct_page',{'page':1,**kw})
    assert batch(block=1)['object_ids']==['a']
    for kw in ({},{'object_ids':['b']}):
        with pytest.raises(ValueError):
            batch(**kw)
    field={'mark':1,'route':'surface','role':'card','move_with':[2]}
    for patch in ({'mark':2},{'text':'rewritten'},{'bbox':[0,0,1,1]}):
        with pytest.raises(ValueError):
            normalize_arguments(state,c,'plan_marked_page',{'page':1,'fields':[{**field,**patch}]})
    c['repair_rounds'] = {'1':[{'object_ids':['b'],'progress':False}]}
    assert batch(block=1)['object_ids'] == ['a']
    c['repair_rounds']['1'].append({'object_ids':['a'],'progress':False})
    with pytest.raises(ValueError, match='local repair limit'):
        batch(block=1)
