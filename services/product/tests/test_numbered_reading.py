"""Explicit numbered reading uses synthetic sources and providers in these tests."""
import json
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from product.materialize import digest, write
from product.runtime import page_reading, text_reading
from product.runtime.host_evidence import normalize_arguments
from product.runtime.reading_targets import revised_lines


def test_job_context_exposes_reading_inventory_even_without_reported_issues(tmp_path):
    from product.experiment import mark_blocks
    from product.runtime.host_evidence import context
    from product.runtime.state import RuntimeState

    state = RuntimeState(tmp_path)
    source = tmp_path/'inputs'/'page.png'
    source.parent.mkdir()
    Image.new('RGB', (120, 40), 'white').save(source)
    marks = [{'id':'t-001', 'mark':1, 'text':'Title', 'label':'title', 'bbox':[8, 8, 45, 18]}]
    blocks = [{**block, 'image':state.register(pic, 'numbered_block_image'),
               'inventory':state.register(inventory, 'numbered_block_inventory')}
              for block, pic, inventory in mark_blocks(source, marks, tmp_path)]
    scene = tmp_path/'scene.json'
    write(scene, {'selected':{}})
    write(state.root/'state.json', {'version':1, 'pages':[{'scene':str(scene)}],
          'host_evidence':{'1':{'blocks':blocks}}, 'questions':{'1':[]}})
    write(tmp_path/'host-baseline/scorecard.json', {'pptx_sha256':'fixture', 'office_status':'UNKNOWN',
          'pages':[{'page':1, 'editable_count':0, 'native_field_count':0, 'ocr_native_hits':[], 'max_local_mae':0}]})

    result = context(state, {'sources':['page.png']})['pages'][0]
    assert result['issue_blocks'] == []
    block, = result['reading_blocks']
    assert set(block) == {'block', 'image', 'inventory'} and block['block'] == 1
    path, receipt = state.evidence(block['inventory'])
    assert receipt['role'] == 'numbered_block_inventory'
    assert json.loads(path.read_text('utf8')) == marks


@pytest.fixture
def page(tmp_path):
    source = tmp_path/'source.png'
    Image.new('RGB', (220, 130), 'white').save(source)
    lines = [
        dict(id='l1', text='Title|Body', bbox=[10, 10, 120, 20], score=.99),
        dict(id='l2', text='30%', bbox=[10, 50, 50, 20], score=.99),
        dict(id='l3', text='Alone', bbox=[80, 50, 70, 20], score=.72),
    ]
    marks = [
        dict(id='detail', text='Body', bbox=[80, 10, 50, 20], label='text', ocr_ids=['l1']),
        dict(id='number', text='30%', bbox=lines[1]['bbox'], label='text', ocr_ids=['l2']),
        dict(id='ocr-review', bbox=lines[2]['bbox'], label='uncertain_ocr', ocr_ids=['l3'],
             intersecting_ocr_ids=['l2','l3']),
        dict(id='layout-text', bbox=lines[2]['bbox'], label='text', intersecting_ocr_ids=['l3']),
        dict(id='layout-graphic', bbox=lines[2]['bbox'], label='image', intersecting_ocr_ids=['l3']),
    ]
    def field(mark):
        return NS(id=mark['id'], text=mark['text'], spans=[], bbox=[*mark['bbox'][:2], 100, 20], align='left')
    scene = NS(text=NS(texts=[field(r) for r in marks[:2]], members={'detail':['l1'],'number':['l2']}),
               source_sha256=digest(source), image_size=[220,130], asset=NS(objects=[]), relations=[],
               selected={r['id']:{'kind':'text','payload':{'text':r['text']}} for r in marks[:2]})
    row = {'snapshot':{'path':str(source)}}
    service = NS(page=lambda c,p:row, graph=lambda r:scene)
    state = NS(job=tmp_path)
    current = {'fonts':tmp_path/'fonts.json', 'ocr':{'1':tmp_path/'ocr.json'},
               'experiment_marks':{'1':tmp_path/'marks.json'}, 'candidates':{}, 'version':3}
    write(current['fonts'], {})
    write(current['ocr']['1'], {'source_sha256':digest(source), 'lines':lines})
    write(current['experiment_marks']['1'], {'source_sha256':digest(source),'rows':marks})
    write(tmp_path/'request.json', {'sources':['source.png']})
    (tmp_path/'operation').mkdir()
    return service,state,current,tmp_path/'operation'


def requested(page, *ids):
    service,state,current,directory = page
    args = normalize_arguments(state,current,'review_numbered_text',{'page':1,'object_ids':list(ids)})
    return text_reading.stage_readings(service,state,current,directory,**args)


def test_explicit_high_score_split_and_uncommitted_readings(page, monkeypatch):
    monkeypatch.delenv('IMAGE2PPT_MOMA_READING_MODEL', raising=False)
    calls = []
    class Actor:
        model = 'fixture'
        def ask(self, prompt, context, images, contract):
            calls.append((images[0].name, Image.open(images[0]).size))
            assert not context
            return dict(text={'detail.png':'Body', 'number.png':'35%', 'l3.png':'Other'}[images[0].name], legible=True)
    monkeypatch.setattr(text_reading, 'planning_actor', lambda _:Actor())
    result = requested(page, 'detail','number','ocr-review','layout-text','layout-graphic','unknown')
    assert calls == [('detail.png',(100,40)), ('number.png',(100,40)), ('l3.png',(140,40))]
    assert result['checked'] == 3 and 'candidate_id' not in result
    items = {r['id']:r for r in result['items']}
    assert items['detail']['status'] == 'observed'
    assert (items['detail']['reading'],items['detail']['before'],items['detail']['legible']) == ('Body','Body',True)
    assert items['number']['reason'] == 'numeric_change_requires_independent_confirmation'
    assert items['layout-text']['target_id'] == items['ocr-review']['target_id'] == 'l3'
    assert items['layout-text']['reading'] == 'Other' and items['layout-text']['reading_status']=='pending_confirmation'
    assert items['layout-text']['delivery'] == 'observed_without_native_field'
    assert items['layout-graphic'] == {'id':'layout-graphic','status':'skipped','reason':'not_a_text_mark'}
    assert items['unknown'] == {'id':'unknown','status':'skipped','reason':'unknown_page_mark'}
    proof = json.loads((page[-1]/'readings.json').read_text('utf8'))
    assert proof['observations'][0]['source_span'] == [6,10]
    assert proof['items'] == result['items']
    assert proof['changes'] == []


def test_skip_ambiguous_geometry_and_unavailable_provider(page,monkeypatch):
    service,state,current,_ = page
    data = json.loads(current['experiment_marks']['1'].read_text('utf8'))
    data['rows'][0]['bbox'] = [0,90,100,20]
    write(current['experiment_marks']['1'],data)
    monkeypatch.setenv('IMAGE2PPT_MOMA_READING_MODEL','')
    def unavailable(_):
        raise RuntimeError('fixture provider unavailable')
    monkeypatch.setattr(text_reading,'planning_actor',unavailable)
    result = requested(page,'detail','number')
    assert result['checked'] == 0 and result['status'] == 'READING_MODEL_UNAVAILABLE'
    assert result['items'][0]['reason'] == 'field_outside_source_line'
    assert result['items'][1]['status'] == 'unavailable'
    data['source_sha256'] = '0'*64
    write(current['experiment_marks']['1'],data)
    with pytest.raises(ValueError,match='different source'):
        requested(page,'detail')


def test_verified_split_stages_just_native_child_and_binds_crop(page,monkeypatch):
    from copy import deepcopy
    service,state,current,directory = page
    row = service.page(current,1)
    scene = service.graph(row)
    original = deepcopy(scene)
    scene.text.texts[0].align = 'center'
    scene.model_dump = lambda:{'selected':scene.selected}
    base = state.job/'base'
    row.update(ir=base/'slideir.json', scene=base/'scene.json')
    write(row['ir'],{})
    write(row['scene'],{'selected':scene.selected})
    service.code_binding = lambda:'fixture'
    monkeypatch.delenv('IMAGE2PPT_MOMA_READING_MODEL', raising=False)
    actor = NS(model='fixture', ask=lambda *args:{'legible':True,'text':'Bode'})
    monkeypatch.setattr(text_reading,'planning_actor',lambda _:actor)
    def verify(pixels, ocr, corrections, fonts):
        line, = ocr['lines']
        assert line['bbox'] == [80,10,50,20] and line['text'] == 'Body'
        return ocr,[dict(ocr_id=line['id'],before='Body',after='Bode',iou_before=.8,iou_after=.9)]
    monkeypatch.setattr(text_reading,'verify_and_apply_ocr_corrections',verify)
    monkeypatch.setattr(text_reading,'adapt',lambda graph,work:{'texts':[t.text for t in graph.text.texts]})
    out = requested(page,'detail')
    assert out['checked']==1 and out['object_ids']==['detail'] and out['status']=='STAGED_LOCAL'
    candidate = current['candidates'][out['candidate_id']]
    assert str(directory/'detail.png') in candidate['files']
    assert scene.selected['detail']['source_reading']['source_span']==[6,10]
    assert scene.text.texts[1].text == original.text.texts[1].text
    proof = json.loads((directory/'readings.json').read_text('utf8'))
    text_reading.authorize_readings(original, scene, proof)


def test_two_view_split_is_not_interpreted_as_whole_line(page,monkeypatch):
    service,state,current,directory = page
    write(state.job/'request.json', {'sources':['source.png'], 'page_reading':True})
    monkeypatch.setenv('IMAGE2PPT_MOMA_READING_MODEL','fixture')
    sessions = []
    class Reader:
        def __init__(self, model):
            self.session = str(len(sessions)+1)
            sessions.append(self.session)
            self.last_usage = {}
        def ask(self,prompt,context,images,contract):
            assert context['field_numbers'] == [1,2]
            second = 'New' if prompt==page_reading.PAGE_PROMPT else 'Word'
            return {'fields':[{'number':i,'text':text,'legible':True,'kind':'ordinary'}
                              for i,text in enumerate(['Body',second],1)]}
    monkeypatch.setattr(page_reading,'MomaReviewer',Reader)
    result = requested(page,'detail','layout-text')
    assert result['checked'] == 2 and len(sessions)==2
    decisions = json.loads((directory/'readings.json').read_text('utf8'))
    detail, layout = decisions['observations']
    assert detail['bbox'] == [80,10,50,20] and detail['before'] == 'Body' and detail['source_span'] == [6,10]
    assert layout['proposed'] == 'New' and layout['written_text'] is None
    item = result['items'][1]
    assert 'reading' not in item and item['reading_status']=='pending_confirmation'
    assert item['readings'] == {'page':{'text':'New','legible':True},'crop':{'text':'Word','legible':True}}
    assert 'candidate_id' not in result


def test_missing_field_in_provider_reply_is_not_a_reading(page,monkeypatch):
    service,state,current,directory = page
    write(state.job/'request.json', {'sources':['source.png'], 'page_reading':True})
    monkeypatch.setenv('IMAGE2PPT_MOMA_READING_MODEL','fixture')
    class Partial:
        def __init__(self, model):
            self.session = 'partial-provider'
            self.last_usage = {}
        def ask(self,prompt,context,images,contract):
            return {'fields':[{'number':1,'text':'','legible':False,'kind':'ordinary'}]}
    monkeypatch.setattr(page_reading,'MomaReviewer',Partial)
    result = requested(page,'detail','layout-text')
    assert result['checked'] == 0 and result['status'] == 'READING_MODEL_UNAVAILABLE'
    first, missing = result['items']
    assert first['status'] == 'unavailable' and first['reason'] == 'no_reading_returned'
    assert missing['status'] == 'unavailable' and missing['unavailable_views'] == {'page':'requested_field_missing'}


def test_automatic_high_confidence_policy_and_split_score(page,monkeypatch):
    service,state,current,directory = page
    monkeypatch.delenv('IMAGE2PPT_MOMA_READING_MODEL', raising=False)
    monkeypatch.setattr(text_reading,'planning_actor',lambda _:object())
    result = text_reading.stage_readings(service,state,current,directory,page=1,object_ids=['detail','number'])
    assert result == {'page':1,'status':'NO_VERIFIED_READING','checked':0}
    lines = [{'id':'l1','text':'Title|Body'}]
    first = {'ocr_id':'l1','before':'Title','after':'Name','source_span':[0,5]}
    last = {'ocr_id':'l1','before':'Body','after':'Main','source_span':[6,10]}
    assert revised_lines(lines,{'child':last}) == {'l1':'Title|Main'}
    assert revised_lines(lines,{'head':first,'child':last}) == {'l1':'Name|Main'}
    assert revised_lines(lines,{'child':dict(last,before='wrong')}) == {}
