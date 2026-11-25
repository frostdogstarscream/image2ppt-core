import json
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.materialize import digest, write
from product.runtime.editable_draft import delivery_page_summaries
from product.runtime.service import RuntimeService


def source_scene(path, size, text_size):
    binding = dict(source_sha256=digest(path), image_size=list(size), method_version='fixture')
    style = dict(id='role', role='section_header', ea='Microsoft YaHei', latin='Arial',
                 size_px=float(text_size), bold=text_size == 20, color='054080')
    text = dict(id='t', style='role', bbox=[4., 8., 70., 20.], text='部门\n职责', z=1)
    shape = dict(id='b', bbox=[2., 2., 88., 40.], z=0, geometry='rect', fill='D0E0F0')
    return {**binding, 'schema':'SceneGraph/1', 'text':{**binding, 'schema':'TextEvidence/1',
            'texts':[text], 'styles':[style], 'members':{'t':['l0','l1']}},
            'surface':{**binding,'schema':'SurfaceEvidence/1','background':'FFFFFF','shapes':[shape]},
            'asset':{**binding,'schema':'AssetEvidence/1','objects':[]},
            'relations':[{'kind':'contains','source':'b','target':'t','status':'AMBIGUOUS','evidence':['overlap']}],
            'ownership':{'t':'UNOWNED','b':'AMBIGUOUS'}}


def test_direct_draft_binds_order_and_keeps_style_disagreement_and_uncertain_ownership(tmp_path):
    jid = 'a'*32
    job = tmp_path/jid
    (job/'inputs').mkdir(parents=True)
    files = ['z.png', 'a.png', 'unprepared.png']
    request = {'orchestrator':'editable_reconstruction', 'visual_review_backend':'moma', 'sources':files}
    write(job/'request.json', request)
    service = RuntimeService(tmp_path)
    state = service.state(jid)
    pages, structure = [], {}
    for i,name in enumerate(files,1):
        size = (i*200, i*100)
        source = job/'inputs'/name
        Image.new('RGB',size).save(source)
        if i==3:
            continue
        graph = source_scene(source, size, 20*i)
        f = job/f'scene{i}.json'
        write(f, graph); pages.append({'scene':str(f)})
        invent = job/f'inventory{i}.json'
        write(invent, {'source_sha256':digest(source),'elements':[
            {'id':'c','category':'card_shape','bbox':[2,2,88,40]},
            {'id':'h','category':'section_header','bbox':[4,8,70,20],'parent_id':'c'}]})
        structure[str(i)] = str(invent)
    write(state.root/'state.json', {'version':3,'pages':pages,'page_structure':structure})
    result = service.call('get_editable_draft',jid)
    assert result['style_families'][0]['weight_disagreement']
    assert result['style_families'][0]['median_relative_size'] == .2
    obj = result['pages'][0]['objects'][0]
    assert obj['owner']=='UNOWNED' and obj['edit_actions']==['replace_paragraph','move'] and obj['text']=='部门\n职责'
    assert result['pages'][0]['relations'][-1]['status']=='PROPOSED'
    assert result['pages'][2]['status']=='MISSING_SCENE_EVIDENCE'
    assert result['pages'][0]['source_sha256']==digest(job/'inputs/z.png')
    assert result==service.call('get_editable_draft',jid)
    assert state.read()['version']==3 and not result['suggestions_grant_pixel_ownership']
    evidence_file,_ = state.evidence(result['evidence']['id'])
    assert json.loads(evidence_file.read_text('utf8'))['style_families']==result['style_families']
    Image.new('RGB',(90,120)).save(job/'inputs/z.png')
    with pytest.raises(ValueError, match='source pixels'):
        service.call('get_editable_draft',jid)
    request.update(orchestrator='opencode',visual_review_backend='opencode')
    write(job/'request.json',request)
    with pytest.raises(ValueError, match='editable reconstruction'):
        service.call('get_editable_draft',jid)


def test_delivery_summary_shows_only_observed_bound_actions(tmp_path, monkeypatch):
    pptx=tmp_path/'result.pptx'; pptx.write_bytes(b'compiled stub')
    ir=tmp_path/'ir.json'; write(ir,{'canvas':{'source_px':[100,50]},'groups':[]})
    current={'composition':{'artifacts':{'pptx':str(pptx)},'files':{str(p):digest(p) for p in [pptx,ir]}},
             'pages':[{'ir':str(ir),'snapshot':{'source_sha256':'a'*64}}]}
    monkeypatch.setattr('product.runtime.editable_draft.inspect_pptx', lambda *args:{'objects':[
        {'class':'plate','name':'source','bbox':[0,0,100,50]},{'class':'text','name':'title','bbox':[1,2,30,5]}]})
    office={'pptx_sha256':digest(pptx),'status':'PASS','office':{'pptx_sha256':digest(pptx),'records':[
        {'id':'title','page':1,'action':'replace','saved_reopened':True,'outside_pixels':0}]}}
    result=delivery_page_summaries(current,office)[0]
    assert result['text_boxes']==1 and result['raster_backing_present'] and result['editable_recall'] is None
    assert result['observed_edit_actions'][0]['id']=='title'
    office['office']['pptx_sha256']='b'*64
    result=delivery_page_summaries(current,office)[0]
    assert result['office_status']=='UNKNOWN' and result['observed_edit_actions']==[]
