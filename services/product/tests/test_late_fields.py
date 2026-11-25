"""Synthetic, source-bound late OCR and field tests. Office is measured separately."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from product.materialize import digest, write
from product.parallel_branches import snapshot
from product.parallel_contracts import AssetEvidence, SurfaceEvidence, TextEvidence
from product.parallel_fusion import fuse
from product.recovery.marked_fields import separated_fields
from product.runtime import late_fields
from product.runtime.plate_audit import visible_divider_lines
from product.runtime.reading_targets import commit_readings
from product.runtime.service import RuntimeService
from product.runtime.state import RuntimeConflict, RuntimeState


def read(path):
    return json.loads(Path(path).read_text('utf8'))


@pytest.fixture
def page(tmp_path):
    image=Image.new('RGB',(280,130),'white')
    draw=ImageDraw.Draw(image)
    draw.text((30,104),'Footer',fill='black')
    draw.text((170,104),'03',fill='blue')
    source=tmp_path/'source.png'
    image.save(source)
    state=RuntimeState(tmp_path)
    snap=snapshot(source,tmp_path/'snapshots')
    binding=dict(source_sha256=snap['source_sha256'],image_size=snap['image_size'],method_version='fixture')
    branches={name:contract.model_validate({**binding,'schema':name.title()+'Evidence/1',
                                           **({'background':'FFFFFF'} if name=='surface' else {})})
              for name,contract in [('text',TextEvidence),('surface',SurfaceEvidence),('asset',AssetEvidence)]}
    graph=fuse(snap,branches,{'lines':[]})
    base=tmp_path/'base'
    write(base/'scene.json',graph.model_dump())
    write(base/'slideir.json',{'nodes':[]})
    image.save(base/'source-plate.png')
    write(tmp_path/'ocr.json',{'source_sha256':digest(source),'lines':[]})
    write(tmp_path/'marks.json',{'source_sha256':digest(source),'rows':[
        {'id':'m-icon','mark':4,'bbox':[5,10,20,20],'label':'icon'}]})
    c={'pages':[dict(snapshot=snap,ir=str(base/'slideir.json'),scene=str(base/'scene.json'))],
       'experiment_marks':{'1':str(tmp_path/'marks.json')},'host_evidence':{'1':{'blocks':[]}},
       'ocr':{'1':str(tmp_path/'ocr.json')},'source_ocr':{'1':str(tmp_path/'ocr.json')},
       'marked_plans':{'1':[{'id':'m-logo','role':'brand','route':'preserve','bbox':[55,10,40,20]}]},
       'composition':{'bound':'unchanged'}}
    return RuntimeService(tmp_path.parent),state,c


def test_register_verified_footers_only_as_numbered_drafts(page,monkeypatch,tmp_path):
    service,state,c=page
    seen=[]
    def local(image,box,directory,cache,**kwargs):
        seen.append(box)
        return {'crop_sha256':'f'*64, 'readings':[{'bbox':box,'text':'Footer' if box[0]==30 else '08','score':.98}]}
    monkeypatch.setattr(late_fields,'source_reading',local)
    finds=[{'bbox':[30,104,30,12],'text':'Footer'},{'bbox':[170,104,12,12],'text':'03'},
           {'bbox':[5,10,20,20],'text':'X'}, {'bbox':[55,10,40,20],'text':'Logo'}]
    rows=late_fields.register(service,state,c,tmp_path/'first',page=1,findings=finds)
    assert [r['status'] for r in rows]==['ready','pending','skipped','skipped']
    footer=rows[0]
    graph=service.graph(c['pages'][0])
    assert graph.ownership[footer['id']]=='RESOLVED' and graph.selected=={}
    assert [t.text for t in graph.text.texts]==['Footer']
    assert graph.text.members[footer['id']]==[footer['line']['id']]
    assert c['marked_plans']['1'][-1]['route']=='text'
    marks=read(c['experiment_marks']['1'])['rows']
    assert [m['mark'] for m in marks]==[4,5,6] and marks[1]['id']==footer['id']
    from product.runtime.marked_routes import bind_plan
    routes=bind_plan([c['marked_plans']['1'][-1]],c['experiment_marks']['1'],graph,
                     c['pages'][0]['snapshot']['path'],separate_cards=True)
    assert routes[0]['target_ids']==[footer['id']]
    assert rows[1]['reason']=='source_readings_unconfirmed' and len(seen)==2
    assert read(c['source_ocr']['1'])['lines']==[]
    assert read(c['ocr']['1'])['lines']==[footer['line']]
    assert c['composition']=={'bound':'unchanged'}
    assert digest(Path(c['pages'][0]['ir']).parent/'source-plate.png')==digest(tmp_path/'base/source-plate.png')
    again=late_fields.register(service,state,c,tmp_path/'again',page=1,findings=[finds[0]])
    assert again==[footer] and len(seen)==2


def test_split_change_history_offsets_are_relative_to_source():
    line={'id':'line','text':'LeftRightCode'}
    a=dict(ocr_id='line',object_id='left',source_span=[0,4],before='Left',after='First long')
    b=dict(ocr_id='line',object_id='right',source_span=[4,9],before='Right',after='Second',status='pending_confirmation')
    commit_readings({'line':line},[a])
    commit_readings({'line':line},[b])
    assert line['text']=='First longSecondCode' and line['ocr_raw_text']=='LeftRightCode'
    assert line['field_readings']['right']['status']=='pending_confirmation'
    assert 'proofread_verified' not in line
    line['text']='changed elsewhere'
    with pytest.raises(RuntimeConflict):
        commit_readings({'line':line},[{**b,'object_id':'third'}])


def test_unnumbered_review_needs_two_blind_original_crop_readings(page,monkeypatch,tmp_path):
    service,state,c=page
    scales=[]
    def local(image,box,directory,cache,*,scale=2):
        scales.append(scale)
        word='30%' if scale==2 else '30元'
        return {'crop_sha256':str(scale)*64,'readings':[{'bbox':box,'text':word,'score':.99}]}
    monkeypatch.setattr(late_fields,'source_reading',local)
    rows=late_fields.register(service,state,c,tmp_path/'discover',page=1,findings=[{'bbox':[30,104,30,12]}])
    assert scales==[2,3] and rows[0]['status']=='pending'
    assert rows[0]['source_reading']=='30%' and rows[0]['observed_text']=='30元'
    assert not service.graph(c['pages'][0]).text.texts


def test_late_child_joins_only_a_uniquely_owned_native_card(tmp_path):
    from types import SimpleNamespace
    proof=tmp_path/'face-proof'
    proof.write_bytes(b'fixture')
    face={'kind':'surface','bbox':[0,0,140,60],'mark_id':'card',
          'owner_evidence':{'file':proof.name,'sha256':digest(proof)}}
    graph=SimpleNamespace(selected={'native-face':face},ownership={'native-face':'RESOLVED'},
        relations=[SimpleNamespace(kind='contains',status='RESOLVED',source='native-face',target='late')])
    plan={'id':'card','bbox':face['bbox'],'origin':'host','route':'surface','role':'card'}
    assert late_fields.verified_parent(graph,'late',[plan],tmp_path) is plan
    assert late_fields.verified_parent(graph,'late',[plan,{**plan,'id':'other'}],tmp_path) is None
    proof.write_bytes(b'different fixture')
    assert late_fields.verified_parent(graph,'late',[plan],tmp_path) is None


def test_separator_needs_distinct_rule_and_visible_hidden_export(tmp_path):
    im=Image.new('RGB',(120,24),'white')
    draw=ImageDraw.Draw(im)
    for x in (8,17,82,91):
        draw.rectangle((x,3,x+3,20),fill='black')
    box=[6,2,91,20]
    for color,expected in [('black','AB|'),('blue','AB')]:
        draw.rectangle((54,3,55,20),fill=color)
        ruled=[]
        fields=separated_fields(np.asarray(im),'AB|CD',box,'101010',dividers=ruled)
        assert fields[0][0]==expected
    im.save(tmp_path/'original.png')
    hidden=tmp_path/'hidden.png'
    im.save(hidden)
    line={'id':'l','bbox':box,'text':'AB|CD'}
    entry={'l':{'source_divider':{**ruled[0],'source_text':'AB|CD'}}}
    assert visible_divider_lines([line],entry,tmp_path/'original.png',hidden,{'sha256':digest(hidden)})[0]['text']=='ABCD'
    ImageDraw.Draw(im).rectangle((54,2,55,21),fill='white')
    im.save(hidden)
    assert visible_divider_lines([line],entry,tmp_path/'original.png',hidden,{'sha256':digest(hidden)})==[line]
