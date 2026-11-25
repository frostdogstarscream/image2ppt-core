"""Exercise native paragraph flow, protected dividers, and local heading weight."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from product.contracts import Style, Text
from product.parallel_contracts import EDITABLE_TEXT_METHOD, native_paragraph_text
from product.recovery.candidate_stage import paragraph_edit_room
from product.runtime.batch_backbone import _merge_structural_ocr_items
from product.runtime.semantic_text import apply_heading_weights, is_body_continuation


def test_edit_room_stops_at_card_content_and_protected_source_without_moving_text():
    source = np.full((120,200,3),250,np.uint8)
    protected = np.zeros((120,200),bool)
    frame = [20.,20.,110.,40.]
    carrier = [10.,10.,150.,68.]
    full, _ = paragraph_edit_room(source,protected,frame,20.,[],[])
    assert full == [20.,20.,110.,60.]
    bounded, cap = paragraph_edit_room(source,protected,frame,20.,[],[carrier])
    assert bounded == [20.,20.,110.,57.] and cap['limited_by'] == 'carrier_edge'
    bounded, cap = paragraph_edit_room(source,protected,frame,20.,[('screen',[65.,70.,75.,40.])],[carrier])
    assert bounded == [20.,20.,110.,49.] and cap['limited_by'] == 'screen'
    protected[64,25] = True
    bounded, _ = paragraph_edit_room(source,protected,frame,20.,[],[])
    assert bounded == [20.,20.,110.,43.]
    source[56:61,80] = 0
    assert paragraph_edit_room(source,protected,frame,20.,[],[])[0] == frame


def test_ocr_lines_cross_blank_asset_box_only_for_continuous_prose():
    source = np.full((120,300,3),248,np.uint8)
    lines = [{'id':f't-{i}', 'ocr_line_id':f'l-{i}', 'text':text, 'bbox':bbox,
              'color_hex':'223344', 'est_size_px':16, 'bold':False}
             for i,(text,bbox) in enumerate([
                 ('请根据实际业务需求配置',[30,20,180,20]),
                 ('文件和图片。',[30,48,100,20])])]
    coarse_icon = [{'id':'icon','kind':'icon','bbox':[0,10,45,64]}]
    def groups(enabled):
        return _merge_structural_ocr_items(source,lines,coarse_icon,editable_paragraphs=enabled)[0]
    assert len(groups(False)) == 2
    merged, = groups(True)
    assert merged['natural_wrap'] and merged['ocr_line_ids'] == ['l-0','l-1']
    assert merged['text'] == '请根据实际业务需求配置\n文件和图片。'
    txt = SimpleNamespace(id='body',text=merged['text'],spans=[])
    style = SimpleNamespace(role='flowing_body_paragraph')
    evidence = SimpleNamespace(method_version=EDITABLE_TEXT_METHOD, members={'body':['l-0','l-1']})
    assert native_paragraph_text(txt,style,evidence) == '请根据实际业务需求配置文件和图片。'
    assert native_paragraph_text(txt,SimpleNamespace(role='title'),evidence) == txt.text
    legacy = SimpleNamespace(method_version='opencode-evidence/1',members=evidence.members)
    assert native_paragraph_text(txt,style,legacy) == txt.text
    assert native_paragraph_text(txt,style,SimpleNamespace(method_version=EDITABLE_TEXT_METHOD,members={})) == txt.text
    txt.spans = [SimpleNamespace(text=txt.text,style='emphasis')]
    assert native_paragraph_text(txt,style,evidence) == txt.text
    txt.spans = []
    txt.text = 'Connect to Office\nPowerPoint。'
    assert native_paragraph_text(txt,style,evidence) == txt.text
    source[44,30:100] = (0,96,160)
    assert len(groups(True)) == 2
    lines[0]['text'],lines[1]['text'] = '线上服务全年开放', '线下服务按需安排'
    assert not is_body_continuation({'align':'left'},lines)


def test_page_heading_weight_uses_real_peer_ink_and_preserves_unrelated_style():
    font_files = [Path('C:/Windows/Fonts')/name for name in ('msyh.ttc','msyhbd.ttc')]
    if not all(p.is_file() for p in font_files):
        pytest.skip('requires installed Chinese regular and bold faces')
    fonts = {'fonts':[{'file':str(p),'family':'Microsoft YaHei','bold':b,'index':0}
                      for b,p in zip((False,True),font_files)]}
    style = Style(id='shared', role='title', ea='Microsoft YaHei',latin='Microsoft YaHei',
                  size_px=30,bold=False,color='123456')
    image = Image.new('RGB',(640,160),'white')
    draw = ImageDraw.Draw(image)
    face = ImageFont.truetype(str(font_files[1]),30)
    texts, inventory = [], {'elements':[]}
    for i,(x,word) in enumerate([(24,'客户服务'),(340,'业务办理')]):
        draw.text((x,20),word,font=face,fill='#123456',anchor='lt')
        box = list(map(float,(x-2,18,125,36)))
        texts.append(Text(id=f't{i}',bbox=box,z=i,style='shared',text=word))
        texts.append(Text(id=f'b{i}',bbox=[float(x),60.,210.,22.],z=i,style='shared',text='这是一段正常的说明文字。'))
        inventory['elements'].append({'category':'section_header','bbox':box})
    graph = SimpleNamespace(text=SimpleNamespace(texts=texts,styles=[style]),selected={},
                            ownership={t.id:'RESOLVED' for t in texts})
    report, = apply_heading_weights(graph,np.array(image),fonts,inventory)
    assert report['preferred_bold'] and all(report['applied_bold'].values())
    assert set(report['members']) == {'t0','t1'}
    assert not style.bold and texts[1].style == texts[3].style == 'shared'
    draw.rectangle((337,16,475,55),fill='white')
    draw.text((340,20),'业务办理',font=ImageFont.truetype(str(font_files[0]),30),fill='#123456',anchor='lt')
    assert apply_heading_weights(graph,np.array(image),fonts,inventory) == []
