from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageFont

from product.contracts import Style, Text
from product.materialize import digest
from product.recovery import marked_fields
from product.recovery.candidate_stage import NativeFrameConflict, _fit_measured_line, native_frame
from product.recovery.field_ink import colour_runs, separate_edge_graphic
from product.recovery.region_text import measure_group, plan_regions
from product.runtime.host_baseline import _reopen_residual_fields, _text_in_retained_artwork


def glyph_row():
    source = np.full((60, 145, 3), 245, 'uint8')
    for i in range(4):
        x = 43 + 17*i
        source[20:33, x:x+9] = (40,45,50) if i < 2 else (12,75,195)
        source[22:28, x+3:x+6] = 245
    source[18:33, 15:29] = [12,75,195]
    return source


@pytest.mark.parametrize('right', [False, True])
def test_detach_only_when_every_source_character_survives(right):
    source = glyph_row()
    if right:
        source = source[:, ::-1].copy()
    text = '甲乙丙丁'
    line = {'text':text, 'bbox':[145-105 if right else 12,14,93,23]}
    detection = {'id':'icon', 'bbox':[145-31 if right else 13,16,18,20]}
    result = separate_edge_graphic(source, line, [detection])
    assert result['graphic']['id'] == 'icon'
    x, y, w, h = result['bbox']
    assert w < 75 and (x+w <= 145-42 if right else x >= 40)
    assert separate_edge_graphic(source, {**line,'text':text+'戊'}, [detection]) is None
    assert separate_edge_graphic(source, line, [{**detection,'bbox':[40,17,21,18]}]) is None


def test_native_runs_erase_both_paints_and_preserve_neighbor(tmp_path):
    source = glyph_row()
    box = [40,15,66,22]
    runs = colour_runs(source, '甲乙丙丁', box)
    assert [(v,c) for v,_,_,c in runs] == [('甲乙','282D32'),('丙丁','0C4BC3')]
    t = Text(id='t', text='甲乙丙丁', style='blue', z=1, bbox=box,
             spans=[{'text':'甲乙','style':'dark'},{'text':'丙丁','style':'blue'}])
    styles = [SimpleNamespace(id=name,color=color,size_px=16) for name,color in [('blue','0C4BC3'),('dark','282D32')]]
    graph = SimpleNamespace(image_size=[145,60], source_sha256='fixture', selected={}, ownership={'t':'RESOLVED'},
                            asset=SimpleNamespace(objects=[]), text=SimpleNamespace(texts=[t],styles=styles,
                            members={'t':['line']}))
    group, = plan_regions(graph, [t], {}, source)
    group['source_field_ink'] = True
    _, clean, _ = measure_group(graph, group, source, tmp_path, [])
    assert np.max(abs(clean[20:33,43:103].astype(int)-245)) <= 3
    np.testing.assert_array_equal(clean[18:33,15:29],source[18:33,15:29])
    t.spans[0].style = 'blue'
    with pytest.raises(ValueError, match='corroborate'):
        measure_group(graph, group, source, tmp_path, [])


def test_field_split_survives_uncertain_colour_only_with_geometry(monkeypatch):
    source = np.full((45, 150, 3), 245, 'uint8')
    for x in (5,20,35,77,92,107):
        source[10:28,x:x+11] = 15
    source[9:29,62:64] = 30
    item = {'id':'title','text':'项目名丨行业类','bbox':[3,7,122,24],'ocr_line_ids':['line'],'color_hex':'164381'}
    monkeypatch.setattr(marked_fields, 'measure_text_appearance', lambda *args: None)
    assert marked_fields.split_items(source,[item],[]) == [item]
    fields = marked_fields.split_items(source,[item], [], inherit_colour=True)
    assert [(f['text'],f['color_hex']) for f in fields] == [('项目名丨','164381'),('行业类','164381')]
    source[9:29,62:64] = 245
    assert marked_fields.split_items(source,[item],[],inherit_colour=True) == [item]


def test_loose_ocr_margin_can_fit_without_covering_unit_or_shrinking_below_measurement():
    for family in ('arial.ttf','DejaVuSans.ttf'):
        try:
            font = ImageFont.truetype(family, 22)
            break
        except OSError:
            continue
    else:
        pytest.skip('requires a real TrueType font')
    style = Style(id='base',role='value',ea='test',latin='test',size_px=22,bold=False,color='1266BB')
    t = Text(id='value',text='9.8',style='base',z=1,bbox=[15,9,66,31])
    fonts = {'fonts':[{'family':'test','bold':False,'file':font.path,'index':0}]}
    source_size = (112,60)
    mask = np.zeros((60,112),bool)
    l, top, r, bot = font.getbbox(t.text)
    mask[9+top:9+bot,15+l:15+r] = np.asarray(font.getmask(t.text)).reshape(bot-top,r-l)>0
    guard = np.zeros_like(mask)
    guard[:,70:90] = True
    with pytest.raises(NativeFrameConflict):
        native_frame(t,{'base':style},fonts,source_size,guard,t.bbox,fit_origin=True)
    fitted, payload, _ = _fit_measured_line(t,{'base':style},fonts,source_size,guard,mask,t.bbox,{'fit_origin':True})
    assert fitted.text == '9.8' and payload['bbox'][0]+payload['bbox'][2] <= 70
    with pytest.raises(NativeFrameConflict):
        _fit_measured_line(t,{'base':style},fonts,source_size,guard|mask,mask,t.bbox,{'fit_origin':True})
    # A real Office overflow on the tested face can resize its own field once.
    edge = int(payload['bbox'][0] + payload['bbox'][2]) + 1
    narrow = np.zeros_like(mask)
    narrow[:,edge:] = True
    minimum = [round(payload['bbox'][2]*1.17),int(payload['bbox'][3])]
    options = {'fit_origin':True,'minimum_size':minimum}
    for foreign_style in (None,{**style.model_dump(),'bold':True}):
        with pytest.raises(NativeFrameConflict):
            _fit_measured_line(t,{'base':style},fonts,source_size,narrow,mask,t.bbox,options,
                               office_style=foreign_style)
    saved_mask = mask.copy()
    adjusted, frame, definition = _fit_measured_line(
        t,{'base':style},fonts,source_size,narrow,mask,t.bbox,options,office_style=style.model_dump())
    assert adjusted.text == t.text and .72*style.size_px <= definition.size_px < style.size_px
    assert frame['bbox'][0]+frame['bbox'][2] <= edge
    np.testing.assert_array_equal(saved_mask, mask)
    with pytest.raises(NativeFrameConflict):
        _fit_measured_line(t,{'base':style},fonts,source_size,narrow,mask,t.bbox,
                           {**options,'minimum_size':[2*minimum[0],minimum[1]]},office_style=style.model_dump())


def test_reopen_restores_only_an_independently_owned_text_frame(monkeypatch,tmp_path):
    from product.recovery import region_state
    original = Text(id='a',text='标题',style='s',z=1,bbox=[4,8,32,15])
    grown = original.model_copy(update={'bbox':[4,8,52,22]})
    record = {'kind':'text','source_owners':['source-plate'],'source_text_definition':original.model_dump()}
    graph = SimpleNamespace(selected={'a':record,'b':{**record,'source_owners':['picture']},'c':record},
                            text=SimpleNamespace(texts=[grown]))
    row = {'ir':str(tmp_path/'slideir.json')}
    service = SimpleNamespace(graph=lambda _:graph, page=lambda *args:row)
    trial = {'composition':{'index':2},'visual_review':{'index':2}}
    monkeypatch.setattr(region_state,'closure',lambda graph, ids: ids|({'a'} if 'c' in ids else set()))
    def rollback(graph, baseline, ids, output, reason):
        assert ids == {'a'} and graph.text.texts[0].bbox == original.bbox
        return output/'slideir.json'
    monkeypatch.setattr(region_state,'rollback',rollback)
    opened, skipped = _reopen_residual_fields(service, trial, 1, {'a','b','c'}, tmp_path/'new')
    assert opened == ['a'] and set(skipped) == {'b','c'}
    assert Path(row['scene']).parent == tmp_path/'new' and 'composition' not in trial


def test_retained_artwork_must_own_visible_text_not_just_enclose_it(tmp_path):
    source = glyph_row()
    img = tmp_path/'source.png'
    Image.fromarray(source).save(img)
    art = tmp_path/'art.png'
    proof = tmp_path/'art.npz'
    box = [40,15,66,22]
    rows = [{'ocr_id':'line','bbox':box,'raster_reading':'甲乙丙丁','plate_reading':None,'status':'retained'}]
    glyphs = source.min(2)<120
    for supported in (True,False):
        rgba = np.dstack([source,glyphs.astype('uint8')*255 if supported else np.zeros_like(glyphs,'uint8')])
        Image.fromarray(rgba).save(art)
        np.savez_compressed(proof,mask=glyphs)
        record = {'bbox':[0,0,145,60],'retained_card_artwork':True,
                  'asset_reference':{'path':art.name,'sha256':digest(art),'source_sha256':digest(img)},
                  'owner_evidence':{'file':proof.name,'sha256':digest(proof)}}
        assert _text_in_retained_artwork(record, rows, tmp_path, digest(img)) == (['line'] if supported else [])
        assert _text_in_retained_artwork(record, [{**rows[0],'plate_reading':'甲乙丙丁'}], tmp_path, digest(img)) == []
