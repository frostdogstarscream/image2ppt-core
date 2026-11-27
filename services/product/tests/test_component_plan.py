import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from product.contracts import Style,Text
from product.runtime.component_plan import PagePlan,TextRule,apply_components,matte_symbol,style_text


def graph():
    style=Style(id='base',role='body',ea='Arial',latin='Arial',size_px=14.,bold=False,color='323232')
    text=Text(id='t0',bbox=[15.,10.,160.,22.],z=3,style=style.id,text='处理文本并核验')
    return SimpleNamespace(image_size=[240,128],source_sha256='0'*64,
        text=SimpleNamespace(styles=[style],texts=[text]),surface=SimpleNamespace(shapes=[],background='FFFFFF'),
        asset=SimpleNamespace(objects=[]),ownership={'t0':'RESOLVED'},
        selected={'t0':{'id':'t0','kind':'text','bbox':text.bbox,'payload':text.model_dump()}})


def test_exact_emphasis_only_changes_its_own_range():
    g=graph()
    result=style_text(g,[TextRule(target_ids=['t0'],normal_color='101010',keep_source_width=True,
        bold=False,emphasis=[dict(source='核验',color='1166BB',bold=True)])])
    t=g.text.texts[0]
    styles={s.id:s for s in g.text.styles}
    assert [(s.text,styles[s.style].bold,styles[s.style].color) for s in t.spans]==[
        ('处理文本并',False,'101010'),('核验',True,'1166BB')]
    assert g.selected['t0']['source_width'] and result[-1]['status']=='NATIVE_TEXT'


def test_actual_path_retains_order_and_does_not_authorize_source_erasure(tmp_path):
    g=graph()
    rgb=np.full((128,240,3),255,np.uint8)
    before=rgb.copy()
    p=PagePlan.model_validate({'components':[dict(role='connector',geometry='path',label='反馈',
        points=[[210,110],[5,110],[5,5]],stroke='446699',arrow_at_end=True)],
        'text_rules':[]})
    report=apply_components(g,rgb,p,tmp_path)
    shape=g.surface.shapes[0]
    assert report[0]['object_ids']==[shape.id] and shape.tail_end=='triangle'
    assert [(p.x,p.y) for p in shape.custom_paths[0].commands]==[(205.,105.),(0.,105.),(0.,0.)]
    assert np.array_equal(rgb,before)


def test_continuous_emphasis_crosses_two_ocr_fields_without_coloring_neighbors():
    g=graph()
    first=g.text.texts[0]
    first.text='形成“知识采集'
    second=first.model_copy(update={'id':'t1','text':'更新与检索”，辅助查询'})
    g.text.texts.append(second)
    g.selected['t1']={'kind':'text','payload':second.model_dump()}
    rule=TextRule(target_ids=['t0','t1'],normal_color='101010',bold=False,
                  emphasis=[dict(source='知识采集更新与检索',color='1166BB',bold=True)])
    results=style_text(g,[rule])
    styles={s.id:s for s in g.text.styles}
    assert all(r['status']=='NATIVE_TEXT' for r in results)
    assert [span.text for t in g.text.texts for span in t.spans if styles[span.style].bold]==['知识采集','更新与检索']
    assert ''.join(s.text for s in second.spans)==second.text


def test_soft_icon_edge_is_not_an_opaque_rectangle(tmp_path):
    path=tmp_path/'symbol.png'
    pixel=np.full((40,40,3),[222,240,251],dtype=np.uint8)
    pixel[10:30,12:28]=[27,106,226]
    Image.fromarray(pixel).save(path)
    assert matte_symbol(path)=='unmixed-on-measured-paper'
    alpha=np.array(Image.open(path))[:,:,3]
    assert not alpha[0].any() and alpha[15,15]>160


def test_confirmed_list_dot_moves_with_its_ocr_line(tmp_path):
    g=graph()
    g.text.members={'t0':['line0']}
    rgb=np.full((128,240,3),255,np.uint8)
    rgb[18:22,9:13]=[18,80,200]
    Image.fromarray(rgb).save(tmp_path/'source.png')
    (tmp_path/'runtime-ocr-protection.json').write_text(json.dumps(
        {'lines':[{'id':'line0','bbox':[15,10,160,22]}]}),'utf8')
    plan=PagePlan.model_validate({'components':[dict(role='pedestal',geometry='ellipse',label='list dot',
        bounds=[8,18,5,5],fill='1260D1')],'text_rules':[]})
    apply_components(g,rgb,plan,tmp_path)
    apply_components(g,rgb,plan,tmp_path)
    assert g.text.texts[0].text=='·处理文本并核验'
    assert not g.surface.shapes
