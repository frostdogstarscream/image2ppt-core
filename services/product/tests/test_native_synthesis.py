import json

import numpy as np
from PIL import Image

from product.materialize import digest
from product.parallel_rebuild import adapt
from product.runtime.native_synthesis import component_outlines, fit_editing_space, observed_box, text_backplate


def test_component_geometry_uses_visible_outline_and_distinguishes_coordinate_forms():
    image=np.full((140,240,3),255,dtype=np.uint8)
    image[40:110,70:190]=[218,228,244]
    boxes=component_outlines(image)
    for identifier,box in [('e01',[70,40,190,110]),('inv-1',[70,40,120,70])]:
        assert observed_box({'id':identifier,'bbox':box},240,140,boxes)==[70,40,120,70]
    image[30:55,20:120]=[2,102,220]
    assert text_backplate(image,[39,35,45,14])==[20,30,100,25]


def test_synthesis_adapter_omits_full_page_picture_only_for_explicit_variant(tmp_path):
    from types import SimpleNamespace
    src=tmp_path/'source.png'
    Image.new('RGB',(90,55),'white').save(src)
    sha=digest(src)
    graph=SimpleNamespace(source_sha256=sha,selected={},image_size=[90,55],
        surface=SimpleNamespace(background='FFFFFF',shapes=[]),text=SimpleNamespace(styles=[]),
        raster_backing={'path':'source.png','source_sha256':sha,'sha256':sha,'bbox':[0,0,90,55]})
    assert [n['id'] for n in adapt(graph,tmp_path)['nodes']]==['source-plate']
    graph.raster_backing={'method':'native_synthesis','source_sha256':sha}
    blank=adapt(graph,tmp_path)
    assert blank['nodes']==blank['assets']==[]


def test_wrapping_uses_free_column_before_neighbor():
    ir={'canvas':{'source_px':[250,150]},'nodes':[
        {'id':'a','type':'text','bbox':[10,10,70,20],'text':'可以修改的完整说明段落'},
        {'id':'b','type':'text','bbox':[90,10,35,15],'text':'邻居'},
        {'id':'c','type':'text','bbox':[10,45,70,20],'text':'下行'}]}
    fit_editing_space(ir,[[5,5,210,110]])
    x,y,w,h=ir['nodes'][0]['bbox']
    assert ir['nodes'][0]['wrap'] and x+w<=90 and y+h<=45

def test_repeated_carriers_need_distinct_sibling_content_and_photo_evidence():
    from product.runtime.native_synthesis import repeated_panels
    boxes = [[x, 50, 100, 195] for x in (5, 114, 223)]
    texts = [[x+5, y, 80, 12] for x, _, _, _ in boxes for y in (140, 165)]
    assert repeated_panels(boxes, texts, [], (340, 300)) == []
    assert repeated_panels(boxes, texts, [[10, 55, 90, 70]], (340, 300)) == boxes


def test_measurement_preserves_both_halves_of_clipped_icon():
    import cv2
    from product.runtime.native_synthesis import measured_icon_box
    rgb = np.full((100, 150, 3), 250, dtype='uint8')
    cv2.circle(rgb, (72, 42), 15, (6, 110, 230), 5)
    box = measured_icon_box(rgb, [66, 38, 32, 33])
    x, y, w, h = box
    assert x <= 55 and y <= 25 and x+w >= 89 and y+h >= 60
