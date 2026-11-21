import cv2
import numpy as np

from product.recovery.circular_badge import circular_badge
from product.recovery.marked_fields import footer_fields


def test_badge_hole_and_surrounding_pixels():
    source = np.full((120, 120, 3), 250, np.uint8)
    cv2.circle(source, (60, 60), 27, (230, 236, 242), -1)
    plate = source.copy()
    cv2.circle(source, (60, 60), 9, (40, 72, 170), 3)
    bbox = [43, 43, 34, 34]
    result = circular_badge(source, plate, bbox)
    assert result is not None
    assert result[2][17, 17, 3] == 0
    assert result[2][17, 26, 3] > 240
    plate[40, 60] = 0
    assert circular_badge(source, plate, bbox) is None


def test_footer_split_requires_source_gutters_and_preserves_wording():
    source = np.full((200, 200, 3), 255, np.uint8)
    for x in (12, 22, 32, 80, 90):
        source[181:191, x:x+8] = 50
    fields = footer_fields(source, 'ABCDE', [10, 180, 90, 12])
    assert [p[0] for p in fields] == ['ABC', 'DE']
    assert footer_fields(source, 'ABCDE', [10, 20, 90, 12]) == []


def test_layer_choice_checks_background_in_transparent_hole():
    from product.recovery.circular_badge import select_badge_ink

    measured = np.zeros((10,10,4),np.uint8)
    measured[2:8,4:6] = [30,60,190,255]
    clean = measured.copy()
    clean[2:8,4:6,0] = 35
    source = np.full((10,10,3),240,np.uint8)
    source[2:8,4:6] = clean[2:8,4:6,:3]
    assert select_badge_ink(source,'f0f0f0',measured,clean)[1]['selected']=='LayerD'
    dirty = clean.copy()
    dirty[:,:,:3][clean[:,:,3]==0] = 240
    dirty[:,:,3] = 255
    assert select_badge_ink(source,'f0f0f0',measured,dirty)[1]['selected']=='color-unmix'


def test_compiler_checks_all_split_fields_from_source(tmp_path):
    import json
    from types import SimpleNamespace
    from PIL import Image
    from product.contracts import Style, Text
    from product.parallel_pipeline import effective_texts

    source = np.full((200, 200, 3), 255, np.uint8)
    for x in (12, 22, 32, 80, 90):
        source[181:191, x:x+8] = 50
    Image.fromarray(source).save(tmp_path/'source.png')
    style = Style(id='s-01',role='body',ea='Arial',latin='Arial',size_px=12,bold=False,color='323232')
    text = Text(id='t-001',bbox=[10,180,90,12],z=1,style=style.id,text='ABCDE')
    graph = SimpleNamespace(text=SimpleNamespace(texts=[text],styles=[style]),selected={text.id:{}})
    path=tmp_path/'slideir.json'
    path.write_text(json.dumps({'nodes':[{'type':'text','id':'t-001','text':'incorrect'},
                                          {'type':'text','id':'t-001-field-2','text':'incorrect'}]}),'utf8')
    expected=effective_texts(graph,path)
    assert [field.text for field in expected.texts]==['ABC','DE']
