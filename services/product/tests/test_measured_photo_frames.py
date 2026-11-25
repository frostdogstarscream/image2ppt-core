"""Single-photo recovery from bounded, independently measured edges."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.contracts import Style, Text
from product.parallel_contracts import SceneGraph
from product.recovery.region_crop import stage_region_crops
from product.runtime.marked_routes import iou, measured_picture_frame
from test_region_crop import scene


def test_nearby_picture_and_caption_are_not_part_of_selected_photo():
    image = np.full((160, 300, 3), 247, np.uint8)
    image[22:112, 15:137] = [174,192,231]
    image[22:112, 143:265] = [48,64,108]  # A stronger edge just across the gutter.
    image[118:123, 20:125] = 45
    hint, peer = [17,23,117,82], [145,23,119,82]
    frame, method = measured_picture_frame(image, hint, [peer])
    assert method == 'FOUR_MEASURED_EDGES' and iou(frame, [15,22,122,90]) > .94
    assert frame[0]+frame[2] < 143 and frame[1]+frame[3] < 118
    assert measured_picture_frame(image, [12,18,256,100], [hint,peer])[1] == 'MARK_CONTAINS_ANOTHER_PICTURE'
    assert measured_picture_frame(image, [0,0,300,160], [])[1] == 'PICTURE_BOUNDARY_NOT_OBJECT_SCOPED'
    assert measured_picture_frame(np.full_like(image,247), hint, [peer])[0] is None


def test_measured_crop_keeps_its_frame_and_neighbour(tmp_path):
    src, baseline = scene(tmp_path)
    box = [20,20,40,40]
    seed = {'id':'photo-1','bbox':box,'kind':'layout:image','measured_frame':'FOUR_MEASURED_EDGES'}
    ir, ids, _ = stage_region_crops(baseline, tmp_path/'crop', seeds=[seed], single_object_only=True)
    assert len(ids)==1
    record = json.loads((ir.parent/'scene.json').read_bytes())['selected'][ids[0]]
    assert record['bbox']==box and record['move_safety']=='MOVABLE'
    plate = np.asarray(Image.open(ir.parent/'source-plate.png').convert('RGB'))
    assert np.array_equal(plate[70:100,150:200],src[70:100,150:200])
    assert (plate[20:60,20:60].min(2)>245).all()


def test_photo_does_not_swallow_unresolved_business_text(tmp_path):
    _, baseline = scene(tmp_path)
    manifest = baseline/'scene.json'
    graph = SceneGraph.model_validate_json(manifest.read_bytes())
    graph.text.styles.append(Style(id='s',role='body',ea='Arial',latin='Arial',size_px=11,bold=False,color='223344'))
    graph.text.texts.append(Text(id='t-label', text='2026', bbox=[27,31,23,11], style='s', z=1))
    graph.ownership['t-label'] = 'AMBIGUOUS'
    manifest.write_text(graph.model_dump_json(by_alias=True), encoding='utf8')
    seed = {'id':'photo-1','bbox':[20,20,40,40],'kind':'layout:image','measured_frame':'FOUR_MEASURED_EDGES'}
    ir, ids, report = stage_region_crops(baseline, tmp_path/'crop', seeds=[seed], single_object_only=True)
    assert ir.parent==baseline and not ids
    assert report[0]['reason']=='UNRESOLVED_TEXT_IN_PICTURE' and report[0]['text_ids']==['t-label']
