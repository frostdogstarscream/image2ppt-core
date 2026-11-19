import numpy as np
import pytest
from PIL import Image
from types import SimpleNamespace

from product.runtime.text_reading import authorize_readings, quantity_changed
from product.recovery.layered_card import contained_peers, frame_layers
from product.materialize import digest


def scene(text):
    return SimpleNamespace(
        source_sha256='source', text=SimpleNamespace(texts=[SimpleNamespace(id='t', text=text)])
    )


def test_card_keeps_verified_photo_hole_and_rejects_crossing_photo(tmp_path):
    box = [10, 10, 120, 90]
    photo = [22, 40, 36, 25]
    mask = np.zeros((130, 160), bool)
    mask[40:65, 22:58] = True
    f = tmp_path / 'owner.npz'
    np.savez_compressed(f, mask=mask)
    child = {
        'kind': 'asset',
        'bbox': photo,
        'move_safety': 'MOVABLE',
        'owner_evidence': {'file': f.name, 'sha256': digest(f)},
    }
    assert contained_peers(box, [child])
    assert not contained_peers(box, [{**child, 'bbox': [4, 40, 36, 25]}])
    assert not contained_peers(box, [{**child, 'move_safety': 'MOVE_UNSAFE'}])
    patch = np.full((90, 120, 3), 245, np.uint8)
    patch[60:75, 15:100] = [20, 60, 140]
    result = frame_layers(patch, np.full_like(patch, 248), box, [child], tmp_path, [])
    assert result is not None
    rgba = result[-1]
    assert not rgba[30:55, 12:48, 3].any()
    assert rgba[65, 20, 3] == 255


@pytest.mark.parametrize(
    'old,new',
    [
        ('26项', '26套'),
        ('30%', '30‰'),
        ('8万元', '8元'),
        ('1.5 kg', '15 kg'),
        ('二十公里', '二十公斤'),
        ('单位：万元', '单位：元'),
    ],
)
def test_conflicting_quantities_are_never_written_as_guesses(old, new):
    assert quantity_changed(old, new)


def test_layered_parent_retains_child_shape_and_text_without_opaque_artwork(tmp_path):
    box = [0, 0, 160, 110]
    surface_box = [22, 32, 105, 30]
    surface = {'id':'s-label', 'kind':'surface', 'bbox':surface_box, 'restore_bbox':surface_box,
               'payload':{'geometry':'roundRect', 'corner_radius_px':4}}
    title_box = [35,38,60,16]
    title = {'id':'t-label','kind':'text','bbox':title_box}
    paper = np.full((110,160,3),244,np.uint8)
    patch = paper.copy()
    patch[32:62,22:127] = [23,68,135]
    title_mask = np.zeros(patch.shape[:2],bool)
    title_mask[40:52,45:47] = True
    face_mask = np.zeros_like(title_mask)
    face_mask[32:62,22:127] = True
    # The verified native face already hides this unclaimed coloured speck.
    # It cannot be moved into a separate picture above the shape.
    face_mask[48,120] = False
    for record,mask in [(surface,face_mask & ~title_mask),(title,title_mask)]:
        path = tmp_path/(record['id']+'.npz')
        np.savez_compressed(path, mask=mask)
        record['owner_evidence']={'file':path.name,'sha256':digest(path)}
    assert contained_peers(box,[surface,title],layered=True)
    assert not contained_peers(box,[surface,title])
    assert not contained_peers(box,[surface,{**title,'bbox':[115,38,32,16]}],layered=True)
    assert not contained_peers(box,[{**surface,'restore_bbox':[-2,0,40,10]}],layered=True)
    result=frame_layers(patch,paper,box,[surface,title],tmp_path,[])
    assert result is not None
    assert not result[-1][32:62,22:127,3].any()


@pytest.mark.parametrize('kind', ['asset', 'surface'])
def test_native_frame_updates_child_deletion_backing_without_changing_its_owner(tmp_path, kind):
    from product.recovery.layered_card import bind_photo_backing

    old = tmp_path / 'original.npz'
    mask = np.ones((8, 9), dtype=bool)
    np.savez_compressed(old, mask=mask, restore_bbox=[10, 11, 3, 2], backing_patch=np.zeros((2, 3, 3)))
    ref = {'file': old.name, 'sha256': digest(old)}
    record = {'id': 'child', 'kind': kind, 'owner_evidence': ref, 'payload': {}}
    field = np.arange(6 * 8 * 3).reshape(6, 8, 3)
    bind_photo_backing(tmp_path, record, 's-frame', [9, 9, 8, 6], field)
    assert ref['sha256'] == digest(old) and record['owner_evidence']['prior'] == ref
    assert record['payload']['background_parent_id'] == 's-frame'
    expected_type = 'surface' if kind == 'surface' else 'photo'
    assert record['owner_evidence']['method'] == 'native-card-under-verified-'+expected_type+'/1'
    with np.load(tmp_path / record['owner_evidence']['file']) as proof:
        assert np.array_equal(proof['mask'], mask)
        assert np.array_equal(proof['backing_patch'], field[2:4, 1:4])


def test_recolor_preserves_pale_picture_details_and_blends_the_paper_tint(tmp_path):
    patch = np.full((130, 210, 3), 243, dtype=np.uint8)
    patch[16:29, 32:100] = [225, 238, 245]
    patch[50:108, 70:150] = [31, 91, 140]
    patch[50:72, 91:108] = 246
    parent = np.full_like(patch, 246)
    result = frame_layers(patch, parent, [0, 0, 210, 130], [], tmp_path, [])
    assert result is not None
    rgba = result[-1]
    assert rgba[60, 100, 3] == 255
    assert rgba[22, 50, 3] < 64


def test_longer_reading_keeps_its_origin_and_protects_next_label(monkeypatch):
    from product.runtime.text_reading import reading_frame
    obj = SimpleNamespace(id='t1', bbox=[20,30,135,36], style='s', align='left')
    style = SimpleNamespace(id='s', ea='face', bold=False, size_px=26)
    graph = SimpleNamespace(text=SimpleNamespace(texts=[obj], styles=[style]))
    fonts = {'fonts':[{'family':'face','bold':False,'file':'mock.ttc'}]}
    monkeypatch.setattr('PIL.ImageFont.truetype', lambda *a,**kw:SimpleNamespace(getlength=lambda s:26*len(s)))
    bbox = reading_frame(obj,'京南会议室','京南综合会议',graph,fonts,1100)
    assert bbox == [20,30,162,36] and obj.bbox == [20,30,135,36]
    graph.text.texts.append(SimpleNamespace(id='t2',bbox=[165,35,50,25]))
    assert reading_frame(obj,'京南会议室','京南综合会议',graph,fonts,1100) is None


def test_detected_caption_edge_stays_on_slide_when_photo_is_removed():
    from product.recovery.region_crop import _separate_photo_caption

    src = np.full((120, 170, 3), 180, np.uint8)
    src[72:95, 10:131] = [43, 85, 160]
    crop = {'box': [5, 10, 131, 70], 'seed_boxes': [[10, 10, 120, 84]], 'kinds': ['layout:image']}
    word = SimpleNamespace(bbox=[22, 80, 66, 12])
    assert _separate_photo_caption(src, crop, [word])['box'] == [5, 10, 131, 62]
    src[72:95, 26:131] = 180
    assert _separate_photo_caption(src, crop, [word]) == crop


def test_tentative_reading_requires_two_crops_and_a_ranked_choice():
    data = {
        'object_id': 't',
        'before': '资料整里',
        'after': '资料整理',
        'status': 'pending_confirmation',
        'alternatives': ['资料整里', '资料整理'],
        'context_sha256': 'b',
        'crop_sha256': 'a',
        'choice_receipt': 'choice',
    }
    authorize_readings(
        scene(data['before']), scene(data['after']), {'source_sha256': 'source', 'changes': [data]}
    )
    with pytest.raises(ValueError):
        authorize_readings(
            scene(data['before']),
            scene(data['after']),
            {'source_sha256': 'source', 'changes': [{**data, 'choice_receipt': None}]},
        )
