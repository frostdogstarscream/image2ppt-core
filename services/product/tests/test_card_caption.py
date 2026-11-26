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


def test_previously_verified_transparent_strokes_leave_their_holes_to_the_card(tmp_path):
    box, bounds = [10, 10, 120, 90], [44, 35, 11, 24]
    source = np.full((120, 150, 3), 249, np.uint8)
    source[10:100, 10:130] = 231
    source[36:58, 48:51] = [20, 65, 150]
    mask = np.zeros(source.shape[:2], bool)
    mask[36:58, 48:51] = True
    file, owner = tmp_path/'line.png', tmp_path/'owner.npz'
    pixels = np.zeros((24, 11, 4), np.uint8)
    pixels[1:-1, 4:7] = [20, 65, 150, 255]
    Image.fromarray(pixels).save(file)
    np.savez_compressed(owner, mask=mask, restore_bbox=bounds, backing_patch=np.full((24,11,3), 231, np.uint8))
    child = {'kind':'asset', 'id':'line', 'bbox':bounds, 'restore_bbox':bounds, 'source_sha256':'src',
             'complete_owner_scope':True,
             'owner_evidence':{'file':owner.name,'sha256':digest(owner)},
             'asset_reference':{'path':file.name,'sha256':digest(file),'source_sha256':'src','bbox':bounds}}
    verified = {'line':[digest(file),digest(owner)]}
    assert contained_peers(box,[child],layered=True,verified_assets=verified)
    for changed in ({'move_safety':'MOVE_UNSAFE'}, {'complete_owner_scope':False}, {'restore_bbox':[0,5,70,66]},
                    {'source_sha256':'other-page'}):
        assert not contained_peers(box,[{**child,**changed}],layered=True,verified_assets=verified)
    assert not contained_peers(box,[child],layered=True,verified_assets={'line':['stale',digest(owner)]})
    clean = np.full((90,120,3),231,np.uint8)
    parent = np.full_like(clean,249)
    def stage():
        return frame_layers(clean,parent,box,[child],tmp_path,[],verified_assets=verified,source_rgb=source)
    result = stage()
    assert result is not None and np.array_equal(result[3],mask[10:100,10:130])
    assert not result[-1][25:49,34:45,3].any()
    # A neighboring stroke inside a transparent hole cannot be erased.
    source[40,45] = 0
    assert stage() is None
    source[40,45] = 231
    pixels[5,1] = [8,20,112,255]
    Image.fromarray(pixels).save(file)
    with pytest.raises(ValueError,match='picture changed'):
        stage()
    child['asset_reference']['sha256'] = verified['line'][0] = digest(file)
    assert stage() is None


def test_dark_frame_edge_requires_quiet_exterior_and_continuous_face():
    from product.recovery.layered_card import _continuous_frame_edge
    source = np.full((120,160,3),252,np.uint8)
    box = [21,20,110,76]
    parent = np.full((76,110,3),252,np.uint8)
    source[20:96,21:131] = [192,220,249]
    assert _continuous_frame_edge(source,parent,box)
    source[17:30,50] = [10,60,110]
    assert not _continuous_frame_edge(source,parent,box)
    source[17:30,50] = 252
    source[20:30,50] = [192,220,249]
    source[20,77] = [8,12,16]
    assert not _continuous_frame_edge(source,parent,box)


def test_child_receipt_binds_the_page_scene_and_committed_presentation(tmp_path):
    from product.runtime.batch_backbone import committed_graphic_evidence
    ir, scene_file = tmp_path/'slideir.json', tmp_path/'scene.json'
    ir.write_text('{}'); scene_file.write_text('{}')
    page={'ir':str(ir),'scene':str(scene_file)}
    scene=SimpleNamespace(source_sha256='src',selected={'symbol':{'kind':'asset','source_sha256':'src',
        'complete_owner_scope':True,'asset_reference':{'sha256':'pixels'},'owner_evidence':{'sha256':'mask'}}})
    record={'page':2,'status':'COMMITTED','validation':{'status':'PASS'},'validated_bundle':{'irs':[digest(ir)]},
            'ir':str(ir),'files':{str(scene_file):digest(scene_file)}}
    state={'candidates':{'candidate':record}}
    assert committed_graphic_evidence(state,page,2,scene)=={'symbol':['pixels','mask']}
    assert not committed_graphic_evidence(state,page,1,scene)
    record['status']='STAGED_LOCAL'
    assert not committed_graphic_evidence(state,page,2,scene)
    record['status']='COMMITTED'
    scene_file.write_text('{"changed":true}')
    assert not committed_graphic_evidence(state,page,2,scene)


def test_separate_cards_precede_section_and_nested_detections_count_once():
    from product.recovery.layered_card import ordered_card_frames
    def mark(identifier, box):
        return {'id':identifier,'label':'card_frame','bbox':box}
    marks=[mark('section',[20,20,440,240]),mark('left',[30,55,198,180]),mark('right',[240,55,205,180]),
           mark('left-inset',[35,120,185,102]),mark('right-copy',[243,56,200,178]),
           {'id':'icon','label':'icon','bbox':[260,68,30,30]}]
    plan=ordered_card_frames(marks)
    assert plan==ordered_card_frames(list(reversed(marks)))
    assert plan[-1]['id']=='section'
    assert set(plan[-1]['required_card_frames'])=={'left','right'}
    assert all(not r['required_card_frames'] for r in plan[:-1])


def test_marked_surface_cannot_flatten_unbuilt_sibling_frames(tmp_path,monkeypatch):
    from test_badge_panel_layers import flat_card_scene
    from product.recovery.marked_surfaces import stage_marked_surfaces
    from product.recovery.layered_card import geometry_marks
    from product.recovery import container_stage
    base, *_ = flat_card_scene(tmp_path)
    rows=[{'id':'text','bbox':[110,60,70,24],'ocr_line_ids':['one']},
          {'id':'second','bbox':[110,110,90,24],'ocr_line_ids':['two']}]
    frames=geometry_marks(base/'source.png',rows)
    assert frames
    def block_early_fallback(*args,**kwargs):
        pytest.fail('an unresolved section reached the early container fallback')
    monkeypatch.setattr(container_stage,'stage_containers_and_surfaces',block_early_fallback)
    out,ids=stage_marked_surfaces(base,tmp_path/'section',
        [{**frames[0],'required_card_frames':['unbuilt-left','unbuilt-right']}],set())
    assert ids==[] and out.parent==base and not (tmp_path/'section').exists()


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


def test_clean_caption_paper_survives_where_native_text_owned_strokes(tmp_path):
    box = [10,10,120,90]
    source = np.full((120,150,3),249,np.uint8)
    patch = np.full((90,120,3),231,np.uint8)
    blue = [20,65,150]
    patch[50:70,20:100] = blue
    source[10:100,10:130] = patch
    mask = np.zeros(source.shape[:2],bool)
    mask[66:71,42:47] = True
    mask[72,44] = True
    source[mask] = 255
    backing = patch[55:64,31:39].copy()
    backing[7,3] = 255  # The unproven pixel must stay with its owner.
    owner = tmp_path/'word.npz'
    np.savez_compressed(owner,mask=mask,restore_bbox=[41,65,8,9],backing_patch=backing)
    child = {'kind':'text','bbox':[41,65,8,9],'owner_evidence':{'file':owner.name,'sha256':digest(owner)}}
    result = frame_layers(patch,np.full_like(patch,249),box,[child],tmp_path,[],
                          verified_assets={'photo':['prior-photo','owner']},source_rgb=source)
    rgba = result[-1]
    assert rgba[58,34].tolist()==[*blue,255]
    assert rgba[62,34,3]==0
