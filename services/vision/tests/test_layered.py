"""Layered v3 contracts on synthetic fixtures. They establish tool-chain behaviour only,
not real-image accuracy or human acceptance."""
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from image2ppt_vision.layered import components, emit_ir, inpaint, qa, segment, semantics, tables, textmask
from image2ppt_vision.layered.perceive import bullet_icon, perceive, plausible_text

ROOT = Path(__file__).resolve().parents[3]


def page(w=640, h=360, color=(246, 248, 252)):
    return np.full((h, w, 3), color, np.uint8)


def write(rgb, text, org, scale=1.0, color=(30, 40, 60), thickness=2):
    cv2.putText(rgb, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def test_ink_mask_keeps_adjacent_tile_icon_out_of_text():
    rgb = page()
    cv2.rectangle(rgb, (40, 40), (66, 66), (20, 110, 230), -1)
    write(rgb, 'REPORT', (80, 64))
    line = {'id': 't0', 'text': 'REPORT', 'bbox': [38, 38, 200, 30], 'score': .99}
    records, text_mask, icon_mask = textmask.analyze_page(rgb, [line])
    record = records[0]
    assert record['status'] == 'ok' and record['icon_candidates']
    assert icon_mask[50:58, 48:58].all() and not text_mask[50:58, 48:58].any()
    assert text_mask[45:65, 80:200].sum() > 200
    assert record['ink_bbox'][0] >= 76


def test_badge_glyphs_are_measured_inside_the_badge():
    rgb = page(color=(20, 110, 230))
    cv2.circle(rgb, (60, 60), 14, (255, 255, 255), -1, cv2.LINE_AA)
    write(rgb, '02', (50, 66), .45, (20, 110, 230), 1)
    line = {'id': 't0', 'text': '02', 'bbox': [48, 50, 24, 21], 'score': .99}
    records, text_mask, _ = textmask.analyze_page(rgb, [line])
    assert records[0]['badge']
    ring = np.zeros(text_mask.shape, np.uint8)
    cv2.circle(ring, (60, 60), 14, 1, 2)
    assert not (text_mask & (ring > 0)).any()
    assert text_mask.sum() > 30


def test_split_only_strips_icon_like_characters():
    split = textmask.split_icon_text('8>REPORT', [80, 40, 120, 24], [[40, 40, 30, 24]])
    assert split is None or split['lead'] in ('8', '8>')
    assert textmask.split_icon_text('国有企业', [80, 40, 72, 24], [[40, 40, 24, 24]]) is None


def test_plausible_text_drops_lone_low_confidence_cjk():
    box = [0, 0, 20, 20]
    assert not plausible_text({'text': '品', 'bbox': box, 'score': .78})
    assert plausible_text({'text': '品', 'bbox': box, 'score': .97})
    assert plausible_text({'text': '1', 'bbox': box, 'score': .7})


def test_inpaint_changes_only_masked_pixels_and_detects_solid():
    rgb = page()
    write(rgb, 'TEXT', (100, 100))
    mask = np.zeros(rgb.shape[:2], bool); mask[70:110, 90:200] = True
    out, method = inpaint.inpaint(rgb, mask, None)
    assert method == 'opencv_telea'
    assert (out[~mask] == rgb[~mask]).all()
    assert inpaint.solid_color(out) == 'F6F8FC'
    assert inpaint.solid_color(rgb) is None


def test_calm_holes_are_filled_smoothly_and_never_by_the_model(monkeypatch):
    rgb = page()
    rgb[:, 320:] = (200, 220, 240)
    rgb[200:260, 400:460] = np.random.default_rng(1).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    mask = np.zeros(rgb.shape[:2], bool)
    mask[60:160, 60:260] = True
    mask[215:245, 415:445] = True
    monkeypatch.setattr(inpaint, 'lama', lambda image, hole, model: np.zeros_like(image))
    out, method = inpaint.inpaint(rgb, mask, model=object())
    assert method == 'lama'
    assert np.abs(out[60:160, 60:260].astype(int)-(246, 248, 252)).max() <= 2
    assert (out[220:240, 420:440] == 0).all()
    assert (out[~mask] == rgb[~mask]).all()


def test_segment_classifies_flat_card_gradient_and_icon():
    rgb = page()
    card = np.zeros(rgb.shape[:2], np.uint8)
    cv2.rectangle(card, (40, 40), (300, 200), 1, -1)
    rgb[card > 0] = (255, 255, 255)
    xs = np.linspace(0, 1, 200)
    band = (np.array([30, 100, 220])[None]+xs[:, None]*np.array([0, 100, 0])[None]).astype(np.uint8)
    rgb[250:300, 360:560] = band[None]
    icon = np.zeros_like(card)
    cv2.circle(icon, (100, 100), 14, 1, -1)
    rgb[icon > 0] = (220, 60, 40)
    rgb[96:104, 92:108] = (255, 255, 255)
    grad = np.zeros_like(card); grad[250:300, 360:560] = 1
    masks = [(card > 0, {'source': 'sam2'}), (grad > 0, {'source': 'sam2'}), (icon > 0, {'source': 'sam2'})]
    empty = np.zeros(card.shape, bool)
    elements, erase = segment.classify(rgb, masks, text_mask=empty, icon_mask=empty)
    kinds = sorted((e['kind'], e.get('geometry')) for e in elements)
    assert ('native_shape', 'rect') in kinds
    assert ('raster', None) in kinds
    gradient = next(e for e in elements if e['bbox'][0] == 360)
    assert gradient['kind'] == 'native_shape' and gradient['fill_gradient']['stops'][0]['offset'] == 0
    assert abs(gradient['fill_gradient']['angle_deg']) < 5 or abs(gradient['fill_gradient']['angle_deg']-360) < 5
    assert erase[card > 0].all()
    raster = next(e for e in elements if e['kind'] == 'raster')
    rgba = segment.cutout(rgb, raster)
    assert rgba.shape[2] == 4
    assert set(np.unique(rgba[:, :, 3])).issubset({0, 255})


def _panel_page():
    """A textured photo panel split into three SAM pieces, a detector icon on it, and a flat card."""
    rgb = page()
    rng = np.random.default_rng(7)
    rgb[60:240, 100:400] = rng.integers(0, 255, (180, 300, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
    panel = (yy >= 60) & (yy < 240) & (xx >= 100) & (xx < 400)
    left = xx < 258+12*np.sin(yy/9.)
    upper = yy < 150+10*np.sin(xx/11.)
    pieces = [(panel & left, {'source': 'sam2'}), (panel & ~left & upper, {'source': 'sam2'}),
              (panel & ~left & ~upper, {'source': 'sam2'})]
    cv2.rectangle(rgb, (450, 260), (620, 340), (255, 255, 255), -1)
    card = np.zeros(rgb.shape[:2], bool); card[260:341, 450:621] = True
    icon = np.zeros(rgb.shape[:2], np.uint8)
    cv2.circle(icon, (180, 150), 12, 1, -1)
    rgb[icon > 0] = (220, 30, 30)
    detector = {'id': '', 'kind': 'raster', 'source': 'textmask_icon', 'alpha': 'mask', 'absorbed': 0,
                'mask': icon > 0, 'bbox': segment._bbox(icon > 0), 'area': int(icon.sum())}
    return rgb, pieces+[(card, {'source': 'sam2'})], detector


def test_textured_pieces_become_one_object_and_erase_equals_alpha():
    rgb, masks, detector = _panel_page()
    detected = detector['area']
    empty = np.zeros(rgb.shape[:2], bool)
    elements, _ = segment.classify(rgb, masks, text_mask=empty, icon_mask=empty)
    H, W = rgb.shape[:2]
    elements, erase = segment.blockify(elements+[detector], W, H, grad=segment.gradient(rgb))
    blocks = [e for e in elements if e.get('role') == 'block']
    assert len(blocks) == 1 and blocks[0]['members'] == 3
    x, y, w, h = blocks[0]['bbox']
    assert abs(x-100) <= 2 and abs(y-60) <= 2 and abs(x+w-400) <= 2 and abs(y+h-240) <= 2
    rgba = segment.cutout(rgb, blocks[0])
    assert int(rgba[:, :, 3].min()) == 255
    assert any(e['kind'] == 'native_shape' and e['bbox'][0] == 450 for e in elements)
    alpha = np.zeros(erase.shape, bool)
    for e in elements:
        alpha |= e['mask']
    rim = cv2.dilate(alpha.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert (erase == rim).all()
    icon = next(e for e in elements if e.get('role') == 'icon')
    assert icon['level'] == 1 and blocks[0]['level'] == 0
    assert icon['mask'].sum() > detected


def test_shaded_card_stays_pixels_and_flat_card_stays_a_shape():
    """A flat centre with a soft shadow rim is not repainted as one flat fill."""
    rgb = page()
    shaded = np.zeros(rgb.shape[:2], np.uint8)
    cv2.rectangle(shaded, (40, 40), (280, 220), 1, -1)
    dist = cv2.distanceTransform(shaded, cv2.DIST_L2, 5)
    shade = np.clip(1-dist/14, 0, 1)
    tone = (255-40*shade)[..., None]*np.ones(3)
    rgb[shaded > 0] = tone[shaded > 0].astype(np.uint8)
    flat = np.zeros(rgb.shape[:2], np.uint8)
    cv2.rectangle(flat, (340, 60), (600, 220), 1, -1)
    rgb[flat > 0] = (255, 255, 255)
    empty = np.zeros(shaded.shape, bool)
    elements, _ = segment.classify(rgb, [(shaded > 0, {'source': 'sam2'}), (flat > 0, {'source': 'sam2'})],
                                   text_mask=empty, icon_mask=empty)
    H, W = rgb.shape[:2]
    elements, erase = segment.blockify(elements, W, H, grad=segment.gradient(rgb))
    assert not any(e['kind'] == 'native_shape' and e['bbox'][0] == 40 for e in elements)
    assert any(e['kind'] == 'native_shape' and e['bbox'][0] == 340 for e in elements)
    assert any(e['kind'] == 'raster' and e['bbox'][0] == 40 for e in elements)
    assert erase[60:200, 60:260].all()


def test_icon_on_a_card_is_lifted_with_its_pale_badge():
    rgb = page()
    card = np.zeros(rgb.shape[:2], np.uint8)
    cv2.rectangle(card, (40, 40), (300, 200), 1, -1)
    rgb[card > 0] = (255, 255, 255)
    cv2.circle(rgb, (100, 100), 16, (225, 236, 252), -1)
    cv2.rectangle(rgb, (90, 90), (110, 110), (20, 90, 220), -1)
    empty = np.zeros(card.shape, bool)
    elements, _ = segment.classify(rgb, [(card > 0, {'source': 'sam2'})], text_mask=empty, icon_mask=empty)
    icons = [e for e in elements if e.get('source') == 'shape_residual']
    assert len(icons) == 1
    x, y, w, h = icons[0]['bbox']
    assert w >= 30 and h >= 30 and x <= 85 and y <= 85


def _piece(x, y, w, h, H, W, mask=None):
    m = np.zeros((H, W), bool)
    if mask is None:
        m[y:y+h, x:x+w] = True
    else:
        m[y:y+h, x:x+w] = mask
    return {'kind': 'raster', 'source': 'sam2', 'bbox': [x, y, w, h], 'area': int(m.sum()), 'mask': m}


def test_seams_split_objects_and_join_one_colour_block():
    """Pieces join only across a calm seam: a band does not pull in the cards under it,
    two chevrons with an edge stay two, and one surface cut in two becomes one."""
    H, W = 400, 800
    rgb = page(W, H)
    piece = lambda *box, mask=None: _piece(*box, H, W, mask)
    band = piece(20, 20, 760, 40)
    cards = [piece(20+190*k, 62, 170, 120) for k in range(4)]
    halves = [piece(40, 220, 100, 120), piece(140, 220, 100, 120)]
    a, b = segment._chevron(120, 40, 14), segment._chevron(120, 40, 14)
    chevrons = [piece(300, 220, 120, 40, mask=a), piece(406, 220, 120, 40, mask=b)]
    sparse = piece(560, 220, 200, 120)
    sparse['mask'][:] = False; sparse['mask'][230:330:4, 570:750] = True; sparse['area'] = int(sparse['mask'].sum())
    inner = piece(590, 240, 120, 80)
    rgb[band['mask']] = (200, 220, 240)
    for card in cards:
        rgb[card['mask']] = (255, 255, 255)
    for half in halves:
        rgb[half['mask']] = (230, 180, 120)
    rgb[chevrons[0]['mask']] = (30, 90, 200)
    rgb[chevrons[1]['mask'] & ~chevrons[0]['mask']] = (60, 140, 230)
    rgb[inner['mask']] = (100, 160, 90)
    groups = segment.merge_blocks([band, *cards, *halves, *chevrons, sparse, inner], W, H,
                                  grad=segment.gradient(rgb))
    sizes = sorted(len(items) for _, items in groups)
    assert sizes == [1]*7+[2, 2]
    assert any(len(items) == 2 and items[0]['bbox'][1] == 220 and items[0]['bbox'][0] in (40, 140) for _, items in groups)
    photo = page(W, H)
    rng = np.random.default_rng(11)
    photo[100:300, 100:400] = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:H, :W]
    sky = (yy >= 100) & (yy < 300) & (xx >= 100) & (xx < 400) & (yy < 170+20*np.sin(xx/15.))
    photo[sky] = (150, 190, 235)
    header = piece(100, 60, 300, 40)
    photo[header['mask']] = (20, 110, 200)
    upper = _piece(100, 100, 300, 120, H, W, sky[100:220, 100:400])
    lower_photo = piece(100, 100, 300, 200)
    lower_photo['mask'] &= ~sky; lower_photo['area'] = int(lower_photo['mask'].sum())
    grad = segment.gradient(photo)
    assert len(segment.merge_blocks([upper, lower_photo], W, H, grad=grad)) == 1
    assert len(segment.merge_blocks([header, upper, lower_photo], W, H, grad=grad)) == 2
    over = page(W, H)
    lower = piece(20, 50, 300, 60)
    over[band['mask']] = (200, 220, 240)
    over[lower['mask']] = (90, 90, 90)
    assert len(segment.merge_blocks([band, lower], W, H, grad=segment.gradient(over))) == 2


def test_stray_specks_do_not_stretch_a_chevron_over_its_neighbours():
    H, W = 300, 900
    rgb = page(W, H)
    shape = segment._chevron(200, 60, 20)
    chevrons = [_piece(40+186*k, 100, 200, 60, H, W, shape) for k in range(3)]
    for k, c in enumerate(chevrons):
        rgb[c['mask']] = (30, 90, 200) if k % 2 == 0 else (40, 170, 150)
    speckled = {**chevrons[1], 'mask': chevrons[1]['mask'].copy()}
    speckled['mask'][120:123, 620:623] = True
    speckled['mask'][110:112, 60:62] = True
    speckled.update(bbox=segment._bbox(speckled['mask']), area=int(speckled['mask'].sum()))
    objects, _ = segment.blockify([chevrons[0], speckled, chevrons[2]], W, H, grad=segment.gradient(rgb))
    boxes = sorted(o['bbox'] for o in objects if o['kind'] == 'raster')
    assert [b[0] for b in boxes] == [40, 226, 412] and all(b[2] == 200 for b in boxes)
    assert all(o['geometry'] == 'chevron' for o in objects if o['kind'] == 'raster')


def test_a_photo_piece_grows_over_its_calm_sky_and_stops_at_the_header():
    rgb = page(1280, 720)
    H, W = rgb.shape[:2]
    rgb[40:330, 40:340] = (255, 255, 255)
    rgb[50:90, 50:330] = (20, 110, 200)
    sky = np.linspace(150, 210, 70)[:, None, None]*np.ones((1, 280, 3))
    rgb[90:160, 50:330] = sky.astype(np.uint8)
    rgb[160:230, 50:330] = np.random.default_rng(2).integers(0, 255, (70, 280, 3), dtype=np.uint8)
    found = _piece(60, 160, 260, 70, H, W)
    header = np.zeros((H, W), bool); header[50:90, 50:330] = True
    grad = segment.gradient(rgb)
    objects = segment.complete_pictures([found], header, grad, H, W)
    assert len(objects) == 1 and objects[0].get('completed')
    x, y, w, h = objects[0]['bbox']
    assert abs(x-50) <= 3 and abs(y-90) <= 3 and abs(x+w-330) <= 3 and abs(y+h-230) <= 3
    assert not (objects[0]['mask'] & header).any()
    yy, xx = np.mgrid[:H, :W]
    hill = (yy >= 720-120*np.exp(-((xx-80)/90.)**2)) & (xx < 300)
    noise = np.random.default_rng(4).integers(0, 255, (H, W, 3), dtype=np.uint8)
    rgb[hill] = noise[hill]
    edge = _piece(0, 640, 200, 80, H, W, hill[640:720, 0:200])
    assert segment.complete_pictures([edge], header, segment.gradient(rgb), H, W) == []


def test_three_levels_lift_to_a_repaired_layer_beneath():
    """Card, header on the card, icon on the header: each layer is repaired where the
    level above it sits, so moving the header shows the card and not a second header."""
    rgb = page()
    H, W = rgb.shape[:2]
    rng = np.random.default_rng(3)
    rgb[40:240, 60:300] = rng.integers(60, 200, (200, 240, 3), dtype=np.uint8)
    rgb[60:90, 80:240] = (20, 90, 200)
    icon = np.zeros((H, W), np.uint8)
    cv2.circle(icon, (100, 75), 8, 1, -1)
    rgb[icon > 0] = (230, 40, 40)
    card = np.zeros((H, W), bool); card[40:240, 60:300] = True
    header = np.zeros((H, W), bool); header[60:90, 80:240] = True
    detector = {'id': '', 'kind': 'raster', 'source': 'textmask_icon', 'alpha': 'mask', 'absorbed': 0,
                'mask': icon > 0, 'bbox': segment._bbox(icon > 0), 'area': int(icon.sum())}
    empty = np.zeros((H, W), bool)
    elements, _ = segment.classify(rgb, [(card, {'source': 'sam2'}), (header, {'source': 'sam2'})],
                                   text_mask=empty, icon_mask=empty)
    elements, erase = segment.blockify(elements+[detector], W, H, grad=segment.gradient(rgb))
    level = lambda pred: {e['level'] for e in elements if pred(e)}
    assert level(lambda e: e.get('role') == 'block') == {0}
    assert level(lambda e: e['kind'] == 'native_shape') == {1}
    assert 2 in level(lambda e: e.get('role') == 'icon')
    assert erase[card].all()
    beneath_header, _ = inpaint.inpaint(rgb, segment.layer_erase(elements, 0, H, W), None)
    assert np.abs(beneath_header[75, 200].astype(int)-(20, 90, 200)).sum() > 60
    beneath_icon, _ = inpaint.inpaint(rgb, segment.layer_erase(elements, 1, H, W), None)
    assert np.abs(beneath_icon[75, 100].astype(int)-(20, 90, 200)).sum() < 30
    assert (beneath_icon[100:230, 70:290] == rgb[100:230, 70:290]).all()


def _rebuilt(rgb, objects=()):
    H, W = rgb.shape[:2]
    empty = np.zeros((H, W), bool)
    return components.rebuild(rgb, list(objects), W, H, segment.gradient(rgb), text_mask=empty, icon_mask=empty)


def test_rebuild_nests_card_header_and_badge_as_three_native_levels():
    rgb = page()
    cv2.rectangle(rgb, (40, 40), (400, 300), (255, 255, 255), -1)
    cv2.rectangle(rgb, (40, 40), (400, 300), (200, 210, 225), 1)
    rgb[60:120, 60:380] = (20, 90, 200)
    rgb[75:105, 80:110] = (240, 150, 30)
    elements, _ = _rebuilt(rgb)
    assert [e['kind'] for e in elements] == ['native_shape']*3
    by_level = {e['level']: e for e in elements}
    assert sorted(by_level) == [0, 1, 2]
    assert by_level[0]['bbox'] == [40, 40, 361, 261] and by_level[0]['fill'] == 'FFFFFF'
    assert by_level[0]['line']['color'] == 'C8D2E1'
    assert by_level[1]['bbox'] == [60, 60, 320, 60] and by_level[1]['geometry'] == 'rect'
    assert by_level[2]['bbox'] == [80, 75, 30, 30] and by_level[2]['fill'] == 'F0961E'


def test_rebuild_fits_a_gradient_band():
    rgb = page()
    t = np.linspace(0, 1, 300)[None, :, None]
    rgb[100:180, 100:400] = np.rint((1-t)*np.array([20, 90, 200])+t*np.array([40, 180, 230])).astype(np.uint8)
    elements, _ = _rebuilt(rgb)
    assert len(elements) == 1
    band = elements[0]
    assert band['kind'] == 'native_shape' and band['bbox'] == [100, 100, 300, 80]
    stops = band['fill_gradient']['stops']
    hexrgb = lambda h: np.array([int(h[i:i+2], 16) for i in (0, 2, 4)])
    assert np.abs(hexrgb(stops[0]['color'])-(20, 90, 200)).max() <= 4
    assert np.abs(hexrgb(stops[-1]['color'])-(40, 180, 230)).max() <= 4


def test_rebuild_finds_a_white_card_on_a_white_page_by_its_border():
    rgb = np.full((360, 640, 3), 255, np.uint8)
    cv2.rectangle(rgb, (60, 60), (360, 260), (225, 230, 238), 1)
    elements, _ = _rebuilt(rgb)
    assert len(elements) == 1
    card = elements[0]
    assert card['kind'] == 'native_shape' and card['source'] == 'contour'
    assert card['bbox'] == [60, 60, 301, 201] and card['line']['color'] == 'E1E6EE'


def test_rebuild_textured_card_falls_back_to_a_picture_with_a_smooth_hole():
    """The picture under a native header shows no second header when the header moves."""
    rgb = page()
    rng = np.random.default_rng(1)
    rgb[40:300, 40:400] = rng.integers(60, 200, (260, 360, 3), dtype=np.uint8)
    rgb[80:140, 80:300] = (20, 90, 200)
    elements, _ = _rebuilt(rgb)
    picture = next(e for e in elements if e['kind'] == 'raster')
    header = next(e for e in elements if e['kind'] == 'native_shape')
    assert picture['fallback'] and picture['bbox'] == [40, 40, 360, 260] and picture['level'] == 0
    assert header['bbox'] == [80, 80, 220, 60] and header['level'] == 1
    H, W = rgb.shape[:2]
    beneath = components.smooth_under(rgb, segment.layer_erase(elements, 0, H, W))
    hole = beneath[90:130, 90:290].astype(int)
    assert np.abs(hole.mean((0, 1))-(130, 130, 130)).max() < 25
    assert np.abs(hole-(20, 90, 200)).sum(2).min() > 60
    assert (beneath[200:290, 50:390] == rgb[200:290, 50:390]).all()


def test_rebuild_background_is_original_outside_components_and_smooth_under_them():
    rgb = page()
    rgb[:, :, 2] = np.linspace(230, 252, 640).astype(np.uint8)[None, :]
    rgb[60:200, 80:320] = (20, 90, 200)
    elements, erase = _rebuilt(rgb)
    assert [e['kind'] for e in elements] == ['native_shape']
    background = components.smooth_under(rgb, erase)
    assert (background[~erase] == rgb[~erase]).all()
    under = background[70:190, 90:310].astype(int)
    assert np.abs(under-rgb[20:21, 90:310].astype(int)).max() <= 3


def test_duplicate_under_objects_flags_unrepaired_layers(tmp_path):
    rng = np.random.default_rng(5)
    under = np.full((100, 100, 3), 240, np.uint8)
    under[20:60, 20:60] = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    Image.fromarray(under).save(tmp_path/'background.png')
    repaired = np.full((100, 100, 3), 240, np.uint8)
    Image.fromarray(repaired).save(tmp_path/'repaired.png')
    faded = (240+(under.astype(float)-240)*.5).astype(np.uint8)
    Image.fromarray(faded).save(tmp_path/'faded.png')
    rgba = np.dstack([under[20:60, 20:60], np.full((40, 40), 255, np.uint8)])
    (tmp_path/'assets').mkdir()
    Image.fromarray(rgba, 'RGBA').save(tmp_path/'assets/a.png')
    flat = np.dstack([np.full((40, 40, 3), 240, np.uint8), np.full((40, 40), 255, np.uint8)])
    Image.fromarray(flat, 'RGBA').save(tmp_path/'assets/f.png')
    element = lambda i, asset, beneath: {'id': i, 'kind': 'raster', 'bbox': [20, 20, 40, 40],
                                         'asset': asset, 'beneath': beneath}
    perception = {'elements': [element('dup', 'assets/a.png', 'background.png'),
                               element('ok', 'assets/a.png', 'repaired.png'),
                               element('faded', 'assets/a.png', 'faded.png'),
                               element('flat', 'assets/f.png', 'repaired.png')]}
    assert [d['id'] for d in qa.duplicate_under_objects(tmp_path, perception)] == ['dup', 'faded']


def test_filled_picture_edge_is_hard():
    """The outer pixel of a filled picture is replaced by the pixel inside it."""
    rgb = np.full((80, 120, 3), 240, np.uint8)
    rgb[10:70, 10:110] = (20, 80, 160)
    rgb[10, 10:110] = (240, 240, 240)
    rgb[69, 10:110] = (240, 240, 240)
    rgb[10:70, 10] = (240, 240, 240)
    rgb[10:70, 109] = (240, 240, 240)
    mask = np.zeros((80, 120), bool)
    mask[10:70, 10:110] = True
    element = {'bbox': [10, 10, 100, 60], 'alpha': 'mask', 'mask': mask}
    rgba = segment.cutout(rgb, element)
    assert element['bbox'] == [10, 10, 100, 60]
    assert set(np.unique(rgba[:, :, 3])).issubset({0, 255})
    assert tuple(int(v) for v in rgba[0, 5, :3]) == (20, 80, 160)
    assert int(rgba[0, 5, 3]) == 255
    assert tuple(int(v) for v in rgba[30, 50, :3]) == (20, 80, 160)


def test_card_hides_nothing_unaccounted():
    """A native card is opaque: an uncandidated decoration inside becomes a cutout."""
    rgb = page()
    card = np.zeros(rgb.shape[:2], np.uint8)
    cv2.rectangle(card, (40, 40), (300, 200), 1, -1)
    rgb[card > 0] = (255, 255, 255)
    cv2.circle(rgb, (200, 120), 10, (240, 120, 0), -1)
    empty = np.zeros(card.shape, bool)
    elements, _ = segment.classify(rgb, [(card > 0, {'source': 'sam2'})], text_mask=empty, icon_mask=empty)
    assert any(e['kind'] == 'raster' and e['source'] == 'shape_residual' for e in elements)


def test_ruled_grid_becomes_table_with_cells():
    rgb = page()
    for y in (60, 100, 140, 180):
        cv2.line(rgb, (60, y), (420, y), (90, 90, 90), 1)
    for x in (60, 180, 300, 420):
        cv2.line(rgb, (x, 60), (x, 180), (90, 90, 90), 1)
    found = tables.detect_tables(rgb)
    assert len(found) == 1 and len(found[0]['xs']) == 4 and len(found[0]['ys']) == 4
    lines = [{'id': 't0', 'text': 'A1', 'bbox': [70, 70, 30, 20]}, {'id': 't1', 'text': 'C3', 'bbox': [310, 150, 30, 20]}]
    rows, owned, styles = tables.fill_cells(rgb, found[0], lines, {})
    assert rows[0][0] == 'A1' and rows[2][2] == 'C3' and owned == ['t0', 't1'] and len(styles) == 9


def style(i, x, y, text='line', size=20., color='222222', bold=False):
    return {'id': i, 'text': text, 'ink_bbox': [x, y, 160, 20], 'size_px': size, 'color': color,
            'bold': bold, 'origin': [x-1, y-4], 'family': '微软雅黑', 'fitted': True}


def test_paragraphs_merge_compatible_lines_only():
    styles = {s['id']: s for s in [style('a', 100, 100), style('b', 100, 126), style('c', 100, 152, size=30),
                                     style('d', 400, 100), style('e', 100, 300)]}
    groups = semantics.deterministic(styles, [])
    assert {'ids': ['a', 'b'], 'align': 'left', 'source': 'deterministic'} in groups
    assert sorted(i for g in groups for i in g['ids']) == sorted(styles)
    proposal = {'paragraphs': [{'ids': ['a', 'b', 'c'], 'align': 'left'}, {'ids': ['d', 'e']}]}
    accepted, issues = semantics.apply_model(proposal, styles, [], groups)
    assert len(issues) == 2 and sorted(i for g in accepted for i in g['ids']) == sorted(styles)


def test_residual_violation_ignores_retained_text():
    perception = {'ocr': [{'id': 't0', 'bbox': [10, 10, 50, 20]}],
                  'residual_ocr': [{'text': 'x', 'bbox': [12, 12, 30, 10], 'score': .9},
                                   {'text': 'ghost', 'bbox': [200, 200, 30, 10], 'score': .8},
                                   {'text': 'faint', 'bbox': [300, 300, 30, 10], 'score': .3}]}
    assert [h['text'] for h in qa.residual_violations(perception, {'t0'})] == ['ghost']
    assert qa.text_recall([{'text': 'AB C'}], [{'text': 'ABX'}]) == round(2/3, 4)


def fake_ocr(lines):
    calls = {'n': 0}

    def ocr(rgb):
        calls['n'] += 1
        return [dict(line) for line in lines] if calls['n'] == 1 else []
    return ocr


@pytest.mark.skipif(shutil.which('node') is None, reason='node required for IR validation')
def test_perceive_and_emit_produce_valid_layered_ir(tmp_path):
    rgb = page()
    cv2.rectangle(rgb, (30, 30), (330, 220), (255, 255, 255), -1)
    cv2.rectangle(rgb, (50, 50), (76, 76), (20, 110, 230), -1)
    write(rgb, 'REPORT', (90, 74))
    write(rgb, 'SECOND', (50, 140), .8)
    source = tmp_path/'p.png'
    Image.fromarray(rgb).save(source)
    lines = [{'text': 'REPORT', 'bbox': [48, 48, 200, 30], 'polygon': [], 'score': .99},
             {'text': 'SECOND', 'bbox': [48, 118, 140, 28], 'polygon': [], 'score': .99}]
    card = np.zeros(rgb.shape[:2], bool); card[30:221, 30:331] = True
    result = perceive(source, tmp_path/'out', ocr=fake_ocr(lines), sam=lambda image: [(card, {'source': 'sam2'})])
    assert result['counts']['ink_ok'] == 2
    styles = {}
    for record in result['text']:
        s = style(record['id'], *record['ink_bbox'][:2])
        s.update(text=next(l['text'] for l in result['ocr'] if l['id'] == record['id']), ink_bbox=record['ink_bbox'])
        styles[record['id']] = s
    paragraphs = semantics.deterministic(styles, [e['bbox'] for e in result['elements'] if e['kind'] == 'native_shape'])
    ir = emit_ir.emit(tmp_path/'out', result, styles, paragraphs, tmp_path/'ir')
    assert ir['background'].get('asset_id') or ir['background']['color']
    assert {n['type'] for n in ir['nodes']} >= {'text'}
    check = subprocess.run(['node', '--input-type=module', '-e',
                            "import('./scripts/ir.mjs').then(async m=>{const fs=await import('node:fs');"
                            "m.validateIR(JSON.parse(fs.readFileSync(process.argv[1],'utf8')));console.log('ok')})"
                            ".catch(e=>{console.error(e.message);process.exit(1)})", str(tmp_path/'ir'/'slideir.json')],
                           cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    assert check.returncode == 0, check.stderr
    perception = json.loads((tmp_path/'out'/'perception.json').read_text(encoding='utf-8'))
    assert perception['files']['background.png'] == perception['background']['sha256']


def test_wide_blank_splits_one_ocr_line_into_two_phrases():
    rgb = page()
    write(rgb, 'ABCD', (40, 64)); write(rgb, 'EFGH', (200, 64))
    line = {'id': 't0', 'text': 'ABCDEFGH', 'bbox': [36, 38, 260, 34], 'score': .99}
    parts = textmask.split_wide_gaps(rgb, [line])
    assert [p['text'] for p in parts] == ['ABCD', 'EFGH']
    assert parts[0]['bbox'][0]+parts[0]['bbox'][2] < 200 <= parts[1]['bbox'][0]+4
    tight = {'id': 't1', 'text': 'ABCD', 'bbox': [36, 38, 100, 34], 'score': .99}
    assert textmask.split_wide_gaps(rgb, [tight]) == [tight]


def test_lone_letter_before_a_text_line_is_a_bullet_icon():
    icon = {'text': 'Q', 'bbox': [100, 50, 20, 22], 'score': .99}
    label = {'text': 'REVIEW', 'bbox': [130, 50, 90, 22], 'score': .99}
    badge = {'text': '9', 'bbox': [100, 50, 20, 22], 'score': .99}
    assert bullet_icon(icon, [icon, label])
    assert not bullet_icon(badge, [badge, label])
    assert not bullet_icon(icon, [icon, {'text': 'FAR', 'bbox': [300, 50, 60, 22], 'score': .99}])


def test_every_ocr_line_needs_an_owner():
    from image2ppt_vision.layered.pipeline import ocr_coverage_gaps
    perception = {'ocr': [{'id': i} for i in ('t000', 't001', 't002', 't003')]}
    ir = {'nodes': [{'id': 'p000', 'type': 'text', 'evidence_ids': ['t000']}],
          'assets': [{'id': 'el-001', 'contains_evidence_ids': ['t001']}],
          'unresolved_evidence_ids': ['t002']}
    assert ocr_coverage_gaps(perception, ir) == ['t003']


def test_doubtful_short_lines_stay_pixels_and_tall_junk_is_dropped():
    from image2ppt_vision.layered.pipeline import doubtful_lines
    perception = {'ocr': [{'id': 'a', 'score': .62}, {'id': 'b', 'score': .93}, {'id': 'c', 'score': .93},
                          {'id': 'd', 'score': .7}]}
    styles = {'a': {'text': 'B', 'fitted': False}, 'b': {'text': '\u54c1', 'fitted': False},
              'c': {'text': 'AI', 'fitted': False}, 'd': {'text': 'B', 'fitted': True}}
    assert doubtful_lines(perception, styles) == {'a', 'b'}
    assert not plausible_text({'text': 'AB', 'bbox': [0, 0, 28, 95], 'score': .58})
    assert plausible_text({'text': 'AB', 'bbox': [0, 0, 28, 95], 'score': .95})


def test_shape_around_text_is_left_in_place():
    rgb = page()
    write(rgb, 'ABCDEF', (72, 88))
    cv2.rectangle(rgb, (124, 42), (127, 102), (30, 40, 60), -1)
    line = {'id': 't0', 'text': 'ABCDEF', 'bbox': [66, 50, 120, 45], 'score': .99}
    records, text_mask, _ = textmask.analyze_page(rgb, [line])
    assert records[0]['status'] == 'ink_exceeds_box'
    assert not text_mask.any()


def test_chevron_ribbon_fits_as_a_chevron():
    ribbon = segment._chevron(120, 36, 14)
    fit = segment.fit_geometry(ribbon)
    assert fit['geometry'] == 'chevron' and fit['iou'] >= .95


def test_tiny_same_color_fragments_merge_and_icons_are_not_sliced():
    rgb = page()
    left = np.zeros(rgb.shape[:2], bool)
    right = np.zeros_like(left)
    left[40:52, 40:52] = True
    right[40:52, 56:68] = True
    rgb[left | right] = (10, 40, 180)
    merged = segment.merge_fragments(rgb, [
        {'mask': left, 'bbox': segment._bbox(left), 'area': int(left.sum()), 'meta': {'source': 'sam2'}},
        {'mask': right, 'bbox': segment._bbox(right), 'area': int(right.sum()), 'meta': {'source': 'sam2'}},
    ])
    assert len(merged) == 1 and merged[0]['meta']['source'] == 'merged_fragment'
    icon = np.zeros_like(left)
    icon[80:120, 80:120] = True
    rgb[icon] = (200, 30, 30)
    half = icon.copy()
    half[:, 100:] = False
    elements, _ = segment.classify(rgb, [(half, {'source': 'sam2'}), (icon & ~half, {'source': 'sam2'})],
                                   text_mask=np.zeros_like(left), icon_mask=icon)
    assert not any(e['source'] == 'sam2' and (e['mask'] & icon).any() for e in elements)
    assert len(segment.icon_fallbacks(rgb, [[80, 80, 40, 40]], elements)) == 1


def test_solid_icon_disc_promotes_to_an_ellipse():
    rgb = page()
    cv2.circle(rgb, (120, 120), 22, (20, 120, 220), -1, cv2.LINE_AA)
    cv2.circle(rgb, (120, 120), 6, (255, 255, 255), -1, cv2.LINE_AA)
    empty = np.zeros(rgb.shape[:2], bool)
    elements, _ = segment.classify(rgb, [], text_mask=empty, icon_mask=empty,
                                   extra=segment.circle_candidates(rgb))
    disc = next(e for e in elements if e['kind'] == 'native_shape')
    assert disc['geometry'] == 'ellipse' or (
        disc['geometry'] == 'roundRect' and disc['corner_radius_px'] >= .4*min(disc['bbox'][2:]))


def test_flat_regions_promote_a_card_and_keep_the_inner_card():
    rgb = page()
    cv2.rectangle(rgb, (40, 40), (360, 250), (255, 255, 255), -1)
    cv2.rectangle(rgb, (60, 70), (180, 150), (225, 236, 252), -1)
    empty = np.zeros(rgb.shape[:2], bool)
    extra = segment.flat_region_candidates(rgb)
    elements, _ = segment.classify(rgb, [], text_mask=empty, icon_mask=empty, extra=extra)
    shapes = [e for e in elements if e['kind'] == 'native_shape']
    assert len(shapes) >= 2
    assert all(e['source'] == 'flat_region' for e in shapes)


def test_emitted_family_is_one_the_preview_can_draw():
    from image2ppt_vision.layered.typography import layered_font_catalog, measure
    rgb = page()
    write(rgb, 'REPORT', (40, 80))
    perception = {'image_size': [rgb.shape[1], rgb.shape[0]],
                  'ocr': [{'id': 't0', 'text': 'REPORT', 'bbox': [36, 52, 180, 40], 'score': .99}],
                  'text': [{'id': 't0', 'status': 'ok', 'ink_bbox': [40, 58, 140, 28],
                            'color': '1E283C', 'bold_ratio': .08}]}
    styles, _ = measure(rgb, perception)
    _, installed = layered_font_catalog()
    assert styles['t0']['family'] in installed
    assert styles['t0']['family'] != 'Source Han Sans SC'


def test_missing_font_manifest_keeps_installed_families():
    from image2ppt_vision.layered.typography import layered_font_catalog, office_family, page_family
    catalog, installed = layered_font_catalog(r'C:\no\such\layered-fonts\manifest.json')
    assert catalog and all(item['family'] != 'Source Han Sans SC' for item in catalog)
    assert office_family([{'family': 'Source Han Sans SC'}, {'family': '微软雅黑'}], {'微软雅黑'}) == '微软雅黑'
    assert office_family([{'family': 'Source Han Sans SC'}], installed) is None or 'Source Han Sans SC' not in installed
    assert page_family(['微软雅黑']*3+['黑体'], '宋体') == '微软雅黑'
    assert page_family(['微软雅黑', '黑体'], '宋体') == '宋体'
