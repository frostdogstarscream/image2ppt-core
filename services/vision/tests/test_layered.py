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

from image2ppt_vision.layered import emit_ir, inpaint, qa, segment, semantics, tables, textmask
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
    assert segment.cutout(rgb, raster).shape[2] == 4


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


def style(i, x, y, text='line', size=20., color='222222', bold=False, w=160, h=20):
    return {'id': i, 'text': text, 'ink_bbox': [x, y, w, h], 'size_px': size, 'color': color,
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


def _ids(groups):
    return [g['ids'] for g in groups]


def test_same_card_wrapped_lines_merge_across_measurement_jitter():
    """Architecture-card body: one sentence wrapped onto lines whose ink, size and colour jitter."""
    h = 18
    card = [40, 60, 240, 180]
    # Same card, but a bluer bold title stays its own box.
    title = style('title', 58, 78, '线索研判', size=18, color='0956D7', bold=True, w=120, h=h)
    body = style('body', 58, 112, '数据碰撞', size=16, color='536B8C', w=150, h=h)
    # Left edge shifts by one em, sampled grey is ~50 levels off, and width-limiting
    # shrunk this line from the same 16px size to 75%.
    wrap = style('wrap', 58+h, 112+h+4, '关联分析', size=12, color='889FB8', w=150, h=h)
    wrap['width_limited_from_px'] = 16
    # Centered short last line whose center misses the old 0.6em band but stays inside one em.
    head = style('head', 70, 160, '校园餐监督', size=16, color='0E2760', bold=True, w=160, h=h)
    # Center is 16px off (inside 1.35em, outside the old 0.6em band); the left edge is not shared.
    short = style('short', 122, 160+h+3, '专项监督', size=16, color='122C68', bold=True, w=88, h=h)
    styles = {s['id']: s for s in (title, body, wrap, head, short)}
    assert 40 < semantics._lab_distance(body['color'], wrap['color']) <= 56
    assert semantics._lab_distance(title['color'], body['color']) > 56
    groups = semantics.deterministic(styles, [card])
    assert _ids(groups) == [['title'], ['body', 'wrap'], ['head', 'short']]
    assert next(g for g in groups if g['ids']==['head', 'short'])['align'] == 'center'
    proposal = {'paragraphs': [{'ids': ['body', 'wrap'], 'align': 'left'},
                               {'ids': ['head', 'short'], 'align': 'center'},
                               {'ids': ['title'], 'align': 'left'}]}
    accepted, issues = semantics.apply_model(proposal, styles, [card], groups)
    assert issues == [] and _ids(accepted) == [['body', 'wrap'], ['head', 'short'], ['title']]
    # Same ink with a bold flicker (stroke-ratio noise) still joins; a real weight change does not.
    flicker = {s['id']: s for s in (style('a', 58, 112, size=16, color='536B8C', bold=True, w=140, h=h),
                                     style('b', 58, 112+h+4, size=16, color='5A7394', bold=False, w=140, h=h))}
    assert semantics._lab_distance('536B8C', '5A7394') <= 18
    assert _ids(semantics.deterministic(flicker, [card])) == [['a', 'b']]
    distinct = {s['id']: s for s in (style('a', 58, 112, size=16, color='536B8C', bold=True, w=140, h=h),
                                      style('b', 58, 112+h+4, size=16, color='889FB8', bold=False, w=140, h=h))}
    assert _ids(semantics.deterministic(distinct, [card])) == [['a'], ['b']]


def test_same_line_midword_wrap_merges_despite_color_and_size():
    """02 slide 3 intro: 依|托 is one baseline split into two OCR boxes."""
    h = 34
    # Right piece sits 2px higher, so y-order would see it first. Colour, bold and
    # fitted size all disagree with the vertical merge rules.
    left = style('t003', 400, 178, '面向监督检查与审查调查两大业务方向，依',
                 size=30.3, color='1170C0', bold=False, w=520, h=h)
    right = style('t004', 942, 176, '托数据治理、纪检知识库、大模型及专业智能体，',
                  size=46, color='111111', bold=True, w=480, h=h)
    assert semantics._lab_distance(left['color'], right['color']) > semantics._COLOR_GAP
    assert abs(left['size_px']-right['size_px'])/right['size_px'] > semantics._SIZE_RATIO
    band = [0, 140, 1800, 120]
    styles = {s['id']: s for s in (left, right)}
    assert semantics.compatible(left, right, [band]) == 'left'
    groups = semantics.deterministic(styles, [band])
    assert _ids(groups) == [['t003', 't004']]
    accepted, issues = semantics.apply_model(
        {'paragraphs': [{'ids': ['t004', 't003'], 'align': 'left'}]}, styles, [band], groups)
    assert issues == [] and accepted[0]['ids'] == ['t003', 't004']
    node = emit_ir._paragraph_node('p000', styles, ['t003', 't004'], 'left', 1920, 1080, 1280/1920)
    assert node['text'] == left['text']+right['text']
    assert node['text_override_reason'] == 'Same-baseline OCR fragments joined without a line break.'
    wrapped = emit_ir._paragraph_node('p001', {s['id']: s for s in (
        style('a', 40, 40, '上一行', h=20), style('b', 40, 70, '下一行', h=20))}, ['a', 'b'], 'left', 1920, 1080, 1)
    assert wrapped['text'] == '上一行\n下一行' and 'text_override_reason' not in wrapped
    # A column on the same baseline, one gutter away, stays its own box.
    column = style('col', 1422+120, 178, '审查调查', size=30.3, color='1170C0', w=160, h=h)
    wide = {**styles, column['id']: column}
    assert _ids(semantics.deterministic(wide, [band])) == [['t003', 't004'], ['col']]
    assert semantics.compatible(right, column, [band]) is None
    # The same 22px gap across two cards does not merge.
    card_l, card_r = [390, 160, 540, 70], [930, 160, 520, 70]
    assert semantics._container(left['ink_bbox'], [card_l, card_r]) != semantics._container(
        right['ink_bbox'], [card_l, card_r])
    assert semantics.compatible(left, right, [card_l, card_r]) is None
    assert _ids(semantics.deterministic(styles, [card_l, card_r])) == [['t004'], ['t003']]


def test_cross_column_and_cross_card_lines_do_not_merge():
    h = 18
    # Stacked cards share an x position and a one-line gap. Only the card boundary separates them.
    above = [40, 40, 200, 100]
    below = [40, 136, 200, 90]
    top = style('top', 58, 112, '整改跟踪', size=16, color='536B8C', w=140, h=h)
    bottom = style('bottom', 58, 140, '闭环管理', size=16, color='536B8C', w=140, h=h)
    stacked = {s['id']: s for s in (top, bottom)}
    assert _ids(semantics.deterministic(stacked, [above, below])) == [['top'], ['bottom']]
    # One band, two columns. The right column starts a hair lower and barely overlaps the left line,
    # which must not swallow the left column's wrapped second line.
    band = [20, 40, 700, 200]
    left = style('left', 50, 80, '证据梳理', size=16, color='536B8C', w=170, h=h)
    wrap = style('wrap', 50+h, 80+h+6, '调查取证', size=16, color='5E7596', w=170, h=h)
    graze = style('graze', 205, 88, '审查调查', size=16, color='536B8C', w=100, h=h)
    right = style('right', 480, 86, '案件审理', size=16, color='536B8C', w=140, h=h)
    columns = {s['id']: s for s in (left, wrap, graze, right)}
    groups = semantics.deterministic(columns, [band])
    assert _ids(groups) == [['left', 'wrap'], ['right'], ['graze']]
    rejected, issues = semantics.apply_model(
        {'paragraphs': [{'ids': ['left', 'right'], 'align': 'left'}, {'ids': ['wrap', 'graze']}]},
        columns, [band], groups)
    assert any(i.startswith('incompatible_paragraph:') for i in issues)
    assert {tuple(g['ids']) for g in rejected} == {('left', 'wrap'), ('graze',), ('right',)}
    # A clearly different size directly underneath still ends the paragraph.
    heading = style('h', 50, 80, size=16, w=170, h=h)
    other = style('o', 50, 80+h+4, size=28, w=170, h=h)
    assert _ids(semantics.deterministic({s['id']: s for s in (heading, other)}, [band])) == [['h'], ['o']]


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
