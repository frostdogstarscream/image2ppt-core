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
