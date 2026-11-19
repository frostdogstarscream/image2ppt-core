"""Focused unit & compiler integration tests for the OpenCode native-first mainline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from product import fonts, scorecard
from product.materialize import digest, write
from product.parallel_branches import snapshot
from product.parallel_contracts import SceneGraph
from product.parallel_pipeline import effective_texts
from product.parallel_rebuild import build
from product.parallel_residual_gate import inspect_residuals
from product.recovery.chart_stage import stage_native_charts
from product.recovery.container_stage import stage_containers_and_surfaces
from product.recovery.gradient_backing import fit_gradient
from product.recovery.native_shapes import estimate_canvas_field, try_fit_native_shape_v2
from product.runtime.compiler import compile_local
from product.runtime.independent_review import compute_inventory_and_form_metrics
from product.runtime.page_structure import OCRCorrection, verify_and_apply_ocr_corrections
from product.tolerance_config import load_tolerance_config


def test_tolerance_config_keeps_hard_gates_immutable(tmp_path):
    bad_override = tmp_path / 'override.json'
    write(bad_override, {
        'profile': 'custom',
        'hard_gates': {'forbid_duplicate_text': False, 'forbid_unknown_erasure': False},
        'fidelity': {'min_ssim': 0.82},
    })
    cfg = load_tolerance_config(bad_override)
    assert cfg['profile'] == 'custom'
    assert cfg['fidelity']['min_ssim'] == 0.82
    assert cfg['hard_gates']['forbid_duplicate_text'] is True
    assert cfg['hard_gates']['forbid_unknown_erasure'] is True
    assert cfg['hard_gates']['forbid_residual_ink_ghost'] is True


def test_multi_sided_gradient_fit_unlocks_blocked_left_heading():
    # Heading next to sequence badge on left: left lateral strip is protected, top/bottom/right are clean sky.
    h, w = 110, 340
    yy, xx = np.mgrid[:h, :w]
    sky = np.clip(np.dstack([242 + 0.01 * xx, 248 + 0.005 * yy, np.full_like(xx, 254, dtype=float)]), 0, 255).astype(np.uint8)
    src = sky.copy()
    src[35:68, 90:260] = [12, 48, 128]
    protected = np.zeros((h, w), dtype=bool)
    protected[28:75, 10:86] = True  # Sequence badge `08 |` immediately on the left

    backing, report = fit_gradient(src, [88, 33, 176, 38], protected, context=18, allow_vertical=True)
    assert report['allow_vertical_context'] is True
    assert report['validation_p95_rgb'] <= 2.5
    assert np.abs(backing[35:68, 90:260].astype(int) - sky[35:68, 90:260].astype(int)).max() <= 2


def test_native_shapes_pentagon_ellipse_gradient_and_scenery_guard(tmp_path):
    w, h = 640, 420
    # Sky gradient top + dark cityscape footer at bottom 15%
    yy, _ = np.mgrid[:h, :w]
    sky = np.dstack([
        np.clip(244 - (yy * 8 // h), 0, 255),
        np.clip(249 - (yy * 5 // h), 0, 255),
        np.full((h, w), 254),
    ]).astype(np.uint8)
    img = Image.fromarray(sky)
    draw = ImageDraw.Draw(img)
    # Bottom scenery silhouette (touches bottom edge, wide textured banner)
    for bx in range(0, w, 28):
        bh_b = 28 + (bx * 13) % 30
        draw.rectangle((bx, h - bh_b, bx + 24, h - 1), fill=(165, 202, 234))

    # 1) White bordered card on sky
    draw.rounded_rectangle((30, 30, 310, 290), radius=8, fill=(250, 252, 255), outline=(205, 222, 242), width=1)
    # 2) Right-pointing pentagon ribbon inside card
    draw.polygon([(31, 31), (185, 31), (208, 52), (185, 73), (31, 73)], fill=(20, 102, 198))
    # 3) Circular ellipse badge on right half of slide
    draw.ellipse((380, 50, 444, 114), fill=(32, 132, 245))
    # 4) Vertical linear-gradient rectangle bar on right half of slide
    arr = np.array(img)
    bar_h = 110
    for r in range(bar_h):
        t = r / float(bar_h - 1)
        arr[150 + r, 390:436] = [int(25 + 140 * t), int(125 + 85 * t), 252]

    field, bg_hex, scenery_guard, is_flat = estimate_canvas_field(arr)
    assert not is_flat
    assert scenery_guard[-4:, :].mean() > 0.5  # Footer scenery protected!
    assert len(bg_hex) == 6

    # Direct fit checks on pentagon, ellipse, and gradient bar
    pent_fit = try_fit_native_shape_v2(
        arr, [31, 31, 178, 43], np.full((43, 178, 3), [250, 252, 255], float), [], np.zeros((43, 178), bool)
    )
    assert pent_fit is not None and pent_fit['geometry'] == 'pentagon'

    ell_fit = try_fit_native_shape_v2(
        arr, [380, 50, 65, 65], field[50:115, 380:445], [], np.zeros((65, 65), bool)
    )
    assert ell_fit is not None and ell_fit['geometry'] == 'ellipse'

    grad_fit = try_fit_native_shape_v2(
        arr, [390, 150, 46, 110], field[150:260, 390:436], [], np.zeros((110, 46), bool)
    )
    assert grad_fit is not None and grad_fit['fill_gradient'] is not None

    # Full container stage -> SlideIR -> fixed compiler integration!
    src_path = tmp_path / 'slide.png'
    Image.fromarray(arr).save(src_path)
    manifest = fonts.font_manifest([])
    rec = next(r for r in manifest['fonts'] if r['family'] == 'Arial' and not r['bold'])
    font_data = {'schema': 'FontManifest/1', 'fonts': [rec]}
    snap = snapshot(src_path, tmp_path / 'snap')
    graph = SceneGraph.model_validate({
        'schema': 'SceneGraph/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'transform': [1, 0, 0, 1, 0, 0],
        'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'method_version': 'test',
                 'styles': [{'id': 'st', 'role': 'body', 'ea': 'Arial', 'latin': 'Arial', 'size_px': 16.0, 'bold': False, 'color': '1E283C'}],
                 'texts': [], 'members': {}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'method_version': 'test', 'objects': []},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'method_version': 'test', 'background': 'F4F9FE', 'shapes': []},
        'ownership': {}, 'relations': [],
    })
    base_ir, _ = build(graph, snap, font_data, tmp_path / 'base', strategy='off', text_candidates=False, surface_candidates=False)
    cont_ir, cont_ids = stage_containers_and_surfaces(base_ir.parent, tmp_path / 'cont')
    assert len(cont_ids) >= 3
    cont_graph = SceneGraph.model_validate_json((cont_ir.parent / 'scene.json').read_text('utf8'))
    compiled = compile_local([cont_ir], tmp_path / 'compiled', [effective_texts(cont_graph, cont_ir)])
    sc = scorecard.score_page(src_path, compiled['pptx'])
    assert sc['object_counts'].get('shape', 0) >= 3


def test_native_chart_stage_compiles_valid_chart_node(tmp_path):
    w, h = 480, 300
    img = Image.new('RGB', (w, h), (250, 252, 255))
    draw = ImageDraw.Draw(img)
    # 3 data bars
    draw.rectangle((90, 170, 130, 220), fill=(40, 130, 240))
    draw.rectangle((190, 130, 230, 220), fill=(40, 130, 240))
    draw.rectangle((290, 90, 330, 220), fill=(40, 130, 240))
    src_path = tmp_path / 'chart.png'
    img.save(src_path)

    manifest = fonts.font_manifest([])
    rec = next(r for r in manifest['fonts'] if r['family'] == 'Arial' and not r['bold'])
    font_data = {'schema': 'FontManifest/1', 'fonts': [rec]}
    snap = snapshot(src_path, tmp_path / 'snap')
    texts = [
        {'id': 't1', 'text': 'Q1', 'bbox': [95.0, 228.0, 30.0, 18.0], 'style': 'st', 'z': 1},
        {'id': 't2', 'text': 'Q2', 'bbox': [195.0, 228.0, 30.0, 18.0], 'style': 'st', 'z': 2},
        {'id': 't3', 'text': 'Q3', 'bbox': [295.0, 228.0, 30.0, 18.0], 'style': 'st', 'z': 3},
        {'id': 'v1', 'text': '50', 'bbox': [95.0, 148.0, 28.0, 16.0], 'style': 'st', 'z': 4},
        {'id': 'v2', 'text': '90', 'bbox': [195.0, 108.0, 28.0, 16.0], 'style': 'st', 'z': 5},
        {'id': 'v3', 'text': '130', 'bbox': [295.0, 68.0, 32.0, 16.0], 'style': 'st', 'z': 6},
    ]
    graph = SceneGraph.model_validate({
        'schema': 'SceneGraph/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'transform': [1, 0, 0, 1, 0, 0],
        'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'method_version': 'test',
                 'styles': [{'id': 'st', 'role': 'body', 'ea': 'Arial', 'latin': 'Arial', 'size_px': 14.0, 'bold': False, 'color': '1E283C'}],
                 'texts': texts, 'members': {t['id']: [t['id']] for t in texts}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'method_version': 'test', 'objects': []},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'method_version': 'test', 'background': 'FAFCFF', 'shapes': []},
        'ownership': {t['id']: 'RESOLVED' for t in texts}, 'relations': [],
    })
    base_ir, _ = build(graph, snap, font_data, tmp_path / 'base', strategy='off', text_candidates=False, surface_candidates=False)
    layout = {'regions': [{'label': 'chart', 'bbox': [60.0, 60.0, 310.0, 195.0], 'score': 0.91}]}
    ch_ir, ch_ids = stage_native_charts(base_ir.parent, tmp_path / 'chart-staged', layout=layout, fonts=font_data)
    assert ch_ids == ['ch-01']
    ch_graph = SceneGraph.model_validate_json((ch_ir.parent / 'scene.json').read_text('utf8'))
    compiled = compile_local([ch_ir], tmp_path / 'compiled', [effective_texts(ch_graph, ch_ir)])
    sc = scorecard.score_page(src_path, compiled['pptx'])
    assert sc['object_counts'].get('chart', 0) == 1


def test_verified_ocr_proofreading_accepts_single_char_typo_and_rejects_hallucination():
    manifest = fonts.font_manifest([])
    cjk = next((r for r in manifest['fonts'] if r.get('cjk')), manifest['fonts'][0])
    img = Image.new('RGB', (220, 60), (40, 95, 165))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(cjk['file'], 22, index=cjk['index'])
    draw.text((20, 16), '会议材料整理', font=font, fill=(250, 252, 255))
    tb = draw.textbbox((20, 16), '会议材料整理', font=font)
    src = np.array(img)
    ocr = {
        'lines': [
            {'id': 'l1', 'text': '会议材料管理', 'score': 0.89, 'bbox': [float(tb[0]), float(tb[1]), float(tb[2] - tb[0]), float(tb[3] - tb[1])]},
            {'id': 'l2', 'text': '会议材料管理', 'score': 0.89, 'bbox': [float(tb[0]), float(tb[1]), float(tb[2] - tb[0]), float(tb[3] - tb[1])]},
        ]
    }
    corrections = [
        OCRCorrection(ocr_id='l1', corrected_text='会议材料整理', reason='typo fix'),
        OCRCorrection(ocr_id='l2', corrected_text='完全不同的无关句子内容', reason='hallucinated'),
    ]
    updated, applied = verify_and_apply_ocr_corrections(src, ocr, corrections, fonts={'fonts': [cjk]})
    assert len(applied) == 1 and applied[0]['ocr_id'] == 'l1'
    assert updated['lines'][0]['text'] == '会议材料整理'
    assert updated['lines'][1]['text'] == '会议材料管理'


def test_residual_gate_allows_minor_surface_shade_deviation_while_blocking_ink_ghost(tmp_path):
    w, h = 160, 80
    src = np.full((h, w, 3), [20, 100, 195], dtype=np.uint8)
    src[25:45, 40:110] = [250, 252, 255]  # White text ink over blue ribbon
    src_path = tmp_path / 'source.png'
    Image.fromarray(src).save(src_path)
    Image.fromarray(np.full((h, w, 3), 250, dtype=np.uint8)).save(tmp_path / 'source-plate.png')

    mask = np.zeros((h, w), dtype=bool)
    mask[25:45, 40:110] = True
    backing_patch = np.full((30, 90, 3), [20, 100, 195], dtype=np.uint8)
    proof_file = tmp_path / 't1-owner.npz'
    np.savez_compressed(proof_file, mask=mask, restore_bbox=[30, 20, 90, 30], backing_patch=backing_patch)

    scene = {
        'source_sha256': digest(src_path),
        'raster_backing': {'path': 'source-plate.png', 'sha256': digest(tmp_path / 'source-plate.png')},
        'selected': {
            's-001': {'id': 's-001', 'kind': 'surface', 'bbox': [10, 10, 140, 60]},
            't1': {'id': 't1', 'kind': 'text', 'bbox': [40, 25, 70, 20],
                   'owner_evidence': {'file': 't1-owner.npz', 'sha256': digest(proof_file)}},
        },
    }
    write(tmp_path / 'scene.json', scene)
    write(tmp_path / 'slideir.json', {
        'nodes': [
            {'id': 's-001', 'type': 'shape', 'z': 1, 'bbox': [10, 10, 140, 60]},
            {'id': 't1', 'type': 'text', 'z': 2, 'bbox': [40, 25, 70, 20]},
        ]
    })

    office_dir = tmp_path / 'office'
    office_dir.mkdir()
    # Case 1: Native gradient shape rendered at [32, 112, 206] (12 RGB shade diff, zero white text ghosting) -> PASS!
    del_clean = np.full((h, w, 3), [32, 112, 206], dtype=np.uint8)
    Image.fromarray(del_clean).save(office_dir / '01-t1-delete.png')
    res1 = inspect_residuals(tmp_path / 'slideir.json', office_dir, 1, tmp_path / 'res1.json')
    t1_r1 = next(r for r in res1 if r['id'] == 't1')
    assert t1_r1['residual_check_status'] == 'PASS'
    assert t1_r1['minor_deviation_pixels'] > 0

    # Case 2: Old white text ink left behind ([235, 240, 248], closer to src than backing!) -> HARD FAIL!
    del_ghost = del_clean.copy()
    del_ghost[28:40, 50:80] = [235, 240, 248]
    Image.fromarray(del_ghost).save(office_dir / '01-t1-delete.png')
    res2 = inspect_residuals(tmp_path / 'slideir.json', office_dir, 1, tmp_path / 'res2.json')
    t1_r2 = next(r for r in res2 if r['id'] == 't1')
    assert t1_r2['residual_check_status'] == 'FAIL'
    assert t1_r2['residual_pixels_confirmed'] > 0


def test_inventory_recall_and_form_metrics():
    sc = {
        'image_size': [1672, 941],
        'native_editable_content_fraction': 0.62,
        'movable_raster_content_fraction': 0.24,
        'objects': [
            {'class': 'text', 'name': 't-001', 'text': '\u5e94\u7528\u6210\u6548\u4e0e\u672a\u6765\u5c55\u671b', 'bbox': [155, 24, 400, 44]},
            {'class': 'shape', 'name': 's-001', 'bbox': [52, 170, 500, 430]},
            {'class': 'opaque_picture', 'name': 'r-001', 'bbox': [587, 248, 250, 160]},
        ],
    }
    inv = {
        'elements': [
            {'id': 'i1', 'category': 'title', 'bbox': [158, 24, 398, 43], 'label': '应用成效与未来展望'},
            {'id': 'i2', 'category': 'card_shape', 'bbox': [52, 170, 500, 430], 'label': 'card'},
            {'id': 'i3', 'category': 'photo', 'bbox': [587, 248, 249, 157], 'label': 'photo'},
            {'id': 'i4', 'category': 'logo', 'bbox': [1440, 33, 183, 58], 'label': 'logo'},
        ]
    }
    m = compute_inventory_and_form_metrics(sc, inv)
    assert m['main_title_editable'] is True
    assert m['recall'] is None
    assert m['proper_form_object_fraction'] == 0.75
    sc['objects'][0]['text'] += '1'
    sc['objects'][1]['class'] = 'opaque_picture'
    inv['elements'][2]['category'] = 'table'
    changed = compute_inventory_and_form_metrics(sc, inv)
    assert changed['matched_items'] == 0
    assert not changed['main_title_editable']
    assert compute_inventory_and_form_metrics(sc, None)['recall'] is None


def test_marked_picture_routes_use_the_crop_api():
    # Both photo routes must compile their keyword arguments against the real crop API.
    import ast
    import inspect
    from product.recovery.region_crop import stage_region_crops
    from product.runtime.batch_backbone import _chain_stage_from_spec
    signature = inspect.signature(stage_region_crops)
    calls = [n for n in ast.walk(ast.parse(inspect.getsource(_chain_stage_from_spec)))
             if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='stage_region_crops']
    assert len(calls) >= 2
    for call in calls:
        signature.bind(*[None for _ in call.args], **{kw.arg:None for kw in call.keywords})


def test_review_controls_reject_flat_output_and_respect_unknown_response(tmp_path):
    from types import SimpleNamespace
    from product.runtime.review_control import verify_reviewer
    reviewer=SimpleNamespace(model='review-fixture',session='fixture',ask=lambda *args:{'defects':{}})
    budget=SimpleNamespace(reserve=lambda *args:True)
    assert verify_reviewer(reviewer,tmp_path/'failed-control',budget)['passed'] is False
    unknown=SimpleNamespace(reserve=lambda *args:False)
    assert verify_reviewer(reviewer,tmp_path/'unknown-control',unknown)['status']=='UNKNOWN'


def test_diagram_groups_only_all_native_detected_bars():
    from product.recovery.container_stage import attach_bar_groups
    boxes = [(0, 8, 6, 2), (10, 6, 6, 4), (20, 3, 6, 7)]
    objects = {str(i):{'kind':'surface', 'bbox':list(box), 'payload':{'geometry':'rect'}}
               for i, box in enumerate(boxes)}
    objects['2']['kind'] = 'asset'
    attach_bar_groups(objects, [boxes])
    assert 'move_members' not in objects['0']
    objects['2']['kind'] = 'surface'
    attach_bar_groups(objects, [boxes])
    assert objects['0']['move_members'] == ['0', '1', '2']


def test_only_verified_source_readings_can_change_native_text():
    from types import SimpleNamespace
    import pytest
    from product.runtime.text_reading import authorize_readings
    def scene(value):
        return SimpleNamespace(source_sha256='src', text=SimpleNamespace(texts=[SimpleNamespace(id='t1',text=value)]))
    original, revised = scene('abx'), scene('abc')
    proof = {'source_sha256':'src', 'changes':[dict(object_id='t1',before='abx',after='abc',
             crop_sha256='crop',iou_before=.6,iou_after=.65)]}
    authorize_readings(original, revised, proof)
    with pytest.raises(ValueError):
        authorize_readings(original, revised, {**proof, 'source_sha256':'other'})
    with pytest.raises(ValueError):
        authorize_readings(original, scene('abd'), proof)
    with pytest.raises(ValueError):
        authorize_readings(original, revised, None)
