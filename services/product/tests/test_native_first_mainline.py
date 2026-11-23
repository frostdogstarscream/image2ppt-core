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
    assert pent_fit is not None and pent_fit['geometry'] == 'custom' and pent_fit['custom_paths']

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


def test_queue_loop_reads_utf8_goal_and_survives_worker_errors(tmp_path, monkeypatch):
    import threading
    import time
    from product import app as app_mod

    q = app_mod.Queue(tmp_path / 'jobs')
    j1 = q.root / ('a' * 32)
    j1.mkdir(parents=True)
    # Contains Chinese + em-dash + emoji that fails GBK decode if read without encoding='utf-8'.
    (j1 / 'request.json').write_text(
        json.dumps({
            'orchestrator': 'opencode',
            'runtime_seconds': 60,
            'sources': ['01.png'],
            'goal': '将原图恢复为尽可能全原生可编辑的 PPTX — 测试 UTF-8 🚀',
        }, ensure_ascii=False),
        encoding='utf-8',
    )
    (j1 / 'result.json').write_text(json.dumps({'status': 'SUCCEEDED'}), encoding='utf-8')

    j2 = q.root / ('b' * 32)
    j2.mkdir(parents=True)
    (j2 / 'request.json').write_text('{bad json', encoding='utf-8')

    with q.connect() as db:
        db.execute('INSERT INTO jobs VALUES (?,?,?)', (j1.name, 'QUEUED', 1.0))
        db.execute('INSERT INTO jobs VALUES (?,?,?)', (j2.name, 'QUEUED', 2.0))

    class _FakeProc:
        pid = 99999
        def poll(self):
            return 0
        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(app_mod.subprocess, 'Popen', lambda *a, **kw: _FakeProc())
    t = threading.Thread(target=q.loop, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            s1 = q.status(j1.name)['status']
            s2 = q.status(j2.name)['status']
            if s1 == 'SUCCEEDED' and s2 == 'FAILED':
                break
            time.sleep(0.05)
        assert q.status(j1.name)['status'] == 'SUCCEEDED'
        assert q.status(j2.name)['status'] == 'FAILED'
        assert t.is_alive()
    finally:
        q.stop_event.set()
        t.join(timeout=3.0)


def test_region_candidates_and_candidate_stage_roundtrip_vertical_fallback(tmp_path):
    from product.recovery.candidate_stage import stage_background
    from product.recovery.region_candidates import adaptive_candidates

    manifest = fonts.font_manifest([])
    rec = next(r for r in manifest['fonts'] if r['family'] == 'Arial' and not r['bold'])
    font_data = {'schema': 'FontManifest/1', 'fonts': [rec]}

    w, h = 280, 120
    img = Image.new('RGB', (w, h), (242, 248, 254))
    draw = ImageDraw.Draw(img)
    # Draw a sequence badge / icon immediately left of Heading (`x=20..80`), blocking left lateral strip
    badge_bbox = [20, 38, 65, 34]
    draw.rectangle((20, 38, 84, 71), fill=(25, 118, 210))
    font = ImageFont.truetype(rec['file'], 20, index=rec['index'])
    draw.text((88, 42), 'Heading', font=font, fill=(15, 50, 120))
    tb = draw.textbbox((88, 42), 'Heading', font=font)
    tbbox = [float(tb[0]), float(tb[1]), float(tb[2] - tb[0]), float(tb[3] - tb[1])]

    src_path = tmp_path / 'src.png'
    img.save(src_path)
    snap = snapshot(src_path, tmp_path / 'snap')
    graph = SceneGraph.model_validate({
        'schema': 'SceneGraph/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'transform': [1, 0, 0, 1, 0, 0],
        'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'method_version': 'test',
                 'styles': [{'id': 'st', 'role': 'title', 'ea': 'Arial', 'latin': 'Arial', 'size_px': 20.0, 'bold': False, 'color': '0F3278'}],
                 'texts': [{'id': 't1', 'text': 'Heading', 'bbox': tbbox, 'style': 'st', 'z': 1}],
                 'members': {'t1': ['l1']}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'method_version': 'test',
                  'objects': [{'id': 'a-badge', 'kind': 'icon', 'bbox': badge_bbox, 'layers': ['l.png'], 'replaces': [],
                               'components': ['badge'], 'parts': [{'name': 'badge', 'bbox': badge_bbox, 'layers': ['l.png']}],
                               'required_owners': [], 'text_policy': 'exclude_independent_text', 'intrinsic_text': '',
                               'description': 'badge', 'prompt_en': 'badge'}]},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [w, h], 'method_version': 'test', 'background': 'F2F8FE', 'shapes': []},
        'ownership': {'t1': 'RESOLVED', 'a-badge': 'RESOLVED'}, 'relations': [],
    })
    ir, _ = build(graph, snap, font_data, tmp_path / 'base', strategy='off', text_candidates=False, surface_candidates=False)
    trials = list(adaptive_candidates(graph, 't1', src_path, None, tmp_path / 'trials', baseline_directory=ir.parent))
    passed = next(t for t in trials if t['status'] == 'PRECHECK_PASS')
    assert passed['provenance'].get('allow_vertical_context') is True
    staged_ir = stage_background(tmp_path / 'trials' / passed['candidate'], graph, ir.parent, font_data, tmp_path / 'staged')
    assert staged_ir.is_file()


def test_residual_gate_alpha_asset_masking_and_occluded_surface_plate_check(tmp_path):
    w, h = 160, 120
    src = np.full((h, w, 3), 245, dtype=np.uint8)
    src[20:100, 20:100] = [215, 232, 250]  # Surface shape s-001
    src[30:90, 110:150] = [190, 210, 240]  # Occluded surface shape s-002
    src_path = tmp_path / 'source.png'
    Image.fromarray(src).save(src_path)

    clean_plate = np.full((h, w, 3), 245, dtype=np.uint8)
    plate_path = tmp_path / 'source-plate.png'
    Image.fromarray(clean_plate).save(plate_path)

    # Higher-z transparent asset a-001 over s-001 (only center circle is opaque, corners transparent)
    a_png = np.zeros((80, 80, 4), dtype=np.uint8)
    yy, xx = np.mgrid[:80, :80]
    circle = ((xx - 40) ** 2 + (yy - 40) ** 2) <= 15 ** 2
    a_png[circle] = [20, 100, 210, 255]
    a_file = tmp_path / 'a-001.png'
    Image.fromarray(a_png).save(a_file)
    a_mask = np.zeros((h, w), dtype=bool)
    a_mask[20:100, 20:100] = circle
    a_proof = tmp_path / 'a-001-owner.npz'
    np.savez_compressed(a_proof, mask=a_mask, restore_bbox=[20, 20, 80, 80], backing_patch=np.full((80, 80, 3), [215, 232, 250], np.uint8))

    s1_mask = np.zeros((h, w), dtype=bool)
    s1_mask[20:100, 20:100] = ~circle
    s1_proof = tmp_path / 's-001-owner.npz'
    np.savez_compressed(s1_proof, mask=s1_mask, restore_bbox=[20, 20, 80, 80], backing_patch=np.full((80, 80, 3), 245, np.uint8))

    s2_mask = np.zeros((h, w), dtype=bool)
    s2_mask[30:90, 110:150] = True
    s2_proof = tmp_path / 's-002-owner.npz'
    np.savez_compressed(s2_proof, mask=s2_mask, restore_bbox=[110, 30, 40, 60], backing_patch=np.full((60, 40, 3), 245, np.uint8))

    scene = {
        'source_sha256': digest(src_path),
        'raster_backing': {'path': 'source-plate.png', 'sha256': digest(plate_path)},
        'selected': {
            's-001': {'id': 's-001', 'kind': 'surface', 'bbox': [20, 20, 80, 80],
                      'owner_evidence': {'file': 's-001-owner.npz', 'sha256': digest(s1_proof)}},
            'a-001': {'id': 'a-001', 'kind': 'asset', 'bbox': [20, 20, 80, 80],
                      'asset_reference': {'path': 'a-001.png', 'bbox': [20, 20, 80, 80]},
                      'owner_evidence': {'file': 'a-001-owner.npz', 'sha256': digest(a_proof)}},
            's-002': {'id': 's-002', 'kind': 'surface', 'bbox': [110, 30, 40, 60],
                      'owner_evidence': {'file': 's-002-owner.npz', 'sha256': digest(s2_proof)}},
            'r-001': {'id': 'r-001', 'kind': 'asset', 'region_crop': True, 'move_safety': 'MOVE_UNSAFE',
                      'bbox': [108, 28, 44, 64]},
        },
    }
    write(tmp_path / 'scene.json', scene)
    write(tmp_path / 'slideir.json', {
        'nodes': [
            {'id': 's-001', 'type': 'shape', 'z': 1, 'bbox': [20, 20, 80, 80]},
            {'id': 's-002', 'type': 'shape', 'z': 2, 'bbox': [110, 30, 40, 60]},
            {'id': 'r-001', 'type': 'image', 'z': 3, 'bbox': [108, 28, 44, 64]},
            {'id': 'a-001', 'type': 'image', 'z': 4, 'bbox': [20, 20, 80, 80]},
        ]
    })

    office_dir = tmp_path / 'office'
    office_dir.mkdir()
    Image.fromarray(clean_plate).save(office_dir / '01-s-001-delete.png')
    Image.fromarray(clean_plate).save(office_dir / '01-s-002-delete.png')
    Image.fromarray(clean_plate).save(office_dir / '01-a-001-delete.png')

    res = inspect_residuals(tmp_path / 'slideir.json', office_dir, 1, tmp_path / 'res-ok.json')
    by_id = {r['id']: r for r in res}
    assert by_id['s-001']['residual_check_status'] == 'PASS'
    assert by_id['s-001']['checked_owner_pixels'] > 0
    assert by_id['s-002']['residual_check_status'] == 'PASS'
    assert 'occluded' in by_id['s-002'].get('reason', '')

    # If plate still holds uncleared source ink under occluded s-002 -> HARD FAIL!
    dirty_plate = clean_plate.copy()
    dirty_plate[40:70, 120:140] = [190, 210, 240]
    Image.fromarray(dirty_plate).save(plate_path)
    scene['raster_backing']['sha256'] = digest(plate_path)
    write(tmp_path / 'scene.json', scene)
    res_fail = inspect_residuals(tmp_path / 'slideir.json', office_dir, 1, tmp_path / 'res-fail.json')
    assert next(r for r in res_fail if r['id'] == 's-002')['residual_check_status'] == 'FAIL'


def test_restage_batch_and_crop_isolates_failing_asset_seeds(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import product.runtime.batch_backbone as bb

    img_p = tmp_path / 'src.png'
    Image.new('RGB', (120, 80), (245, 248, 252)).save(img_p)
    snap = snapshot(img_p, tmp_path / 'snap')
    base_dir = tmp_path / 'base'
    base_dir.mkdir()
    staged_dir = tmp_path / 'staged'
    staged_dir.mkdir()
    base_scene = {
        'schema': 'SceneGraph/1', 'source_sha256': snap['source_sha256'], 'image_size': [120, 80],
        'transform': [1, 0, 0, 1, 0, 0], 'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [120, 80], 'method_version': 'test', 'styles': [], 'texts': [], 'members': {}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [120, 80], 'method_version': 'test', 'objects': [
            {'id': 'a-008', 'kind': 'icon', 'bbox': [10, 10, 20, 20], 'layers': ['l-032'], 'replaces': [], 'components': ['c'], 'parts': [{'name': 'c', 'bbox': [10, 10, 20, 20], 'layers': ['l-032']}], 'required_owners': [], 'text_policy': 'exclude_independent_text', 'intrinsic_text': '', 'description': 'a', 'prompt_en': 'a'},
        ]},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [120, 80], 'method_version': 'test', 'background': 'F5F8FC', 'shapes': []},
        'ownership': {'a-008': 'RESOLVED'}, 'relations': [],
    }
    write(base_dir / 'scene.json', base_scene)
    write(base_dir / 'slideir.json', {'nodes': []})
    staged_scene = {
        **base_scene,
        'selected': {
            'r-008': {'id': 'r-008', 'kind': 'asset', 'bbox': [10, 10, 20, 20], 'region_crop': False, 'seed_ids': ['l-032'], 'move_safety': 'MOVABLE'},
            'r-009': {'id': 'r-009', 'kind': 'asset', 'bbox': [50, 10, 20, 20], 'region_crop': False, 'seed_ids': ['l-033'], 'move_safety': 'MOVABLE'},
        },
    }
    write(staged_dir / 'scene.json', staged_scene)
    write(staged_dir / 'slideir.json', {'nodes': []})

    captured_skips = []

    def fake_chain(**kwargs):
        captured_skips.append(set(kwargs['skip_ids']))
        out_base = Path(kwargs['directory']) / kwargs['tag'] / 'out'
        out_base.mkdir(parents=True, exist_ok=True)
        write(out_base / 'scene.json', {**base_scene, 'selected': {'r-009': staged_scene['selected']['r-009']}})
        write(out_base / 'slideir.json', {'nodes': []})
        return out_base, ['r-009'], []

    monkeypatch.setattr(bb, '_chain_stage_from_spec', fake_chain)
    fonts_p = tmp_path / 'fonts.json'
    write(fonts_p, {'schema': 'FontManifest/1', 'fonts': []})
    fake_svc = SimpleNamespace(page=lambda cur, p: {'snapshot': snap, 'layers': None})
    current = {'fonts': str(fonts_p)}
    cand = {
        'id': 'c-batch', 'page': 1, 'object_id': 'r-008', 'object_ids': ['r-008', 'r-009'],
        'ir': str(staged_dir / 'slideir.json'),
        'batch_spec': {'initial_baseline': str(base_dir), 'cache_dir': str(tmp_path), 'trials': []},
    }
    office_res = {
        'residuals': [[{'id': 'r-008', 'residual_check_status': 'FAIL'}]],
        'replacements': {'records': []},
        'office': {},
    }
    assert bb.restage_batch_candidate_subset(fake_svc, None, current, tmp_path / 'iso1', cand, office_res) is True
    assert {'r-008', 'l-032'} <= captured_skips[0]
    assert cand['object_ids'] == ['r-009']
    assert cand['isolation_attempts'] == 1
    assert {'r-008', 'l-032'} <= set(cand['isolated_skip_ids'])

    # crop_spec isolation should drop failing seed 'layout:l-012' and keep prior skips across passes
    crop_stage = tmp_path / 'crop-staged'
    crop_stage.mkdir()
    crop_scene = {
        **base_scene,
        'selected': {
            'r-011': {'id': 'r-011', 'kind': 'asset', 'bbox': [10, 10, 20, 20], 'region_crop': True, 'seed_ids': ['layout:l-012'], 'move_safety': 'MOVABLE'},
            'r-012': {'id': 'r-012', 'kind': 'asset', 'bbox': [60, 10, 20, 20], 'region_crop': True, 'seed_ids': ['layout:l-013'], 'move_safety': 'MOVABLE'},
        },
    }
    write(crop_stage / 'scene.json', crop_scene)
    write(crop_stage / 'slideir.json', {'nodes': []})
    crop_skips = []

    def fake_stage_crops(src_base, out_dir, *, seeds=None, skip_seed_ids=None, **kwargs):
        crop_skips.append(set(skip_seed_ids or ()))
        out_dir.mkdir(parents=True, exist_ok=True)
        write(out_dir / 'scene.json', {**base_scene, 'selected': {'r-012': crop_scene['selected']['r-012']}})
        ir_f = out_dir / 'slideir.json'
        write(ir_f, {'nodes': []})
        return ir_f, ['r-012'], []

    monkeypatch.setattr('product.recovery.region_crop.stage_region_crops', fake_stage_crops)
    crop_cand = {
        'id': 'c-crop', 'page': 1, 'object_id': 'r-011', 'object_ids': ['r-011', 'r-012'],
        'ir': str(crop_stage / 'slideir.json'),
        'crop_spec': {
            'initial_baseline': str(base_dir),
            'seeds': [{'id': 'layout:l-012'}, {'id': 'layout:l-013'}],
            'protected_text_ids': [],
            'upgrade_containers': False,
        },
    }
    res_crop = {'residuals': [[{'id': 'r-011', 'residual_check_status': 'FAIL'}]], 'replacements': {'records': []}, 'office': {}}
    assert bb.restage_batch_candidate_subset(fake_svc, None, current, tmp_path / 'crop-iso', crop_cand, res_crop) is True
    assert {'r-011', 'layout:l-012'} <= crop_skips[0]
    assert crop_cand['object_ids'] == ['r-012']
    assert crop_cand['isolation_attempts'] == 1


def test_native_frame_edit_headroom_and_peer_clamp(monkeypatch):
    from types import SimpleNamespace
    from product.contracts import Text, Style
    from product.recovery.candidate_stage import native_frame

    text = Text(id='t', text='测试标题', bbox=[20., 20., 60., 20.], z=1, style='body')
    style = Style(id='body', role='body', ea='EA', latin='Latin', size_px=18, bold=False, color='111111')
    monkeypatch.setattr(
        'product.recovery.candidate_stage.ImageFont.truetype',
        lambda *args, **kwargs: SimpleNamespace(
            getbbox=lambda _: [0, 2, 74, 18],
            getlength=lambda s: 74.0 if s == '测试标题' else 110.0,
        ),
    )
    monkeypatch.setattr('product.recovery.candidate_stage.font_line_height', lambda *args: 22.)
    fonts_data = {'fonts': [{'family': family, 'bold': False, 'file': 'synthetic.ttf', 'index': 0} for family in ('EA', 'Latin')]}

    tight = native_frame(text, {'body': style}, fonts_data, (300, 120), np.zeros((120, 300), bool), [15, 15, 90, 30])
    assert tight['bbox'] == [20., 20., 75., 23.]

    wide = native_frame(text, {'body': style}, fonts_data, (300, 120), np.zeros((120, 300), bool), [15, 15, 90, 30], edit_headroom=True)
    # 110 * 1.05 + 6 = 122 > 74 * 1.07 + 6 = 86, height = 22 + 3 = 25
    assert wide['bbox'] == [20., 20., 122., 25.]

    # Peer at x>=110 clamps headroom box width safely below x=110 without triggering conflict
    protected = np.zeros((120, 300), bool)
    protected[:, 110:] = True
    clamped = native_frame(text, {'body': style}, fonts_data, (300, 120), protected, [15, 15, 90, 30], edit_headroom=True)
    assert 74 <= clamped['bbox'][2] <= 90
    assert clamped['bbox'][0] + clamped['bbox'][2] <= 110


def test_verify_pictures_cache_fastpath_and_mutation_detection(tmp_path):
    import zipfile, pytest
    from product.parallel_office import verify_pictures

    rgba = np.zeros((6, 8, 4), np.uint8)
    rgba[1:5, 2:7] = [25, 120, 210, 200]
    asset = tmp_path / 'icon.png'
    Image.fromarray(rgba).save(asset)
    ir = tmp_path / 'ir.json'
    write(ir, {'assets': [{'id': 'icon', 'path': 'icon.png', 'sha256': digest(asset)}],
               'nodes': [{'id': 'logo', 'type': 'image', 'asset_id': 'icon'}]})
    pptx = tmp_path / 'fixture.pptx'
    xml = ('<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
           'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
           '<p:pic><p:nvPicPr><p:cNvPr name="logo"/></p:nvPicPr>'
           '<p:blipFill><a:blip r:embed="rId1"/></p:blipFill></p:pic></p:sld>')
    with zipfile.ZipFile(pptx, 'w') as z:
        z.writestr('ppt/slides/slide1.xml', xml)
        z.writestr('ppt/slides/_rels/slide1.xml.rels', '<Relationships><Relationship Id="rId1" Target="../media/icon.png"/></Relationships>')
        z.write(asset, 'ppt/media/icon.png')

    pic_cache = {}
    verify_pictures(pptx, [ir], cache=pic_cache)
    assert (1, 'logo') in pic_cache['verified_media_sha']
    verify_pictures(pptx, [ir], cache=pic_cache)
    verify_pictures(pptx, [ir], deleted=(1, 'logo'), cache=pic_cache)

    mutated = tmp_path / 'mutated.pptx'
    rgba2 = rgba.copy()
    rgba2[2, 3, 0] = 200
    bad_asset = tmp_path / 'bad.png'
    Image.fromarray(rgba2).save(bad_asset)
    with zipfile.ZipFile(mutated, 'w') as z:
        z.writestr('ppt/slides/slide1.xml', xml)
        z.writestr('ppt/slides/_rels/slide1.xml.rels', '<Relationships><Relationship Id="rId1" Target="../media/icon.png"/></Relationships>')
        z.write(bad_asset, 'ppt/media/icon.png')
    with pytest.raises(ValueError):
        verify_pictures(mutated, [ir], cache=pic_cache)


def test_runner_deadline_preserves_min_agent_window_for_stale_baseline(tmp_path, monkeypatch):
    import product.runtime.runner as runner_mod
    from product.runtime.state import RuntimeState

    monkeypatch.setattr('time.time', lambda: 10000.0)
    monkeypatch.setattr('time.monotonic', lambda: 0.0)

    def _run_case(name, runtime_s, started_age):
        job = tmp_path / name
        (job / 'inputs').mkdir(parents=True)
        Image.new('RGB', (40, 30), 'white').save(job / 'inputs/1.png')
        write(job / 'request.json', {'sources': ['1.png'], 'runtime_seconds': runtime_s, 'host_stage0': True})
        write(job / 'host-baseline/started.json', {'time': 10000.0 - started_age})
        ws = job / 'agent'
        write(ws / 'opencode.json', {'providers': {'p': {}}})
        RuntimeState(job)
        monkeypatch.setattr(runner_mod, 'prepare_workspace', lambda _: ws)
        monkeypatch.setattr(runner_mod, 'wait_for_mcp', lambda *a, **k: True)

        class DummyClient:
            def verify(self):
                return {'version': runner_mod.PINNED_VERSION}
            def request(self, method, route, **kwargs):
                if route.endswith('/wait'):
                    write(job / 'result.json', {'status': 'REVIEW_REQUIRED'})
                    return {}
                return {}
        monkeypatch.setattr(RuntimeState, 'read', lambda self: {'status': 'REVIEW_REQUIRED' if (job / 'result.json').exists() else 'RUNNING'})
        runner_mod._run(job, DummyClient())
        return json.loads((job / 'runtime/session.json').read_text('utf-8'))

    s1 = _run_case('j1', runtime_s=1800, started_age=1600)
    assert s1['deadline'] == 10000.0 + 900
    s2 = _run_case('j2', runtime_s=0, started_age=100)
    assert s2['deadline'] == 10000.0
    s3 = _run_case('j3', runtime_s=1800, started_age=100)
    assert s3['deadline'] == 10000.0 - 100 + 1800


def test_inspect_residuals_target_ids_preserves_higher_z_sibling_masking(tmp_path):
    w, h = 160, 120
    src = np.full((h, w, 3), 245, dtype=np.uint8)
    src[20:100, 20:100] = [215, 232, 250]
    src[30:90, 110:150] = [190, 210, 240]
    yy, xx = np.mgrid[:80, :80]
    circle = ((xx - 40) ** 2 + (yy - 40) ** 2) <= 15 ** 2
    a_png = np.zeros((80, 80, 4), dtype=np.uint8)
    a_png[circle] = [20, 100, 210, 255]
    src_patch = src[20:100, 20:100].copy()
    src_patch[circle] = [20, 100, 210]
    src[20:100, 20:100] = src_patch
    Image.fromarray(src).save(tmp_path / 'source.png')

    a_file = tmp_path / 'a-001.png'
    Image.fromarray(a_png).save(a_file)
    plate = np.full((h, w, 3), 245, dtype=np.uint8)
    Image.fromarray(plate).save(tmp_path / 'source-plate.png')

    s1_mask = np.zeros((h, w), dtype=bool)
    s1_mask[20:100, 20:100] = True
    s1_proof = tmp_path / 's-001-owner.npz'
    np.savez_compressed(s1_proof, mask=s1_mask, restore_bbox=[20, 20, 80, 80], backing_patch=np.full((80, 80, 3), 245, np.uint8))
    s2_mask = np.zeros((h, w), dtype=bool)
    s2_mask[30:90, 110:150] = True
    s2_proof = tmp_path / 's-002-owner.npz'
    np.savez_compressed(s2_proof, mask=s2_mask, restore_bbox=[110, 30, 40, 60], backing_patch=np.full((60, 40, 3), 245, np.uint8))

    scene = {
        'schema': 'SceneGraph/1', 'source_sha256': digest(tmp_path / 'source.png'), 'image_size': [w, h],
        'transform': [1, 0, 0, 1, 0, 0], 'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': digest(tmp_path / 'source.png'), 'image_size': [w, h], 'method_version': 'test', 'styles': [], 'texts': [], 'members': {}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': digest(tmp_path / 'source.png'), 'image_size': [w, h], 'method_version': 'test', 'objects': []},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': digest(tmp_path / 'source.png'), 'image_size': [w, h], 'method_version': 'test', 'background': 'F5F5F5', 'shapes': []},
        'ownership': {'s-001': 'RESOLVED', 's-002': 'RESOLVED', 'a-001': 'RESOLVED'}, 'relations': [],
        'raster_backing': {'path': 'source-plate.png', 'sha256': digest(tmp_path / 'source-plate.png')},
        'selected': {
            's-001': {'id': 's-001', 'kind': 'surface', 'bbox': [20, 20, 80, 80], 'owner_evidence': {'file': 's-001-owner.npz', 'sha256': digest(s1_proof)}},
            's-002': {'id': 's-002', 'kind': 'surface', 'bbox': [110, 30, 40, 60], 'owner_evidence': {'file': 's-002-owner.npz', 'sha256': digest(s2_proof)}},
            'a-001': {'id': 'a-001', 'kind': 'asset', 'bbox': [20, 20, 80, 80], 'asset_reference': {'path': 'a-001.png', 'bbox': [20, 20, 80, 80]}},
        },
    }
    write(tmp_path / 'scene.json', scene)
    write(tmp_path / 'slideir.json', {
        'nodes': [
            {'id': 's-001', 'type': 'shape', 'z': 1, 'bbox': [20, 20, 80, 80]},
            {'id': 's-002', 'type': 'shape', 'z': 2, 'bbox': [110, 30, 40, 60]},
            {'id': 'a-001', 'type': 'image', 'z': 4, 'bbox': [20, 20, 80, 80]},
        ]
    })
    office_dir = tmp_path / 'office'
    office_dir.mkdir()
    del_s1 = plate.copy()
    del_s1[20:100, 20:100][circle] = [20, 100, 210]  # higher-z a-001 opaque pixels remain rendered above s-001
    Image.fromarray(del_s1).save(office_dir / '01-s-001-delete.png')

    res = inspect_residuals(tmp_path / 'slideir.json', office_dir, 1, tmp_path / 'res-scoped.json', target_ids={'s-001'})
    assert [r['id'] for r in res] == ['s-001']
    assert res[0]['residual_check_status'] == 'PASS'


def test_validate_candidate_scopes_incremental_office_check_and_keeps_full_bundle_unscoped(tmp_path, monkeypatch):
    from product.runtime.service import RuntimeService
    from product.runtime.state import RuntimeState

    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT', str(tmp_path / 'leases'))
    jobs_root = tmp_path / 'jobs'
    job_dir = jobs_root / 'aabbccdd'
    (job_dir / 'inputs').mkdir(parents=True)
    src_p = job_dir / 'inputs/1.png'
    Image.new('RGB', (80, 60), (240, 244, 248)).save(src_p)
    snap = snapshot(src_p, job_dir / 'snap')
    write(job_dir / 'request.json', {'sources': ['1.png'], 'orchestrator': 'opencode'})
    fonts_p = job_dir / 'fonts.json'
    write(fonts_p, {'schema': 'FontManifest/1', 'fonts': []})

    base_scene = {
        'schema': 'SceneGraph/1', 'source_sha256': snap['source_sha256'], 'image_size': [80, 60],
        'transform': [1, 0, 0, 1, 0, 0], 'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [80, 60], 'method_version': 'test',
                 'styles': [{'id': 'st', 'role': 'body', 'ea': 'Arial', 'latin': 'Arial', 'size_px': 12.0, 'bold': False, 'color': '000000'}],
                 'texts': [{'id': 't-001', 'text': 'hi', 'bbox': [10.0, 10.0, 20.0, 12.0], 'style': 'st', 'z': 1}],
                 'members': {'t-001': ['l1']}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [80, 60], 'method_version': 'test', 'objects': []},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [80, 60], 'method_version': 'test', 'background': 'F0F4F8', 'shapes': []},
        'ownership': {'t-001': 'RESOLVED'}, 'relations': [],
    }
    scene_b = {
        **base_scene,
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [80, 60], 'method_version': 'test', 'objects': [
            {'id': 'r-new', 'kind': 'icon', 'bbox': [40, 10, 10, 10], 'layers': ['l.png'], 'replaces': [], 'components': ['c'],
             'parts': [{'name': 'c', 'bbox': [40, 10, 10, 10], 'layers': ['l.png']}], 'required_owners': [],
             'text_policy': 'exclude_independent_text', 'intrinsic_text': '', 'description': 'r', 'prompt_en': 'r'},
        ]},
        'ownership': {'t-001': 'RESOLVED', 'r-new': 'RESOLVED'},
    }

    calls = []

    def fake_compile(irs, out_dir, texts):
        out_dir.mkdir(parents=True, exist_ok=True)
        pptx_p = out_dir / 'out.pptx'
        pptx_p.write_bytes(b'fake-pptx-' + str(len(calls)).encode())
        return {'pptx': str(pptx_p)}

    def fake_office_check(state, pptx, irs, directory, *, target_ids=None):
        calls.append(target_ids)
        return {'status': 'PASS', 'files': {}, 'target_ids': target_ids}

    monkeypatch.setattr('product.runtime.compiler.compile_local', fake_compile)
    monkeypatch.setattr(RuntimeService, 'office_check', staticmethod(fake_office_check))

    svc = RuntimeService(jobs_root)
    state = RuntimeState(job_dir)

    # Case A: original.selected is empty -> full-page unscoped check (target_ids=None) and sets validated_bundle
    b1 = job_dir / 'b1'
    c1 = job_dir / 'c1'
    b1.mkdir(); c1.mkdir()
    write(b1 / 'scene.json', {**base_scene, 'selected': {}})
    write(b1 / 'slideir.json', {'nodes': []})
    proof = c1 / 't1.npz'
    np.savez_compressed(proof, mask=np.zeros((60, 80), bool), restore_bbox=[0, 0, 10, 10], backing_patch=np.zeros((10, 10, 3), np.uint8))
    write(c1 / 'scene.json', {**base_scene, 'selected': {'t-001': {'id': 't-001', 'kind': 'text', 'owner_evidence': {'file': 't1.npz', 'sha256': digest(proof)}}}})
    write(c1 / 'slideir.json', {'nodes': []})
    cur1 = {
        'fonts': str(fonts_p),
        'pages': [{'scene': str(b1 / 'scene.json'), 'ir': str(b1 / 'slideir.json'), 'snapshot': snap}],
        'candidates': {'cand-1': {
            'id': 'cand-1', 'page': 1, 'object_id': 't-001', 'object_ids': ['t-001'], 'kind': 'batch',
            'status': 'STAGED_LOCAL', 'ir': str(c1 / 'slideir.json'),
            'parent_ir_sha256': digest(b1 / 'slideir.json'), 'parent_scene_sha256': digest(b1 / 'scene.json'),
            'code': svc.code_binding(), 'files': {},
        }},
    }
    svc.validate_candidate(state, cur1, job_dir / 'val1', candidate_id='cand-1')
    assert calls[-1] is None
    assert 'validated_bundle' in cur1['candidates']['cand-1']

    # Case B: original.selected already has t-001, candidate adds r-new -> scoped check ['r-new'], no validated_bundle
    b2 = job_dir / 'b2'
    c2 = job_dir / 'c2'
    b2.mkdir(); c2.mkdir()
    proof2 = b2 / 't1.npz'
    np.savez_compressed(proof2, mask=np.zeros((60, 80), bool), restore_bbox=[0, 0, 10, 10], backing_patch=np.zeros((10, 10, 3), np.uint8))
    sel_t1 = {'id': 't-001', 'kind': 'text', 'owner_evidence': {'file': 't1.npz', 'sha256': digest(proof2)}}
    write(b2 / 'scene.json', {**scene_b, 'selected': {'t-001': sel_t1}})
    write(b2 / 'slideir.json', {'nodes': []})
    asset_p = c2 / 'r-new.png'
    Image.new('RGBA', (10, 10), (0, 0, 0, 255)).save(asset_p)
    sel_r = {'id': 'r-new', 'kind': 'asset', 'region_crop': True, 'move_safety': 'MOVE_UNSAFE',
             'asset_reference': {'path': 'r-new.png', 'sha256': digest(asset_p)}}
    write(c2 / 'scene.json', {**scene_b, 'selected': {'t-001': sel_t1, 'r-new': sel_r}})
    write(c2 / 'slideir.json', {'nodes': []})
    cur2 = {
        'fonts': str(fonts_p),
        'pages': [{'scene': str(b2 / 'scene.json'), 'ir': str(b2 / 'slideir.json'), 'snapshot': snap}],
        'candidates': {'cand-2': {
            'id': 'cand-2', 'page': 1, 'object_id': 'r-new', 'object_ids': ['r-new'], 'kind': 'batch',
            'status': 'STAGED_LOCAL', 'ir': str(c2 / 'slideir.json'),
            'parent_ir_sha256': digest(b2 / 'slideir.json'), 'parent_scene_sha256': digest(b2 / 'scene.json'),
            'code': svc.code_binding(), 'files': {},
        }},
    }
    svc.validate_candidate(state, cur2, job_dir / 'val2', candidate_id='cand-2')
    assert calls[-1] == ['r-new']
    assert 'validated_bundle' not in cur2['candidates']['cand-2']


def test_office_queue_cancels_queued_tickets_for_completed_jobs(tmp_path, monkeypatch):
    from product.runtime.office_bridge import OfficeQueue

    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT', str(tmp_path / 'leases'))
    root = tmp_path / 'jobs'
    for identifier in ('a' * 32, 'b' * 32):
        write(root / identifier / 'runtime/o/check/worker-state.json', {'status': 'QUEUED'})
    write(root / ('a' * 32) / 'result.json', {'status': 'REVIEW_REQUIRED'})
    ticket = OfficeQueue(root).claim()
    assert ticket['ticket'].startswith('b' * 32 + '/')
    first = json.loads((root / ('a' * 32) / 'runtime/o/check/worker-state.json').read_text('utf-8'))
    assert first['status'] == 'CANCELLED'


def test_deepseek_provider_alias_workspace_and_redaction(tmp_path, monkeypatch):
    from product.runtime.provider import (
        DEEPSEEK_BASE_URL, MOMA_BASE_URL, all_secret_env_values, configure_model, provider_id,
    )
    from product.runtime.runner import prepare_workspace

    for name in (
        'PLANNER_API_KEY', 'PLANNER_BASE_URL', 'PLANNER_MODEL',
        'OPENCODE_GO_API_KEY', 'MOMA_API_KEY', 'MOMA_API_KEYS',
        'MOMA_API_KEY_BACKUP', 'MOMA_API_KEY_2', 'IMAGE2PPT_AGENT_API_KEY',
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('IMAGE2PPT_AGENT_PROVIDER', 'deepseek')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-synthetic-ds-key')
    monkeypatch.setenv('MOMA_API_KEYS', 'moma-k1\nmoma-k2')
    monkeypatch.setenv('IMAGE2PPT_AGENT_MODEL', 'deepseek-v4.1-flash')

    job = tmp_path / ('d' * 32)
    (job / 'inputs').mkdir(parents=True)
    write(job / 'request.json', {
        'sources': ['01.png'],
        'agent_inference_provider': 'deepseek',
        'agent_model': 'deepseek-v4.1-flash',
        'agent_steps': 32,
        'host_stage0': True,
    })

    workspace = prepare_workspace(job)
    encoded = (workspace / 'opencode.json').read_text('utf-8')
    cfg = json.loads(encoded)
    pid = provider_id('deepseek')
    assert pid == 'image2ppt-deepseek'
    prov = cfg['providers'][pid]
    assert prov['settings']['baseURL'] == DEEPSEEK_BASE_URL
    assert prov['env'] == ['DEEPSEEK_API_KEY', 'IMAGE2PPT_AGENT_API_KEY']
    assert prov['body'] == {'thinking': {'type': 'disabled'}}
    assert prov['models']['vision']['modelID'] == 'deepseek-flash'
    assert 'sk-synthetic-ds-key' not in encoded
    assert 'moma-k1' not in encoded

    secrets = set(all_secret_env_values())
    assert {'sk-synthetic-ds-key', 'moma-k1', 'moma-k2'} <= secrets

    # Switching back to MoMA must restore PLANNER_BASE_URL to MoMA rather than leaving DeepSeek URL.
    moma_job = tmp_path / ('e' * 32)
    write(moma_job / 'request.json', {'sources': ['01.png'], 'agent_inference_provider': 'moma'})
    monkeypatch.delenv('IMAGE2PPT_AGENT_MODEL', raising=False)
    sel, settings_obj = configure_model(moma_job)
    assert sel == 'moma'
    assert settings_obj.base_url == MOMA_BASE_URL
    assert settings_obj.api_key == 'moma-k1'


def test_moma_dual_key_failover_and_cross_provider_review(tmp_path, monkeypatch):
    import httpx
    import pytest
    from pydantic import BaseModel
    from product.runtime.provider import (
        DEEPSEEK_BASE_URL, MOMA_BASE_URL, OpenAICompatReviewer, resolve_review_config,
    )

    class _Reply(BaseModel):
        ok: bool

    seen_auth: list[str] = []

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def post(self, url, *, json=None, headers=None):
            seen_auth.append(headers['Authorization'])
            req = httpx.Request('POST', url)
            if headers['Authorization'].endswith('key-one'):
                return httpx.Response(429, request=req, text='rate limit')
            return httpx.Response(
                200,
                request=req,
                json={
                    'model': 'qwen/qwen3.8-max',
                    'usage': {'total_tokens': 12},
                    'choices': [{'finish_reason': 'stop', 'message': {'content': '{"ok": true}'}}],
                },
            )

    monkeypatch.setattr(httpx, 'Client', _FakeClient)
    rev = OpenAICompatReviewer(
        provider='moma',
        base_url=MOMA_BASE_URL,
        api_keys=['key-one', 'key-two'],
        model='qwen/qwen3.8-max',
        session='sess-1',
    )
    out = rev.ask('sys', {'k': 1}, [], _Reply)
    assert out == {'ok': True}
    assert seen_auth == ['Bearer key-one', 'Bearer key-two']
    assert rev.reported_model == 'qwen/qwen3.8-max'

    for name in ('PLANNER_API_KEY', 'PLANNER_BASE_URL', 'PLANNER_MODEL', 'OPENCODE_GO_API_KEY',
                 'IMAGE2PPT_REVIEW_PROVIDER', 'IMAGE2PPT_REVIEW_MODEL', 'IMAGE2PPT_AGENT_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-ds-test')
    monkeypatch.setenv('MOMA_API_KEY', 'mm-test-1')
    monkeypatch.setenv('MOMA_API_KEYS', 'mm-test-1,mm-test-2')
    monkeypatch.setenv('IMAGE2PPT_AGENT_PROVIDER', 'deepseek')
    monkeypatch.setenv('IMAGE2PPT_AGENT_MODEL', 'deepseek-v4.1-flash')

    job_ds = tmp_path / ('f' * 32)
    write(job_ds / 'request.json', {'sources': ['01.png'], 'agent_inference_provider': 'deepseek', 'agent_model': 'deepseek-v4.1-flash'})
    r_prov, r_url, r_keys, r_model = resolve_review_config(job_ds)
    assert (r_prov, r_url, r_keys, r_model) == ('moma', MOMA_BASE_URL, ['mm-test-1', 'mm-test-2'], 'qwen/qwen3.8-max')

    job_mm = tmp_path / ('1' * 32)
    write(job_mm / 'request.json', {'sources': ['01.png'], 'agent_inference_provider': 'moma', 'agent_model': 'qwen/qwen3.8-max'})
    r2_prov, r2_url, r2_keys, r2_model = resolve_review_config(job_mm)
    assert (r2_prov, r2_url, r2_keys, r2_model) == ('deepseek', DEEPSEEK_BASE_URL, ['sk-ds-test'], 'deepseek-flash')

    monkeypatch.setenv('IMAGE2PPT_REVIEW_PROVIDER', 'deepseek')
    monkeypatch.setenv('IMAGE2PPT_REVIEW_MODEL', 'deepseek-flash')
    with pytest.raises(ValueError, match='differ'):
        resolve_review_config(job_ds)


def test_scoped_target_ids_and_sibling_residual_attribution(tmp_path, monkeypatch):
    from product.runtime.batch_backbone import _collect_office_defects
    from product.runtime.service import RuntimeService
    from product.runtime.state import RuntimeState

    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT', str(tmp_path / 'leases'))
    jobs_root = tmp_path / 'jobs'
    job_dir = jobs_root / ('2' * 32)
    (job_dir / 'inputs').mkdir(parents=True)
    src_p = job_dir / 'inputs/1.png'
    Image.new('RGB', (1200, 600), (245, 245, 245)).save(src_p)
    snap = snapshot(src_p, job_dir / 'snap')
    write(job_dir / 'request.json', {'sources': ['1.png'], 'orchestrator': 'opencode'})
    fonts_p = job_dir / 'fonts.json'
    write(fonts_p, {'schema': 'FontManifest/1', 'fonts': []})

    def _asset_stub(oid, box):
        return {
            'id': oid, 'kind': 'icon', 'bbox': box, 'layers': ['l.png'], 'replaces': [],
            'components': ['c'], 'parts': [{'name': 'c', 'bbox': box, 'layers': ['l.png']}],
            'required_owners': [], 'text_policy': 'exclude_independent_text',
            'intrinsic_text': '', 'description': oid, 'prompt_en': oid,
        }

    base_scene = {
        'schema': 'SceneGraph/1', 'source_sha256': snap['source_sha256'], 'image_size': [1200, 600],
        'transform': [1, 0, 0, 1, 0, 0], 'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [1200, 600],
                 'method_version': 'test', 'styles': [], 'texts': [], 'members': {}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [1200, 600],
                  'method_version': 'test', 'objects': [
                      _asset_stub('r-001', [20, 20, 40, 30]),
                      _asset_stub('r-005', [1068, 376, 41, 47]),
                      _asset_stub('r-019', [1047, 484, 84, 77]),
                  ]},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [1200, 600],
                    'method_version': 'test', 'background': 'F5F5F5', 'shapes': [
                        {'id': 's-006', 'bbox': [1023, 358, 61, 105], 'z': 1, 'geometry': 'rect', 'fill': 'FFFFFF'},
                    ]},
        'ownership': {'r-001': 'RESOLVED', 'r-005': 'RESOLVED', 'r-019': 'RESOLVED', 's-006': 'RESOLVED'},
        'relations': [],
    }
    bdir = job_dir / 'b'
    cdir = job_dir / 'c'
    bdir.mkdir(); cdir.mkdir()
    r1_proof_b = bdir / 'r1.npz'
    r1_proof_c = cdir / 'r1.npz'
    r5_proof_b = bdir / 'r5.npz'
    r5_proof_c = cdir / 'r5.npz'
    r19_proof = cdir / 'r19.npz'
    s6_proof = cdir / 's6.npz'
    for pf in (r1_proof_b, r1_proof_c, r5_proof_b, r5_proof_c, r19_proof, s6_proof):
        np.savez_compressed(pf, mask=np.zeros((600, 1200), bool), restore_bbox=[0, 0, 10, 10], backing_patch=np.zeros((10, 10, 3), np.uint8))

    sel_r5 = {'id': 'r-005', 'kind': 'asset', 'bbox': [1068, 376, 41, 47], 'restore_bbox': [1068, 376, 41, 47],
              'owner_evidence': {'file': 'r5.npz', 'sha256': digest(r5_proof_b)}}
    # Distant committed object at [20, 20, 40, 30] must remain excluded from scoped targets.
    sel_r1 = {'id': 'r-001', 'kind': 'asset', 'bbox': [20, 20, 40, 30], 'restore_bbox': [20, 20, 40, 30],
              'owner_evidence': {'file': 'r1.npz', 'sha256': digest(r1_proof_b)}}
    sel_r1_c = {**sel_r1, 'owner_evidence': {'file': 'r1.npz', 'sha256': digest(r1_proof_c)}}
    sel_r5_c = {**sel_r5, 'owner_evidence': {'file': 'r5.npz', 'sha256': digest(r5_proof_c)}}
    sel_r19 = {'id': 'r-019', 'kind': 'asset', 'bbox': [1047, 484, 84, 77], 'restore_bbox': [1047, 484, 84, 77],
               'owner_evidence': {'file': 'r19.npz', 'sha256': digest(r19_proof)}}
    sel_s6 = {'id': 's-006', 'kind': 'surface', 'bbox': [1023, 358, 61, 105], 'restore_bbox': [1023, 358, 61, 105],
              'owner_evidence': {'file': 's6.npz', 'sha256': digest(s6_proof)}}

    write(bdir / 'scene.json', {**base_scene, 'selected': {'r-001': sel_r1, 'r-005': sel_r5}})
    write(bdir / 'slideir.json', {'nodes': []})
    cand_scene_dict = {**base_scene,
                       'selected': {'r-001': sel_r1_c, 'r-005': sel_r5_c, 'r-019': sel_r19, 's-006': sel_s6}}
    write(cdir / 'scene.json', cand_scene_dict)
    write(cdir / 'slideir.json', {'nodes': []})

    seen_targets = []
    monkeypatch.setattr('product.runtime.compiler.compile_local', lambda irs, out, texts: (out.mkdir(parents=True, exist_ok=True), (out / 'd.pptx').write_bytes(b'x'), {'pptx': str(out / 'd.pptx')})[2])
    def _fake_check(state, pptx, irs, directory, *, target_ids=None):
        seen_targets.append(target_ids)
        return {'status': 'PASS', 'files': {}, 'target_ids': target_ids}
    monkeypatch.setattr(RuntimeService, 'office_check', staticmethod(_fake_check))

    svc = RuntimeService(jobs_root)
    cur = {
        'fonts': str(fonts_p),
        'pages': [{'scene': str(bdir / 'scene.json'), 'ir': str(bdir / 'slideir.json'), 'snapshot': snap}],
        'candidates': {'cand-s6': {
            'id': 'cand-s6', 'page': 1, 'object_id': 'r-019', 'object_ids': ['r-019', 's-006'], 'kind': 'batch',
            'status': 'STAGED_LOCAL', 'ir': str(cdir / 'slideir.json'),
            'parent_ir_sha256': digest(bdir / 'slideir.json'), 'parent_scene_sha256': digest(bdir / 'scene.json'),
            'code': svc.code_binding(), 'files': {},
        }},
    }
    svc.validate_candidate(RuntimeState(job_dir), cur, job_dir / 'val', candidate_id='cand-s6')
    # r-005 overlaps newly staged s-006 -> must be checked; distant r-001 must remain excluded.
    assert seen_targets == [['r-005', 'r-019', 's-006']]

    cand_graph = SceneGraph.model_validate(cand_scene_dict)
    office_res = {
        'residuals': [[
            {'id': 'r-005', 'residual_check_status': 'FAIL', 'residual_pixels_confirmed': 4},
            {'id': 'r-019', 'residual_check_status': 'PASS'},
            {'id': 's-006', 'residual_check_status': 'PASS'},
        ]],
        'replacements': {'records': []},
        'office': {},
    }
    _, bad_ids = _collect_office_defects(office_res, cand_graph, ['r-019', 's-006'])
    assert bad_ids == {'s-006'}


def test_scoped_ids_error_lists_invalid_ids(tmp_path):
    import pytest
    from product.runtime.host_evidence import normalize_arguments, scoped_ids
    from product.runtime.state import RuntimeState

    job = tmp_path / ('3' * 32)
    job.mkdir()
    state = RuntimeState(job)
    write(job / 'marks.json', {'rows': [{'mark': 1, 'id': 't-001'}, {'mark': 43, 'id': 'r-008'}]})
    write(job / 'issues.json', {'items': [{'mark': 1, 'id': 't-001'}]})
    ref = state.register(job / 'issues.json', 'numbered_questions')['id']
    cur = {'experiment_marks': {'1': str(job / 'marks.json')}, 'questions': {'1': [{'block': 1, 'questions': ref}]}}

    with pytest.raises(ValueError, match='r-008'):
        scoped_ids(state, cur, 1, None, ['t-001', 'r-008'])
    with pytest.raises(ValueError, match=r'mark 43 \(r-008\)'):
        normalize_arguments(state, cur, 'plan_marked_page', {
            'page': 1,
            'fields': [
                {'mark': 1, 'route': 'text', 'role': 'title', 'move_with': []},
                {'mark': 43, 'route': 'asset', 'role': 'icon', 'move_with': []},
            ],
        })


def test_openai_compat_reviewer_injects_json_hint_when_missing():
    from pydantic import BaseModel
    from product.recovery.container_stage import _ring_parent_bg
    from product.recovery.marked_surfaces import stage_marked_surfaces
    from product.runtime.provider import OpenAICompatReviewer

    assert callable(stage_marked_surfaces)

    class _Out(BaseModel):
        ok: bool

    rev = OpenAICompatReviewer(
        provider='deepseek',
        base_url='https://api.deepseek.com/v1',
        api_keys=['k-dummy'],
        model='deepseek-flash',
        session='s-1',
    )
    b1 = rev._build_body('Transcribe glyphs strictly.', {}, [], _Out)
    assert 'json' in b1['messages'][0]['content'].lower()
    exact_sys = 'Classify rows and return strict JSON.'
    b2 = rev._build_body(exact_sys, {}, [], _Out)
    assert b2['messages'][0]['content'] == exact_sys

    canvas = np.full((60, 80, 3), 240, dtype=np.uint8)
    canvas[20:40, 20:50] = [20, 90, 180]
    ring = _ring_parent_bg(canvas, [20, 20, 30, 20])
    assert ring is not None and ring[1] <= 1.0 and np.allclose(ring[0], [240, 240, 240])


def test_check_local_retries_transient_com_error_only(tmp_path, monkeypatch):
    import product.parallel_edit_probe as edit_probe_mod
    import product.parallel_office as office_mod
    import product.parallel_residual_gate as res_mod
    import product.runtime.office as rt_office

    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT', str(tmp_path / 'leases'))
    monkeypatch.setattr(rt_office.time, 'sleep', lambda _s: None)

    pptx = tmp_path / 'deck.pptx'
    pptx.write_bytes(b'pptx-fixture')
    ir = tmp_path / 'slideir.json'
    write(ir, {'nodes': [{'id': 's-001', 'type': 'shape'}]})

    class com_error(Exception):
        def __init__(self, msg='(-2146959355, "Server execution failed")'):
            super().__init__(msg)
            self.hresult = -2146959355

    attempts = {'n': 0}

    def _flaky_inspect(p, irs, out, *, render_only=False, target_ids=None):
        attempts['n'] += 1
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'partial.txt').write_text('stale', encoding='utf-8')
        if attempts['n'] == 1:
            raise com_error()
        return {'status': 'PASS_APP_AUTOMATED', 'pptx_sha256': digest(pptx), 'records': []}

    monkeypatch.setattr(office_mod, 'inspect', _flaky_inspect)
    monkeypatch.setattr(edit_probe_mod, 'inspect_text', lambda *a, **kw: {'status': 'NOT_APPLICABLE'})
    monkeypatch.setattr(res_mod, 'inspect_residuals', lambda *a, **kw: [])
    monkeypatch.setattr(res_mod, 'residual_accepted', lambda r: True)

    res1 = rt_office.check_local(pptx, [ir], tmp_path / 'run1')
    assert attempts['n'] == 2
    assert res1['status'] == 'PASS'

    # Non-transient OfficeValidationError must NOT be retried.
    val_attempts = {'n': 0}

    def _validation_fail(p, irs, out, *, render_only=False, target_ids=None):
        val_attempts['n'] += 1
        raise office_mod.OfficeValidationError('s-001', 'outside pixels changed')

    monkeypatch.setattr(office_mod, 'inspect', _validation_fail)
    res2 = rt_office.check_local(pptx, [ir], tmp_path / 'run2')
    assert val_attempts['n'] == 1
    assert res2['status'] == 'REVIEW_REQUIRED'
    assert res2['office']['failed_object_id'] == 's-001'


def test_host_baseline_allows_newly_converted_native_text_font_shift_but_rejects_whole_page_regression():
    from product.runtime.host_baseline import regressions

    row_base = {
        'page': 1,
        'source_sha256': 'src-sha',
        'editable_count': 25,
        'ocr_native_hits': [],
        'native_field_count': 0,
        'native_text_regions': [],
        'passed_edits': ['r-001/delete', 'r-001/move-scale'],
        'local_fidelity': {
            'whole-page': {'mae_rgb': 0.5, 'ssim': 0.99, 'delta_e2000_p95': 1.0},
            't-005': {'mae_rgb': 0.6, 'ssim': 0.999, 'delta_e2000_p95': 1.1},
            'l-003': {'mae_rgb': 0.5, 'ssim': 0.999, 'delta_e2000_p95': 1.1},
        },
    }
    row_cand = {
        **row_base,
        'editable_count': 38,
        'ocr_native_hits': ['l-003'],
        'native_field_count': 13,
        'native_text_regions': ['l-003', 't-005'],
        'local_fidelity': {
            'whole-page': {'mae_rgb': 4.1, 'ssim': 0.926, 'delta_e2000_p95': 2.64},
            't-005': {'mae_rgb': 58.6, 'ssim': 0.067, 'delta_e2000_p95': 54.4},
            'l-003': {'mae_rgb': 50.6, 'ssim': 0.115, 'delta_e2000_p95': 53.6},
        },
    }
    base = {'verified': True, 'pages': [row_base]}
    cand = {'verified': True, 'pages': [row_cand]}
    assert regressions(base, cand) == []

    bad_whole = {'verified': True, 'pages': [{
        **row_cand,
        'local_fidelity': {
            **row_cand['local_fidelity'],
            'whole-page': {'mae_rgb': 45.0, 'ssim': 0.45, 'delta_e2000_p95': 35.0},
        },
    }]}
    assert regressions(base, bad_whole) == ['p1:LOCAL_VISUAL_REGRESSION:whole-page']


def test_disjoint_batch_partition_and_verify_pictures_set(tmp_path):
    from zipfile import ZipFile
    from product.parallel_office import _partition_disjoint_batches, _px_window, verify_pictures

    w0 = _px_window([(10, 10, 40, 20)], 1.0, 1.0, 400, 300, pad_px=16)
    w1 = _px_window([(200, 10, 40, 20)], 1.0, 1.0, 400, 300, pad_px=16)
    w2 = _px_window([(42, 12, 30, 20)], 1.0, 1.0, 400, 300, pad_px=16)  # overlaps w0 with pad=16
    batches = _partition_disjoint_batches(['a', 'b', 'c'], {'a': w0, 'b': w1, 'c': w2}, max_batch_size=8)
    assert batches == [['a', 'b'], ['c']]

    # verify_pictures accepts a set of deleted (page, nid) pairs alongside cache
    a_png = tmp_path / 'a1.png'
    Image.new('RGBA', (16, 16), (10, 20, 30, 255)).save(a_png)
    b_png = tmp_path / 'a2.png'
    Image.new('RGBA', (16, 16), (40, 50, 60, 255)).save(b_png)
    ir_path = tmp_path / 'slideir.json'
    write(ir_path, {
        'schema_version': '0.1',
        'assets': [
            {'id': 'a-1', 'path': 'a1.png', 'sha256': digest(a_png)},
            {'id': 'a-2', 'path': 'a2.png', 'sha256': digest(b_png)},
        ],
        'nodes': [
            {'id': 'a-1', 'type': 'image', 'asset_id': 'a-1'},
            {'id': 'a-2', 'type': 'image', 'asset_id': 'a-2'},
        ],
    })
    pptx = tmp_path / 'deck.pptx'
    with ZipFile(pptx, 'w') as zf:
        zf.writestr('ppt/slides/slide1.xml', (
            '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<p:cSld><p:spTree>'
            '<p:pic><p:nvPicPr><p:cNvPr id="2" name="a-1"/></p:nvPicPr>'
            '<p:blipFill><a:blip r:embed="rId1"/></p:blipFill></p:pic>'
            '<p:pic><p:nvPicPr><p:cNvPr id="3" name="a-2"/></p:nvPicPr>'
            '<p:blipFill><a:blip r:embed="rId2"/></p:blipFill></p:pic>'
            '</p:spTree></p:cSld></p:sld>'
        ))
        zf.writestr('ppt/slides/_rels/slide1.xml.rels', (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="../media/img1.png"/>'
            '<Relationship Id="rId2" Target="../media/img2.png"/>'
            '</Relationships>'
        ))
        zf.write(a_png, 'ppt/media/img1.png')
        zf.write(b_png, 'ppt/media/img2.png')
    cache = {}
    verify_pictures(pptx, [ir_path], cache=cache)
    del_pptx = tmp_path / 'del.pptx'
    with ZipFile(del_pptx, 'w') as zf:
        zf.writestr('ppt/slides/slide1.xml', (
            '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<p:cSld><p:spTree/></p:cSld></p:sld>'
        ))
        zf.writestr('ppt/slides/_rels/slide1.xml.rels', (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
        ))
    verify_pictures(del_pptx, [ir_path], {(1, 'a-1'), (1, 'a-2')}, cache=cache)


def test_region_crop_clears_contained_caption_island_and_isolates_adjacent_crops(tmp_path):
    from product.recovery.region_crop import stage_region_crops

    w, h = 440, 260
    card = np.full((h, w, 3), 248, dtype=np.uint8)
    src = card.copy()
    # Two adjacent photos with 10px gap so _content blur doesn't bridge them, while 2px grow_seed expansion puts them 6px apart
    src[20:115, 20:114] = [90, 130, 170]
    src[35:65, 35:95] = [210, 225, 240]
    src[80:115, 20:114] = [18, 86, 160]  # dark-blue caption bar inside photo 1
    src[20:115, 124:218] = [40, 90, 140]
    src[35:65, 140:200] = [235, 242, 250]  # high-contrast interior detail in photo 2
    src[80:115, 124:218] = [18, 86, 160]  # adjacent photo 2 dark-blue bar
    # Paint dark-blue island left on plate after earlier white caption t-001 was erased
    plate = src.copy()
    t_mask = np.zeros((h, w), dtype=bool)
    t_mask[90:104, 38:96] = True
    src[t_mask] = [255, 255, 255]
    plate[t_mask] = [18, 86, 160]

    base = tmp_path / 'base'
    base.mkdir()
    Image.fromarray(src).save(base / 'source.png')
    Image.fromarray(np.dstack([plate, np.full((h, w), 255, np.uint8)])).save(base / 'source-plate.png')
    proof_file = base / 't-001-owner.npz'
    np.savez_compressed(proof_file, mask=t_mask, restore_bbox=[38, 90, 58, 14], backing_patch=np.full((14, 58, 3), [18, 86, 160], np.uint8))

    src_sha = digest(base / 'source.png')
    plate_sha = digest(base / 'source-plate.png')
    proof_sha = digest(proof_file)
    graph = SceneGraph.model_validate({
        'schema': 'SceneGraph/1',
        'source_sha256': src_sha,
        'image_size': [w, h],
        'transform': [1, 0, 0, 1, 0, 0],
        'method_version': 'test',
        'text': {
            'schema': 'TextEvidence/1', 'source_sha256': src_sha, 'image_size': [w, h], 'method_version': 'test',
            'styles': [{'id': 'st', 'role': 'caption', 'ea': 'Arial', 'latin': 'Arial', 'size_px': 12.0, 'bold': False, 'color': 'FFFFFF'}],
            'texts': [{'id': 't-001', 'bbox': [38, 90, 58, 14], 'z': 5, 'style': 'st', 'text': 'Caption'}],
            'members': {},
        },
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': src_sha, 'image_size': [w, h], 'method_version': 'test', 'objects': []},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': src_sha, 'image_size': [w, h], 'method_version': 'test', 'background': 'F8F8F8', 'shapes': []},
        'relations': [],
        'ownership': {'t-001': 'RESOLVED'},
        'raster_backing': {'path': 'source-plate.png', 'sha256': plate_sha, 'source_sha256': src_sha, 'bbox': [0, 0, w, h]},
        'selected': {
            't-001': {
                'id': 't-001', 'kind': 'text', 'bbox': [38, 90, 58, 14], 'restore_bbox': [38, 90, 58, 14],
                'state': 'STAGED', 'source_sha256': src_sha, 'complete_owner_scope': True,
                'owner_evidence': {'file': 't-001-owner.npz', 'sha256': proof_sha, 'method': 'test'},
            },
        },
        'transactions': [],
    })
    write(base / 'scene.json', graph.model_dump())
    write(base / 'slideir.json', {'schema_version': '0.1', 'nodes': []})

    out = tmp_path / 'out'
    _, staged, _ = stage_region_crops(
        base,
        out,
        seeds=[
            {'id': 'layout:p1', 'kind': 'layout:image', 'bbox': [24, 24, 86, 87]},
            {'id': 'layout:p2', 'kind': 'layout:image', 'bbox': [128, 24, 86, 87]},
        ],
    )
    assert len(staged) == 2
    out_plate = np.array(Image.open(out / 'source-plate.png').convert('RGB'))
    # Both adjacent photo regions AND the wholly-contained caption island must now be clean card white (no dark blue smeared band!)
    assert int(np.abs(out_plate[90:104, 38:96].astype(int) - 248).max()) <= 3
    assert int(np.abs(out_plate[20:115, 20:114].astype(int) - 248).max()) <= 3
    assert int(np.abs(out_plate[20:115, 124:218].astype(int) - 248).max()) <= 3


def test_ring_parent_patch_gradient_fallback_and_repair_rounds_per_block(tmp_path):
    from product.recovery.container_stage import _discover_candidates, _ring_parent_patch
    from product.runtime.host_evidence import normalize_arguments
    from product.runtime.state import RuntimeState

    h, w = 100, 140
    yy, _ = np.mgrid[:h, :w]
    canvas_field = np.dstack([240 - 0.12 * yy, 245 - 0.08 * yy, np.full((h, w), 252.0)]).astype(np.float32)
    work = np.clip(np.rint(canvas_field), 0, 255).astype(np.uint8)
    # Add outer shadow/texture band in exterior ring to raise flat err > 6.5 while matching canvas_field on >= 50% of ring
    work[12:15, 20:100] = [195, 210, 230]
    work[18:80, 24:96] = [255, 255, 255]
    patch_res = _ring_parent_patch(work, canvas_field, [24, 18, 72, 62], pad=5)
    assert patch_res is not None
    assert patch_res[2] <= 6.5

    # Verify _discover_candidates 3-arg signature compatibility used by marked_surfaces
    cands = _discover_candidates(work, canvas_field, np.zeros((h, w), bool))
    assert any(k == 'box' for k, _ in cands)

    # repair_rounds allows repairing block 2 after block 1 made no progress
    job = tmp_path / 'job'
    job.mkdir()
    st = RuntimeState(job)
    write(job / 'q1.json', {'items': [{'id': 't-001'}]})
    write(job / 'q2.json', {'items': [{'id': 't-002'}]})
    r1 = st.register(job / 'q1.json', 'numbered_questions')['id']
    r2 = st.register(job / 'q2.json', 'numbered_questions')['id']
    cur = {
        'questions': {'1': [{'block': 1, 'questions': r1}, {'block': 2, 'questions': r2}]},
        'repair_rounds': {'1': [{'progress': False, 'candidate_id': None, 'object_ids': ['t-001']}]},
    }
    assert normalize_arguments(st, cur, 'batch_reconstruct_page', {'page': 1, 'block': 2})['object_ids'] == ['t-002']
    import pytest
    with pytest.raises(ValueError, match='local repair limit'):
        normalize_arguments(st, cur, 'batch_reconstruct_page', {'page': 1, 'block': 1})
    crop_args = normalize_arguments(st, cur, 'crop_region', {'page': 1})
    assert crop_args == {'page': 1, 'allow_movable': True, 'allow_move_unsafe': False, 'upgrade_containers': True}


def test_page_structure_ocr_correction_preserves_numbers_and_fixes_cjk_typo():
    manifest = fonts.font_manifest([])
    cjk = next((f for f in manifest['fonts'] if f.get('cjk')), None)
    assert cjk is not None
    w, h = 240, 60
    img = Image.new('RGB', (w, h), (18, 86, 160))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(cjk['file'], 24, index=int(cjk.get('index', 0)))
    draw.text((20, 14), '政策解读与咨询', fill=(255, 255, 255), font=font)
    arr = np.asarray(img)
    ocr = {'lines': [{'id': 'l001', 'text': '政策解读与瓷询', 'bbox': [16, 10, 200, 38], 'score': 0.98}]}
    _, applied = verify_and_apply_ocr_corrections(
        arr,
        json.loads(json.dumps(ocr)),
        [
            OCRCorrection(ocr_id='l001', corrected_text='政策解读与咨询', reason='typo'),
            OCRCorrection(ocr_id='l001', corrected_text='政策1解读与咨询', reason='bad-num'),
        ],
        fonts={'fonts': [cjk]},
    )
    assert len(applied) == 1
    assert applied[0]['after'] == '政策解读与咨询'


def test_p1_tail_fixes_and_native_capabilities(tmp_path, monkeypatch):
    import pytest
    from product.recovery.adaptive_policy import AdaptiveBudget
    from product.recovery.candidate_stage import _expose_staged_dir
    from product.recovery.native_shapes import estimate_card_shadow, try_fit_native_shape_v2
    from product.recovery.region_crop import _detect_and_prepare_rounded_photo
    from product.recovery.vector_icons import try_vectorize_monochrome_icon, upgrade_staged_icon_to_vector_shape
    from product.runtime.batch_backbone import _deoverlap_same_row_ocr_lines
    from product.runtime.review_control import ControlAnswer
    from product.parallel_contracts import SceneGraph
    from product.parallel_office import _node_bleed_margin_px
    from product.parallel_rebuild import adapt
    from product.parallel_store import Store

    # 1. Windows transient directory rename retry + copy fallback
    src_dir = tmp_path / 'staged'
    src_dir.mkdir()
    (src_dir / 'file.txt').write_text('ok', encoding='utf8')
    out_dir = tmp_path / 'final'
    orig_rename = Path.rename
    attempts = {'n': 0}

    def flaky_rename(self, target):
        if self == src_dir and attempts['n'] < 8:
            attempts['n'] += 1
            raise PermissionError('transient AV lock')
        return orig_rename(self, target)

    monkeypatch.setattr(Path, 'rename', flaky_rename)
    _expose_staged_dir(src_dir, out_dir)
    assert (out_dir / 'file.txt').read_text('utf8') == 'ok'

    # 2. Same-row OCR de-overlap including score < 0.85 bloated prefix ('08|')
    row_img = np.full((60, 420, 3), 248, dtype=np.uint8)
    row_img[16:40, 20:54] = [20, 60, 140]      # '08'
    row_img[15:41, 66:69] = [20, 60, 140]      # '|'
    row_img[16:40, 108:300] = [10, 40, 100]    # '应用成效与未来展望'
    ocr_row = {'lines': [
        {'id': 'l000', 'text': '08|', 'bbox': [18, 12, 108, 32], 'polygon': [[18, 12], [126, 12], [126, 44], [18, 44]], 'score': 0.78},
        {'id': 'l001', 'text': '应用成效与未来展望', 'bbox': [108, 12, 198, 32], 'polygon': [[108, 12], [306, 12], [306, 44], [108, 44]], 'score': 0.99},
    ]}
    assert _deoverlap_same_row_ocr_lines(row_img, ocr_row) is True
    b0, b1 = ocr_row['lines'][0]['bbox'], ocr_row['lines'][1]['bbox']
    assert b0[0] + b0[2] <= b1[0]
    assert b0[0] + b0[2] >= 69

    # 3. ControlAnswer accepts direct letter dict
    ans = ControlAnswer.model_validate({'A': 'clean', 'B': 'wrong_unit', 'C': 'old_number', 'D': 'flattened_card'})
    assert ans.defects['B'] == 'wrong_unit'

    # 4. Multi-page AdaptiveBudget scales defaults while respecting explicit overrides
    b4 = AdaptiveBudget(Store(tmp_path / 'b4'), page_count=4)
    assert b4.limits['model_calls'] == 48 and b4.limits['office_validations'] == 240
    b4_ovr = AdaptiveBudget(Store(tmp_path / 'b4o'), {'model_calls': 15}, page_count=4)
    assert b4_ovr.limits['model_calls'] == 15 and b4_ovr.limits['office_validations'] == 240

    # 5. Rounded photo detection & corner cleanup
    w_p, h_p = 100, 80
    backing = np.full((h_p, w_p, 3), 255, dtype=np.uint8)
    photo = np.full((h_p, w_p, 3), 255, dtype=np.uint8)
    yy_p, xx_p = np.mgrid[:h_p, :w_p]
    r_true = 12
    inside = np.ones((h_p, w_p), bool)
    for cy_c, cx_c in ((r_true, r_true), (r_true, w_p - 1 - r_true), (h_p - 1 - r_true, r_true), (h_p - 1 - r_true, w_p - 1 - r_true)):
        quad = ((yy_p < r_true) if cy_c == r_true else (yy_p > h_p - 1 - r_true)) & ((xx_p < r_true) if cx_c == r_true else (xx_p > w_p - 1 - r_true))
        inside &= (~quad) | (((xx_p - cx_c) ** 2 + (yy_p - cy_c) ** 2) <= r_true ** 2)
    photo[inside] = [30, 90, 180]
    res_round = _detect_and_prepare_rounded_photo(photo, backing)
    assert res_round is not None
    cleaned_px, est_r = res_round
    assert 9.0 <= est_r <= 15.0
    assert int(cleaned_px[0, 0, 2]) > 140  # outer corner inpainted away from white backing

    # 6. Monochrome icon vectorization + relation rewrite + foreground z-ordering
    ic_w, ic_h = 48, 48
    backing_ic = np.full((ic_h, ic_w, 3), 255, dtype=np.uint8)
    yy_i, xx_i = np.mgrid[:ic_h, :ic_w]
    ring = ((xx_i - 23.5) ** 2 + (yy_i - 23.5) ** 2 <= 18 ** 2) & ((xx_i - 23.5) ** 2 + (yy_i - 23.5) ** 2 >= 8 ** 2)
    rgba_ic = np.zeros((ic_h, ic_w, 4), dtype=np.uint8)
    rgba_ic[ring] = [16, 100, 220, 255]
    src_ic = backing_ic.copy()
    src_ic[ring] = [16, 100, 220]
    vec_shape = try_vectorize_monochrome_icon(rgba_ic, src_ic, backing_ic, [100, 100, ic_w, ic_h], 's-v001', 1)
    assert vec_shape is not None and vec_shape.geometry == 'custom' and len(vec_shape.custom_paths[0].commands) >= 6

    # Full SceneGraph upgrade_staged_icon_to_vector_shape + foreground_shape z-order above card & container asset
    vbase = tmp_path / 'vbase'
    vbase.mkdir()
    src_full = np.full((180, 220, 3), 245, dtype=np.uint8)
    src_full[40:40 + ic_h, 40:40 + ic_w][ring] = [16, 100, 220]
    Image.fromarray(src_full).save(vbase / 'source.png')
    plate_full = np.full((180, 220, 4), 245, dtype=np.uint8); plate_full[..., 3] = 255
    Image.fromarray(plate_full).save(vbase / 'source-plate.png')
    Image.fromarray(rgba_ic).save(vbase / 'a-001.png')
    cont_rgba = np.zeros((20, 120, 4), dtype=np.uint8); cont_rgba[..., 3] = 255; cont_rgba[..., :3] = [20, 80, 160]
    Image.fromarray(cont_rgba).save(vbase / 'a-c01.png')
    a_mask = np.zeros((180, 220), bool); a_mask[40:40 + ic_h, 40:40 + ic_w] = ring
    np.savez_compressed(vbase / 'a-001-owner.npz', mask=a_mask, restore_bbox=[40, 40, ic_w, ic_h], backing_patch=np.full((ic_h, ic_w, 3), 245, np.uint8))
    c_mask = np.zeros((180, 220), bool); c_mask[10:30, 20:140] = True
    np.savez_compressed(vbase / 'a-c01-owner.npz', mask=c_mask, restore_bbox=[20, 10, 120, 20], backing_patch=np.full((20, 120, 3), 245, np.uint8))
    ssha = digest(vbase / 'source.png')
    vg = SceneGraph.model_validate({
        'schema': 'SceneGraph/1', 'source_sha256': ssha, 'image_size': [220, 180], 'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': ssha, 'image_size': [220, 180], 'method_version': 'test', 'styles': [], 'texts': [], 'members': {}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': ssha, 'image_size': [220, 180], 'method_version': 'test', 'objects': [
            {'id': 'a-001', 'kind': 'icon', 'bbox': [40, 40, ic_w, ic_h], 'layers': ['l.png'], 'components': ['whole'],
             'parts': [{'name': 'whole', 'bbox': [40, 40, ic_w, ic_h], 'layers': ['l.png']}], 'exclusions': [], 'replaces': [],
             'required_owners': [], 'text_policy': 'exclude_independent_text', 'intrinsic_text': '', 'description': 'icon', 'prompt_en': 'icon'},
            {'id': 'a-c01', 'kind': 'decoration', 'bbox': [20, 10, 120, 20], 'layers': ['l.png'], 'components': ['whole'],
             'parts': [{'name': 'whole', 'bbox': [20, 10, 120, 20], 'layers': ['l.png']}], 'exclusions': [], 'replaces': [],
             'required_owners': [], 'text_policy': 'exclude_independent_text', 'intrinsic_text': '', 'description': 'bar', 'prompt_en': 'bar'},
        ]},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': ssha, 'image_size': [220, 180], 'method_version': 'test', 'background': 'F5F5F5', 'shapes': []},
        'relations': [{'source': 'a-c01', 'target': 'a-001', 'kind': 'connects', 'status': 'RESOLVED', 'evidence': ['test']}],
        'ownership': {'a-001': 'RESOLVED', 'a-c01': 'RESOLVED'},
        'raster_backing': {'path': 'source-plate.png', 'sha256': digest(vbase / 'source-plate.png'), 'source_sha256': ssha, 'bbox': [0, 0, 220, 180]},
        'selected': {
            'a-001': {'id': 'a-001', 'kind': 'asset', 'bbox': [40, 40, ic_w, ic_h], 'restore_bbox': [40, 40, ic_w, ic_h],
                      'payload': {'method': 'test'},
                      'asset_reference': {'path': 'a-001.png', 'sha256': digest(vbase / 'a-001.png'), 'source_sha256': ssha, 'bbox': [40, 40, ic_w, ic_h], 'crop_offset': [40, 40]},
                      'owner_evidence': {'file': 'a-001-owner.npz', 'sha256': digest(vbase / 'a-001-owner.npz')}},
            'a-c01': {'id': 'a-c01', 'kind': 'asset', 'container_layer': True, 'bbox': [20, 10, 120, 20], 'restore_bbox': [20, 10, 120, 20],
                      'payload': {'method': 'test'},
                      'asset_reference': {'path': 'a-c01.png', 'sha256': digest(vbase / 'a-c01.png'), 'source_sha256': ssha, 'bbox': [20, 10, 120, 20], 'crop_offset': [20, 10]},
                      'owner_evidence': {'file': 'a-c01-owner.npz', 'sha256': digest(vbase / 'a-c01-owner.npz')}},
        },
        'transactions': [],
    })
    write(vbase / 'scene.json', vg.model_dump())
    write(vbase / 'slideir.json', adapt(vg, vbase))
    vout = tmp_path / 'vout'
    upgraded = upgrade_staged_icon_to_vector_shape(vbase, 'a-001', vout)
    assert upgraded is not None and upgraded[1] == 's-v001'
    up_ir = json.loads((vout / 'slideir.json').read_text('utf8'))
    z_by_id = {n['id']: n['z'] for n in up_ir['nodes']}
    assert z_by_id['s-v001'] > z_by_id['a-c01']
    up_sg = SceneGraph.model_validate_json((vout / 'scene.json').read_text('utf8'))
    assert up_sg.relations[0].target == 's-v001'

    # 7. Straight arrow connector fitting, shadow estimation, and COM bleed margin
    arrow_plate = np.full((24, 90, 3), 250, dtype=np.uint8)
    arrow_plate[10:14, 6:72] = [30, 90, 200]
    for dx in range(14):
        half_h = max(1, round((14 - dx) * 0.65))
        arrow_plate[12 - half_h:12 + half_h, 72 + dx] = [30, 90, 200]
    fit = try_fit_native_shape_v2(arrow_plate, [0, 0, 90, 24], np.full((24, 90, 3), 250.0), [], np.zeros((24, 90), bool), cand_kind='box')
    assert fit is not None and fit['geometry'] == 'straightConnector1' and fit['tail_end'] == 'triangle'

    cfield = np.full((40, 80, 3), 240.0, dtype=np.float32)
    splate = np.full((40, 80, 3), 240, dtype=np.uint8)
    hmask = np.zeros((40, 80), bool); hmask[20:30, 10:70] = True
    splate[hmask] = [228, 228, 228]
    sh_obj = estimate_card_shadow(splate, cfield, hmask)
    assert sh_obj is not None and 0.06 <= sh_obj.opacity <= 0.22
    assert _node_bleed_margin_px({'shadow': sh_obj.model_dump(), 'line': {'width_pt': 2.0, 'tail_end': 'triangle'}}) >= 15


def test_p2_text_recall_composite_cards_and_crop_vector_icons(tmp_path):
    from product.parallel_fusion import glyph_evidence
    from product.parallel_rebuild import adapt
    from product.recovery.vector_icons import upgrade_crops_to_vector_shapes

    # (a) Dark-ink stroke core & collinear alpha > 1.0 glyph evidence
    backing_a = np.full((40, 80, 3), [232, 242, 252], dtype=np.uint8)
    src_a = backing_a.copy()
    src_a[10:28, 16:22] = [1, 5, 62]     # deep multi-stroke navy core
    src_a[10:28, 26:32] = [10, 25, 68]   # collinear overshoot alpha ~ 1.15 vs foreground 1C366B
    fg_mask, unassigned = glyph_evidence(src_a, [0, 0, 80, 40], '1C366B', backing_a)
    assert fg_mask.any()
    assert not unassigned.any()
    assert float(unassigned.max()) <= 20.0

    # (b) Composite subtle white card + #E4EFFB header band + flush two-tone pentagon ribbon pair
    h_img, w_img = 300, 540
    sky = np.full((h_img, w_img, 3), [244, 249, 254], dtype=np.uint8)
    src_b = sky.copy()
    # Outer white card [40, 40, 460, 220] with subtle border and #E4EFFB header [42, 42, 456, 40]
    cx0, cy0, cw, ch = 40, 40, 460, 220
    src_b[cy0:cy0 + ch, cx0:cx0 + cw] = [253, 254, 255]
    src_b[cy0:cy0 + 2, cx0:cx0 + cw] = [234, 241, 250]
    src_b[cy0 + ch - 2:cy0 + ch, cx0:cx0 + cw] = [234, 241, 250]
    src_b[cy0:cy0 + ch, cx0:cx0 + 2] = [234, 241, 250]
    src_b[cy0:cy0 + ch, cx0 + cw - 2:cx0 + cw] = [234, 241, 250]
    hx0, hy0, hw, hh = 42, 42, 456, 40
    src_b[hy0:hy0 + hh, hx0:hx0 + hw] = [228, 239, 251]
    # Flush two-tone pentagon ribbon [42, 42, 160, 40]: cap [0..52] + pentagon [52..160]
    rw, tip = 160, 14
    for ry_l in range(hh):
        dx = int(round(tip * (1.0 - abs(ry_l - (hh - 1) * 0.5) / ((hh - 1) * 0.5))))
        max_c = (rw - tip) + dx
        src_b[hy0 + ry_l, hx0:hx0 + min(52, max_c)] = [37, 137, 249]
        if max_c > 52:
            src_b[hy0 + ry_l, hx0 + 52:hx0 + max_c] = [16, 98, 194]

    cbase = tmp_path / 'cbase'
    cbase.mkdir()
    Image.fromarray(src_b).save(cbase / 'source.png')
    Image.fromarray(src_b).save(cbase / 'source-plate.png')
    csha = digest(cbase / 'source.png')
    psha = digest(cbase / 'source-plate.png')
    sg_b = SceneGraph.model_validate({
        'schema': 'SceneGraph/1',
        'source_sha256': csha,
        'image_size': [w_img, h_img],
        'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': csha, 'image_size': [w_img, h_img], 'method_version': 'test', 'styles': [], 'texts': [], 'members': {}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': csha, 'image_size': [w_img, h_img], 'method_version': 'test', 'objects': []},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': csha, 'image_size': [w_img, h_img], 'method_version': 'test', 'background': 'F4F9FE', 'shapes': []},
        'ownership': {},
        'relations': [],
        'raster_backing': {'path': 'source-plate.png', 'sha256': psha, 'source_sha256': csha, 'bbox': [0, 0, w_img, h_img]},
        'selected': {},
        'transactions': [],
    })
    write(cbase / 'scene.json', sg_b.model_dump())
    write(cbase / 'slideir.json', adapt(sg_b, cbase))
    ir_out, staged_ids = stage_containers_and_surfaces(cbase, tmp_path / 'cout')
    out_dir = ir_out.parent
    sg_out = SceneGraph.model_validate_json((out_dir / 'scene.json').read_text('utf8'))
    assert len(staged_ids) >= 4
    assert any(s.geometry == 'custom' and s.custom_paths for s in sg_out.surface.shapes)
    compiled_b = compile_local([ir_out], tmp_path / 'composite_compiled', [effective_texts(sg_out, ir_out)])
    import zipfile as _zf
    with _zf.ZipFile(compiled_b['pptx']) as z_c:
        s1_xml = z_c.read('ppt/slides/slide1.xml').decode('utf8')
    assert '<a:custGeom>' in s1_xml
    for sid in staged_ids:
        rec = sg_out.selected[sid]
        if rec.get('kind') == 'surface':
            with np.load(out_dir / rec['owner_evidence']['file'], allow_pickle=False) as d:
                assert int(d['mask'].sum()) >= 16
    ribbon_recs = [
        sg_out.selected[sid]
        for sid in staged_ids
        if sg_out.selected.get(sid, {}).get('method') == 'container-surface-v2-two-tone-ribbon/1'
    ]
    assert len(ribbon_recs) == 2
    with np.load(out_dir / ribbon_recs[0]['owner_evidence']['file'], allow_pickle=False) as d0:
        m0 = d0['mask'].astype(bool)
    with np.load(out_dir / ribbon_recs[1]['owner_evidence']['file'], allow_pickle=False) as d1:
        m1 = d1['mask'].astype(bool)
    assert int((m0 & m1).sum()) == 0

    # (c) Tight cutout region_crop vectorization via upgrade_crops_to_vector_shapes
    rcbase = tmp_path / 'rcbase'
    rcbase.mkdir()
    ic_w, ic_h = 48, 48
    yy_i, xx_i = np.mgrid[:ic_h, :ic_w]
    ring = ((xx_i - 23.5) ** 2 + (yy_i - 23.5) ** 2 <= 18 ** 2) & ((xx_i - 23.5) ** 2 + (yy_i - 23.5) ** 2 >= 8 ** 2)
    rgba_rc = np.zeros((ic_h, ic_w, 4), dtype=np.uint8)
    rgba_rc[ring] = [16, 100, 220, 255]
    src_rc = np.full((180, 220, 3), 245, dtype=np.uint8)
    src_rc[40:40 + ic_h, 40:40 + ic_w][ring] = [16, 100, 220]
    plate_rc = np.full((180, 220, 3), 245, dtype=np.uint8)
    Image.fromarray(src_rc).save(rcbase / 'source.png')
    Image.fromarray(plate_rc).save(rcbase / 'source-plate.png')
    Image.fromarray(rgba_rc).save(rcbase / 'r-001.png')
    r_mask = np.zeros((180, 220), bool)
    r_mask[40:40 + ic_h, 40:40 + ic_w] = ring
    np.savez_compressed(
        rcbase / 'r-001-owner.npz',
        mask=r_mask,
        restore_bbox=[40, 40, ic_w, ic_h],
        backing_patch=np.full((ic_h, ic_w, 3), 245, dtype=np.uint8),
    )
    rsha = digest(rcbase / 'source.png')
    rpsha = digest(rcbase / 'source-plate.png')
    sg_rc = SceneGraph.model_validate({
        'schema': 'SceneGraph/1',
        'source_sha256': rsha,
        'image_size': [220, 180],
        'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': rsha, 'image_size': [220, 180], 'method_version': 'test', 'styles': [], 'texts': [], 'members': {}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': rsha, 'image_size': [220, 180], 'method_version': 'test', 'objects': [
            {'id': 'r-001', 'kind': 'illustration', 'bbox': [40, 40, ic_w, ic_h], 'layers': ['l.png'], 'components': ['whole'],
             'parts': [{'name': 'whole', 'bbox': [40, 40, ic_w, ic_h], 'layers': ['l.png']}], 'exclusions': [], 'replaces': [],
             'required_owners': [], 'text_policy': 'exclude_independent_text', 'intrinsic_text': '', 'description': 'ring', 'prompt_en': 'ring'},
        ]},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': rsha, 'image_size': [220, 180], 'method_version': 'test', 'background': 'F5F5F5', 'shapes': []},
        'ownership': {'r-001': 'RESOLVED'},
        'relations': [],
        'raster_backing': {'path': 'source-plate.png', 'sha256': rpsha, 'source_sha256': rsha, 'bbox': [0, 0, 220, 180]},
        'selected': {
            'r-001': {
                'id': 'r-001', 'kind': 'asset', 'region_crop': True, 'bbox': [40, 40, ic_w, ic_h],
                'restore_bbox': [40, 40, ic_w, ic_h], 'payload': {'method': 'test'},
                'asset_reference': {'path': 'r-001.png', 'sha256': digest(rcbase / 'r-001.png'), 'source_sha256': rsha,
                                    'bbox': [40, 40, ic_w, ic_h], 'crop_offset': [40, 40]},
                'owner_evidence': {'file': 'r-001-owner.npz', 'sha256': digest(rcbase / 'r-001-owner.npz')},
            },
        },
        'transactions': [],
    })
    write(rcbase / 'scene.json', sg_rc.model_dump())
    write(rcbase / 'slideir.json', adapt(sg_rc, rcbase))
    vec_ir, vec_ids = upgrade_crops_to_vector_shapes(rcbase, ['r-001'], tmp_path / 'rcwork')
    assert vec_ids == ['s-v001']
    sg_vec = SceneGraph.model_validate_json((vec_ir.parent / 'scene.json').read_text('utf8'))
    assert any(s.id == 's-v001' and s.geometry == 'custom' for s in sg_vec.surface.shapes)


def test_container_stage_refreshes_overlying_backing_patches_and_excludes_card_corner_sky(tmp_path):
    from product.parallel_rebuild import adapt
    from product.recovery.container_stage import stage_containers_and_surfaces
    from product.recovery.native_shapes import try_fit_two_tone_ribbon_pair

    w, h = 560, 280
    # Canvas sky [243,249,254], bordered card [40:520, 40:240], header [225,238,252], two-tone ribbon flush at card TL with r=9 sky corner
    plate = np.full((h, w, 3), [243, 249, 254], dtype=np.uint8)
    plate[40:240, 40:520] = [210, 224, 240]
    plate[41:83, 41:519] = [225, 238, 252]
    plate[83:239, 41:519] = [254, 255, 255]
    # Two-tone ribbon at [41:241, 41:83] (w=200, h=42): left cap x=41..101, right body x=101..241 (tip at 241)
    rx, ry, rw, rh = 41, 41, 200, 42
    yy, xx = np.mgrid[:rh, :rw]
    half_h = (rh - 1) / 2.0
    max_x = 186.0 + (1.0 - np.abs(yy - half_h) / half_h) * (rw - 1 - 186.0)
    in_rib = xx <= max_x + 0.5
    # Rounded TL corner of card exposes sky [243,249,254]
    r_tl = 9.0
    tl_sky = (yy < r_tl) & (xx < r_tl) & (((xx - r_tl) ** 2 + (yy - r_tl) ** 2) > r_tl ** 2)
    rib_sub = plate[ry:ry + rh, rx:rx + rw]
    rib_sub[in_rib & (xx < 60) & (~tl_sky)] = [35, 135, 249]
    rib_sub[in_rib & (xx >= 60)] = [16, 98, 198]
    rib_sub[tl_sky] = [243, 249, 254]
    plate[ry:ry + rh, rx:rx + rw] = rib_sub

    # Direct unit test of sky-corner exclusion in try_fit_two_tone_ribbon_pair
    par_hdr = np.full((rh, rw, 3), [225, 238, 252], dtype=np.float64)
    pair = try_fit_two_tone_ribbon_pair(plate, [rx, ry, rw, rh], par_hdr, np.zeros((rh, rw), bool))
    assert pair is not None
    assert pair[0]['geometry'] == 'custom'
    assert not bool(pair[0]['erase_mask_local'][0, 0])
    assert int(pair[0]['erase_mask_local'][:6, :6].sum()) < 26

    # Full container_stage test verifying overlying text t-001 backing_patch synchronization
    base = tmp_path / 'base'
    base.mkdir()
    Image.fromarray(plate).save(base / 'source.png')
    Image.fromarray(plate).save(base / 'source-plate.png')
    t_mask = np.zeros((h, w), dtype=bool)
    t_mask[50:74, 52:76] = True
    # Stale pre-container backing patch differs from ribbon [35,135,249] by 11 RGB
    stale_bp = np.full((24, 24, 3), [24, 132, 247], dtype=np.uint8)
    t_proof = base / 't-001-owner.npz'
    np.savez_compressed(t_proof, mask=t_mask, restore_bbox=[52, 50, 24, 24], backing_patch=stale_bp)
    ssha = digest(base / 'source.png')
    psha = digest(base / 'source-plate.png')
    sg = SceneGraph.model_validate({
        'schema': 'SceneGraph/1', 'source_sha256': ssha, 'image_size': [w, h], 'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': ssha, 'image_size': [w, h], 'method_version': 'test',
                 'styles': [{'id': 'st', 'role': 'body', 'ea': 'Arial', 'latin': 'Arial', 'size_px': 18.0, 'bold': True, 'color': 'FFFFFF'}],
                 'texts': [{'id': 't-001', 'text': '01', 'bbox': [52.0, 50.0, 24.0, 24.0], 'style': 'st', 'z': 5}],
                 'members': {'t-001': ['l1']}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': ssha, 'image_size': [w, h], 'method_version': 'test', 'objects': []},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': ssha, 'image_size': [w, h], 'method_version': 'test', 'background': 'F3F9FE', 'shapes': []},
        'ownership': {'t-001': 'RESOLVED'}, 'relations': [],
        'raster_backing': {'path': 'source-plate.png', 'sha256': psha, 'source_sha256': ssha, 'bbox': [0, 0, w, h]},
        'selected': {
            't-001': {
                'id': 't-001', 'kind': 'text', 'bbox': [52.0, 50.0, 24.0, 24.0], 'restore_bbox': [52, 50, 24, 24],
                'payload': {'id': 't-001', 'text': '01', 'bbox': [52.0, 50.0, 24.0, 24.0], 'style': 'st', 'z': 5},
                'owner_evidence': {'file': 't-001-owner.npz', 'sha256': digest(t_proof)},
            },
        },
        'transactions': [{'id': 't-001', 'owner_evidence': {'file': 't-001-owner.npz', 'sha256': digest(t_proof)}}],
    })
    write(base / 'scene.json', sg.model_dump())
    write(base / 'slideir.json', adapt(sg, base))

    out_dir = tmp_path / 'out'
    _, staged_ids = stage_containers_and_surfaces(base, out_dir)
    assert staged_ids
    sg_out = SceneGraph.model_validate_json((out_dir / 'scene.json').read_text('utf8'))
    new_t_ref = sg_out.selected['t-001']['owner_evidence']
    assert new_t_ref['sha256'] == digest(out_dir / new_t_ref['file'])
    with np.load(out_dir / new_t_ref['file'], allow_pickle=False) as d:
        bp_updated = d['backing_patch']
    assert np.allclose(bp_updated.mean(axis=(0, 1)), [35, 135, 249], atol=2.0)


def test_collect_office_defects_blames_only_direct_innermost_surface():
    from product.runtime.batch_backbone import _collect_office_defects

    sg = SceneGraph.model_validate({
        'schema': 'SceneGraph/1', 'source_sha256': 'a' * 64, 'image_size': [600, 500], 'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': 'a' * 64, 'image_size': [600, 500], 'method_version': 'test', 'styles': [], 'texts': [], 'members': {}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': 'a' * 64, 'image_size': [600, 500], 'method_version': 'test', 'objects': []},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': 'a' * 64, 'image_size': [600, 500], 'method_version': 'test', 'background': 'FFFFFF', 'shapes': []},
        'ownership': {}, 'relations': [],
        'selected': {
            't-005': {'id': 't-005', 'kind': 'text', 'bbox': [15, 15, 40, 30], 'restore_bbox': [15, 15, 40, 30]},
            's-ribbon': {'id': 's-ribbon', 'kind': 'surface', 'bbox': [0, 0, 80, 60], 'restore_bbox': [0, 0, 80, 60]},
            's-adj': {'id': 's-adj', 'kind': 'surface', 'bbox': [80, 0, 140, 60], 'restore_bbox': [80, 0, 140, 60]},
            's-header': {'id': 's-header', 'kind': 'surface', 'bbox': [0, 0, 500, 60], 'restore_bbox': [0, 0, 500, 60]},
            's-outer': {'id': 's-outer', 'kind': 'surface', 'bbox': [0, 0, 500, 400], 'restore_bbox': [0, 0, 500, 400]},
        },
    })
    office_res = {
        'residuals': [[{'id': 't-005', 'residual_check_status': 'FAIL'}]],
        'replacements': {'records': []},
        'office': {},
    }
    _, bad_ids = _collect_office_defects(office_res, sg, ['s-ribbon', 's-adj', 's-header', 's-outer'])
    assert bad_ids == {'s-ribbon'}


def test_split_decorative_ocr_keeps_chinese_comma_and_merges_paragraph():
    from product.runtime.batch_backbone import split_decorative_ocr_line, _merge_structural_ocr_items

    src = np.full((200, 500, 3), 255, dtype=np.uint8)
    # Paint dark ink strips inside lines so split_decorative_ocr_line sees text runs
    src[44:60, 25:340] = 50
    t_val, b_val, aff = split_decorative_ocr_line(
        src, {'id': 'l1', 'bbox': [20, 40, 350, 25], 'text': '内置多项智能体能力，可用于公文撰写、'}
    )
    assert t_val.endswith('、')
    assert aff == ''

    raw = [
        {'id': 't-1', 'ocr_line_id': 'l1', 'text': '随着模型能力发展和', 'bbox': [100.0, 50.0, 145.0, 21.0],
         'color_hex': '58667F', 'est_size_px': 15.4, 'bold': False, 'stripped_affix': ''},
        {'id': 't-2', 'ocr_line_id': 'l2', 'text': '业务需求变化，持续', 'bbox': [101.0, 75.0, 144.0, 20.0],
         'color_hex': '56647C', 'est_size_px': 15.4, 'bold': False, 'stripped_affix': ''},
        {'id': 't-3', 'ocr_line_id': 'l3', 'text': '优化平台能力', 'bbox': [98.0, 94.0, 101.0, 26.0],
         'color_hex': '4D5F7B', 'est_size_px': 18.4, 'bold': False, 'stripped_affix': ''},
    ]
    cleaned, _ = _merge_structural_ocr_items(src, raw, [])
    assert len(cleaned) == 1
    assert cleaned[0]['ocr_line_ids'] == ['l1', 'l2', 'l3']


def test_validate_candidate_pass2_scopes_recheck_and_merges_untouched_pass_records(tmp_path, monkeypatch):
    import shutil
    from product.runtime.service import RuntimeService
    from product.runtime.state import RuntimeState
    import product.runtime.batch_backbone as bb

    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT', str(tmp_path / 'leases'))
    jobs_root = tmp_path / 'jobs'
    job_dir = jobs_root / 'ccddeeff'
    (job_dir / 'inputs').mkdir(parents=True)
    src_p = job_dir / 'inputs/1.png'
    Image.new('RGB', (200, 120), (245, 248, 252)).save(src_p)
    snap = snapshot(src_p, job_dir / 'snap')
    write(job_dir / 'request.json', {'sources': ['1.png'], 'orchestrator': 'opencode'})
    fonts_p = job_dir / 'fonts.json'
    write(fonts_p, {'schema': 'FontManifest/1', 'fonts': []})

    base_scene = {
        'schema': 'SceneGraph/1', 'source_sha256': snap['source_sha256'], 'image_size': [200, 120],
        'transform': [1, 0, 0, 1, 0, 0], 'method_version': 'test',
        'text': {'schema': 'TextEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [200, 120], 'method_version': 'test',
                 'styles': [{'id': 'st', 'role': 'body', 'ea': 'Arial', 'latin': 'Arial', 'size_px': 12.0, 'bold': False, 'color': '000000'}],
                 'texts': [
                     {'id': 't-far', 'text': 'far', 'bbox': [10.0, 10.0, 30.0, 14.0], 'style': 'st', 'z': 1},
                     {'id': 't-near', 'text': 'near', 'bbox': [120.0, 60.0, 30.0, 14.0], 'style': 'st', 'z': 2},
                 ],
                 'members': {'t-far': ['l1'], 't-near': ['l2']}},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [200, 120], 'method_version': 'test', 'objects': [
            {'id': 'r-bad', 'kind': 'icon', 'bbox': [145, 60, 20, 20], 'layers': ['l.png'], 'replaces': [], 'components': ['c'],
             'parts': [{'name': 'c', 'bbox': [145, 60, 20, 20], 'layers': ['l.png']}], 'required_owners': [],
             'text_policy': 'exclude_independent_text', 'intrinsic_text': '', 'description': 'b', 'prompt_en': 'b'},
        ]},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': snap['source_sha256'], 'image_size': [200, 120], 'method_version': 'test', 'background': 'F5F8FC', 'shapes': []},
        'ownership': {'t-far': 'RESOLVED', 't-near': 'RESOLVED', 'r-bad': 'RESOLVED'}, 'relations': [],
    }

    bdir = job_dir / 'base'; bdir.mkdir()
    p1dir = job_dir / 'pass1'; p1dir.mkdir()
    p2dir = job_dir / 'pass2'; p2dir.mkdir()
    plate_img = np.full((120, 200, 3), [245, 248, 252], dtype=np.uint8)
    Image.fromarray(plate_img).save(p1dir / 'source-plate.png')
    plate_p2 = plate_img.copy()
    plate_p2[60:80, 145:165] = [100, 120, 140]  # only r-bad region restored on plate
    Image.fromarray(plate_p2).save(p2dir / 'source-plate.png')

    def _mk_proof(d, name, box):
        p = d / f'{name}.npz'
        np.savez_compressed(p, mask=np.zeros((120, 200), bool), restore_bbox=box, backing_patch=np.zeros((box[3], box[2], 3), np.uint8))
        return {'file': p.name, 'sha256': digest(p)}

    write(bdir / 'scene.json', {**base_scene, 'selected': {}})
    write(bdir / 'slideir.json', {'nodes': []})

    sel_far = {'id': 't-far', 'kind': 'text', 'bbox': [10, 10, 30, 14], 'restore_bbox': [10, 10, 30, 14], 'owner_evidence': _mk_proof(p1dir, 't-far', [10, 10, 30, 14])}
    sel_near = {'id': 't-near', 'kind': 'text', 'bbox': [120, 60, 30, 14], 'restore_bbox': [120, 60, 30, 14], 'owner_evidence': _mk_proof(p1dir, 't-near', [120, 60, 30, 14])}
    sel_bad = {'id': 'r-bad', 'kind': 'asset', 'bbox': [145, 60, 20, 20], 'restore_bbox': [145, 60, 20, 20], 'owner_evidence': _mk_proof(p1dir, 'r-bad', [145, 60, 20, 20])}
    write(p1dir / 'scene.json', {**base_scene, 'selected': {'t-far': sel_far, 't-near': sel_near, 'r-bad': sel_bad}})
    write(p1dir / 'slideir.json', {'nodes': []})

    # Copy identical owner proofs for pass2 (same sha256 -> unmodified!)
    shutil.copyfile(p1dir / 't-far.npz', p2dir / 't-far.npz')
    shutil.copyfile(p1dir / 't-near.npz', p2dir / 't-near.npz')
    write(p2dir / 'scene.json', {**base_scene, 'selected': {'t-far': sel_far, 't-near': sel_near}})
    write(p2dir / 'slideir.json', {'nodes': []})

    calls = []

    def fake_compile(irs, out_dir, texts):
        out_dir.mkdir(parents=True, exist_ok=True)
        pptx_p = out_dir / 'out.pptx'
        pptx_p.write_bytes(b'pptx-' + str(len(calls)).encode())
        return {'pptx': str(pptx_p)}

    def fake_office(state, pptx, irs, directory, *, target_ids=None):
        calls.append(target_ids)
        if len(calls) == 1:
            return {
                'status': 'REVIEW_REQUIRED', 'files': {},
                'residuals': [[
                    {'id': 't-far', 'residual_check_status': 'PASS', 'residual_pixels': 0, 'source_bound': True, 'deleted': True},
                    {'id': 't-near', 'residual_check_status': 'PASS', 'residual_pixels': 0, 'source_bound': True, 'deleted': True},
                    {'id': 'r-bad', 'residual_check_status': 'FAIL', 'residual_pixels': 12},
                ]],
                'replacements': {'records': []}, 'office': {},
            }
        return {
            'status': 'PASS', 'files': {},
            'residuals': [[
                {'id': oid, 'residual_check_status': 'PASS', 'residual_pixels': 0, 'source_bound': True, 'deleted': True}
                for oid in (target_ids or [])
            ]],
            'replacements': {'records': []}, 'office': {},
        }

    def fake_restage(svc, state, current, iso_dir, cand, res):
        cand['ir'] = str(p2dir / 'slideir.json')
        cand['object_ids'] = ['t-far', 't-near']
        cand['object_id'] = 't-far'
        cand['isolation_attempts'] = 1
        cand['files'] = {str(p): digest(p) for p in p2dir.rglob('*') if p.is_file()}
        return True

    monkeypatch.setattr('product.runtime.compiler.compile_local', fake_compile)
    monkeypatch.setattr(RuntimeService, 'office_check', staticmethod(fake_office))
    monkeypatch.setattr(bb, 'restage_batch_candidate_subset', fake_restage)

    svc = RuntimeService(jobs_root)
    state = RuntimeState(job_dir)
    cur = {
        'fonts': str(fonts_p),
        'pages': [{'scene': str(bdir / 'scene.json'), 'ir': str(bdir / 'slideir.json'), 'snapshot': snap}],
        'candidates': {'cand-p2': {
            'id': 'cand-p2', 'page': 1, 'object_id': 't-far', 'object_ids': ['t-far', 't-near', 'r-bad'], 'kind': 'batch',
            'status': 'STAGED_LOCAL', 'ir': str(p1dir / 'slideir.json'),
            'parent_ir_sha256': digest(bdir / 'slideir.json'), 'parent_scene_sha256': digest(bdir / 'scene.json'),
            'code': svc.code_binding(), 'files': {},
        }},
    }
    res = svc.validate_candidate(state, cur, job_dir / 'val', candidate_id='cand-p2')
    assert res['status'] == 'PASS'
    assert calls[0] is None
    assert calls[1] == ['t-near']
    merged_ids = {r['id'] for r in res['residuals'][0]}
    assert merged_ids == {'t-far', 't-near'}


def test_region_crop_decomposes_tinted_circle_badges_expands_bounded_composite_and_fits_accent_bar():
    from product.recovery.region_crop import _detect_icon_in_tinted_circle, grow_seed
    from product.recovery.native_shapes import try_fit_native_shape_v2

    # 1. Tinted light-blue circle badge with high-contrast dark-blue inner icon
    h, w = 112, 112
    backing = np.full((h, w, 3), [234, 244, 253], dtype=np.uint8)
    pixels = backing.copy()
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(xx - 55.5, yy - 55.5)
    pixels[dist <= 51.0] = [221, 237, 252]  # subtle 13-RGB tinted circle
    pixels[(yy >= 34) & (yy < 78) & (xx >= 32) & (xx < 80)] = [14, 92, 204]  # dark-blue icon
    # Must still decompose when layout detector tags the circular icon badge as `layout:image`
    res_inner = _detect_icon_in_tinted_circle(pixels, backing, np.zeros((h, w), bool), ['layout:image'])
    assert res_inner is not None
    (ix0, iy0, iw, ih), inner_bg = res_inner
    assert 28 <= ix0 <= 34 and 30 <= iy0 <= 36 and 46 <= iw <= 56 and 42 <= ih <= 52
    assert np.allclose(inner_bg.mean(axis=(0, 1)), [221, 237, 252], atol=2.0)

    # 1b. Off-center / clipped tinted badge crop (77x109, like PP-DocLayoutV3 on Shield badge)
    clip_px = pixels[2:111, 20:97]
    clip_bg = backing[2:111, 20:97]
    res_clip = _detect_icon_in_tinted_circle(clip_px, clip_bg, np.zeros(clip_px.shape[:2], bool), ['layout:image'])
    assert res_clip is not None
    (cx0, cy0, cw, ch), clip_inner_bg = res_clip
    assert 8 <= cx0 <= 16 and 28 <= cy0 <= 36 and 46 <= cw <= 56 and 42 <= ch <= 52
    assert np.allclose(clip_inner_bg.mean(axis=(0, 1)), [221, 237, 252], atol=2.0)

    # 2. Bounded composite illustration (e.g. diagonal rising arrow 220x140 inside 1200x700 slide)
    content = np.zeros((700, 1200), dtype=bool)
    for k in range(180):
        ry = 360 - int(k * 0.6)
        rx = 700 + k
        content[ry - 8:ry + 8, rx - 8:rx + 8] = True
    box_g, cuts_g = grow_seed(content, [695, 320, 64, 64], (1200, 700))
    assert not cuts_g
    assert box_g[2] >= 170 and box_g[3] >= 100

    # 3. Compact high-contrast vertical accent bar `|` (8x27) fits as native `rect`
    plate = np.full((80, 80, 3), [234, 244, 252], dtype=np.uint8)
    plate[20:47, 30:38] = [14, 72, 165]
    par = np.full((27, 8, 3), [234, 244, 252], dtype=np.float64)
    bar_fit = try_fit_native_shape_v2(plate, [30, 20, 8, 27], par, [], np.zeros((27, 8), bool), cand_kind='line')
    assert bar_fit is not None
    assert bar_fit['geometry'] == 'rect'
    assert bar_fit['fill'] == '0E48A5'




