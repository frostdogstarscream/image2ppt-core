"""Synthetic region-crop transactions; actual Office evidence is separate."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.object_contracts import ClosedObject
from product.parallel_contracts import SceneGraph
from product.parallel_fusion import fuse
from product.parallel_rebuild import build
from product.parallel_residual_gate import inspect_residuals, residual_accepted
from product.recovery.region_crop import grow_seed, stage_region_crops, _content
from test_parallel import branches


def _icon(box):
    return ClosedObject(id=box[0], description='complex graphic', prompt_en='complex graphic', kind='illustration',
                        bbox=box[1], layers=['layer-01.png'], components=['whole'],
                        text_policy='exclude_independent_text',
                        parts=[{'name': 'whole', 'bbox': box[1], 'layers': ['layer-01.png']}])


def scene(tmp_path, *, band=False):
    rng = np.random.default_rng(7)
    src = np.full((120, 240, 3), 255, np.uint8)
    src[20:60, 20:60] = rng.integers(0, 200, (40, 40, 3))
    src[70:100, 150:200] = rng.integers(40, 160, (30, 50, 3))
    if band:
        src[105:112, :] = rng.integers(0, 200, (7, 240, 3))
    (tmp_path / 'in').mkdir()
    path = tmp_path / 'in' / 'source.png'
    Image.fromarray(src).save(path)
    snap, evidence = branches(path)
    for box in [('a-1', [22, 22, 30, 30]), ('a-2', [152, 72, 40, 20])]:
        evidence['asset'].objects.append(_icon(box))
    graph = fuse(snap, evidence, {'lines': []})
    baseline = tmp_path / 'baseline'
    build(graph, snap, {'fonts': []}, baseline, strategy='off', text_candidates=False)
    return src, baseline


def test_isolated_complex_regions_become_movable_crops(tmp_path):
    src, baseline = scene(tmp_path)
    ir_file, staged, report = stage_region_crops(baseline, tmp_path / 'crops')
    assert len(staged) == 2
    out = ir_file.parent
    graph = SceneGraph.model_validate_json((out / 'scene.json').read_text('utf8'))
    plate = np.array(Image.open(out / 'source-plate.png').convert('RGB'))
    for rid in staged:
        record = graph.selected[rid]
        assert record['region_crop'] and record['move_safety'] == 'MOVABLE'
        x, y, w, h = record['bbox']
        # Grown from the seed to the complete connected graphic.
        crop = np.array(Image.open(out / record['asset_reference']['path']).convert('RGBA'))
        assert (crop[..., 3] == 255).all()
        assert np.array_equal(crop[..., :3], src[y:y + h, x:x + w])
        assert np.abs(plate[y:y + h, x:x + w].astype(int) - 255).max() <= 4
        assert record['owner_evidence']['method'] == 'region-crop-lateral-backing/1'
    boxes = sorted(graph.selected[r]['bbox'] for r in staged)
    assert boxes[0][0] <= 20 and boxes[0][0] + boxes[0][2] >= 60
    # Outside the crops nothing changed.
    union = np.zeros(src.shape[:2], bool)
    for rid in staged:
        x, y, w, h = graph.selected[rid]['bbox']; union[y:y + h, x:x + w] = True
    assert np.array_equal(plate[~union], src[~union])
    ir = json.loads(ir_file.read_text('utf8'))
    z = {n['id']: n['z'] for n in ir['nodes']}
    assert all(z[r] > z['source-plate'] for r in staged)
    with np.load(out / 'ownership.npz', allow_pickle=False) as data:
        assert (data['ownership'][union] == 5).all()


def test_move_unsafe_crop_erases_nothing_and_is_not_a_residual_pass(tmp_path):
    src, baseline = scene(tmp_path)
    ir_file, staged, _ = stage_region_crops(baseline, tmp_path / 'crops', allow_movable=False, allow_move_unsafe=True)
    out = ir_file.parent
    plate = np.array(Image.open(out / 'source-plate.png').convert('RGB'))
    assert np.array_equal(plate, src)
    graph = SceneGraph.model_validate_json((out / 'scene.json').read_text('utf8'))
    assert all(graph.selected[r]['move_safety'] == 'MOVE_UNSAFE' and 'owner_evidence' not in graph.selected[r]
               for r in staged)
    rows = inspect_residuals(ir_file, tmp_path / 'office', 1, tmp_path / 'residual.json')
    assert {r['residual_check_status'] for r in rows} == {'NOT_APPLICABLE'}
    assert all(residual_accepted(r) for r in rows)
    assert not residual_accepted({'residual_check_status': 'NOT_APPLICABLE'})
    assert not residual_accepted({'residual_check_status': 'NOT_APPLICABLE', 'move_safety': 'MOVE_UNSAFE'})
    assert not residual_accepted({'residual_check_status': 'UNKNOWN', 'move_safety': 'MOVE_UNSAFE'})

    # Tampering with plate beneath a MOVE_UNSAFE crop must fail residual inspection.
    tampered = plate.copy()
    rx, ry, rw, rh = graph.selected[staged[0]]['bbox']
    tampered[ry:ry + rh, rx:rx + rw] = 0
    Image.fromarray(tampered).save(out / 'source-plate.png')
    graph.raster_backing['sha256'] = ''
    (out / 'scene.json').write_text(json.dumps(graph.model_dump()), 'utf8')
    tampered_rows = inspect_residuals(ir_file, tmp_path / 'office', 1, tmp_path / 'residual-bad.json')
    assert any(r['residual_check_status'] == 'FAIL' for r in tampered_rows)


def test_default_disallows_move_unsafe_crops(tmp_path):
    _, baseline = scene(tmp_path)
    _, staged, report = stage_region_crops(baseline, tmp_path / 'crops-default', allow_movable=False)
    assert staged == []
    assert any(r.get('reason') == 'MOVE_UNSAFE_DISABLED' for r in report)
    assert not (tmp_path / 'crops-default').exists()


def test_repairable_text_is_protected_from_seeding_and_swallowing(tmp_path):
    from product.contracts import Style, Text
    from product.recovery.region_crop import collect_crop_seeds, plan_region_crops

    src = np.full((180, 260, 3), 255, np.uint8)
    rng = np.random.default_rng(11)
    src[30:90, 20:140] = rng.integers(0, 180, (60, 120, 3))
    (tmp_path / 'in').mkdir()
    path = tmp_path / 'in' / 'src.png'
    Image.fromarray(src).save(path)
    snap, evidence = branches(path)
    evidence['text'].styles.append(Style(id='s1', role='body', ea='Arial', latin='Arial', size_px=16.0, bold=False, color='101010'))
    evidence['text'].texts.append(Text(id='t-rep', text='RepairMe', bbox=[115, 45, 70, 22], style='s1', z=3))
    evidence['text'].members['t-rep'] = ['l-1']
    evidence['asset'].objects.append(_icon(('a-left', [25, 35, 70, 45])))
    graph = fuse(snap, evidence, {'lines': [{'id': 'l-1', 'text': 'RepairMe', 'bbox': [115, 45, 70, 22], 'polygon': [[115, 45], [185, 45], [185, 67], [115, 67]], 'score': 0.99}]})
    assert graph.ownership['t-rep'] == 'RESOLVED'

    seeds, protected = collect_crop_seeds(graph, unrecoverable_text_ids=set())
    assert 't-rep' in protected
    assert all(s['id'] != 't-rep' for s in seeds)
    planned, _ = plan_region_crops(graph, src, seeds, protected_text_ids=protected)
    assert len(planned) == 1
    cx, cy, cw, ch = planned[0]['box']
    assert cx + cw <= 115  # shrunk away from protected repairable text


def test_movable_crop_on_committed_card_preserves_plate_layer_algebra(tmp_path):
    from product.contracts import Shape
    from product.object_extract import rectangle

    size = (260, 180)
    canvas_bg = np.array([255, 255, 255], np.uint8)
    card_bg = np.array([235, 240, 248], np.uint8)
    src = np.full((180, 260, 3), canvas_bg, np.uint8)
    src[20:160, 20:240] = card_bg
    src[50:110, 60:140] = np.random.default_rng(3).integers(10, 150, (60, 80, 3))
    (tmp_path / 'in').mkdir()
    path = tmp_path / 'in' / 'card.png'
    Image.fromarray(src).save(path)
    snap, evidence = branches(path)
    evidence['surface'].shapes.append(Shape(id='s-card', bbox=[20, 20, 220, 140], z=1, geometry='rect', fill='EBF0F8'))
    evidence['asset'].objects.append(_icon(('a-in-card', [62, 52, 76, 56])))
    graph = fuse(snap, evidence, {'lines': []})
    baseline = tmp_path / 'baseline'
    build(graph, snap, {'fonts': []}, baseline, strategy='off', text_candidates=False)

    # Simulate committed surface card over canvas source-plate (canvas pixels remain in plate beneath card).
    bgraph = SceneGraph.model_validate_json((baseline / 'scene.json').read_text('utf8'))
    plate = np.array(Image.open(baseline / 'source-plate.png').convert('RGB'))
    plate[20:160, 20:240] = canvas_bg
    Image.fromarray(np.dstack([plate, np.full((180, 260), 255, np.uint8)])).save(baseline / 'source-plate.png')
    from product.materialize import digest
    bgraph.raster_backing['sha256'] = digest(baseline / 'source-plate.png')
    bgraph.selected['s-card'] = {
        'id': 's-card', 'kind': 'surface', 'bbox': [20, 20, 220, 140],
        'state': 'COMMITTED', 'source_sha256': bgraph.source_sha256,
    }
    (baseline / 'scene.json').write_text(json.dumps(bgraph.model_dump()), 'utf8')

    _, staged, report = stage_region_crops(baseline, tmp_path / 'crops-card')
    assert len(staged) == 1
    out = tmp_path / 'crops-card'
    cgraph = SceneGraph.model_validate_json((out / 'scene.json').read_text('utf8'))
    rec = cgraph.selected[staged[0]]
    assert rec['move_safety'] == 'MOVABLE'
    # Layer-algebra invariant: source-plate under committed card remains untouched canvas white.
    out_plate = np.array(Image.open(out / 'source-plate.png').convert('RGB'))
    assert np.array_equal(out_plate[50:110, 60:140], np.broadcast_to(canvas_bg, (60, 80, 3)))
    # While owner backing_patch stores the card color revealed on deletion!
    with np.load(out / rec['owner_evidence']['file'], allow_pickle=False) as d:
        assert np.abs(d['backing_patch'].astype(int) - card_bg).max() <= 3


def test_transparent_upgrade_for_circle_icon_vs_framed_photo(tmp_path):
    import cv2
    src = np.full((160, 280, 3), 250, np.uint8)
    # 1) Unframed circular graphic with large transparent perimeter inside tight box -> upgrades to RGBA asset.
    cv2.circle(src, (60, 80), 28, (30, 120, 210), -1)
    # 2) Solid 4-edge framed rectangle (photo-like layout:image) -> stays opaque MOVABLE crop.
    src[30:110, 160:240] = np.random.default_rng(5).integers(20, 180, (80, 80, 3))
    (tmp_path / 'in').mkdir()
    path = tmp_path / 'in' / 'mix.png'
    Image.fromarray(src).save(path)
    snap, evidence = branches(path)
    evidence['asset'].objects.append(_icon(('a-circle', [32, 52, 56, 56])))
    graph = fuse(snap, evidence, {'lines': []})
    baseline = tmp_path / 'baseline'
    build(graph, snap, {'fonts': []}, baseline, strategy='off', text_candidates=False)

    seeds = [
        {'id': 'a-circle', 'kind': 'asset', 'bbox': [32, 52, 56, 56], 'text': ''},
        {'id': 'layout:l-photo', 'kind': 'layout:image', 'bbox': [160, 30, 80, 80], 'text': ''},
    ]
    _, staged, report = stage_region_crops(baseline, tmp_path / 'crops-up', seeds=seeds)
    assert len(staged) == 2
    out = tmp_path / 'crops-up'
    cgraph = SceneGraph.model_validate_json((out / 'scene.json').read_text('utf8'))
    by_seed = {cgraph.selected[s]['seed_ids'][0]: cgraph.selected[s] for s in staged}
    assert by_seed['a-circle']['region_crop'] is False
    circle_png = np.array(Image.open(out / by_seed['a-circle']['asset_reference']['path']).convert('RGBA'))
    assert (circle_png[..., 3] == 0).mean() >= 0.12
    assert by_seed['layout:l-photo']['region_crop'] is True
    assert by_seed['layout:l-photo']['move_safety'] == 'MOVABLE'


def test_oversized_merge_rejected_and_single_seed_subsplit_and_skip_seed_ids(tmp_path):
    from product.recovery.region_crop import plan_region_crops

    src = np.full((120, 240, 3), 255, np.uint8)
    src[20:50, 20:80] = 40
    src[20:50, 130:190] = 60
    # Single large region spanning wide horizontal blocks separated by blank gutter.
    seeds = [
        {'id': 'a-one', 'kind': 'asset', 'bbox': [20, 20, 60, 30], 'text': ''},
        {'id': 'a-two', 'kind': 'asset', 'bbox': [130, 20, 60, 30], 'text': ''},
    ]
    src_obj_graph = branches(tmp_path / 'dummy.png' if False else (lambda p: (Image.fromarray(src).save(p), p)[1])(tmp_path / 's.png'))[0]
    g = SceneGraph.model_validate({'schema': 'SceneGraph/1', 'source_sha256': src_obj_graph['source_sha256'],
        'image_size': [240, 120], 'method_version': 't/1', 'ownership': {},
        'text': {'schema': 'TextEvidence/1', 'source_sha256': src_obj_graph['source_sha256'], 'image_size': [240, 120], 'method_version': 't/1'},
        'asset': {'schema': 'AssetEvidence/1', 'source_sha256': src_obj_graph['source_sha256'], 'image_size': [240, 120], 'method_version': 't/1'},
        'surface': {'schema': 'SurfaceEvidence/1', 'source_sha256': src_obj_graph['source_sha256'], 'image_size': [240, 120], 'method_version': 't/1', 'background': 'FFFFFF'}})

    # One single-seed box spanning 20..190 (width=170, area > 0.15 * page) splits cleanly into 2 components!
    big_seed = [{'id': 'layout:big', 'kind': 'layout:image', 'bbox': [18, 18, 174, 34], 'text': ''}]
    accepted, rep = plan_region_crops(g, src, big_seed, max_area_fraction=0.14)
    assert len(accepted) >= 2
    assert all(c['box'][2] * c['box'][3] <= 0.14 * 240 * 120 for c in accepted)

    # skip_seed_ids skips matching seed ID and ignores sequential r-001.
    kept_without_skip, _ = plan_region_crops(g, src, seeds, skip_seed_ids={'r-001'})
    assert len(kept_without_skip) == 2
    kept_with_skip, _ = plan_region_crops(g, src, seeds, skip_seed_ids={'a-one'})
    assert len(kept_with_skip) == 1 and kept_with_skip[0]['seed_ids'] == ['a-two']


def test_continuing_content_keeps_small_seed_and_stays_move_unsafe(tmp_path):
    src, _ = scene(tmp_path, band=True)
    content = _content(src)
    box, cuts = grow_seed(content, [100, 104, 20, 10], (240, 120))
    assert cuts and box[2] <= 24
    box, cuts = grow_seed(content, [30, 30, 10, 10], (240, 120))
    assert not cuts and box[0] <= 20 and box[0] + box[2] >= 60


def test_no_remaining_content_means_no_crop(tmp_path):
    _, baseline = scene(tmp_path)
    graph = SceneGraph.model_validate_json((baseline / 'scene.json').read_text('utf8'))
    _, staged, report = stage_region_crops(baseline, tmp_path / 'crops',
                                           seeds=[{'id': 'blank', 'kind': 'asset', 'bbox': [100, 10, 20, 10]}])
    assert staged == [] and report[0]['reason'] == 'NO_VISIBLE_CONTENT'
    assert not (tmp_path / 'crops').exists() and graph.selected == {}


def test_omni_surface_proposal_uses_current_canvas_and_ring_adapter(tmp_path):
    from PIL import ImageDraw
    from product.recovery.marked_surfaces import stage_marked_surfaces
    source = tmp_path/'input.png'
    image = Image.new('RGB', (260, 148), 'white')
    ImageDraw.Draw(image).rounded_rectangle((40, 35, 218, 100), radius=5, fill='#daeaf4', outline='#367cbc', width=2)
    image.save(source)
    snap, evidence = branches(source)
    graph = fuse(snap, evidence, {'lines':[]})
    base = tmp_path/'base'
    build(graph, snap, {'fonts':[]}, base, strategy='off', text_candidates=False)
    ir, ids = stage_marked_surfaces(base, tmp_path/'staged', [{'id':'m-card','bbox':[40,35,179,66]}], set())
    assert ids and any(obj['type']=='shape' for obj in json.loads(ir.read_text())['nodes'])
