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
    ir_file, staged, _ = stage_region_crops(baseline, tmp_path / 'crops', allow_movable=False, include_in_place=True)
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
    assert not residual_accepted({'residual_check_status': 'UNKNOWN', 'move_safety': 'MOVE_UNSAFE'})


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

def test_parent_fill_and_canvas_have_separate_delete_references(tmp_path):
    from product.materialize import digest
    from product.recovery.region_crop import _under_parent
    source = np.full((90, 120, 3), 255, np.uint8)
    card = [8, 8, 100, 70]
    source[8:78, 8:108] = (212, 232, 248)
    source[32:42, 50:66] = (20, 75, 130)
    proof = tmp_path / 'parent.npz'
    np.savez_compressed(proof,backing_patch=np.full((70,100,3),255,np.uint8))
    parent = {'kind':'surface','method':'container-surface-confirmed/1','restore_bbox':card,
              'payload':{'fill':'D4E8F8','geometry':'roundRect','corner_radius_px':5},
              'owner_evidence':{'file':proof.name,'sha256':digest(proof)}}
    visible, canvas = _under_parent(parent,[48,30,20,15],source,tmp_path)
    assert np.all(visible == [212,232,248]) and np.all(canvas == 255)
    assert _under_parent(parent,[8,8,20,15],source,tmp_path) is None


def test_default_delivery_skips_unsafe_and_keeps_stable_crop_ids(tmp_path):
    _, baseline = scene(tmp_path)
    ir, ids, details = stage_region_crops(baseline,tmp_path/'unsafe',allow_movable=False)
    assert not ids and all(r['reason']=='IN_PLACE_ONLY' for r in details)
    seeds=[{'id':'a-1','kind':'asset','bbox':[22,22,30,30]},
           {'id':'a-2','kind':'asset','bbox':[152,72,40,20]}]
    _, both, _ = stage_region_crops(baseline,tmp_path/'both',seeds=seeds)
    _, only, _ = stage_region_crops(baseline,tmp_path/'only',seeds=seeds[1:])
    assert only==both[1:]

def test_cropping_inside_native_card_does_not_paint_card_color_on_canvas(tmp_path):
    from product.contracts import Shape
    from product.materialize import digest, write
    from product.parallel_rebuild import adapt
    src = np.full((120,240,3),255,np.uint8)
    src[10:102,15:215] = [212,232,248]
    src[44:63,72:96] = [5,100,180]
    path = tmp_path/'source.png'
    Image.fromarray(src).save(path)
    snap, evidence = branches(path)
    shape = Shape(id='s-card',bbox=[15,10,200,92],z=1,geometry='rect',fill='D4E8F8')
    evidence['surface'].shapes.append(shape)
    g = fuse(snap,evidence,{'lines':[]})
    b = tmp_path/'before'
    build(g,snap,{'fonts':[]},b,strategy='off',text_candidates=False,surface_candidates=False)
    g = SceneGraph.model_validate_json((b/'scene.json').read_text('utf8'))
    mask=np.zeros(src.shape[:2],bool);mask[10:102,15:215]=True
    proof=b/'card.npz'
    np.savez_compressed(proof,mask=mask,restore_bbox=[15,10,200,92],backing_patch=np.full((92,200,3),255,np.uint8))
    rec={'id':'s-card','kind':'surface','method':'container-surface-confirmed/1','bbox':shape.bbox,
         'restore_bbox':shape.bbox,'payload':shape.model_dump(),'owner_evidence':{'file':proof.name,'sha256':digest(proof)}}
    g.selected['s-card']=rec
    plate=b/g.raster_backing['path']
    Image.fromarray(np.full_like(src,255)).save(plate)
    g.raster_backing['sha256']=digest(plate)
    write(b/'scene.json',g.model_dump());write(b/'slideir.json',adapt(g,b))
    ir,ids,_=stage_region_crops(b,tmp_path/'after',seeds=[{'id':'manual-icon','kind':'asset','bbox':[70,42,28,23]}])
    assert len(ids)==1
    g=SceneGraph.model_validate_json((ir.parent/'scene.json').read_text('utf8'))
    child=g.selected[ids[0]]
    assert child['payload']['background_parent_id']=='s-card'
    assert np.all(np.asarray(Image.open(ir.parent/g.raster_backing['path']).convert('RGB'))==255)
    with np.load(ir.parent/child['owner_evidence']['file']) as data:
        assert np.all(data['backing_patch']==[212,232,248])
