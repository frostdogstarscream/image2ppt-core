"""Small-first opaque region crops for content that remains in the page raster.

A region crop copies the current safe plate pixels of a rectangle into an
independent, opaque picture at the identical position. It asserts no semantics
(text inside it stays intrinsic and is never re-typeset) and therefore needs no
text, mask or object approval:

* ``MOVE_UNSAFE``: the plate keeps identical pixels underneath. Nothing is erased;
  the delivered page is pixel-identical. Moving the crop would reveal a copy,
  which is reported explicitly and excluded from residual PASS claims.
* ``MOVABLE``: only when the crop is isolated from committed objects and an
  independent lateral/surface backing model fits the surrounding ring tightly.
  The whole rectangle's pixels move into the crop and the plate receives that
  backing, with a normal owner receipt for actual-deletion residual checks.

Crops start from the smallest seed (an object that did not become editable) and
grow only to avoid cutting connected visible content or partially covering a
remaining text line.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

from ..materialize import digest, write
from ..parallel_store import key
from ..object_contracts import ClosedObject
from ..object_extract import rectangle
from ..parallel_contracts import SceneGraph

METHOD = 'source-plate-region-crop/1'
MAX_GROW_PX = 48
MAX_AREA_FRACTION = 0.25
MIN_CONTENT_FRACTION = 0.01
MOVABLE_RING_ERROR = 4.0


def _content(plate: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (5, 5))
    var = cv2.blur(gray * gray, (5, 5)) - mean * mean
    return np.sqrt(np.maximum(var, 0)) > 6.0


def _clip(box, size):
    w_img, h_img = size
    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x1 = min(w_img, int(np.ceil(box[0] + box[2])))
    y1 = min(h_img, int(np.ceil(box[1] + box[3])))
    return [x0, y0, max(0, x1 - x0), max(0, y1 - y0)]


def _union(a, b):
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return [x0, y0, x1 - x0, y1 - y0]


def _intersects(a, b, gap=0):
    return not (a[0] + a[2] + gap <= b[0] or b[0] + b[2] + gap <= a[0]
                or a[1] + a[3] + gap <= b[1] or b[1] + b[3] + gap <= a[1])


def _contains(outer, inner):
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[0] + outer[2] >= inner[0] + inner[2] and outer[1] + outer[3] >= inner[1] + inner[3])


def grow_seed(content: np.ndarray, seed, size, *, max_grow=MAX_GROW_PX):
    """Grow a seed box to its connected visible content; return (box, cuts_content)."""
    seed = _clip([seed[0] - 2, seed[1] - 2, seed[2] + 4, seed[3] + 4], size)
    window = _clip([seed[0] - max_grow, seed[1] - max_grow, seed[2] + 2 * max_grow, seed[3] + 2 * max_grow], size)
    wx, wy, ww, wh = window
    local = cv2.dilate(content[wy:wy + wh, wx:wx + ww].astype(np.uint8), np.ones((3, 3), np.uint8))
    count, labels = cv2.connectedComponents(local, connectivity=8)
    sx, sy = seed[0] - wx, seed[1] - wy
    hit = [int(v) for v in np.unique(labels[sy:sy + seed[3], sx:sx + seed[2]]) if v > 0]
    if not hit:
        return seed, False
    mask = np.isin(labels, hit)
    touches = ((wx > 0 and mask[:, 0].any()) or (wx + ww < size[0] and mask[:, -1].any())
               or (wy > 0 and mask[0].any()) or (wy + wh < size[1] and mask[-1].any()))
    if touches:
        # Content continues beyond the bounded neighbourhood (texture, long rule,
        # page-wide decoration). Small-first: keep the seed instead of swallowing it.
        return seed, True
    ys, xs = np.nonzero(mask)
    grown = [wx + int(xs.min()), wy + int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    return _clip(_union(seed, [grown[0] - 1, grown[1] - 1, grown[2] + 2, grown[3] + 2]), size), False


def default_seeds(graph: SceneGraph) -> list[dict]:
    """Unselected visual assets; native text needs an explicit fallback decision."""
    rows = []
    for a in graph.asset.objects:
        if a.id not in graph.selected:
            rows.append({'id': a.id, 'kind': 'asset', 'bbox': list(a.bbox), 'text': a.intrinsic_text})
    return rows


def plan_region_crops(graph: SceneGraph, plate: np.ndarray, seeds: list[dict] | None = None, *,
                      max_area_fraction=MAX_AREA_FRACTION, protected_text_boxes=()) -> tuple[list[dict], list[dict]]:
    size = (plate.shape[1], plate.shape[0])
    content = _content(plate)
    seeds = default_seeds(graph) if seeds is None else seeds
    text_seeds = {s['id'] for s in seeds if s['kind'] == 'text'}
    protected = [t.bbox for t in graph.text.texts if t.id not in graph.selected and t.id not in text_seeds]
    protected.extend(protected_text_boxes)
    committed = [(row.get('restore_bbox') or row.get('bbox'), row.get('kind') == 'surface' or row.get('container_layer'))
                 for row in graph.selected.values() if row.get('restore_bbox') or row.get('bbox')]
    max_area = max_area_fraction * size[0] * size[1]
    crops, report = [], []
    for seed in seeds:
        box, cuts = grow_seed(content, seed['bbox'], size)
        if any(_intersects(box, t) for t in protected):
            report.append({'seed_ids': [seed['id']], 'bbox': box, 'state': 'SKIPPED', 'reason': 'PROTECTED_TEXT'})
            continue
        crops.append({'box': box, 'seed_ids': [seed['id']], 'texts': [seed.get('text') or ''], 'cuts': cuts})
    changed = True
    while changed:
        changed = False
        merged = []
        for crop in crops:
            for other in merged:
                if not _intersects(other['box'], crop['box'], gap=2):
                    continue
                union = _union(other['box'], crop['box'])
                if union[2] * union[3] > max_area or any(_intersects(union,t) for t in protected):
                    continue
                if any(_intersects(union, box) and not (parent and _contains(box, union))
                       for box, parent in committed):
                    continue
                other['box'] = union
                other['seed_ids'].extend(i for i in crop['seed_ids'] if i not in other['seed_ids'])
                other['texts'].extend(crop['texts'])
                other['cuts'] |= crop['cuts']
                changed = True
                break
            else:
                merged.append(crop)
        crops = merged
    accepted = []
    page = float(size[0] * size[1])
    for crop in crops:
        x, y, w, h = crop['box']
        row = {'seed_ids': crop['seed_ids'], 'bbox': crop['box']}
        if w < 2 or h < 2:
            report.append({**row, 'state': 'SKIPPED', 'reason': 'EMPTY'}); continue
        if w * h > max_area_fraction * page:
            report.append({**row, 'state': 'SKIPPED', 'reason': 'TOO_LARGE_FOR_SMALL_FIRST_CROP'}); continue
        if float(content[y:y + h, x:x + w].mean()) < MIN_CONTENT_FRACTION:
            report.append({**row, 'state': 'SKIPPED', 'reason': 'NO_VISIBLE_CONTENT'}); continue
        accepted.append(crop)
    return accepted, report


def _under_parent(record, box, source, baseline):
    # Only a single proven flat interior can be treated as backing for its child.
    shape = record.get('payload', {})
    fill = shape.get('fill', '')
    if (record.get('kind') != 'surface' or record.get('method') != 'container-surface-confirmed/1'
            or shape.get('fill_gradient') or shape.get('geometry') not in ('rect','roundRect')
            or len(fill) != 6):
        return None
    x,y,w,h = box
    px,py,pw,ph = map(int,record['restore_bbox'])
    margin = int(np.ceil(max(shape.get('corner_radius_px') or 0, 2*(shape.get('line_width_pt') or 0))))+4
    if not _contains([px+margin,py+margin,pw-2*margin,ph-2*margin],box):
        return None
    color = np.array(list(bytes.fromhex(fill)),dtype=np.uint8)
    border = source[y-2:y+h+2,x-2:x+w+2]
    ring = np.concatenate([border[:2].reshape(-1,3),border[-2:].reshape(-1,3),
                           border[2:-2,:2].reshape(-1,3),border[2:-2,-2:].reshape(-1,3)])
    if np.percentile(np.abs(ring.astype(float)-color).max(1),95)>MOVABLE_RING_ERROR:
        return None
    ref = record['owner_evidence']
    proof = (baseline/ref['file']).resolve()
    if not proof.is_relative_to(baseline.resolve()) or digest(proof)!=ref['sha256']:
        raise ValueError('parent backing binding mismatch')
    with np.load(proof,allow_pickle=False) as receipt:
        canvas = receipt['backing_patch'][y-py:y-py+h,x-px:x-px+w].copy()
    if canvas.shape != (h,w,3):
        raise ValueError('parent backing footprint mismatch')
    return np.broadcast_to(color,(h,w,3)).copy(),canvas


def stage_region_crops(baseline: str | Path, output: str | Path, *, seeds: list[dict] | None = None,
                       allow_movable: bool = True, include_in_place: bool = False, max_area_fraction=MAX_AREA_FRACTION,
                       skip_ids: set[str] | None = None, ocr_lines=None) -> tuple[Path, list[str], list[dict]]:
    """Stage crops on a built baseline directory; return (slideir path, staged ids, report)."""
    from ..object_unmix import backing_from_evidence
    from ..parallel_fusion import fuse_incremental_scene
    from ..parallel_rebuild import adapt, array_sha
    baseline, output = Path(baseline), Path(output)
    skip = set(skip_ids or ())
    graph = SceneGraph.model_validate_json((baseline / 'scene.json').read_text('utf8'))
    rb = graph.raster_backing or {}
    plate_file = (baseline / rb.get('path', 'source-plate.png')).resolve()
    if not plate_file.is_relative_to(baseline.resolve()) or digest(plate_file) != rb.get('sha256'):
        raise ValueError('region crop source-plate binding mismatch')
    plate_rgba = np.array(Image.open(plate_file).convert('RGBA'))
    plate = plate_rgba[..., :3].copy()
    source = np.array(Image.open(baseline / 'source.png').convert('RGB'))
    if digest(baseline / 'source.png') != graph.source_sha256 or source.shape != plate.shape:
        raise ValueError('region crop source snapshot mismatch')
    size = (plate.shape[1], plate.shape[0])
    # Crops render above the plate, native surfaces and container layers, and
    # below committed foreground text/assets. Their pixels therefore equal the
    # source appearance, except inside committed foreground owners, where the
    # verified backing is used so no old ink is duplicated beneath edited text.
    foreground = np.zeros(plate.shape[:2], bool)
    for record in graph.selected.values():
        ref = record.get('owner_evidence')
        if not ref or record.get('region_crop') or record.get('container_layer'):
            continue
        if record.get('kind') not in ('text', 'asset'):
            continue
        proof = (baseline / ref['file']).resolve()
        if not proof.is_relative_to(baseline.resolve()) or digest(proof) != ref['sha256']:
            raise ValueError('region crop owner evidence binding mismatch')
        with np.load(proof, allow_pickle=False) as data:
            foreground |= data['mask'].astype(bool)
    appearance = np.where(foreground[..., None], plate, source)
    protection_file = baseline/'runtime-ocr-protection.json'
    if ocr_lines is None and protection_file.is_file():
        inventory = json.loads(protection_file.read_text(encoding='utf8'))
        if inventory.get('source_sha256') != graph.source_sha256:
            raise ValueError('OCR protection source mismatch')
        ocr_lines = inventory.get('lines',[])
    released = set(graph.selected)
    released.update(s['id'] for s in seeds or [] if s['kind']=='text')
    handled = {lid for tid in released for lid in graph.text.members.get(tid, [])}
    protected_boxes = [line['bbox'] for line in ocr_lines or [] if line['id'] not in handled]
    crops, report = plan_region_crops(graph, appearance, seeds, max_area_fraction=max_area_fraction,
                                      protected_text_boxes=protected_boxes)
    if not crops:
        return baseline / 'slideir.json', [], report
    if output.exists():
        raise ValueError('region crop output already exists')
    shutil.copytree(baseline, output)
    used = {x.id for x in [*graph.text.texts, *graph.asset.objects, *graph.surface.shapes]}
    staged = []
    asset_kinds = {obj.id: obj.kind for obj in graph.asset.objects}
    movable_mask = np.zeros(plate.shape[:2], bool)
    for crop in crops:
        rid = 'r-' + key([graph.source_sha256,sorted(crop['seed_ids']),crop['box']])[:16]
        if rid in used or rid in skip or skip.intersection(crop['seed_ids']):
            continue
        used.add(rid)
        x, y, w, h = crop['box']
        intrinsic = ' / '.join(t.strip() for t in crop['texts'] if t and t.strip())[:200]
        crop_obj = ClosedObject(
            id=rid, kind='illustration', bbox=[x, y, w, h], layers=['source-plate.png'],
            components=['whole'], parts=[{'name': 'whole', 'bbox': [x, y, w, h], 'layers': ['source-plate.png']}],
            exclusions=[], replaces=[], required_owners=[],
            text_policy='intrinsic_label' if intrinsic else 'exclude_independent_text', intrinsic_text=intrinsic,
            description='opaque source region crop (no semantic claim)', prompt_en='opaque source region crop')
        # Host fusion alone derives object ownership and semantic relations.
        fused_try = fuse_incremental_scene(baseline, base_graph=graph, added_assets=[crop_obj])
        if fused_try.ownership.get(rid) != 'RESOLVED':
            report.append({'id': rid, 'state': 'SKIPPED', 'reason': f"host fusion ownership={fused_try.ownership.get(rid)}",
                           'bbox': [x, y, w, h], 'seed_ids': crop['seed_ids']})
            continue
        pixels = appearance[y:y + h, x:x + w].copy()
        safety, backing_meta, backing = 'MOVE_UNSAFE', {'reason': 'not attempted'}, None
        overlaps = [r for r in graph.selected.values() if (r.get('restore_bbox') or r.get('bbox'))
                    and _intersects(crop['box'], r.get('restore_bbox') or r['bbox'])]
        parent_patch = None
        if not allow_movable:
            backing_meta = {'reason': 'movable crops disabled'}
        elif overlaps:
            pair = _under_parent(overlaps[0], crop['box'], source, baseline) if len(overlaps)==1 and not crop['cuts'] else None
            if pair is None:
                backing_meta = {'reason': 'overlaps committed object without proven flat interior'}
            else:
                backing, parent_patch = pair
                backing_meta = {'background_parent_id':overlaps[0]['id'],'method':'verified-native-parent-interior'}
                safety = 'MOVABLE'
        elif crop['cuts']:
            backing_meta = {'reason': 'crop boundary cuts continuing content'}
        else:
            try:
                backing, error, evidence = backing_from_evidence(plate, [x, y, w, h], movable_mask, graph.surface)
                if np.isfinite(error) and error <= MOVABLE_RING_ERROR:
                    safety = 'MOVABLE'
                    backing_meta = {'ring_error': float(error), 'background_evidence': evidence}
                else:
                    backing_meta = {'reason': 'surrounding backing not independently fitted', 'ring_error': float(error)}
            except ValueError as exc:
                backing_meta = {'reason': f'backing unavailable: {exc}'}
        if safety == 'MOVE_UNSAFE' and not include_in_place:
            report.append({'state':'SKIPPED','reason':'IN_PLACE_ONLY','seed_ids':crop['seed_ids'],
                           'bbox':crop['box'],'detail':backing_meta})
            continue
        png = output / f'{rid}.png'
        rgba = np.dstack([pixels, np.full((h, w), 255, np.uint8)])
        if safety == 'MOVABLE' and all(asset_kinds.get(i) in ('icon', 'logo') for i in crop['seed_ids']):
            from reconstruction import extract_rgba
            delta = np.abs(pixels.astype(float) - backing).max(axis=2)
            perimeter = np.concatenate((delta[0], delta[-1], delta[:, 0], delta[:, -1]))
            if not crop['cuts'] and np.max(perimeter) <= MOVABLE_RING_ERROR:
                matte = extract_rgba(pixels, backing, delta > 2)
                alpha = matte[..., 3:4] / 255.0
                rendered = matte[..., :3] * alpha + backing * (1 - alpha)
                if np.abs(rendered - pixels).max() <= 3:
                    rgba = matte
                    backing_meta['matte_method'] = 'verified-backing-unmix'
        Image.fromarray(rgba).save(png)
        graph = fused_try
        record = {
            'id': rid, 'kind': 'asset', 'region_crop': True, 'move_safety': safety,
            'bbox': [x, y, w, h], 'restore_bbox': [x, y, w, h], 'seed_ids': crop['seed_ids'],
            'payload': {'method': METHOD, 'boundary_cuts_content': crop['cuts'], **backing_meta},
            'state': 'STAGED', 'quality': 'REVIEW_REQUIRED', 'human_acceptance': False,
            'source_sha256': graph.source_sha256, 'original_owner': 'source-plate',
            'complete_owner_scope': safety == 'MOVABLE',
            'asset_reference': {'path': png.name, 'sha256': digest(png), 'source_sha256': graph.source_sha256,
                                'bbox': [x, y, w, h], 'crop_offset': [x, y]},
        }
        if safety == 'MOVABLE':
            region = rectangle([x, y, w, h], size)
            proof = output / f'{rid}-owner.npz'
            np.savez_compressed(proof, mask=region, restore_bbox=[x, y, w, h], backing_patch=backing.astype(np.uint8))
            record['mask_sha256'] = array_sha(region.astype('uint8'))
            record['owner_evidence'] = {'file': proof.name, 'sha256': digest(proof), 'method': 'region-crop-lateral-backing/1'}
            plate[y:y + h, x:x + w] = np.clip(np.rint(parent_patch if parent_patch is not None else backing), 0, 255).astype(np.uint8)
            movable_mask |= region
        graph.transactions.append(record)
        graph.selected[rid] = record
        staged.append(rid)
        report.append({'id': rid, 'state': 'STAGED', 'move_safety': safety, 'bbox': [x, y, w, h],
                       'seed_ids': crop['seed_ids'], **backing_meta})
    if not staged:
        shutil.rmtree(output, ignore_errors=True)
        return baseline / 'slideir.json', [], report
    if movable_mask.any():
        Image.fromarray(np.dstack([plate, plate_rgba[..., 3]])).save(output / plate_file.name)
        graph.raster_backing = {**rb, 'sha256': digest(output / plate_file.name)}
        if (output / 'ownership.npz').is_file() and (output / 'ownership.json').is_file():
            with np.load(output / 'ownership.npz', allow_pickle=False) as data:
                ownership, uncertain = data['ownership'].copy(), data['uncertain_text'].copy()
            ownership[movable_mask] = 5
            np.savez_compressed(output / 'ownership.npz', ownership=ownership, uncertain_text=uncertain)
            meta = json.loads((output / 'ownership.json').read_text('utf8'))
            meta['map_sha256'] = digest(output / 'ownership.npz')
            meta.setdefault('classes', {})['5'] = 'region_crop'
            write(output / 'ownership.json', meta)
    write(output / 'region-crops.json', {'schema': 'RegionCrops/1', 'method': METHOD, 'records': report})
    ir = adapt(graph, output)
    write(output / 'scene.json', graph.model_dump())
    write(output / 'slideir.json', ir)
    return output / 'slideir.json', staged, report
