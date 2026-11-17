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
    """Every scene object that did not become an editable owner."""
    rows = []
    for t in graph.text.texts:
        if t.id not in graph.selected:
            rows.append({'id': t.id, 'kind': 'text', 'bbox': list(t.bbox), 'text': t.text})
    for a in graph.asset.objects:
        if a.id not in graph.selected:
            rows.append({'id': a.id, 'kind': 'asset', 'bbox': list(a.bbox), 'text': a.intrinsic_text})
    return rows


def plan_region_crops(graph: SceneGraph, plate: np.ndarray, seeds: list[dict] | None = None, *,
                      max_area_fraction=MAX_AREA_FRACTION) -> tuple[list[dict], list[dict]]:
    size = (plate.shape[1], plate.shape[0])
    content = _content(plate)
    seeds = default_seeds(graph) if seeds is None else seeds
    remaining_text = [s for s in default_seeds(graph) if s['kind'] == 'text']
    crops, report = [], []
    for seed in seeds:
        box, cuts = grow_seed(content, seed['bbox'], size)
        crops.append({'box': box, 'seed_ids': [seed['id']], 'texts': [seed.get('text') or ''], 'cuts': cuts})
    changed = True
    while changed:
        changed = False
        # Never cut a remaining text line: absorb partially covered lines whole.
        for crop in crops:
            for t in remaining_text:
                tb = _clip(t['bbox'], size)
                if _intersects(crop['box'], tb) and not _contains(crop['box'], tb):
                    crop['box'] = _clip(_union(crop['box'], [tb[0] - 2, tb[1] - 2, tb[2] + 4, tb[3] + 4]), size)
                    if t['id'] not in crop['seed_ids']:
                        crop['seed_ids'].append(t['id']); crop['texts'].append(t.get('text') or '')
                    changed = True
        merged = []
        for crop in crops:
            target = next((m for m in merged if _intersects(m['box'], crop['box'], gap=2)), None)
            if target is None:
                merged.append(crop)
                continue
            target['box'] = _union(target['box'], crop['box'])
            target['seed_ids'] += [i for i in crop['seed_ids'] if i not in target['seed_ids']]
            target['texts'] += crop['texts']
            target['cuts'] = target['cuts'] or crop['cuts']
            changed = True
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


def _committed_boxes(graph: SceneGraph) -> list[list[float]]:
    boxes = []
    for record in graph.selected.values():
        box = record.get('restore_bbox') or record.get('bbox')
        if box and len(box) == 4:
            boxes.append([float(v) for v in box])
    return boxes


def stage_region_crops(baseline: str | Path, output: str | Path, *, seeds: list[dict] | None = None,
                       allow_movable: bool = True, max_area_fraction=MAX_AREA_FRACTION) -> tuple[Path, list[str], list[dict]]:
    """Stage crops on a built baseline directory; return (slideir path, staged ids, report)."""
    from ..object_unmix import backing_from_evidence
    from ..parallel_rebuild import adapt, array_sha
    baseline, output = Path(baseline), Path(output)
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
    crops, report = plan_region_crops(graph, plate, seeds, max_area_fraction=max_area_fraction)
    if not crops:
        return baseline / 'slideir.json', [], report
    if output.exists():
        raise ValueError('region crop output already exists')
    shutil.copytree(baseline, output)
    committed = _committed_boxes(graph)
    used = {x.id for x in [*graph.text.texts, *graph.asset.objects, *graph.surface.shapes]}
    staged, seq = [], 1
    movable_mask = np.zeros(plate.shape[:2], bool)
    for crop in crops:
        while f'r-{seq:03d}' in used:
            seq += 1
        rid = f'r-{seq:03d}'; seq += 1; used.add(rid)
        x, y, w, h = crop['box']
        pixels = appearance[y:y + h, x:x + w].copy()
        safety, backing_meta, backing = 'MOVE_UNSAFE', {'reason': 'not attempted'}, None
        isolated = not any(_intersects(crop['box'], b) for b in committed)
        if not allow_movable:
            backing_meta = {'reason': 'movable crops disabled'}
        elif not isolated:
            backing_meta = {'reason': 'overlaps committed editable object'}
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
        png = output / f'{rid}.png'
        Image.fromarray(np.dstack([pixels, np.full((h, w), 255, np.uint8)])).save(png)
        intrinsic = ' / '.join(t for t in crop['texts'] if t)[:200]
        graph.asset.objects.append(ClosedObject(
            id=rid, kind='illustration', bbox=[x, y, w, h], layers=['source-plate.png'],
            components=['whole'], parts=[{'name': 'whole', 'bbox': [x, y, w, h], 'layers': ['source-plate.png']}],
            exclusions=[], replaces=[], required_owners=[],
            text_policy='intrinsic_label' if intrinsic else 'exclude_independent_text', intrinsic_text=intrinsic,
            description='opaque source region crop (no semantic claim)', prompt_en='opaque source region crop'))
        # Geometric ownership only: identical pixels at identical position.
        graph.ownership[rid] = 'RESOLVED'
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
            plate[y:y + h, x:x + w] = np.clip(np.rint(backing), 0, 255).astype(np.uint8)
            movable_mask |= region
        graph.transactions.append(record)
        graph.selected[rid] = record
        staged.append(rid)
        report.append({'id': rid, 'state': 'STAGED', 'move_safety': safety, 'bbox': [x, y, w, h],
                       'seed_ids': crop['seed_ids'], **backing_meta})
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
