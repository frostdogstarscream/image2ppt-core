"""Separate source-measured symbols from circular or flat native panel faces."""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts import intersection
from ..materialize import digest, write
from ..object_contracts import ClosedObject
from ..parallel_contracts import SceneGraph
from ..parallel_fusion import fuse_incremental_scene
from ..parallel_rebuild import adapt, array_sha
from ..parallel_store import key
from .circular_badge import measure_flat_symbol, measure_symbol, select_badge_ink


def represented_badge_evidence(graph):
    return {alias for record in graph.selected.values() for alias in record.get('badge_evidence_ids', ())}


def _layer_image(baseline, graph):
    ref = baseline / 'badge-layer.json'
    if ref.is_file():
        binding = json.loads(ref.read_text('utf8'))
        path = Path(binding['path'])
        if binding['source_sha256'] == graph.source_sha256 and path.is_file() and digest(path) == binding['sha256']:
            with Image.open(path) as image:
                if image.size == tuple(graph.image_size):
                    return np.asarray(image.convert('RGBA'))
    return None


def _parent_backing(baseline, graph, parents, lower, box, face, mask, plate_tile, source_tile):
    """Keep the physical plate below a committed native face clean when adding a child."""
    x, y, w, h = box
    if any(intersection(box, r['bbox']) < w*h or r.get('source_sha256') != graph.source_sha256
           for r in parents):
        return None
    composite, covered = lower
    local = np.s_[y:y+h, x:x+w]
    if not covered[local].all() or abs(composite[local][..., :3].astype(float)-face).max() > 10:
        return None
    parents = sorted(parents, key=lambda r: (r['payload']['z'], -r['bbox'][2]*r['bbox'][3]))
    for parent in parents:
        proof = parent.get('owner_evidence')
        if not proof or parent['payload'].get('geometry') not in ('rect','roundRect','ellipse','pentagon','chevron'):
            return None
        path = baseline/proof['file']
        if not path.resolve().is_relative_to(baseline.resolve()) or digest(path) != proof['sha256']:
            raise ValueError('native symbol parent proof changed')
        with np.load(path, allow_pickle=False) as data:
            rx, ry, rw, rh = map(int, data['restore_bbox'])
            if not (rx <= x and ry <= y and x+w <= rx+rw and y+h <= ry+rh):
                return None
            owner = data['mask'][local]
            # A real parent face has measured paper at the child's perimeter.
            # Do not claim strokes already owned by an immutable surface.
            if np.any(owner & mask):
                return None
            if parent is parents[0]:
                physical = data['backing_patch'][y-ry:y-ry+h, x-rx:x-rx+w, :3].copy()
            if parent is parents[-1]:
                # Constrain the footprint to the fitted surface, including the
                # angled side of a ribbon (its bounding rectangle is broader).
                around = ~mask
                if not around.any() or np.mean(owner[around]) < .8:
                    return None
    old = plate_tile[mask]
    if not np.all(np.all(old == source_tile[mask],axis=1) | np.all(old == physical[mask],axis=1)):
        return None
    return physical, parents[-1]['id']


def _retained_symbol(directory, record, box, mask, face, source, parents):
    """A generated card-art layer can release an entire measured symbol, never photo pixels."""
    payload = record.get('payload', {})
    if (payload.get('method') != 'retained-card-foreground'
            or payload.get('background_parent_id') not in {r['id'] for r in parents}):
        return None
    x, y, w, h = box
    ax, ay, aw, ah = map(round, record['bbox'])
    if not (ax <= x and ay <= y and x+w <= ax+aw and y+h <= ay+ah):
        return None
    pic, proof = record['asset_reference'], record['owner_evidence']
    for ref, name in ((pic, 'path'), (proof, 'file')):
        path = (directory/ref[name]).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != ref['sha256']:
            raise ValueError('retained panel artwork binding changed')
    with Image.open(directory/pic['path']) as image:
        rgba = np.array(image.convert('RGBA'))
    with np.load(directory/proof['file'], allow_pickle=False) as data:
        saved = {k:data[k] for k in data.files}
    tile = np.s_[y-ay:y-ay+h, x-ax:x-ax+w]
    owner = saved['mask'][y:y+h,x:x+w]
    if rgba.shape[:2] != (ah, aw) or not np.array_equal(owner, mask):
        return None
    alpha = rgba[tile][...,3:] / 255.
    original = rgba[tile][...,:3]*alpha + face*(1-alpha)
    if np.any(alpha[...,0][~mask] > .06) or abs(original[mask]-source[mask]).max() > 12:
        return None
    rgba[tile][mask] = 0
    owner[mask] = False
    return rgba, saved


def stage_badge_symbols(baseline, output, layout, *, wanted=None, skip=(), panel_regions=None):
    graph = SceneGraph.model_validate_json((baseline / 'scene.json').read_text('utf8'))
    source_path = baseline / 'source.png'
    plate_path = baseline / graph.raster_backing['path']
    if digest(source_path) != graph.source_sha256 or digest(plate_path) != graph.raster_backing['sha256']:
        raise ValueError('badge source or plate binding changed')
    with Image.open(source_path) as image:
        source = np.asarray(image.convert('RGB'))
    with Image.open(plate_path) as image:
        plate = np.array(image.convert('RGBA'))
    icons = {obj.id: obj.bbox for obj in graph.asset.objects
             if obj.kind == 'icon' and obj.text_policy == 'exclude_independent_text'
             and graph.ownership.get(obj.id) == 'RESOLVED'}
    locators = dict(icons)
    explicit_icons = set(icons)
    if layout and layout.get('source_sha256', graph.source_sha256) == graph.source_sha256:
        for region in layout.get('regions', ()):
            if region['label'] in ('image', 'figure', 'icon', 'seal') and region.get('score', 0) >= .5:
                alias = 'layout:' + region['id'].removeprefix('layout:')
                locators[alias] = region['bbox']
                if region['label'] == 'icon':
                    explicit_icons.add(alias)
    protected = [t.bbox for t in graph.text.texts]
    ledger = baseline / 'runtime-ocr-protection.json'
    if ledger.is_file():
        ocr = json.loads(ledger.read_text('utf8'))
        if ocr.get('source_sha256', graph.source_sha256) != graph.source_sha256:
            raise ValueError('badge OCR protection source changed')
        protected.extend(line['bbox'] for line in ocr['lines'])
    protected.extend(a.bbox for a in graph.asset.objects if graph.ownership.get(a.id) != 'RESOLVED'
                     and a.kind == 'icon')
    protected.extend(obj.bbox for obj in graph.asset.objects
                     if obj.kind == 'logo' or obj.text_policy == 'intrinsic_label')
    groups = {}
    for alias, box in locators.items():
        if box[2] * box[3] > .045 * source.shape[0] * source.shape[1]:
            continue
        result = measure_symbol(source, box)
        if result is None and alias in explicit_icons:
            result = measure_flat_symbol(source, box)
        if result:
            family = 'badge-ink' if 'disc' in result else 'flat-icon'
            bounds = tuple(result.get('disc', result['bbox']))
            group = groups.setdefault((family, bounds), (result, set()))
            group[1].add(alias)
    layer = _layer_image(baseline, graph) if groups else None
    staged = []
    decisions = []
    claimed = np.zeros(source.shape[:2], bool)
    lower = None
    for (family, bounds), (candidate, aliases) in sorted(groups.items()):
        if panel_regions is not None and not any(intersection(bounds, box) >= bounds[2]*bounds[3] for box in panel_regions):
            continue
        if not aliases.intersection(explicit_icons):
            continue
        oid = 'a-' + key([graph.source_sha256, family, list(bounds)])[:16]
        if (oid in graph.selected or oid in skip or aliases.intersection(skip)
                or (wanted is not None and not (aliases | {oid}).intersection(wanted))):
            continue
        box = candidate['bbox']
        if (any(intersection(bounds, other) for other in protected)
                or any(intersection(box, other) for name, other in icons.items() if name not in aliases)):
            continue
        x, y, w, h = box
        tile = np.s_[y:y+h, x:x+w]
        mask = candidate['mask']
        peers = [r for r in graph.selected.values() if intersection(box, r['bbox'])]
        parents = [r for r in peers if r['kind'] == 'surface']
        artwork = [r for r in peers if r['kind'] != 'surface']
        if len(artwork) > 1 or np.any(claimed[tile] & mask):
            continue
        face = np.array(list(bytes.fromhex(candidate['face'])), dtype='uint8')
        backing = np.broadcast_to(face, (h,w,3))
        physical = backing
        parent_id = None
        retained = None
        current_dir = output if staged else baseline
        if artwork:
            retained = _retained_symbol(current_dir, artwork[0], box, mask, face, source[tile], parents)
            if retained is None:
                continue
        reuse_artwork = retained is not None and not np.any(retained[1]['mask'])
        if reuse_artwork:
            if artwork[0]['id'] in skip or list(map(round, artwork[0]['bbox'])) != list(box):
                continue
            oid = artwork[0]['id']
        if parents:
            from .region_crop import _render_lower_composite
            if retained is not None:
                lower_graph = graph.model_copy(update={'selected':{k:r for k,r in graph.selected.items()
                                                                  if k != artwork[0]['id']}})
                visible_face = _render_lower_composite(plate, lower_graph, current_dir)
            else:
                if lower is None:
                    lower = _render_lower_composite(plate, graph, current_dir)
                visible_face = lower
            result = _parent_backing(current_dir, graph, parents, visible_face, box, face, mask, plate[tile][..., :3], source[tile])
            if result is None:
                continue
            physical, parent_id = result
        elif not np.array_equal(source[tile][mask], plate[tile][..., :3][mask]):
            continue
        excluded = {name for name in ('LayerD','color-unmix') if oid+'-'+name in skip}
        rgba, decision = select_badge_ink(source[tile], candidate['face'], candidate['rgba'],
                                         layer[tile] if layer is not None else None, excluded=excluded, strict_holes=True)
        decisions.append({'object_id':oid, 'disc':candidate.get('disc'), 'bbox':box, 'evidence_ids':sorted(aliases), **decision})
        if rgba is None:
            continue
        obj = ClosedObject(id=oid, kind='icon', bbox=box, layers=['source-plate.png'],
                           components=['whole'], parts=[{'name':'whole','bbox':box,'layers':['source-plate.png']}],
                           text_policy='exclude_independent_text', description='Source-measured symbol',
                           prompt_en='Source-measured symbol')
        fused = fuse_incremental_scene(baseline, base_graph=graph, added_assets=[obj])
        if fused.ownership.get(oid) != 'RESOLVED':
            continue
        graph = fused
        if not staged:
            shutil.copytree(baseline, output)
        if retained is not None and not reuse_artwork:
            leftover, saved = retained
            original = graph.selected[artwork[0]['id']]
            pic_path = output/original['asset_reference']['path']
            proof_path = output/original['owner_evidence']['file']
            Image.fromarray(leftover).save(pic_path)
            np.savez_compressed(proof_path, **saved)
            original['asset_reference']['sha256'] = digest(pic_path)
            original['owner_evidence']['sha256'] = digest(proof_path)
            original['mask_sha256'] = array_sha(saved['mask'].astype('uint8'))
            graph.transactions = [original if t['id'] == original['id'] else t for t in graph.transactions]
            if original['id'] not in staged:
                staged.append(original['id'])
        if retained is not None:
            lower = None
        png = output / (oid + '.png')
        Image.fromarray(rgba).save(png)
        full_mask = np.zeros_like(claimed)
        full_mask[tile] = mask
        claimed |= full_mask
        proof = output / (oid + '-owner.npz')
        np.savez_compressed(proof, mask=full_mask, restore_bbox=box, backing_patch=backing)
        plate[tile][..., :3][mask] = physical[mask]
        method = 'measured-badge-ink' if family == 'badge-ink' else 'measured-flat-icon'
        record = {'id':oid, 'kind':'asset', 'bbox':box, 'restore_bbox':box, 'state':'STAGED',
                  'move_safety':'MOVABLE', 'complete_owner_scope':True, 'quality':'REVIEW_REQUIRED',
                  'source_sha256':graph.source_sha256, 'original_owner':'source-plate',
                  'badge_evidence_ids':sorted(aliases), **({'disc_bbox':list(bounds)} if family == 'badge-ink' else {}),
                  'variant_key':oid+'-'+decision['selected'], 'mask_sha256':array_sha(full_mask.astype('uint8')),
                  'payload':{'method':method, 'selection':decision, **({'background_parent_id':parent_id} if parent_id else {})},
                  'owner_evidence':{'file':proof.name,'sha256':digest(proof),'method':method},
                  'asset_reference':{'path':png.name,'sha256':digest(png),'source_sha256':graph.source_sha256,
                                     'bbox':box,'crop_offset':[x,y]}}
        graph.selected[oid] = record
        if reuse_artwork:
            graph.transactions = [record if t['id'] == oid else t for t in graph.transactions]
        else:
            graph.transactions.append(record)
        if parent_id:
            parent = graph.selected[parent_id]
            children = set(parent.get('contained_children', ())) | {oid}
            parent['contained_children'] = sorted(children)
            for transaction in graph.transactions:
                if transaction['id'] == parent_id:
                    transaction['contained_children'] = sorted(children)
        staged.append(oid)
    if decisions:
        write(output.parent/(output.name+'-trials.json'), {'source_sha256':graph.source_sha256, 'objects':decisions})
    if not staged:
        return baseline, []
    Image.fromarray(plate).save(output / plate_path.name)
    graph.raster_backing = {**graph.raster_backing, 'sha256':digest(output / plate_path.name)}
    maps_file = output / 'ownership.npz'
    if maps_file.is_file():
        with np.load(maps_file, allow_pickle=False) as data:
            maps = {name:data[name].copy() for name in data.files}
        maps['ownership'][claimed] = 0
        np.savez_compressed(maps_file, **maps)
        metadata = output / 'ownership.json'
        write(metadata, {**json.loads(metadata.read_text('utf8')), 'map_sha256':digest(maps_file)})
    write(output / 'scene.json', graph.model_dump())
    write(output / 'transactions.json', graph.transactions)
    write(output / 'slideir.json', adapt(graph, output))
    return output, staged
