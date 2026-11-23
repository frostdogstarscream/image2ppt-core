"""Stage the symbol on a measured disc before its parent surfaces are reconstructed."""
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
from .circular_badge import measure_symbol, select_badge_ink


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
    if layout and layout.get('source_sha256', graph.source_sha256) == graph.source_sha256:
        for region in layout.get('regions', ()):
            if region['label'] in ('image', 'figure', 'icon', 'seal') and region.get('score', 0) >= .5:
                locators['layout:' + region['id'].removeprefix('layout:')] = region['bbox']
    protected = [t.bbox for t in graph.text.texts]
    ledger = baseline / 'runtime-ocr-protection.json'
    if ledger.is_file():
        ocr = json.loads(ledger.read_text('utf8'))
        if ocr.get('source_sha256', graph.source_sha256) != graph.source_sha256:
            raise ValueError('badge OCR protection source changed')
        protected.extend(line['bbox'] for line in ocr['lines'])
    occupied = [r['bbox'] for r in graph.selected.values()]
    protected.extend(obj.bbox for obj in graph.asset.objects
                     if obj.kind == 'logo' or obj.text_policy == 'intrinsic_label')
    groups = {}
    for alias, box in locators.items():
        if box[2] * box[3] > .045 * source.shape[0] * source.shape[1]:
            continue
        result = measure_symbol(source, box)
        if result:
            disc = tuple(result['disc'])
            group = groups.setdefault(disc, (result, set()))
            group[1].add(alias)
    layer = _layer_image(baseline, graph) if groups else None
    staged = []
    claimed = np.zeros(source.shape[:2], bool)
    for disc, (candidate, aliases) in sorted(groups.items()):
        if panel_regions is not None and not any(intersection(disc, box) >= disc[2]*disc[3] for box in panel_regions):
            continue
        if not aliases.intersection(icons):
            continue
        oid = 'a-' + key([graph.source_sha256, 'badge-ink', list(disc)])[:16]
        if (oid in graph.selected or oid in skip or aliases.intersection(skip)
                or (wanted is not None and not (aliases | {oid}).intersection(wanted))):
            continue
        box = candidate['bbox']
        if (any(intersection(disc, other) for other in protected + occupied)
                or any(intersection(box, other) for name, other in icons.items() if name not in aliases)):
            continue
        x, y, w, h = box
        tile = np.s_[y:y+h, x:x+w]
        mask = candidate['mask']
        if not np.array_equal(source[tile][mask], plate[tile][..., :3][mask]):
            continue
        excluded = {name for name in ('LayerD','color-unmix') if oid+'-'+name in skip}
        rgba, decision = select_badge_ink(source[tile], candidate['face'], candidate['rgba'],
                                         layer[tile] if layer is not None else None, excluded=excluded, strict_holes=True)
        if rgba is None:
            continue
        obj = ClosedObject(id=oid, kind='icon', bbox=box, layers=['source-plate.png'],
                           components=['whole'], parts=[{'name':'whole','bbox':box,'layers':['source-plate.png']}],
                           text_policy='exclude_independent_text', description='Symbol on a circular base',
                           prompt_en='Symbol on a circular base')
        fused = fuse_incremental_scene(baseline, base_graph=graph, added_assets=[obj])
        if fused.ownership.get(oid) != 'RESOLVED':
            continue
        graph = fused
        if not staged:
            shutil.copytree(baseline, output)
        png = output / (oid + '.png')
        Image.fromarray(rgba).save(png)
        face = np.array(list(bytes.fromhex(candidate['face'])), dtype='uint8')
        backing = np.broadcast_to(face, (h,w,3))
        full_mask = np.zeros_like(claimed)
        full_mask[tile] = mask
        claimed |= full_mask
        proof = output / (oid + '-owner.npz')
        np.savez_compressed(proof, mask=full_mask, restore_bbox=box, backing_patch=backing)
        plate[tile][..., :3][mask] = face
        record = {'id':oid, 'kind':'asset', 'bbox':box, 'restore_bbox':box, 'state':'STAGED',
                  'move_safety':'MOVABLE', 'complete_owner_scope':True, 'quality':'REVIEW_REQUIRED',
                  'source_sha256':graph.source_sha256, 'original_owner':'source-plate',
                  'badge_evidence_ids':sorted(aliases), 'disc_bbox':list(disc),
                  'variant_key':oid+'-'+decision['selected'], 'mask_sha256':array_sha(full_mask.astype('uint8')),
                  'payload':{'method':'measured-badge-ink', 'selection':decision},
                  'owner_evidence':{'file':proof.name,'sha256':digest(proof),'method':'measured-badge-ink'},
                  'asset_reference':{'path':png.name,'sha256':digest(png),'source_sha256':graph.source_sha256,
                                     'bbox':box,'crop_offset':[x,y]}}
        graph.selected[oid] = record
        graph.transactions.append(record)
        occupied.append(box)
        staged.append(oid)
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
    write(output / 'slideir.json', adapt(graph, output))
    return output, staged
