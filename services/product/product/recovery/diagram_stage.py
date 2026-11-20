"""Stage a connected illustration from its measured silhouette on a verified background."""
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

from ..materialize import digest, write
from ..object_contracts import ClosedObject
from ..object_extract import rectangle
from ..object_unmix import backing_from_evidence
from ..parallel_contracts import SceneGraph
from ..parallel_fusion import fuse_incremental_scene
from ..parallel_rebuild import adapt, array_sha
from ..parallel_store import key
from .region_crop import _extract_rgba, _render_lower_composite, MOVABLE_RING_ERROR


def stage_diagram(baseline, output, route):
    graph = SceneGraph.model_validate_json((baseline/'scene.json').read_text('utf8'))
    src = baseline/'source.png'
    plate_file = baseline/graph.raster_backing['path']
    if digest(src)!=graph.source_sha256 or digest(plate_file)!=graph.raster_backing['sha256']:
        raise ValueError('changed diagram source')
    box = list(map(round, route['bbox']))
    x,y,w,h = box
    source = np.asarray(Image.open(src).convert('RGB'))
    plate = np.array(Image.open(plate_file).convert('RGBA'))
    size = graph.image_size
    guard = np.zeros(source.shape[:2],bool)
    for t in graph.text.texts:
        guard |= rectangle(t.bbox,size)
    line_file = baseline/'runtime-ocr-protection.json'
    if line_file.exists():
        for line in json.loads(line_file.read_text('utf8'))['lines']:
            guard |= rectangle(line['bbox'],size)
    for record in graph.selected.values():
        if record['kind']=='surface' or record.get('container_layer'):
            continue
        ref = record.get('owner_evidence')
        if ref:
            if digest(baseline/ref['file'])!=ref['sha256']:
                raise ValueError('changed neighbor ownership')
            with np.load(baseline/ref['file'],allow_pickle=False) as proof:
                guard |= proof['mask'].astype(bool)
        else:
            guard |= rectangle(record['bbox'],size)
    lower, covered = _render_lower_composite(plate[...,:3],graph,baseline)
    backing, err, evidence = backing_from_evidence(lower,box,guard,graph.surface)
    if not np.isfinite(err) or err>MOVABLE_RING_ERROR:
        raise ValueError('diagram background not verified')
    tile = source[y:y+h,x:x+w]
    blocked = guard[y:y+h,x:x+w]
    backing = np.asarray(backing).clip(0,255).astype('uint8')
    signal = np.abs(tile.astype(float)-backing).max(2)
    # Reconnect only visible foreground, allowing a short antialias gap.
    anchors = route['anchor_boxes']
    radius = max(1,round(min(min(b[2:]) for b in anchors)*.06))
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    ink = (signal>max(9,err*3)) & ~blocked
    linked = cv2.morphologyEx(ink.astype('uint8'),cv2.MORPH_CLOSE,brush)
    linked[blocked] = 0
    _, labels = cv2.connectedComponents(linked)
    common = None
    for ax,ay,aw,ah in anchors:
        sample=labels[max(0,round(ay)-y):min(h,round(ay+ah)-y),max(0,round(ax)-x):min(w,round(ax+aw)-x)]
        ids,counts=np.unique(sample,return_counts=True)
        hits=set(ids[(ids!=0)&(counts>=.10*aw*ah)])
        common = hits if common is None else common & hits
    if len(common or ())!=1:
        raise ValueError('connector does not link all independent icon anchors')
    component=(labels==next(iter(common))).astype('uint8')
    contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    silhouette=np.zeros_like(component)
    cv2.drawContours(silhouette,contours,-1,1,cv2.FILLED)
    if (silhouette.astype(bool)&blocked).any():
        raise ValueError('diagram silhouette includes protected content')
    mask=silhouette.astype(bool)
    # No unrelated marks are admitted by a shared bounding rectangle.
    rgba=_extract_rgba(tile,backing,mask)
    rgba[~mask,3]=0
    if np.mean(rgba[...,3]==0)<.10:
        raise ValueError('diagram lacks a transparent perimeter')
    oid='a-'+key([graph.source_sha256,route['id']])[:16]
    obj=ClosedObject(id=oid,kind='illustration',bbox=box,layers=['source-plate.png'],
        components=['whole'],parts=[{'name':'whole','bbox':box,'layers':['source-plate.png']}],
        text_policy='exclude_independent_text',description='Connected illustration',prompt_en='Connected illustration')
    graph=fuse_incremental_scene(baseline,base_graph=graph,added_assets=[obj])
    if graph.ownership.get(oid)!='RESOLVED' or oid in graph.selected:
        raise ValueError('diagram ownership not resolved')
    shutil.copytree(baseline,output)
    png=output/(oid+'.png')
    Image.fromarray(rgba).save(png)
    full=np.zeros_like(guard)
    full[y:y+h,x:x+w]=mask
    proof=output/(oid+'-owner.npz')
    np.savez_compressed(proof,mask=full,restore_bbox=box,backing_patch=backing)
    writable=mask & ~covered[y:y+h,x:x+w]
    plate[y:y+h,x:x+w,:3][writable]=backing[writable]
    Image.fromarray(plate).save(output/plate_file.name)
    graph.raster_backing={**graph.raster_backing,'sha256':digest(output/plate_file.name)}
    ownership_file = output/'ownership.npz'
    if ownership_file.exists():
        with np.load(ownership_file,allow_pickle=False) as data:
            maps={k:data[k].copy() for k in data.files}
        maps['ownership'][full]=0
        np.savez_compressed(ownership_file,**maps)
        meta_path=output/'ownership.json'
        meta=json.loads(meta_path.read_text('utf8'))
        meta['map_sha256']=digest(ownership_file)
        write(meta_path,meta)
    record={'id':oid,'kind':'asset','bbox':box,'restore_bbox':box,'mark_id':route['id'],
        'seed_ids':[route['id'],*route['diagram_members']],'state':'STAGED','move_safety':'MOVABLE',
        'quality':'REVIEW_REQUIRED','original_owner':'source-plate','source_sha256':graph.source_sha256,
        'complete_owner_scope':True,'mask_sha256':array_sha(full.astype('uint8')),
        'owner_evidence':{'file':proof.name,'sha256':digest(proof),'method':'connected-illustration'},
        'asset_reference':{'path':png.name,'sha256':digest(png),'source_sha256':graph.source_sha256,'bbox':box,'crop_offset':[x,y]},
        'payload':{'method':'connected-illustration','background_evidence':evidence,'ring_error':float(err)}}
    graph.selected[oid]=record
    graph.transactions.append(record)
    write(output/'slideir.json',adapt(graph,output))
    write(output/'scene.json',graph.model_dump())
    return output/'slideir.json',oid
