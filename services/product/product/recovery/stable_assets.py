"""Reuse source-owner closures with bound prior actual Office edit evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..materialize import digest
from ..parallel_contracts import SceneGraph


def verified_candidate(obj,source,protected,records):
    record=next((r for r in records if r['id']==obj.id and r.get('verified_replacement')),None)
    if record is None:return None
    proof=record['verified_replacement']
    for field in ('scene','owner','acceptance'):
        if digest(proof[field]['path'])!=proof[field]['sha256']:
            raise ValueError('stable replacement evidence binding mismatch')
    scene_file=Path(proof['scene']['path'])
    scene=SceneGraph.model_validate_json(scene_file.read_text('utf8'))
    original=np.array(Image.open(scene_file.parent/'source.png').convert('RGB'))
    if digest(scene_file.parent/'source.png')!=scene.source_sha256 or not np.array_equal(original,source):
        raise ValueError('stable replacement source changed')
    definition=next((a for a in scene.asset.objects if a.id==obj.id),None)
    if definition is None or definition.model_dump()!=obj.model_dump():return None
    selected=scene.selected.get(obj.id,{})
    if not selected.get('complete_owner_scope') or selected.get('source_sha256')!=scene.source_sha256:
        raise ValueError('stable source owner closure missing')
    if selected['owner_evidence']['sha256']!=proof['owner']['sha256']:
        raise ValueError('stable owner proof differs from its scene')
    asset=selected['asset_reference']
    if asset['sha256']!=record['sha256'] or digest(record['path'])!=record['sha256'] or selected['bbox']!=record['bbox']:
        raise ValueError('stable asset identity changed')
    acceptance=json.loads(Path(proof['acceptance']['path']).read_text('utf8'))
    if acceptance['source_sha256']!=scene.source_sha256:raise ValueError('stable Office source mismatch')
    office=acceptance['office'];artifact=acceptance['artifacts']
    if office['status']!='PASS_APP_AUTOMATED' or office['pptx_sha256']!=artifact['pptx_sha256']:
        raise ValueError('stable actual Office evidence unavailable')
    if digest(artifact['pptx'])!=artifact['pptx_sha256']:raise ValueError('stable Office package changed')
    page=acceptance.get('page',1)
    edits=[r for r in office['records'] if r['id']==obj.id and r['page']==page]
    if not {'move-scale','delete'}<={r['action'] for r in edits} or any(not r['saved_reopened'] or r['outside_pixels'] for r in edits):
        raise ValueError('stable edit evidence incomplete')
    residual=next((r for r in acceptance['owner_residual'] if r['id']==obj.id),None)
    if (not residual or residual['residual_check_status']!='PASS' or residual['residual_pixels_confirmed']
            or residual['unexplained_pixels'] or residual['reference_sha256']!=proof['owner']['sha256']):
        raise ValueError('stable source residual evidence incomplete')
    with np.load(proof['owner']['path'],allow_pickle=False) as data:
        mask=data['mask'];box=data['restore_bbox'].tolist();patch=data['backing_patch']
    if mask.dtype!=bool or mask.shape!=source.shape[:2] or not mask.any():raise ValueError('invalid stable owner mask')
    if hashlib.sha256(mask.astype('uint8').tobytes()).hexdigest()!=selected['mask_sha256']:
        raise ValueError('stable mask differs from source-owner record')
    x,y,w,h=map(int,box)
    if patch.dtype!=np.uint8 or patch.shape!=(h,w,3) or min(x,y)<0 or x+w>source.shape[1] or y+h>source.shape[0]:
        raise ValueError('invalid stable backing patch')
    outside=mask.copy();outside[y:y+h,x:x+w]=False
    if outside.any():raise ValueError('stable owner scope is incomplete')
    bx,by,bw,bh=map(round,record['bbox']);rgba=np.array(Image.open(record['path']).convert('RGBA'))
    if rgba.shape!=(bh,bw,4):raise ValueError('stable PNG dimensions changed')
    if (mask&protected).any() or (protected[by:by+bh,bx:bx+bw]&(rgba[...,3]>0)).any():
        raise ValueError('stable owner intersects current protected peer')
    return {'id':obj.id,'kind':'asset','bbox':record['bbox'],'restore_bbox':box,'mask':mask,'patch':patch,
            'rgba':rgba,'background_error':selected.get('background_error',0),
            'payload':{**selected['payload'],'method':'stable-reuse','original_file':record['path'],
                'original_sha256':record['sha256'],'stable_id':obj.id,'stable_bbox':record['bbox'],
                'verified_replacement':proof,'current_peer_integrity':'PASS'}}
