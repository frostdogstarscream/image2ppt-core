"""Independent source and ownership rechecks for Agent-produced whole PNGs."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..materialize import digest, write
from ..object_extract import load_page
from ..parallel_candidate_contracts import ReconstructionCandidate, PreflightResult
from ..parallel_rebuild import asset_candidates, source_owner, backing_from_evidence
from ..parallel_regions import regions
from ..parallel_store import key


def _load(path):
    return json.loads(Path(path).read_text('utf8'))


def verify_asset_candidate(scene, original, object_id, source_file, directory, output, *,
                           baseline_directory, layer_directory):
    """Recreate seed, bounded halo, owner closure and erase patch before readiness.

    Existing product preflight and source-owner thresholds are reused. Office
    and independent visual approval are subsequent operations.
    """
    from .asset_retry import protected_for_asset, complete_source_halo
    if baseline_directory is None:raise ValueError('asset tool needs a host-bound baseline')
    baseline=Path(baseline_directory).resolve()
    candidate=ReconstructionCandidate.model_validate(_load(directory/'candidate.json'))
    preflight=PreflightResult.model_validate(_load(directory/'preflight.json'))
    payload=_load(directory/'payload.json')
    if (candidate.status!='PRECHECK_PASS' or candidate.hard_gate!='PASS' or preflight.status!='PASS'
            or candidate.representation!='TRANSPARENT_PNG' or candidate.object_ids!=[object_id]
            or candidate.candidate_id!=preflight.candidate_id):
        raise ValueError('complete asset candidate did not pass its preflight')
    if candidate.source_sha256!=scene.source_sha256 or candidate.scene_sha256!=key(scene.model_dump()):
        raise ValueError('asset candidate source/scene binding mismatch')
    if set(candidate.references)!={'payload.json','pixels.npz'}:
        raise ValueError('asset candidate references are incomplete')
    for name,sha in candidate.references.items():
        if digest(directory/name)!=sha:raise ValueError('asset candidate evidence changed')
    source_obj=next(obj for obj in original.asset.objects if obj.id==object_id)
    obj=next(obj for obj in scene.asset.objects if obj.id==object_id)
    if (payload.get('id')!=object_id or payload.get('kind')!='asset'
            or payload.get('source_asset_definition')!=source_obj.model_dump()
            or payload.get('proposed_asset_definition')!=obj.model_dump()
            or source_obj.model_dump()!=obj.model_dump()):
        raise ValueError('asset tool changed semantic object definition')
    if scene.ownership[object_id]!='RESOLVED' or regions(scene)[1][object_id]['member_ids']!=[object_id]:
        raise ValueError('asset target has unresolved or indivisible ownership')
    if digest(baseline/'source.png')!=scene.source_sha256:
        raise ValueError('asset baseline source binding mismatch')
    if payload.get('baseline_scene_sha256')!=digest(baseline/'scene.json'):
        raise ValueError('asset baseline scene binding mismatch')
    source=np.array(Image.open(source_file).convert('RGB'))
    saved=np.array(Image.open(directory/'source.png').convert('RGB'))
    if not np.array_equal(saved,source):raise ValueError('asset candidate source pixels changed')
    with np.load(directory/'pixels.npz',allow_pickle=False) as data:
        if (data['bbox'].shape!=(4,) or data['rgba_bbox'].shape!=(4,)
                or not np.issubdtype(data['bbox'].dtype,np.integer)
                or not np.issubdtype(data['rgba_bbox'].dtype,np.integer)):
            raise ValueError('asset pixel boxes must contain four integers')
        mask=data['mask'];patch=data['patch'];box=data['bbox'].astype(int).tolist()
        rgba=data['rgba'];rgba_box=data['rgba_bbox'].astype(int).tolist()
        seed=data['seed_rgba'];saved_protected=data['protected']
    if (mask.dtype!=bool or mask.shape!=source.shape[:2] or not mask.any()
            or rgba.dtype!=np.uint8 or patch.dtype!=np.uint8
            or seed.dtype!=np.uint8 or seed.shape!=(*source.shape[:2],4)):
        raise ValueError('invalid asset pixel evidence')
    protected=protected_for_asset(scene,baseline,object_id)
    if not np.array_equal(protected,saved_protected) or (mask&protected).any():
        raise ValueError('asset candidate altered protected peers')
    surface=scene.surface.model_copy(deep=True)
    surface.shapes=[shape for shape in surface.shapes if scene.ownership[shape.id]=='RESOLVED']
    if layer_directory is None or payload.get('layer_manifest_sha256')!=digest(Path(layer_directory)/'manifest.json'):
        raise ValueError('asset LayerD input is not bound to the host configuration')
    layers=load_page(layer_directory)
    if not np.array_equal(np.array(Image.open(layers.source_path).convert('RGB')),source):
        raise ValueError('asset LayerD evidence belongs to different source pixels')
    seeds=asset_candidates(obj,source,layers,protected,(),'auto',surface=surface)
    if not any(np.array_equal(seed,proposal) for proposal,_ in seeds):
        raise ValueError('asset segmentation seed lacks original LayerD/source evidence')
    completed,halo=complete_source_halo(obj,seed,source,protected,surface)
    x,y,w,h=rgba_box
    if (min(x,y)<0 or min(w,h)<=0 or x+w>source.shape[1] or y+h>source.shape[0]
            or rgba.shape!=(h,w,4)):
        raise ValueError('asset RGBA scope is invalid')
    full=np.zeros_like(completed);full[y:y+h,x:x+w]=rgba
    if not np.array_equal(full,completed):raise ValueError('asset differs from bounded source segmentation')
    asset=payload.get('asset_reference',{})
    filename=asset.get('path','')
    if Path(filename).name!=filename or not filename:
        raise ValueError('asset PNG path is invalid')
    if digest(directory/filename)!=asset.get('sha256'):
        raise ValueError('asset PNG hash mismatch')
    if not np.array_equal(np.array(Image.open(directory/filename).convert('RGBA')),rgba):
        raise ValueError('asset PNG pixels were resampled or changed')
    owner_box,backing,owner,owner_proof=source_owner(obj,full,source,protected,surface)
    if owner_box!=box or not np.array_equal(owner,mask):
        raise ValueError('asset owner closure differs from independent source components')
    rx,ry,rw,rh=box;local=mask[ry:ry+rh,rx:rx+rw]
    expected=source[ry:ry+rh,rx:rx+rw].copy();expected[local]=backing[local]
    if not np.array_equal(expected,patch):raise ValueError('asset patch changed pixels beyond the proven owner')
    backing_rgba,error,_=backing_from_evidence(source,rgba_box,protected,surface)
    alpha=rgba[...,3:4]/255
    reconstructed=rgba[...,:3]*alpha+backing_rgba*(1-alpha)
    visible=rgba[...,3]>0
    if not visible.any():raise ValueError('asset is fully transparent')
    p95=float(np.percentile(np.abs(reconstructed-source[y:y+h,x:x+w]).max(2)[visible],95))
    if not np.isfinite(error) or error>6 or p95>24:
        raise ValueError('asset failed existing independent background/recomposition gates')
    baseline_scene=_load(baseline/'scene.json')
    for record in baseline_scene['selected'].values():
        proof=record['owner_evidence'];file=(baseline/proof['file']).resolve()
        if not file.is_relative_to(baseline) or digest(file)!=proof['sha256']:
            raise ValueError('baseline owner proof changed')
        with np.load(file,allow_pickle=False) as data:
            if (mask&data['mask'].astype(bool)).any():raise ValueError('asset overlaps a verified old owner')
    source_crop=source[y:y+h,x:x+w];written=source.copy()
    written[ry:ry+rh,rx:rx+rw][local]=patch[local]
    Image.fromarray(source_crop).save(output/'source-region.png')
    Image.fromarray(written[y:y+h,x:x+w]).save(output/'deleted-region.png')
    checker=np.indices((h,w)).sum(0)//8%2
    checker=np.repeat(np.where(checker[...,None],215,245).astype('uint8'),3,2)
    preview=Image.fromarray(checker).convert('RGBA');preview.alpha_composite(Image.fromarray(rgba))
    preview.convert('RGB').save(output/'asset-preview.png')
    write(output/'candidate-scene.json',scene.model_dump())
    checks={name:'PASS' for name in preflight.checks}
    write(output/'asset-independent-checks.json',{'checks':checks,'owner':owner_proof,
        'source_recomposition_p95':p95,'source_halo':halo,'published':False})
    return {'status':'CANDIDATE_READY','representation':'TRANSPARENT_PNG','candidate_directory':str(directory),
        'candidate_scene':str((output/'candidate-scene.json').resolve()),'source_sha256':scene.source_sha256,
        'scene_sha256':key(scene.model_dump()),'checks':checks,
        'preview_files':[{'path':str(output/name),'role':role} for name,role in [
            ('source-region.png','source_region'),('asset-preview.png','transparent_asset_preview'),
            ('deleted-region.png','backing_after_asset_removal')]],
        'office':'NOT_RUN','independent_visual_review':'NOT_RUN','published':False}
