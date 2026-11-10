"""Automatic use of the semi-automatic source/background RGBA estimator.

The ORIGINAL source and a separately estimated clean backing are the evidence.
This is an alternative matte, never a correction of LayerD/SAM or an Alpha truth.
Only the existing conversation-independent helpers are reused; no OCR/model call.
"""
from pathlib import Path
from functools import lru_cache
import cv2
import numpy as np
from PIL import Image
from .materialize import digest
from .object_extract import rectangle
from .object_registry import fingerprint
from .object_ownership import background


@lru_cache(maxsize=1)
def binding():
    shared=Path(__file__).resolve().parents[3]/'ppt-rebuild/src/reconstruction.py'
    return {'method':'source-background-unmix/1','implementation_sha256':digest(shared),'adapter_sha256':digest(__file__)}


def candidate(obj,source,ir,root,blocked,*,other_objects=()):
    """Return full-canvas RGBA plus pending background restoration, or reject.

    MoMA supplies semantic scope. Lateral samples estimate backing, native
    geometry cross-checks it, original pixels supply colour and edge evidence.
    Every independent text/neighbor conflict is rejected, never punched out.
    """
    from reconstruction import background_model,extract_rgba
    size=(source.shape[1],source.shape[0]);root=Path(root).resolve()
    nodes={n['id']:n for n in ir['nodes']};assets={a['id']:a for a in ir['assets']}
    if obj.kind not in ('icon','logo','illustration'):raise ValueError('unmix_kind_unsupported')
    if any(i not in nodes or nodes[i]['type']!='image' for i in obj.replaces):raise ValueError('unmix_owner_unsupported')
    if any(set(g.get('members',[]))&set(obj.replaces) for g in ir.get('groups',[])):
        raise ValueError('group_transaction_requires_group_preservation')
    boxes=[obj.bbox,*[part.bbox for part in obj.parts],*[nodes[i]['bbox'] for i in obj.replaces]]
    left=max(0,int(np.floor(min(b[0] for b in boxes)))-4)
    top=max(0,int(np.floor(min(b[1] for b in boxes)))-4)
    right=min(size[0],int(np.ceil(max(b[0]+b[2] for b in boxes)))+4)
    bottom=min(size[1],int(np.ceil(max(b[1]+b[3] for b in boxes)))+4)
    box=[left,top,right-left,bottom-top];x,y,w,h=box
    excluded=blocked.copy()
    for other in other_objects:
        if other.id!=obj.id:
            for part in other.parts:excluded|=rectangle(part.bbox,size)
    # Negative boxes protect sampling/ownership; they do not cut visible strokes.
    for part in obj.exclusions:excluded|=rectangle(part.bbox,size)
    for n in ir['nodes']:
        if n['type']=='image' and n['id'] not in obj.replaces:excluded|=rectangle(n['bbox'],size)
    estimate,ring_error=background_model(source,box,excluded)
    if not np.isfinite(ring_error) or ring_error>6:raise ValueError('unmix_background_not_clean')
    z=min([nodes[i]['z'] for i in obj.replaces] or [10**9])
    native,known=background(ir,size,z)
    # Native fields are a cross-check, not source ground truth: an earlier VLM
    # may have picked a slightly wrong card colour. Preserve the disagreement
    # for independent source/delete review instead of silently changing a card.
    border=np.zeros((h,w),bool);border[[0,-1],:]=True;border[:,[0,-1]]=True
    check=border & known[y:y+h,x:x+w] & ~excluded[y:y+h,x:x+w]
    native_check={'known_samples':int(check.sum()),'median':None,'p95':None}
    if check.any():
        error=np.abs(estimate.astype('int16')-native[y:y+h,x:x+w].astype('int16')).max(2)[check]
        native_check.update(median=float(np.median(error)),p95=float(np.percentile(error,95)))
    crop=source[y:y+h,x:x+w]
    delta=np.abs(crop.astype('int16')-estimate.astype('int16')).max(2)
    core=delta>14
    if core.sum()<6:raise ValueError('unmix_no_separated_foreground')
    # Same bounded edge search as the semi-automatic route. Only source
    # differences enter the matte; no opacity/colour is manufactured by dilation.
    edge=cv2.dilate(core.astype('uint8'),np.ones((3,3),np.uint8))>0
    mask=edge & (delta>max(1,ring_error))
    # A too-small VLM scope must fail rather than silently sever a component.
    sides=[(left>0,mask[:,0]),(right<size[0],mask[:,-1]),(top>0,mask[0]),(bottom<size[1],mask[-1])]
    if any(active and pixels.any() for active,pixels in sides):raise ValueError('unmix_scope_cuts_foreground')
    if (mask & excluded[y:y+h,x:x+w]).any():raise ValueError('unmix_protected_content')
    for part in obj.parts:
        roi=rectangle(part.bbox,size)[y:y+h,x:x+w]
        if not (roi & mask).any():raise ValueError('missing_required_parts')
    rgba=extract_rgba(crop,estimate,mask)
    alpha=rgba[...,3:4].astype(float)/255
    rendered=rgba[...,:3]*alpha+estimate*(1-alpha)
    error=np.abs(rendered-crop)[mask]
    reconstruction_mae=float(error.mean());p95=float(np.percentile(error,95))
    if not np.isfinite(reconstruction_mae) or reconstruction_mae>1 or p95>2:
        raise ValueError('unmix_recomposition_failed')
    full=np.zeros((*source.shape[:2],4),np.uint8);full[y:y+h,x:x+w]=rgba
    backgrounds=[];prior=Image.new('RGBA',size);retained_pixels=0
    for owner in sorted(obj.replaces,key=lambda i:nodes[i]['z']):
        n=nodes[owner];asset=assets[n['asset_id']];file=(root/asset['path']).resolve()
        if not file.is_relative_to(root):raise ValueError('unmix_asset_path_escape')
        if asset.get('sha256') and digest(file)!=asset['sha256']:raise ValueError('unmix_asset_hash_mismatch')
        ox,oy,ow,oh=[round(v) for v in n['bbox']]
        if any(k in n for k in ('rotation','crop','flip_h','flip_v','transform')):raise ValueError('unmix_owner_transform')
        old=np.array(Image.open(file).convert('RGBA'))
        if old.shape!=(oh,ow,4):raise ValueError('unmix_owner_dimensions')
        if not np.all(old[...,3]==255):
            # Preserve pre-existing white/soft-alpha samples that cannot be
            # identified from source/background contrast alone. This original
            # ownership is still an independent semantic-review hypothesis.
            prior.alpha_composite(Image.fromarray(old),(ox,oy))
            continue
        if not np.array_equal(old[...,:3],source[oy:oy+oh,ox:ox+ow]):raise ValueError('unmix_owner_not_source_crop')
        oldregion=rectangle(n['bbox'],size)
        if (oldregion & blocked).any():raise ValueError('unmix_old_owner_text_or_locked_pixels')
        for neighbor in ir['nodes']:
            if neighbor['type']=='image' and neighbor['id'] not in obj.replaces and (oldregion & rectangle(neighbor['bbox'],size)).any():
                raise ValueError('unmix_old_owner_neighbor')
        patch=old.copy();patch[...,:3]=estimate[oy-y:oy-y+oh,ox-x:ox-x+ow]
        backgrounds.append({'id':'restored-bg-'+fingerprint({'owner':owner,'object':obj.id})[:24],
            'owner':owner,'bbox':n['bbox'],'old_z':n['z'],'rgba':patch,'source_asset_sha256':digest(file),
            'status':'PENDING_BACKGROUND_REVIEW','non_owned_pixels_unchanged':False,
            'backing_method':'source_lateral_background_plane','source_unmix_scope_hypothesis':True,
            'unverified_old_scope_pixels':int((rgba[oy-y:oy-y+oh,ox-x:ox-x+ow,3]==0).sum())})
    prior=np.array(prior);keep=(full[...,3]==0)&(prior[...,3]>0)
    if (keep & blocked).any():raise ValueError('unmix_protected_old_alpha')
    full[keep]=prior[keep];retained_pixels=int(keep.sum())
    # Recheck all retained Alpha against ORIGINAL pixels. A good result cannot
    # be inferred merely from having copied an old representation.
    local=full[y:y+h,x:x+w];active=local[...,3]>0;alpha=local[...,3:4]/255
    final_error=np.abs(local[...,:3]*alpha+estimate*(1-alpha)-crop)[active]
    if final_error.size and (final_error.mean()>2 or np.percentile(final_error,95)>4):
        raise ValueError('unmix_retained_alpha_recomposition_failed')
    reconstruction_mae=float(final_error.mean());p95=float(np.percentile(final_error,95))
    metadata={'schema':'SourceUnmixCandidate/1','scope_bbox':box,'ring_error':ring_error,
        'reconstruction_mae':reconstruction_mae,'reconstruction_p95':p95,'estimated_alpha':True,
        'retained_old_alpha_pixels':retained_pixels,
        'native_background_comparison':native_check,
        'method_binding':binding(),'independent_review_required':True}
    return full,metadata,backgrounds,estimate,mask
