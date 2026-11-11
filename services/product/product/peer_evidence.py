"""Source-verified peer masks for foreground and shared text-bearing backings."""
import json
from pathlib import Path
import numpy as np
from .contracts import intersection
from .materialize import digest
from .object_extract import rectangle


def roles(graph):
    surfaces={s.id:s for s in graph.surface.shapes}
    aliases={}
    for relation in graph.relations:
        if (relation.kind=='part_of' and relation.status=='RESOLVED' and relation.target in surfaces
                and 'asset-background-intent' in relation.evidence):
            aliases.setdefault(relation.source,[]).append(relation.target)
    return {a.id:{'role':'BACKGROUND_PANEL_CANDIDATE' if a.id in aliases else
                  'FOREGROUND_OBJECT' if a.kind in ('icon','logo') else 'UNCERTAIN_COMPOSITE',
                  'surface_ids':aliases.get(a.id,[])} for a in graph.asset.objects}


def panel_evidence(asset,surface,graph,source):
    """Fit independently corroborated source backing, retaining all outliers."""
    size=source.shape[1::-1]
    scope=rectangle(asset.bbox,size)&rectangle(surface.bbox,size)
    excluded=np.zeros(source.shape[:2],bool)
    for text in graph.text.texts:excluded|=rectangle(text.bbox,size)
    classified=roles(graph)
    for peer in graph.asset.objects:
        if classified[peer.id]['role']!='BACKGROUND_PANEL_CANDIDATE':excluded|=rectangle(peer.bbox,size)
    x,y,w,h=map(round,surface.bbox)
    yy,xx=np.indices(source.shape[:2]);u=(xx-x)/max(w,1);v=(yy-y)/max(h,1)
    fill=np.array([int(surface.fill[i:i+2],16) for i in (0,2,4)])
    plausible=np.abs(source.astype(float)-fill).max(2)<=20
    samples=scope&~excluded&plausible&((xx%4)==0)&((yy%4)==0)
    if samples.sum()<100:return None,{'status':'UNKNOWN','reason':'insufficient independent panel pixels'}
    design=np.stack((np.ones_like(u),u,v),2)
    keep=samples.copy()
    for _ in range(3):
        values=source[keep].astype(float);matrix=design[keep]
        coeff=np.linalg.lstsq(matrix,values,rcond=None)[0]
        predicted=design@coeff
        error=np.abs(source.astype(float)-predicted).max(2)
        keep=samples&(error<=4)
        if keep.sum()<100:return None,{'status':'UNKNOWN','reason':'panel source fit is discontinuous'}
    agreement=float(keep.sum()/samples.sum())
    sy,sx=np.where(keep)
    if agreement<.85 or np.ptp(sx)<w*.5 or np.ptp(sy)<h*.3:
        return None,{'status':'UNKNOWN','reason':'panel continuity or spatial support unproved','agreement':agreement}
    backing=np.clip(np.rint(predicted),0,255).astype('uint8')
    # A backing relation authorizes no erasure. These pixels only become usable
    # source context. Every other contrast pixel remains protected unless the
    # existing source glyph model independently explains the target below.
    clean=scope&(np.abs(source.astype(float)-backing.astype(float)).max(2)<=6)
    return (clean,backing,scope),{'status':'SOURCE_BACKING_VERIFIED','agreement':agreement,
        'support_pixels':int(keep.sum()),'background_pixels':int(clean.sum()),
        'coefficients':coeff.tolist(),'method':'surface-intent+source-planar-inliers/1'}


def verified_owner(identifier,graph,source,baseline):
    if baseline is None:return None
    baseline=Path(baseline).resolve()
    scene=json.loads((baseline/'scene.json').read_text('utf8'))
    if scene['source_sha256']!=graph.source_sha256 or digest(baseline/'source.png')!=graph.source_sha256:
        raise ValueError('peer owner source binding mismatch')
    record=scene['selected'].get(identifier)
    if not record:return None
    if record.get('kind')!='asset' or not record.get('complete_owner_scope') or record.get('source_sha256')!=graph.source_sha256:
        raise ValueError('peer source owner closure unproved')
    proof=record.get('owner_evidence',{});file=(baseline/proof.get('file','')).resolve()
    if not file.is_relative_to(baseline) or not file.is_file() or digest(file)!=proof.get('sha256'):
        raise ValueError('peer owner proof mismatch')
    with np.load(file,allow_pickle=False) as data:mask=data['mask']
    import hashlib
    if mask.dtype!=bool or hashlib.sha256(mask.tobytes()).hexdigest()!=record.get('mask_sha256'):
        raise ValueError('peer source owner mask mismatch')
    if mask.shape!=source.shape[:2]:raise ValueError('peer owner mask dimensions differ')
    return mask


def protection(text,graph,source,blocked,baseline=None,*,purpose='erase',diagnostics=None):
    from .parallel_fusion import glyph_evidence
    classified=roles(graph);surfaces={s.id:s for s in graph.surface.shapes}
    size=source.shape[1::-1];context=blocked.copy();panels=[];rows=[]
    for peer in graph.text.texts:
        if peer.id==text.id:continue
        mask=rectangle(peer.bbox,size);context|=mask
        rows.append((peer.id,'TEXT',mask,{'method':'independent-text-scope'}))
    for peer in graph.asset.objects:
        role=classified[peer.id];mask=verified_owner(peer.id,graph,source,baseline)
        if mask is not None:
            context|=mask
            rows.append((peer.id,role['role'],mask,{'method':'hash-bound-verified-owner-mask'}))
            continue
        mask=rectangle(peer.bbox,size)
        report={'method':'unproved-owner-conservative-scope'}
        if role['role']=='BACKGROUND_PANEL_CANDIDATE' and intersection(peer.bbox,text.bbox):
            containing=[surfaces[i] for i in role['surface_ids'] if intersection(surfaces[i].bbox,text.bbox)>0]
            if not containing:
                context|=mask;rows.append((peer.id,role['role'],mask,report));continue
            surface=max(containing,key=lambda b:intersection(b.bbox,text.bbox))
            evidence,report=panel_evidence(peer,surface,graph,source)
            report['surface_id']=surface.id
            if evidence is not None:
                # Reading context does not claim pixels. The existing independent
                # local fit must still pass before a source-derived glyph mask
                # can exempt any target ink from this backing's veto.
                clean,_,_=evidence
                panels.append((peer.id,mask&~clean,report))
                continue
        context|=mask;rows.append((peer.id,role['role'],mask,report))
    if purpose=='context':return context
    if purpose not in ('erase','both'):raise ValueError('unsupported protection purpose')
    allowed=np.zeros(source.shape[:2],bool)
    if panels:
        from .parallel_rebuild import background_model
        from .recovery.gradient_backing import fit_gradient
        from .recovery.text_mask_completion import TextMaskCompletion
        from .recovery.background_adapter import RecoveryBlocked
        x,y,w,h=map(round,text.bbox);box=[x-3,y-3,w+6,h+6]
        try:
            if min(box[:2])<0 or x+w+3>size[0] or y+h+3>size[1]:raise ValueError('target leaves source')
            backing,_=background_model(source,box,context)
            style=next(s for s in graph.text.styles if s.id==text.style)
            ink,_=glyph_evidence(source,box,style.color,backing)
            original=np.zeros(source.shape[:2],bool);original[y-3:y+h+3,x-3:x+w+3]=ink
            reference,fit=fit_gradient(source,box,context)
            completion=TextMaskCompletion.analyze(source,original,context,box,reference,glyph_height=style.size_px)
            if fit['validation_p95_rgb']<=3 and not completion.report['unassigned_contrast_pixels']:
                allowed=completion.masks['semantic_completion']
        except (ValueError,RecoveryBlocked) as exc:
            for _,_,report in panels:report['target_mask_failure']=str(exc)
    protected=context.copy()
    for identifier,mask,report in panels:
        mask=mask&~allowed
        report.update(target_authorized_pixels=int(allowed.sum()),remaining_unknown_pixels=int(mask.sum()))
        protected|=mask
        rows.append((identifier,'BACKGROUND_PANEL_CANDIDATE',mask,report))
    if diagnostics is not None:
        for identifier,role,mask,evidence in rows:
            diagnostics.append({'object_id':identifier,'role':role,'mask':mask,'evidence':evidence})
    return (protected,context) if purpose=='both' else protected


def save_diagnostics(directory,text,graph,source,protected,context,rows):
    from .materialize import write
    from .parallel_store import key
    from PIL import Image
    directory=Path(directory)
    old=np.zeros(source.shape[:2],bool);size=source.shape[1::-1]
    for peer in [*[t for t in graph.text.texts if t.id!=text.id],*graph.asset.objects]:old|=rectangle(peer.bbox,size)
    target=rectangle(text.bbox,size);masks={'protected':protected,'context_protected':context,'legacy_bbox_protected':old}
    for index,row in enumerate(rows):
        mask=row.pop('mask');name='peer_'+str(index);masks[name]=mask
        row.update(mask_key=name,pixels=int(mask.sum()),target_intersection_pixels=int((mask&target).sum()))
    np.savez_compressed(directory/'peer-protection.npz',**masks)
    write(directory/'peer-protection.json',{'schema':'PeerProtectionEvidence/1','source_sha256':graph.source_sha256,
        'scene_sha256':key(graph.model_dump()),'object_id':text.id,'mask_sha256':digest(directory/'peer-protection.npz'),
        'legacy_protected_pixels':int(old.sum()),'protected_pixels':int(protected.sum()),
        'removed_bbox_protection_pixels':int((old&~protected).sum()),
        'readable_context_pixels':int((~context).sum()),'peers':rows,
        'authority':'source context is read-only; original TextMaskCompletion and transactional ownership remain mandatory'})
    x,y,w,h=map(round,text.bbox);left=max(0,x-16);top=max(0,y-8);right=min(size[0],x+w+16);bottom=min(size[1],y+h+8)
    preview=source[top:bottom,left:right].copy()
    preview[protected[top:bottom,left:right]]=[220,60,60]
    Image.fromarray(preview).save(directory/'peer-protection.png')
