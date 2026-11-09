"""Alpha-first object extraction with conservative, atomic SlideIR replacement.

The semantic inventory selects ownership. Connected components supply pixels only.
Unavailable SAM and failed gates are explicit fallbacks, never rectangular masks.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
import cv2
from .object_contracts import ObjectPlan
from .materialize import digest, write

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'ppt-rebuild/src'))
from layerd_loader import load_page, recompose


def rectangle(box,size,pad=0):
    w,h=size;x,y,bw,bh=box
    x0=max(0,int(np.floor(x-pad)));y0=max(0,int(np.floor(y-pad)))
    x1=min(w,int(np.ceil(x+bw+pad)));y1=min(h,int(np.ceil(y+bh+pad)))
    result=np.zeros((h,w),bool);result[y0:y1,x0:x1]=True
    return result


def component_support(alpha,box,size):
    """Return whole components, including disconnected fragments and edge pixels."""
    inside=rectangle(box,size);support=alpha>0
    count,labels,stats,_=cv2.connectedComponentsWithStats(support.astype('uint8'),8)
    selected=np.zeros_like(support);rejected=[]
    for index in np.unique(labels[inside & support]):
        component=labels==index;area=int(stats[index,cv2.CC_STAT_AREA]);overlap=int((component & inside).sum())
        # Box is a localization hint, never a hard cut. Reject implausibly large
        # connected regions; pass them to SAM instead of truncating the component.
        if overlap/max(1,area)<.85 or np.any(component & ~rectangle(box,size,pad=16)):
            rejected.append(int(index));continue
        selected|=component
    return selected,rejected


def visible_rgba(layers,masks):
    """Source-over selected layer pixels, attenuated by every higher layer.

    We export the visible contribution only, not hidden inpainted completions.
    Straight RGB is reconstructed from summed premultiplied contributions.
    """
    h,w=layers[0].shape[:2];trans=np.ones((h,w),np.float32)
    weighted=np.zeros((h,w,3),np.float32);alpha=np.zeros((h,w),np.float32)
    for rgba,mask in reversed(list(zip(layers,masks))):
        a=rgba[...,3].astype(np.float32)/255
        owned=a*trans*mask
        weighted+=rgba[...,:3]*owned[...,None];alpha+=owned
        trans*=1-a
    rgb=np.divide(weighted,alpha[...,None],out=np.zeros_like(weighted),where=alpha[...,None]>0)
    return np.dstack([np.rint(rgb).clip(0,255).astype('uint8'),np.rint(alpha*255).clip(0,255).astype('uint8')])


def load_sam(directory,obj,source_hash,plan_hash,size,version):
    """Only consume source/plan/version-bound, finite, full-canvas masks."""
    if directory is None:return None,'sam_not_configured'
    root=Path(directory).resolve();receipt=root/(obj.id+'.json')
    if not receipt.is_file():return None,'sam_result_missing'
    meta=json.loads(receipt.read_text(encoding='utf-8'))
    required={'source_sha256':source_hash,'plan_sha256':plan_hash,'version':version,'status':'SUCCEEDED','image_size':list(size)}
    if any(meta.get(k)!=v for k,v in required.items()):return None,'sam_binding_mismatch'
    file=(root/meta['mask_file']).resolve()
    if not file.is_relative_to(root) or digest(file)!=meta['mask_sha256']:return None,'sam_artifact_mismatch'
    if not meta.get('checkpoint_sha256') or not meta.get('code_revision'):return None,'sam_provenance_missing'
    with np.load(file,allow_pickle=False) as value:
        masks=value['masks'];scores=value['scores']
    if masks.ndim!=3 or masks.shape[1:]!=(size[1],size[0]) or len(scores)!=len(masks) or not np.isfinite(masks).all() or not np.isfinite(scores).all():return None,'sam_invalid_mask'
    roi=rectangle(obj.bbox,size);selected=np.zeros_like(roi)
    for mask,score in zip(masks,scores):
        mask=mask>.5;area=int(mask.sum())
        if score<.5 or not area:continue
        if (mask & roi).sum()/area>=.7:selected|=mask
    if not selected.any():return None,'sam_no_owned_candidate'
    return selected,None


def extract(planfile,layerdir,baseline_ir,output,*,route='alpha',sam_dir=None,rejected=None):
    """A completed directory is immutable; callers choose a new revision to retry."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    planfile=Path(planfile);plan=ObjectPlan.model_validate(json.loads(planfile.read_text(encoding='utf-8')))
    page=load_page(layerdir)
    if plan.source_sha256!=digest(page.source_path) or plan.image_size!=list(page.size):raise ValueError('object plan source mismatch')
    baseline_ir=Path(baseline_ir);ir=json.loads(baseline_ir.read_text(encoding='utf-8'))
    if tuple(ir['canvas']['source_px'])!=page.size:raise ValueError('baseline dimensions mismatch')
    baseline_source=(baseline_ir.parent/ir['source_image']).resolve()
    if not baseline_source.is_relative_to(baseline_ir.parent.resolve()):raise ValueError('baseline source escapes workspace')
    if not np.array_equal(np.array(Image.open(baseline_source).convert('RGB')),np.array(Image.open(page.source_path).convert('RGB'))):raise ValueError('baseline pixels mismatch')
    for a in ir['assets']:
        src=(baseline_ir.parent/a['path']).resolve()
        if not src.is_relative_to(baseline_ir.parent.resolve()):raise ValueError('baseline asset escapes workspace')
        target=output/a['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    shutil.copy2(baseline_source,output/ir['source_image'])
    layers=[np.array(Image.open(l.path).convert('RGBA')) for l in page.layers]
    foreground=visible_rgba(layers,[np.full((page.size[1],page.size[0]),l.z>0 and l.role!='background',bool) for l in page.layers])[...,3]>0
    layer_ids={l.file:i for i,l in enumerate(page.layers)}
    source=np.array(Image.open(page.source_path).convert('RGB')).astype('int16');composite=recompose(page).astype('int16')
    text_masks={n['id']:rectangle(n['bbox'],page.size) for n in ir['nodes'] if n['type']=='text'}
    text_mask=np.zeros((page.size[1],page.size[0]),bool)
    for m in text_masks.values():text_mask|=m
    occupied=np.zeros_like(text_mask);records=[];assets=[];original_texts=copy.deepcopy([n for n in ir['nodes'] if n['type']=='text'])
    for obj in plan.objects:
        row={'id':obj.id,'description':obj.description,'route_requested':route,'status':'FALLBACK','method':None,'reasons':[],'replaces':obj.replaces}
        records.append(row)
        if obj.id in (rejected or {}):row['reasons']=['independent_visual_review'];row['review_findings']=rejected[obj.id];continue
        if not set(obj.layers).issubset(layer_ids):row['reasons']=['unknown_layer'];continue
        masks=[np.zeros_like(text_mask) for _ in layers];difficult=False
        for name in obj.layers:
            index=layer_ids[name];mask,component_rejected=component_support(layers[index][...,3],obj.bbox,page.size)
            masks[index]=mask;difficult|=bool(component_rejected)
        method='alpha'
        if difficult or not any(m.any() for m in masks):
            if route=='alpha':row['reasons']=['alpha_ambiguous_or_empty'];continue
            support,error=load_sam(sam_dir,obj,plan.source_sha256,digest(planfile),page.size,route)
            if error:row['reasons']=[error];continue
            # Preserve partial-alpha boundary pixels adjacent to SAM support. No
            # geometric dilation of opaque pixels or invented missing strokes.
            neighbor=cv2.dilate(support.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)
            for name in obj.layers:
                index=layer_ids[name];a=layers[index][...,3]
                masks[index]=(support|((a>0)&(a<255)&neighbor))&(a>0)
            method=route
        rgba=visible_rgba(layers,masks);support=rgba[...,3]>0
        row.update(method=method,visible_pixels=int(support.sum()))
        if not support.any():row['reasons']=['no_visible_pixels'];continue
        # Native text always wins, including when a model mislabels it intrinsic.
        # Reject the entire proposal rather than punching glyph-sized holes into it.
        if (support & text_mask).any():row['reasons']=['protected_native_text_overlap'];continue
        if (support & occupied).any():row['reasons']=['object_ownership_overlap'];continue
        nodes={n['id']:n for n in ir['nodes']}
        if any(i not in nodes or nodes[i]['type']=='text' for i in obj.replaces):row['reasons']=['invalid_replacement_owner'];continue
        # Old crop rectangles often include blank padding. Permit that padding
        # only when no other foreground evidence would be removed. Native shapes
        # still need complete geometric ownership, not a colour resemblance.
        roi=rectangle(obj.bbox,page.size,pad=3)
        unsafe=False
        for identifier in obj.replaces:
            old=nodes[identifier];old_roi=rectangle(old['bbox'],page.size)
            if old['type']=='image':
                if (old_roi & ~rectangle(obj.bbox,page.size,pad=16)).any() or (old_roi & foreground & ~support).any():unsafe=True
            elif (old_roi & ~roi).any():unsafe=True
        if unsafe:row['reasons']=['replacement_contains_other_content'];continue
        # Existing opaque crops must be removed atomically or remain untouched.
        conflicts=[n['id'] for n in ir['nodes'] if n['type']=='image' and n['id'] not in obj.replaces and (rectangle(n['bbox'],page.size)&support).any()]
        if conflicts:row['reasons']=['unclaimed_existing_image'];row['conflicts']=conflicts;continue
        conflicts=[n['id'] for n in ir['nodes'] if n['type']=='shape' and n['id'] not in obj.replaces
                   and n['bbox'][2]*n['bbox'][3]<=obj.bbox[2]*obj.bbox[3]*1.2
                   and (rectangle(n['bbox'],page.size)&support).any()]
        if conflicts:row['reasons']=['unclaimed_existing_shape'];row['conflicts']=conflicts;continue
        # LayerD itself can have altered edge colours. This measures source
        # contribution, not semantic completeness or final-slide acceptance.
        mae=float(np.abs(composite-source)[support].mean());row['layerd_source_mae']=round(mae,4)
        if mae>10:row['reasons']=['layerd_source_fidelity'];continue
        ys,xs=np.nonzero(support);x0,y0,x1,y1=int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
        filename='object-'+obj.id+'.png';Image.fromarray(rgba[y0:y1,x0:x1]).save(output/filename)
        maskname='object-'+obj.id+'-mask.png';Image.fromarray(support.astype('uint8')*255).save(output/maskname)
        assetid='extracted-'+obj.id;evid='ev-extracted-'+obj.id;nodeid='extracted-'+obj.id
        if nodeid in nodes:raise ValueError('generated object ID collision')
        old_asset_ids={nodes[i].get('asset_id') for i in obj.replaces}
        old_evidence=set(e for i in obj.replaces for e in nodes[i].get('evidence_ids',[]))
        for a in ir['assets']:
            if a['id'] in old_asset_ids:old_evidence.update(a.get('contains_evidence_ids',[]))
        candidate=copy.deepcopy(ir)
        candidate['nodes']=[n for n in candidate['nodes'] if n['id'] not in obj.replaces]
        candidate['assets']=[a for a in candidate['assets'] if a['id'] not in old_asset_ids]
        candidate['evidence']=[e for e in candidate['evidence'] if e['id'] not in old_evidence]
        if any(set(g.get('members',[]))&set(obj.replaces) for g in candidate.get('groups',[])):
            row['reasons']=['existing_group_requires_whole_group_transaction'];continue
        z=min([nodes[i]['z'] for i in obj.replaces] or [max(n['z'] for n in nodes.values())+1])
        box=[x0,y0,x1-x0,y1-y0]
        candidate['nodes'].append({'id':nodeid,'type':'image','bbox':box,'z':z,'asset_id':assetid,'evidence_ids':[]})
        candidate['assets'].append({'id':assetid,'path':filename,'purpose':'module','sha256':digest(output/filename),'contains_evidence_ids':[evid],'description':obj.description})
        candidate['evidence'].append({'id':evid,'kind':'image','bbox':box,'description':obj.description})
        # Commit only after all gates; no mutation from a failed member survives.
        ir=candidate;occupied|=support
        row.update(status='EXTRACTED',bbox=box,asset=filename,asset_sha256=digest(output/filename),mask=maskname,reasons=[],semantic_acceptance=False)
        assets.append(filename)
    # Keep native text fields, including z, byte-for-byte identical. Stable sort
    # preserves same-z ordering without silently changing the frozen baseline.
    ir['nodes'].sort(key=lambda n:n['z'])
    if [n for n in ir['nodes'] if n['type']=='text']!=original_texts:raise ValueError('text baseline changed')
    ir.setdefault('review_notes',[]).append('Alpha/SAM object extraction experiment; semantic completeness and human acceptance are not implied by structural gates.')
    write(output/'slideir.json',ir)
    write(output/'authoring.json',{'tool':'image2PPT object extraction','model_id':'frozen-object-plan','ir_sha256':digest(output/'slideir.json'),'decisions':['Alpha-first atomic object replacement; frozen text styles']})
    report={'schema':'ObjectExtraction/1','route':route,'source_sha256':plan.source_sha256,'plan_sha256':digest(planfile),'manifest_sha256':digest(Path(layerdir)/'manifest.json'),'objects':records,'extracted':sum(r['status']=='EXTRACTED' for r in records),'fallback':sum(r['status']=='FALLBACK' for r in records),'human_acceptance':False}
    write(output/'extraction.json',report)
    return output/'slideir.json',report
