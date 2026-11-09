"""Staged background restoration, never an implicit background classification.

Non-owned old pixels are retained exactly. Hidden backing comes only from the
existing lower LayerD stack and needs an independent deletion-preview review.
No white keying, glyph erasing, or new image-generation model is involved.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from .materialize import digest
from .object_registry import fingerprint
from .object_extract import rectangle


def restoration(ir,obj,rgba,layers,names,root,source,hard_text,unknown_text,*,native=False,owner_scope=False):
    root=Path(root).resolve();size=(source.shape[1],source.shape[0]);support=rgba[...,3]>0
    nodes={n['id']:n for n in ir['nodes']};assets={a['id']:a for a in ir['assets']}
    if not obj.replaces:return None
    # The model cannot silently relabel another native shape as a background.
    if any(i not in nodes or nodes[i]['type']!='image' for i in obj.replaces):return None
    selected=min(names.index(name) for name in obj.layers)
    if selected==0:return None
    backing=Image.new('RGBA',size)
    for layer in layers[:selected]:backing=Image.alpha_composite(backing,Image.fromarray(layer))
    under=np.array(backing)
    if np.any(support & (under[...,3]!=255)):return None
    native_rgb=native_known=None
    if native:
        from .object_ownership import background
        native_rgb,native_known=background(ir,size,min(nodes[i]['z'] for i in obj.replaces))
        for node in ir['nodes']:
            if node['type']=='image' and node['id'] not in obj.replaces:
                native_known[rectangle(node['bbox'],size)]=False
    rows=[]
    for owner in obj.replaces:
        node=nodes[owner]
        if any(k in node for k in ('rotation','crop','flip_h','flip_v','transform')):return None
        asset=assets[node['asset_id']];file=(root/asset['path']).resolve()
        if not file.is_relative_to(root):raise ValueError('background source escapes scene')
        old=np.array(Image.open(file).convert('RGBA'));x,y,w,h=[round(v) for v in node['bbox']]
        if old.shape!=(h,w,4) or not np.all(old[...,3]==255) or not np.array_equal(old[...,:3],source[y:y+h,x:x+w]):return None
        region=rectangle(node['bbox'],size)
        # Source native glyph evidence cannot become a raster background patch.
        if (region & (hard_text|unknown_text) & ~support).any():return None
        a=rgba[y:y+h,x:x+w,3].astype('float32')/255
        foreground=rgba[y:y+h,x:x+w,:3].astype('float32')
        colour=old[...,:3].astype('float32');remaining=1-a
        solved=np.divide(colour-foreground*a[...,None],remaining[...,None],
            out=under[y:y+h,x:x+w,:3].astype('float32').copy(),where=remaining[...,None]>.005)
        partial=(a>0)&(a<.995)
        unphysical=partial & ((solved.min(2)<-2)|(solved.max(2)>257))
        # LayerD edges may differ slightly from uploaded RGB. Do not harden
        # Alpha or invent a foreground colour to force an exact inverse. Keep
        # the existing lower-layer backing and require explicit visual review.
        solved[unphysical]=under[y:y+h,x:x+w,:3][unphysical]
        patch=old.copy();inside=a>0
        if native:
            if np.any(inside & ~native_known[y:y+h,x:x+w]):return None
            if owner_scope:
                # This is a DIFFERENT, explicitly unverified ownership claim:
                # every old crop pixel must have native geometric backing, no
                # competing image or text evidence. Independent source review
                # must additionally prove the old crop contains ONLY this object
                # and its background before any scene/asset publication.
                if not native_known[y:y+h,x:x+w].all() or (region & (hard_text|unknown_text)).any():return None
                inside=np.ones_like(inside)
            patch[inside,:3]=native_rgb[y:y+h,x:x+w][inside]
        else:patch[inside,:3]=np.rint(solved[inside]).clip(0,255).astype('uint8')
        # Preserve every non-owned source pixel byte-for-byte, including unknown
        # pixels. A reviewer must reject leftover icon/neighbor content here.
        if not np.array_equal(patch[~inside],old[~inside]):raise ValueError('non-owned background pixels changed')
        name='restored-bg-'+fingerprint({'owner':owner,'object':obj.id})[:24]
        rows.append({'id':name,'owner':owner,'bbox':node['bbox'],'old_z':node['z'],'rgba':patch,
            'source_asset_sha256':digest(file),'status':'PENDING_BACKGROUND_REVIEW',
            'non_owned_pixels_unchanged':not owner_scope,'backing_layers':names[:selected],
            'backing_method':'verified_native_fill' if native else 'layerd_inverse',
            'native_owner_scope_hypothesis':bool(owner_scope),
            'unverified_old_scope_pixels':int((a==0).sum()) if owner_scope else 0,
            'noninvertible_edge_pixels':int(unphysical.sum())})
    return rows
