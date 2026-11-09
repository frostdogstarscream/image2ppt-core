"""Conservative source-pixel ownership; rectangles are search regions, not ink.

Only supported solid native backgrounds explain opaque source-crop padding.
Unknown geometry/transform/colour is never silently classified as background.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .object_extract import rectangle


def colour(value):
    return tuple(bytes.fromhex(value)) if isinstance(value,str) and len(value)==6 else None


def shape_pixels(node,size):
    canvas=Image.new('L',size);draw=ImageDraw.Draw(canvas)
    x,y,w,h=node['bbox'];box=(x,y,x+w-1,y+h-1)
    geometry=node.get('geometry');fill=node.get('fill')
    if fill=='none':return np.zeros((size[1],size[0]),bool),False
    if colour(fill) is None or node.get('fill_gradient'):return np.array(canvas)>0,False
    if geometry=='rect':draw.rectangle(box,fill=255)
    elif geometry=='ellipse':draw.ellipse(box,fill=255)
    elif geometry=='roundRect':
        radius=node.get('corner_radius_px')
        # The maximum-radius pill is contained in every valid rounded rectangle,
        # regardless of the renderer's unspecified corner adjustment.
        if radius is None:
            radius=min(w,h)/2
            draw.rounded_rectangle(box,radius=radius,fill=255)
            import cv2
            edge=max(1,int(np.ceil(node.get('line',{}).get('width_pt',0)/.75)))
            return cv2.erode(np.array(canvas),np.ones((2*edge+1,2*edge+1),np.uint8)).astype(bool),True
        draw.rounded_rectangle(box,radius=radius,fill=255)
    else:return np.array(canvas)>0,False
    # Boundaries vary between renderers. They remain unknown.
    import cv2
    edge=max(1,int(np.ceil(node.get('line',{}).get('width_pt',0)/.75)))
    return cv2.erode(np.array(canvas),np.ones((2*edge+1,2*edge+1),np.uint8)).astype(bool),True


def background(ir,size,z):
    rgb=np.zeros((size[1],size[0],3),np.uint8)
    base=colour(ir.get('background',{}).get('color'))
    known=np.full((size[1],size[0]),base is not None,bool)
    if base:rgb[:]=base
    for n in sorted(ir['nodes'],key=lambda n:n['z']):
        if n['z']>=z or n['type']!='shape':continue
        region=rectangle(n['bbox'],size);mask,supported=shape_pixels(n,size)
        known[region]=False
        if supported:rgb[mask]=colour(n['fill']);known[mask]=True
    return rgb,known


def image_pixels(ir,node,root,size,source):
    """Return true visible content of a legacy image, with explainable padding.

    Transparency, white ink and soft edges retain ownership. Padding exclusion
    applies only to an exact opaque source crop over a known native background.
    """
    if any(k in node for k in ('rotation','crop','flip_h','flip_v','transform')):
        return rectangle(node['bbox'],size),False
    asset=next((a for a in ir['assets'] if a['id']==node.get('asset_id')),None)
    if asset is None:return rectangle(node['bbox'],size),False
    root=Path(root).resolve();file=(root/asset['path']).resolve()
    if not file.is_relative_to(root):raise ValueError('asset path escape')
    from .materialize import digest
    if asset.get('sha256') and digest(file)!=asset['sha256']:raise ValueError('asset hash mismatch')
    x,y,w,h=[round(v) for v in node['bbox']]
    image=Image.open(file).convert('RGBA').resize((w,h),Image.Resampling.BICUBIC)
    rgba=np.array(image);mask=np.zeros((size[1],size[0]),bool)
    if min(x,y)<0 or x+w>size[0] or y+h>size[1]:return rectangle(node['bbox'],size),False
    content=rgba[...,3]>0
    original=source[y:y+h,x:x+w]
    if np.all(rgba[...,3]==255) and np.array_equal(rgba[...,:3],original):
        bg,known=background(ir,size,node['z'])
        # Only exact, native-explainable pixels are padding. AA remains content.
        padding=known[y:y+h,x:x+w] & (np.abs(original.astype('int16')-bg[y:y+h,x:x+w]).max(2)<=1)
        content&=~padding
    mask[y:y+h,x:x+w]=content
    return mask,True


def text_evidence(ir,source,size):
    hard=np.zeros((size[1],size[0]),bool);uncertain=hard.copy();records=[]
    for node in ir['nodes']:
        if node['type']!='text':continue
        roi=rectangle(node['bbox'],size)
        colors={node.get('color')}
        for p in node.get('paragraphs',[]):
            colors.add(p.get('color'))
            colors.update(r.get('color') for r in p.get('runs',[]))
        colors=[colour(c) for c in colors if colour(c) is not None]
        bg,known=background(ir,size,node['z'])
        if not colors:
            uncertain|=roi;records.append({'id':node['id'],'evidence':'missing_native_style'});continue
        ink=np.zeros_like(roi);faint=np.zeros_like(roi)
        for c in colors:
            v=np.array(c,dtype=np.float32)-bg.astype('float32')
            observed=source.astype('float32')-bg.astype('float32')
            denominator=(v*v).sum(2)
            t=(observed*v).sum(2)/np.maximum(denominator,1)
            error=np.abs(observed-v*t[...,None]).max(2)
            # Native colour + source antialias mixture are independent evidence
            # from an Alpha proposal; this is not a mask approving itself.
            ink|=roi & known & (denominator>25) & (t>.12) & (t<1.05) & (error<3)
            faint|=roi & known & (denominator>25) & (t>.005) & (t<=.12) & (error<3)
        from .object_text import native_glyph
        glyph=native_glyph(node,size)
        confirmed=ink & glyph if glyph is not None else np.zeros_like(ink)
        hard|=confirmed
        uncertain|=(roi & ~known)|(ink & ~confirmed)|faint
        records.append({'id':node['id'],'source_colour_ink':int(ink.sum()),'unknown_background':int((roi & ~known).sum())})
    return hard,uncertain,records


def replacement_proof(ir,obj,support,root,source,size,cache=None,background_assets=None):
    nodes={n['id']:n for n in ir['nodes']};claimed=set(obj.replaces)
    if not set(obj.required_owners)<=claimed:return False,'incomplete_owner_set',{}
    if any(i not in nodes or nodes[i]['type']=='text' for i in claimed):return False,'invalid_replacement_owner',{}
    if any(set(g.get('members',[])) & claimed for g in ir.get('groups',[])):
        return False,'group_transaction_requires_group_preservation',{}
    proof={};unclaimed=[];pending=False
    for n in ir['nodes']:
        if n['type']!='image':continue
        expected=(background_assets or {}).get(n['id'])
        if expected:
            asset=next(a for a in ir['assets'] if a['id']==n['asset_id'])
            if asset.get('sha256')!=expected:return False,'stable_background_changed',proof
            continue
        cache=cache if cache is not None else {}
        if n['id'] not in cache:cache[n['id']]=image_pixels(ir,n,root,size,source)
        mask,known=cache[n['id']]
        if n['id'] in claimed:
            missing=int((mask & ~support).sum());proof[n['id']]={'unexplained_pixels':missing,'known':known}
            if not known:return False,'replacement_contains_other_content',proof
            if missing:pending=True
        elif (mask & support).any():unclaimed.append(n['id'])
    if unclaimed:return False,'incomplete_owner_set',{'unclaimed':unclaimed,**proof}
    # Small native primitives inside the object may be old fragments. Preserve
    # large cards/backgrounds; never delete either based on box intersection.
    for n in ir['nodes']:
        if n['type']!='shape' or n['id'] in claimed:continue
        mask,known=shape_pixels(n,size)
        region=rectangle(n['bbox'],size)
        if region.sum()>max(1,support.sum())*4:continue
        if (mask & support).any() or (not known and (region & support).any()):
            return False,'unclaimed_native_contribution',{'shape':n['id'],**proof}
    for i in claimed:
        n=nodes[i]
        if n['type']=='shape':
            mask,known=shape_pixels(n,size)
            if not known or (mask & ~support).any():return False,'shape_not_wholly_owned',proof
    if pending:return False,'replacement_contains_other_content',proof
    return True,None,proof
