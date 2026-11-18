"""Compare experimental foreground candidates inside the existing owner gate."""
import base64
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image

from .experiment import invoke
from .materialize import digest, write


def candidate_seeds(op, obj, source, protected, directory, layer_page):
    x,y,w,h=map(round,obj.bbox)
    H,W=source.shape[:2]
    pad=max(8,round(min(w,h)*.15))
    x0,y0=max(0,x-pad),max(0,y-pad)
    x1,y1=min(W,x+w+pad),min(H,y+h+pad)
    image=Image.fromarray(source[y0:y1,x0:x1])
    if op=='local-layerd':
        result=invoke(op,image,directory)
        available=[]
        for row in result.get('layers',[]):
            rgba=np.asarray(Image.open(row['path']).convert('RGBA'))
            support=rgba[...,3]>8
            if rgba.shape[:2]!=(y1-y0,x1-x0) or not support.any():
                continue
            inside=support[max(0,y-y0):y+h-y0,max(0,x-x0):x+w-x0].sum()
            if inside>=.9*support.sum():
                available.append((rgba,{'method':op,'layer_sha256':row['sha256']}))
    else:
        from .object_closure import assemble
        arrays=[np.array(Image.open(l.path).convert('RGBA')) for l in layer_page.layers]
        seed,ambiguous,missing=assemble(obj,arrays,[l.file for l in layer_page.layers],(W,H))
        coarse=seed[y0:y1,x0:x1,3]
        if ambiguous or missing or not coarse.any():
            write(directory/'receipt.json',{'status':'rejected','reason':'no unambiguous coarse LayerD mask'})
            return []
        encoded=io.BytesIO()
        Image.fromarray(coarse).save(encoded,'PNG')
        result=invoke(op,image,directory,mask=base64.b64encode(encoded.getvalue()).decode())
        available=[(np.array(Image.open(result['rgba']).convert('RGBA')),{'method':op,'coarse_mask_source':'LayerD'})] if 'rgba' in result else []
    seeds=[]
    for candidate,meta in available:
        if candidate.shape!=(y1-y0,x1-x0,4) or (protected[y0:y1,x0:x1] & (candidate[...,3]>0)).any():
            continue
        full=np.zeros((H,W,4),np.uint8)
        full[y0:y1,x0:x1]=candidate
        seeds.append((full,{**meta,'source_window':[x0,y0,x1-x0,y1-y0]}))
    return seeds


def trace_asset(png_file, stage, identifier, directory):
    import fitz
    image=Image.open(png_file).convert('RGBA')
    array=np.array(image)
    colors=array[array[...,3]>200,:3]//32
    if colors.size==0 or len(np.unique(colors,axis=0))>16:
        return
    result=invoke('vtracer',image,directory)
    if 'svg' not in result:return
    path=Path(result['svg'])
    with fitz.open(stream=path.read_bytes(),filetype='svg') as doc:
        rect=doc[0].rect
        pix=doc[0].get_pixmap(matrix=fitz.Matrix(image.width/rect.width,image.height/rect.height),alpha=True)
        rendered=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,4)
    if rendered.shape!=array.shape:
        write(directory/'selection.json',{'status':'rejected','reason':'render size mismatch'})
        return
    alpha=array[...,3:4]/255
    own=array[...,:3]*alpha
    coverage=np.maximum(rendered[...,3],array[...,3])>8
    errors=[]
    for bg in (0,255):
        err=np.abs(own+bg*(1-alpha)-rendered[...,:3]-bg*(1-rendered[...,3:4]/255))
        errors.append(float(np.percentile(err.max(2)[coverage],95)))
    if max(errors)>15:
        write(directory/'selection.json',{'status':'rejected','contrast_p95':errors})
        return
    target=stage/(identifier+'.svg')
    target.write_bytes(path.read_bytes())
    sidecar=stage/'vector-assets.json'
    records=json.loads(sidecar.read_text('utf8')) if sidecar.exists() else {}
    records[identifier]={'path':target.name,'sha256':digest(target),'png_sha256':digest(png_file)}
    write(sidecar,records)
    write(directory/'selection.json',{'status':'staged','contrast_p95':errors})

def contrast_edge_score(graphic):
    """Proxy for spill on a changed background on near-flat, antialiased edges."""
    import cv2
    import numpy as np

    pixels = np.asarray(graphic, dtype=float)
    alpha = pixels[..., 3] / 255
    core = alpha >= .97
    if not core.any():
        return None
    distance, labels = cv2.distanceTransformWithLabels((~core).astype('uint8'),cv2.DIST_L2,5,
                                                        labelType=cv2.DIST_LABEL_PIXEL)
    edge = (alpha > .08) & (alpha < .92) & (distance <= 3)
    if edge.sum() < 8:
        return None
    colors = np.zeros((int(labels.max())+1,3))
    colors[labels[core]] = pixels[...,:3][core]
    a = alpha[edge,None]
    rgb = pixels[...,:3][edge]
    # Treat the adjacent opaque interior as a local foreground estimate.
    # This is an edge proxy, not a labelled alternate-background reference.
    intended = colors[labels[edge]]
    backdrop = np.array([112,54,140])
    difference = abs((rgb*a+backdrop*(1-a))-(intended*a+backdrop*(1-a))).max(1)
    return {'p90':float(np.percentile(difference,90)), 'samples':int(edge.sum()),
            'visible':int((alpha>.08).sum())}

