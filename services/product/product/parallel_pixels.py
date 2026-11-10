"""Verified RGBA contribution masks; a search box is never treated as ink."""
import numpy as np
from PIL import Image
from .materialize import digest

def graphic_ink(stable,size,source_sha256):
    result=np.zeros((size[1],size[0]),bool)
    for r in stable:
        if r.get('source_sha256',source_sha256)!=source_sha256 or digest(r['path'])!=r['sha256']:continue
        x,y,w,h=[round(v) for v in r['bbox']]
        if min(x,y)<0 or x+w>size[0] or y+h>size[1]:continue
        rgba=np.array(Image.open(r['path']).convert('RGBA'))
        if rgba.shape==(h,w,4):result[y:y+h,x:x+w]|=rgba[...,3]>0
    return result
