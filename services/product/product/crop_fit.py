"""Expand a small source crop to include connected ink cut by its boundary."""
import math
import cv2
import numpy as np
from PIL import Image
from .contracts import Page,intersection


def complete_crops(page,source):
    image=np.asarray(Image.open(source).convert('RGB'))
    height,width=image.shape[:2];data=page.model_dump();changes=[]
    for crop in data['crops']:
        x,y,w,h=crop['bbox']
        if w*h>width*height*.04:continue
        pad=min(40,max(12,round(max(w,h)*.6)))
        left=max(0,math.floor(x)-pad);top=max(0,math.floor(y)-pad)
        right=min(width,math.ceil(x+w)+pad);bottom=min(height,math.ceil(y+h)+pad)
        roi=image[top:bottom,left:right].astype(np.float32)
        corners=np.concatenate([roi[:3,:3].reshape(-1,3),roi[:3,-3:].reshape(-1,3),
                                roi[-3:,:3].reshape(-1,3),roi[-3:,-3:].reshape(-1,3)])
        bg=np.median(corners,axis=0)
        mask=(np.linalg.norm(roi-bg,axis=2)>65).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        original=labels[max(0,round(y)-top):min(roi.shape[0],round(y+h)-top),
                        max(0,round(x)-left):min(roi.shape[1],round(x+w)-left)]
        ids,freq=np.unique(original,return_counts=True)
        boxes=[stats[i] for i,n in zip(ids,freq) if i and n>=3 and stats[i,4]>=5]
        if not boxes:continue
        nx=max(0,min(x,left+min(s[0] for s in boxes)-2));ny=max(0,min(y,top+min(s[1] for s in boxes)-2))
        nr=min(width,max(x+w,left+max(s[0]+s[2] for s in boxes)+2))
        nb=min(height,max(y+h,top+max(s[1]+s[3] for s in boxes)+2))
        new=[float(nx),float(ny),float(nr-nx),float(nb-ny)]
        if new==crop['bbox'] or any(intersection(new,t.bbox)>1 for t in page.texts):continue
        # Do not follow large connected page backgrounds into unrelated content.
        if new[2]*new[3]>w*h*2.5:continue
        changes.append({'id':crop['id'],'before':crop['bbox'],'after':new,'reason':'source connected ink at crop edge'})
        crop['bbox']=new
    return Page.model_validate(data),changes
