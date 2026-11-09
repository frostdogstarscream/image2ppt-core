"""Deterministic ink measurements and enlarged comparison crops; advisory only."""
import math
import numpy as np
import cv2
from PIL import Image,ImageDraw


def compare_text(source,preview,page,output):
    before=Image.open(source).convert('RGB');after=Image.open(preview).convert('RGB')
    if before.size!=after.size:raise ValueError('preview dimensions mismatch')
    styles={s.id:s for s in page.styles};a=np.asarray(before,dtype=np.int16);b=np.asarray(after,dtype=np.int16)
    metrics=[];areas=[]
    for t in page.texts:
        style=styles[t.style]
        if t.spans or '\n' in t.text:continue
        color=np.array([int(style.color[i:i+2],16) for i in (0,2,4)])
        if sum(color)>600:continue
        x,y,w,h=t.bbox;box=(max(0,int(x)-8),max(0,int(y)-8),min(before.width,math.ceil(x+w)+12),min(before.height,math.ceil(y+h)+12))
        def ink(array):
            data=array[box[1]:box[3],box[0]:box[2]]
            mask=np.sqrt(((data-color)**2).sum(axis=2))<65
            # Thin decoration rules must not inflate text width (one-pixel source
            # rules and two-pixel exported rules otherwise produce false ratios).
            count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
            for label in range(1,count):
                _,_,cw,ch,_=stats[label]
                if (cw>4*ch and ch<max(4,style.size_px*.15)) or (cw<3 and ch>4*cw):
                    mask[labels==label]=False
            yy,xx=np.where(mask)
            if len(xx)<20:return None
            # Trim isolated antialiasing pixels by requiring at least two ink pixels per row/column.
            xs=np.where(mask.sum(axis=0)>=2)[0];ys=np.where(mask.sum(axis=1)>=2)[0]
            if not len(xs) or not len(ys):return None
            return [int(xs[-1]-xs[0]+1),int(ys[-1]-ys[0]+1)]
        src=ink(a);dst=ink(b)
        if not src or not dst:continue
        wr=dst[0]/src[0];hr=dst[1]/src[1]
        metric={'id':t.id,'style':t.style,'source_ink':src,'render_ink':dst,
                'width_ratio':round(wr,3),'height_ratio':round(hr,3),
                'suspected_size_mismatch':wr<.84 or wr>1.18 or hr<.78 or hr>1.25}
        metrics.append(metric)
        areas.append((abs(math.log(wr))+abs(math.log(hr)),t.id,box))
    # Preserve a group overview and focus on the largest measurable mismatches.
    areas=sorted(areas,reverse=True)[:8]
    tiles=[]
    for _,id,box in areas:
        left=before.crop(box);right=after.crop(box)
        scale=min(2,700/max(1,left.width))
        size=(max(1,round(left.width*scale)),max(1,round(left.height*scale)))
        left=left.resize(size);right=right.resize(size)
        row=Image.new('RGB',(size[0]*2+12,size[1]+22),'white')
        ImageDraw.Draw(row).text((2,2),id+'   SOURCE | RENDER',fill='black')
        row.paste(left,(0,22));row.paste(right,(size[0]+12,22));tiles.append(row)
    if not tiles:return metrics,None
    board=Image.new('RGB',(max(t.width for t in tiles),sum(t.height for t in tiles)),(220,220,220));y=0
    for tile in tiles:board.paste(tile,(0,y));y+=tile.height
    board.save(output)
    return metrics,output


def compare_graphics(source,preview,page,output):
    before=Image.open(source).convert('RGB');after=Image.open(preview).convert('RGB')
    targets=[*page.crops,*[s for s in page.shapes if s.bbox[2]*s.bbox[3]<before.width*before.height*.02]]
    rows=[]
    for target in targets[:24]:
        x,y,w,h=target.bbox
        box=(max(0,int(x)-25),max(0,int(y)-25),min(before.width,int(x+w)+25),min(before.height,int(y+h)+25))
        a=before.crop(box);b=after.crop(box)
        scale=min(2,360/max(1,a.width),180/max(1,a.height));size=(max(1,round(a.width*scale)),max(1,round(a.height*scale)))
        row=Image.new('RGB',(740,202),'white');ImageDraw.Draw(row).text((2,2),target.id+' SOURCE | RENDER',fill='black')
        row.paste(a.resize(size),(0,22));row.paste(b.resize(size),(375,22));rows.append(row)
    if not rows:return None
    # Two columns keep details readable without one excessively tall image.
    board=Image.new('RGB',(1480,202*((len(rows)+1)//2)),(220,220,220))
    for i,row in enumerate(rows):board.paste(row,((i%2)*740,(i//2)*202))
    board.save(output);return output
