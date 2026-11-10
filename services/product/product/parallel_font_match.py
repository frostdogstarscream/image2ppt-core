"""Rank installed faces jointly against independent source glyph samples."""
import threading
import numpy as np
from PIL import Image,ImageDraw,ImageFont

_LOCAL=threading.local()
def face(path,index,size):
    cache=getattr(_LOCAL,'fonts',None)
    if cache is None:cache={};_LOCAL.fonts=cache
    k=(path,index,max(1,int(size)))
    if k not in cache:
        if len(cache)>=512:cache.pop(next(iter(cache)))
        cache[k]=ImageFont.truetype(path,k[2],index=index)
    return cache[k]

def match(rows,fonts,current):
    if not rows:return None
    cjk=any(any(ord(c)>255 for c in r['text']) for r in rows)
    candidates=[f for f in fonts['fonts'] if f['cjk'] or not cjk]
    ranked=[]
    for record in candidates:
        estimates=[]
        try:
            base=face(record['file'],record['index'],32)
            for r in rows:
                box=base.getbbox(r['text']);estimates.append(32*r['ink_width']/max(1,box[2]-box[0]))
            centre=int(round(float(np.median(estimates))))
            for size in range(max(4,centre-1),min(200,centre+1)+1):
                font=face(record['file'],record['index'],size);scores=[]
                for r in rows:
                    gb=font.getbbox(r['text']);gw=max(1,gb[2]-gb[0]);gh=max(1,gb[3]-gb[1])
                    glyph=Image.new('L',(gw,gh));ImageDraw.Draw(glyph).text((-gb[0],-gb[1]),r['text'],font=font,fill=255)
                    alpha=np.array(glyph)/255.;h,w=r['source'].shape[:2];best=float('inf')
                    for dx in (-1,0,1):
                        for dy in (-1,0,1):
                            left,top=r['ink_left']+dx,r['ink_top']+dy;canvas=np.zeros((h,w),float)
                            sx,sy=max(0,-left),max(0,-top);tx,ty=max(0,left),max(0,top)
                            cw,ch=min(w-tx,gw-sx),min(h-ty,gh-sy)
                            if min(cw,ch)<=0:continue
                            canvas[ty:ty+ch,tx:tx+cw]=alpha[sy:sy+ch,sx:sx+cw]
                            render=r['foreground']*canvas[...,None]+r['background']*(1-canvas[...,None])
                            clipping=max(0,left+gw-w)+max(0,top+gh-h)
                            score=float(np.abs(render-r['source']).mean())+clipping*2
                            best=min(best,score)
                    scores.append(best)
                ranked.append({'family':record['family'],'bold':record['bold'],'size_px':float(size),
                               'source_pixel_mae':float(np.mean(scores)),'cjk':cjk})
        except (ValueError,OSError):continue
    ranked.sort(key=lambda r:r['source_pixel_mae'])
    if not ranked:return None
    best=ranked[0];family=current.ea if cjk else current.latin
    prior=next((r for r in ranked if r['family']==family and r['bold']==current.bold),None)
    if prior and prior['source_pixel_mae']<=best['source_pixel_mae']+.35:best=prior
    return {**best,'representative_ids':[r['id'] for r in rows],'ranked_candidates':ranked[:6],
            'method':'installed-face-source-glyph-group/1','independent_final_render_required':True}
