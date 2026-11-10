"""Explicit OCR geometry adapter; MoMA supplies roles, not guessed ink position."""
import numpy as np
from PIL import ImageFont
from .parallel_store import key


def anchor(graph,ocr,source,fonts,store,*,fitted=False):
    graph=graph.model_copy(deep=True);lines={r['id']:r for r in ocr.get('lines',[])}
    styles={s.id:s for s in graph.text.styles};faces={(f['family'],f['bold']):f for f in fonts['fonts']};changes=[]
    users={i:sorted([t for t in graph.text.texts if i in graph.text.members.get(t.id,[])],key=lambda t:t.bbox[0]) for i in lines}
    def font(style,text):
        family=style.ea if any(ord(c)>255 for c in text) else style.latin
        f=faces[(family,style.bold)];return ImageFont.truetype(f['file'],max(1,round(style.size_px)),index=f['index'])
    for t in graph.text.texts:
        if graph.ownership[t.id]!='RESOLVED':continue
        ids=graph.text.members.get(t.id,[])
        if not ids or any(i not in lines for i in ids):continue
        boxes=[lines[i]['bbox'] for i in ids];x=min(b[0] for b in boxes);y=min(b[1] for b in boxes)
        right=max(b[0]+b[2] for b in boxes);bottom=max(b[1]+b[3] for b in boxes)
        if len(ids)==1 and len(users[ids[0]])>1:
            ordered=users[ids[0]];index=ordered.index(t)
            try:weights=[font(styles[a.style],a.text).getlength(a.text) for a in ordered]
            except (KeyError,OSError):continue
            total=max(1,sum(weights));start=sum(weights[:index])/total;end=sum(weights[:index+1])/total
            width=right-x;left_cut=x+round(width*start);right_cut=x+round(width*end)
            # Locate a real whitespace cut near the predicted run boundary.
            crop=source[y:bottom,x:right].astype(float)
            border=np.concatenate([crop[0],crop[-1]],axis=0);bg=np.median(border,axis=0)
            ink=np.abs(crop-bg).max(2)>35;empty=~ink.any(0)
            for which,target in [('left',left_cut-x),('right',right_cut-x)]:
                if (which=='left' and index==0) or (which=='right' and index==len(ordered)-1):continue
                candidates=np.flatnonzero(empty & (np.abs(np.arange(width)-target)<=max(12,width*.08)))
                if len(candidates):
                    cut=int(candidates[np.argmin(np.abs(candidates-target))])+x
                    if which=='left':left_cut=cut
                    else:right_cut=cut
            sub=ink[:,max(0,left_cut-x):min(width,right_cut-x)]
            if not sub.any():continue
            yy,xx=np.nonzero(sub);x=left_cut+int(xx.min());right=left_cut+int(xx.max())+1;y+=int(yy.min());bottom=y+int(yy.max()-yy.min()+1)
        before=list(t.bbox);box=[float(x),float(y),float(right-x),float(bottom-y)]
        if fitted:
            try:
                s=styles[t.style];face=font(s,t.text);first=t.text.split('\n')[0];glyph=face.getbbox(first)
                line_count=len(t.text.split('\n'));tops=sorted(set(lines[i]['bbox'][1] for i in ids))
                spacing=float(np.median(np.diff(tops))) if len(tops)>1 and line_count>1 else (t.line_spacing_px or s.size_px*1.2)
                if line_count>1:t.line_spacing_px=spacing
                widths=[]
                for content in t.text.split('\n'):
                    widths.append(sum(font(s,c).getlength(c) for c in content))
                fw=max(max(widths),float(right-x));fh=max(s.size_px,(line_count-1)*spacing+s.size_px,float(bottom-y+max(0,glyph[1])))
                left=max(0.,x-glyph[0]-(fw-max(widths))*(.5 if t.align=='center' else 1 if t.align=='right' else 0))
                top=max(0.,y-glyph[1]);fw=min(fw,graph.image_size[0]-left);fh=min(fh,graph.image_size[1]-top)
                box=[round(left,2),round(top,2),round(fw,2),round(fh,2)]
            except (KeyError,OSError):continue
        t.bbox=box
        changes.append({'id':t.id,'before':before,'after':box,'ocr_members':ids,'font_bearing_adjusted':fitted})
    if changes:graph.references.append(store.evidence({'schema':'OCRTextGeometryAdapter/1','source_sha256':graph.source_sha256,'changes':changes}))
    return graph
