"""Recover a colored, numbered section tab and its pale continuation from the plate."""
import cv2
import numpy as np


def title_bands(plate, records, text_rows):
    rgb = plate[..., :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    vivid = ((hsv[..., 1] > 130) & (hsv[..., 2] > 110)).astype('uint8')
    contours, _ = cv2.findContours(vivid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = plate.copy()
    bands = []
    selected_text = {r['id'] for r in records if r['kind'] == 'text'}
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if not 2.2 < w/h < 8 or h < 10 or cv2.contourArea(contour)<.88*w*h:
            continue
        inside = [t for t in text_rows if x<=t.bbox[0] and y<=t.bbox[1]
                  and t.bbox[0]+t.bbox[2]<=x+w and t.bbox[1]+t.bbox[3]<=y+h]
        if len(inside)<2 or any(t.id not in selected_text for t in inside):
            continue
        inside.sort(key=lambda t:t.bbox[0])
        if not inside[0].text.strip().isdigit():
            continue
        pad=max(2,round(h*.08))
        if y<pad or y+h+2*pad>=rgb.shape[0]:
            continue
        end=min(rgb.shape[1],x+4*w)
        samples=rgb[y+h+pad:y+h+2*pad,x:end].astype(float)
        floor=np.median(samples,axis=0)
        middle=np.median(rgb[y+h//4:y+3*h//4,x:end],axis=0)
        contrast=np.max(np.abs(middle-floor),axis=1)
        stops=np.flatnonzero(contrast[w:]<5)
        if not len(stops):
            continue
        length=w+int(stops[0])
        if length-w<h*.5 or np.max(np.ptp(middle[w+pad:length-pad],axis=0),initial=0)>22:
            continue
        # Do not replace a picture, an existing container or an unresolved text row.
        def overlaps(b):
            a,b_y,c,d=b
            return a<x+length and a+c>x and b_y<y+h and b_y+d>y
        if any(overlaps(r['bbox']) for r in records if r['kind']!='text') or any(
                overlaps(t.bbox) and t not in inside for t in text_rows):
            continue
        # Locate the measured color break between the number and title.
        start=max(1,round(inside[0].bbox[0]+inside[0].bbox[2]-x))
        stop=min(w-1,round(inside[1].bbox[0]-x))
        if stop<=start:
            continue
        jumps=np.linalg.norm(np.diff(middle[:w],axis=0),axis=1)
        split=start+int(np.argmax(jumps[start-1:stop]))
        if jumps[split-1]<8:
            continue
        outline=np.zeros((h,w),'uint8')
        cv2.drawContours(outline,[contour-[x,y]],-1,1,cv2.FILLED)
        edges=w-1-np.argmax(outline[:,::-1],axis=1)
        inset=edges[h//2]-np.mean(edges[[max(1,h//12),h-1-max(1,h//12)]])
        if inset>h*.4:
            continue
        colors=[np.median(middle[a:b],axis=0) for a,b in
                ((pad,split-pad),(split+pad,w-h//3),(w+pad,length-pad))]
        if not all(np.isfinite(c).all() for c in colors):
            continue
        # Include antialias pixels outside the saturated component. Only write
        # colors explained by the three fills or their edge blends with the floor.
        left=max(0,x-1); right=min(rgb.shape[1],x+length+1)
        top=y-1; bottom=y+h+1
        tile=rgb[top:bottom,left:right].astype(float)
        under=np.median(rgb[y+h+pad:y+h+2*pad,left:right],axis=0)
        if np.percentile(np.ptp(samples[:,:length],axis=0),95)>14:
            continue
        error=np.full(tile.shape[:2],np.inf)
        for first,second in [(under,c) for c in colors]+[(colors[0],colors[1]),(colors[1],colors[2])]:
            vector=second-first
            alpha=np.clip(np.sum((tile-first)*vector,axis=2)/np.maximum(1,np.sum(vector*vector,axis=-1)),0,1)
            error=np.minimum(error,abs(tile-(first+alpha[...,None]*vector)).max(axis=2))
        unknown=(error>22).astype('uint8')
        if np.any(unknown):
            _,_,stats,_=cv2.connectedComponentsWithStats(unknown)
            if stats[1:,cv2.CC_STAT_AREA].max(initial=0)>max(4,h*.15):
                continue
        mask=(unknown==0) & (abs(tile-under).max(axis=2)>1)
        clean[top:bottom,left:right,:3][mask]=np.broadcast_to(under,tile.shape)[mask]
        bands.append({'id':'band-'+inside[0].id,'bbox':[x,y,length,h],'tab_width':w,
                      'split':split,'tab_geometry':'hexagon' if inset>h*.06 else 'rect',
                      'colors':[''.join(f'{round(v):02X}' for v in c) for c in colors]})
    return clean, sorted(bands,key=lambda b:b['bbox'][:2])
