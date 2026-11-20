"""Recover separately editable discs, light symbols and tracks in connected illustrations."""
import cv2
import numpy as np
from PIL import Image


def diagram_parts(rgb, rgba, anchors):
    """Return ordered local parts only for a row of evidenced circular badges."""
    h, w = rgb.shape[:2]
    yy, xx = np.indices((h, w))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    vivid = ((hsv[..., 1] > 150) & (rgba[..., 3] > 100)).astype('uint8')
    contours, _ = cv2.findContours(vivid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    discs = []
    used = set()
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        cx, cy = x+(width-1)/2, y+(height-1)/2
        radius = (width+height)/4
        if radius < 9 or abs(width-height) > radius*.18:
            continue
        if not .88 < cv2.contourArea(contour)/(np.pi*radius**2) < 1.10:
            continue
        matches = [i for i,(a,b,c,d) in enumerate(anchors) if a<=cx<=a+c and b<=cy<=b+d]
        if len(matches)!=1 or matches[0] in used:
            continue
        dist = np.hypot(xx-cx, yy-cy)
        fill = np.median(rgb[(dist < radius*.88) & vivid.astype(bool)], axis=0)
        if not np.isfinite(fill).all():
            continue
        channel = int(np.argmin(fill))
        light = np.clip((rgb[...,channel]-fill[channel])/(255-fill[channel]),0,1)
        symbol = (dist < radius*.74) & (light>.17)
        solid = symbol & (light>.78)
        if not .015 < solid.sum()/(np.pi*radius**2) < .42:
            continue
        if (np.ptp(rgb[solid].astype(float),axis=1)>38).mean()>.15:
            continue
        rim = (dist>radius*1.04) & (dist<radius*1.12)
        if not rim.any() or (rgb[rim].min(axis=1)>190).mean()<.7:
            continue
        glyph = np.full_like(rgba,255)
        glyph[...,3] = np.where(symbol,np.rint(255*light),0)
        discs.append((cx,cy,radius,fill,glyph))
        used.add(matches[0])
    discs.sort(key=lambda d:d[0])
    if len(discs)<2 or len(discs)!=len(anchors):
        return []
    if any(b[0]-a[0] < 1.5*(a[2]+b[2]) for a,b in zip(discs,discs[1:])):
        return []

    # Look for a narrow colored ridge between neighboring discs. Interpolate
    # beneath a disc only when visible samples support the connecting curve.
    xs, ys, radii = np.array([d[:3] for d in discs]).T
    col = np.arange(w)
    expected = np.interp(col,xs,ys)
    radius = np.median(radii)
    chroma = np.ptp(rgb.astype(float),axis=2)
    offset = max(2,round(radius*.12))
    ridge = chroma-(np.roll(chroma,offset,axis=0)+np.roll(chroma,-offset,axis=0))/2
    corridor = (abs(yy-expected)<radius*.48) & (rgba[...,3]>0)
    peak = np.argmax(np.where(corridor,ridge,-1000),axis=0)
    gap = (col>xs[0]) & (col<xs[-1])
    for cx,_,r,_,_ in discs:
        gap &= abs(col-cx)>1.35*r
    hits = gap & (ridge[peak,col]>10)
    if hits.sum()<max(8,.55*gap.sum()):
        return []
    coeff=np.polyfit(col[hits],peak[hits],min(2,len(discs)-1))
    track=np.polyval(coeff,col)
    if np.percentile(abs(track[hits]-peak[hits]),80)>radius*.12:
        return []
    # Smooth only hidden spans. Keep the narrow visible ridge in its measured position.
    track[hits]=peak[hits]
    stroke = np.median(rgb[peak[hits],col[hits]],axis=0)
    line = np.zeros_like(rgba)
    trail = (xx>=xs[0]) & (xx<=xs[-1])
    line[...,:3]=stroke
    line[...,3]=np.where(trail,np.clip(radius*.052+.5-abs(yy-track),0,1)*255,0)
    arrow=rgba.copy()
    band=trail & (abs(yy-track)<offset*1.4)
    line[band]=rgba[band]
    arrow[band]=0
    original=arrow.copy()
    # Remap exposed neighboring columns along the curve, using premultiplied
    # colors to prevent dark edges when sampling transparent pixels.
    premul=original.astype(float)
    premul[...,:3]*=premul[...,3:]/255
    for index,(cx,cy,r,_,_) in enumerate(discs):
        scope=np.hypot(xx-cx,yy-cy)<1.42*r
        left,right=round(cx-r*1.46),round(cx+r*1.46)
        if left<0 or right>=w:
            return []
        fraction=np.clip((col-left)/(right-left),0,1)
        mix=np.zeros_like(premul)
        for side,weight in ((left,1-fraction),(right,fraction)):
            if side<xs[0] or side>xs[-1]:
                continue
            sample=np.clip(np.rint(yy-track+track[side]).astype(int),0,h-1)
            mix+=premul[sample,side]*weight[None,:,None]
        mix[...,:3]*=255/np.maximum(mix[...,3:],1)
        arrow[scope]=mix[scope].clip(0,255).astype('uint8')
        line[scope,:3]=stroke
        line[...,3][scope]=np.where(trail,np.clip(radius*.052+.5-abs(yy-track),0,1)*255,0)[scope]
    def picture(name,pixels):
        img=Image.fromarray(pixels)
        bounds=img.getchannel('A').getbbox()
        if bounds is None:
            return None
        a,b,c,d=bounds
        return {'name':name,'bbox':[a,b,c-a,d-b],'image':img.crop(bounds)}
    parts=[picture('arrow',arrow),picture('connector',line)]
    for i,(cx,cy,r,color,glyph) in enumerate(discs,1):
        # Include half of the white stroke on each side of the ellipse path.
        outer=r*1.09
        parts.append({'name':f'disc-{i}','bbox':[cx-outer,cy-outer,2*outer,2*outer],
                      'fill':''.join(f'{round(v):02X}' for v in color),'stroke_pt':.75*.18*r})
        parts.append(picture(f'symbol-{i}',glyph))
    return parts if all(parts) else []
