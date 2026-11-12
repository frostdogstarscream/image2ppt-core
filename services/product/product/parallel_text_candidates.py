"""Read-only text hypotheses. Background evidence does not authorize bbox erasure."""
import numpy as np
import cv2
from .object_extract import rectangle
from .parallel_fusion import glyph_evidence


class TextUncertainty(ValueError):
    def __init__(self, code, message, evidence=None, artifacts=None):
        super().__init__(message)
        self.code, self.evidence = code, evidence or {}
        self.artifacts=artifacts or {}


def peer_protection(text, graph, source, blocked, *, baseline_directory=None, diagnostics=None, purpose='erase'):
    from .peer_evidence import protection
    if baseline_directory is not None:
        import json
        from pathlib import Path
        retained=Path(baseline_directory)/'runtime-ocr-protection.json'
        if retained.exists():
            evidence=json.loads(retained.read_text('utf8'))
            if evidence['source_sha256']!=graph.source_sha256:
                raise ValueError('retained OCR protection source hash mismatch')
            own=set(graph.text.members.get(text.id,[])) if graph.ownership.get(text.id)=='RESOLVED' else set()
            blocked=blocked.copy()
            for line in evidence['lines']:
                if line['id'] not in own:
                    blocked |= rectangle(line['bbox'],source.shape[1::-1],pad=1)
    return protection(text,graph,source,blocked,baseline_directory,purpose=purpose,diagnostics=diagnostics)


def perimeter_background(source, box, protected):
    x,y,w,h = box
    if x < 3 or y < 3 or x+w+3 > source.shape[1] or y+h+3 > source.shape[0]:
        raise TextUncertainty('BACKGROUND_NONUNIFORM', 'independent four-sided samples unavailable')
    regions = [(slice(y-3,y),slice(x,x+w)), (slice(y+h,y+h+3),slice(x,x+w)),
               (slice(y,y+h),slice(x-3,x)), (slice(y,y+h),slice(x+w,x+w+3))]
    samples=[]; medians=[]
    for rows,cols in regions:
        values=source[rows,cols][~protected[rows,cols]].astype(float)
        if len(values) < 6:
            raise TextUncertainty('PROTECTED_NEIGHBOR', 'insufficient unprotected perimeter samples')
        samples.append(values);medians.append(np.median(values,axis=0))
    color=np.median(np.concatenate(samples),axis=0)
    error=float(max(np.percentile(np.abs(v-color).max(1),95) for v in samples))
    if error>4 or np.abs(np.array(medians)-color).max()>3:
        raise TextUncertainty('BACKGROUND_NONUNIFORM', 'four sides do not prove a flat backing', {'perimeter_error':error})
    return np.broadcast_to(np.rint(color).astype('uint8'),(h,w,3)).copy(),error


def generate(text, graph, source, fonts, blocked, strategy):
    from .parallel_rebuild import text_candidate
    if strategy == 'T1-exact':
        return text_candidate(text,graph,source,fonts,blocked)
    from .materialize import font_gate
    from types import SimpleNamespace
    styles={s.id:s for s in graph.text.styles}
    font_gate(SimpleNamespace(styles=[styles[i] for i in {text.style,*[s.style for s in text.spans]}],texts=[text]),fonts)
    protected=peer_protection(text,graph,source,blocked)
    # Font-fit boxes may crop original strokes. Expand the evidence window,
    # never the native layout, and require clean closure on all four sides.
    x,y,w,h=map(round,text.bbox)
    margin=8
    box=[max(0,x-margin),max(0,y-margin),min(source.shape[1],x+w+margin)-max(0,x-margin),
         min(source.shape[0],y+h+margin)-max(0,y-margin)]
    x,y,w,h=box
    backing,error=perimeter_background(source,box,protected)
    if strategy == 'T3-joint-backing':
        # Until all containing foreground owners can be verified together this
        # strategy must keep an explicit receipt, not silently approximate a card.
        raise TextUncertainty('REGION_CLOSURE_UNPROVEN', 'joint backing requires all region owners and independent backing proof')
    colors={styles[text.style].color,*[styles[s.style].color for s in text.spans]}
    ink=np.zeros((h,w),bool);unknown=np.ones((h,w),bool)
    for color in colors:
        certain,uncertain=glyph_evidence(source,box,color,backing)
        ink|=certain;unknown&=uncertain
    # Screenshot ringing can overshoot a planned foreground color. Only an
    # observed pixel completely enclosed by eight already-proven glyph pixels
    # may use an extrapolated soft-alpha fit. This never grows a glyph boundary
    # or fills a missing source stroke, and preserves the original residual bound.
    enclosed=cv2.filter2D(ink.astype('uint8'),-1,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT)==8
    recovered=np.zeros_like(ink)
    crop=source[y:y+h,x:x+w].astype(float);bg=backing.astype(float)
    for color in colors:
        fg=np.array([int(color[i:i+2],16) for i in (0,2,4)],float)
        direction=fg-bg;den=(direction*direction).sum(2)
        alpha=((crop-bg)*direction).sum(2)/np.maximum(den,1)
        residual=np.abs(bg+alpha[...,None]*direction-crop).max(2)
        recovered |= unknown & enclosed & (alpha>1) & (alpha<=1.25) & (residual<=32) & (den>100)
    ink|=recovered;unknown&=~recovered
    details={'unknown_pixels':int(unknown.sum()),'perimeter_error':error}
    if unknown.any() or ink.sum()<2:
        raise TextUncertainty('FOREGROUND_COLOR_MISMATCH', 'unexplained pixels retained',details,
                              {'uncertain-mask':unknown,'candidate-mask':ink,'proposed-backing':backing})
    # Reject cropped strokes; don't make a clean interior while leaving outside ink.
    if ink[0].any() or ink[-1].any() or ink[:,0].any() or ink[:,-1].any():
        raise TextUncertainty('OWNER_ESCAPES_SCOPE','glyph touches candidate boundary',details)
    full=np.zeros(source.shape[:2],bool);full[y:y+h,x:x+w]=ink
    if (full&protected).any():
        raise TextUncertainty('PROTECTED_NEIGHBOR','glyph intersects protected peer',details)
    patch=source[y:y+h,x:x+w].copy();patch[ink]=backing[ink]
    payload=text.model_dump()
    # Reserve the same bounded localization slack for content replacement while
    # preserving the left/top anchor, font and source wording. This changes no
    # erase mask; peer rectangles veto even an invisible expanded text frame.
    tx,ty,tw,th=text.bbox
    native=[tx,ty,min(tw+margin*2,source.shape[1]-tx),th]
    if (rectangle(native,(source.shape[1],source.shape[0])) & protected).any():
        raise TextUncertainty('PROTECTED_NEIGHBOR','replacement frame would intersect peer')
    payload['bbox']=native
    return {'id':text.id,'kind':'text','bbox':box,'mask':full,'patch':patch,
            'payload':payload,'background_error':error,
            'method':'soft-alpha+independent-four-sided-flat-background/1',
            'background_evidence':{'method':'four-sided-samples/1','error':error,'color':backing[0,0].tolist(),
                                   'enclosed_overshoot_pixels':int(recovered.sum())}}
