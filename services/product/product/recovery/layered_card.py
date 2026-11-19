"""Recover the face of a detected frame after its independent photos have been removed."""
import cv2
import numpy as np
from PIL import Image
from ..contracts import intersection
from ..materialize import digest
from ..parallel_store import key
from .native_shapes import _fit_linear_gradient, estimate_canvas_field


def geometry_marks(source, text_rows):
    from .container_stage import _discover_candidates
    rgb = np.asarray(Image.open(source).convert('RGB'))
    canvas, _, scenery, flat = estimate_canvas_field(rgb)
    sha = digest(source)
    rows = []
    for kind, box in _discover_candidates(rgb, canvas, scenery, is_flat_canvas=flat):
        if kind != 'box':
            continue
        children = [row for row in text_rows if row.get('ocr_line_ids') and
                    intersection(box, row['bbox']) >= .98 * row['bbox'][2] * row['bbox'][3]]
        if len(children) < 2:
            continue
        rows.append({'id':'m-'+key([sha,'card-frame',box])[:16], 'label':'card_frame',
                     'bbox':box, 'sources':[{'model':'pixel-frame','label':'card_frame'}]})
    return rows


def contained_peers(box, peers):
    return all(intersection(box,r['bbox']) >= r['bbox'][2]*r['bbox'][3] and
               (r['kind']=='text' or r['kind']=='asset' and r.get('move_safety')=='MOVABLE')
               and r.get('owner_evidence') for r in peers)


def frame_layers(patch, parent, box, peers, baseline, ignore_boxes):
    """Fit exposed card paper; retain unexplained interior art and leave photo holes transparent."""
    x, y, w, h = box
    sample = np.ones((h,w),bool)
    owned = np.zeros_like(sample)
    photos = np.zeros_like(sample)
    for r in peers:
        ref = r['owner_evidence']
        f = baseline/ref['file']
        if digest(f) != ref['sha256']:
            raise ValueError('card child source evidence changed')
        with np.load(f,allow_pickle=False) as proof:
            mask = proof['mask'][y:y+h,x:x+w].astype(bool)
        owned |= mask
        if r['kind']=='asset':
            ax,ay,aw,ah = map(round,r['bbox'])
            occupied = mask[ay-y:ay-y+ah,ax-x:ax-x+aw]
            if occupied.shape!=(ah,aw) or not occupied.all():
                return None
            photos |= mask
    for b in [r['bbox'] for r in peers]+list(ignore_boxes):
        ax,ay,aw,ah=map(round,b)
        sample[max(0,ay-y):max(0,min(h,ay-y+ah)),max(0,ax-x):max(0,min(w,ax-x+aw))]=False
    margin=max(2,round(min(w,h)*.018))
    sample[:margin]=sample[-margin:]=False
    sample[:,:margin]=sample[:,-margin:]=False
    smooth=cv2.GaussianBlur(patch,(5,5),0)
    sample &= np.max(np.abs(smooth.astype(float)-patch),axis=2)<3
    if sample.sum()<.035*w*h:
        return None
    colors=patch[sample].astype(int)//16
    histogram=np.bincount(colors @ np.array([256,16,1]),minlength=4096)
    mode=int(histogram.argmax())
    center=np.median(patch[sample][colors @ np.array([256,16,1])==mode],axis=0)
    sample &= np.abs(patch.astype(float)-center).max(2)<22
    # Evidence for a backing must extend across the card rather than a photo or label.
    sy,sx=np.where(sample)
    if len(sx)<.035*w*h or np.ptp(sx)<.75*w or np.ptp(sy)<.55*h:
        return None
    # Trim compression noise and text halos around the dominant paper colour.
    distance=np.abs(patch.astype(float)-center).max(2)
    sample &= distance <= max(5.0,float(np.percentile(distance[sample],75)))
    fit=_fit_linear_gradient(patch,sample,14)
    if fit is None:
        return None
    field,_,gradient=fit
    fill=np.rint(np.median(field[sample],axis=0)).astype('uint8')
    delta=np.abs(patch.astype(float)-field).max(2)
    # The exterior must meet a quiet page backing. Reject a picture or text crossing the edge.
    rim=np.zeros_like(sample)
    rim[[0,-1],:]=True
    rim[:,[0,-1]]=True
    exterior_error=np.abs(patch.astype(float)-parent).max(2)
    bad=(exterior_error>35)&rim
    if bad.sum()>max(3,round(.015*rim.sum())):
        return None
    ink=(delta>12)&~photos
    # Keep complete islands such as photographs (including their light interiors).
    contours,_=cv2.findContours(ink.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        ax,ay,aw,ah=cv2.boundingRect(contour)
        if min(aw,ah)>=12 and aw*ah<.55*w*h and cv2.contourArea(contour)>.64*aw*ah:
            local=np.zeros_like(sample,dtype='uint8')
            cv2.drawContours(local,[contour],-1,1,-1)
            ink |= local.astype(bool)
    ink &= ~photos
    rgba=np.dstack([patch,np.uint8(ink)*255])
    return fill, gradient, field, owned, ink, rgba
