"""Bound SAM3 masks to planned regions; uncertain masks explicitly use rectangles."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import plan_first


def select(masks, scores, box, mode):
    """A positive exemplar may find identical objects elsewhere: reject them."""
    h, w = masks.shape[1:]; x, y, bw, bh = box
    roi = np.zeros((h, w), bool); roi[y:y+bh, x:x+bw] = True
    eligible = []
    for i, mask in enumerate(masks):
        area = int(mask.sum()); inside = int((mask & roi).sum())
        if area < 16 or inside/max(area, 1) < .85: continue
        ys, xs = np.where(mask)
        if not (x <= float(xs.mean()) < x+bw and y <= float(ys.mean()) < y+bh): continue
        rank = inside/max(bw*bh, 1) * float(scores[i])
        eligible.append((rank, i))
    indices = [i for _, i in sorted(eligible, reverse=True)]
    if mode == 'object': indices = indices[:1]
    mask = np.any(masks[indices], axis=0) if indices else np.zeros((h, w), bool)
    return mask, indices


def guide(rgb, lines, tables, payload, source_hash, directory):
    directory = Path(directory); result = plan_first.read(directory/'result.json')
    box_policy = result.get('box_policy', 'prompt')
    if box_policy not in ('prompt', 'crop-hard'):
        raise ValueError('unknown SAM3 box policy')
    _, _, report = plan_first.guide(rgb, lines, tables, payload, source_hash, hard_boxes=box_policy == 'crop-hard')
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    h, w = rgb.shape[:2]
    if result['source_sha256'] != source_hash or result['plan_payload_sha256'] != expected or result['image_size'] != [w, h]:
        raise ValueError('SAM3 source/plan/image binding mismatch')
    rows = {r['id']: r for r in result['regions']}; owned = np.zeros((h, w), bool)
    masks = {}; grad = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(grad, 40, 100) > 0
    text = np.zeros((h, w), np.uint8)
    for line in lines:
        x, y, bw, bh = [int(round(v)) for v in line['bbox']]
        text[max(0,y-2):min(h,y+bh+2), max(0,x-2):min(w,x+bw+2)] = 1
    for item in report['regions']:
        if item['status'] != 'guided': continue
        row = rows.get(item['id'])
        if row is None: raise ValueError('missing SAM3 region result')
        file = directory/result['precision']/row['mask_file']
        if file.parent != directory/result['precision'] or plan_first.sha(file) != row['sha256']:
            raise ValueError('SAM3 mask hash/path mismatch')
        with np.load(file, allow_pickle=False) as z:
            raw, scores = z['masks'], z['scores']
        if raw.shape[1:] != (h,w) or raw.ndim != 3 or len(scores) != len(raw) or not np.isfinite(scores).all():
            raise ValueError('invalid SAM3 mask dimensions or scores')
        mask, selected = select(raw.astype(bool), scores, item['bbox'], row['segmentation_mode'])
        x,y,bw,bh = item['bbox']; roi = np.zeros((h,w),bool); roi[y:y+bh,x:x+bw]=True
        if box_policy == 'crop-hard':
            if row.get('crop_bbox_px') != [x,y,x+bw,y+bh] or np.any(raw[:, ~roi]):
                raise ValueError('SAM3 hard box violation')
            mask &= roi
        evidence = edge & roi & ~text.astype(bool)
        near = cv2.dilate(mask.astype(np.uint8), np.ones((5,5),np.uint8)) > 0
        coverage = float((evidence & near).sum()/max(evidence.sum(),1))
        reason = 'no_local_candidate' if not selected else ('missing_graphic_edges' if coverage < .92 else None)
        if (mask & owned).any(): reason = 'overlapping_alpha_ownership'
        if reason:
            mask = roi; item['sam3_status'] = 'rectangle_fallback'; item['reasons'].append(reason)
        else:
            item['sam3_status'] = 'accepted_mask'
        mask &= ~owned; owned |= mask; masks[item['id']] = mask
        item.update(sam3_selected_indices=selected, sam3_candidates=len(raw), graphic_edge_coverage=round(coverage,4))
    work = rgb.copy()
    colour=np.median(np.concatenate((rgb[0],rgb[-1],rgb[:,0],rgb[:,-1])),axis=0).astype(np.uint8)
    work[owned]=colour
    report.update(segmentation_backend='sam3-prompted', sam3_result_sha256=plan_first.sha(directory/'result.json'),
                  box_policy=box_policy,
                  checkpoint_sha256=result['checkpoint_sha256'], precision=result['precision'])
    return work, owned, report, masks


def install(elements, report, masks, h, w, out):
    from .review import _hard_rect, _next_id
    from .segment import assign_levels
    for item in report['regions']:
        if item['status'] != 'guided': continue
        mask=masks[item['id']]; ys,xs=np.where(mask)
        if not len(xs): continue
        box=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)]
        crop=_hard_rect(box,h,w); crop.update(mask=mask,area=int(mask.sum()),alpha='mask',
            source='sam3_prompted' if item['sam3_status']=='accepted_mask' else 'sam3_rectangle_fallback',
            fallback=item['sam3_status']!='accepted_mask',plan_region_id=item['id'],id=_next_id(elements))
        kept=[]
        for old in elements:
            overlap=old['mask'] & mask
            if not overlap.any(): kept.append(old);continue
            if old['kind']=='native_shape':
                # A small mask hole must not leave an entire duplicate native icon
                # behind the raster. Keep only genuinely crossing/enclosing shapes.
                if overlap.sum()/max(old['mask'].sum(),1) < .8:
                    kept.append(old)
                else:
                    item.setdefault('replaced_native_ids',[]).append(old['id'])
                continue
            old['mask'] &= ~mask; old['area']=int(old['mask'].sum())
            if old['area']: old['alpha']='mask';kept.append(old)
        kept.append(crop);elements[:]=kept;item['element_id']=crop['id']
        Image.fromarray(mask.astype(np.uint8)*255).save(Path(out)/'masks'/('sam3-'+item['id']+'.png'))
    assign_levels(elements)
    erase=np.zeros((h,w),bool)
    for e in elements:erase |= e['mask']
    return elements,erase
