"""Bounded local mask repair for detected icon damage before object extraction."""
import numpy as np

from .loop_review import clip, intersect
from .loop_checks import graphic_losses


def repair_icons(source, cleaned, text_mask, records, *, inpaint, model=None, limit=8):
    H, W = source.shape[:2]
    icons = [clip(box, [W, H]) for r in records for box in r.get('icon_candidates', [])]
    findings = graphic_losses(source, cleaned, [r.get('ocr_bbox', r.get('ink_bbox')) for r in records
                                              if r.get('ocr_bbox') or r.get('ink_bbox')],
                             protected=icons, stage='inpainting')
    damaged = [box for box in icons if any(intersect(box, f['bbox']) for f in findings)]
    result = cleaned.copy(); actions = []; retained = []
    for box in damaged[:limit]:
        x, y, w, h = box; cx, cy, cw, ch = clip(box, [W, H], max(24, min(w, h)))
        mask = text_mask[cy:cy+ch, cx:cx+cw].copy()
        mask[y-cy:y-cy+h, x-cx:x-cx+w] = False
        candidate, method = inpaint(source[cy:cy+ch, cx:cx+cw], mask, model)
        local = result.copy(); local[y:y+h, x:x+w] = candidate[y-cy:y-cy+h, x-cx:x-cx+w]
        remaining = graphic_losses(source[y:y+h, x:x+w], local[y:y+h, x:x+w])
        if remaining:
            result[y:y+h, x:x+w] = source[y:y+h, x:x+w]
            retained.append(box); action = 'source_icon_retained'
        else:
            result[y:y+h, x:x+w] = local[y:y+h, x:x+w]
            action = 'local_mask_repair'
        actions.append({'bbox': box, 'context_bbox': [cx, cy, cw, ch], 'action': action, 'method': method})
    return result, {'actions': actions, 'unprocessed_regions': damaged[limit:], 'retained_source_regions': retained,
                    'basis': 'source edge check on detected icon regions; visual acceptance remains required'}
