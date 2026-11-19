"""Continuous local colour plane fitted only to held-out, unprotected context."""
import cv2
import numpy as np

from ..object_extract import rectangle
from .background_adapter import RecoveryBlocked, _sha


def fit_gradient(source, bbox, protected, context=14, *, allow_vertical: bool = False):
    h, w = source.shape[:2]
    scope = rectangle(bbox, (w, h))
    window = rectangle(bbox, (w, h), pad=context)
    bx, by, bw, bh = map(round, bbox)
    if not allow_vertical:
        # Default lateral-only mode: stays on the same horizontal rows to avoid
        # crossing tight card borders above or below a line of text.
        window[:max(0, by)] = False
        window[min(h, by+bh):] = False
    yy, xx = np.nonzero(window)
    if len(xx) == 0:
        raise RecoveryBlocked('INSUFFICIENT_CONTEXT', 'empty context window')
    x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
    y, x = np.mgrid[y0:y1, x0:x1]
    nx, ny = (x-x0)/max(1, x1-x0), (y-y0)/max(1, y1-y0)
    rgb = source[y0:y1, x0:x1].astype(float)
    clean = (~scope & window & ~protected)[y0:y1, x0:x1]
    min_side = max(30, bh*2)

    if not allow_vertical:
        if min(int((clean & (x < bx)).sum()), int((clean & (x >= bx+bw)).sum())) < min_side:
            raise RecoveryBlocked('INSUFFICIENT_CONTEXT', 'both lateral sides need independent unprotected samples')
        left_mask = clean & (x < bx)
        right_mask = clean & (x >= bx+bw)
        med_l = np.median(rgb[left_mask], axis=0)
        med_r = np.median(rgb[right_mask], axis=0)
        nxl = float(np.median(nx[left_mask]))
        nxr = float(np.median(nx[right_mask]))
        slope_x = (med_r - med_l) / max(1e-3, nxr - nxl)
        seed_pred = med_l + (nx - nxl)[..., None] * slope_x
        seed_res = rgb - seed_pred
        near_h = clean & (np.abs(seed_res).max(2) <= 24.0)
        if int(near_h.sum()) >= 60:
            ny_c = ny - float(np.median(ny[near_h]))
            denom = float((ny_c[near_h] ** 2).sum())
            if denom > 1e-3:
                slope_y = (ny_c[near_h, None] * seed_res[near_h]).sum(0) / denom
                seed_pred = seed_pred + ny_c[..., None] * slope_y
    else:
        # Multi-sided context mode: when a heading sits next to a sequence number,
        # icon, or page margin so one lateral strip is blocked, combine consistent
        # unprotected strips from {left, right, top, bottom}.
        min_strip = max(24, min(bh, bw))
        strips = {
            'left': clean & (x < bx) & (y >= by) & (y < by + bh),
            'right': clean & (x >= bx + bw) & (y >= by) & (y < by + bh),
            'top': clean & (y < by),
            'bottom': clean & (y >= by + bh),
        }
        active = {k: m for k, m in strips.items() if int(m.sum()) >= min_strip}
        if len(active) < 2:
            raise RecoveryBlocked('INSUFFICIENT_CONTEXT', 'need at least two independent context strips')
        # Anchor on the largest lateral strip if available, else the largest strip overall.
        anchor_key = max(
            active.keys(),
            key=lambda k: (1 if k in ('left', 'right') else 0, int(active[k].sum())),
        )
        anchor_med = np.median(rgb[active[anchor_key]], axis=0)
        compatible_keys = [
            k for k, m in active.items()
            if float(np.abs(np.median(rgb[m], axis=0) - anchor_med).max()) <= 14.0
        ]
        if len(compatible_keys) < 2:
            raise RecoveryBlocked('INSUFFICIENT_CONTEXT', 'context strips disagree across horizontal/vertical boundary')
        compat_union = np.zeros_like(clean)
        for k in compatible_keys:
            compat_union |= active[k]
        clean = compat_union
        seed_pred = np.broadcast_to(anchor_med, rgb.shape).copy()

    excluded_foreground_pixels = 0
    coarse_design = np.stack([np.ones_like(nx), nx, ny], axis=2)
    coarse_keep = clean & (np.abs(rgb - seed_pred).max(2) <= 18.0)
    for _ in range(5):
        if coarse_keep.sum() < 100:
            break
        coarse_coeff = np.linalg.lstsq(coarse_design[coarse_keep], rgb[coarse_keep], rcond=None)[0]
        coarse_err = np.abs(coarse_design @ coarse_coeff - rgb).max(2)
        med = float(np.median(coarse_err[coarse_keep]))
        mad = float(np.median(np.abs(coarse_err[coarse_keep] - med)))
        updated = clean & (coarse_err <= max(4.0, med + 3.5 * mad))
        if updated.sum() < 100 or np.array_equal(updated, coarse_keep):
            break
        coarse_keep = updated
    if coarse_keep.sum() >= 100:
        coarse_coeff = np.linalg.lstsq(coarse_design[coarse_keep], rgb[coarse_keep], rcond=None)[0]
        coarse_err = np.abs(coarse_design @ coarse_coeff - rgb).max(2)
        foreign = clean & (coarse_err > 20.0)
        if foreign.any():
            foreign_dilated = (cv2.dilate(foreign.astype('uint8'), np.ones((3, 3), np.uint8)) > 0) & clean
            candidate_clean = clean & ~foreign_dilated
            lateral_ok = (
                allow_vertical
                or (int((candidate_clean & (x < bx)).sum()) >= min_side
                    and int((candidate_clean & (x >= bx+bw)).sum()) >= min_side)
            )
            if lateral_ok and int(candidate_clean.sum()) >= max(130, round(0.55 * int(clean.sum()))):
                excluded_foreground_pixels = int(clean.sum() - candidate_clean.sum())
                clean = candidate_clean
    validation = clean & ((x + 3*y) % 5 == 0)
    training = clean & ~validation
    if training.sum() < 100 or validation.sum() < 30:
        raise RecoveryBlocked('INSUFFICIENT_CONTEXT', 'gradient needs independent context samples')
    fits = []
    for degree in (1, 2, 3):
        design = np.stack([np.ones_like(nx), nx, *[ny**power for power in range(1, degree+1)]], axis=2)
        keep = training.copy()
        for _ in range(6):
            coefficients = np.linalg.lstsq(design[keep], rgb[keep], rcond=None)[0]
            error = np.abs(design @ coefficients - rgb).max(2)
            median = np.median(error[keep])
            limit = max(3.0, median + 3 * np.median(np.abs(error[keep] - median)))
            updated = training & (error <= limit)
            if updated.sum() < 100:
                raise RecoveryBlocked('INSUFFICIENT_CONTEXT', 'gradient inlier set is too small')
            if np.array_equal(updated, keep):
                break
            keep = updated
        coefficients = np.linalg.lstsq(design[keep], rgb[keep], rcond=None)[0]
        prediction = design @ coefficients
        errors = np.abs(prediction - rgb).max(2)
        fits.append((float(np.percentile(errors[validation], 95)), degree,
                     prediction, coefficients, errors, keep))
        if fits[-1][0] <= 3:
            break
    p95, degree, prediction, coefficients, errors, keep = min(fits, key=lambda fit: fit[0])
    report = {'schema': 'LocalGradientBacking/1', 'source_pixels_sha256': _sha(source),
              'bbox': list(bbox), 'context': context, 'degree': degree,
              'allow_vertical_context': allow_vertical,
              'training_pixels': int(training.sum()), 'inlier_pixels': int(keep.sum()),
              'validation_pixels': int(validation.sum()),
              'excluded_foreground_context_pixels': excluded_foreground_pixels,
              'validation_p95_rgb': p95,
              'validation_mae_rgb': float(errors[validation].mean()),
              'coefficients': coefficients.tolist(),
              'fit_candidates': [{'degree': fit[1], 'validation_p95_rgb': fit[0]} for fit in fits],
              'reference': ('multi-sided unprotected context strips' if allow_vertical
                            else 'independent unprotected lateral source strips outside the text search scope')}
    result = source.copy()
    result[y0:y1, x0:x1] = np.clip(np.rint(prediction), 0, 255).astype('uint8')
    result.setflags(write=False)
    return result, report
