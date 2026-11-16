"""Continuous local colour plane fitted only to held-out, unprotected context."""
import cv2
import numpy as np

from ..object_extract import rectangle
from .background_adapter import RecoveryBlocked, _sha


def fit_gradient(source, bbox, protected, context=14):
    h, w = source.shape[:2]
    scope = rectangle(bbox, (w, h))
    window = rectangle(bbox, (w, h), pad=context)
    # The text may sit in a card with horizontal borders immediately above or
    # below it. Lateral strips stay on the same backing at every sampled row.
    _, by, _, bh = map(round, bbox)
    window[:max(0, by)] = False
    window[min(h, by+bh):] = False
    yy, xx = np.nonzero(window)
    x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
    y, x = np.mgrid[y0:y1, x0:x1]
    nx, ny = (x-x0)/max(1, x1-x0), (y-y0)/max(1, y1-y0)
    rgb = source[y0:y1, x0:x1].astype(float)
    clean = (~scope & window & ~protected)[y0:y1, x0:x1]
    bx, _, bw, _ = map(round, bbox)
    min_side = max(30, bh*2)
    if min(int((clean & (x < bx)).sum()), int((clean & (x >= bx+bw)).sum())) < min_side:
        raise RecoveryBlocked('INSUFFICIENT_CONTEXT', 'both lateral sides need independent unprotected samples')
    # Exclude discrete high-contrast foreign foreground intrusions (such as an
    # adjacent unmodeled icon or badge) only when both lateral sides still
    # retain substantial independent support on the shared continuous backing.
    excluded_foreground_pixels = 0
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
            if (int((candidate_clean & (x < bx)).sum()) >= min_side
                    and int((candidate_clean & (x >= bx+bw)).sum()) >= min_side
                    and int(candidate_clean.sum()) >= max(130, round(0.55 * int(clean.sum())))):
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
    # Outliers are reported on the complete independent validation set. A fit
    # cannot remove inconvenient validation pixels to approve itself.
    report = {'schema': 'LocalGradientBacking/1', 'source_pixels_sha256': _sha(source),
              'bbox': list(bbox), 'context': context, 'degree': degree,
              'training_pixels': int(training.sum()), 'inlier_pixels': int(keep.sum()),
              'validation_pixels': int(validation.sum()),
              'excluded_foreground_context_pixels': excluded_foreground_pixels,
              'validation_p95_rgb': p95,
              'validation_mae_rgb': float(errors[validation].mean()),
              'coefficients': coefficients.tolist(),
              'fit_candidates': [{'degree': fit[1], 'validation_p95_rgb': fit[0]} for fit in fits],
              'reference': 'independent unprotected lateral source strips outside the text search scope'}
    result = source.copy()
    result[y0:y1, x0:x1] = np.clip(np.rint(prediction), 0, 255).astype('uint8')
    result.setflags(write=False)
    return result, report
