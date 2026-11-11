"""Continuous local colour plane fitted only to held-out, unprotected context."""
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
    if min(int((clean & (x < bx)).sum()), int((clean & (x >= bx+bw)).sum())) < max(30, bh*2):
        raise RecoveryBlocked('INSUFFICIENT_CONTEXT', 'both lateral sides need independent unprotected samples')
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
              'validation_p95_rgb': p95,
              'validation_mae_rgb': float(errors[validation].mean()),
              'coefficients': coefficients.tolist(),
              'fit_candidates': [{'degree': fit[1], 'validation_p95_rgb': fit[0]} for fit in fits],
              'reference': 'independent unprotected lateral source strips outside the text search scope'}
    result = source.copy()
    result[y0:y1, x0:x1] = np.clip(np.rint(prediction), 0, 255).astype('uint8')
    result.setflags(write=False)
    return result, report
