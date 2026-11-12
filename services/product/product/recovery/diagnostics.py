"""Independent context and whole-search-scope checks for regional trials."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..materialize import digest, write
from ..object_extract import rectangle


def assess_trial(source, mask, raw, protected, bbox, reference, reference_report, metadata):
    if mask.dtype != bool or mask.shape != source.shape[:2] or (mask & protected).any():
        raise ValueError('invalid or protected trial mask')
    if raw.dtype != np.uint8 or raw.shape != source.shape:
        raise ValueError('invalid raw trial')
    written = source.copy()
    written[mask] = raw[mask]
    scope = rectangle(bbox, source.shape[1::-1]) & ~protected
    source_signal = np.abs(source.astype(float)-reference.astype(float)).max(2)
    error = np.abs(written.astype(float)-reference.astype(float)).max(2)
    residual = scope & (source_signal > 6) & (error > 6)
    missing = scope & (source_signal > 6) & ~mask
    kernel = np.ones((3, 3), np.uint8)
    seam = ((cv2.dilate(mask.astype('uint8'), kernel) > 0)
            & ~(cv2.erode(mask.astype('uint8'), kernel) > 0) & scope)
    seam_p95 = float(np.percentile(error[seam], 95)) if seam.any() else 0.0
    seam_max = float(error[seam].max()) if seam.any() else 0.0
    changed = (written != source).any(2)
    checks = {'reference_context': reference_report['validation_p95_rgb'] <= 3,
              'complete_visible_scope': not missing.any(),
              'no_visible_original_signal': not residual.any(),
              'background_colour': float(error[scope].mean()) <= 3 and float(np.percentile(error[scope], 95)) <= 5,
              'seam': seam_p95 <= 5 and seam_max <= 8,
              'protected_integrity': not (changed & protected).any(),
              'outside_mask_integrity': not (changed & ~mask).any(),
              'raw_writeback_identity': np.array_equal(written[mask], raw[mask])}
    report = {'schema': 'R12RegionalResidualReport/1', **metadata,
              'status': 'PRECHECK_PASS' if all(checks.values()) else 'REVIEW_REQUIRED',
              'checks': checks, 'mask_pixels': int(mask.sum()), 'changed_pixels': int(changed.sum()),
              'visible_source_pixels_outside_mask': int(missing.sum()),
              'remaining_source_signal_pixels': int(residual.sum()),
              'scope_error_mae_rgb': float(error[scope].mean()),
              'scope_error_p95_rgb': float(np.percentile(error[scope], 95)),
              'seam_p95_rgb': seam_p95, 'seam_max_rgb': seam_max,
              'protected_changed_pixels': int((changed & protected).sum()),
              'raw_protected_changed_pixels': int(((raw != source).any(2) & protected).sum()),
              'raw_mask_mean_rgb': raw[mask].mean(0).tolist(),
              'written_mask_mean_rgb': written[mask].mean(0).tolist(),
              'reference_mask_mean_rgb': reference[mask].mean(0).tolist(),
              'raw_mask_luma_bias': float((raw[mask].astype(float)-reference[mask].astype(float)).mean()),
              'writeback_raw_max_difference': int(np.abs(written[mask].astype(int)-raw[mask].astype(int)).max()),
              'reference_context': reference_report,
              'measurement_interpretation': {
                  'reference_dependent_quality': 'MEASURED' if checks['reference_context'] else 'UNKNOWN',
                  'remaining_source_signal_pixels': 'Difference from the estimated reference inside the search scope; not a count of missing mask pixels or proof of antialiasing.',
                  'visible_source_pixels_outside_mask': 'Source contrast outside the erase mask, relative to the estimated reference.',
                  'limitation': 'If reference_context fails, residual, colour, seam and visible-scope estimates cannot establish actual visual defects. Independent reliable background evidence is required; pixel ownership checks remain binding.'},
              'publication': 'REVIEW_REQUIRED', 'human_acceptance': False,
              'reference_limit': 'Source-context interpolation and source text geometry; original unoccluded pixels are unavailable'}
    return report, written, residual


def save_trial(directory, source, mask, raw, protected, bbox, reference, reference_report, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    report, written, residual = assess_trial(source, mask, raw, protected, bbox, reference, reference_report, metadata)
    difference = np.abs(written.astype(int)-source.astype(int)).max(2).astype('uint8')
    residual_image = np.zeros_like(source)
    residual_image[residual] = [255, 40, 20]
    x, y, w, h = map(round, bbox)
    images = {'source': source, 'mask': mask.astype('uint8')*255, 'raw_inpaint': raw,
              'written_back': written, 'difference_map': difference, 'residual_map': residual_image,
              'source_crop': source[y:y+h, x:x+w], 'raw_crop': raw[y:y+h, x:x+w],
              'written_crop': written[y:y+h, x:x+w]}
    for name, array in images.items():
        Image.fromarray(array).save(directory/(name+'.png'))
    write(directory/'residual_report.json', report)
    files = {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()}
    write(directory/'hashes.json', files)
    return report
