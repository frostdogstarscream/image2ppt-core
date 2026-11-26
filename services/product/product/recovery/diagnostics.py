"""Independent context and whole-search-scope checks for regional trials."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..materialize import digest, write
from ..object_extract import rectangle
from .background_adapter import _high_frequency, _ring


def _nonplanar_backing_proof(source, written, raw, mask, protected, scope, seam, metadata):
    method = str((metadata or {}).get('method') or '')
    provenance = (metadata or {}).get('provenance') or {}
    origin = str(provenance.get('background_origin') or '')
    is_layerd = method.startswith('layerd-') or origin == 'layerd'
    is_inpaint = method == 'big-lama' or origin == 'big-lama'
    if not (is_layerd or is_inpaint):
        return None
    ring = _ring(mask, protected, radius=8)
    if int(ring.sum()) < 24:
        return {'passed': False, 'reason': 'NO_CLEAN_CONTEXT_RING'}
    if is_layerd:
        ring_delta = np.abs(source.astype(float) - raw.astype(float)).mean(2)[ring]
        ring_mae = float(ring_delta.mean())
        ring_p95 = float(np.percentile(ring_delta, 95))
        if ring_mae > 3.5 or ring_p95 > 9.0:
            return {'passed': False, 'reason': 'LAYERD_RING_MISMATCH',
                    'ring_mae': round(ring_mae, 4), 'ring_p95': round(ring_p95, 4)}
    k5 = np.ones((5, 5), np.uint8)
    outer2 = (cv2.dilate(mask.astype('uint8'), k5) > 0) & ~mask & ~protected
    eroded = cv2.erode(mask.astype('uint8'), k5) > 0
    inner2 = (mask & ~eroded) if (mask & ~eroded).any() else mask
    if not outer2.any():
        return {'passed': False, 'reason': 'NO_OUTER_BOUNDARY'}
    source_blur = cv2.GaussianBlur(source.astype(np.float32), (5, 5), 1.2)
    bg_rough_p95 = float(np.percentile(np.abs(source.astype(np.float32) - source_blur).max(2)[ring], 95))
    written_blur = cv2.GaussianBlur(written.astype(np.float32), (5, 5), 1.2)
    seam_jump_p95 = float(np.percentile(np.abs(written.astype(np.float32) - written_blur).max(2)[seam], 95)) if seam.any() else 0.0
    boundary_bias = float(np.abs(written[inner2].astype(float).mean(0) - source[outer2].astype(float).mean(0)).max())
    if is_layerd:
        outer_paste_p95 = float(np.percentile(np.abs(source[outer2].astype(float) - raw[outer2].astype(float)).max(1), 95))
        seam_ok = outer_paste_p95 <= 6.0 or (
            seam_jump_p95 <= max(8.5, bg_rough_p95 * 1.6)
            and boundary_bias <= max(12.0, bg_rough_p95 * 2.0)
        )
    else:
        seam_ok = (seam_jump_p95 <= max(8.5, bg_rough_p95 * 1.6)
                   and boundary_bias <= max(12.0, bg_rough_p95 * 2.0))
    old_hf = _high_frequency(source)
    new_hf = _high_frequency(written)
    raw_hf = _high_frequency(raw) if is_layerd else np.zeros_like(new_hf)
    old_ink = (mask & ((old_hf - raw_hf) > 8.0)) if is_layerd else (mask & (old_hf > 10.0))
    if not old_ink.any():
        return {'passed': False, 'reason': 'NO_SOURCE_STROKE_EVIDENCE'}
    eff_new_hf = np.maximum(0.0, new_hf - raw_hf) if is_layerd else new_hf
    residual_fraction = float(np.mean(eff_new_hf[old_ink] > 6.0))
    ablation_ok = (residual_fraction <= 0.02
                   and float(eff_new_hf[old_ink].mean()) <= max(2.5, 0.25 * float(old_hf[old_ink].mean())))
    adjacent_unmasked = (outer2 & scope & (eff_new_hf > max(14.0, bg_rough_p95 * 2.0)))
    scope_ok = int(adjacent_unmasked.sum()) == 0
    return {
        'passed': bool(seam_ok and ablation_ok and scope_ok),
        'mode': 'layerd_ring_ablation' if is_layerd else 'inpaint_seam_ablation',
        'seam_jump_p95': round(seam_jump_p95, 4),
        'boundary_bias_rgb': round(boundary_bias, 4),
        'bg_rough_p95': round(bg_rough_p95, 4),
        'residual_fraction': round(residual_fraction, 5),
        'adjacent_unmasked_stroke_pixels': int(adjacent_unmasked.sum()),
        'checks': {'seam': seam_ok, 'ablation': ablation_ok, 'scope': scope_ok},
    }


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
    if metadata.get('method') == 'interletter' and reference_report.get('method') == 'interletter':
        # The local fit independently measured variation in untouched gaps.
        # Keep original gap pixels; at stroke sites the complete fitted ground
        # must have been written and the seams must match nearby actual paper.
        noise = reference_report['gap_noise_rgb']
        missing = scope & (source_signal > noise) & ~mask
        residual = (scope & mask & (error > 3)) | missing
        checks.update(reference_context=reference_report['validation_p95_rgb'] <= 8,
                      complete_visible_scope=not missing.any(), no_visible_original_signal=not residual.any(),
                      background_colour=not (error[mask] > 3).any(), seam=seam_p95 <= noise and seam_max <= noise)
    quality_mode = 'MEASURED' if checks['reference_context'] else 'UNKNOWN'
    nonplanar = None
    if not all(checks.values()) and checks['protected_integrity'] and checks['outside_mask_integrity'] and checks['raw_writeback_identity']:
        nonplanar = _nonplanar_backing_proof(source, written, raw, mask, protected, scope, seam, metadata)
        if nonplanar and nonplanar['passed']:
            checks.update(reference_context=True, complete_visible_scope=True,
                          no_visible_original_signal=True, background_colour=True, seam=True)
            residual = np.zeros_like(mask)
            quality_mode = 'MEASURED_RING_ABLATION'
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
              **({'nonplanar_backing_proof': nonplanar} if nonplanar is not None else {}),
              'measurement_interpretation': {
                  'reference_dependent_quality': quality_mode,
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
