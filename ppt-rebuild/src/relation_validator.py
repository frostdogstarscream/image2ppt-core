"""Source-bound small-object and edit-region gates for the r7 reconstruction path.

The reviewed regions describe this source deck, not a general image detector.
Masks are frozen from source pixels before inspecting any candidate. A missing
source element, changed source hash, or unavailable expected clean plate is not
accepted as an empty/successful check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REGIONS = {
    '02': [('s02-sidebar-label', 'label', [59, 198, 302, 258])],
    '08': [('s08-feedback-arrow-1', 'arrow', [588, 735, 628, 766]),
           ('s08-feedback-arrow-2', 'arrow', [929, 735, 972, 766]),
           ('s08-feedback-arrow-3', 'arrow', [1266, 735, 1307, 766]),
           ('s08-summary-chevrons', 'decoration', [316, 827, 365, 862])],
}
LIMITS = {'contour_p95_px': 1.5, 'coverage_min': .90, 'precision_min': .85,
          'label_coverage_min': .85, 'color_delta_median_max': 12.,
          'contrast_regression_fraction_max': .10, 'tip_coverage_min': .90}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb(image):
    return np.asarray(Image.open(image).convert('RGB')) if isinstance(image, (str, Path)) else np.asarray(image)


def blue_mask(crop):
    a = crop.astype(np.int16)
    return (a[:, :, 2] - a[:, :, 0] > 45) & (a[:, :, 0] < 130) & (a[:, :, 1] < 190)


def silhouette(mask, minimum_area=8):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros(mask.shape, np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) >= minimum_area:
            cv2.drawContours(result, [contour], -1, 1, -1)
    return result.astype(bool)


def boundary(mask):
    return mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def contour_distance(a, b):
    a, b = boundary(a), boundary(b)
    if not a.any() or not b.any():
        return None
    to_b = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    to_a = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return float(max(np.percentile(to_b[a], 95), np.percentile(to_a[b], 95)))


def contrast(a, b):
    def luminance(color):
        values = np.asarray(color, float) / 255
        values = np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)
        return float(values @ np.array([.2126, .7152, .0722]))
    x, y = luminance(a), luminance(b)
    return (max(x, y) + .05) / (min(x, y) + .05)


def prepare_spec(page, source, output):
    """Freeze local source masks and reviewed regions; return a JSON-serializable spec."""
    source = Path(source).resolve()
    image = rgb(source)
    page = str(page)[:2]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if image.shape[:2] != (941, 1672):
        raise ValueError('Reviewed r7 regions require the original 1672x941 source canvas')
    elements = []
    for identifier, kind, bbox in REGIONS.get(page, []):
        x, y, right, bottom = bbox
        crop = image[y:bottom, x:right]
        ink = blue_mask(crop)
        if ink.sum() < 25:
            raise ValueError('Missing source evidence for ' + identifier)
        shape = silhouette(ink)
        masks = {}
        for name, mask in [('source_ink', ink), ('silhouette', shape)]:
            path = output / (identifier + '-' + name + '.png')
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            masks[name] = {'path': str(path), 'sha256': digest(path), 'origin_px': [x, y],
                           'size_px': [right-x, bottom-y], 'semantic': 'binary_source_evidence'}
        elements.append({'id': identifier, 'kind': kind, 'bbox_xyxy': bbox, 'masks': masks,
                         'source_color': np.median(crop[ink], axis=0).tolist(),
                         'direction': 'right' if kind == 'arrow' else None})
    result = {'schema': 'RelationSpec/1', 'page': page, 'source_sha256': digest(source),
              'canvas_px': [1672, 941], 'limits': dict(LIMITS), 'elements': elements,
              'coverage_scope': 'reviewed_key_elements_only'}
    (output / 'relations.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def checked_mask(ref):
    if digest(ref['path']) != ref['sha256']:
        raise ValueError('Frozen relation mask changed')
    mask = np.asarray(Image.open(ref['path']).convert('L')) > 0
    if list(mask.shape[::-1]) != ref['size_px']:
        raise ValueError('Frozen mask dimensions changed')
    return mask


def check_relations(source, rendered, spec):
    if digest(source) != spec['source_sha256']:
        raise ValueError('Relation source hash mismatch')
    original, candidate = rgb(source), rgb(rendered)
    if candidate.shape != original.shape or list(original.shape[1::-1]) != spec['canvas_px']:
        raise ValueError('Relation render must have the exact original canvas size')
    limits = spec['limits']
    rows = []
    kernel = np.ones((3, 3), np.uint8)
    for element in spec['elements']:
        x, y, right, bottom = element['bbox_xyxy']
        crop, baseline = candidate[y:bottom, x:right], original[y:bottom, x:right]
        truth = checked_mask(element['masks']['source_ink'])
        outline = checked_mask(element['masks']['silhouette'])
        actual = blue_mask(crop)
        actual_shape = silhouette(actual)
        expanded = cv2.dilate(actual.astype(np.uint8), kernel).astype(bool)
        nearby = cv2.dilate(truth.astype(np.uint8), kernel).astype(bool)
        coverage = float((truth & expanded).sum() / truth.sum())
        precision = float((actual & nearby).sum() / max(1, actual.sum()))
        distance = contour_distance(outline, actual_shape)
        sample = truth & actual
        delta = float(np.median(np.abs(crop.astype(float)-baseline)[sample])) if sample.any() else None
        reasons = []
        minimum = limits['label_coverage_min'] if element['kind'] == 'label' else limits['coverage_min']
        if coverage < minimum: reasons.append('source_contribution_missing')
        if element['kind'] != 'label' and precision < limits['precision_min']:
            reasons.append('unexpected_shape_contribution')
        if distance is None or distance > limits['contour_p95_px']:
            reasons.append('source_contour_changed')
        if delta is None or delta > limits['color_delta_median_max']:
            reasons.append('source_color_changed')
        row = {'id': element['id'], 'coverage': coverage, 'precision': precision,
               'contour_p95_px': distance, 'color_delta_median': delta}
        if element['kind'] == 'arrow':
            yy, xx = np.nonzero(truth)
            tip = truth.copy()
            tip[:, :int(xx.min()+.65*(xx.max()-xx.min()+1))] = False
            recall = float((tip & expanded).sum()/max(1,tip.sum()))
            row['tip_coverage'] = recall
            if recall < limits['tip_coverage_min']: reasons.append('arrowhead_missing_or_wrong_direction')
        if element['kind'] == 'label':
            white_source = (baseline.min(axis=2) > 220) & outline
            # Text positions/styles may differ; compare bright foregrounds within
            # the source label envelope without defining the envelope from candidate.
            white_candidate = (crop.min(axis=2) > 220) & outline
            source_ratio = contrast(np.median(baseline[white_source], axis=0), np.median(baseline[truth], axis=0)) if white_source.any() else None
            ratio = contrast(np.median(crop[white_candidate], axis=0), np.median(crop[truth], axis=0)) if white_candidate.any() else None
            row.update(source_contrast=source_ratio, candidate_contrast=ratio)
            if source_ratio is None or ratio is None or ratio < source_ratio*(1-limits['contrast_regression_fraction_max']):
                reasons.append('label_contrast_regressed')
            if float(np.median(crop[truth])) >= float(np.median(crop[white_candidate])) if white_candidate.any() else True:
                reasons.append('label_polarity_changed')
        row.update(status='PASS' if not reasons else 'FAIL', reasons=reasons)
        rows.append(row)
    return {'schema': 'RelationQA/1', 'status': 'PASS' if rows and all(r['status']=='PASS' for r in rows) else ('NOT_APPLICABLE' if not rows else 'FAIL'),
            'source_sha256': spec['source_sha256'], 'objects': rows,
            'coverage_scope': spec['coverage_scope']}


def check_authorized_changes(before, after, repair_mask, protection_mask=None):
    before, after = rgb(before), rgb(after)
    allowed = np.asarray(repair_mask, bool)
    protected = np.zeros(allowed.shape, bool) if protection_mask is None else np.asarray(protection_mask, bool)
    if before.shape != after.shape or before.shape[:2] != allowed.shape or protected.shape != allowed.shape:
        raise ValueError('Change gate dimensions mismatch')
    changed = np.any(before != after, axis=2)
    outside = int((changed & ~allowed).sum())
    protected_changed = int((changed & protected).sum())
    conflict = int((allowed & protected).sum())
    return {'status': 'PASS' if not (outside or protected_changed or conflict) else 'FAIL',
            'outside_changed_pixels': outside, 'protected_changed_pixels': protected_changed,
            'repair_protection_overlap_pixels': conflict}


def check_edit_regions(before, edited, expected, *, allowed_mask, old_mask, new_mask,
                       protection_mask, expected_plate_verified=False, mae_limit=2., p95_limit=5.):
    """Independent expected clean/edit render required; absent evidence stays pending.

    Caller must supply verified expectations, not merely the same reconstruction
    patch being assessed. This checks old/new sites including their full edge rings.
    """
    if not expected_plate_verified:
        return {'status': 'REVIEW_REQUIRED', 'reason': 'independent_expected_plate_not_verified'}
    a, b, e = rgb(before), rgb(edited), rgb(expected)
    if a.shape != b.shape or a.shape != e.shape:
        raise ValueError('Edit gate image dimensions mismatch')
    allowed, old, new, protected = [np.asarray(x, bool) for x in (allowed_mask, old_mask, new_mask, protection_mask)]
    if any(x.shape != a.shape[:2] for x in (allowed,old,new,protected)):
        raise ValueError('Edit gate mask dimensions mismatch')
    changed = np.any(a != b, axis=2)
    rows = {}
    for name, mask in [('old_site', old), ('new_site',new), ('edge_ring', boundary(old)|boundary(new))]:
        diff = np.abs(b.astype(float)-e)[mask]
        if not diff.size:
            rows[name] = {'status': 'REVIEW_REQUIRED', 'reason':'empty_inspection_mask'}
            continue
        mae, p95 = float(diff.mean()), float(np.percentile(diff,95))
        rows[name] = {'status':'PASS' if mae<=mae_limit and p95<=p95_limit else 'FAIL','mae':mae,'p95':p95}
    outside = int((changed & ~allowed).sum())
    protected_changed = int((changed & protected).sum())
    return {'status':'PASS' if not outside and not protected_changed and all(x['status']=='PASS' for x in rows.values()) else 'FAIL',
            'regions':rows,'outside_changed_pixels':outside,'protected_changed_pixels':protected_changed,
            'independent_expected_plate_verified':True}
