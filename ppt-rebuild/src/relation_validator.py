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
from qa_status import aggregate_checks
from relation_catalog import REGIONS, catalog_page


LIMITS = {'contour_p95_px': 1.5, 'coverage_min': .90, 'precision_min': .85,
          'label_coverage_min': .85, 'color_delta_median_max': 12.,
          'contrast_regression_fraction_max': .10, 'tip_coverage_min': .90,
          'text_coverage_min': .65, 'text_pixel_ratio_min': .50,
          'text_shift_tolerance_px': 3}

KINDS = {'label': 'LABEL_CONTRAST', 'arrow': 'ARROW_CONNECTIVITY',
         'decoration': 'DECORATION_PRESENCE', 'LABEL_CONTRAST': 'LABEL_CONTRAST',
         'ARROW_CONNECTIVITY': 'ARROW_CONNECTIVITY', 'ICON_PRESENCE': 'ICON_PRESENCE',
         'SURFACE_GEOMETRY': 'SURFACE_GEOMETRY', 'TEXT_OVER_SURFACE': 'TEXT_OVER_SURFACE',
         'CLEAN_PLATE': 'CLEAN_PLATE', 'DECORATION_PRESENCE': 'DECORATION_PRESENCE'}


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
    return mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


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


def _box(value, canvas):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('Relation requires a four-coordinate bbox_xyxy')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) or int(x) != x for x in value):
        raise ValueError('Relation bbox coordinates must be finite integers')
    x, y, right, bottom = map(int, value)
    if not (0 <= x < right <= canvas[0] and 0 <= y < bottom <= canvas[1]):
        raise ValueError('Relation bbox lies outside source canvas')
    return x, y, right, bottom


def segment(crop, element):
    method = element.get('segmentation', 'blue')
    if method == 'blue':
        return blue_mask(crop)
    if method == 'color_distance':
        color = np.asarray(element.get('foreground_rgb'), float)
        if color.shape != (3,) or not np.isfinite(color).all():
            raise ValueError('color_distance requires foreground_rgb')
        return np.max(np.abs(crop.astype(float)-color), axis=2) <= element.get('color_tolerance', 30)
    if method == 'surface_boundary':
        edge = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), 30, 80) > 0
        band = np.ones(edge.shape, bool)
        if min(edge.shape) > 12:
            band[6:-6, 6:-6] = False
        return edge & band
    raise ValueError('Unsupported source segmentation: ' + str(method))


def prepare_spec(page, source, output, *, config=None, ir=None):
    """Freeze local source masks and reviewed regions; return a JSON-serializable spec."""
    source = Path(source).resolve()
    image = rgb(source)
    page = str(page)
    if len(page) >= 2 and page[:2].isdigit():
        page = page[:2]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    canvas = list(image.shape[1::-1])
    declarations, catalog = catalog_page(page, digest(source), config, ir)
    elements = []
    for declared in declarations:
        element = dict(declared)
        identifier, kind = element['id'], KINDS.get(element.get('kind'))
        # Unsupported or absent evidence remains visible in the frozen inventory.
        if kind is None or element.get('evidence_missing') or not element.get('applicable', True):
            element['masks'] = {}
            elements.append(element)
            continue
        x, y, right, bottom = _box(element.get('bbox_xyxy'), canvas)
        crop = image[y:bottom, x:right]
        if kind in ('CLEAN_PLATE', 'TEXT_OVER_SURFACE'):
            element.update(masks={}, evidence_missing='independent_relation_evidence_required')
            elements.append(element)
            continue
        if kind == 'SURFACE_GEOMETRY':
            element.update(segmentation='surface_boundary', geometry_scope='visible_outer_boundary_only')
        ink = segment(crop, element)
        if ink.sum() < element.get('minimum_source_pixels', 25):
            element.update(masks={}, evidence_missing='source_contribution_not_resolved')
            elements.append(element)
            continue
        shape = ink if kind == 'SURFACE_GEOMETRY' else silhouette(ink)
        masks = {}
        frozen_masks = [('source_ink', ink), ('silhouette', shape)]
        if kind == 'LABEL_CONTRAST':
            if not element.get('text_target_id') or not element.get('text_bbox_xyxy'):
                element['text_evidence_missing'] = 'label_text_owner_or_region_missing'
            else:
                tx, ty, tr, tb = _box(element['text_bbox_xyxy'], canvas)
                if not (x <= tx < tr <= right and y <= ty < tb <= bottom):
                    raise ValueError('Label text evidence must lie within label region')
                text_region = np.zeros(ink.shape, bool)
                text_region[ty-y:tb-y, tx-x:tr-x] = True
                frozen_masks.append(('text_contribution', (crop.min(axis=2) > 220) & shape & text_region))
        for name, mask in frozen_masks:
            path = output / (identifier + '-' + name + '.png')
            if path.parent != output:
                raise ValueError('Relation ID must not contain path separators')
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            masks[name] = {'path': str(path), 'sha256': digest(path), 'origin_px': [x, y],
                           'size_px': [right-x, bottom-y], 'semantic': 'binary_source_evidence'}
        element.update(masks=masks, source_color=np.median(crop[ink], axis=0).tolist())
        elements.append(element)
    result = {'schema': 'RelationSpec/1', 'page': page, 'source_sha256': digest(source),
              'canvas_px': canvas, 'limits': dict(LIMITS), 'elements': elements,
              'expected_relation_ids': [row['id'] for row in declarations], **catalog}
    (output / 'relations.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def checked_mask(ref):
    if digest(ref['path']) != ref['sha256']:
        raise ValueError('Frozen relation mask changed')
    mask = np.asarray(Image.open(ref['path']).convert('L')) > 0
    if list(mask.shape[::-1]) != ref['size_px']:
        raise ValueError('Frozen mask dimensions changed')
    return mask


def relation_coverage(spec, rows):
    expected = spec.get('expected_relation_ids', [r['id'] for r in spec['elements']])
    counts = {status: sum(r['status'] == status for r in rows)
              for status in ('PASS', 'FAIL', 'REVIEW_REQUIRED', 'NOT_APPLICABLE')}
    executed = sum(bool(r.get('executed', True)) for r in rows if r['status'] != 'NOT_APPLICABLE')
    applicable = len(expected)-counts['NOT_APPLICABLE']
    evaluated = sum(bool(r.get('executed', True)) for r in rows if r['status'] != 'NOT_APPLICABLE')
    return {'schema': 'RelationCoverage/1', 'page': spec['page'], 'expected': len(expected),
            'defined': len(spec['elements']), 'executed': executed,
            'passed': counts['PASS'], 'failed': counts['FAIL'],
            'review_required': counts['REVIEW_REQUIRED'], 'not_applicable': counts['NOT_APPLICABLE'],
            'execution_coverage': executed/applicable if applicable else None,
            'executed_pass_rate': counts['PASS']/evaluated if evaluated else None,
            'coverage_scope': spec['coverage_scope']}


def check_relations(source, rendered, spec):
    if digest(source) != spec['source_sha256']:
        raise ValueError('Relation source hash mismatch')
    original, candidate = rgb(source), rgb(rendered)
    if candidate.shape != original.shape or list(original.shape[1::-1]) != spec['canvas_px']:
        raise ValueError('Relation render must have the exact original canvas size')
    limits = {**LIMITS, **spec.get('limits', {})}
    rows = []
    kernel = np.ones((3, 3), np.uint8)
    expected_ids = spec.get('expected_relation_ids', [e['id'] for e in spec['elements']])
    actual_ids = [e['id'] for e in spec['elements']]
    if len(set(actual_ids)) != len(actual_ids) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Duplicate frozen relation ID')
    if set(actual_ids)-set(expected_ids):
        raise ValueError('Undeclared relation in frozen specification')
    for missing in set(expected_ids)-set(actual_ids):
        rows.append({'id': missing, 'status': 'FAIL', 'reasons': ['required_relation_missing'], 'executed': False})
    for element in spec['elements']:
        kind = KINDS.get(element.get('kind'))
        row = {'id': element['id'], 'kind': element.get('kind'),
               'repairable': element.get('repairable', element.get('kind') in ('label', 'arrow', 'decoration')),
               'executed': False}
        if not element.get('applicable', True):
            reason = element.get('not_applicable_reason')
            row.update(status='NOT_APPLICABLE' if reason else 'FAIL',
                       reason=reason or 'not_applicable_reason_missing', reasons=[reason or 'not_applicable_reason_missing'])
            rows.append(row)
            continue
        if kind is None or element.get('evidence_missing'):
            row.update(status='FAIL' if element.get('evidence_missing') == 'expected_scene_object_missing' else 'REVIEW_REQUIRED',
                       reasons=[element.get('evidence_missing') or 'unsupported_relation_kind'])
            rows.append(row)
            continue
        if kind in ('CLEAN_PLATE', 'TEXT_OVER_SURFACE'):
            row.update(status='REVIEW_REQUIRED', reasons=['independent_relation_evidence_required'])
            rows.append(row)
            continue
        x, y, right, bottom = _box(element['bbox_xyxy'], spec['canvas_px'])
        crop, baseline = candidate[y:bottom, x:right], original[y:bottom, x:right]
        truth = checked_mask(element['masks']['source_ink'])
        outline = checked_mask(element['masks']['silhouette'])
        for name in ('source_ink', 'silhouette'):
            ref = element['masks'][name]
            if ref['origin_px'] != [x, y] or ref['size_px'] != [right-x, bottom-y]:
                raise ValueError('Relation mask origin/extent mismatch')
        if not truth.any():
            row.update(status='REVIEW_REQUIRED', reasons=['empty_source_inspection_mask'])
            rows.append(row)
            continue
        actual = segment(crop, element)
        actual_shape = actual if kind == 'SURFACE_GEOMETRY' else silhouette(actual)
        expanded = cv2.dilate(actual.astype(np.uint8), kernel).astype(bool)
        nearby = cv2.dilate(truth.astype(np.uint8), kernel).astype(bool)
        coverage = float((truth & expanded).sum() / truth.sum())
        precision = float((actual & nearby).sum() / max(1, actual.sum()))
        distance = contour_distance(outline, actual_shape)
        sample = truth & actual
        delta = float(np.median(np.abs(crop.astype(float)-baseline)[sample])) if sample.any() else None
        reasons = []
        minimum = limits['label_coverage_min'] if kind == 'LABEL_CONTRAST' else limits['coverage_min']
        if coverage < minimum: reasons.append('source_contribution_missing')
        if kind not in ('LABEL_CONTRAST', 'ICON_PRESENCE', 'SURFACE_GEOMETRY') and precision < limits['precision_min']:
            reasons.append('unexpected_shape_contribution')
        if kind not in ('ICON_PRESENCE', 'SURFACE_GEOMETRY') and (distance is None or distance > limits['contour_p95_px']):
            reasons.append('source_contour_changed')
        if kind not in ('SURFACE_GEOMETRY', 'ICON_PRESENCE') and (delta is None or delta > limits['color_delta_median_max']):
            reasons.append('source_color_changed')
        row.update(coverage=coverage, precision=precision, contour_p95_px=distance,
                   color_delta_median=delta, executed=True)
        if kind == 'SURFACE_GEOMETRY':
            row['geometry_scope'] = element.get('geometry_scope', 'visible_outer_boundary_only')
            # Canny samples establish visible boundary presence. They do not
            # independently certify rounded geometry, fill or child occlusion.
            row['precision_review'] = {'status': 'REVIEW_REQUIRED',
                'reason': 'surface_contour_fill_and_occlusion_evidence_incomplete'}
        if kind == 'ICON_PRESENCE':
            row['inspection_scope'] = 'source_contribution_presence_and_position_not_color_fidelity'
        if kind == 'ARROW_CONNECTIVITY':
            yy, xx = np.nonzero(truth)
            tip = truth.copy()
            direction = element.get('direction')
            if direction == 'right': tip[:, :int(xx.min()+.65*(xx.max()-xx.min()+1))] = False
            elif direction == 'left': tip[:, int(xx.min()+.35*(xx.max()-xx.min()+1)):] = False
            elif direction == 'down': tip[:int(yy.min()+.65*(yy.max()-yy.min()+1)), :] = False
            elif direction == 'up': tip[int(yy.min()+.35*(yy.max()-yy.min()+1)):, :] = False
            else:
                row.update(status='FAIL' if reasons else 'REVIEW_REQUIRED', reasons=reasons+['arrow_direction_not_declared'])
                rows.append(row)
                continue
            recall = float((tip & expanded).sum()/max(1,tip.sum()))
            row.update(tip_coverage=recall, direction=direction,
                       connectivity_scope='source_arrow_contour_and_tip')
            if recall < limits['tip_coverage_min']: reasons.append('arrowhead_missing_or_wrong_direction')
        if kind == 'LABEL_CONTRAST':
            text_ref = element['masks'].get('text_contribution')
            if text_ref is None or not element.get('text_target_id'):
                row.update(status='FAIL' if reasons else 'REVIEW_REQUIRED', reasons=reasons+['label_text_contribution_not_bound'])
                rows.append(row)
                continue
            if text_ref['origin_px'] != [x, y] or text_ref['size_px'] != [right-x, bottom-y]:
                raise ValueError('Label text mask origin/extent mismatch')
            white_source = checked_mask(text_ref)
            tx, ty, tr, tb = _box(element['text_bbox_xyxy'], spec['canvas_px'])
            text_region = np.zeros(outline.shape, bool)
            text_region[max(0,ty-y):min(bottom-y,tb-y),max(0,tx-x):min(right-x,tr-x)] = True
            white_candidate = (crop.min(axis=2) > 220) & outline & text_region
            radius = int(limits['text_shift_tolerance_px'])
            text_nearby = cv2.dilate(white_candidate.astype(np.uint8), np.ones((2*radius+1,2*radius+1), np.uint8)).astype(bool)
            glyph_coverage = float((white_source & text_nearby).sum()/max(1, white_source.sum()))
            pixel_ratio = float(white_candidate.sum()/max(1, white_source.sum()))
            row.update(text_target_id=element['text_target_id'], source_text_pixels=int(white_source.sum()),
                       candidate_text_pixels=int(white_candidate.sum()), text_coverage=glyph_coverage,
                       text_pixel_ratio=pixel_ratio)
            if not white_source.any():
                row.update(status='FAIL' if reasons else 'REVIEW_REQUIRED', reasons=reasons+['source_label_text_evidence_empty'])
                rows.append(row)
                continue
            if glyph_coverage < limits['text_coverage_min'] or pixel_ratio < limits['text_pixel_ratio_min']:
                reasons.append('label_text_contribution_missing')
            source_ratio = contrast(np.median(baseline[white_source], axis=0), np.median(baseline[truth], axis=0)) if white_source.any() else None
            ratio = contrast(np.median(crop[white_candidate], axis=0), np.median(crop[truth], axis=0)) if white_candidate.any() else None
            row.update(source_contrast=source_ratio, candidate_contrast=ratio)
            if source_ratio is None or ratio is None or ratio < source_ratio*(1-limits['contrast_regression_fraction_max']):
                reasons.append('label_contrast_regressed')
            if float(np.median(crop[truth])) >= float(np.median(crop[white_candidate])) if white_candidate.any() else True:
                reasons.append('label_polarity_changed')
        row.update(status='FAIL' if reasons else ('REVIEW_REQUIRED' if kind == 'SURFACE_GEOMETRY' else 'PASS'),
                   reasons=reasons or (['surface_contour_fill_and_occlusion_evidence_incomplete'] if kind == 'SURFACE_GEOMETRY' else []))
        rows.append(row)
    if rows:
        status = aggregate_checks({r['id']: r for r in rows})['status']
    else:
        status = 'NOT_APPLICABLE' if spec.get('not_applicable_reason') else 'REVIEW_REQUIRED'
    coverage = relation_coverage(spec, rows)
    coverage['status'] = status
    return {'schema': 'RelationQA/1', 'status': status,
            'source_sha256': spec['source_sha256'], 'objects': rows,
            'coverage_scope': spec['coverage_scope'], 'coverage': coverage,
            'reason': spec.get('not_applicable_reason') if status == 'NOT_APPLICABLE' else
                      ('relation_scope_not_defined' if not rows else None)}


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
    a, b = rgb(before), rgb(edited)
    e = rgb(expected) if expected is not None else None
    if a.shape != b.shape or (e is not None and a.shape != e.shape):
        raise ValueError('Edit gate image dimensions mismatch')
    allowed, old, new, protected = [np.asarray(x, bool) for x in (allowed_mask, old_mask, new_mask, protection_mask)]
    if any(x.shape != a.shape[:2] for x in (allowed,old,new,protected)):
        raise ValueError('Edit gate mask dimensions mismatch')
    changed = np.any(a != b, axis=2)
    rows = {}
    for name, mask in [('old_site', old), ('new_site',new), ('edge_ring', boundary(old)|boundary(new))]:
        if not expected_plate_verified or e is None:
            rows[name] = {'status': 'REVIEW_REQUIRED', 'reason': 'independent_expected_plate_not_verified'}
            continue
        diff = np.abs(b.astype(float)-e)[mask]
        if not diff.size:
            rows[name] = {'status': 'REVIEW_REQUIRED', 'reason':'empty_inspection_mask'}
            continue
        mae, p95 = float(diff.mean()), float(np.percentile(diff,95))
        rows[name] = {'status':'PASS' if mae<=mae_limit and p95<=p95_limit else 'FAIL','mae':mae,'p95':p95}
    outside = int((changed & ~allowed).sum())
    protected_changed = int((changed & protected).sum())
    checks = {**rows, 'authorized_write_boundary': {'status': 'FAIL' if outside else 'PASS'},
              'protected_pixels': {'status': 'FAIL' if protected_changed else 'PASS'}}
    return {'status': aggregate_checks(checks)['status'],
            'regions':rows,'outside_changed_pixels':outside,'protected_changed_pixels':protected_changed,
            'independent_expected_plate_verified':bool(expected_plate_verified and e is not None)}
