"""Bind a visual finding to marks on the audited source, before offering an edit."""
from math import isfinite

from ..parallel_store import key
from .marked_routes import iou


def locate(defect, rows, size):
    result = dict(defect)
    by_number = {row['mark']:row for row in rows}
    numbers = list(dict.fromkeys(result.pop('mark_numbers', []) or []))
    space = result.pop('bbox_space', None)
    raw_box = result.pop('bbox', None)
    result.update(mark_ids=[], bbox=None, binding='unlocated')
    if numbers:
        if any(number not in by_number for number in numbers):
            return {**result, 'binding':'unknown_mark', 'reported_marks':numbers}
        matched = [by_number[number] for number in numbers]
        return {**result, 'binding':'numbered', 'mark_ids':[r['id'] for r in matched],
                'bbox':matched[0]['bbox'] if len(matched) == 1 else None}
    if not raw_box:
        return result
    if len(raw_box) != 4 or space not in ('source_px_xywh','normalized_xywh'):
        return {**result, 'binding':'unspecified_geometry'}
    x, y, w, h = raw_box
    if space == 'normalized_xywh':
        x, y, w, h = x*size[0], y*size[1], w*size[0], h*size[1]
    if (not all(isfinite(v) for v in (x,y,w,h)) or min(x,y)<0 or min(w,h)<1
            or x+w>size[0]+1e-6 or y+h>size[1]+1e-6):
        return {**result, 'binding':'invalid_geometry'}
    box = [round(x),round(y),round(x+w)-round(x),round(y+h)-round(y)]
    candidates = sorted(((iou(box,r['bbox']),r['id']) for r in rows), reverse=True)
    matches = [oid for score,oid in candidates if score >= .5]
    if len(matches) == 1:
        return {**result, 'binding':'matched_geometry', 'mark_ids':matches, 'bbox':box}
    if matches:
        return {**result, 'binding':'ambiguous_geometry', 'candidate_ids':matches[:10]}
    return {**result, 'binding':'new_region', 'bbox':box}


def repair_inputs(current, page, ids):
    """Count only changed inputs for this exact target as a reason to retry."""
    k = str(page)
    plans = {r['id']:r for r in current.get('marked_plans', {}).get(k, [])}
    reviews = current.get('review_issues', {}).get(k, {})
    plate = current.get('plate_issues', {}).get(k, {})
    late = current.get('late_fields', {}).get(k, {})
    fingerprints = {}
    for oid in ids:
        plan = plans.get(oid, {})
        signature = {name:plan[name] for name in ('route','role','shape') if name in plan}
        signature['move_with'] = sorted(plan.get('move_with', []))
        field = late.get(oid, {})
        fingerprints[oid] = key({'plan':signature, 'review':sorted({r['finding_id'] for r in reviews.get(oid, [])}),
                                 'plate':plate.get(oid), 'source_reading':field.get('source_reading'),
                                 'source_views':[v.get('crop_sha256') for v in field.get('source_views', [])]})
    return fingerprints


def expose_review(service, state, current, directory, audit):
    from .host_evidence import questions
    from ..materialize import write
    routing = []
    for result in audit.get('pages', []):
        verdict = result['vlm_audit']
        if verdict['status'] not in ('READY','CACHED'):
            continue
        page = result['page']
        k = str(page)
        work = directory/f'review-targets-{page}'
        work.mkdir(parents=True, exist_ok=True)
        issues = {}
        for defect in verdict.get('critical_defects', []):
            ids = defect.get('mark_ids', [])
            finding = {'page':page, 'finding_id':defect.get('finding_id'), 'binding':defect.get('binding','unlocated')}
            if (defect.get('binding') == 'new_region' and defect.get('bbox') and
                    defect['category'] in ('missing_text','text_corruption')):
                from .late_fields import register
                discovered = register(service,state,current,work/f"new-{defect['finding_id']}",
                                      page=page,findings=[{'bbox':defect['bbox'], 'text':''}])
                ids = [r['id'] for r in discovered if r.get('id') and r['status'] in ('ready','existing')]
                finding['discovery'] = discovered
                if ids:
                    finding['binding'] = 'source_verified_discovery'
            routing.append({**finding, 'object_ids':ids})
            for oid in ids:
                issues.setdefault(oid, []).append({field:defect.get(field) for field in
                                                  ('finding_id','category','edit_action','description')})
        current.setdefault('review_issues', {})[k] = issues
        current.setdefault('questions', {})[k] = questions(state,current,page,
            current.get('regional_failures', {}).get(k, []),work)
    write(directory/'review-routing.json', routing)
    return state.register(directory/'review-routing.json','review_routing')['id']
