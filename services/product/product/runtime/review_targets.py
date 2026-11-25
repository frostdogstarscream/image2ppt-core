"""Bind a visual finding to marks on the audited source, before offering an edit."""
from functools import lru_cache
import json
from math import isfinite
from pathlib import Path

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


@lru_cache(maxsize=64)
def _reading_report_index(path, mtime_ns, size):
    """Index small evidence signatures once per report revision, without opening the images."""
    report = json.loads(path.read_text('utf8'))
    source = report.get('source_sha256')
    by_id = {}
    for row in report.get('observations', []):
        oid, crop = row.get('object_id'), row.get('crop_sha256')
        if not oid or not crop:
            continue
        readings = by_id.setdefault(oid, set())
        reading = row.get('reading')
        if isinstance(reading, str) and reading.strip():
            readings.add(key([crop, row.get('context_sha256'), row.get('model'), reading.strip(), row.get('legible')]))
        for view_name, view in row.get('views', {}).items():
            answer = view.get('answer') or {}
            text = answer.get('text')
            if view.get('status') == 'ok' and view.get('source_sha256') == source and isinstance(text, str) and text.strip():
                readings.add(key([crop, view_name, view.get('model'), text.strip(), answer.get('legible')]))
    for item in report.get('items', []):
        if item.get('status') == 'observed' and item.get('target_id') in by_id:
            by_id.setdefault(item['id'], set()).update(by_id[item['target_id']])
    return source, by_id


def repair_inputs(current, page, ids):
    """Count only changed inputs for this exact target as a reason to retry."""
    k = str(page)
    plans = {r['id']:r for r in current.get('marked_plans', {}).get(k, [])}
    reviews = current.get('review_issues', {}).get(k, {})
    plate = current.get('plate_issues', {}).get(k, {})
    late = current.get('late_fields', {}).get(k, {})
    source = current.get('host_evidence', {}).get(k, {}).get('source_sha256')
    readings = {oid: set() for oid in ids}
    if source:
        for file in current.get('reading_reports', {}).get(k, []):
            path = Path(file)
            try:
                stat = path.stat()
                bound_source, index = _reading_report_index(path, stat.st_mtime_ns, stat.st_size)
            except (OSError, ValueError):
                # Archived optional observations cannot unlock a retry without their contents.
                continue
            if bound_source == source:
                for oid in readings:
                    readings[oid].update(index.get(oid, ()))
    fingerprints = {}
    for oid in ids:
        plan = plans.get(oid, {})
        route, role = plan.get('route'), plan.get('role')
        signature = {name:plan[name] for name in ('route','shape') if name in plan}
        signature['move_with'] = sorted(set(plan.get('move_with', [])) - {oid})
        # Roles change execution only when they affect picture lettering,
        # protected project names or an automatically owned parent card.
        if role in ('brand', 'screenshot', 'dashboard'):
            signature['artwork_text'] = True
            if route == 'asset' and role != 'brand':
                signature['intrinsic_ui'] = True
        elif role == 'project':
            signature['protected_project'] = True
        if route == 'surface' and plan.get('origin') == 'host':
            signature['group_owned_children'] = True
            signature['eligible_card_parent'] = role in ('card', 'role_box')
        field = late.get(oid, {})
        fingerprints[oid] = key({'plan':signature, 'review':sorted({r['finding_id'] for r in reviews.get(oid, [])}),
                                 'plate':plate.get(oid), 'source_reading':field.get('source_reading'),
                                 'observed_readings':sorted(readings[oid]),
                                 'source_views':[v.get('crop_sha256') for v in field.get('source_views', [])]})
    return fingerprints


def explain_empty_retry(requested, routes, graph, failures):
    """Explain a scoped no-op from the actual bound plan and pre-trial graph."""
    routes_by_id = {row['id']: row for row in routes}
    errors = {row['object_id']: row for row in failures if row.get('object_id')}
    known_text = {obj.id for obj in graph.text.texts}
    outcomes = []
    for object_id in dict.fromkeys(requested):
        route = routes_by_id.get(object_id, {})
        targets = sorted(set(route.get('target_ids', ())))
        selected = sorted(({object_id} | set(targets)) & graph.selected.keys())
        failed = [errors[i] for i in dict.fromkeys([object_id, *targets]) if i in errors]
        if failed:
            reason = 'EXECUTOR_REPORTED_FAILURE'
        elif object_id in graph.selected and graph.selected[object_id].get('kind') == 'text':
            reason = 'SELECTED_TEXT_NOT_REOPENED_BY_BATCH'
        elif route.get('route') == 'preserve':
            reason = 'PRESERVED_BY_PLAN'
        elif object_id not in graph.selected and targets and set(targets).issubset(graph.selected):
            reason = 'ROUTE_BOUND_ONLY_TO_SELECTED_OBJECTS'
        elif route.get('route') == 'text' and not targets and object_id not in known_text:
            reason = 'TEXT_MARK_HAS_NO_BOUND_SCENE_TEXT'
        elif object_id in known_text and graph.ownership.get(object_id) != 'RESOLVED':
            reason = 'TEXT_OWNERSHIP_UNRESOLVED'
        elif object_id in graph.selected:
            reason = 'SELECTED_OBJECT_NO_CHANGE_STAGED'
        else:
            reason = 'NO_CANDIDATE_RECORDED_CAUSE_UNKNOWN'
        outcomes.append({'object_id': object_id, 'route': route.get('route'), 'reason': reason,
                         'bound_target_ids': targets, 'existing_selected_ids': selected,
                         'executor_failures': [{'id': row['object_id'], 'failure': row.get('failure'),
                                                'reason': row.get('reason')} for row in failed]})
    return outcomes


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
