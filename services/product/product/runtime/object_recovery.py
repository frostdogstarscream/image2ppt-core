"""Rebuild useful edits from this job on the current page using only source IDs and routes."""
from copy import copy, deepcopy
import json
from pathlib import Path

from ..materialize import digest, write
from ..parallel_store import BudgetExceeded, key
from ..pipeline import PipelineError
from .state import RuntimeConflict


def _read(path):
    return json.loads(Path(path).read_bytes())


def _representation(ir):
    result = {node['id']: node['type'] for node in ir['nodes'] if node['id'] != 'source-plate'}
    result.update((group['id'], sorted(group['member_ids'])) for group in ir.get('groups', []))
    return result


def _matches(item, shapes, selected):
    # r-001 etc. are allocated in staging order and can refer to a different photo.
    # Rebind such pictures by their source seeds before comparing their Office type.
    if item.get('crop_seeds'):
        return any(sorted(record.get('seed_ids', [])) == item['crop_seeds']
                   and shapes.get(oid) == item['representation'] for oid, record in selected.items())
    return (shapes.get(item['object_id']) == item['representation']
            and all(shapes.get(oid) == kind for oid,kind in item.get('member_types', {}).items()))


def remember_edits(state, current, candidate, ir, graph, checked):
    """An edit receipt supplies a retry lead. It never authorizes writes on another parent."""
    if candidate.get('kind') != 'batch' or not candidate.get('object_ids'):
        return
    office = checked.get('office', {})
    deck = checked.get('pptx_sha256')
    if (not deck or office.get('pptx_sha256') != deck
            or office.get('status') not in {'PASS_APP_AUTOMATED', 'FAIL_APP_AUTOMATED'}):
        return
    actions = {}
    for row in office.get('records', []):
        if row.get('page') == 1 and row.get('outside_pixels') == 0 and row.get('saved_reopened') is True:
            actions.setdefault(row['id'], set()).add(row['action'])
    verified = {oid for oid, actions_done in actions.items() if {'move-scale', 'delete'} <= actions_done}
    representations = _representation(_read(ir))
    planned = {p['id']: p for p in current.get('marked_plans', {}).get(str(candidate['page']), [])}
    source_ids = {item.id for item in graph.asset.objects} | {line.id for line in graph.text.texts}
    residuals = {item['id']:item for page in checked.get('residuals', []) for item in page}
    items = []
    for oid in sorted(set(candidate['object_ids'])):
        record = graph.selected.get(oid, {})
        shape = representations.get(oid)
        if not shape or not record.get('owner_evidence') or record.get('move_safety') == 'MOVE_UNSAFE':
            continue
        members_checked = isinstance(shape, list) and set(shape) <= verified
        if oid not in verified and not members_checked:
            continue
        if residuals.get(oid, {}).get('residual_check_status') == 'FAIL' or office.get('failed_object_id') == oid:
            continue
        if any(r.get('id') == oid and (not r.get('content_matches') or not r.get('siblings_unchanged')
               or not r.get('frame_unchanged') or r.get('outside_pixels', 0)
               or r.get('layout', {}).get('overflow') or r.get('layout', {}).get('unexpected_wrap'))
               for r in checked.get('replacements', {}).get('records', [])):
            continue
        mark = record.get('mark_id') or next((s for s in record.get('seed_ids', ()) if s in planned), oid)
        if mark not in planned and mark not in source_ids and not (record.get('kind') == 'surface' and mark == oid):
            continue
        plan = planned.get(mark)
        # Geometry and wording must be re-read from the current, source-bound inventory.
        field = {k:plan[k] for k in ('id','route','role','move_with','origin','shape') if k in plan} if plan else None
        crop_seeds = sorted(record.get('seed_ids', ())) if oid.startswith('r-') else []
        items.append({'object_id':oid, 'source_id':mark, 'field':field, 'representation':shape, 'crop_seeds':crop_seeds,
                      'evidence':'group_children' if oid not in verified else 'object_move_delete'})
    routes = candidate.get('batch_spec', {}).get('marked_routes', [])
    if routes and any(record.get('move_members') for record in graph.selected.values()):
        from .marked_routes import plan_move_groups
        owners = {}
        plan_move_groups(routes, graph, sources=owners)
        for gid, members in representations.items():
            if not isinstance(members, list) or not set(members).intersection(candidate['object_ids']):
                continue
            owner = owners.get(tuple(sorted(members)))
            if not owner or gid not in verified and not set(members) <= verified:
                continue
            plan = planned.get(owner['id'])
            if plan:
                items.append({'object_id':gid, 'source_id':owner['id'],
                              'field':{k:plan[k] for k in ('id','route','role','move_with','origin','shape') if k in plan},
                              'representation':members, 'evidence':'object_move_delete' if gid in verified else 'group_children'})
    if items:
        for item in items:
            if isinstance(item['representation'], list):
                item['member_types'] = {member:representations[member] for member in item['representation']}
        payload = {'page':candidate['page'], 'source':graph.source_sha256, 'code':candidate['code'],
                   'candidate':candidate['id'], 'office_pptx':deck, 'items':items}
        write(state.root/'object-recovery'/f"{candidate['page']}"/(key(payload)+'.json'), payload)


def recover_page(service, state, current, directory, page):
    """Rebind missing edits to the current source and commit only newly Office-checked objects."""
    receipts = sorted((state.root/'object-recovery'/str(page)).glob('*.json'))
    if not receipts:
        return []
    row = service.page(current, page)
    code = service.code_binding()
    source = digest(row['snapshot']['path'])
    already = _representation(_read(row['ir']))
    selected = _read(row['scene'])['selected'] if row.get('scene') else {}
    plans = {p['id']:p for p in current.get('marked_plans', {}).get(str(page), [])}
    inventory_path = current.get('experiment_marks', {}).get(str(page))
    inventory = _read(inventory_path) if inventory_path else {}
    known_marks = {mark['id'] for mark in inventory.get('rows', [])} if inventory.get('source_sha256') == source else set()
    missing = {}
    for path in receipts:
        receipt = _read(path)
        if key(receipt) != path.stem:
            raise RuntimeConflict('object recovery receipt changed')
        if (receipt['page'], receipt['source'], receipt['code']) != (page, source, code):
            continue
        for item in receipt['items']:
            oid, mark, field = item['object_id'], item['source_id'], item['field']
            if _matches(item, already, selected):
                continue
            if field and mark in plans and plans[mark]['route'] != field['route']:
                continue
            if field and not {mark, *field.get('move_with', [])} <= known_marks:
                continue
            # A previously checked group is preferable to an older single-picture version.
            identity = mark if item.get('crop_seeds') else oid
            if identity not in missing or isinstance(item['representation'], list):
                missing[identity] = item
    if not missing:
        return []
    parent_sha = digest(row['ir'])
    identity = key([source, parent_sha, missing])
    history = current.setdefault('object_recovery', {}).setdefault(str(page), {})
    if identity in history or state.remaining_seconds() == 0:
        return []
    attempt = {'parent_ir_sha256':parent_sha, 'requested_ids':sorted(missing), 'restored_ids':[]}
    history[identity] = attempt
    # Reuse the production transaction; failed staging and validation must leave the parent untouched.
    for item in missing.values():
        field = item['field']
        if field:
            previous = plans.get(field['id'])
            if not previous or (not previous.get('move_with') and field.get('move_with') and previous.get('origin')=='host'):
                plans[field['id']] = field
    destination = directory / identity[:16]
    destination.mkdir(parents=True, exist_ok=True)
    attempt['trials'] = []
    batches = [missing]
    validation_state = copy(state)
    validation_state.reserve_final_office_check = len(current['pages']) > 1
    for index, batch in enumerate(batches):
        if state.remaining_seconds() == 0:
            break
        trial = deepcopy(current)
        trial.setdefault('marked_plans', {})[str(page)] = list(plans.values())
        targets = sorted({target for item in batch.values() for target in
                         (item['source_id'], *(item.get('field') or {}).get('move_with', []))})
        work = destination/str(index)
        work.mkdir()
        receipt = {'targets':targets, 'parent_ir_sha256':digest(service.page(current,page)['ir'])}
        attempt['trials'].append(receipt)
        try:
            staged = service.batch_reconstruct_page(validation_state, trial, work, page=page, object_ids=targets,
                include_assets=True, include_containers=True, include_region_crops=False, verify_office=True)
            receipt.update(status=staged['status'], candidate_id=staged.get('candidate_id'),
                           office_attempts=staged.get('office_attempts', 0))
            if staged['status'] == 'VALIDATED':
                candidate = trial['candidates'][staged['candidate_id']]
                delivered = _representation(_read(candidate['ir']))
                candidate_scene = Path(candidate['ir']).parent/'scene.json'
                candidate_objects = _read(candidate_scene)['selected'] if candidate_scene.exists() else {}
                restored = sorted(oid for oid,item in batch.items() if _matches(item, delivered, candidate_objects))
                if restored:
                    answer = service.commit_region(validation_state, trial, work/'commit', candidate_id=candidate['id'])
                    if answer['status'] == 'COMMITTED':
                        receipt.update(status='COMMITTED', restored_ids=restored, ir_sha256=digest(candidate['ir']))
                        attempt['restored_ids'].extend(restored)
                        trial['object_recovery'][str(page)][identity] = attempt
                        current.update(trial)
            if index == 0 and len(batch) > 1 and staged['status'] not in {'OFFICE_UNAVAILABLE','NO_CANDIDATES_STAGED'}:
                batches.extend({oid:item} for oid,item in batch.items() if oid not in attempt['restored_ids'])
        except RuntimeConflict:
            raise
        except (BudgetExceeded, PipelineError, OSError, ValueError) as error:
            receipt.update(status='KEPT_PARENT', reason=str(error))
            break
    attempt['status'] = 'COMMITTED' if attempt['restored_ids'] else 'KEPT_PARENT'
    current['object_recovery'][str(page)][identity] = attempt
    write(destination/'object-recovery.json', attempt)
    return attempt['restored_ids']
