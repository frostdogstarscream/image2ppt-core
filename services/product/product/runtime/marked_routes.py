"""Bind numbered plan entries to source pixels and existing scene objects."""

from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image

from ..contracts import intersection
from ..materialize import digest, write
from ..parallel_store import key


def apply_move_groups(ir, movement):
    """Bring members together only when all crossed nonmembers are disjoint."""
    ordered = sorted(ir['nodes'], key=lambda n: n['z'])
    used = {oid for group in ir.get('groups', []) for oid in group['member_ids']}
    original_group_count = len(ir.get('groups', []))
    issues = []
    for ids in movement:
        members = set(ids)
        if len(members) < 2 or members & used:
            issues.append({'members': ids, 'failure':'GROUP_MEMBER_ALREADY_ASSIGNED',
                           'conflict_ids': sorted(members & used)})
            continue
        candidates = [(n, i) for i, n in enumerate(ordered) if n['id'] in members]
        if len(candidates) != len(members):
            issues.append({'members':ids, 'failure':'GROUP_MEMBER_MISSING',
                           'conflict_ids':sorted(members - {node['id'] for node, _ in candidates})})
            continue
        first = candidates[0][1]
        blockers = sorted({
            other['id']
            for n, idx in candidates
            for other in ordered[first:idx]
            if other['id'] not in members and intersection(n['bbox'], other['bbox']) > 0
        })
        if blockers:
            issues.append({'members':ids, 'failure':'GROUP_Z_ORDER_CONFLICT', 'conflict_ids':blockers})
            continue
        children = [n for n, _ in candidates]
        ordered = [*ordered[:first], *children, *[n for n in ordered[first:] if n['id'] not in members]]
        ir['groups'].append({'id': 'g-' + key(sorted(ids))[:16], 'member_ids': [n['id'] for n in children]})
        used.update(members)
    if len(ir.get('groups', [])) > original_group_count:
        for z, node in enumerate(ordered):
            node['z'] = z
    return issues


def plan_move_groups(routes, graph, *, issues=None, sources=None):
    resolved = {k: [k] for k in graph.selected}
    for oid, record in graph.selected.items():
        for alias in record.get('badge_evidence_ids', ()):
            resolved.setdefault(alias.removeprefix('layout:'), []).append(oid)
    for row in route_trace(routes, graph):
        if row['selected_ids']:
            resolved[row['mark_id']] = row['selected_ids']
    groups = []
    for r in routes:
        ids = [r['id'], *r.get('move_with', [])]
        owned = set()
        face_ids = resolved.get(r['id'], [])
        if r['route']=='surface' and len(face_ids)==1:
            face = face_ids[0]
            if graph.selected[face]['kind']=='surface':
                owned.update(graph.selected[face].get('contained_children', ()))
                linked = {k for k,rec in graph.selected.items() if rec.get('payload',{}).get('background_parent_id')==face}
                owned.update(linked)
                if linked and r.get('group_owned_children'):
                    # A measured card may have several candidate child frames in its
                    # inventory. Use its committed photo/overlay ownership and confirmed
                    # text relations once that card has actually been reconstructed.
                    owned.update(rel.target for rel in graph.relations if rel.source==face
                                 and rel.kind=='contains' and rel.status=='RESOLVED' and rel.target in graph.selected)
        if len(ids) < 2 and not (owned and r.get('group_owned_children')):
            continue
        missing = [i for i in ids if not resolved.get(i)]
        if missing:
            if issues is not None:
                issues.append({'object_id':r['id'], 'kind':'group', 'bbox':r['bbox'],
                               'failure':'GROUP_MEMBER_MISSING', 'conflict_ids':missing})
            continue
        members = owned | {oid for i in ids for oid in resolved[i]}
        group = sorted(members)
        if len(group) > 1 and group not in groups:
            groups.append(group)
            if sources is not None:
                sources[tuple(group)] = r
    return groups


def stage_planned_groups(routes, graph, baseline, output, skip_ids):
    """Group only independently accepted layers. Failed grouping cannot alter their scene or IR."""
    import shutil
    from ..parallel_rebuild import adapt
    issues, sources, proposals = [], {}, []
    existing = {oid:rec['move_members'] for oid,rec in graph.selected.items() if rec.get('move_members')}
    replaced = {}
    planned = plan_move_groups(routes, graph, issues=issues, sources=sources)
    owners = {}
    for group in planned:
        for oid in group:
            owners.setdefault(oid, []).append(sources[tuple(group)]['id'])
    for group in planned:
        members = set(group)
        if any(members <= set(old) for old in existing.values()):
            continue
        route = sources[tuple(group)]
        shared = [oid for oid in group if len(owners[oid])>1]
        if shared:
            issues.append({'object_id':route['id'],'kind':'group','bbox':route['bbox'],
                           'failure':'GROUP_AMBIGUOUS_OWNER','conflict_ids':shared})
            continue
        blockers = members & skip_ids
        reason = 'GROUP_MEMBER_EXCLUDED'
        overlaps = [oid for oid,other in existing.items() if members.intersection(other)]
        if not blockers:
            reason = 'GROUP_MEMBER_ALREADY_ASSIGNED'
            # A photo verified later can extend its own card group. Keep previous
            # membership available until the expanded group survives Z-order checks.
            own_face = [oid for oid in members if oid in graph.selected and graph.selected[oid]['kind']=='surface']
            extend = (route['route']=='surface' and len(overlaps)==1
                      and set(existing[overlaps[0]]) < members
                      and any(face in existing[overlaps[0]] and
                              iou(route.get('measured_label_face') or route['bbox'],graph.selected[face]['bbox']) >= .92
                              for face in own_face))
            if not extend:
                blockers.update(i for oid in overlaps for i in existing[oid] if i in members)
        if blockers:
            issues.append({'object_id':route['id'], 'kind':'group', 'bbox':route['bbox'],
                           'failure':reason, 'conflict_ids':sorted(blockers)})
            continue
        if overlaps:
            replaced[tuple(group)] = overlaps[0]
        proposals.append(group)
    if not proposals:
        return baseline, [], issues

    # adapt also materializes image assets. Keep every trial in its own directory,
    # including trials where layering conflicts eventually reject all proposed groups.
    shutil.copytree(baseline, output)
    for group in proposals:
        old_owner = replaced.get(tuple(group))
        if old_owner is not None:
            graph.selected[old_owner].pop('move_members')
        graph.selected[group[0]]['move_members'] = group
    conflicts = []
    ir = adapt(graph, output, group_issues=conflicts)
    written = {frozenset(g['member_ids']) for g in ir.get('groups', [])}
    accepted = []
    for group in proposals:
        if frozenset(group) in written:
            accepted.extend(group)
            continue
        graph.selected[group[0]].pop('move_members',None)
        old_owner = replaced.get(tuple(group))
        if old_owner is not None:
            graph.selected[old_owner]['move_members'] = existing[old_owner]
        route = sources[tuple(group)]
        issue = next((c for c in conflicts if set(c['members'])==set(group)), {})
        issues.append({'object_id':route['id'], 'kind':'group','bbox':route['bbox'],
                       'failure':issue.get('failure','GROUP_NOT_EMITTED'), 'conflict_ids':issue.get('conflict_ids',[])})
    if not accepted:
        return baseline, [], issues
    if len(accepted) < sum(map(len,proposals)):
        ir = adapt(graph, output)
    write(output/'scene.json', graph.model_dump())
    write(output/'slideir.json', ir)
    return output, accepted, issues


def iou(a, b):
    shared = intersection(a, b)
    return shared / max(1, a[2] * a[3] + b[2] * b[3] - shared)


def card_move_members(rows, *, source_path=None):
    """Select independent text and graphics inside the innermost numbered face."""
    text_rows = [r for r in rows if r.get('ocr_ids') and r['label'] == 'text']
    text_ids = {r['id'] for r in text_rows}
    source = None
    if source_path is not None:
        with Image.open(source_path) as image:
            source = np.array(image.convert('RGB'))
    cards = {}
    for row in rows:
        box = row['bbox']
        if row['label'] == 'card_frame' or (
            row['label'] == 'icon' and row.get('sources') and row.get('intersecting_ocr_ids')
            and any(t['bbox'][2]*t['bbox'][3] <= .65*box[2]*box[3] and
                    intersection(t['bbox'], box) >= .98*t['bbox'][2]*t['bbox'][3] for t in text_rows)
            and (source is None or measured_label_face(source,box) is not None)
        ):
            cards[row['id']] = row
    # A detector can name the same face as an icon; use its independently measured frame.
    frames = [r['bbox'] for r in cards.values() if r['label']=='card_frame']
    cards = {oid:r for oid,r in cards.items() if r['label']=='card_frame' or
             not any(iou(r['bbox'], box)>=.85 for box in frames)}
    leaf = {}
    for oid, row in cards.items():
        box = row['bbox']
        if any(other['id'] != oid and other['bbox'][2]*other['bbox'][3] < .8*box[2]*box[3]
               and intersection(box,other['bbox']) >= .95*other['bbox'][2]*other['bbox'][3]
               for other in cards.values()):
            continue
        candidates = []
        for child in rows:
            cb = child['bbox']
            if child['id'] in cards or cb[2]*cb[3] >= .90*box[2]*box[3]:
                continue
            if intersection(box,cb) < .98*cb[2]*cb[3]:
                continue
            if child['id'] in text_ids or child['label'] in ('image','icon','illustration','decoration','photo'):
                candidates.append(child)
        # Explicit scene or asset records take precedence over duplicating detector windows.
        chosen = []
        for child in sorted(candidates, key=lambda c:(c['id'] not in text_ids,
                                                       not c['id'].startswith('a-'), c['id'])):
            if child['id'] not in text_ids and any(iou(child['bbox'], other['bbox']) >= .7 for other in chosen):
                continue
            chosen.append(child)
        if any(c['id'] in text_ids for c in chosen):
            leaf[oid] = [c['id'] for c in chosen]
    return leaf


def measured_label_face(source, hint):
    """Return a bounded, nearly rectangular connected fill. The shape fitter owns final admission."""
    x,y,w,h = map(round,hint)
    reach = max(2, round(min(w,h)*.12))
    left,top = max(0,x-reach), max(0,y-reach)
    tile = source[top:min(source.shape[0], y+h+reach), left:min(source.shape[1], x+w+reach)]
    roi = source[y:y+h,x:x+w]
    if min(w,h)<4 or roi.shape[:2] != (h,w):
        return None
    face = np.median(roi.reshape(-1,3),axis=0)
    spread = np.quantile(np.abs(roi-face).max(2),.65)
    tolerance = np.clip(2*spread+12,14,64)
    mask = (np.abs(tile-face).max(2)<tolerance).astype('uint8')
    _, _, components, _ = cv2.connectedComponentsWithStats(mask,8)
    for bx,by,bw,bh,area in components[1:]:
        box = [int(left+bx),int(top+by),int(bw),int(bh)]
        if (min(bx,by)<=0 or bx+bw>=tile.shape[1] or by+bh>=tile.shape[0] or
                area < .72*bw*bh or iou(box,hint)<.80):
            continue
        return box
    return None


def picture_frame(source, box):
    """Refine a pictorial mark to one connected frame, dropping adjacent captions."""
    height, width = source.shape[:2]
    x, y, w, h = map(round, box)
    pad = 12
    left, top = max(0, x - pad), max(0, y - pad)
    tile = source[top : min(height, y + h + pad), left : min(width, x + w + pad)]
    if min(tile.shape[:2]) < 16:
        return None
    ring = np.concatenate((tile[0], tile[-1], tile[:, 0], tile[:, -1]))
    color = np.median(ring, axis=0)
    ink = np.abs(tile.astype(float) - color).max(2) > 9
    _, _, components, _ = cv2.connectedComponentsWithStats(ink.astype('uint8'), 8)
    matches = []
    for bx, by, bw, bh, area in components[1:]:
        if min(bw, bh) < 14 or area < 0.10 * bw * bh:
            continue
        if bx == 0 or by == 0 or bx + bw == tile.shape[1] or by + bh == tile.shape[0]:
            continue
        frame = [int(left + bx), int(top + by), int(bw), int(bh)]
        score = iou(frame, box)
        if score >= 0.60:
            matches.append((score, frame))
    return max(matches)[1] if matches else None


def measured_picture_frame(source, box, peer_boxes):
    """Find one picture without following ink into a neighbouring picture or panel."""
    from ..recovery.region_crop import MAX_MEASURED_PICTURE_FRACTION
    height, width = source.shape[:2]
    x, y, w, h = map(round, box)
    if min(w, h) < 14 or x < 0 or y < 0 or x+w > width or y+h > height:
        return None, 'INVALID_PICTURE_BOUNDS'
    if w*h > MAX_MEASURED_PICTURE_FRACTION*width*height:
        return None, 'PICTURE_BOUNDARY_NOT_OBJECT_SCOPED'
    peers = [other for other in peer_boxes if iou(box, other) < .80]
    if any(intersection(box, p) > .65*p[2]*p[3] for p in peers):
        return None, 'MARK_CONTAINS_ANOTHER_PICTURE'

    # A detected rectangle can end inside the last row of a photo. Locate its
    # actual transition into the surrounding card. A gap shared with a peer is
    # divided at its midpoint, so the stronger edge of that peer cannot win.
    limits = [0, width, 0, height]
    for px, py, pw, ph in peers:
        if min(y+h, py+ph)-max(y,py) > .45*min(h,ph):
            if px+pw <= x:
                limits[0] = max(limits[0], round((px+pw+x)/2))
            if px >= x+w:
                limits[1] = min(limits[1], round((px+x+w)/2))
        if min(x+w, px+pw)-max(x,px) > .45*min(w,pw):
            if py+ph <= y:
                limits[2] = max(limits[2], round((py+ph+y)/2))
            if py >= y+h:
                limits[3] = min(limits[3], round((py+y+h)/2))
    radius = max(4, round(.11*min(w,h)))
    edges = []
    for side, (axis, position, start, span, direction) in enumerate((
        (source, x, y, h, 1), (source, x+w, y, h, -1),
        (source.swapaxes(0,1), y, x, w, 1), (source.swapaxes(0,1), y+h, x, w, -1),
    )):
        inset = max(1, round(span*.12))
        band = axis[start+inset:start+span-inset]
        low = max(3, position-radius, limits[side] if direction==1 else 0)
        high = min(axis.shape[1]-3, position+radius, limits[side] if direction==-1 else axis.shape[1])
        scores = []
        for edge in range(low, high+1):
            before = np.median(band[:, edge-3:edge], axis=1)
            after = np.median(band[:, edge:edge+3], axis=1)
            outside = before if direction==1 else after
            # Looking into the page background should see a smooth gutter,
            # even if the interior contains rules, text or a sharp horizon.
            spread = np.percentile(abs(outside-np.median(outside,axis=0)).max(1), 75)
            contrast = np.percentile(abs(before-after).max(1), 35)
            if contrast >= 8 and spread <= max(8, .5*contrast):
                scores.append((min(contrast,48)/(1+spread/12)/(1+abs(edge-position)/radius),edge))
        if not scores:
            return None, 'NO_MEASURED_' + ('LEFT','RIGHT','TOP','BOTTOM')[side] + '_EDGE'
        edges.append(max(scores)[1])
    left, right, top, bottom = edges
    frame = [left,top,right-left,bottom-top]
    if iou(frame,box) < .65 or any(intersection(frame,p) for p in peers):
        return None, 'PICTURE_EDGE_CONFLICT'
    return frame, 'FOUR_MEASURED_EDGES'


def bind_plan(fields, inventory_path, graph, source_path, *, separate_cards=False):
    inventory = json.loads(Path(inventory_path).read_text(encoding='utf8'))
    if inventory['source_sha256'] != graph.source_sha256 or digest(source_path) != graph.source_sha256:
        raise ValueError('marked plan belongs to a different source')
    marks = {r['id']: r for r in inventory['rows']}
    if separate_cards:
        from ..recovery.layered_card import ordered_card_frames
        marks.update((r['id'],r) for r in ordered_card_frames(inventory['rows']))
    source = np.asarray(Image.open(source_path).convert('RGB'))
    routes = []
    for field in fields:
        mark = marks[field['id']]
        route, box = field['route'], mark['bbox']
        row = {**field, 'bbox': box, 'label': mark['label'], 'target_ids': []}
        if separate_cards and route=='surface' and field.get('origin')=='host':
            row['group_owned_children'] = True
        if separate_cards and route == 'surface' and mark['label'] == 'icon':
            row['measured_label_face'] = measured_label_face(source, box)
        row.pop('required_card_frames', None)
        if route == 'surface' and mark.get('required_card_frames'):
            row['required_card_frames'] = mark['required_card_frames']
        if route == 'text' and mark.get('ocr_ids') and mark['label'] == 'text' and separate_cards:
            # Scope the retry to the marked field. Two touching frames can own different words.
            row['target_ids'] = [t.id for t in graph.text.texts if t.id == mark['id']]
        elif route in ('text', 'table', 'preserve'):
            for t in graph.text.texts:
                covered = intersection(t.bbox, box) / max(1, t.bbox[2] * t.bbox[3])
                if t.id == mark['id'] or covered >= 0.90:
                    row['target_ids'].append(t.id)
            if route == 'preserve':
                row['target_ids'].extend(
                    obj.id for obj in graph.asset.objects if intersection(obj.bbox, box) > 0
                 )
        elif route == 'asset' and mark['label'] == 'connected_diagram':
            row['diagram_members'] = mark['member_ids']
            row['anchor_boxes'] = [marks[i]['bbox'] for i in mark['member_ids']]
        elif route == 'asset':
            if mark['label'] == 'image':
                if separate_cards:
                    peers = [r['bbox'] for r in marks.values() if r['label']=='image' and r['id']!=mark['id']]
                    row['picture_frame'], row['picture_boundary'] = measured_picture_frame(source, box, peers)
                    if not row['picture_frame'] and row['picture_boundary'].startswith('NO_MEASURED_'):
                        component = picture_frame(source, box)
                        if component and not any(intersection(component,p) for p in peers if iou(box,p)<.80):
                            row['picture_frame'], row['picture_boundary'] = component, 'CLOSED_COMPONENT'
                else:
                    row['picture_frame'] = picture_frame(source, box)
                row['intrinsic_ui'] = any(
                    word in field.get('role', '').casefold()
                    for word in ('screenshot', 'dashboard', '截图', '工作台')
                )
            for obj in graph.asset.objects:
                if obj.id == mark['id'] or iou(obj.bbox, box) >= 0.75:
                    row['target_ids'].append(obj.id)
            if (
                mark['id'].startswith('m-')
                and not row['target_ids']
                and not row.get('picture_frame')
                and not row.get('picture_boundary')
                and not mark.get('intersecting_ocr_ids')
            ):
                kind = 'icon' if mark['label'] == 'icon' else 'illustration'
                row['new_asset'] = {
                    'id': mark['id'],
                    'kind': kind,
                    'bbox': box,
                    'description': 'Numbered ' + kind,
                    'prompt_en': 'Isolate the complete marked ' + kind,
                    'text_policy': 'exclude_independent_text',
                    'components': [kind],
                    'parts': [{'name': kind, 'bbox': box}],
                }
                row['target_ids'].append(mark['id'])
        routes.append(row)
    # UI detectors label controls and enclosing cards alike as icons. Treat
    # containment as a surface hypothesis; the surface stage must prove its edges.
    planned = {r['id'] for r in routes}
    for mark in marks.values():
        if mark['label'] != 'icon' or mark['id'] in planned or not mark.get('intersecting_ocr_ids'):
            continue
        box = mark['bbox']
        if box[2] < 60 or box[3] < 22 or box[2] < box[3] * 1.4:
            continue
        children = [
            t.id
            for t in graph.text.texts
            if 0.93 * t.bbox[2] * t.bbox[3] <= intersection(t.bbox, box)
            and t.bbox[2] * t.bbox[3] <= 0.55 * box[2] * box[3]
        ]
        if children:
            routes.append(
                {
                    'id': mark['id'],
                    'bbox': box,
                    'label': 'icon',
                    'route': 'surface',
                    'origin': 'detected_container',
                    'target_ids': children,
                    'move_with': [],
                }
            )
    return routes


def route_trace(routes, graph):
    """Resolve each plan against the records produced by this exact stage."""
    trace = []
    for route in routes:
        mark = route['id']
        targets = set(route.get('target_ids', [])) | {mark}
        expected = {
            'surface': {'surface'},
            'table': {'table'},
            'chart': {'chart'},
            'asset': {'asset'},
            'text': {'text'},
            'preserve': set(),
        }[route['route']]
        ids = [
            k
            for k, rec in graph.selected.items()
            if rec['kind'] in expected
            and (k in targets or rec.get('mark_id') == mark or mark in rec.get('seed_ids', [])
                 or targets.intersection(rec.get('badge_evidence_ids', ())))
        ]
        if not ids and route['route']=='surface':
            box = route.get('measured_label_face') or route['bbox']
            same = [oid for oid,rec in graph.selected.items() if rec['kind']=='surface' and iou(box,rec['bbox'])>=.92]
            if len(same)==1:
                ids = same
        trace.append(
            {
                'mark_id': mark,
                'route': route['route'],
                'bbox': route['bbox'],
                'origin': route.get('origin', 'planner'),
                'target_ids': sorted(targets),
                'move_with': route.get('move_with', []),
                'selected_ids': ids,
            }
        )
    return trace


def bind_asset_layers(proposals, layers_dir, source_sha):
    """Only seed a layer when a connected alpha component is contained in the mark."""
    if not proposals:
        return []
    manifest = json.loads((Path(layers_dir) / 'manifest.json').read_text(encoding='utf8'))
    if manifest['source_sha256'] != source_sha:
        raise ValueError('visual mark layer source mismatch')
    layers = {p['id']: [] for p in proposals}
    for entry in manifest['layers']:
        if not entry['z']:
            continue
        name = entry['file']
        path = (Path(layers_dir) / name).resolve()
        if not path.is_relative_to(Path(layers_dir).resolve()) or digest(path) != entry['sha256']:
            raise ValueError('invalid marked asset layer binding')
        with Image.open(path) as image:
            alpha = np.asarray(image.convert('RGBA'))[..., 3]
        _, _, stats, _ = cv2.connectedComponentsWithStats((alpha > 24).astype('uint8'), 8)
        for p in proposals:
            if any(
                area >= 4 and intersection(p['bbox'], [int(x), int(y), int(w), int(h)]) >= w * h * 0.95
                for x, y, w, h, area in stats[1:]
            ):
                layers[p['id']].append(name)
    return [
        {**p, 'layers': names, 'parts': [{**part, 'layers': names} for part in p['parts']]}
        for p in proposals
        if (names := layers[p['id']]) and len(names) <= 12
    ]
