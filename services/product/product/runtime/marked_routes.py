"""Bind numbered plan entries to source pixels and existing scene objects."""

from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image

from ..contracts import intersection
from ..materialize import digest
from ..parallel_store import key


def apply_move_groups(ir, movement):
    """Bring members together only when all crossed nonmembers are disjoint."""
    ordered = sorted(ir['nodes'], key=lambda n: n['z'])
    used = set()
    for ids in movement:
        members = set(ids)
        if len(members) < 2 or members & used:
            continue
        candidates = [(n, i) for i, n in enumerate(ordered) if n['id'] in members]
        if len(candidates) != len(members):
            continue
        first = candidates[0][1]
        if any(
            intersection(n['bbox'], other['bbox']) > 0
            for n, idx in candidates
            for other in ordered[first:idx]
            if other['id'] not in members
        ):
            continue
        children = [n for n, _ in candidates]
        last = candidates[-1][1]
        pre_children = []
        post_nonmembers = []
        for n in ordered[first:last]:
            if n['id'] in members:
                continue
            is_lower_layer = (
                n.get('type') == 'shape'
                or (n.get('type') == 'image' and (n['id'] == 'source-plate' or n['id'].startswith(('a-c', 'r-'))))
            )
            if (
                is_lower_layer
                and n['id'] not in used
                and all(intersection(n['bbox'], c['bbox']) == 0 for c in children)
                and all(intersection(n['bbox'], p['bbox']) == 0 for p in post_nonmembers)
            ):
                pre_children.append(n)
            else:
                post_nonmembers.append(n)
        tail = [n for n in ordered[last + 1:] if n['id'] not in members]
        ordered = [*ordered[:first], *pre_children, *children, *post_nonmembers, *tail]
        ir['groups'].append({'id': 'g-' + key(sorted(ids))[:16], 'member_ids': [n['id'] for n in children]})
        used.update(members)
    if used:
        for z, node in enumerate(ordered):
            node['z'] = z


def plan_move_groups(routes, graph):
    resolved = {k: [k] for k in graph.selected}
    for row in route_trace(routes, graph):
        resolved[row['mark_id']] = row['selected_ids']
    groups = []
    for r in routes:
        ids = [r['id'], *r.get('move_with', [])]
        if len(ids) > 1 and all(resolved.get(i) for i in ids):
            group = sorted({oid for i in ids for oid in resolved[i]})
            if len(group) > 1 and group not in groups:
                groups.append(group)
    return groups


def _backing_edge_evidence(plate, box):
    """Look for a continuous dark outline flanked by lighter pixels on both sides of a card edge."""
    x, y, w, h = map(round, box)
    evidence = []
    for side, plane, edge, outward, start, length, depth in (
        ('top',plate,y,-1,x,w,h), ('bottom',plate,y+h-1,1,x,w,h),
        ('left',plate.T,x,-1,y,h,w), ('right',plate.T,x+w-1,1,y,h,w),
    ):
        if depth < 24 or length < 36:
            continue
        # Read only the central span to avoid curved corners. The card interior and the
        # farther exterior jointly distinguish a narrow dark band from a smooth gradient.
        along = np.arange(max(0,start+length//5), min(plane.shape[1],start+length-length//5))
        if not along.size or min(edge+outward*16,edge-outward*10)<0 or max(edge+outward*16,edge-outward*10)>=plane.shape[0]:
            continue
        def median_band(a,b):
            return np.median(plane[edge+outward*np.arange(a,b)[:,None],along],axis=0)
        near = median_band(2,7)
        darkening = np.minimum(median_band(12,17),median_band(-10,-5)) - near
        fraction = float((darkening >= 3).mean())
        if fraction >= .8 and np.median(darkening) >= 4:
            evidence.append({'side':side,'darkened_fraction':round(fraction,3),
                             'median_luma_drop':round(float(np.median(darkening)),2)})
    return evidence


def resolved_surface_groups(graph, directory, *, diagnostics=None):
    """Discover movable surface/child assemblies using the frozen scene and owner receipts."""
    directory = Path(directory).resolve()
    guard_file = directory / 'runtime-ocr-protection.json'
    if not guard_file.is_file():
        return []
    guard = json.loads(guard_file.read_text('utf8'))
    if guard.get('source_sha256') != graph.source_sha256:
        return []
    objects = {obj.id: obj for obj in [*graph.surface.shapes, *graph.asset.objects, *graph.text.texts]}
    children = {}
    doubtful = set()
    for relation in graph.relations:
        if relation.status != 'RESOLVED' or not relation.evidence:
            doubtful.update((relation.source, relation.target))
        elif relation.kind in ('contains', 'part_of'):
            parent, child = ((relation.source, relation.target) if relation.kind == 'contains'
                             else (relation.target, relation.source))
            children.setdefault(parent, set()).add(child)
    groups = []
    backing = None
    for parent, members in children.items():
        ids = {parent, *members}
        if parent in members or ids & doubtful or not ids <= graph.selected.keys():
            continue
        panel = graph.selected[parent]
        if panel['kind'] != 'surface' or not any(graph.selected[i]['kind'] in ('asset', 'text') for i in members):
            continue
        if any(graph.ownership.get(i) != 'RESOLVED' or not graph.selected[i].get('complete_owner_scope')
               or graph.selected[i].get('move_safety') == 'MOVE_UNSAFE' for i in ids):
            continue
        box = panel['bbox']
        if any(intersection(box, objects[i].bbox) < objects[i].bbox[2] * objects[i].bbox[3] for i in members):
            continue
        if any(intersection(box, obj.bbox) > 0 for i, obj in objects.items() if i not in ids):
            continue
        owned_lines = {line for i in members for line in graph.text.members.get(i, [])}
        if any(intersection(box, row['bbox']) > 0 and row.get('id') not in owned_lines for row in guard['lines']):
            continue
        for i in ids:
            record = graph.selected[i]
            proof = record.get('owner_evidence') or {}
            path = (directory / proof.get('file', '')).resolve()
            if (record.get('source_sha256') != graph.source_sha256 or not path.is_relative_to(directory)
                    or not path.is_file() or digest(path) != proof.get('sha256')):
                break
        else:
            if backing is None:
                ref = graph.raster_backing or {}
                path = (directory / ref.get('path','')).resolve()
                if (ref.get('source_sha256') != graph.source_sha256 or not path.is_relative_to(directory)
                        or not path.is_file() or digest(path) != ref.get('sha256')):
                    return []
                with Image.open(path) as image:
                    backing = np.asarray(image.convert('L'),dtype=np.int16)
            edges = _backing_edge_evidence(backing, box)
            if edges:
                if diagnostics is not None:
                    diagnostics.append({'object_id':'g-'+key(sorted(ids))[:16], 'backing_id':parent, 'kind':'group', 'bbox':box,
                        'failure':'BACKGROUND_REPAIR_FAILED', 'reason':'Visible edge on the exposed group backing',
                        'members':[parent,*sorted(members)], 'observed_edges':edges,
                        'backing_sha256':graph.raster_backing['sha256']})
                continue
            groups.append([parent, *sorted(members)])
    return groups


def iou(a, b):
    shared = intersection(a, b)
    return shared / max(1, a[2] * a[3] + b[2] * b[3] - shared)


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


def bind_plan(fields, inventory_path, graph, source_path):
    inventory = json.loads(Path(inventory_path).read_text(encoding='utf8'))
    if inventory['source_sha256'] != graph.source_sha256 or digest(source_path) != graph.source_sha256:
        raise ValueError('marked plan belongs to a different source')
    marks = {r['id']: r for r in inventory['rows']}
    source = np.asarray(Image.open(source_path).convert('RGB'))
    routes = []
    for field in fields:
        mark = marks[field['id']]
        route, box = field['route'], mark['bbox']
        row = {**field, 'bbox': box, 'label': mark['label'], 'target_ids': []}
        if route in ('text', 'table', 'preserve'):
            for t in graph.text.texts:
                covered = intersection(t.bbox, box) / max(1, t.bbox[2] * t.bbox[3])
                if t.id == mark['id'] or covered >= 0.90:
                    row['target_ids'].append(t.id)
            if route == 'preserve':
                row['target_ids'].extend(
                    obj.id for obj in graph.asset.objects if intersection(obj.bbox, box) > 0
                )
        elif route == 'asset':
            if mark['label'] == 'image':
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
            'asset': {'asset'},
            'text': {'text'},
            'preserve': set(),
        }[route['route']]
        ids = [
            k
            for k, rec in graph.selected.items()
            if rec['kind'] in expected
            and (k in targets or rec.get('mark_id') == mark or mark in rec.get('seed_ids', []))
        ]
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
