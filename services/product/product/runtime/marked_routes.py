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
        ordered = [*ordered[:first], *children, *[n for n in ordered[first:] if n['id'] not in members]]
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
        elif route == 'asset' and mark['label'] == 'connected_diagram':
            row['diagram_members'] = mark['member_ids']
            row['anchor_boxes'] = [marks[i]['bbox'] for i in mark['member_ids']]
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
            and (k in targets or rec.get('mark_id') == mark or mark in rec.get('seed_ids', [])
                 or targets.intersection(rec.get('badge_evidence_ids', ())))
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
