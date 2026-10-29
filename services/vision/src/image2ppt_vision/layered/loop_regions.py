"""Local source retention and grouping, adapted from the v3.1 prototypes.

Only a new proposal is changed. Every pixel-backed object discloses its edit
scope. Retention punches sibling images instead of covering duplicate ink.
"""
import heapq
import copy
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ..storage import digest
from .loop_review import clip, intersect

MAX_REGION_SHARE = .16
MAX_PAGE_RETENTION = .55


def union(a, b):
    x, y = min(a[0], b[0]), min(a[1], b[1])
    return [x, y, max(a[0]+a[2], b[0]+b[2])-x, max(a[1]+a[3], b[1]+b[3])-y]


def inside(a, b):
    return intersect(a, b) >= a[2]*a[3]-.01


def restore(ir, proposal, regions):
    """Close over intersecting text/table objects, then replace bounded regions.

    Large closures are refused, not silently turned into whole-slide pictures.
    Parent background shapes may remain beneath a patch; sibling raster pixels
    and native text never do. Every removed evidence ID keeps one owner.
    """
    proposal = Path(proposal)
    W, H = ir['canvas']['source_px']
    pending = [{'bbox': clip(r['bbox'], [W, H], r.get('pad', 3)),
                'reasons': {r.get('code', 'source_retention')}} for r in regions]
    changed = True
    while changed:
        changed = False
        for region in pending:
            box = region['bbox']
            for node in ir['nodes']:
                closed = node['type'] in ('text', 'table', 'chart') or (node['type'] == 'shape' and
                         (node.get('geometry') != 'rect' or node.get('fill_gradient') or node.get('line')))
                if closed and intersect(node['bbox'], box):
                    expanded = clip(union(box, node['bbox']), [W, H])
                    if expanded != box:
                        region['bbox'] = box = expanded
                        changed = True
            for other in list(pending):
                if other is region or not intersect(box, other['bbox']):
                    continue
                region['bbox'] = union(box, other['bbox'])
                region['reasons'].update(other['reasons'])
                pending.remove(other)
                changed = True
                break
            if changed:
                break
    prior = [n['bbox'] for n in ir['nodes'] if n['id'].startswith('retained-')]
    occupancy = np.zeros((H, W), bool)
    for box in prior:
        x, y, w, h = clip(box, [W, H]); occupancy[y:y+h, x:x+w] = True
    reports, rejected = [], []
    source = Image.open(proposal/ir['source_image']).convert('RGB')
    used_ids = {n['id'] for n in ir['nodes']} | {a['id'] for a in ir['assets']}
    for region in sorted(pending, key=lambda r: (r['bbox'][1], r['bbox'][0])):
        x, y, w, h = box = clip(region['bbox'], [W, H])
        if any(inside(box, b) for b in prior):
            continue
        trial = occupancy.copy(); trial[y:y+h, x:x+w] = True
        if w*h > MAX_REGION_SHARE*W*H or trial.mean() > MAX_PAGE_RETENTION:
            rejected.append({'bbox': box, 'code': 'retention_limit', 'reasons': sorted(region['reasons'])})
            continue
        occupancy = trial
        owned, removed, trimmed = set(), [], []
        assets = {a['id']: a for a in ir['assets']}
        for node in list(ir['nodes']):
            if not intersect(node['bbox'], box):
                continue
            if node['type'] in ('text', 'table', 'chart') or inside(node['bbox'], box):
                owned.update(node['evidence_ids'])
                if node['type'] == 'image':
                    asset = assets[node['asset_id']]
                    owned.update(asset['contains_evidence_ids'])
                    ir['assets'].remove(asset)
                ir['nodes'].remove(node); removed.append(node['id'])
            elif node['type'] == 'shape':
                # Flat, borderless rectangles can be cropped exactly into four
                # pieces. Other presets were closed over above.
                nx, ny, nw, nh = node['bbox']; right, bottom = nx+nw, ny+nh
                ix, iy, iright, ibottom = max(x, nx), max(y, ny), min(x+w, right), min(y+h, bottom)
                boxes = [[nx, ny, nw, iy-ny], [nx, ibottom, nw, bottom-ibottom],
                         [nx, iy, ix-nx, ibottom-iy], [iright, iy, right-iright, ibottom-iy]]
                ir['nodes'].remove(node); removed.append(node['id'])
                pieces = []
                for part in boxes:
                    if min(part[2:]) <= 0: continue
                    piece = copy.deepcopy(node); piece['id'] = node['id']+'-clip-'+str(len(pieces))
                    piece['bbox'] = part; piece['evidence_ids'] = node['evidence_ids'] if not pieces else []
                    pieces.append(piece)
                ir['nodes'].extend(pieces); trimmed.extend(p['id'] for p in pieces)
            elif node['type'] == 'image':
                asset = assets[node['asset_id']]
                with Image.open(proposal/asset['path']) as image:
                    rgba = np.asarray(image.convert('RGBA')).copy()
                nx, ny, nw, nh = node['bbox']; ah, aw = rgba.shape[:2]
                left, right = max(0, math.floor((x-nx)/nw*aw)), min(aw, math.ceil((x+w-nx)/nw*aw))
                top, bottom = max(0, math.floor((y-ny)/nh*ah)), min(ah, math.ceil((y+h-ny)/nh*ah))
                rgba[top:bottom, left:right, 3] = 0
                if not rgba[:, :, 3].any():
                    owned.update(node['evidence_ids']); owned.update(asset['contains_evidence_ids'])
                    ir['assets'].remove(asset); ir['nodes'].remove(node); removed.append(node['id'])
                    continue
                relative = f'assets/{node["id"]}-clear-{len(reports)}.png'
                Image.fromarray(rgba).save(proposal/relative)
                asset.update(path=relative, sha256=digest(proposal/relative))
                trimmed.append(node['id'])
        # Include evidence deliberately left unresolved by the original emitter.
        evidence = {e['id']: e for e in ir['evidence']}
        owned.update(i for i in ir.get('unresolved_evidence_ids', []) if inside(evidence[i]['bbox'], box))
        for asset in ir['assets']:
            for eid in list(asset['contains_evidence_ids']):
                if evidence[eid]['kind'] == 'text' and inside(evidence[eid]['bbox'], box):
                    owned.add(eid)
            asset['contains_evidence_ids'] = [i for i in asset['contains_evidence_ids'] if i not in owned]
        ir['unresolved_evidence_ids'] = [i for i in ir.get('unresolved_evidence_ids', []) if i not in owned]
        index = 0
        while f'retained-{index:03d}' in used_ids:
            index += 1
        aid = f'retained-{index:03d}'; used_ids.add(aid); eid = 'ev-'+aid
        relative = f'assets/{aid}.png'; source.crop((x, y, x+w, y+h)).save(proposal/relative)
        ir['evidence'].append({'id': eid, 'kind': 'image', 'bbox': box,
                               'description': 'Source pixels preserved after bounded quality repair.'})
        ir['assets'].append({'id': aid, 'path': relative, 'sha256': digest(proposal/relative), 'purpose': 'module',
                             'contains_evidence_ids': sorted(owned | {eid}),
                             'description': 'move_resize_only: source pixels; internal content is not native editable.'})
        ir['nodes'].append({'id': aid, 'type': 'image', 'asset_id': aid, 'bbox': box, 'evidence_ids': [],
                            'z': max((n['z'] for n in ir['nodes']), default=-1)+1})
        reports.append({'id': aid, 'bbox': box, 'reasons': sorted(region['reasons']),
                        'removed_nodes': removed, 'trimmed_rasters': trimmed,
                        'evidence_ids': sorted(owned), 'editable': 'move_resize_only'})
    source.close()
    ir['groups'] = []  # Rebuild once the final object membership is known.
    for z, node in enumerate(sorted(ir['nodes'], key=lambda n: n['z'])):
        node['z'] = z
    return {'regions': reports, 'rejected': rejected, 'retained_area_fraction': round(float(occupancy.mean()), 4)}


def safe_order(nodes, groups):
    """v3.1 overlap partial order: grouping must not change visible paint order."""
    ordered = sorted(nodes, key=lambda n: n['z'])
    owner = {n['id']: n['id'] for n in ordered}
    for group in groups:
        for nid in group['member_ids']:
            owner[nid] = group['id']
    vertices = set(owner.values()); edges = {v: set() for v in vertices}; degree = {v: 0 for v in vertices}
    priority = {v: min(i for i, n in enumerate(ordered) if owner[n['id']] == v) for v in vertices}
    for i, a in enumerate(ordered):
        for b in ordered[i+1:]:
            u, v = owner[a['id']], owner[b['id']]
            if u != v and intersect(a['bbox'], b['bbox']) and v not in edges[u]:
                edges[u].add(v); degree[v] += 1
    ready = [(priority[v], v) for v in vertices if degree[v] == 0]; heapq.heapify(ready); out = []
    while ready:
        _, v = heapq.heappop(ready)
        out.extend(n['id'] for n in ordered if owner[n['id']] == v)
        for successor in sorted(edges[v]):
            degree[successor] -= 1
            if degree[successor] == 0:
                heapq.heappush(ready, (priority[successor], successor))
    return out if len(out) == len(nodes) else None


def group_modules(ir, perception):
    W, H = ir['canvas']['source_px']; groups, used, report = [], set(), []
    # Real containers only. Never invent a page grid or a reading-order group.
    candidates = [e for e in perception['elements'] if .008*W*H <= e['bbox'][2]*e['bbox'][3] <= .32*W*H
                  and e['bbox'][2] >= .08*W and e['bbox'][3] >= .04*H]
    for element in sorted(candidates, key=lambda e: e['bbox'][2]*e['bbox'][3]):
        box = element['bbox']
        members = [n for n in ir['nodes'] if inside(n['bbox'], box)]
        if len(members) < 2 or any(n['id'] in used for n in members):
            continue
        if any(n['type'] == 'text' and intersect(n['bbox'], box) and n not in members for n in ir['nodes']):
            continue
        group = {'id': f'module-{len(groups):03d}', 'member_ids': [n['id'] for n in members]}
        order = safe_order(ir['nodes'], groups+[group])
        if order is None:
            continue
        groups.append(group); used.update(group['member_ids'])
        report.append({'id': group['id'], 'source_object': element['id'], 'bbox': box,
                       'member_ids': group['member_ids'], 'movement_acceptance': False})
    order = safe_order(ir['nodes'], groups) or [n['id'] for n in sorted(ir['nodes'], key=lambda n: n['z'])]
    nodes = {n['id']: n for n in ir['nodes']}
    ir['nodes'] = [nodes[i] for i in order]
    for z, node in enumerate(ir['nodes']):
        node['z'] = z
    ir['groups'] = groups
    return report
