"""Plan and bind original-site observations on disposable Office copies."""
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageChops

from .materialize import digest


def edit_sites(irfile, target_ids=None, *, include_unverified=False):
    irfile = Path(irfile)
    ir = json.loads(irfile.read_text('utf8'))
    scene = json.loads((irfile.parent / 'scene.json').read_text('utf8'))
    if ir.get('source_sha256') != scene.get('source_sha256'):
        raise ValueError('original-site scene belongs to another image')
    selected = scene['selected']
    types = {node['id']: node['type'] for node in ir['nodes']}
    images = {oid for oid, kind in types.items() if kind == 'image' and oid != 'source-plate'
              and selected.get(oid, {}).get('move_safety') != 'MOVE_UNSAFE'}
    moves = sorted(images if target_ids is None else images.intersection(target_ids))
    children = {}
    for edge in scene.get('relations', []):
        if edge['kind'] in ('contains','part_of'):
            backing, child = ((edge['source'],edge['target']) if edge['kind']=='contains'
                              else (edge['target'],edge['source']))
            children.setdefault(backing, {}).setdefault(child, []).append(edge)
    pairs, unverified = [], []
    for backing, candidates in children.items():
        if types.get(backing) != 'shape' or not images.intersection(candidates):
            continue
        if target_ids is not None and not set(target_ids).intersection({backing,*candidates}):
            continue
        if len(candidates) != 1:
            unverified.append({'backing':backing,'members':sorted(candidates),'reason':'MULTIPLE_CHILDREN'})
            continue
        front, edges = next(iter(candidates.items()))
        if any(edge['status'] != 'RESOLVED' or not edge.get('evidence') for edge in edges):
            unverified.append({'backing':backing,'members':[front],'reason':'UNPROVEN_RELATION'})
            continue
        if (front in images and types.get(backing) == 'shape'
                and selected.get(backing, {}).get('kind') == 'surface'
                and (target_ids is None or front in target_ids or backing in target_ids)):
            pairs.append((backing, front))
    if include_unverified:
        return moves, sorted(pairs), unverified
    return moves, sorted(pairs)


def visible_box(box, rotation):
    x, y, w, h = box[:4]
    angle = math.radians(rotation)
    width = abs(w * math.cos(angle)) + abs(h * math.sin(angle))
    height = abs(w * math.sin(angle)) + abs(h * math.cos(angle))
    return x + (w - width) / 2, y + (h - height) / 2, width, height


def site_geometry(irfile):
    """Derive the observation footprint from the bound IR, including prior restoration extents."""
    from .parallel_office import _node_bleed_margin_px
    irfile = Path(irfile)
    ir = json.loads(irfile.read_text('utf8'))
    scene = json.loads((irfile.parent/'scene.json').read_text('utf8'))
    if not ir.get('source_sha256') or ir['source_sha256'] != scene.get('source_sha256'):
        raise ValueError('original-site geometry lacks a bound source')
    sw, sh = ir['canvas']['source_px']
    pw, ph = ir['canvas']['target_px']
    sx, sy = pw/sw, ph/sh
    geometry = {}
    for node in ir['nodes']:
        oid = node['id']
        if oid == 'source-plate':
            continue
        x, y, w, h = node['bbox']
        bounds = (x*sx, y*sy, w*sx, h*sy)
        rotation = node.get('rotation_deg', 0)
        left, top, width, height = visible_box(bounds, rotation)
        right, bottom = left + width, top + height
        restore = scene['selected'].get(oid, {}).get('restore_bbox')
        if restore:
            rx, ry, rw, rh = restore
            left, top = min(left, rx*sx), min(top, ry*sy)
            right, bottom = max(right, (rx+rw)*sx), max(bottom, (ry+rh)*sy)
        pad = _node_bleed_margin_px(node) + 12
        window = (max(0, math.floor(left)-pad), max(0, math.floor(top)-pad),
                  min(pw, math.ceil(right)+pad), min(ph, math.ceil(bottom)+pad))
        geometry[oid] = (tuple(v*.75 for v in bounds) + (rotation,), window, pad-12)
    return geometry, (pw, ph)


def observation_window(geometry, members):
    boxes = [geometry[member][1] for member in members]
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def clear_position(signature, site, slide_width, slide_height, gap):
    """Move the whole transformed picture beyond the observation crop, including bleed."""
    x, y, w, h, rotation = signature[:5]
    left, top, right, bottom = site
    bx, by, bw, bh = visible_box((x, y, w, h), rotation)
    choices = [(right + gap - bx, 0), (left - gap - bx - bw, 0),
               (0, bottom + gap - by), (0, top - gap - by - bh)]
    # Large images may only clear the crop on the pasteboard of the test copy.
    def score(offset):
        dx, dy = offset
        inside = bx + dx >= 0 and by + dy >= 0 and bx + dx + bw <= slide_width and by + dy + bh <= slide_height
        return not inside, abs(dx) + abs(dy)
    dx, dy = min(choices, key=score)
    return x + dx, y + dy


def site_file(page, members, action):
    token = hashlib.sha256(json.dumps([page, members, action]).encode()).hexdigest()[:24]
    return f'{page:02d}-{token}-site.png'


def record_site(output, page, members, action, window, rendered, render_file, original_frames, reopened_frames):
    left, top, right, bottom = window
    if right <= left or bottom <= top:
        raise ValueError('original site is outside the rendered slide')
    path = Path(output) / site_file(page, members, action)
    Image.fromarray(rendered[top:bottom, left:right]).save(path)
    render_hash = digest(render_file)
    full = Path(output) / f'{render_hash}-reopen.png'
    if not full.is_file():
        shutil.copyfile(render_file, full)
    return {'page': page, 'members': list(members), 'action': action, 'window': list(window),
            'saved_reopened': True, 'outside_pixels': 0, 'visual_status': 'REQUIRES_IMAGE_INSPECTION',
            'original_frames_pt': {m: list(original_frames[m][:5]) for m in members},
            'reopened_frames_pt': {m: list(reopened_frames[m][:5]) if m in reopened_frames else None for m in members},
            'crop': path.name, 'sha256': digest(path), 'render_sha256': render_hash}


def checked_sites(directory, result, irfile, page, target_ids=None):
    moves, pairs, unverified = edit_sites(irfile, target_ids, include_unverified=True)
    skipped = [row for row in result.get('unverified_original_site_groups', []) if row['page']==page]
    if skipped != [{'page':page,**row} for row in unverified]:
        raise ValueError('unverified joint-edit scope is missing or changed')
    expected = {('move-clear', (name,)) for name in moves} | {
        ('delete-together', members) for members in pairs}
    rows = [row for row in result.get('original_sites', []) if row['page'] == page]
    seen = set()
    render_rows = {}
    if rows:
        geometry, canvas = site_geometry(irfile)
        before_path = Path(directory)/f'page-{page:02d}-before.png'
        if not before_path.is_file() or result.get('original_site_previews',{}).get(str(page))!=digest(before_path):
            raise ValueError('missing bound Office before render for original-site verification')
        with Image.open(before_path) as img:
            if img.size!=canvas:raise ValueError('original-site before render size mismatch')
            before = np.asarray(img.convert('RGB'))
    for row in rows:
        action, members = row['action'], tuple(row['members'])
        identity = action, members
        filename = site_file(page, members, action)
        path = Path(directory) / filename
        if (identity not in expected or identity in seen or row.get('saved_reopened') is not True
                or type(row.get('outside_pixels')) is not int or row['outside_pixels'] != 0
                or row['crop'] != filename or not path.is_file() or row['sha256'] != digest(path)):
            raise ValueError('missing or mismatched original-site observation')
        window = observation_window(geometry, members)
        if tuple(row.get('window', ())) != window:
            raise ValueError('original-site crop does not cover the bound object and restoration extent')
        old, new = row.get('original_frames_pt', {}), row.get('reopened_frames_pt', {})
        if set(old) != set(members) or set(new) != set(members):
            raise ValueError('original-site edit lacks the reopened inventory of its members')
        for member in members:
            frame = old[member]
            if (not isinstance(frame,list) or len(frame)!=5 or not all(math.isfinite(v) for v in frame)
                    or any(abs(a-b)>.15 for a,b in zip(frame,geometry[member][0]))):
                raise ValueError('original-site frame does not match the bound IR')
            if action == 'delete-together':
                if new[member] is not None:
                    raise ValueError('original-site joint deletion left a member present')
                continue
            after = new[member]
            if not isinstance(after,list) or len(after)!=5 or not all(math.isfinite(v) for v in after) or min(after[2:4])<=0:
                raise ValueError('original-site move lacks a valid reopened frame')
            x,y,w,h = visible_box(after,after[4])
            left,top,right,bottom = (v*.75 for v in window)
            if x+w>left and x<right and y+h>top and y<bottom:
                raise ValueError('reopened picture still covers the original observation site')
        render_hash = row.get('render_sha256','')
        if len(render_hash)!=64 or any(ch not in '0123456789abcdef' for ch in render_hash):
            raise ValueError('original-site edit lacks its actual reopened render hash')
        full = Path(directory) / f'{render_hash}-reopen.png'
        if not full.is_file() or digest(full) != render_hash:
            raise ValueError('missing actual reopened render for original-site evidence')
        render_rows.setdefault(render_hash, []).append(row)
        seen.add(identity)
    if seen != expected:
        raise ValueError('Office worker did not supply all original-site observations')
    for render_hash, group in render_rows.items():
        # Batch moves use disjoint picture sites; the union covers only the actual
        # old and new footprints of the members verified above.
        with Image.open(Path(directory)/f'{render_hash}-reopen.png') as image:
            full_image = image.convert('RGB')
        if full_image.size != canvas:
            raise ValueError('original-site reopened render size mismatch')
        delta = (before != np.asarray(full_image)).any(axis=2)
        outside = delta.copy()
        pw,ph=canvas
        for row in group:
            left, top, right, bottom = row['window']
            with Image.open(Path(directory)/row['crop']) as crop:
                if (crop.size != (right-left,bottom-top)
                        or ImageChops.difference(full_image.crop(row['window']),crop.convert('RGB')).getbbox()):
                    raise ValueError('original-site crop pixels do not match the actual reopened render')
            if not delta[top:bottom,left:right].any():
                raise ValueError('no visible edit in the original observation site')
            for member in row['members']:
                margin = geometry[member][2]
                for key in ('original_frames_pt','reopened_frames_pt'):
                    frame = row[key][member]
                    if frame is None:
                        continue
                    x,y,w,h = visible_box([v/.75 for v in frame[:4]],frame[4])
                    x0,y0=max(0,math.floor(x)-margin),max(0,math.floor(y)-margin)
                    x1,y1=min(pw,math.ceil(x+w)+margin),min(ph,math.ceil(y+h)+margin)
                    if x1>x0 and y1>y0:
                        for peer in group:
                            if peer is row:
                                continue
                            px,py,pr,pb=peer['window']
                            if x0<pr and x1>px and y0<pb and y1>py:
                                raise ValueError('another batch edit intersects the original-site observation')
                        if key=='reopened_frames_pt' and not delta[y0:y1,x0:x1].any():
                            raise ValueError('no visible evidence at the reported in-slide move destination')
                        outside[y0:y1,x0:x1]=False
        if outside.any():
            raise ValueError('reopened render changed pixels outside the original-site edit transaction')
    return rows
