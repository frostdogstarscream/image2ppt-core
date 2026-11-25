"""Exercise real PowerPoint group selection and an individual member's solid fill."""
import shutil

import numpy as np
from PIL import Image

from .materialize import digest
from .office_sites import clear_position, observation_window, record_site, visible_box


def group_edit_targets(ir, target_ids=None):
    nodes = {n['id']: n for n in ir['nodes']}
    groups = []
    for group in ir.get('groups', []):
        ids = group['member_ids']
        if target_ids is not None and group['id'] not in target_ids and not set(ids).issubset(target_ids):
            continue
        if not any(nodes[i]['type'] in ('image', 'text') for i in ids):
            continue
        child = next((i for i in ids if nodes[i]['type'] == 'shape' and nodes[i].get('fill')
                      and not nodes[i].get('fill_gradient') and nodes[i].get('fill_opacity', 1) > 0), None)
        groups.append((group['id'], list(ids), child))
    return groups


def probe_group(app, pptx, output, page, target, before, original, signature, verify_media, margins, geometry):
    from .parallel_office import _px_window
    group_id, members, child_id = target
    ph, pw = before.shape[:2]
    records = []
    observations = []
    for action in (('group-move', 'member-recolor') if child_id else ('group-move',)):
        stem = f'{page:02d}-{group_id}-{action}'
        copy = output / (stem + '.pptx')
        rendered = output / (stem + '-reopen.png')
        shutil.copyfile(pptx, copy)
        deck = app.Presentations.Open(str(copy), ReadOnly=False, Untitled=False, WithWindow=False)
        try:
            slide = deck.Slides(page)
            sw, sh = float(deck.PageSetup.SlideWidth), float(deck.PageSetup.SlideHeight)
            sx, sy = pw / sw, ph / sh
            group = slide.Shapes.Item(group_id)
            if group.Type != 6 or {str(s.Name) for s in group.GroupItems} != set(members):
                raise ValueError('PowerPoint group membership differs from compiled scene')
            if action == 'group-move':
                old = [float(group.Left), float(group.Top), float(group.Width), float(group.Height)]
                pad = max(margins.get(i, 4) for i in members) / min(sx, sy)
                x0, y0, x1, y1 = observation_window(geometry, members)
                site = (x0/sx, y0/sy, x1/sx, y1/sy)
                left, top = clear_position([*old, float(group.Rotation)], site, sw, sh, pad + 3)
                group.Left, group.Top = left, top
                dx, dy = left - old[0], top - old[1]
            else:
                fill = group.GroupItems.Item(child_id).Fill
                original_color = int(fill.ForeColor.RGB)
                color = 0x2363CD if original_color != 0x2363CD else 0xD36C24
                fill.ForeColor.RGB = color
            deck.Save()
        finally:
            deck.Close()
        verify_media(copy)
        deck = app.Presentations.Open(str(copy), ReadOnly=True, Untitled=False, WithWindow=False)
        try:
            slide = deck.Slides(page)
            group = slide.Shapes.Item(group_id)
            if group.Type != 6 or {str(s.Name) for s in group.GroupItems} != set(members):
                raise ValueError('PowerPoint lost native group after save/reopen')
            actual = signature(slide)
            if set(actual) != set(original):
                raise ValueError('group edit changed the slide inventory')
            changed = members if action == 'group-move' else [child_id]
            if action == 'member-recolor' and int(group.GroupItems.Item(child_id).Fill.ForeColor.RGB) != color:
                raise ValueError('member recolor did not persist')
            allowed = np.zeros((ph, pw), dtype=bool)
            for name, old in original.items():
                expected = list(old[:5])
                if action == 'group-move' and name in members:
                    expected[0] += dx
                    expected[1] += dy
                if any(abs(a-b) > .12 for a,b in zip(actual[name][:5], expected)) or actual[name][5:] != old[5:]:
                    raise ValueError('group edit changed sibling or failed to move every member')
                if name in changed:
                    for frame in (old, actual[name]):
                        window = _px_window([visible_box(frame,frame[4])], sx, sy, pw, ph, margins.get(name,4))
                        x, y, right, bottom = window
                        allowed[y:bottom, x:right] = True
            slide.Export(str(rendered), 'PNG', pw, ph)
        finally:
            deck.Close()
        with Image.open(rendered) as image:
            after = np.asarray(image.convert('RGB'))
        diff = (after != before).any(2)
        if not diff.any() or (diff & ~allowed).any():
            raise ValueError('group edit has no visible effect or changed pixels outside its members')
        record = {'page':page, 'id':group_id if action == 'group-move' else child_id,
                  'group_id':group_id, 'members':members, 'action':action, 'saved_reopened':True,
                  'outside_pixels':0, 'changed_pixels':int(diff.sum()), 'render':rendered.name,
                  'render_sha256':digest(rendered), 'edited_pptx_sha256':digest(copy)}
        if action == 'group-move':
            record.update(offset_pt=[dx,dy])
        else:
            record.update(initial_fill_rgb_com=original_color, reopened_fill_rgb_com=color)
        site = record_site(output,page,changed,action,observation_window(geometry,changed),after,rendered,original,actual)
        site['group_id'] = group_id
        if action == 'member-recolor':
            site.update(initial_fill_rgb_com=original_color, reopened_fill_rgb_com=color)
        observations.append(site)
        records.append(record)
    return records, observations
