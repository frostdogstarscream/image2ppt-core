"""Track source-pixel dependencies between model-selected raster objects."""
import json

import numpy as np
from PIL import Image


def overlapping_ink(parent, child, output):
    left, top, width, height = map(int, parent['bbox'])
    cx, cy, cw, ch = map(int, child['bbox'])
    x0, y0 = max(left, cx), max(top, cy)
    x1, y1 = min(left + width, cx + cw), min(top + height, cy + ch)
    if x1 <= x0 or y1 <= y0:
        return 0
    with Image.open(output / parent['path']) as image:
        a = np.asarray(image.convert('RGBA'))[y0-top:y1-top, x0-left:x1-left, 3]
    with Image.open(output / child['path']) as image:
        b = np.asarray(image.convert('RGBA'))[y0-cy:y1-cy, x0-cx:x1-cx, 3]
    return int(((a > 128) & (b > 128)).sum())


def restore_dependencies(state, output):
    """Recover older plate provenance from recorded successful model actions."""
    selected = []
    for row in state.get('history', []):
        result = row.get('result', {})
        if not isinstance(result, dict) or 'error' in result:
            continue
        folder = output / f"turn-{row['turn']:03}"
        action_file = folder / 'action.json'
        if not action_file.exists():
            continue
        action = json.loads(action_file.read_text(encoding='utf-8'))
        name = action['action']
        if name == 'draft' and (folder / 'draft.json').exists():
            ir = json.loads((folder / 'draft.json').read_text(encoding='utf-8'))
            selected = [n['asset_id'] for n in ir['nodes'] if n['type'] == 'image']
        elif name == 'objects':
            selected = result['selected_objects']
        elif name == 'plate' and result.get('id') in state['assets']:
            supplied = action.get('arguments', {}).get('ir')
            cleared = selected if supplied is None else [
                n['asset_id'] for n in supplied['nodes'] if n['type'] == 'image']
            state['assets'][result['id']].setdefault('cleared_asset_ids', cleared.copy())
    plates = {a['sha256']: a for a in state['assets'].values() if 'cleared_asset_ids' in a}
    for asset in state['assets'].values():
        if 'requires_asset_ids' in asset:
            continue
        plate = plates.get(asset.get('cropped_from_sha256'))
        if plate is not None:
            asset['requires_asset_ids'] = [identifier for identifier in plate['cleared_asset_ids']
                if identifier != asset['id'] and overlapping_ink(asset, state['assets'][identifier], output) >= 12]


def dependency_conflicts(ir, state):
    selected = {n['asset_id']: n.get('z', 0) for n in ir['nodes'] if n['type'] == 'image'}
    conflicts = []
    for identifier, z in selected.items():
        for child in state['assets'][identifier].get('requires_asset_ids', []):
            if child not in selected or selected[child] <= z:
                conflicts.append({'parent': identifier, 'required_child_above_parent': child})
    return conflicts


def remove_text_plan(arguments, state):
    """Apply explicit model text ownership choices and invalidate derived crops."""
    import copy
    ir = copy.deepcopy(state['current_ir'])
    texts = {n['id']: n for n in ir['nodes'] if n['type'] == 'text'}
    selected = set(arguments.get('ids', []))
    region_id = arguments.get('within_inventory')
    if region_id is not None:
        x, y, width, height = state['visual_inventory'][region_id]['bbox']
        for identifier, node in texts.items():
            nx, ny, nw, nh = node.get('source_text_bbox', node['bbox'])
            if nx >= x and ny >= y and nx + nw <= x + width and ny + nh <= y + height:
                selected.add(identifier)
    if not selected or not selected <= texts.keys() or not arguments.get('reason'):
        raise ValueError('remove_text needs existing text IDs or within_inventory and a reason')
    invalid = []
    for node in ir['nodes']:
        if node['type'] != 'image':
            continue
        asset = state['assets'][node['asset_id']]
        ax, ay, aw, ah = asset['bbox']
        for identifier in selected.intersection(asset.get('cleaned_text_ids', [])):
            tx, ty, tw, th = texts[identifier].get('source_text_bbox', texts[identifier]['bbox'])
            if min(ax+aw, tx+tw) > max(ax, tx) and min(ay+ah, ty+th) > max(ay, ty):
                invalid.append(node['asset_id'])
                break
    ir['nodes'] = [n for n in ir['nodes'] if n['id'] not in selected
                   and not (n['type'] == 'image' and n['asset_id'] in invalid)]
    ir['review_notes'].append(arguments['reason'])
    return ir, {'raster_text_node_ids': sorted(selected), 'deselected_stale_assets': invalid,
                'instruction': 'Regenerate plate from original source. Re-extract affected visual modules '
                               'from the new plate, with those parent assets deselected.',
                'next_actions': ['plate', 'extract_many', 'objects', 'compile']}
