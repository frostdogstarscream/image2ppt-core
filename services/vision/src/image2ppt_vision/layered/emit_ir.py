"""SlideIR emission for the layered pipeline.

Layer order: cleaned background < native shapes / tables / cutouts (perception
order, larger first) < native text. Every OCR line has exactly one owner: a
text or table node, a cutout that still shows it, or the cleaned background.
"""
import shutil
import statistics
from pathlib import Path

from PIL import Image

from ..storage import digest, write_json


def _inside(box, outer):
    cx, cy = box[0]+box[2]/2, box[1]+box[3]/2
    return outer[0] <= cx <= outer[0]+outer[2] and outer[1] <= cy <= outer[1]+outer[3]


def _clip(box, W, H):
    x, y, w, h = box
    x, y = min(max(0., x), W-1), min(max(0., y), H-1)
    return [round(x, 3), round(y, 3), round(max(1., min(w, W-x)), 3), round(max(1., min(h, H-y)), 3)]


def _paragraph_node(pid, styles, ids, align, W, H, scale):
    lines = [styles[i] for i in ids]
    first = lines[0]
    size_px = statistics.median(s['size_px'] for s in lines)
    ox = first['ink_bbox'][0]-first['origin'][0]
    left = min(s['ink_bbox'][0] for s in lines)-ox
    right = max(s['ink_bbox'][0]+s['ink_bbox'][2] for s in lines)+ox
    tops = [s['ink_bbox'][1] for s in lines]
    pitch = statistics.median(b-a for a, b in zip(tops, tops[1:])) if len(lines) > 1 else None
    height = (pitch*(len(lines)-1) if pitch else 0)+size_px*1.45
    slack = max(4., size_px*.6)
    if align == 'center':
        cx = (left+right)/2
        box = [cx-(right-left)/2-slack, first['origin'][1], right-left+2*slack, height]
    elif align == 'right':
        box = [left-slack, first['origin'][1], right-left+slack, height]
    else:
        box = [first['origin'][0], first['origin'][1], right-first['origin'][0]+slack, height]
    node = {'id': pid, 'type': 'text', 'bbox': _clip(box, W, H), 'evidence_ids': list(ids),
            'text': '\n'.join(s['text'] for s in lines),
            'font': {'latin': first['family'], 'ea': first['family'], 'size_pt': round(size_px*scale*.75, 3)},
            'bold': bool(first['bold']), 'color': first['color'].upper(), 'align': align, 'wrap': False}
    if pitch:
        node['line_spacing_pt'] = round(pitch*scale*.75, 3)
    if len(lines) == 1 and first.get('runs'):
        node['runs'] = first['runs']
    return node


def emit(perception_dir, perception, styles, paragraphs, output, *, notes=(), source_crops=()):
    perception_dir, output = Path(perception_dir), Path(output)
    if output.exists():
        raise FileExistsError('new output required')
    (output/'assets').mkdir(parents=True)
    (output/'source').mkdir()
    shutil.copyfile(perception_dir/'source.png', output/'source/page.png')
    W, H = perception['image_size']
    scale = 1280/W
    ocr = {t['id']: t for t in perception['ocr']}
    records = {r['id']: r for r in perception['text']}
    native_ids = {i for p in paragraphs for i in p['ids']}
    table_ids = {i for t in perception['tables'] for i in t['owned_text_ids']}
    boxes = {i: t['bbox'] for i, t in ocr.items()}
    evidence, icon_parts = [], []
    for t in perception['ocr']:
        split = records.get(t['id'], {}).get('icon_text_split') if t['id'] in native_ids else None
        evidence.append({'id': t['id'], 'kind': 'text', 'bbox': _clip(t['bbox'], W, H),
                         'text': split['native'] if split else t['text']})
        if split:
            part, box = t['id']+'-icon', records[t['id']]['icon_candidates'][0]
            evidence.append({'id': part, 'kind': 'text', 'bbox': _clip(box, W, H), 'text': split['lead']+split['trail'],
                             'description': f'OCR read "{t["text"]}"; these characters are the adjacent icon and stay in its pixels.'})
            icon_parts.append(part); boxes[part] = box
    retained = [i for i in ocr if i not in native_ids and i not in table_ids and i not in source_crops]+icon_parts
    nodes, assets, z = [], [], 0
    claimed = set()
    for element in perception['elements']:
        eid = element['id']
        kind = 'shape' if element['kind'] == 'native_shape' else 'image'
        evidence.append({'id': 'ev-'+eid, 'kind': kind, 'bbox': _clip(element['bbox'], W, H),
                         'description': f'{element["source"]} {element["kind"]}'})
        node = {'id': eid, 'type': kind, 'bbox': _clip(element['bbox'], W, H), 'z': z, 'evidence_ids': ['ev-'+eid]}
        if kind == 'shape':
            node.update(geometry=element['geometry'], fill=element['fill'])
            if element.get('corner_radius_px'):
                node['corner_radius_px'] = min(element['corner_radius_px'], min(node['bbox'][2:])/2)
            if element.get('fill_gradient'):
                node['fill_gradient'] = element['fill_gradient']
            if element.get('line'):
                node['line'] = {'color': element['line']['color'],
                                'width_pt': round(element['line']['width_px']*scale*.75, 3)}
        else:
            relative = f'assets/{eid}.png'
            shutil.copyfile(perception_dir/element['asset'], output/relative)
            if digest(output/relative) != element['asset_sha256']:
                raise ValueError('cutout integrity mismatch: '+eid)
            inside = [i for i in retained if i not in claimed and _inside(boxes[i], element['bbox'])]
            claimed.update(inside)
            assets.append({'id': eid, 'path': relative, 'purpose': 'module', 'sha256': digest(output/relative),
                           'contains_evidence_ids': ['ev-'+eid, *inside],
                           'description': 'RGBA cutout from the text-free page; alpha from segmentation mask.'})
            node['asset_id'] = eid
            node['evidence_ids'] = []
        nodes.append(node); z += 1
    if source_crops:
        with Image.open(perception_dir/'source.png') as image:
            page = image.convert('RGB')
        for tid in source_crops:
            x, y, w, h = records[tid]['ink_bbox']
            x0, y0 = max(0, round(x)-3), max(0, round(y)-3)
            x1, y1 = min(W, round(x+w)+3), min(H, round(y+h)+3)
            box = [x0, y0, x1-x0, y1-y0]
            relative = f'assets/crop-{tid}.png'
            page.crop((x0, y0, x1, y1)).save(output/relative)
            assets.append({'id': 'crop-'+tid, 'path': relative, 'purpose': 'module', 'sha256': digest(output/relative),
                           'contains_evidence_ids': [tid],
                           'description': 'Source pixels for a doubtful OCR line (likely an icon), not re-typed.'})
            nodes.append({'id': 'crop-'+tid, 'type': 'image', 'bbox': box, 'z': z, 'evidence_ids': [], 'asset_id': 'crop-'+tid})
            z += 1
    for table in perception['tables']:
        tid = table['id']
        evidence.append({'id': 'ev-'+tid, 'kind': 'shape', 'bbox': _clip(table['bbox'], W, H),
                         'description': 'ruled grid table region'})
        xs, ys = table['xs'], table['ys']
        th = H*scale
        heights = [(b-a)*scale for a, b in zip(ys, ys[1:])]
        heights[-1] += table['bbox'][3]*th/H-sum(heights)
        sizes = [styles[i]['size_pt'] for i in table['owned_text_ids'] if i in styles] or [10.]
        family = next((styles[i]['family'] for i in table['owned_text_ids'] if i in styles), '微软雅黑')
        nodes.append({'id': tid, 'type': 'table', 'bbox': _clip(table['bbox'], W, H), 'z': z,
                      'evidence_ids': ['ev-'+tid, *table['owned_text_ids']], 'rows': table['rows'],
                      'font': {'latin': family, 'ea': family, 'size_pt': statistics.median(sizes)},
                      'column_tracks': [{'mode': 'fixed', 'value': round((b-a)*scale, 3)} for a, b in zip(xs, xs[1:])],
                      'row_heights': [round(v, 4) for v in heights], 'cell_styles': table['cell_styles']})
        z += 1
    for k, paragraph in enumerate(paragraphs):
        node = _paragraph_node(f'p{k:03d}', styles, paragraph['ids'], paragraph['align'], W, H, scale)
        node['z'] = z; z += 1
        node['source_text_bbox'] = _clip(ocr[paragraph['ids'][0]]['bbox'], W, H)
        nodes.append(node)
    background = {'color': perception['background'].get('solid_color') or _median_color(perception_dir/'background.png')}
    rest = [i for i in retained if i not in claimed]
    if not perception['background'].get('solid_color'):
        shutil.copyfile(perception_dir/'background.png', output/'assets/background.png')
        assets.append({'id': 'background', 'path': 'assets/background.png', 'purpose': 'cleaned_background',
                       'sha256': digest(output/'assets/background.png'), 'contains_evidence_ids': rest,
                       'description': 'LaMa-repaired page with native text, shapes and cutouts removed.'})
        background['asset_id'] = 'background'
        unresolved = []
    else:
        unresolved = rest
    ir = {'schema_version': '0.1', 'source_image': 'source/page.png', 'source_sha256': digest(output/'source/page.png'),
          'canvas': {'source_px': [W, H], 'target_px': [1280, round(H*scale)]}, 'background': background,
          'review_status': 'needs_review',
          'review_notes': ['Layered v3 reconstruction: cleaned background, native text/shapes, RGBA cutouts.',
                           'Fonts are installed candidates fitted to source ink, not identified originals.', *notes],
          'evidence': evidence, 'nodes': nodes, 'assets': assets, 'groups': []}
    if unresolved:
        ir['unresolved_evidence_ids'] = unresolved
    write_json(output/'slideir.json', ir)
    return ir


def _median_color(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('RGB')).reshape(-1, 3)
    return ''.join(f'{int(v):02X}' for v in np.median(pixels, 0))
