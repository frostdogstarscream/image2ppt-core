"""Measure OCR fields on an actual Office slide with its native text cleared."""
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image

from .contracts import intersection
from .materialize import digest
from .parallel_office import leaf_shapes


def surviving_edges(old, new):
    # A white letter erased onto white paper shares its former RGB value.
    # Only count the old pattern when a local stroke edge also survives.
    original = old - cv2.GaussianBlur(old,(0,0),1.3)
    after = new - cv2.GaussianBlur(new,(0,0),1.3)
    strong = np.abs(original).max(2) > 16
    if strong.sum() < 8:
        return None
    direction = original[strong]
    magnitude = np.linalg.norm(direction,axis=1)
    projected = (direction * after[strong]).sum(1)/magnitude
    return round(float((projected > np.maximum(4,magnitude*.15)).mean()),3)


def export_text_layer(slide, ir_path, output, width, height):
    """Use a disposable read-only presentation. The caller closes it without saving."""
    ir_path, output = Path(ir_path), Path(output)
    ir = json.loads(ir_path.read_text('utf8'))
    folder = ir_path.parent
    text_nodes = [n for n in ir['nodes'] if n['type'] == 'text']
    native = {n['id'] for n in text_nodes}
    found = set()
    actual_text = {}
    tables = {n['id']:n for n in ir['nodes'] if n['type'] == 'table'}
    cells = {}
    for shape in leaf_shapes(slide.Shapes):
        name = str(shape.Name)
        if name in tables and shape.HasTable:
            x,y,w,h = tables[name]['bbox']
            for row in range(1,shape.Table.Rows.Count+1):
                for column in range(1,shape.Table.Columns.Count+1):
                    cell = shape.Table.Cell(row,column).Shape
                    value = str(cell.TextFrame.TextRange.Text)
                    if not value.strip():
                        continue
                    cell_id = f'{name}/{row}/{column}'
                    cells[cell_id] = [x+(cell.Left-shape.Left)*w/shape.Width,
                                      y+(cell.Top-shape.Top)*h/shape.Height,
                                      cell.Width*w/shape.Width,cell.Height*h/shape.Height]
                    actual_text[cell_id] = value
                    cell.TextFrame.TextRange.Text = ''
        if name in native and shape.HasTextFrame and shape.TextFrame.HasText:
            found.add(name)
            actual_text[name] = str(shape.TextFrame.TextRange.Text)
            shape.TextFrame.TextRange.Text = ''
    if found != native:
        raise ValueError('native text inventory differs from the actual Office slide')
    slide.Export(str(output.resolve()), 'PNG', width, height)
    result = {'preview': output.name, 'preview_sha256': digest(output), 'hidden_text_objects': len(found),
              'fields': [], 'type_conflicts': [], 'status': 'INVENTORY_UNAVAILABLE'}
    source_file = folder/'source.png'
    inventory = folder/'runtime-ocr-protection.json'
    if not inventory.is_file() or not source_file.is_file():
        return result
    evidence = json.loads(inventory.read_text('utf8'))
    if evidence['source_sha256'] != digest(source_file):
        raise ValueError('OCR and source do not match in the text-layer inspection')
    scene = json.loads((folder/'scene.json').read_text('utf8'))
    graph_text = scene['text']
    by_line = {}
    for name, ids in graph_text['members'].items():
        for identifier in ids:
            by_line.setdefault(identifier, []).append(name)
    structure = folder/'page-structure.json'
    roles = json.loads(structure.read_text('utf8')).get('graphic_or_uncertain_ocr', {}) if structure.exists() else {}
    frames = folder/'embedded-ui-frames.json'
    ui = json.loads(frames.read_text('utf8')) if frames.exists() else []
    with Image.open(source_file) as image:
        source = np.asarray(image.convert('RGB'))
    with Image.open(output) as image:
        cleared = np.asarray(image.convert('RGB'))
    backing = scene.get('raster_backing') or {}
    plate = None
    if backing.get('path'):
        path = (folder/backing['path']).resolve()
        if not path.is_relative_to(folder.resolve()) or digest(path) != backing['sha256']:
            raise ValueError('text-layer inspection requires the bound background image')
        with Image.open(path) as image:
            plate = np.asarray(image.convert('RGBA'))
        if plate.shape[:2] != source.shape[:2]:
            raise ValueError('background dimensions differ from source')
        result['plate_sha256'] = backing['sha256']
    if source.shape != cleared.shape:
        raise ValueError('text-layer inspection requires the source pixel coordinate system')
    for line in evidence['lines']:
        oid = line['id']
        x, y, w, h = map(round, line['bbox'])
        selected = sorted(found.intersection(by_line.get(oid, [])))
        selected.extend(name for name,box in cells.items() if intersection(box,line['bbox']) >= .95*w*h
                        and ''.join(line['text'].split()) in ''.join(actual_text[name].split()))
        disposition = 'native' if selected else 'no_native_text'
        if oid in roles:
            disposition = roles[oid]['disposition']
            if disposition == 'graphic':
                conflicts = set(selected)
                for node in text_nodes:
                    if (node.get('text', '').strip() == line['text'].strip() and
                            intersection(node['bbox'], line['bbox']) >= .6*max(1, w*h)):
                        conflicts.add(node['id'])
                if conflicts:
                    result['type_conflicts'].append({'ocr_id':oid, 'wrong_text_objects':sorted(conflicts)})
        elif not selected and (line.get('is_logo') or any(
                left <= x and top <= y and x+w <= left+fw and y+h <= top+fh for left,top,fw,fh in ui)):
            disposition = 'picture_content'
        score = plate_score = cleared_edges = plate_edges = None
        if 0 <= x < x+w <= width and 0 <= y < y+h <= height:
            old = source[y:y+h,x:x+w].astype(float)
            new = cleared[y:y+h,x:x+w].astype(float)
            border = np.median(np.concatenate([old[0],old[-1],old[:,0],old[:,-1]]),axis=0)
            strong = np.max(np.abs(old-border),axis=2) >= 40
            if strong.sum() >= 8:
                score = round(float((np.abs(old[strong]-new[strong]).max(axis=1)<=14).mean()),3)
                cleared_edges = surviving_edges(old,new)
                if plate is not None:
                    patch = plate[y:y+h,x:x+w][strong]
                    matches = (np.abs(old[strong]-patch[:,:3]).max(axis=1) <= 14) & (patch[:,3] >= 240)
                    plate_score = round(float(matches.mean()),3)
                    pixels = plate[y:y+h,x:x+w]
                    alpha = pixels[:,:,3:4].astype(float)/255
                    plate_edges = surviving_edges(old,pixels[:,:,:3]*alpha+255*(1-alpha))
        result['fields'].append({'ocr_id':oid,'text':line['text'],'bbox':line['bbox'],
            'representation':disposition,'source_object_ids':by_line.get(oid, []),
            'native_ids':selected,'source_ink_similarity':score,'plate_ink_similarity':plate_score,
            'cleared_stroke_edges':cleared_edges,'plate_stroke_edges':plate_edges})
    blockers = []
    for field in result['fields']:
        if (roles.get(field['ocr_id'],{}).get('visual_role') == 'graphic'
                and field['representation'] == 'graphic' and not field['native_ids']):
            continue
        ids = field['native_ids']
        normalize = lambda value: ''.join(value.split())
        value = normalize(field['text'])
        actual = ''.join(normalize(actual_text[name]) for name in ids)
        if not ids:
            reason = 'MISSING_NATIVE_TEXT'
        elif not value or value not in actual:
            reason = 'CONTENT_REQUIRES_RECONCILIATION'
        elif max(field['cleared_stroke_edges'] or 0,field['plate_stroke_edges'] or 0) >= .5:
            reason = 'POSSIBLE_BACKING_TEXT'
        else:
            continue
        blockers.append({'ocr_id':field['ocr_id'], 'text':field['text'], 'reason':reason})
    result.update(status='MEASURED', inventory_sha256=digest(inventory), source_sha256=digest(source_file),
                  coverage_status='INCOMPLETE' if blockers or result['type_conflicts'] else (
                      'INVENTORY_COVERED' if result['fields'] else 'UNVERIFIED'), blockers=blockers)
    return result
