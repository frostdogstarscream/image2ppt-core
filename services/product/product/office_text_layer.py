"""Measure OCR fields on an actual Office slide with its native text cleared."""
from pathlib import Path
import json

import numpy as np
from PIL import Image

from .contracts import intersection
from .materialize import digest
from .parallel_office import leaf_shapes


def export_text_layer(slide, ir_path, output, width, height):
    """Use a disposable read-only presentation. The caller closes it without saving."""
    ir_path, output = Path(ir_path), Path(output)
    ir = json.loads(ir_path.read_text('utf8'))
    folder = ir_path.parent
    text_nodes = [n for n in ir['nodes'] if n['type'] == 'text']
    native = {n['id'] for n in text_nodes}
    found = set()
    for shape in leaf_shapes(slide.Shapes):
        name = str(shape.Name)
        if name in native and shape.HasTextFrame and shape.TextFrame.HasText:
            found.add(name)
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
    if source.shape != cleared.shape:
        raise ValueError('text-layer inspection requires the source pixel coordinate system')
    for line in evidence['lines']:
        oid = line['id']
        x, y, w, h = map(round, line['bbox'])
        selected = sorted(found.intersection(by_line.get(oid, [])))
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
        score = None
        if 0 <= x < x+w <= width and 0 <= y < y+h <= height:
            old = source[y:y+h,x:x+w].astype(float)
            new = cleared[y:y+h,x:x+w].astype(float)
            border = np.median(np.concatenate([old[0],old[-1],old[:,0],old[:,-1]]),axis=0)
            strong = np.max(np.abs(old-border),axis=2) >= 40
            if strong.sum() >= 8:
                score = round(float((np.abs(old[strong]-new[strong]).max(axis=1)<=14).mean()),3)
        result['fields'].append({'ocr_id':oid,'text':line['text'],'bbox':line['bbox'],
            'representation':disposition,'source_object_ids':by_line.get(oid, []),
            'native_ids':selected,'source_ink_similarity':score})
    result.update(status='MEASURED', inventory_sha256=digest(inventory), source_sha256=digest(source_file))
    return result
