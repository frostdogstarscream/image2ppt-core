"""Read-only edit proposals from the current evidence; no proposal owns source pixels."""
import json
from collections import Counter
from pathlib import Path
from statistics import median

from ..contracts import intersection
from ..materialize import digest, write
from ..parallel_store import key
from ..scorecard import inspect_pptx


ACTIONS = {
    'text': ['replace_text', 'move'], 'surface': ['recolor', 'move', 'delete'],
    'asset': ['replace_image', 'move', 'resize', 'delete'],
    'table': ['replace_cell'], 'chart': ['edit_data_if_source_values_known'],
    'group': ['move_together'], 'raster': ['retain_appearance'],
}
ROLES = {
    'card_shape': 'surface', 'ribbon_shape': 'surface', 'badge_shape': 'surface', 'divider': 'surface',
    'photo': 'asset', 'logo': 'asset', 'icon': 'asset', 'chart': 'chart', 'table': 'table',
    'title': 'text', 'section_header': 'text', 'body_text': 'text', 'background_scenery': 'raster',
}


def _style_families(pages):
    """Group comparable roles at comparable rendered scales; supply suggestions, not new style values."""
    families = []
    for page in pages:
        for style in page.get('styles', []):
            role = style['role']
            height_fraction = style['size_px'] / page['image_size'][1]
            rgb = bytes.fromhex(style['color'])
            family = next((f for f in families if f['role'] == role and f['font_ea'] == style['ea']
                           and f['font_latin'] == style['latin']
                           and max(abs(a-b) for a,b in zip(f['_rgb'], rgb)) <= 12
                           and abs(f['_height'] - height_fraction) <= .09 * min(f['_height'], height_fraction)), None)
            member = {'page': page['page'], 'style_id': style['id'], 'source_sha256': page['source_sha256'],
                      'scene_sha256': page['scene_sha256'], 'relative_size': height_fraction, 'bold': style['bold']}
            if family is None:
                family = {'role': role, 'font_ea': style['ea'], 'font_latin': style['latin'],
                          '_rgb': rgb, '_height': height_fraction, 'members': []}
                families.append(family)
            family['members'].append(member)
    res = []
    for f in families:
        members = f['members']
        if len({m['source_sha256'] for m in members}) < 2:
            continue
        weights = {m['bold'] for m in members}
        res.append({'id': 'style-'+key(members)[:12], 'role': f['role'], 'ea': f['font_ea'], 'latin': f['font_latin'],
                    'color_sample': f['_rgb'].hex().upper(), 'members': members,
                    'median_relative_size': median(m['relative_size'] for m in members),
                    'weight': next(iter(weights)) if len(weights) == 1 else None,
                    'weight_disagreement': len(weights) > 1})
    return res


def _page_objects(graph, inventory):
    objects = []
    styles = {s.id: s.role for s in graph.text.styles}
    relations = [r.model_dump() for r in graph.relations]
    for kind, source in [('text', graph.text.texts), ('surface', graph.surface.shapes), ('asset', graph.asset.objects)]:
        for obj in source:
            record = {'id': obj.id, 'bbox': obj.bbox, 'kind': kind, 'owner': graph.ownership[obj.id],
                      'selected': obj.id in graph.selected, 'edit_actions': ACTIONS[kind]}
            if kind == 'text':
                members = graph.text.members.get(obj.id, [])
                record.update(role=styles[obj.style], text=obj.text, style_id=obj.style, ocr_line_ids=members,
                              emphasis=[span.model_dump() for span in obj.spans])
                if len(members) > 1:
                    record['edit_actions'] = ['replace_paragraph', 'move']
            elif kind == 'asset':
                record.update(role=obj.kind, layers=obj.layers)
            else:
                record.update(role=obj.geometry, fill=obj.fill)
            objects.append(record)

    # Inventory IDs are in a separate namespace. Matching is strictly geometric, by class and
    # unique strong overlap. Ambiguous proposals remain explicit, never become graph IDs.
    refs = {}
    observed = list(objects)
    elements = inventory.get('elements', [])
    for i, el in enumerate(elements):
        kind = ROLES.get(el.get('category'), 'raster')
        bbox = el.get('bbox', [])
        if len(bbox) != 4 or min(bbox[2:]) <= 0:
            continue
        candidates = []
        for obj in observed:
            if obj['kind'] != kind:
                continue
            area = intersection(obj['bbox'], bbox)
            if area >= .8 * max(bbox[2]*bbox[3], obj['bbox'][2]*obj['bbox'][3]):
                candidates.append(obj)
        eid = el.get('id') or f'row-{i+1}'
        if len(candidates) == 1:
            linked = candidates[0]
            linked.setdefault('inventory_ids', []).append(eid)
            reference = linked['id']
        else:
            reference = f'inventory/{i+1}/{eid}'
            objects.append({'id': reference, 'kind': kind, 'role': el.get('category'), 'bbox': bbox,
                            'owner': 'UNVERIFIED', 'selected': False, 'edit_actions': ACTIONS[kind],
                            'inventory_ids': [eid], 'possible_matches': [s['id'] for s in candidates]})
        refs.setdefault(eid, []).append(reference)
    for el in elements:
        parent = refs.get(el.get('parent_id'), [])
        child = refs.get(el.get('id'), [])
        if len(parent) == len(child) == 1 and child != parent:
            relations.append({'kind': 'contains', 'source': parent[0], 'target': child[0], 'status': 'PROPOSED',
                              'evidence': ['page_structure_parent']})
    return objects, relations


def get_editable_draft(service, state, current, request):
    """Return current evidence directly to the agent, also registering a complete immutable snapshot."""
    pages = []
    for page, _ in enumerate(request['sources'], 1):
        source = service.source(state, page)
        source_sha = digest(source)
        info = {'page': page, 'source_sha256': source_sha}
        pages.append(info)
        row = current['pages'][page-1] if len(current['pages']) >= page else None
        if not row:
            info.update(status='MISSING_SCENE_EVIDENCE', objects=[], relations=[])
            continue
        graph = service.graph(row)
        if graph.source_sha256 != source_sha:
            raise ValueError(f'page {page} scene refers to different source pixels')
        inventory = {}
        inventory_file = current.get('page_structure', {}).get(str(page))
        if inventory_file:
            inventory = json.loads(Path(inventory_file).read_text('utf8'))
            if inventory['source_sha256'] != source_sha:
                raise ValueError(f'page {page} structure inventory changed source')
            info['inventory_sha256'] = digest(inventory_file)
        objects, relations = _page_objects(graph, inventory)
        info.update(status='DRAFT', image_size=graph.image_size, scene_sha256=digest(row['scene']),
                    objects=objects, relations=relations, styles=[s.model_dump() for s in graph.text.styles])
    draft = {'schema': 'EditableDeckDraft/1', 'scene_version': current['version'], 'pages': pages,
             'style_families': _style_families(pages), 'suggestions_grant_pixel_ownership': False}
    draft_sha = key(draft)
    file = state.root/'drafts'/(draft_sha[:24]+'.json')
    if file.is_file():
        if key(json.loads(file.read_text('utf8'))) != draft_sha:
            raise ValueError('draft evidence changed')
    else:
        write(file, draft)
    return {**draft, 'evidence': state.register(file, 'editable_deck_draft')}


def delivery_page_summaries(current, office=None):
    """Report exact exported object counts and only evidence bound to the compiled PPTX bytes."""
    composition = current['composition']
    pptx = composition['artifacts']['pptx']
    sha = digest(pptx)
    if composition['files'].get(str(pptx)) != sha:
        raise ValueError('delivery description requires bound PPTX')
    office = office or {}
    app = office.get('office', {})
    bound = office.get('pptx_sha256') == app.get('pptx_sha256') == sha
    summaries = []
    for page, entry in enumerate(current['pages'], 1):
        ir = json.loads(Path(entry['ir']).read_text('utf8'))
        if composition['files'].get(str(entry['ir'])) != digest(entry['ir']):
            raise ValueError('delivery description requires bound page IR')
        measured = inspect_pptx(pptx, tuple(ir['canvas']['source_px']), page)
        classes = Counter(obj['class'] for obj in measured['objects'])
        rasters = [{'id': obj.get('name'), 'bbox': obj['bbox'], 'kind': obj['class']} for obj in measured['objects']
                   if obj['class'] in {'asset','opaque_picture','duplicate_picture','plate'}]
        records = [{'id': r.get('id'), 'action': r.get('action'), 'saved_reopened': r.get('saved_reopened'),
                    'outside_pixels': r.get('outside_pixels')} for r in app.get('records', [])
                   if bound and r.get('page') == page]
        failures = current.get('regional_failures', {}).get(str(page), [])
        layer = app.get('text_layers', {}).get(str(page)) if bound else None
        if layer:
            reasons = {r['object_id']: r.get('reason') or r.get('failure') for r in failures if r.get('object_id')}
            layer = {**layer, 'fields': [
                {**field, 'reconstruction_failures': [reasons[oid] for oid in field.get('source_object_ids', []) if oid in reasons]}
                for field in layer['fields']]}
        summaries.append({'page': page, 'source_sha256': entry['snapshot']['source_sha256'], 'pptx_sha256': sha,
                          'object_counts': dict(classes), 'text_boxes': classes['text'], 'tables': classes['table'],
                          'charts': classes['chart'], 'shapes': classes['shape'], 'groups_in_ir': len(ir.get('groups', [])),
                          'raster_objects': rasters, 'raster_backing_present': bool(classes['plate']),
                          'perception_failures': entry.get('failures', []), 'editable_recall': None,
                          'office_status': office.get('status', 'UNKNOWN') if bound else 'UNKNOWN',
                          'text_layer_inspection': layer,
                          'observed_edit_actions': records,
                          'region_failures': failures})
    return summaries
