"""Reviewed relation inventory, independent of the objects surviving in a candidate.

This is a bounded Xuzhou source catalog, not an automatic detector. A caller may
provide another source-bound catalog for any canvas. Missing IR/evidence keeps
the declared inventory pending instead of silently removing required checks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path


SOURCE_HASHES = {
    '02': '3f1daa8db89fd43e9a595361fcc320283cf456eea867c9c2b9bcf54adc91d371',
    '03': 'd23931d294330686fdafbe9717443d93733456fabf518ac0ce9bb18ebb64e6cf',
    '04': '0bdd402aa3008667a44d1c2faddc2f7b0a7b449f0d519a9fbb96721a1ebec059',
    '08': '5b42f349c8990408c19591c18910c247af8d50c93fa1c46e250af236e61e80a5',
    '10': 'a730d74f496b3787e75bbca499b7d2a9b2434bf1d57e6bbdc04f3ba7ea12f224',
}

# Legacy tuples remain exposed by relation_validator for callers/tests.
REGIONS = {
    '02': [('s02-sidebar-label', 'label', [59, 198, 302, 258])],
    '08': [('s08-feedback-arrow-1', 'arrow', [588, 735, 628, 766]),
           ('s08-feedback-arrow-2', 'arrow', [929, 735, 972, 766]),
           ('s08-feedback-arrow-3', 'arrow', [1266, 735, 1307, 766]),
           ('s08-summary-chevrons', 'decoration', [316, 827, 365, 862])],
}


def _entry(identifier, kind, **kwargs):
    return {'id': identifier, 'kind': kind, 'repairable': False, **kwargs}


def default_catalog():
    pages = {p: {'source_sha256': h, 'elements': []} for p, h in SOURCE_HASHES.items()}
    for page, rows in REGIONS.items():
        for identifier, kind, bbox in rows:
            row = _entry(identifier, kind, bbox_xyxy=bbox, repairable=True,
                         segmentation='blue', direction='right' if kind == 'arrow' else None)
            if kind == 'label':
                row.update(text_target_id='paragraph-l007', text_bbox_xyxy=[93, 207, 248, 249])
            pages[page]['elements'].append(row)
    for name in ('crane', 'building', 'laptop', 'organization', 'analysis', 'handshake'):
        pages['03']['elements'].append(_entry('icon-'+name, 'ICON_PRESENCE',
            evidence_id='ev-icon-'+name, segmentation='blue'))
    for index in (1, 8, 15, 23):
        identifier = f'surface-04-{index:02d}'
        pages['04']['elements'].append(_entry(identifier, 'SURFACE_GEOMETRY',
            evidence_id='ev-'+identifier, object_id=identifier+'-body'))
    # Reviewed source arrows occupy only the gaps between the four layer cards.
    for i, box in enumerate(([815, 293, 858, 316], [815, 453, 858, 478], [815, 626, 858, 649]), 1):
        pages['04']['elements'].append(_entry(f's04-layer-arrow-{i}', 'ARROW_CONNECTIVITY',
            bbox_xyxy=box, direction='up', segmentation='blue'))
    for index in (1, 2, 3):
        identifier = f'surface-10-{index:02d}'
        pages['10']['elements'].append(_entry(identifier, 'SURFACE_GEOMETRY',
            evidence_id='ev-'+identifier, object_id=identifier+'-body'))
    return {'schema': 'RelationCatalog/1', 'coverage_scope': 'reviewed_key_elements_only', 'pages': pages}


def load_catalog(config=None):
    if config is None:
        return default_catalog()
    if isinstance(config, (str, Path)):
        value = json.loads(Path(config).read_text(encoding='utf-8-sig'))
    else:
        value = copy.deepcopy(config)
    if value.get('schema') != 'RelationCatalog/1' or not isinstance(value.get('pages'), dict):
        raise ValueError('Expected RelationCatalog/1 with explicit page entries')
    return value


def catalog_page(page, source_sha256, config=None, ir=None):
    catalog = load_catalog(config)
    declaration = copy.deepcopy(catalog['pages'].get(page, {'elements': []}))
    if declaration.get('source_sha256') not in (None, source_sha256):
        raise ValueError('Reviewed relation catalog belongs to another source image')
    if isinstance(ir, (str, Path)):
        ir = json.loads(Path(ir).read_text(encoding='utf-8-sig'))
    if ir is not None and ir.get('source_sha256') != source_sha256:
        raise ValueError('Relation IR belongs to another source image')
    evidence = {row['id']: row for row in (ir or {}).get('evidence', [])}
    nodes = {row['id']: row for row in (ir or {}).get('nodes', [])}
    rows = declaration.get('elements', [])
    ids = [row.get('id') for row in rows]
    if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError('Relation IDs must be unique nonempty strings')
    for row in rows:
        if 'bbox_xyxy' not in row:
            ev = evidence.get(row.get('evidence_id'))
            if not ev:
                row['evidence_missing'] = 'source_object_evidence_missing'
                continue
            x, y, w, h = ev['bbox']
            row['bbox_xyxy'] = [math.floor(x), math.floor(y), math.ceil(x+w), math.ceil(y+h)]
            row['source_evidence'] = copy.deepcopy(ev)
        if row.get('object_id'):
            node = nodes.get(row['object_id'])
            if not node:
                row['evidence_missing'] = 'expected_scene_object_missing'
            else:
                row['object_snapshot'] = {key: copy.deepcopy(node[key]) for key in ('id', 'type', 'bbox', 'z') if key in node}
        if ir is not None and row.get('text_target_id') and row['text_target_id'] not in nodes:
            row['evidence_missing'] = 'expected_scene_object_missing'
    return rows, {
        'catalog_sha256': hashlib.sha256(json.dumps(catalog, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        'coverage_scope': catalog.get('coverage_scope', 'declared_relations_only'),
        'not_applicable_reason': declaration.get('not_applicable_reason'),
    }
