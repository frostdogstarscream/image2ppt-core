"""Shared saved-plan contract; importing never calls a model."""
import math
import re

SCHEMA = 'SegmentationPlan/1'
KINDS = {'module', 'text', 'graphic', 'shape', 'decoration', 'table', 'chart'}
ACTIONS = {'split_children', 'ocr', 'keep_pixels', 'native_shape_candidate', 'table_candidate', 'chart_candidate'}


def validate_plan(plan):
    if plan.get('schema') not in (SCHEMA, 'SegmentationPlan/2') or not isinstance(plan.get('summary'), str):
        raise ValueError('invalid schema or summary')
    rows = plan.get('regions')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 72:
        raise ValueError('region count outside 1..72')
    if not isinstance(plan.get('risks'), list) or any(not isinstance(r, str) for r in plan['risks']):
        raise ValueError('risks must be strings')
    by_id = {}; warnings = []
    for row in rows:
        if not isinstance(row, dict) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,31}', row.get('id','')):
            raise ValueError('invalid region id')
        if row['id'] in by_id: raise ValueError('duplicate region id')
        by_id[row['id']] = row
        if row.get('kind') not in KINDS or row.get('action') not in ACTIONS:
            raise ValueError('invalid region kind/action')
        if any(not isinstance(row.get(k),str) or len(row[k])>300 for k in ('label','reason')):
            raise ValueError('invalid label/reason')
        box = row.get('bbox')
        if not isinstance(box,list) or len(box)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in box):
            raise ValueError('invalid coordinates')
        if not 0<=box[0]<box[2]<=1000 or not 0<=box[1]<box[3]<=1000:
            raise ValueError('coordinates must be normalized xyxy within the source')
    if sum(r['kind']=='module' for r in rows)>12 or sum(r['kind']!='module' for r in rows)>60:
        raise ValueError('module/object budget exceeded')
    if plan['schema'] == 'SegmentationPlan/2':
        parents = {r.get('parent_id') for r in rows}
        for row in rows:
            if row['action'] == 'keep_pixels' and row['id'] not in parents:
                prompt = row.get('sam_prompt_en')
                if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 180 or not re.search(r'[A-Za-z]', prompt):
                    raise ValueError('SAM3 requires a short English visual prompt')
                if row.get('segmentation_mode') not in ('object', 'composite'):
                    raise ValueError('SAM3 requires object or composite mode')
    for row in rows:
        parent=row.get('parent_id');seen={row['id']};depth=0
        while parent is not None:
            if parent not in by_id or by_id[parent]['kind']!='module' or parent in seen:
                raise ValueError('invalid or cyclic parent relationship')
            seen.add(parent);depth+=1
            if depth>2:raise ValueError('hierarchy deeper than three levels')
            parent=by_id[parent].get('parent_id')
        if row.get('parent_id'):
            a=row['bbox'];b=by_id[row['parent_id']]['bbox']
            if a[0]<b[0] or a[1]<b[1] or a[2]>b[2] or a[3]>b[3]:
                warnings.append({'code':'child_outside_parent','id':row['id'],'parent':row['parent_id']})
    return warnings

