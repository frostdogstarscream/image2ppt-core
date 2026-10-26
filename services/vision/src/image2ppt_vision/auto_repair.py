"""Deterministic, recorded repairs of model drafts.

The raw model response stays on disk unchanged. Repairs only fix geometry and
bookkeeping that the program can compute exactly; every operation is recorded
and the strict validators still decide whether the repaired value is accepted.
Boxes here are model-space normalized_1000_xyxy: containment and overlap ratios
are invariant under the per-axis scaling to source pixels.
"""
from copy import deepcopy
import math
import re

from .auto_provider import fingerprint
from .auto_protocol import transform_model_boxes
from .region_plan import CAPABILITIES, CLASSIFICATIONS

OVERLAP = .025
OWN = .95
WHOLE_PAGE = .8
PAGE_AREA = 1000*1000

CLASSIFICATION_ALIASES = {
    'header': 'decorative', 'footer': 'decorative', 'logo': 'decorative', 'title': 'decorative',
    'banner': 'decorative', 'decoration': 'decorative', 'background': 'decorative', 'page_number': 'decorative',
    'text': 'mixed', 'text_block': 'mixed', 'panel': 'mixed', 'card': 'mixed', 'content': 'mixed',
    'list': 'mixed', 'summary': 'mixed', 'overview': 'mixed', 'other': 'mixed', 'sidebar': 'mixed',
    'legend': 'mixed', 'note': 'mixed', 'notes': 'mixed', 'caption': 'mixed',
    'cards': 'step_cards', 'steps': 'step_cards', 'process': 'flow', 'timeline': 'flow', 'flowchart': 'flow',
    'bar_chart': 'chart', 'graph': 'chart', 'photo': 'photo_map', 'map': 'photo_map', 'image': 'photo_diagram',
    'diagram': 'photo_diagram', 'screenshot': 'ui_mockup',
}


def _valid(b):
    return (isinstance(b, list) and len(b) == 4 and
            all(type(v) in (int, float) and math.isfinite(v) for v in b))


def _area(b):
    return max(0, b[2]-b[0])*max(0, b[3]-b[1])


def _inter(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _contains(outer, inner):
    return (inner[0] >= outer[0]-1e-6 and inner[1] >= outer[1]-1e-6 and
            inner[2] <= outer[2]+1e-6 and inner[3] <= outer[3]+1e-6)


def _overlapping(a, b):
    return _inter(a, b) > min(_area(a), _area(b))*OVERLAP


def _owns(region, text):
    return _inter(region, text) >= _area(text)*OWN


def _gap(a, b):
    return max(a[0]-b[2], b[0]-a[2], 0)**2+max(a[1]-b[3], b[1]-a[3], 0)**2


def _round(b):
    return [round(v, 2) for v in b]


def _record(schema, raw, repaired, operations):
    return {'schema': schema, 'raw_sha256': fingerprint(raw), 'repaired_sha256': fingerprint(repaired),
            'operations': operations}


def _clamp(b):
    c = [min(max(v, 0), 1000) for v in b]
    return c if c[0] < c[2] and c[1] < c[3] else None


def _shrink_options(a, exclude):
    """Boxes that keep the largest part of a while having no area in common with exclude."""
    options = [[a[0], a[1], min(a[2], exclude[0]), a[3]], [max(a[0], exclude[2]), a[1], a[2], a[3]],
               [a[0], a[1], a[2], min(a[3], exclude[1])], [a[0], max(a[1], exclude[3]), a[2], a[3]]]
    return sorted((o for o in options if o[0] < o[2] and o[1] < o[3]), key=lambda o: -_area(o))


def repair_layout(draft, inventory):
    """Return (repaired_draft, AutomaticLayoutRepair/1 record). Never raises on malformed input."""
    value, operations = normalize_draft_envelope(deepcopy(draft), 'RegionLayoutDraft/1')
    try:
        regions = value['regions']
        ok = isinstance(regions, list) and all(
            isinstance(r, dict) and _valid(r.get('bbox')) and isinstance(r.get('subregions', []), list)
            for r in regions)
        texts = [(t['id'], transform_model_boxes({'bbox': t['bbox']}, inventory['image_size'], to_model=True)['bbox'])
                 for t in inventory['texts']]
    except (KeyError, TypeError, ValueError, AttributeError):
        ok = False
    # A whole-page region is a model error, not a geometry slip; never repair it into shape.
    if not ok or any(_area(r['bbox']) >= PAGE_AREA*WHOLE_PAGE for r in regions):
        return value, _record('AutomaticLayoutRepair/1', draft, value, operations)

    color = _normalize_color(value.get('background_color'), allow_none=False)
    if color is not None and color != value.get('background_color'):
        operations.append({'op': 'normalize_color', 'field': 'background_color',
                           'from': value.get('background_color'), 'to': color})
        value['background_color'] = color

    for region in regions:
        label = region.get('classification')
        if label not in CLASSIFICATIONS and isinstance(label, str):
            key = re.sub(r'[\s-]+', '_', label.strip().lower())
            new = key if key in CLASSIFICATIONS else CLASSIFICATION_ALIASES.get(key)
            if new:
                operations.append({'op': 'map_classification', 'region': region.get('id'), 'from': label, 'to': new})
                region['classification'] = new
        clamped = _clamp(region['bbox'])
        if clamped and clamped != region['bbox']:
            operations.append({'op': 'clamp_region', 'region': region.get('id'), 'to': _round(clamped)})
            region['bbox'] = clamped

    def owners_of(box, boxes):
        return [i for i, b in enumerate(boxes) if _owns(b, box)]

    def keeps_every_owner(index, candidate):
        boxes = [r['bbox'] for r in regions]
        trial = list(boxes); trial[index] = candidate
        return all(not owners_of(t, boxes) or owners_of(t, trial) for _, t in texts)

    def introduces_overlap(index, candidate):
        return any(j != index and _overlapping(candidate, r['bbox']) and not _overlapping(regions[index]['bbox'], r['bbox'])
                   for j, r in enumerate(regions))

    def claims_owned_text(index, candidate):
        boxes = [r['bbox'] for r in regions]
        return any(_owns(candidate, t) and not _owns(boxes[index], t) and owners_of(t, boxes) for _, t in texts)

    def can_grow(index, candidate):
        return (_area(candidate) < PAGE_AREA*WHOLE_PAGE and not introduces_overlap(index, candidate)
                and not claims_owned_text(index, candidate))

    # Subregions outside their parent: grow the parent when safe, otherwise clip the child.
    for index, region in enumerate(regions):
        for sub in list(region.get('subregions', [])):
            if not isinstance(sub, dict) or not _valid(sub.get('bbox')) or _contains(region['bbox'], sub['bbox']):
                continue
            clamped = _clamp(sub['bbox']) or sub['bbox']
            grown = _union(region['bbox'], clamped)
            if can_grow(index, grown):
                operations.append({'op': 'grow_region_for_subregion', 'region': region.get('id'),
                                   'subregion': sub.get('id'), 'from': _round(region['bbox']), 'to': _round(grown)})
                region['bbox'] = grown
                sub['bbox'] = clamped

    # Substantial overlaps: cut one region away from the other along the cheapest side.
    for _ in range(3):
        changed = False
        for i, a in enumerate(regions):
            for j, b in enumerate(regions):
                if j <= i or not _overlapping(a['bbox'], b['bbox']):
                    continue
                choices = []
                for index, other in ((i, b), (j, a)):
                    for option in _shrink_options(regions[index]['bbox'], other['bbox']):
                        if keeps_every_owner(index, option):
                            choices.append((_area(regions[index]['bbox'])-_area(option), index, option))
                            break
                if not choices:
                    continue
                _, index, option = min(choices, key=lambda c: c[0])
                operations.append({'op': 'resolve_overlap', 'region': regions[index].get('id'),
                                   'with': (b if index == i else a).get('id'),
                                   'from': _round(regions[index]['bbox']), 'to': _round(option)})
                regions[index]['bbox'] = option
                changed = True
        if not changed:
            break

    # OCR ownership: exactly one region must wholly contain each OCR box.
    for tid, tbox in texts:
        boxes = [r['bbox'] for r in regions]
        owners = owners_of(tbox, boxes)
        if not owners:
            ranked = sorted(range(len(regions)), key=lambda k: (-_inter(boxes[k], tbox), _gap(boxes[k], tbox)))
            for index in ranked[:3]:
                grown = _union(boxes[index], tbox)
                if can_grow(index, grown):
                    operations.append({'op': 'grow_region_for_ocr', 'region': regions[index].get('id'), 'ocr_id': tid,
                                       'from': _round(boxes[index]), 'to': _round(grown)})
                    regions[index]['bbox'] = grown
                    break
        elif len(owners) > 1:
            keep = min(owners, key=lambda k: _area(boxes[k]))
            for index in owners:
                if index == keep:
                    continue
                for option in _shrink_options(boxes[index], tbox):
                    if keeps_every_owner(index, option) and not _owns(option, tbox):
                        operations.append({'op': 'release_shared_ocr', 'region': regions[index].get('id'), 'ocr_id': tid,
                                           'kept_by': regions[keep].get('id'),
                                           'from': _round(boxes[index]), 'to': _round(option)})
                        regions[index]['bbox'] = option
                        break

    # Children must end inside their (possibly shrunk) parent.
    for region in regions:
        kept = []
        for sub in region.get('subregions', []):
            if isinstance(sub, dict) and _valid(sub.get('bbox')) and not _contains(region['bbox'], sub['bbox']):
                b = region['bbox']; s = sub['bbox']
                clipped = [max(b[0], s[0]), max(b[1], s[1]), min(b[2], s[2]), min(b[3], s[3])]
                if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                    operations.append({'op': 'drop_subregion_outside_parent', 'region': region.get('id'),
                                       'subregion': sub.get('id')})
                    continue
                operations.append({'op': 'clip_subregion', 'region': region.get('id'), 'subregion': sub.get('id'),
                                   'from': _round(s), 'to': _round(clipped)})
                sub['bbox'] = clipped
            kept.append(sub)
        if 'subregions' in region:
            region['subregions'] = kept
    return value, _record('AutomaticLayoutRepair/1', draft, value, operations)


# ---------------------------------------------------------------- decisions

ICON_PARTS = {'icon', 'icons', 'logo', 'logos', 'title_icon', 'pictogram', 'symbol', 'symbols', 'emblem'}
NATIVE_ALIASES = {'title': 'section_title', 'header': 'section_title', 'heading': 'section_title',
                  'text': 'labels', 'label': 'labels', 'value': 'values', 'arrows': 'arrow', 'circles': 'circle',
                  'cards': 'card', 'panels': 'panel', 'bar': 'bars', 'badge': 'number_badge', 'badges': 'number_badge'}
RETAINED_ALIASES = {'icon': 'icons', 'logo': 'icons', 'logos': 'icons', 'title_icon': 'icons', 'pictogram': 'icons',
                    'symbol': 'icons', 'image': 'photo', 'images': 'photos', 'picture': 'photo',
                    'background': 'decorative', 'decoration': 'decorative', 'illustration': 'decorative'}


def align_decision(raw):
    """Move parts the fixed compiler cannot draw (icons, logos) to retained pixels.

    Returns (aligned_raw, operations). Malformed input is returned unchanged so
    the strict decision validator reports it.
    """
    value = deepcopy(raw)
    operations = []
    try:
        can = value['can_rebuild']
        native = list(value['native_parts']); can_native = list(can['native_parts'])
        retained = list(value['retained_parts']); reasons = list(value['retain_reasons'])
        if not all(isinstance(p, str) for p in native+can_native+retained+reasons):
            return value, operations
    except (KeyError, TypeError):
        return value, operations
    native_vocabulary = set(CAPABILITIES['native_parts']); retained_vocabulary = set(CAPABILITIES['retained_parts'])

    def add(items, item):
        if item not in items:
            items.append(item)

    def native_list(parts):
        result = []
        for part in parts:
            if part in ICON_PARTS:
                add(retained, 'icons')
                operations.append({'op': 'capability_downgrade', 'part': part, 'to': 'retained/icons',
                                   'reason': 'fixed compiler draws only allowed geometries, not icon artwork'})
            elif part not in native_vocabulary and part in NATIVE_ALIASES:
                operations.append({'op': 'map_native_part', 'from': part, 'to': NATIVE_ALIASES[part]})
                add(result, NATIVE_ALIASES[part])
            else:
                add(result, part)
        return result

    native = native_list(native)
    can_native = native_list(can_native)
    mapped = []
    for part in retained:
        if part not in retained_vocabulary and part in RETAINED_ALIASES:
            operations.append({'op': 'map_retained_part', 'from': part, 'to': RETAINED_ALIASES[part]})
            part = RETAINED_ALIASES[part]
        add(mapped, part)
    retained = [p for p in mapped if p not in native]
    if not operations:
        return value, operations
    # Duplicated operations from native_parts and can_rebuild.native_parts describe one change.
    unique = []
    for item in operations:
        if item not in unique:
            unique.append(item)
    operations = unique
    native = [p for p in native if p not in retained]
    if any(o['op'] == 'capability_downgrade' for o in operations) and 'non_native_geometry' not in reasons and len(reasons) < 6:
        reasons.append('non_native_geometry')
    kind = value.get('decision')
    if native and retained:
        new_kind = 'hybrid'
    elif retained:
        new_kind = 'retain_module'; native = []
    else:
        new_kind = kind
    if new_kind != kind:
        operations.append({'op': 'decision_kind_follows_scope', 'from': kind, 'to': new_kind})
    if new_kind == 'native':
        reasons = []
    value.update(decision=new_kind, native_parts=native, retained_parts=retained, retain_reasons=reasons)
    can['native_parts'] = list(native)
    return value, operations


# ---------------------------------------------------------------- colors

_HEX = re.compile(r'[0-9a-fA-F]{6}')


def _normalize_color(value, *, allow_none):
    """Six-hex RRGGBB, or 'none' where transparency is allowed; None when not convertible."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.lower() in ('none', 'transparent', ''):
        return 'none' if allow_none else None
    match = re.fullmatch(r'rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*(?:,\s*[\d.]+\s*)?\)', text, re.I)
    if match:
        channels = [int(c) for c in match.groups()]
        return ''.join(f'{c:02X}' for c in channels) if all(c <= 255 for c in channels) else None
    text = re.sub(r'^(#|0x)', '', text, flags=re.I)
    if re.fullmatch(r'[0-9a-fA-F]{3}', text):
        text = ''.join(c*2 for c in text)
    elif re.fullmatch(r'[0-9a-fA-F]{8}', text):
        text = text[:6]
    return text.upper() if _HEX.fullmatch(text) else None


def normalize_draft_colors(draft):
    """Normalize color notations in a region draft; return (draft, operations, problems).

    problems lists fields that still hold an unusable color, with the object ID
    and field path, so model feedback can name the exact location.
    """
    value = deepcopy(draft); operations = []; problems = []

    def fix(owner, key, path, allow_none):
        current = owner.get(key)
        new = _normalize_color(current, allow_none=allow_none)
        if new is None:
            problems.append({'field': path, 'value': current,
                             'allowed': 'six hex RRGGBB'+(' or "none"' if allow_none else '')})
        elif new != current:
            operations.append({'op': 'normalize_color', 'field': path, 'from': current, 'to': new})
            owner[key] = new

    for obj in value.get('objects', []) if isinstance(value.get('objects'), list) else []:
        if not isinstance(obj, dict) or obj.get('kind') != 'native_shape':
            continue
        oid = obj.get('id')
        if 'fill' in obj and obj['fill'] is not None:
            fix(obj, 'fill', f'{oid}.fill', True)
        line = obj.get('line')
        if isinstance(line, dict) and 'color' in line:
            if _normalize_color(line['color'], allow_none=True) == 'none':
                operations.append({'op': 'drop_transparent_line', 'field': f'{oid}.line'})
                obj.pop('line')
            else:
                fix(line, 'color', f'{oid}.line.color', False)
        gradient = obj.get('fill_gradient')
        if isinstance(gradient, dict) and isinstance(gradient.get('stops'), list):
            for k, stop in enumerate(gradient['stops']):
                if isinstance(stop, dict) and 'color' in stop:
                    fix(stop, 'color', f'{oid}.fill_gradient.stops[{k}].color', False)
    for update in value.get('text_overrides', []) if isinstance(value.get('text_overrides'), list) else []:
        if not isinstance(update, dict):
            continue
        tid = update.get('id')
        if 'color' in update:
            fix(update, 'color', f'{tid}.color', False)
        for k, run in enumerate(update.get('runs') or []):
            if isinstance(run, dict) and 'color' in run:
                fix(run, 'color', f'{tid}.runs[{k}].color', False)
    return value, operations, problems


# ------------------------------------------------------- draft bookkeeping

GEOMETRY_KEYS = ('type', 'kind', 'shape', 'name', 'geometry', 'preset', 'prst')
GEOMETRY_ALIASES = {'rectangle': 'rect', 'square': 'rect', 'box': 'rect', 'roundrect': 'roundRect',
                    'rounded_rect': 'roundRect', 'roundedrect': 'roundRect', 'rounded_rectangle': 'roundRect',
                    'circle': 'ellipse', 'oval': 'ellipse', 'arrow': 'rightArrow', 'right_arrow': 'rightArrow',
                    'rightarrow': 'rightArrow', 'left_arrow': 'leftArrow', 'leftarrow': 'leftArrow'}
OVERRIDE_BOOKKEEPING = ('status', 'reason', 'action', 'style_group')


def _geometry_name(value):
    from .supervised_elements import GEOMETRIES
    if not isinstance(value, str):
        return None
    if value in GEOMETRIES:
        return value
    folded = {g.lower(): g for g in GEOMETRIES}
    key = re.sub(r'[\s-]+', '_', value.strip()).lower()
    return folded.get(key) or GEOMETRY_ALIASES.get(key)


def normalize_draft_envelope(draft, schema):
    """Drop the echoed response_format marker and fill the fixed schema constant; return (draft, operations)."""
    if not isinstance(draft, dict):
        return draft, []
    value = dict(draft); operations = []
    if value.get('type') == 'json_object':
        value.pop('type')
        operations.append({'op': 'drop_response_format_marker'})
    if 'schema' not in value and 'coordinate_space' in value:
        value['schema'] = schema
        operations.append({'op': 'fill_schema_constant', 'schema': schema})
    return value, operations


REVIEW_ISSUE_FIELDS = ('id', 'kind', 'severity', 'object_ids', 'bbox', 'description')
CONTENT_KINDS = ('text', 'numeric', 'missing', 'duplicate')


def normalize_visual_review(raw, known_ids):
    """Bookkeeping repairs for RegionVisualReviewDraft/1; findings are never removed or softened."""
    value, operations = normalize_draft_envelope(raw, 'RegionVisualReviewDraft/1')
    if not isinstance(value, dict) or not isinstance(value.get('issues'), list):
        return value, operations
    value = deepcopy(value); blocking = False
    for item in value['issues']:
        if not isinstance(item, dict):
            continue
        extra = sorted(set(item) - set(REVIEW_ISSUE_FIELDS))
        if extra and set(REVIEW_ISSUE_FIELDS) <= set(item):
            for key in extra:
                item.pop(key)
            operations.append({'op': 'drop_issue_fields', 'issue': item.get('id'), 'fields': extra})
        if isinstance(item.get('object_ids'), list):
            unknown = [i for i in item['object_ids'] if i not in known_ids]
            if unknown:
                item['object_ids'] = [i for i in item['object_ids'] if i in known_ids]
                operations.append({'op': 'drop_unknown_issue_targets', 'issue': item.get('id'),
                                   'targets': [str(i)[:40] for i in unknown]})
        if item.get('kind') in CONTENT_KINDS and item.get('severity') == 'minor':
            item['severity'] = 'blocking'
            operations.append({'op': 'content_issue_is_blocking', 'issue': item.get('id')})
        blocking = blocking or item.get('severity') == 'blocking'
    if value.get('quality_passed') is True and blocking:
        value['quality_passed'] = False
        operations.append({'op': 'blocking_finding_fails_review'})
    return value, operations


GRADIENT_SCALE = {1: 100000, 100: 1000}   # IR stop offsets run 0..100000


def gradient_problem(gradient):
    """Mirror scripts/ir.mjs fill_gradient rules; return a message or None."""
    if not isinstance(gradient, dict):
        return 'fill_gradient must be an object'
    stops = gradient.get('stops')
    if not (type(gradient.get('angle_deg')) in (int, float) and isinstance(stops, list) and 2 <= len(stops) <= 8):
        return 'fill_gradient needs numeric angle_deg and 2-8 stops'
    if not all(isinstance(s, dict) and type(s.get('offset')) in (int, float) and 0 <= s['offset'] <= 100000
               and _normalize_color(s.get('color'), allow_none=False) == s.get('color') for s in stops):
        return 'gradient stops need offset 0..100000 and six-hex color'
    if stops[0]['offset'] != 0 or stops[-1]['offset'] != 100000 or any(
            b['offset'] <= a['offset'] for a, b in zip(stops, stops[1:])):
        return 'gradient stop offsets must start at 0, end at 100000 and strictly increase'
    return None


def normalize_draft_structure(draft):
    """Fix notation slips the program can resolve exactly; return (draft, operations)."""
    value = deepcopy(draft); operations = []
    for obj in value.get('objects', []) if isinstance(value.get('objects'), list) else []:
        if not isinstance(obj, dict) or obj.get('kind') != 'native_shape':
            continue
        geometry = obj.get('geometry')
        if isinstance(geometry, dict):
            names = {_geometry_name(v) for k, v in geometry.items() if k in GEOMETRY_KEYS} - {None}
            new = names.pop() if len(names) == 1 else None
        else:
            new = _geometry_name(geometry)
        if new and new != geometry:
            operations.append({'op': 'normalize_geometry', 'object': obj.get('id'), 'from': geometry, 'to': new})
            obj['geometry'] = new
        gradient = obj.get('fill_gradient')
        stops = gradient.get('stops') if isinstance(gradient, dict) else None
        if (isinstance(stops, list) and len(stops) >= 2 and
                all(isinstance(s, dict) and type(s.get('offset')) in (int, float) for s in stops)):
            last = stops[-1]['offset']
            scale = GRADIENT_SCALE.get(last) if stops[0]['offset'] == 0 else None
            if scale and all(0 <= s['offset'] <= last for s in stops):
                for s in stops:
                    s['offset'] = round(s['offset']*scale)
                operations.append({'op': 'scale_gradient_offsets', 'object': obj.get('id'), 'factor': scale})
    overrides = value.get('text_overrides')
    if isinstance(overrides, dict) and all(isinstance(v, dict) and v.get('id', k) == k for k, v in overrides.items()):
        value['text_overrides'] = [{'id': k, **v} for k, v in overrides.items()]
        operations.append({'op': 'overrides_keyed_by_id_to_list', 'texts': sorted(overrides)})
    for update in value.get('text_overrides', []) if isinstance(value.get('text_overrides'), list) else []:
        if isinstance(update, dict):
            removed = [k for k in OVERRIDE_BOOKKEEPING if k in update]
            for key in removed:
                update.pop(key)
            if removed:
                operations.append({'op': 'drop_override_bookkeeping', 'text': update.get('id'), 'fields': removed})
    return value, operations


def reconcile_audit(audit, texts):
    """Retention status follows actual crop containment; missing audits are honestly unresolved."""
    operations = []; result = []; seen = set()
    for item in audit if isinstance(audit, list) else []:
        if not isinstance(item, dict) or item.get('id') not in texts or item.get('id') in seen:
            continue
        item = dict(item); tid = item['id']; seen.add(tid)
        retained = texts[tid]['action'] == 'retained_in_asset'
        if retained and item.get('status') != 'retained':
            operations.append({'op': 'audit_follows_retention', 'text': tid, 'from': item.get('status'), 'to': 'retained'})
            item['status'] = 'retained'
        elif not retained and item.get('status') == 'retained':
            operations.append({'op': 'audit_follows_retention', 'text': tid, 'from': 'retained', 'to': 'unresolved'})
            item['status'] = 'unresolved'
        result.append(item)
    for tid, text in texts.items():
        if tid not in seen:
            retained = text['action'] == 'retained_in_asset'
            operations.append({'op': 'audit_missing', 'text': tid, 'to': 'retained' if retained else 'unresolved'})
            result.append({'id': tid, 'status': 'retained' if retained else 'unresolved',
                           'reason': 'No typography audit was returned for this text.', 'style_group': ''})
    return result, operations


# ---------------------------------------------------------------- mappings

TEXT_PART_PRIORITY = ('labels', 'section_title', 'values', 'cell_text', 'category_labels', 'percentage_labels')


def parts_from_mapping(draft):
    """Convert a legacy mapping into per-object part tags; tags already present win."""
    value = deepcopy(draft)
    mapping = value.pop('mapping', None)
    if not isinstance(mapping, dict):
        return value
    tags = {}
    for kind in ('native', 'retained'):
        groups = mapping.get(kind) or {}
        if isinstance(groups, dict):
            for part, refs in groups.items():
                for ref in refs if isinstance(refs, list) else []:
                    if isinstance(ref, str):
                        tags.setdefault(ref, part)
    text_parts = dict(value.get('text_parts') or {})
    for obj in value.get('objects', []) if isinstance(value.get('objects'), list) else []:
        if isinstance(obj, dict) and 'part' not in obj and obj.get('id') in tags:
            obj['part'] = tags[obj['id']]
    for ref, part in tags.items():
        if ref.startswith('text-'):
            text_parts.setdefault(ref, part)
    if text_parts:
        value['text_parts'] = text_parts
    return value


def tagged_view(elements):
    """Accepted region elements as the model sees them: part tags instead of mapping."""
    value = deepcopy(elements)
    for key in ('scope_adjustment', 'program_repair'):
        value.pop(key, None)
    tags = {ref: part for groups in value.pop('mapping', {}).values() for part, refs in groups.items() for ref in refs}
    for obj in value.get('objects', []):
        if obj['id'] in tags:
            obj['part'] = tags[obj['id']]
    value['text_parts'] = {t['id']: tags[t['id']] for t in value.get('text_decisions', []) if t['id'] in tags}
    return value


def derive_mapping(objects, texts, text_parts, routing):
    """Build the native/retained mapping from per-object part tags.

    objects: expanded objects with an optional 'part' tag (removed here).
    texts: {id: text decision} after raster containment set action.
    Returns (objects_without_tags, mapping, scope, operations). The scope is the
    bound decision's parts adjusted only by recorded operations.
    """
    native_vocabulary = set(CAPABILITIES['native_parts']); retained_vocabulary = set(CAPABILITIES['retained_parts'])
    native = list(routing['native_parts']); retained = list(routing['retained_parts'])
    operations = []; mapping = {'native': {}, 'retained': {}}; clean = []

    def default_native():
        for part in TEXT_PART_PRIORITY:
            if part in native:
                return part
        return native[0] if native else 'labels'

    def default_shape():
        return 'panel' if 'panel' in native else (native[0] if native else 'panel')

    for obj in objects:
        obj = dict(obj); part = obj.pop('part', None); oid = obj['id']
        if obj['kind'] == 'raster_asset':
            if part in retained:
                pass
            elif part in native:
                native.remove(part); retained.append(part)
                operations.append({'op': 'native_part_realized_as_raster', 'part': part, 'object': oid})
            elif isinstance(part, str) and (part in retained_vocabulary or part in RETAINED_ALIASES):
                part = RETAINED_ALIASES.get(part, part)
                if part not in retained:
                    retained.append(part)
                    operations.append({'op': 'retained_part_added', 'part': part, 'object': oid})
            else:
                new = retained[0] if retained else 'module_body'
                if new not in retained:
                    retained.append(new)
                operations.append({'op': 'raster_part_assigned', 'from': part, 'part': new, 'object': oid})
                part = new
            mapping['retained'].setdefault(part, []).append(oid)
        else:
            if part in retained or (part in retained_vocabulary and part not in native_vocabulary):
                raise ValueError('retained part must be actual source pixels: native shape '+oid+' is tagged '+part+
                                 '; crop the source as raster_asset or tag a native part')
            if part in native:
                pass
            elif isinstance(part, str) and part in native_vocabulary:
                native.append(part)
                operations.append({'op': 'native_part_added', 'part': part, 'object': oid})
            else:
                new = default_shape()
                if new not in native:
                    native.append(new)
                operations.append({'op': 'shape_part_assigned', 'from': part, 'part': new, 'object': oid})
                part = new
            mapping['native'].setdefault(part, []).append(oid)
        clean.append(obj)
    for tid, text in texts.items():
        if text['action'] != 'native':
            continue
        part = (text_parts or {}).get(tid)
        if part in native:
            pass
        elif isinstance(part, str) and part in native_vocabulary and part not in retained:
            native.append(part)
            operations.append({'op': 'native_part_added', 'part': part, 'object': tid})
        else:
            new = default_native()
            if new not in native:
                native.append(new)
                operations.append({'op': 'native_part_added', 'part': new, 'object': tid})
            if part is not None:
                operations.append({'op': 'text_part_assigned', 'from': part, 'part': new, 'object': tid})
            part = new
        mapping['native'].setdefault(part, []).append(tid)
    for kind, parts in (('native', native), ('retained', retained)):
        for part in list(parts):
            if part not in mapping[kind]:
                parts.remove(part)
                operations.append({'op': 'unrealized_part_dropped', 'part': part, 'kind': kind})
    scope = {'native_parts': native, 'retained_parts': retained}
    return clean, mapping, scope, operations


TEXT_RASTER_OVERLAP = .12   # supervised_elements.raster_text_conflicts threshold
KEEP_TRIMMED = .6


def _xyxy(b):
    return [b[0], b[1], b[0]+b[2], b[1]+b[3]]


GROW_DECLARED = .5


def grow_rasters_to_declared_text(objects, contained_boxes, native_boxes, region):
    """Extend a raster crop to the full OCR box of text it declares to contain.

    Pixel xywh. Applied only when the grown crop stays in the region, grows by at most half
    its area and does not newly cover a native text; dest_bbox grows at the same scale.
    """
    operations = []; limit = _xyxy(region)
    for obj in objects:
        if obj.get('kind') != 'raster_asset' or not _valid(obj.get('source_crop_bbox')) or not _valid(obj.get('dest_bbox')):
            continue
        crop = _xyxy(obj['source_crop_bbox'])
        boxes = [_xyxy(contained_boxes[t]) for t in obj.get('contains_text_ids', []) if t in contained_boxes]
        if not boxes or all(_inter(crop, b) >= _area(b)*OWN for b in boxes):
            continue
        new = crop
        for b in boxes:
            new = _union(new, b)
        if (not _contains(limit, new) or _area(new) > _area(crop)*(1+GROW_DECLARED) or
                any(_inter(new, _xyxy(n)) > max(_inter(crop, _xyxy(n)), _area(_xyxy(n))*TEXT_RASTER_OVERLAP)
                    for n in native_boxes.values())):
            continue
        dest = obj['dest_bbox']; old = obj['source_crop_bbox']; sx, sy = dest[2]/old[2], dest[3]/old[3]
        grown = [dest[0]+(new[0]-crop[0])*sx, dest[1]+(new[1]-crop[1])*sy, (new[2]-new[0])*sx, (new[3]-new[1])*sy]
        if not _contains(limit, _xyxy(grown)):
            continue
        obj['dest_bbox'] = grown
        obj['source_crop_bbox'] = [new[0], new[1], new[2]-new[0], new[3]-new[1]]
        operations.append({'op': 'grow_raster_to_declared_text', 'object': obj.get('id'),
                           'from': _round(old), 'to': _round(obj['source_crop_bbox'])})
    return operations


def trim_rasters_off_native_text(objects, native_boxes, contained_boxes):
    """Cut a raster crop on one side so it stops covering an adjacent native text's OCR box.

    Boxes are pixel xywh. A cut is applied only if the crop keeps most of its area and still
    contains every text it declares; a text that sits inside the crop is left to the model.
    Returns operations; objects are changed in place.
    """
    operations = []
    for obj in objects:
        if obj.get('kind') != 'raster_asset' or not _valid(obj.get('source_crop_bbox')) or not _valid(obj.get('dest_bbox')):
            continue
        declared = [contained_boxes[t] for t in obj.get('contains_text_ids', []) if t in contained_boxes]
        for tid, box in native_boxes.items():
            crop = _xyxy(obj['source_crop_bbox']); text = _xyxy(box)
            if _inter(crop, text) <= _area(text)*TEXT_RASTER_OVERLAP:
                continue
            options = [o for o in _shrink_options(crop, text)
                       if _area(o) >= _area(crop)*KEEP_TRIMMED and
                       all(_inter(o, _xyxy(d)) >= _area(_xyxy(d))*OWN for d in declared)]
            if not options:
                continue
            new = options[0]; dest = obj['dest_bbox']; old = obj['source_crop_bbox']
            sx, sy = dest[2]/old[2], dest[3]/old[3]
            obj['dest_bbox'] = [dest[0]+(new[0]-crop[0])*sx, dest[1]+(new[1]-crop[1])*sy,
                                (new[2]-new[0])*sx, (new[3]-new[1])*sy]
            obj['source_crop_bbox'] = [new[0], new[1], new[2]-new[0], new[3]-new[1]]
            operations.append({'op': 'trim_raster_off_native_text', 'object': obj.get('id'), 'text': tid,
                               'from': _round(old), 'to': _round(obj['source_crop_bbox'])})
    return operations
