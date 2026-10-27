"""Paragraph grouping, alignment and reading order for native text.

The deterministic grouper always runs. The model (one text-only call per page)
may only re-partition lines into paragraphs and choose alignment; every proposed
paragraph must still pass the same geometric compatibility check, and anything
it omits or breaks falls back to the deterministic answer. The model never
sees pixels to place and never edits wording.
"""
import hashlib
import json

SCHEMA = 'LayeredSemantics/1'
INSTRUCTION = (
    'You group OCR lines of one presentation slide into paragraphs. Input json: lines '
    '(id, text, box [x,y,w,h] in pixels, size_px, color, bold) and containers (native shape boxes). '
    'Return json only: {"paragraphs":[{"ids":[...],"align":"left|center|right","role":"title|heading|body|label|number|list"}]}. '
    'Merge consecutive lines of the same text block — a wrapped sentence, a bullet list in one column, '
    'or same-card body lines of one font size — into one paragraph in top-to-bottom order. '
    'Edges, sizes and colours that differ only by measurement noise are still the same block. '
    'Never merge lines from different cards, columns or table cells, and never merge a label with its value. '
    'Every id must appear exactly once. Do not change any text. Order paragraphs in natural reading order.')


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def _center(b):
    return b[0]+b[2]/2


# Wrapped lines of one card share a size, ink colour and edge, but the ink box
# jitters by about an em, sampled colours drift by a few dozen levels, and
# width-limiting can shrink one line toward 75% of its fitted size. A different
# card or column is still a hard stop.
_ALIGN_EM = 1.35
_SIZE_RATIO = .22
_COLOR_GAP = 56
_SAME_INK = 18


def _pad(box):
    return min(12., max(4., .45*box[3]))


def _inside(box, container, pad):
    return (container[0] <= box[0]+pad and container[1] <= box[1]+pad and
            box[0]+box[2] <= container[0]+container[2]+pad and
            box[1]+box[3] <= container[1]+container[3]+pad)


def _container(box, containers):
    pad = _pad(box)
    inside = [c for c in containers if _inside(box, c, pad)]
    return min(inside, key=lambda c: c[2]*c[3], default=None)


def _lab_distance(a, b):
    a = [int(a[i:i+2], 16) for i in (0, 2, 4)]
    b = [int(b[i:i+2], 16) for i in (0, 2, 4)]
    return max(abs(p-q) for p, q in zip(a, b))


def _overlap_width(a, b):
    ba, bb = a['ink_bbox'], b['ink_bbox']
    return max(0., min(ba[0]+ba[2], bb[0]+bb[2]) - max(ba[0], bb[0]))


def _same_column(a, b):
    """b sits under a, rather than grazing a neighbouring column."""
    narrow = min(a['ink_bbox'][2], b['ink_bbox'][2])
    return _overlap_width(a, b) >= .45*max(narrow, 1.)


def _size_close(a, b):
    direct = abs(a['size_px']-b['size_px']) / max(a['size_px'], b['size_px'])
    if direct <= _SIZE_RATIO:
        return True
    # Compare the pre-shrink sizes when width-limiting moved only one line.
    ra = a.get('width_limited_from_px', a['size_px'])
    rb = b.get('width_limited_from_px', b['size_px'])
    return abs(ra-rb) / max(ra, rb) <= .12


def _align_errors(a, b):
    ba, bb = a['ink_bbox'], b['ink_bbox']
    return {'left': abs(ba[0]-bb[0]),
            'center': abs(_center(ba)-_center(bb)),
            'right': abs((ba[0]+ba[2])-(bb[0]+bb[2]))}


def _alignment(a, b):
    """Shared edge. A tight match keeps the previous priority; otherwise the closest edge within one em."""
    h = max(a['ink_bbox'][3], b['ink_bbox'][3], 1.)
    errors = _align_errors(a, b)
    for mode in ('left', 'center', 'right'):
        if errors[mode] <= .6*h:
            return mode
    mode = min(errors, key=errors.get)
    return mode if errors[mode] <= _ALIGN_EM*h else None


def compatible(a, b, containers):
    """Can line b continue the paragraph whose last line is a (a above b)?"""
    ba, bb = a['ink_bbox'], b['ink_bbox']
    h = max(ba[3], bb[3], 1.)
    if a.get('runs') or b.get('runs'):
        return None
    if not _size_close(a, b):
        return None
    color_gap = _lab_distance(a['color'], b['color'])
    if color_gap > _COLOR_GAP or (a['bold'] != b['bold'] and color_gap > _SAME_INK):
        return None
    gap = bb[1]-(ba[1]+ba[3])
    if not -.2*h <= gap <= 1.2*h:
        return None
    if _container(ba, containers) != _container(bb, containers):
        return None
    return _alignment(a, b)


def deterministic(styles, containers):
    lines = sorted(styles.values(), key=lambda s: (s['ink_bbox'][1], s['ink_bbox'][0]))
    paragraphs, used = [], set()
    for line in lines:
        if line['id'] in used:
            continue
        group, align = [line], None
        used.add(line['id'])
        while True:
            last = group[-1]
            below = [s for s in lines if s['id'] not in used and s['ink_bbox'][1] > last['ink_bbox'][1]
                     and _overlap_width(last, s) > 0]
            if not below:
                break
            extended = False
            for candidate in sorted(below, key=lambda s: (s['ink_bbox'][1], s['ink_bbox'][0])):
                mode = compatible(last, candidate, containers)
                if align and mode not in (None, align):
                    h = max(last['ink_bbox'][3], candidate['ink_bbox'][3], 1.)
                    if _align_errors(last, candidate)[align] <= _ALIGN_EM*h:
                        mode = align
                if mode is None or (align and mode != align):
                    # A real next line in this column ends the paragraph. A neighbour
                    # that only grazes the ink can be stepped over.
                    if _same_column(last, candidate):
                        break
                    continue
                align = mode
                group.append(candidate); used.add(candidate['id'])
                extended = True
                break
            if not extended:
                break
        paragraphs.append({'ids': [s['id'] for s in group], 'align': align or 'left', 'source': 'deterministic'})
    return paragraphs


def model_evidence(styles, containers):
    return {'lines': [{'id': s['id'], 'text': s['text'], 'box': [round(v) for v in s['ink_bbox']],
                       'size_px': round(s['size_px'], 1), 'color': s['color'], 'bold': s['bold']}
                      for s in sorted(styles.values(), key=lambda s: (s['ink_bbox'][1], s['ink_bbox'][0]))],
            'containers': [[round(v) for v in c] for c in containers]}


def apply_model(proposal, styles, containers, fallback):
    """Accept valid proposed paragraphs; everything else keeps the deterministic grouping."""
    issues = []
    if not isinstance(proposal, dict) or not isinstance(proposal.get('paragraphs'), list):
        return fallback, ['model_json_shape_invalid']
    accepted, seen = [], set()
    for item in proposal['paragraphs']:
        ids = item.get('ids') if isinstance(item, dict) else None
        if not isinstance(ids, list) or not ids or any(i not in styles or i in seen for i in ids):
            issues.append('invalid_or_duplicate_ids')
            continue
        group = sorted((styles[i] for i in ids), key=lambda s: s['ink_bbox'][1])
        modes = [compatible(a, b, containers) for a, b in zip(group, group[1:])]
        if any(m is None for m in modes):
            issues.append('incompatible_paragraph:'+','.join(ids))
            continue
        align = item.get('align') if item.get('align') in ('left', 'center', 'right') else (modes[0] if modes else 'left')
        if modes and align != modes[0] and not (len(group) == 1):
            align = modes[0]
        accepted.append({'ids': [s['id'] for s in group], 'align': align, 'role': item.get('role'), 'source': 'model'})
        seen.update(ids)
    for paragraph in fallback:
        rest = [i for i in paragraph['ids'] if i not in seen]
        if not rest:
            continue
        if rest != paragraph['ids']:
            for i in rest:
                accepted.append({'ids': [i], 'align': 'left', 'source': 'deterministic_split'})
        else:
            accepted.append(paragraph)
        seen.update(rest)
    return accepted, issues


def group(styles, containers, *, remote=None, model=None, record_dir=None):
    """Return (paragraphs, report). ``remote`` is an auto_transport.SSHRemote."""
    fallback = deterministic(styles, containers)
    report = {'schema': SCHEMA, 'model': model, 'model_used': False, 'issues': [], 'usage': {}}
    if remote is None or not styles:
        return fallback, report
    from uuid import uuid4
    from ..auto_provider import model_body, parse_object, save_new
    evidence = model_evidence(styles, containers)
    body = model_body(model, INSTRUCTION, evidence, [], max_tokens=8000)
    request = {'request_id': uuid4().hex, 'operation': 'model', 'timeout': 180, 'body': body}
    report['evidence_sha256'] = _digest(evidence)
    report['request_sha256'] = _digest(request)
    if record_dir:
        save_new(record_dir/'semantics-request.json', request)
    response = remote.request(request, timeout=240)
    if record_dir:
        save_new(record_dir/'semantics-response.json', response)
    report['usage'] = response.get('usage', {})
    if not response.get('ok'):
        report['issues'].append('model_call_failed:'+str(response.get('error')))
        return fallback, report
    content = response.get('content', '')
    report['response_sha256'] = hashlib.sha256(content.encode('utf-8')).hexdigest()
    try:
        proposal = parse_object(content)
    except (ValueError, json.JSONDecodeError):
        report['issues'].append('model_json_invalid')
        return fallback, report
    paragraphs, issues = apply_model(proposal, styles, containers, fallback)
    report.update(model_used=True, issues=issues)
    return paragraphs, report
