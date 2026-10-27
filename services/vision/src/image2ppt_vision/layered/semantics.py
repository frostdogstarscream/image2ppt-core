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
    'Merge consecutive lines of the same text block (a wrapped sentence, a bullet list in one column) '
    'into one paragraph in top-to-bottom order. Never merge lines from different cards, columns, table cells '
    'or of different size/colour, and never merge a label with its value. Every id must appear exactly once. '
    'Do not change any text. Order paragraphs in natural reading order.')


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def _center(b):
    return b[0]+b[2]/2


def _container(box, containers):
    inside = [c for c in containers if c[0] <= box[0]+1 and c[1] <= box[1]+1 and
              box[0]+box[2] <= c[0]+c[2]+1 and box[1]+box[3] <= c[1]+c[3]+1]
    return min(inside, key=lambda c: c[2]*c[3], default=None)


def _lab_distance(a, b):
    a = [int(a[i:i+2], 16) for i in (0, 2, 4)]
    b = [int(b[i:i+2], 16) for i in (0, 2, 4)]
    return max(abs(p-q) for p, q in zip(a, b))


def compatible(a, b, containers):
    """Can line b continue the paragraph whose last line is a (a above b)?"""
    ba, bb = a['ink_bbox'], b['ink_bbox']
    h = max(ba[3], bb[3])
    if a.get('runs') or b.get('runs'):
        return None
    if abs(a['size_px']-b['size_px']) > .12*max(a['size_px'], b['size_px']):
        return None
    if a['bold'] != b['bold'] or _lab_distance(a['color'], b['color']) > 40:
        return None
    gap = bb[1]-(ba[1]+ba[3])
    if not -.1*h <= gap <= .95*h:
        return None
    if _container(ba, containers) != _container(bb, containers):
        return None
    if abs(ba[0]-bb[0]) <= .6*h:
        return 'left'
    if abs(_center(ba)-_center(bb)) <= .6*h:
        return 'center'
    if abs(ba[0]+ba[2]-bb[0]-bb[2]) <= .6*h:
        return 'right'
    return None


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
            nexts = [s for s in lines if s['id'] not in used and s['ink_bbox'][1] > last['ink_bbox'][1]]
            below = [s for s in nexts if max(0, min(s['ink_bbox'][0]+s['ink_bbox'][2], last['ink_bbox'][0]+last['ink_bbox'][2])
                                         - max(s['ink_bbox'][0], last['ink_bbox'][0])) > 0]
            if not below:
                break
            candidate = min(below, key=lambda s: s['ink_bbox'][1])
            mode = compatible(last, candidate, containers)
            if mode is None or (align and mode != align):
                break
            align = mode
            group.append(candidate); used.add(candidate['id'])
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
