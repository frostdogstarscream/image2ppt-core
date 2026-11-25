"""Resolve a numbered request to the immutable source field and its optional native owner."""
from collections import defaultdict

from ..contracts import intersection
from .marked_routes import iou

TEXT_MARK_LABELS = frozenset({
    'text', 'title', 'doc_title', 'paragraph_title', 'paragraph', 'section_header',
    'number', 'caption', 'figure_caption', 'table_caption', 'header', 'footer', 'footnote', 'uncertain_ocr',
})


def resolve_targets(scene, lines, marks, requested, plans, logo_ids):
    """Never use a resized Office text frame as the bounds for reading source glyphs."""
    objects = {o.id: o for o in scene.text.texts}
    members = scene.text.members
    line_owners = defaultdict(list)
    for oid, ids in members.items():
        for lid in ids:
            if oid in objects:
                line_owners[lid].append(oid)
    intrinsic = set(logo_ids) | {r.source for r in scene.relations if r.kind == 'intrinsic_text'}
    picture_boxes = [p['bbox'] for p in plans if p.get('role') in ('brand', 'screenshot', 'dashboard')
                     or p.get('route') == 'preserve']
    picture_boxes.extend(a.bbox for a in scene.asset.objects if a.kind == 'logo' or a.text_policy == 'intrinsic_label')
    targets, items = {}, []
    for oid in dict.fromkeys(requested):
        item = {'id': oid, 'status': 'skipped'}
        items.append(item)
        if oid not in marks:
            item['reason'] = 'unknown_page_mark'
            continue
        mark = marks[oid]
        obj = objects.get(oid)
        label = mark.get('label')
        if obj:
            line_ids = members.get(oid, [])
        elif label in TEXT_MARK_LABELS:
            line_ids = mark.get('ocr_ids') or [
                lid for lid in mark.get('intersecting_ocr_ids', []) if lid in lines
                and iou(mark['bbox'], lines[lid]['bbox']) >= .7]
        else:
            item['reason'] = 'not_a_text_mark'
            continue
        if len(line_ids) != 1 or line_ids[0] not in lines:
            item['reason'] = 'ambiguous_or_missing_ocr_line'
            continue
        lid = line_ids[0]
        line = lines[lid]
        if obj is None and iou(mark['bbox'], line['bbox']) < .7:
            item['reason'] = 'mark_does_not_match_source_line'
            continue
        # A layout/OCR alias may address a whole line. Bind to an existing text object
        # only when it alone owns exactly this line; otherwise retain a reading as evidence.
        if obj is None and len(line_owners[lid]) == 1:
            owner = objects[line_owners[lid][0]]
            if members[owner.id] == [lid] and owner.text == line['text']:
                obj = owner
        box = line['bbox']
        word = line.get('ocr_raw_text', line['text']) if line.get('field_readings') else line['text']
        span = None
        if obj and obj.text != word:
            source_mark = marks.get(obj.id, {})
            # This guards old jobs whose widened native bounds or corrected text can
            # no longer locate a unique substring in the original OCR line.
            if (not obj.text or word.count(obj.text) != 1 or source_mark.get('ocr_ids') != [lid]
                    or source_mark.get('text') != obj.text):
                item['reason'] = 'ambiguous_source_field'
                continue
            box = source_mark['bbox']
            if intersection(box, line['bbox']) < box[2]*box[3]*.95:
                item['reason'] = 'field_outside_source_line'
                continue
            peers = [marks[peer]['bbox'] for peer in line_owners[lid] if peer != obj.id and peer in marks]
            if (iou(box, line['bbox']) >= .95 or
                    any(intersection(box,b) > .25*min(box[2]*box[3], b[2]*b[3]) for b in peers)):
                item['reason'] = 'field_overlaps_neighbor'
                continue
            start = word.index(obj.text)
            span = [start, start+len(obj.text)]
            word = obj.text
        x, y, width, height = map(round, box)
        if min(width, height) < 1 or min(x, y) < 0 or x+width > scene.image_size[0] or y+height > scene.image_size[1]:
            item['reason'] = 'invalid_source_box'
            continue
        if (oid in intrinsic or (obj and obj.id in intrinsic) or line.get('is_logo')
                or any(intersection(box, b) >= .95*width*height for b in picture_boxes)):
            item['reason'] = 'intrinsic_image_text'
            continue
        if (line.get('reading_status') or (not span and line.get('field_readings'))
                or (obj and scene.selected.get(obj.id, {}).get('source_reading'))):
            item['reason'] = 'already_reviewed_field'
            continue
        target_id = obj.id if obj else lid
        item['target_id'] = target_id
        item['ocr_id'] = lid
        field = dict(line, bbox=[x, y, width, height], text=word)
        if span:
            field['source_span'] = span
            field.pop('ocr_raw_text', None)
        writable = bool(obj and not obj.spans and scene.selected.get(obj.id, {}).get('kind') == 'text')
        targets[target_id] = (field, writable)
    return targets, items


def revised_lines(lines, readings):
    """Assemble revised source lines without granting a split field credit for its siblings."""
    by_line = defaultdict(list)
    for row in readings.values():
        if row.get('after'):
            by_line[row['ocr_id']].append(row)
    adopted = {}
    for line in lines:
        rows = by_line.get(line['id'], [])
        if len(rows) == 1 and 'source_span' not in rows[0]:
            adopted[line['id']] = rows[0]['after']
            continue
        if not rows or any('source_span' not in r for r in rows):
            continue
        original = line['text']
        end = 0
        parts = []
        for row in sorted(rows, key=lambda r:r['source_span']):
            left, right = row['source_span']
            if not end <= left < right <= len(original) or original[left:right] != row['before']:
                break
            parts.extend((original[end:left], row['after']))
            end = right
        else:
            adopted[line['id']] = ''.join(parts) + original[end:]
    return adopted


def commit_readings(lines, changes):
    """Apply source spans together; sibling fields keep their wording and independent status."""
    from .state import RuntimeConflict
    grouped = defaultdict(list)
    for change in changes:
        grouped[change['ocr_id']].append(change)
    updates = {}
    for lid, edits in grouped.items():
        line = lines[lid]
        if len(edits)==1 and 'source_span' not in edits[0]:
            edit=edits[0]
            if line['text']!=edit['before'] or line.get('field_readings'):
                raise RuntimeConflict('source OCR changed before commit')
            status=edit.get('status','source_supported')
            updates[lid]={'ocr_raw_text':edit['before'],'text':edit['after'],
                          'proofread_verified':status!='pending_confirmation','reading_status':status}
            continue
        original=line.get('ocr_raw_text',line['text'])
        history=dict(line.get('field_readings',{}))
        if history:
            previous=revised_lines([{'id':lid,'text':original}],history).get(lid)
            if previous!=line['text']:
                raise RuntimeConflict('source OCR changed before split commit')
        for edit in edits:
            oid=edit['object_id']
            if 'source_span' not in edit or oid in history or line.get('reading_status'):
                raise RuntimeConflict('conflicting source reading spans')
            history[oid]={k:edit[k] for k in ('ocr_id','source_span','before','after','status') if k in edit}
        text=revised_lines([{'id':lid,'text':original}],history).get(lid)
        if text is None:
            raise RuntimeConflict('source OCR span changed before commit')
        updates[lid]={'text':text,'ocr_raw_text':original,'field_readings':history}
    for lid, changes in updates.items():
        lines[lid].update(changes)
