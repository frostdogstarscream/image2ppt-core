"""Read the named background in the delivered package and reconcile it with native fields."""
from difflib import SequenceMatcher
from io import BytesIO
import os
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from PIL import Image

from ..materialize import digest
from ..parallel_branches import local_ocr, snapshot
from ..scorecard import NS, _norm, _overlap, _part_rels, _q, _slide_parts


def named_plate(deck, page):
    with ZipFile(deck) as archive:
        part = _slide_parts(archive)[page - 1]
        rels = _part_rels(archive, part)
        matches = []
        for picture in ET.fromstring(archive.read(part)).findall('.//p:pic', NS):
            name = picture.find('p:nvPicPr/p:cNvPr', NS)
            if name is not None and name.get('name') == 'source-plate':
                blip = picture.find('p:blipFill/a:blip', NS)
                matches.append(rels[blip.get(_q('r:embed'))])
        if len(matches) != 1:
            raise ValueError('expected one named source plate')
        with Image.open(BytesIO(archive.read(matches[0]))) as image:
            rgba = image.convert('RGBA')
            white = Image.new('RGBA', rgba.size, 'white')
            return Image.alpha_composite(white, rgba).convert('RGB')


def match(line, candidates):
    box = line['bbox']
    value = _norm(line['text'])
    matches = []
    for index, item in enumerate(candidates):
        other = item['bbox']
        overlap = _overlap(box, other) / max(1, box[2]*box[3], other[2]*other[3])
        reading = _norm(item['text'])
        similarity = SequenceMatcher(None, value, reading, autojunk=False).ratio()
        if value and reading and overlap >= .45 and similarity >= .65:
            matches.append((overlap * similarity, index))
    return max(matches)[1] if matches else None


def bind_native_paragraphs(lines, fields, members, readings):
    """Bind complete native paragraphs to their actual source rows, including punctuation."""
    by_id = {r['id']: r for r in lines}
    result = []
    for field in fields:
        ids = members.get(field.get('name'), ())
        if len(ids) > 1 and len(set(ids)) == len(ids) and all(i in by_id for i in ids):
            rows = [by_id[i] for i in ids]
            wording = ''.join(_norm(readings.get(i, by_id[i]['text'])) for i in ids)
            if (wording == _norm(field['text'])
                    and all(_overlap(r['bbox'], field['bbox']) >= .5*r['bbox'][2]*r['bbox'][3] for r in rows)
                    and all(a['bbox'][1] + .4*a['bbox'][3] < b['bbox'][1] for a, b in zip(rows, rows[1:]))):
                field = {**field, 'source_rows': {i: _norm(readings.get(i, by_id[i]['text'])) for i in ids}}
        result.append(field)
    return result


def bind_separate_symbols(fields, members, source_fields, graphics):
    """Credit a split label only when its isolated symbol also exists in the edited deck."""
    result = []
    for field in fields:
        ids = members.get(field.get('name'), ())
        entry = source_fields.get(ids[0], {}) if len(ids) == 1 else {}
        symbol = entry.get('separate_symbol')
        if not symbol or _norm(field['text']) != _norm(entry['render_text']):
            result.append(field)
            continue
        original = _norm(entry['text'])
        body = _norm(entry['render_text'])
        x, y, w, h = entry['bbox']
        fx, _, fw, _ = entry['render_bbox']
        if original == body + symbol:
            left, right = fx + fw, x + w
        elif original == symbol + body:
            left, right = x, fx
        else:
            result.append(field)
            continue
        slot = [left, y, right-left, h]
        if (slot[2] > 0 and _overlap(field['bbox'], entry['render_bbox']) >= .5 * fw * h):
            drawn = [g for g in graphics if _overlap(slot, g['bbox']) >=
                     max(.8*g['bbox'][2]*g['bbox'][3], .15*slot[2]*h)]
            if len(drawn) == 1:
                field = {**field, 'separate_symbol': {'ocr_id':ids[0], 'source_text':original,
                         'symbol':symbol, 'object':drawn[0]['name'], 'bbox':drawn[0]['bbox']}}
        result.append(field)
    return result


def native_line_fields(line, fields):
    """Return the physical native fields spelling one horizontal OCR line, in order."""
    text = _norm(line['text'])
    x, y, w, h = line['bbox']
    if not text or w <= 0 or h <= 0:
        return []
    pieces = []
    for field in fields:
        if field.get('source_rows', {}).get(line.get('id')) == text:
            return [field]
        symbol = field.get('separate_symbol', {})
        if symbol.get('ocr_id') == line.get('id') and symbol.get('source_text') == text:
            return [field]
        fx, fy, fw, fh = field['bbox']
        value = _norm(field['text'])
        overlap = _overlap(line['bbox'], field['bbox'])
        if value == text and overlap > .5 * w * h:
            return [field]
        if (not value or fw <= 0 or fh <= 0 or overlap < .5 * fw * fh
                or abs((fy + fh/2) - (y + h/2)) > .45 * h
                or not x <= fx + fw/2 <= x + w):
            continue
        # Table cells are independent fields; do not join adjacent cells or rows.
        pieces.append((fx, fx + fw, field))
    if len(pieces) < 2:
        return []
    pieces.sort(key=lambda item: item[0])
    if any('class' not in field and field.get('name', '').endswith(']') for _, _, field in pieces):
        return []
    if any(left < prev_right - .3*h or left > prev_right + 2*h
           for (_, prev_right, _), (left, _, _) in zip(pieces, pieces[1:])):
        return []
    if pieces[0][0] > x + h or pieces[-1][1] < x + w - h:
        return []
    ordered = [field for _, _, field in pieces]
    return ordered if ''.join(_norm(f['text']) for f in ordered) == text else []


def _visible_fragments(line, detected):
    whole = _norm(line['text'])
    hits = set()
    direct = match(line, detected)
    if direct is not None:
        hits.add(direct)
    for i, item in enumerate(detected):
        fragment = _norm(item['text'])
        box = item['bbox']
        if (len(fragment) >= 2 and fragment in whole
                and _overlap(line['bbox'], box) >= .7 * max(1, box[2]*box[3])):
            hits.add(i)
    return sorted(hits)


def reconcile(lines, detected, fields, marks, readings):
    rows = []
    seen = set()
    for line in lines:
        oid = line['id']
        owners = [m['id'] for m in marks if m.get('label') == 'text' and oid in m.get('ocr_ids', ())]
        recognized = _visible_fragments(line, detected)
        expected = {**line, 'text':readings.get(oid, line['text'])}
        exact = native_line_fields(expected, fields)
        has_field = bool(exact)
        seen.update(recognized)
        symbol = exact[0].get('separate_symbol') if exact else None
        if symbol:
            expected_marks = {i for i,item in enumerate(detected) if _norm(item['text']) == symbol['symbol']
                              and _overlap(symbol['bbox'], item['bbox']) >= .7*item['bbox'][2]*item['bbox'][3]}
            seen.update(expected_marks)
            recognized = [i for i in recognized if i not in expected_marks]
        status = ('intrinsic' if line.get('is_logo') or line.get('intrinsic') else
                  'residual' if has_field and recognized else
                  'native' if has_field else
                  'retained' if recognized else 'unresolved')
        rows.append({'ocr_id':oid,'bbox':line['bbox'],'text':line['text'],'status':status,
                     'mark_ids':owners, 'exact_native_fields':[f['name'] for f in exact if 'name' in f],
                     'symbol_object':symbol['object'] if symbol else None,
                     'native_form': ('label_with_separate_symbol' if symbol else
                                     'paragraph_member' if exact and exact[0].get('source_rows') else
                                     'split_fields' if len(exact) > 1 else 'own_field' if exact else None),
                     'plate_reading':' '.join(detected[i]['text'] for i in recognized) or None})
    for index, item in enumerate(detected):
        if index not in seen:
            box = item['bbox']
            reading = item['text'].strip()
            icons = [m['id'] for m in marks if m.get('label') == 'icon'
                     and _overlap(m['bbox'], box) >= .85 * box[2] * box[3]
                     and _overlap(m['bbox'], box) >= .5 * m['bbox'][2] * m['bbox'][3]]
            source_text = any(_overlap(box, line['bbox']) > 0 for line in lines)
            ambiguous_icon = (len(reading) == 1 and not reading.isalnum() and bool(icons) and not source_text)
            rows.append({'ocr_id':None,'bbox':box,'text':reading,
                         'status':'unconfirmed_icon_reading' if ambiguous_icon else 'plate_only',
                         'mark_ids':icons if ambiguous_icon else []})
    return rows


def audit(job, deck, sha, page, snapshot_row, lines, fields, marks, readings, *, office_view, view_receipt):
    directory = Path(job)/'plate-audit'/sha/f'p{page}'
    directory.mkdir(parents=True, exist_ok=True)
    try:
        plate = named_plate(deck, page)
        if list(plate.size) != list(snapshot_row['image_size']):
            raise ValueError('plate and source dimensions differ')
        path = directory/'visible-plate.png'
        plate.save(path)
        evidence = local_ocr(snapshot(path, directory/'snapshots'), Path(job)/'cache', os.environ['IMAGE2PPT_OCR_MANIFEST'])
        if not office_view.is_file() or digest(office_view) != view_receipt.get('sha256'):
            raise ValueError('missing bound Office text-hidden export')
        with Image.open(office_view) as image:
            if list(image.size) != list(plate.size):
                raise ValueError('text-hidden Office export has the wrong size')
        composite = local_ocr(snapshot(office_view, directory/'snapshots'), Path(job)/'cache',
                              os.environ['IMAGE2PPT_OCR_MANIFEST'])
        rows = reconcile(lines, composite['lines'], fields, marks, readings)
        for row in rows:
            if row['ocr_id'] is not None:
                row['raster_reading'] = row['plate_reading']
                row['plate_reading'] = ' '.join(evidence['lines'][i]['text']
                    for i in _visible_fragments(row, evidence['lines'])) or None
        return {'status':'MEASURED','pptx_sha256':sha,'source_sha256':snapshot_row['source_sha256'],
                'plate_sha256':digest(path), 'text_hidden_sha256':view_receipt['sha256'],
                'ocr_identity':composite['identity'],'rows':rows}
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return {'status':'UNKNOWN','reason':type(exc).__name__,'pptx_sha256':sha,'rows':[]}
