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


def native_line_fields(line, fields):
    """Return the physical native fields spelling one horizontal OCR line, in order."""
    text = _norm(line['text'])
    x, y, w, h = line['bbox']
    if not text or w <= 0 or h <= 0:
        return []
    pieces = []
    for field in fields:
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


def reconcile(lines, detected, fields, marks, readings):
    rows = []
    seen = set()
    for line in lines:
        oid = line['id']
        owners = [m['id'] for m in marks if m.get('label') == 'text' and oid in m.get('ocr_ids', ())]
        recognized = match(line, detected)
        expected = {**line, 'text':readings.get(oid, line['text'])}
        exact = native_line_fields(expected, fields)
        has_field = bool(exact) or match(expected, fields) is not None
        if recognized is not None:
            seen.add(recognized)
        status = ('intrinsic' if line.get('is_logo') or line.get('intrinsic') else
                  'residual' if has_field and recognized is not None else
                  'native' if has_field else
                  'retained' if recognized is not None else 'unresolved')
        rows.append({'ocr_id':oid,'bbox':line['bbox'],'text':line['text'],'status':status,
                     'mark_ids':owners, 'exact_native_fields':[f['name'] for f in exact if 'name' in f],
                     'plate_reading':detected[recognized]['text'] if recognized is not None else None})
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


def audit(job, deck, sha, page, snapshot_row, lines, fields, marks, readings):
    directory = Path(job)/'plate-audit'/sha/f'p{page}'
    directory.mkdir(parents=True, exist_ok=True)
    try:
        plate = named_plate(deck, page)
        if list(plate.size) != list(snapshot_row['image_size']):
            raise ValueError('plate and source dimensions differ')
        path = directory/'visible-plate.png'
        plate.save(path)
        evidence = local_ocr(snapshot(path, directory/'snapshots'), Path(job)/'cache', os.environ['IMAGE2PPT_OCR_MANIFEST'])
        rows = reconcile(lines, evidence['lines'], fields, marks, readings)
        return {'status':'MEASURED','pptx_sha256':sha,'source_sha256':snapshot_row['source_sha256'],
                'plate_sha256':digest(path),'ocr_identity':evidence['identity'],'rows':rows}
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return {'status':'UNKNOWN','reason':type(exc).__name__,'pptx_sha256':sha,'rows':[]}
