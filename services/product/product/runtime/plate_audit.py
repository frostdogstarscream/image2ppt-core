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


def reconcile(lines, detected, fields, marks, readings):
    rows = []
    seen = set()
    for line in lines:
        oid = line['id']
        owners = [m['id'] for m in marks if m.get('label') == 'text' and oid in m.get('ocr_ids', ())]
        recognized = match(line, detected)
        field = match({**line, 'text':readings.get(oid, line['text'])}, fields)
        if recognized is not None:
            seen.add(recognized)
        status = ('intrinsic' if line.get('is_logo') or line.get('intrinsic') else
                  'residual' if field is not None and recognized is not None else
                  'native' if field is not None else
                  'retained' if recognized is not None else 'unresolved')
        rows.append({'ocr_id':oid,'bbox':line['bbox'],'text':line['text'],'status':status,
                     'mark_ids':owners,'plate_reading':detected[recognized]['text'] if recognized is not None else None})
    for index, item in enumerate(detected):
        if index not in seen:
            rows.append({'ocr_id':None,'bbox':item['bbox'],'text':item['text'],'status':'plate_only','mark_ids':[]})
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
