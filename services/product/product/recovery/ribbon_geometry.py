"""Preserve the source-measured right-pointing outline in editable DrawingML."""
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile
import json
import os

from lxml import etree as ET

P = '{http://schemas.openxmlformats.org/presentationml/2006/main}'
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'


def apply_ribbon_geometry(pptx, irs):
    updates = {}
    with ZipFile(pptx) as archive:
        for number, ir in enumerate(irs, 1):
            scene = Path(ir).parent / 'scene.json'
            if not scene.exists():
                continue
            records = json.loads(scene.read_text('utf8'))['selected']
            adjustments = {key: row['arrow_adjustment'] for key, row in records.items() if 'arrow_adjustment' in row}
            if not adjustments:
                continue
            part = f'ppt/slides/slide{number}.xml'
            root = ET.fromstring(archive.read(part))
            matched = False
            for shape in root.iter(P+'sp'):
                props = shape.find(P+'nvSpPr/'+P+'cNvPr')
                name = props.get('name') if props is not None else None
                if name not in adjustments:
                    continue
                geom = shape.find(P+'spPr/'+A+'prstGeom')
                if geom is None or geom.get('prst') != 'pentagon':
                    continue
                matched = True
                geom.set('prst', 'homePlate')
                guides = geom.find(A+'avLst')
                if guides is None:
                    guides = ET.SubElement(geom, A+'avLst')
                guides.clear()
                ET.SubElement(guides, A+'gd', name='adj', fmla=f'val {adjustments.pop(name)}')
            if adjustments:
                import logging
                logging.getLogger(__name__).warning('Missing editable ribbon geometries: %s', sorted(adjustments))
            if not matched:
                continue
            updates[part] = ET.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
        if not updates:
            return
        with NamedTemporaryFile(dir=Path(pptx).parent, delete=False) as temp:
            temporary = Path(temp.name)
        try:
            with ZipFile(temporary,'w') as output:
                for info in archive.infolist():
                    output.writestr(info, updates[info.filename] if info.filename in updates else archive.read(info))
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary,pptx)
    finally:
        temporary.unlink(missing_ok=True)
