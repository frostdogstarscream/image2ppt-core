"""Attach a verified SVG alternative to the same DrawingML picture as its PNG."""
import json
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import posixpath
import uuid
from lxml import etree as ET

from .materialize import digest

REL='http://schemas.openxmlformats.org/package/2006/relationships'
OFF='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
P='http://schemas.openxmlformats.org/presentationml/2006/main'
SVG='http://schemas.microsoft.com/office/drawing/2016/SVG/main'
CT='http://schemas.openxmlformats.org/package/2006/content-types'


def attach(pptx, irs):
    manifests=[Path(ir).parent/'vector-assets.json' for ir in irs]
    if not any(m.is_file() for m in manifests):return
    target=Path(pptx)
    updates={}
    with ZipFile(target) as src:
        for i,manifest in enumerate(manifests,1):
            if not manifest.is_file():continue
            slide=f'ppt/slides/slide{i}.xml'
            rels=f'ppt/slides/_rels/slide{i}.xml.rels'
            root,links=ET.fromstring(src.read(slide)),ET.fromstring(src.read(rels))
            records=json.loads(manifest.read_text(encoding='utf8'))
            for picture in root.iter('{%s}pic'%P):
                props=picture.find('.//{%s}cNvPr'%P)
                if props is None or props.get('name') not in records:continue
                row=records[props.get('name')]
                image=(manifest.parent/row['path']).resolve()
                if image.parent!=manifest.parent.resolve() or digest(image)!=row['sha256']:
                    raise ValueError('SVG file binding mismatch')
                blip=picture.find('.//{%s}blip'%A)
                embed=blip.get('{%s}embed'%OFF)
                pngrel=next(l for l in links if l.get('Id')==embed)
                pngname=posixpath.normpath(posixpath.join('ppt/slides',pngrel.get('Target'))).lstrip('/')
                import hashlib
                if hashlib.sha256(src.read(pngname)).hexdigest()!=row['png_sha256']:
                    raise ValueError('SVG fallback PNG binding mismatch')
                filename='vector-'+row['sha256']+'.svg'
                rid='rIdSvg'+uuid.uuid4().hex
                ET.SubElement(links,'{%s}Relationship'%REL,Id=rid,Target='../media/'+filename,Type=OFF+'/image')
                lst=blip.find('{%s}extLst'%A)
                if lst is None:lst=ET.SubElement(blip,'{%s}extLst'%A)
                ext=ET.SubElement(lst,'{%s}ext'%A,uri='{96DAC541-7B7A-43D3-8B79-37D633B846F1}')
                ET.SubElement(ext,'{%s}svgBlip'%SVG,{'{%s}embed'%OFF:rid},nsmap={'asvg':SVG})
                updates['ppt/media/'+filename]=image.read_bytes()
            updates[slide]=ET.tostring(root)
            updates[rels]=ET.tostring(links)
        content=ET.fromstring(src.read('[Content_Types].xml'))
        if not any(x.get('Extension')=='svg' for x in content):
            ET.SubElement(content,'{%s}Default'%CT,Extension='svg',ContentType='image/svg+xml')
        updates['[Content_Types].xml']=ET.tostring(content)
        temporary=target.with_suffix('.'+uuid.uuid4().hex+'.tmp')
        try:
            with ZipFile(temporary,'w',compression=ZIP_DEFLATED) as dest:
                for info in src.infolist():
                    if info.filename not in updates:dest.writestr(info,src.read(info))
                for name,data in updates.items():dest.writestr(name,data)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary,target)

