"""Convert a validated model page to fixed SlideIR; never execute model code."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
from uuid import uuid4
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image, ImageFont
from fontTools.ttLib import TTFont
from .contracts import Page, intersection


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('tmp-'+uuid4().hex[:8])
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        for attempt in range(50):
            try:temporary.replace(path);break
            except PermissionError:
                if attempt==49:raise
                time.sleep(.02)
    finally:temporary.unlink(missing_ok=True)


def font_gate(page, manifest):
    records = {(f['family'],f['bold']):f for f in manifest['fonts']}
    styles = {s.id:s for s in page.styles}
    cmaps = {}; faces = {}
    for s in page.styles:
        for name in (s.ea,s.latin):
            if (name,s.bold) not in records:
                raise ValueError('font/weight unavailable: '+name)
    def face(style, char):
        name = style.ea if ord(char)>255 else style.latin
        f = records[name,style.bold]
        key = (f['file'],f['index'])
        if key not in cmaps:
            with TTFont(f['file'],fontNumber=f['index'],lazy=True) as font:
                cmaps[key] = font.getBestCmap() or {}
        if not char.isspace() and ord(char) not in cmaps[key]:
            raise ValueError('font does not cover glyph in '+style.id)
        cache_key=(*key,round(style.size_px))
        if cache_key not in faces:
            faces[cache_key]=ImageFont.truetype(f['file'],max(1,round(style.size_px)),index=f['index'])
        return faces[cache_key]
    errors=[]
    for text in page.texts:
        style = styles[text.style]
        spans = [(s.text,styles[s.style]) for s in text.spans] or [(text.text,style)]
        widths = [0.0]; heights = [style.size_px]
        for content,s in spans:
            for c in content:
                if c=='\n':
                    widths.append(0.0); heights.append(s.size_px); continue
                font = face(s,c)
                widths[-1] += font.getlength(c)
                heights[-1] = max(heights[-1],s.size_px)
        if max(widths)>text.bbox[2]+3:
            errors.append(f'text {text.id} unbroken line needs width >= {max(widths):.1f}px; current bbox width={text.bbox[2]:.1f}px. Restore explicit newline characters at the source line boundaries, within this source text area. Do not extend a paragraph across adjacent cards or outside the page; do not shrink correct font size.')
        estimated = (len(heights)-1)*(text.line_spacing_px or max(heights)*1.2)+heights[-1]
        if estimated>text.bbox[3]+4:
            errors.append(f'text {text.id} needs height >= {estimated:.1f}px; current bbox height={text.bbox[3]:.1f}px')
    for i,a in enumerate(page.texts):
        for b in page.texts[i+1:]:
            if intersection(a.bbox,b.bbox)>min(a.bbox[2]*a.bbox[3],b.bbox[2]*b.bbox[3])*.02:
                errors.append('text boxes overlap: '+a.id+' / '+b.id)
    if errors:raise ValueError('\n'.join(errors[:40]))


def materialize(page: Page, source, directory, manifest, model, *, asset_references=None):
    source, directory = Path(source),Path(directory)
    with Image.open(source) as image:
        if list(image.size)!=page.image_size or digest(source)!=page.source_sha256:
            raise ValueError('source binding mismatch')
    font_gate(page,manifest)
    directory.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source,directory/'source.png')
    styles = {s.id:s for s in page.styles}
    def attrs(style):
        return {'font':{'ea':style.ea,'latin':style.latin,'size_pt':style.size_px*.75},
                'bold':style.bold,'color':style.color}
    ir = {'schema_version':'0.1','source_image':'source.png','source_sha256':page.source_sha256,
          'canvas':{'source_px':page.image_size,'target_px':page.image_size},
          'background':{'color':page.background},'review_status':'needs_review',
          'review_notes':['MoMA automatic proposal; no human or Office acceptance',*page.limitations],
          'nodes':[],'evidence':[],'assets':[],'groups':[]}
    for node in [*page.texts,*page.shapes,*page.crops]:
        ev = {'id':'ev-'+node.id,'bbox':node.bbox}
        out = {'id':node.id,'bbox':node.bbox,'z':node.z,'evidence_ids':[ev['id']]}
        if node in page.texts:
            ev.update(kind='text',text=node.text)
            out.update(type='text',text=node.text,**attrs(styles[node.style]),align=node.align,
                       wrap=False,insets=[0,0,0,0],vertical_anchor='top')
            # Materialize every paragraph and every run. No old overrides survive.
            paragraphs = []
            spans = [(s.text,styles[s.style]) for s in node.spans] or [(node.text,styles[node.style])]
            lines = [[]]
            for content,style in spans:
                for i,part in enumerate(content.split('\n')):
                    if i: lines.append([])
                    if part: lines[-1].append({'text':part,**attrs(style)})
            for runs in lines:
                para = {'text':''.join(r['text'] for r in runs),**attrs(styles[node.style]),'align':node.align}
                if runs: para['runs']=runs
                if node.line_spacing_px: para['line_spacing_pt']=node.line_spacing_px*.75
                paragraphs.append(para)
            out['paragraphs']=paragraphs
        elif node in page.shapes:
            ev.update(kind='shape',description=node.geometry)
            out.update(type='shape',geometry=node.geometry,fill=node.fill,
                       line={'color':node.line_color,'width_pt':node.line_width_pt})
            if node.fill_gradient is not None:out['fill_gradient']=node.fill_gradient.model_dump()
            if node.corner_radius_px is not None:
                if node.corner_radius_px>min(node.bbox[2:])/2:raise ValueError('corner radius exceeds shape')
                out['corner_radius_px']=node.corner_radius_px
        else:
            ev.update(kind='image',description=node.description)
            asset='asset-'+node.id
            file=directory/(asset+'.png')
            x,y,w,h=node.bbox
            ref=(asset_references or {}).get(node.id)
            if ref is not None:
                if ref['source_sha256']!=page.source_sha256 or ref['bbox']!=node.bbox:
                    raise ValueError('stable asset scene binding changed; revalidate ownership')
                if digest(ref['path'])!=ref['sha256']:raise ValueError('stable asset hash mismatch')
                shutil.copyfile(ref['path'],file)
            else:
                if node.id.startswith(('extracted-','restored-bg-')):raise ValueError('transparent asset reference missing; source recrop forbidden')
                with Image.open(source) as image:
                    image.crop((round(x),round(y),round(x+w),round(y+h))).save(file)
            ir['assets'].append({'id':asset,'path':file.name,'purpose':'module','sha256':digest(file),
                                 'contains_evidence_ids':[ev['id']],'description':node.description})
            out.update(type='image',asset_id=asset,evidence_ids=[])
        ir['evidence'].append(ev);ir['nodes'].append(out)
    ir['nodes'].sort(key=lambda n:n['z'])
    for z,node in enumerate(ir['nodes']):node['z']=z
    file=directory/'slideir.json';write(file,ir)
    write(directory/'authoring.json',{'tool':'image2PPT product worker','model_id':model,
                                     'ir_sha256':digest(file),'decisions':['VisualPage/1 compiled as data']})
    return file


def verify_package(pptx, pages):
    """Final OOXML must contain exactly the authored text and effective run styles."""
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
        'p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
    with zipfile.ZipFile(pptx) as archive:
        for i,page in enumerate(pages,1):
            xml=ET.fromstring(archive.read(f'ppt/slides/slide{i}.xml'))
            expected=Counter(t.text for t in page.texts)
            actual=Counter()
            shapes=[]
            for sp in xml.findall('.//p:sp',ns):
                paras=sp.findall('p:txBody/a:p',ns)
                if paras:
                    text='\n'.join(''.join(t.text or '' for t in p.findall('.//a:t',ns)) for p in paras)
                    if text: actual[text]+=1;shapes.append((text,paras))
            if expected!=actual: raise ValueError('final PPTX text differs from plan')
            # Compare complete run sequences, retaining duplicate text occurrences.
            styles={s.id:s for s in page.styles}
            wanted=Counter()
            for t in page.texts:
                spans=[(s.text,styles[s.style]) for s in t.spans] or [(t.text,styles[t.style])]
                signature=tuple((c,s.ea,s.latin,int(s.size_px*.75*100+.5),s.bold,s.color.upper())
                                for content,s in spans for c in content if c!='\n')
                wanted[(t.text,signature)]+=1
            got=Counter()
            for text,paras in shapes:
                signature=[]
                for p in paras:
                    for r in p.findall('a:r',ns):
                        prop=r.find('a:rPr',ns)
                        if prop is None: raise ValueError('missing explicit run style')
                        ea=prop.find('a:ea',ns);latin=prop.find('a:latin',ns);color=prop.find('a:solidFill/a:srgbClr',ns)
                        if ea is None or latin is None or color is None: raise ValueError('incomplete run style')
                        signature.extend((c,ea.get('typeface'),latin.get('typeface'),int(prop.get('sz','0')),
                                          prop.get('b','0') in ('1','true'),color.get('val','').upper())
                                         for c in r.findtext('a:t','',ns))
                got[(text,tuple(signature))]+=1
            if wanted!=got: raise ValueError('final PPTX run style mismatch')
