"""Read actual OOXML and edit only diagnostic copies; never rewrite source IR."""
import posixpath
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from uuid import uuid4
from PIL import Image, ImageChops, ImageStat
from .contracts import BuildManifest, CheckResult, ProbeCases
from .storage import digest, inventory, local_file, read_json, verify_files, write_json

NS = {'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def q(name):
    prefix, local = name.split(':')
    return '{' + NS[prefix] + '}' + local


def part_target(owner, target):
    part = posixpath.normpath(posixpath.join(posixpath.dirname(owner), target)) if not target.startswith('/') else target.lstrip('/')
    if part.startswith('../') or part == '..':
        raise ValueError('relationship escapes package')
    return part


def relationships(z, owner):
    file = posixpath.join(posixpath.dirname(owner), '_rels', posixpath.basename(owner) + '.rels')
    if file not in z.namelist():
        return {}
    return {n.get('Id'): part_target(owner, n.get('Target')) for n in ET.fromstring(z.read(file))
            if n.get('TargetMode') != 'External'}


def package_parts(z):
    names = z.namelist()
    if len(names) != len(set(names)) or sum(i.file_size for i in z.infolist()) > 512_000_000:
        raise ValueError('duplicate parts or oversized PPTX')
    if z.testzip() is not None:
        raise ValueError('PPTX CRC failed')
    for file in names:
        if file.endswith('.rels'):
            if file == '_rels/.rels':
                owner = ''
            else:
                directory, name = file.rsplit('/_rels/', 1)
                owner = directory + '/' + name[:-5]
            for n in ET.fromstring(z.read(file)):
                if n.get('TargetMode') != 'External' and part_target(owner, n.get('Target')) not in names:
                    raise ValueError('missing relationship target in ' + file)
    presentation = ET.fromstring(z.read('ppt/presentation.xml'))
    rels = relationships(z, 'ppt/presentation.xml')
    slides = [rels[n.get(q('r:id'))] for n in presentation.findall('p:sldIdLst/p:sldId', NS)]
    if not slides:
        raise ValueError('no slides')
    size = presentation.find('p:sldSz', NS)
    return slides, [int(size.get('cx')), int(size.get('cy'))]


def object_name(element):
    nv = element.find('.//p:cNvPr', NS)
    return nv.get('name') if nv is not None else None


def style_value(props, attr):
    for prop in props:
        if prop is None:
            continue
        if attr == 'bold' and prop.get('b') is not None:
            return prop.get('b') in ('1', 'true')
        if attr == 'color':
            fill = prop.find('a:solidFill', NS)
            if fill is not None:
                rgb = fill.find('a:srgbClr', NS)
                # An explicit unresolved color overrides lower-priority defaults.
                return rgb.get('val', '').upper() if rgb is not None and len(rgb)==0 else None
            if any(prop.find('a:'+kind, NS) is not None for kind in ('noFill','gradFill','blipFill','pattFill','grpFill')):
                return None
    return None  # theme/master inheritance is deliberately not guessed


def inspect_pptx(file):
    with ZipFile(file) as z:
        parts, size = package_parts(z)
        pages = []
        for slide_no, part in enumerate(parts, 1):
            root = ET.fromstring(z.read(part))
            rels = relationships(z, part)
            objects, names, ids = [], set(), set()

            def walk(parent, parent_group=None):
                for element in parent:
                    kind = element.tag.rsplit('}', 1)[-1]
                    if kind not in ('sp','pic','grpSp','cxnSp','graphicFrame'):
                        continue
                    nv = element.find('.//p:cNvPr', NS)
                    if nv is None:
                        raise ValueError('object lacks identity')
                    name, oid = nv.get('name'), nv.get('id')
                    if name in names or oid in ids:
                        raise ValueError('duplicate object identity on slide')
                    names.add(name); ids.add(oid)
                    transform = element.find('p:grpSpPr/a:xfrm' if kind == 'grpSp' else 'p:spPr/a:xfrm', NS)
                    transform_data = {} if transform is None else {n.tag.rsplit('}',1)[-1]: {k:int(v) for k,v in n.attrib.items()} for n in transform}
                    off, ext = transform_data.get('off', {}), transform_data.get('ext', {})
                    obj = {'name':name,'object_id':oid,'kind':kind,'parent_group':parent_group,'z':len(objects),
                           'transform':transform_data,'bbox_emu':[off.get('x'),off.get('y'),ext.get('cx'),ext.get('cy')],
                           'text':None,'runs':[]}
                    body = element.find('p:txBody', NS)
                    if body is not None:
                        paragraphs=[]
                        for pidx, paragraph in enumerate(body.findall('a:p', NS)):
                            level = paragraph.find('a:pPr', NS)
                            lvl = int(level.get('lvl', '0')) + 1 if level is not None else 1
                            defaults = [paragraph.find('a:pPr/a:defRPr',NS),body.find(f'a:lstStyle/a:lvl{lvl}pPr/a:defRPr',NS)]
                            text=''
                            for run in paragraph:
                                if run.tag == q('a:br'):
                                    text += '\n'
                                elif run.tag in (q('a:r'),q('a:fld')):
                                    value = run.findtext('a:t', '', NS)
                                    props = [run.find('a:rPr',NS)] + defaults
                                    obj['runs'].append({'text':value,'paragraph':pidx,'color':style_value(props,'color'),'bold':style_value(props,'bold')})
                                    text += value
                            paragraphs.append(text)
                        obj['text']='\n'.join(paragraphs)
                    if kind == 'pic':
                        blip=element.find('.//a:blip',NS)
                        media=rels.get(blip.get(q('r:embed'))) if blip is not None else None
                        obj['media']=media
                    objects.append(obj)
                    if kind == 'grpSp':
                        walk(element,name)
            walk(root.find('p:cSld/p:spTree',NS))
            pages.append({'slide':slide_no,'part':part,'objects':objects})
        return {'pptx_sha256':digest(file),'crc_passed':True,'relationships_passed':True,
                'slide_count':len(pages),'size_emu':size,'slides':pages}


def check(id, status, method, targets=None, **evidence):
    return CheckResult(id=id,status=status,method=method,targets=targets or [],evidence=evidence).model_dump()


def evaluate(ir, objects):
    actual = {o['name']:o for o in objects['slides'][0]['objects']}
    results=[check('package', 'PASS','PPTX CRC, relationships and object identities')]
    expected_names={n['id'] for n in ir['nodes']} | {g['id'] for g in ir['groups']}
    if ir['background'].get('asset_id'):
        expected_names.add('background-image')
    results.append(check('object_inventory','PASS' if expected_names == set(actual) else 'FAIL','exact IR/object mapping',
                         missing=sorted(expected_names-set(actual)),unexpected=sorted(set(actual)-expected_names)))
    for node in ir['nodes']:
        obj=actual.get(node['id'])
        if obj is not None:
            expected_kind='pic' if node['type']=='image' else 'sp'
            valid=obj['kind']==expected_kind and (node['type']!='text' or obj['text'] is not None)
            results.append(check('object_type:'+node['id'],'PASS' if valid else 'FAIL','IR type versus actual OOXML object kind',
                                 [node['id']],expected=expected_kind,actual=obj['kind']))
        if node['type']!='text' or obj is None:
            continue
        same=node['text']==obj['text']
        results.append(check('text:'+node['id'],'PASS' if same else 'FAIL','exact text including explicit line breaks',[node['id']]))
        if node.get('runs') and same:
            expected=[(r.get('color',node['color']).upper(),r.get('bold',node.get('bold',False))) for r in node['runs'] for _ in r['text']]
            observed=[(r['color'],r['bold']) for r in obj['runs'] for _ in r['text']]
            property_mismatches={name:[i for i,(a,b) in enumerate(zip(expected,observed)) if b[j] is not None and a[j]!=b[j]]
                                 for j,name in enumerate(('color','bold'))}
            unknown_properties={name:[i for i,b in enumerate(observed) if b[j] is None] for j,name in enumerate(('color','bold'))}
            mismatch=sorted(set(property_mismatches['color']+property_mismatches['bold']))
            unknown=any(None in value for value in observed)
            status='FAIL' if mismatch or len(expected)!=len(observed) else 'REVIEW_REQUIRED' if unknown else 'PASS'
            results.append(check('run_style:'+node['id'],status,'effective explicit OOXML run styles; unresolved inheritance is not guessed',
                                 [node['id']],mismatch_character_indices=mismatch,unknown_inheritance=unknown,
                                 property_mismatches=property_mismatches,unknown_properties=unknown_properties,
                                 expected_styles=[list(v) for v in dict.fromkeys(expected)],actual_styles=[list(v) for v in dict.fromkeys(observed)]))
    for group in ir['groups']:
        obj=actual.get(group['id'])
        results.append(check('group_type:'+group['id'],'PASS' if obj and obj['kind']=='grpSp' else 'FAIL',
                             'IR group versus actual OOXML group',[group['id']]))
    results.append(check('source_completeness','NOT_RUN','requires independent source annotation; declared ownership alone is insufficient'))
    results.append(check('visual_review','REVIEW_REQUIRED','actual-file preview provided; no automatic semantic visual acceptance'))
    for software in ('PowerPoint','WPS','human_acceptance'):
        results.append(check(software,'NOT_RUN','target-application/manual acceptance is separate'))
    return {'quality_status':'REVIEW_REQUIRED','checks':results,'known_failures':[c['id'] for c in results if c['status']=='FAIL'],
            'image_fallbacks':[{'id':n['id'],'editable':'move_resize_only'} for n in ir['nodes'] if n['type']=='image'],
            'powerpoint_checked':False,'wps_checked':False,'human_acceptance':False}


def edit_copy(source, target, case):
    before=inspect_pptx(source)
    page=before['slides'][case.slide-1]
    records={o['name']:o for o in page['objects']}
    with ZipFile(source) as original:
        root=ET.fromstring(original.read(page['part']))
        parents={child:parent for parent in root.iter() for child in parent}
        elements={object_name(e):e for e in root.iter() if e.tag in {q('p:sp'),q('p:grpSp'),q('p:pic')}}
        for name in case.targets:
            if name not in records or name not in elements:
                raise ValueError('missing probe target: '+name)
            element=elements[name]
            if case.action=='delete_text':
                if element.tag!=q('p:sp') or element.find('p:txBody',NS) is None:
                    raise ValueError('delete_text requires a native text object')
                parents[element].remove(element)
            elif case.action=='move_group':
                if element.tag!=q('p:grpSp'):
                    raise ValueError('move_group requires actual group')
                off=element.find('p:grpSpPr/a:xfrm/a:off',NS)
                if off is None:
                    raise ValueError('group has no transform')
                for axis,delta in zip(('x','y'),case.delta_px):
                    off.set(axis,str(int(off.get(axis))+round(delta*9525)))
            else:
                if element.tag!=q('p:sp'):
                    raise ValueError('change_fill requires a native shape')
                props=element.find('p:spPr',NS)
                if props is None:
                    raise ValueError('shape has no properties')
                for child in list(props):
                    if child.tag in {q('a:'+s) for s in ('solidFill','gradFill','noFill','blipFill','pattFill','grpFill')}:
                        props.remove(child)
                fill=ET.Element(q('a:solidFill'))
                ET.SubElement(fill,q('a:srgbClr'),{'val':case.fill.upper()})
                # Fill follows xfrm and geometry and precedes line/effects.
                index=next((i for i,c in enumerate(props) if c.tag not in {q('a:xfrm'),q('a:prstGeom'),q('a:custGeom')}),len(props))
                props.insert(index,fill)
        with ZipFile(target,'x',compression=ZIP_DEFLATED) as output:
            for entry in original.infolist():
                output.writestr(entry,ET.tostring(root,encoding='utf-8',xml_declaration=True) if entry.filename==page['part'] else original.read(entry.filename))
    after=inspect_pptx(target)
    current={o['name']:o for o in after['slides'][case.slide-1]['objects']}
    if case.action=='delete_text':
        if any(name in current for name in case.targets) or set(current)!=set(records)-set(case.targets):
            raise ValueError('delete probe object inventory mismatch')
    elif case.action=='move_group':
        for name in case.targets:
            for axis,delta in zip(('x','y'),case.delta_px):
                if current[name]['transform']['off'][axis]!=records[name]['transform']['off'][axis]+round(delta*9525):
                    raise ValueError('group move failed')
            for member in records.values():
                if member['parent_group']==name and current[member['name']]['transform']!=member['transform']:
                    raise ValueError('group child transform changed')
    else:
        with ZipFile(target) as z:
            root=ET.fromstring(z.read(page['part']))
            for name in case.targets:
                item=next(e for e in root.iter(q('p:sp')) if object_name(e)==name)
                if item.find('p:spPr/a:solidFill/a:srgbClr',NS).get('val')!=case.fill.upper():
                    raise ValueError('fill change failed')
    return after


def probe(build_file, cases_file, backend):
    build_file=Path(build_file).resolve(strict=True)
    manifest=BuildManifest.model_validate(read_json(build_file))
    verify_files(build_file.parent,manifest.files)
    cases=ProbeCases.model_validate(read_json(cases_file))
    base=build_file.parent
    source=local_file(base,'result.pptx')
    initial_hash=digest(source)
    directory=base.parent.parent/'probes'/('probe-'+uuid4().hex[:12])
    directory.mkdir(parents=True)
    write_json(directory/'cases.json',cases.model_dump(),exclusive=True)
    original_map=inspect_pptx(source)
    results=[]
    for case in cases.cases:
        target=directory/case.id
        target.mkdir()
        try:
            final=target/'probe.pptx'
            edited=edit_copy(source,final,case)
            write_json(target/'objects.before.json',original_map,exclusive=True)
            write_json(target/'objects.after.json',edited,exclusive=True)
            render=backend.render(final,target/'render')
            if len(render['pages'])!=original_map['slide_count']:
                raise ValueError('probe page count changed')
            checks=[check('operation','PASS','direct OOXML diagnostic copy',case.targets),
                    check('semantic_visual','REVIEW_REQUIRED','source and actual-file probe require visual comparison')]
            before=base/'preview.png'
            after=Path(render['pages'][case.slide-1]['path'])
            if case.roi:
                x,y,w,h=case.roi
                with Image.open(before) as a, Image.open(after) as b:
                    if a.size!=b.size or x+w>a.width or y+h>a.height:
                        raise ValueError('probe ROI outside renderer canvas')
                    box=(x,y,x+w,y+h)
                    a.crop(box).save(target/'before.png')
                    b.crop(box).save(target/'after.png')
                    diff=ImageChops.difference(a.convert('RGB').crop(box),b.convert('RGB').crop(box))
                    diff.save(target/'difference.png')
                    checks.append(check('pixel_difference','REVIEW_REQUIRED','difference is diagnostic, not semantic proof',mean=sum(ImageStat.Stat(diff).mean)/3))
                    if case.expected_solid:
                        crop=b.convert('RGB').crop(box)
                        delta=ImageChops.difference(crop,Image.new('RGB',crop.size,'#'+case.expected_solid))
                        mean=sum(ImageStat.Stat(delta).mean)/3
                        checks.append(check('labelled_residue','PASS' if mean<=1 else 'FAIL','synthetic labelled uniform-background ROI',mean=mean,threshold=1))
            if case.related_image:
                obj=next(o for o in original_map['slides'][case.slide-1]['objects'] if o['name']==case.related_image)
                objects={o['name']:o for o in original_map['slides'][case.slide-1]['objects']}
                def page_box(item):
                    x,y,w,h=item['bbox_emu']
                    parent=item['parent_group']
                    while parent:
                        group=objects[parent]; transform=group['transform']
                        sx=transform['ext']['cx']/transform['chExt']['cx']
                        sy=transform['ext']['cy']/transform['chExt']['cy']
                        x=transform['off']['x']+(x-transform['chOff']['x'])*sx
                        y=transform['off']['y']+(y-transform['chOff']['y'])*sy
                        w*=sx; h*=sy; parent=group['parent_group']
                    return [v/9525 for v in (x,y,w,h)]
                image_box=page_box(obj); relations=[]
                for name in case.targets:
                    other=objects[name]; box=page_box(other)
                    overlap=max(0,min(image_box[0]+image_box[2],box[0]+box[2])-max(image_box[0],box[0]))*max(0,min(image_box[1]+image_box[3],box[1]+box[3])-max(image_box[1],box[1]))
                    relations.append({'target':name,'text':other['text'],'bbox_px':box,'intersection_area_px2':overlap,
                                      'image_drawn_after_target':obj['z']>other['z'],'target_z':other['z']})
                checks.append(check('image_relationships','REVIEW_REQUIRED','actual package geometry and paint order; overlap is risk evidence only',
                                    case.targets,image=case.related_image,image_bbox_px=image_box,image_z=obj['z'],relations=relations))
                from io import BytesIO
                with ZipFile(source) as z, Image.open(BytesIO(z.read(obj['media']))) as icon:
                    rgba=icon.convert('RGBA')
                    icon.save(target/'related-image.png')
                    for label,color in [('white','white'),('dark','#172B4D')]:
                        bg=Image.new('RGBA',rgba.size,color); bg.alpha_composite(rgba); bg.convert('RGB').save(target/(label+'.png'))
                    grid=Image.new('RGBA',rgba.size,'white')
                    from PIL import ImageDraw
                    draw=ImageDraw.Draw(grid)
                    for y in range(0,grid.height,8):
                        for x in range(0,grid.width,8):
                            if (x//8+y//8)%2: draw.rectangle((x,y,x+7,y+7),fill='#B8B8B8')
                    grid.alpha_composite(rgba); grid.convert('RGB').save(target/'checkerboard.png')
                    border=rgba.crop((0,0,rgba.width,1)).convert('RGB')
                    checks.append(check('image_boundary','REVIEW_REQUIRED','alpha and border diagnostics do not establish a defect',
                                        [case.related_image],mode=icon.mode,alpha_extrema=list(rgba.getchannel('A').getextrema()),top_border_mean=ImageStat.Stat(border).mean))
            files=inventory(target,[p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()])
            results.append({'id':case.id,'execution_status':'COMPLETED','checks':checks,'files':files})
        except Exception as error:
            results.append({'id':case.id,'execution_status':'FAILED','error':str(error),'checks':[check('probe','FAIL',type(error).__name__)]})
    if digest(source)!=initial_hash:
        raise ValueError('original PPTX changed during probe')
    report={'build_sha256':digest(build_file),'source_pptx_sha256':initial_hash,'original_preserved':True,
            'method':'direct_final_pptx_copy','cases':results,'executed_count':len(results),'human_acceptance':False,
            'execution_status':'FAILED' if any(r['execution_status']=='FAILED' for r in results) else 'COMPLETED'}
    write_json(directory/'report.json',report,exclusive=True)
    return {'report':str(directory/'report.json'),**report}
