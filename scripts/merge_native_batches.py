"""Join Artifact Tool-authored native shape/image batches, preserving IR z order.

No new visual objects are authored here. Charts, tables, hyperlinks, background
image assets and external relationships are intentionally unsupported by this
opt-in recovery path. The ordinary adapter/finalizer still runs afterwards.
"""
import argparse,json,posixpath
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from xml.etree import ElementTree as ET

NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
REL='http://schemas.openxmlformats.org/package/2006/relationships';CT='http://schemas.openxmlformats.org/package/2006/content-types'
for prefix,uri in NS.items():ET.register_namespace(prefix,uri)

def _read(path):
    with ZipFile(path) as z:
        if z.testzip():raise ValueError('invalid ZIP CRC')
        return {n:z.read(n) for n in z.namelist()}

def merge_batches(manifest_file,target):
    manifest=json.loads(Path(manifest_file).read_text(encoding='utf8'));target=Path(target)
    if target.exists():raise FileExistsError('batch target must be new')
    parts=_read(manifest['base']);content=ET.fromstring(parts['[Content_Types].xml']);slide_trees={};relations={};copied_images=0
    for number,irfile in enumerate(manifest['irs'],1):
        ir=json.loads(Path(irfile).read_text(encoding='utf8'))
        if ir['background'].get('asset_id') or any(n['type'] not in ('text','shape','image') for n in ir['nodes']):raise ValueError('unsupported batch node or background')
        slide_trees[number]=(ET.fromstring(parts[f'ppt/slides/slide{number}.xml']),ir)
        relfile=f'ppt/slides/_rels/slide{number}.xml.rels'
        relations[number]=ET.fromstring(parts[relfile]) if relfile in parts else ET.Element('{'+REL+'}Relationships')
    for index,batch in enumerate(manifest['batches'],1):
        number=batch['slide'];root,ir=slide_trees[number];tree=root.find('p:cSld/p:spTree',NS);extra=_read(batch['path'])
        other=ET.fromstring(extra['ppt/slides/slide1.xml']);other_tree=other.find('p:cSld/p:spTree',NS)
        relpath='ppt/slides/_rels/slide1.xml.rels';source_rels=ET.fromstring(extra[relpath]) if relpath in extra else []
        lookup={r.get('Id'):r for r in source_rels};mapping={};current_ids={r.get('Id') for r in relations[number]}
        highest=max(int(e.get('id','0')) for e in root.findall('.//p:cNvPr',NS))
        for child in other_tree:
            if child.tag not in ('{'+NS['p']+'}sp','{'+NS['p']+'}pic'):continue
            child=deepcopy(child)
            for element in child.iter():
                for key,value in list(element.attrib.items()):
                    if key.startswith('{'+NS['r']+'}'):
                        if value not in mapping:
                            rel=lookup.get(value)
                            if rel is None or not rel.get('Type','').endswith('/image') or rel.get('TargetMode')=='External':raise ValueError('unsupported batch relationship')
                            target_uri=rel.get('Target','')
                            old=posixpath.normpath(target_uri.lstrip('/') if target_uri.startswith('/') else posixpath.join('ppt/slides',target_uri))
                            if not old.startswith('ppt/media/') or old not in extra:raise ValueError('invalid batch image target')
                            new=f'ppt/media/batch-{index}-{posixpath.basename(old)}'
                            if new in parts and parts[new]!=extra[old]:raise ValueError('batch media collision')
                            if new not in parts:parts[new]=extra[old];copied_images+=1
                            rid=f'rIdBatch{index}_{len(mapping)+1}'
                            if rid in current_ids:raise ValueError('relationship collision')
                            current_ids.add(rid);mapping[value]=rid
                            ET.SubElement(relations[number],'{'+REL+'}Relationship',{'Id':rid,'Type':rel.get('Type'),'Target':posixpath.relpath(new,'ppt/slides')})
                            ext=posixpath.splitext(new)[1][1:]
                            if not any(c.tag=='{'+CT+'}Default' and c.get('Extension')==ext for c in content):
                                source_ct=ET.fromstring(extra['[Content_Types].xml']);match=next((c for c in source_ct if c.tag=='{'+CT+'}Default' and c.get('Extension')==ext),None)
                                if match is None:raise ValueError('missing image content type')
                                content.append(deepcopy(match))
                        element.set(key,mapping[value])
                if element.tag=='{'+NS['p']+'}cNvPr':highest+=1;element.set('id',str(highest))
            tree.append(child)
    counts=[]
    for number,(root,ir) in slide_trees.items():
        tree=root.find('p:cSld/p:spTree',NS);objects={}
        for item in list(tree):
            if item.tag not in ('{'+NS['p']+'}sp','{'+NS['p']+'}pic'):continue
            nv=item.find('.//p:cNvPr',NS)
            if nv is None or nv.get('name') in objects:raise ValueError('duplicate or missing batch object name')
            objects[nv.get('name')]=item
        ordered=sorted(ir['nodes'],key=lambda n:n['z'])
        if set(objects)!={n['id'] for n in ordered}:raise ValueError('merged batch object inventory mismatch')
        for item in objects.values():tree.remove(item)
        for node in ordered:tree.append(objects[node['id']])
        ids=[e.get('id') for e in root.findall('.//p:cNvPr',NS)]
        if len(ids)!=len(set(ids)):raise ValueError('duplicate native object numeric ID')
        parts[f'ppt/slides/slide{number}.xml']=ET.tostring(root,encoding='utf-8',xml_declaration=True)
        parts[f'ppt/slides/_rels/slide{number}.xml.rels']=ET.tostring(relations[number],encoding='utf-8',xml_declaration=True)
        counts.append({'slide':number,'objects':len(objects),'order':[n['id'] for n in ordered]})
    ET.register_namespace('',CT)
    parts['[Content_Types].xml']=ET.tostring(content,encoding='utf-8',xml_declaration=True)
    with ZipFile(target,'x',ZIP_DEFLATED) as z:
        for name,data in parts.items():z.writestr(name,data)
    return {'mode':'artifact_tool_native_batches','slides':counts,'copied_images':copied_images,'authoring_engine_unchanged':True,'human_acceptance':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('target');a=p.parse_args();print(json.dumps(merge_batches(a.manifest,a.target)))
