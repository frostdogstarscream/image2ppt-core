"""Number text discovered after initial perception and verify it against the upload."""
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts import Style, Text, intersection
from ..experiment import mark_blocks
from ..materialize import digest, write
from ..parallel_branches import local_ocr, snapshot
from ..parallel_fusion import fuse
from ..parallel_store import key
from ..parallel_typography import measure_text_appearance, snap_ooxml_size_px
from .host_evidence import read
from .marked_routes import iou
from .state import RuntimeConflict


def source_reading(image, box, directory, cache, *, scale=2):
    x, y, w, h = box
    margin = max(1, round(h * (.45 if scale == 2 else .25)))
    left, top = max(0, x-margin), max(0, y-margin)
    crop = image.crop((left, top, min(image.width,x+w+margin), min(image.height,y+h+margin)))
    path = directory/f'crop-{scale}.png'
    crop.resize((crop.width*scale,crop.height*scale), Image.Resampling.LANCZOS).save(path)
    snap = snapshot(path,directory/'source-crops')
    result = local_ocr(snap,cache,os.environ['IMAGE2PPT_OCR_MANIFEST'])
    if result.get('source_sha256') != snap['source_sha256']:
        raise RuntimeConflict('late source crop OCR binding changed')
    hits = []
    for line in result['lines']:
        lx, ly, lw, lh = line['bbox']
        candidate = [left+round(lx/scale), top+round(ly/scale), round(lw/scale), round(lh/scale)]
        candidate[2]=min(candidate[2],image.width-candidate[0])
        candidate[3]=min(candidate[3],image.height-candidate[1])
        if line.get('text','').strip() and iou(box,candidate) >= .45:
            hits.append({'text':line['text'].strip(),'bbox':candidate,'score':line['score']})
    return {'crop_sha256':snap['source_sha256'],'crop_file':str(path),'ocr_identity':result.get('identity'),
            'readings':hits}


def verified_parent(graph, field_id, plans, backing):
    """Only an independently committed, enclosing face may gain a movement partner."""
    plans=[p for p in plans if p.get('origin')=='host' and p.get('route')=='surface'
           and p.get('role') in ('card','role_box')]
    hosts=[]
    for relation in graph.relations:
        if (relation.target!=field_id or relation.kind!='contains' or relation.status!='RESOLVED'
                or graph.ownership.get(relation.source)!='RESOLVED'):
            continue
        face=graph.selected.get(relation.source,{})
        owner=face.get('owner_evidence') or {}
        if face.get('kind')!='surface' or not owner.get('file'):
            continue
        file=(backing/owner['file']).resolve()
        if not file.is_relative_to(backing.resolve()) or not file.is_file() or digest(file)!=owner.get('sha256'):
            continue
        aliases={face.get('mark_id'),relation.source,*face.get('seed_ids',[])}
        matched=[p for p in plans if p['id'] in aliases or iou(p['bbox'],face['bbox'])>=.9]
        if len(matched)==1:
            hosts.append((face['bbox'],matched[0]))
    if not hosts:
        return None
    box,plan=min(hosts,key=lambda pair:pair[0][2]*pair[0][3])
    if all(other_plan is plan or (other[2]*other[3]>1.2*box[2]*box[3] and
           intersection(box,other)>=.98*box[2]*box[3]) for other,other_plan in hosts):
        return plan
    return None


def register(service, state, current, directory, *, page, findings):
    """Bind observations to stable marks. Only original-source readings may create draft text."""
    k = str(page)
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    row = service.page(current,page)
    graph = service.graph(row)
    snap = row['snapshot']
    source = Path(snap['path'])
    sha = digest(source)
    inventory = read(current['experiment_marks'][k])
    ocr = read(current['ocr'][k])
    if sha != graph.source_sha256 or sha != inventory['source_sha256'] or sha != ocr.get('source_sha256'):
        raise RuntimeConflict('late source field binding changed')
    with Image.open(source) as im:
        image = im.convert('RGB')
    pixels = np.asarray(image)
    rows = inventory['rows']
    stored = current.setdefault('late_fields',{}).setdefault(k,{})
    next_mark = max((m['mark'] for m in rows),default=0) + 1
    new_marks, changed_marks, accepted, results = [], [], [], []
    plans = current.get('marked_plans',{}).get(k,[])
    artwork = [p['bbox'] for p in plans if p.get('role') in ('brand','screenshot','dashboard') or p.get('route')=='preserve']
    artwork += [a.bbox for a in graph.asset.objects if a.kind=='logo' or a.text_policy=='intrinsic_label']
    for finding in findings:
        raw = finding.get('bbox') or []
        if len(raw) != 4 or not all(isinstance(n,(int,float)) and np.isfinite(n) for n in raw):
            results.append({'id':None,'bbox':raw,'status':'skipped','reason':'invalid_source_box'})
            continue
        x,y,w,h = box = list(map(round,raw))
        if min(w,h)<1 or min(x,y)<0 or x+w>image.width or y+h>image.height:
            results.append({'id':None,'bbox':box,'status':'skipped','reason':'outside_source'})
            continue
        oid = 't-late-'+key([sha,page,box])[:16]
        previous = next((r for r in stored.values() if iou(r['bbox'],box)>=.8),None)
        if previous and not (previous['status']=='pending' and (
                previous['reason'].startswith('source_reading_unavailable_') or
                finding.get('text') and finding['text']!=previous.get('observed_text'))):
            results.append(previous)
            continue
        if previous:
            oid=previous['id']
            x,y,w,h=box=previous['bbox']
        item = {'id':oid,'bbox':box,'status':'pending','reason':'unconfirmed_source','source_sha256':sha}
        results.append(item)
        if any(intersection(box,b)>=.8*w*h for b in artwork) or any(
                m['label']=='icon' and iou(m['bbox'],box)>=.55 for m in rows):
            item.update(status='skipped',reason='intrinsic_graphic')
            continue
        known = [line for line in ocr['lines'] if intersection(line['bbox'],box)>.35*min(
                 w*h, line['bbox'][2]*line['bbox'][3])]
        direct = known[0] if len(known)==1 and iou(box,known[0]['bbox'])>=.65 else None
        if known and direct is None:
            item.update(status='skipped',reason='overlaps_existing_fields')
            continue
        if direct:
            owners = [t.id for t in graph.text.texts if direct['id'] in graph.text.members.get(t.id,[])]
            if owners:
                item.update(id=owners[0] if len(owners)==1 else None, status='existing' if len(owners)==1 else 'skipped',
                            reason='existing_source_field' if len(owners)==1 else 'overlaps_split_fields')
                continue
        if any(intersection(t.bbox,box)>.5*min(w*h,t.bbox[2]*t.bbox[3]) for t in graph.text.texts):
            item.update(status='skipped',reason='overlaps_existing_text')
            continue
        stored[oid] = item
        line_id = direct['id'] if direct else 'ocr-'+oid
        parents=[m for m in rows if m['label'] in ('card_frame','container','surface') and
                 intersection(m['bbox'],box)>=.95*w*h and m['bbox'][2]*m['bbox'][3]>w*h]
        parent=min(parents,key=lambda m:m['bbox'][2]*m['bbox'][3],default=None)
        mark = {'mark':next_mark,'id':oid,'label':'late_text_observation','bbox':box,'text':'',
                'parent':parent['id'] if parent else None,
                'sources':[],'ocr_ids':[]}
        if previous:
            mark=next(m for m in rows if m['id']==oid)
            changed_marks.append(mark)
        else:
            new_marks.append(mark)
            next_mark += 1
        item['mark'] = mark['mark']
        work = directory/oid
        work.mkdir(exist_ok=True)
        try:
            first = source_reading(image,box,work,state.job/'cache')
            item['source_views'] = [first]
            if len(first['readings'])!=1:
                item['reason'] = 'no_unique_source_reading'
                continue
            hit = first['readings'][0]
            witness = (finding.get('text') or '').strip()
            if not witness:
                second = source_reading(image,box,work,state.job/'cache',scale=3)
                item['source_views'].append(second)
                if len(second['readings'])==1 and second['readings'][0]['score']>=.85:
                    witness = second['readings'][0]['text']
            item.update(source_reading=hit['text'],observed_text=witness)
            if (hit['score']<.85 or ''.join(witness.split())!=''.join(hit['text'].split())
                    or not any(c.isalnum() for c in hit['text'])):
                item['reason']='source_readings_unconfirmed'
                continue
            if direct and direct['text'] != hit['text']:
                item['reason']='existing_ocr_disagrees'
                continue
            text_box = direct['bbox'] if direct else hit['bbox']
            look = measure_text_appearance(pixels,text_box)
            if not look:
                item['reason']='no_separable_source_ink'
                continue
        except RuntimeConflict:
            raise
        except (OSError,RuntimeError,KeyError,ValueError,ImportError) as exc:
            item['reason']='source_reading_unavailable_'+type(exc).__name__
            continue
        line = dict(direct or {}, id=line_id, bbox=text_box, text=hit['text'],score=hit['score'])
        nearest = min(graph.text.styles,key=lambda s:abs(s.size_px-text_box[3]),default=None)
        style = Style(id='s-'+oid,role='source text',ea=nearest.ea if nearest else 'Microsoft YaHei',
                      latin=nearest.latin if nearest else 'Arial',color=look['color_hex'],
                      bold=nearest.bold if nearest else False,size_px=snap_ooxml_size_px(text_box[3]*.88))
        similar = next((s for s in graph.text.styles if (s.ea,s.latin,s.bold,s.color,s.size_px)==
                        (style.ea,style.latin,style.bold,style.color,style.size_px)),None)
        if similar:
            style=similar
        if len(graph.text.texts)>=400 or (not similar and len(graph.text.styles)>=100):
            item['reason']='source_evidence_capacity'
            continue
        text = Text(id=oid,text=line['text'],bbox=text_box,style=style.id,z=900)
        working = [ln for ln in ocr['lines'] if ln['id']!=line_id]+[line]
        proposed = graph.text.model_copy(deep=True)
        if not similar:
            proposed.styles.append(style)
        proposed.texts.append(text)
        proposed.members[oid]=[line_id]
        fused = fuse(snap,{'text':proposed,'asset':graph.asset,'surface':graph.surface},{'lines':working})
        if fused.ownership[oid]!='RESOLVED':
            item['reason']='source_ownership_unresolved'
            continue
        graph.text=proposed
        graph.ownership[oid]='RESOLVED'
        graph.relations.extend(r for r in fused.relations if oid in (r.source,r.target))
        ocr['lines']=working
        mark.update(label='text',text=line['text'],bbox=text_box,ocr_ids=[line_id])
        group_parent=verified_parent(graph,oid,plans,Path(row['ir']).parent)
        if group_parent:
            mark['parent']=group_parent['id']
            partners=group_parent.setdefault('move_with',[])
            if oid not in partners:
                partners.append(oid)
        accepted.append(oid)
        item.update(status='ready',reason='source_crop_confirmed',line=line)
    if new_marks or changed_marks:
        marks_file=directory/'late-marks.json'
        write(marks_file,{**inventory,'rows':rows+new_marks})
        current['experiment_marks'][k]=str(marks_file)
        evidence=current['host_evidence'][k]
        evidence['inventory_sha256']=digest(marks_file)
        for block,pic,info in mark_blocks(source,new_marks,directory):
            evidence['blocks'].append({**block,'image':state.register(pic,'numbered_block_image'),
                                      'inventory':state.register(info,'numbered_block_inventory')})
    if accepted:
        # Copy the exact committed IR, masks and artwork. Only draft source evidence is extended.
        backing = directory/'backing'
        shutil.copytree(Path(row['ir']).parent,backing)
        ir=backing/Path(row['ir']).name
        write(backing/'scene.json',graph.model_dump())
        write(backing/'runtime-ocr-protection.json',{'source_sha256':sha,'lines':ocr['lines']})
        row.update(ir=str(ir),scene=str(backing/'scene.json'))
        ocr['line_count']=len(ocr['lines'])
        write(directory/'verified-source-ocr.json',ocr)
        current['ocr'][k]=str(directory/'verified-source-ocr.json')
        for m in [*new_marks,*changed_marks]:
            if m['id'] in accepted:
                ledger=current.setdefault('scene_relations',{}).setdefault(k,{}).setdefault('source_fields',{})
                ledger.setdefault(m['ocr_ids'][0],{'text':m['text'],'bbox':m['bbox'],'render_text':m['text'],
                                                  'render_bbox':m['bbox'],'separate_symbol':None})
                current['marked_plans'][k].append({'id':m['id'],'bbox':m['bbox'],'route':'text','role':'field',
                                                   'source_text':m['text'],'origin':'host','move_with':[]})
    write(directory/'source-findings.json',results)
    return results
