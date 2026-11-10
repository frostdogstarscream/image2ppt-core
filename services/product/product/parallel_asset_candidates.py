"""Bounded asset diagnostics. Legacy candidate bytes and acceptance are unchanged."""
from pathlib import Path
import numpy as np
from PIL import Image
from .materialize import write,digest
from .object_extract import rectangle
from .parallel_rebuild import asset_candidates,source_owner,OwnerConflict,array_sha


def diagnose(graph,source,layers,stable,sam,text_protection,directory,object_ids):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    original={a['path']:digest(a['path']) for a in stable};results=[]
    surface=graph.surface.model_copy(deep=True)
    surface.shapes=[s for s in surface.shapes if graph.ownership[s.id]=='RESOLVED']
    for obj in graph.asset.objects:
        if obj.id not in object_ids:continue
        reserved=text_protection.copy();peers={}
        for record in stable:
            if record['id']==obj.id:continue
            x,y,w,h=map(round,record['bbox']);rgba=np.array(Image.open(record['path']).convert('RGBA'))
            mask=np.zeros(source.shape[:2],bool);mask[y:y+h,x:x+w]=rgba[...,3]>0
            reserved|=mask;peers[record['id']]=mask
        for text in graph.text.texts:
            peers[text.id]=rectangle(text.bbox,tuple(graph.image_size))&text_protection
        proposals=[]
        for assets in (stable,()):
            try:proposals.extend(asset_candidates(obj,source,layers,reserved,assets,'auto',sam.get(obj.id),None,surface))
            except (ValueError,OSError) as exc:results.append({'object_id':obj.id,'status':'UNKNOWN','reason':str(exc)})
        seen=set()
        for rgba,meta in proposals[:6]:
            sha=array_sha(rgba)
            if sha in seen:continue
            seen.add(sha);folder=directory/(obj.id+'-'+sha);folder.mkdir()
            Image.fromarray(rgba).save(folder/'candidate.png')
            row={'object_id':obj.id,'method':meta['method'],'source_sha256':graph.source_sha256,
                 'candidate_rgba_sha256':sha,'candidate_png_sha256':digest(folder/'candidate.png'),
                 'automatic_acceptance':False}
            try:
                source_owner(obj,rgba,source,reserved,surface)
                row.update(status='OWNER_PRECHECK_ONLY',reason='actual editing and final visual review still required')
            except OwnerConflict as exc:
                x,y,w,h=exc.box
                Image.fromarray(exc.component.astype('uint8')*255).save(folder/'component-mask.png')
                Image.fromarray(exc.conflict.astype('uint8')*255).save(folder/'protected-conflict-mask.png')
                overlay=source[y:y+h,x:x+w].copy();overlay[exc.conflict]=[255,0,0]
                Image.fromarray(overlay).save(folder/'protected-conflict.png')
                row.update(exc.report)
                row['protected_objects']=[{'id':name,'overlap_pixels':int((mask[y:y+h,x:x+w]&exc.conflict).sum()),
                    'basis':'stable alpha' if name in {a['id'] for a in stable} else 'text localization intersected with protected ink; semantic attribution unproven'}
                    for name,mask in peers.items() if (mask[y:y+h,x:x+w]&exc.conflict).any()]
                row['evidence_files']={p.name:digest(p) for p in folder.iterdir() if p.is_file()}
            except ValueError as exc:row.update(status='UNKNOWN',reason=str(exc))
            write(folder/'owner-conflict.json',row);results.append(row)
    if any(digest(file)!=sha for file,sha in original.items()):raise ValueError('stable asset modified during diagnostics')
    write(directory/'asset-candidates.json',{'candidates':results,'stable_bytes_preserved':True,'published':False})
    return results
