"""Source-derived group calibration; installed faces and shared styles only."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import ImageFont
from .object_extract import rectangle
from .parallel_branches import canonical_fonts
from .parallel_store import key


def calibrate(graph,source,fonts,store,ocr=None,stable=(),*,adaptive=False):
    from reconstruction import background_model
    graph=graph.model_copy(deep=True);graph.text=canonical_fonts(graph.text,fonts,store)
    binding=key({'source':graph.source_sha256,'fonts':[(r['family'],r['bold'],r.get('sha256')) for r in fonts['fonts']],
                 'stable_rgba':[(r['sha256'],r['bbox']) for r in stable],
                 'members':[(t.text,t.bbox,t.style,[s.model_dump() for s in t.spans],graph.ownership[t.id]) for t in graph.text.texts],
                 'method':'r10-group-calibration/4','adaptive':adaptive})
    for reference in graph.references:
        file=Path(reference)
        if not file.is_file():continue
        record=json.loads(file.read_text(encoding='utf-8'))
        if record.get('schema')=='GroupTypographyCalibration/1' and record.get('binding')==binding:
            byid={s.id:s.model_dump() for s in graph.text.styles}
            if all(byid.get(r['style'])==r['after'] for r in record['changes']):return graph
    installed={(f['family'],f['bold']):f for f in fonts['fonts']}
    size=tuple(graph.image_size)
    from .parallel_pixels import graphic_ink
    blocked=graphic_ink(stable,size,graph.source_sha256)
    samples=defaultdict(list);visual=defaultdict(list);styles={s.id:s for s in graph.text.styles}
    ocr_lines={r['id']:r for r in (ocr or {}).get('lines',[])}
    for t in graph.text.texts:
        if graph.ownership[t.id]!='RESOLVED' or t.spans:continue
        content=t.text.split('\n')[0];box=t.bbox
        ids=graph.text.members.get(t.id,[])
        if '\n' in t.text:
            if not ids or ids[0] not in ocr_lines:continue
            first=ocr_lines[ids[0]]
            if content.replace(' ','')!=first['text'].replace(' ',''):continue
            box=first['bbox']
        x,y,w,h=[round(v) for v in box]
        if min(w,h)<=0 or x<0 or y<0 or x+w>source.shape[1] or y+h>source.shape[0]:continue
        s=styles[t.style]
        try:
            backing,error=background_model(source,[x,y,w,h],blocked)
            if error>4:continue
            crop=source[y:y+h,x:x+w];delta=np.abs(crop.astype(float)-backing).max(2)
            ink=(delta>40)&~blocked[y:y+h,x:x+w]
            if ink.sum()<8:continue
            # Foreground comes from actual high-contrast interiors, also on dark
            # backgrounds. Never use 'darkest pixels' for white lettering.
            strong=ink&(delta>=np.percentile(delta[ink],80));values=crop[strong]
            quant=values//8;colors,counts=np.unique(quant,axis=0,return_counts=True);dominant=colors[counts.argmax()]
            selected=(quant==dominant).all(1);color=np.median(values[selected],axis=0).round()
            model=np.array([int(s.color[i:i+2],16) for i in (0,2,4)])
            if np.abs(color-model).max()>55:continue
            family=s.ea if any(ord(c)>255 for c in content) else s.latin
            f=installed.get((family,s.bold))
            if f is None:
                f=next((r for r in fonts['fonts'] if r.get('cjk') or not any(ord(c)>255 for c in content)),None)
            if f is None:continue
            font=ImageFont.truetype(f['file'],max(1,round(s.size_px)),index=f['index'])
            xx=np.nonzero(ink)[1];target=int(xx.max()-xx.min()+1)
            box=font.getbbox(content);rendered=max(1,box[2]-box[0]);ratio=target/rendered
            if .5<=ratio<=1.8:
                samples[s.id].append((color,ratio,t.id))
                yy,xx=np.nonzero(ink)
                visual[s.id].append({'id':t.id,'text':content,'source':crop.astype(float),'background':backing.astype(float),
                    'foreground':color,'ink_left':int(xx.min()),'ink_top':int(yy.min()),'ink_width':target})
        except (ValueError,OSError):continue
    changes=[]
    for sid,rows in samples.items():
        s=styles[sid];before=s.model_dump();color=np.median(np.stack([r[0] for r in rows]),axis=0).round().astype(int)
        s.color=''.join(f'{v:02X}' for v in color)
        # One role-level size adjustment, never independent old/new box styles.
        ratio=float(np.median([r[1] for r in rows]));s.size_px=round(s.size_px*ratio,2)
        from .parallel_font_match import match
        representatives=sorted(visual[sid],key=lambda r:-len(r['text']))[:2]
        choices={'fonts':[f for f in fonts['fonts'] if f['family'] in (s.ea,s.latin)]} if adaptive else fonts
        selection=match(representatives,choices,s)
        if selection is not None:
            if selection['cjk']:s.ea=selection['family']
            else:s.latin=selection['family']
            s.bold=selection['bold'];s.size_px=selection['size_px']
            if (s.latin,s.bold) not in installed:s.latin=selection['family']
            if (s.ea,s.bold) not in installed:s.ea=selection['family']
        changes.append({'style':sid,'before':before,'after':s.model_dump(),'source_members':[r[2] for r in rows],'font_selection':selection})
    if changes:graph.references.append(store.evidence({'schema':'GroupTypographyCalibration/1','source_sha256':graph.source_sha256,'binding':binding,'changes':changes}))
    return graph
