"""Source-derived group calibration; installed faces and shared styles only."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import ImageFont
from .object_extract import rectangle
from .parallel_branches import canonical_fonts
from .parallel_store import key


def snap_ooxml_size_px(value: float) -> float:
    """Snap pixel font size to 0.2px steps so size_px * 0.75 * 100 is an exact integer."""
    return float(round(round(max(4.0, min(400.0, float(value))) * 5.0) * 0.2, 1))


def measure_text_appearance(source, bbox, blocked=None):
    """Measure foreground glyph RGB, stroke ink mask and local backing class from source pixels."""
    from reconstruction import background_model
    h_img, w_img = source.shape[:2]
    x, y, w, h = [round(v) for v in bbox]
    if min(w, h) <= 0 or x < 0 or y < 0 or x + w > w_img or y + h > h_img:
        return None
    if blocked is None:
        blocked = np.zeros((h_img, w_img), bool)
    backing = None
    error = 999.0
    try:
        backing, error = background_model(source, [x, y, w, h], blocked)
    except (ValueError, OSError):
        pass
    if backing is None or error > 8.0:
        rx0, ry0 = max(0, x - 4), max(0, y - 3)
        rx1, ry1 = min(w_img, x + w + 4), min(h_img, y + h + 3)
        ring = np.ones((ry1 - ry0, rx1 - rx0), bool)
        ring[y - ry0:y + h - ry0, x - rx0:x + w - rx0] = False
        ring &= ~blocked[ry0:ry1, rx0:rx1]
        ring_vals = source[ry0:ry1, rx0:rx1][ring].astype(float)
        if len(ring_vals) < 10:
            return None
        bg_med = np.median(ring_vals, axis=0)
        error = float(np.percentile(np.abs(ring_vals - bg_med).max(1), 90))
        backing = np.broadcast_to(bg_med, (h, w, 3)).copy()
    bg_class = 'FLAT' if error <= 3.5 else ('GRADIENT' if error <= 14.0 else 'COMPLEX')
    crop = source[y:y+h, x:x+w]
    delta = np.abs(crop.astype(float) - backing.astype(float)).max(2)
    threshold = max(34.0, min(65.0, error * 2.2))
    ink = (delta > threshold) & ~blocked[y:y+h, x:x+w]
    if int(ink.sum()) < 8:
        return None
    strong = ink & (delta >= np.percentile(delta[ink], 75))
    values = crop[strong].astype(float)
    quant = values.astype(int) // 12
    colors, counts = np.unique(quant, axis=0, return_counts=True)
    dominant = colors[counts.argmax()]
    seed = np.median(values[(quant == dominant).all(1)], axis=0)
    in_cluster = np.abs(values - seed).max(1) <= 18.0
    agreement = float(in_cluster.mean())
    if agreement < 0.50:
        return None
    fg_rgb = np.clip(np.median(values[in_cluster], axis=0).round(), 0, 255).astype(int)
    hex_color = ''.join(f'{v:02X}' for v in fg_rgb)
    return {'color_rgb': fg_rgb, 'color_hex': hex_color, 'agreement': agreement,
            'backing': backing, 'backing_error': float(error), 'backing_class': bg_class, 'ink': ink}


def calibrate(graph,source,fonts,store,ocr=None,stable=(),*,adaptive=False):
    from .contracts import Style
    graph=graph.model_copy(deep=True);graph.text=canonical_fonts(graph.text,fonts,store)
    binding=key({'source':graph.source_sha256,'fonts':[(r['family'],r['bold'],r.get('sha256')) for r in fonts['fonts']],
                 'stable_rgba':[(r['sha256'],r['bbox']) for r in stable],
                 'members':[(t.text,t.bbox,t.style,[s.model_dump() for s in t.spans],graph.ownership[t.id]) for t in graph.text.texts],
                 'method':'r12-group-calibration/5','adaptive':adaptive})
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
    raw_samples=defaultdict(list);styles={s.id:s for s in graph.text.styles}
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
        measured=measure_text_appearance(source,[x,y,w,h],blocked)
        if measured is None or measured['backing_error']>6.0:continue
        color=measured['color_rgb'].astype(float)
        model=np.array([int(s.color[i:i+2],16) for i in (0,2,4)],float)
        # Trust high-agreement pixel ink measurements even when the VLM guessed
        # a default color (e.g. black instead of blue); reject only low-agreement noise.
        if np.abs(color-model).max()>55 and measured['agreement']<0.60:continue
        ink=measured['ink'];backing=measured['backing'];crop=source[y:y+h,x:x+w]
        family=s.ea if any(ord(c)>255 for c in content) else s.latin
        f=installed.get((family,s.bold))
        if f is None:
            f=next((r for r in fonts['fonts'] if r.get('cjk') or not any(ord(c)>255 for c in content)),None)
        if f is None:continue
        try:
            font=ImageFont.truetype(f['file'],max(1,round(s.size_px)),index=f['index'])
            yy,xx=np.nonzero(ink)
            target=int(xx.max()-xx.min()+1)
            target_h=int(yy.max()-yy.min()+1)
            fbox=font.getbbox(content)
            rendered=max(1,fbox[2]-fbox[0])
            rendered_h=max(1,fbox[3]-fbox[1])
            ratio=min(target/rendered,(target_h/rendered_h)*1.08)
            if .5<=ratio<=1.8:
                raw_samples[s.id].append((color,ratio,t,{'id':t.id,'text':content,'source':crop.astype(float),
                    'background':backing.astype(float),'foreground':color,
                    'ink_left':int(xx.min()),'ink_top':int(yy.min()),'ink_width':target,'ink_height':target_h}))
        except (ValueError,OSError):continue
    # If texts sharing one style have genuinely distinct measured ink colors
    # (e.g. blue header vs black body grouped together), split into color-clean styles.
    samples=defaultdict(list);visual=defaultdict(list)
    for sid,entries in raw_samples.items():
        s=styles[sid]
        med_color=np.median(np.stack([e[0] for e in entries]),axis=0)
        if all(np.abs(e[0]-med_color).max()<=35 for e in entries):
            for color,ratio,t,vis in entries:
                samples[sid].append((color,ratio,t.id));visual[sid].append(vis)
            continue
        clusters=[]
        for entry in entries:
            placed=False
            for cluster in clusters:
                center=np.median(np.stack([c[0] for c in cluster]),axis=0)
                if np.abs(entry[0]-center).max()<=35:
                    cluster.append(entry);placed=True;break
            if not placed:clusters.append([entry])
        clusters.sort(key=lambda c:-len(c))
        for idx,cluster in enumerate(clusters):
            if idx==0:
                target_sid=sid
            else:
                c_rgb=np.median(np.stack([c[0] for c in cluster]),axis=0).round().astype(int)
                hex_c=''.join(f'{v:02X}' for v in c_rgb)
                target_sid=f'{sid[:36]}-c{hex_c.lower()}'
                if target_sid not in styles:
                    derived=Style.model_validate({**s.model_dump(),'id':target_sid,'color':hex_c})
                    graph.text.styles.append(derived);styles[target_sid]=derived
                for _,_,t,_ in cluster:
                    if t.id not in graph.selected:t.style=target_sid
            for color,ratio,t,vis in cluster:
                samples[target_sid].append((color,ratio,t.id));visual[target_sid].append(vis)
    changes=[]
    for sid,rows in samples.items():
        s=styles[sid];before=s.model_dump();color=np.median(np.stack([r[0] for r in rows]),axis=0).round().astype(int)
        s.color=''.join(f'{v:02X}' for v in color)
        # One role-level size adjustment, never independent old/new box styles.
        ratio=float(np.median([r[1] for r in rows]));s.size_px=snap_ooxml_size_px(s.size_px*ratio)
        from .parallel_font_match import match
        representatives=sorted(visual[sid],key=lambda r:-len(r['text']))[:2]
        choices={'fonts':[f for f in fonts['fonts'] if f['family'] in (s.ea,s.latin)]} if adaptive else fonts
        selection=match(representatives,choices,s)
        if selection is not None:
            if selection['cjk']:s.ea=selection['family']
            else:s.latin=selection['family']
            s.bold=selection['bold'];s.size_px=snap_ooxml_size_px(selection['size_px'])
            if (s.latin,s.bold) not in installed:s.latin=selection['family']
            if (s.ea,s.bold) not in installed:s.ea=selection['family']
        changes.append({'style':sid,'before':before,'after':s.model_dump(),'source_members':[r[2] for r in rows],'font_selection':selection})
    # Snap peer styles of the same role/weight/family within 1.2px jitter to a clean page tier.
    by_role=defaultdict(list)
    for row in changes:
        st=styles[row['style']]
        by_role[(st.role,st.bold,st.ea)].append((st,len(row['source_members']),row))
    for group in by_role.values():
        group.sort(key=lambda item:item[0].size_px)
        tier=[]
        def flush_tier(items):
            if len(items)<2:return
            weights=sum(w for _,w,_ in items)
            snapped=snap_ooxml_size_px(sum(st.size_px*w for st,w,_ in items)/max(1,weights))
            for st,_,rec in items:
                st.size_px=snapped;rec['after']=st.model_dump()
        for item in group:
            if not tier or abs(item[0].size_px-tier[0][0].size_px)<=1.2:tier.append(item)
            else:flush_tier(tier);tier=[item]
        flush_tier(tier)
    if changes:graph.references.append(store.evidence({'schema':'GroupTypographyCalibration/1','source_sha256':graph.source_sha256,'binding':binding,'changes':changes}))
    return graph
