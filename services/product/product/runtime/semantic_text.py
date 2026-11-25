"""Page-local type and line-flow observations for editable reconstruction."""
from collections import defaultdict
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..contracts import intersection
from ..parallel_contracts import continuous_chinese_lines
from ..parallel_typography import measure_text_appearance


def is_body_continuation(item, lines):
    """Distinguish OCR line breaks inside Chinese prose from authored label/bullet breaks."""
    if item.get('align') != 'left' or not continuous_chinese_lines([line['text'] for line in lines]):
        return False
    for prev, nxt in zip(lines, lines[1:]):
        x, _, width, height = prev['bbox']
        if abs(x - nxt['bbox'][0]) > height * .45 or nxt['bbox'][2] > width * 1.12:
            return False
    return True


def plain_line_gap(source, bbox):
    """A tight uniform stripe can separate OCR lines even inside a coarse asset box.

    This observation only authorizes grouping OCR members. Pixel ownership and
    background repair still run on the source for the resulting object.
    """
    x, y, w, h = map(round, bbox)
    if h < 3 or w < 8 or min(x, y) < 0:
        return False
    stripe = source[y:y+h, x:x+w].astype(float)
    if stripe.shape[:2] != (h,w) or np.ptp(stripe.reshape(-1,3), axis=0).max() > 24:
        return False
    xx,yy = np.meshgrid(np.linspace(-1,1,w),np.linspace(-1,1,h))
    design = np.stack([xx,yy,np.ones_like(xx)], axis=-1).reshape(-1,3)
    observed = stripe.reshape(-1,3)
    predicted = design @ np.linalg.lstsq(design,observed,rcond=None)[0]
    return np.abs(observed-predicted).max() <= 12


def _ink_density(alpha):
    ys, xs = np.nonzero(alpha >= .50)
    if not len(xs):
        return None
    return float(alpha[ys.min():ys.max()+1, xs.min():xs.max()+1].mean())


def apply_heading_weights(graph, source, fonts, inventory):
    """Compare actual heading ink with both installed faces across aligned page peers."""
    styles = {s.id: s for s in graph.text.styles}
    faces = {(f['family'], f['bold']): f for f in fonts['fonts']}
    groups = []
    for text in graph.text.texts:
        if (graph.ownership.get(text.id) != 'RESOLVED' or text.spans or text.id in graph.selected
                or '\n' in text.text or not re.search(r'[\u3400-\u9fff]', text.text)):
            continue
        x, y, w, h = text.bbox
        if min(w,h) <= 0:
            continue
        # Match the actual title/heading reading to a structure hint. A city,
        # numeric label, or short row in a table is not a heading by size alone.
        role = next((e['category'] for e in inventory.get('elements', [])
                     if e.get('category') in ('title','section_header')
                     and intersection(text.bbox, e['bbox']) >= .86*w*h), None)
        if role is None:
            continue
        below = any(t.id != text.id and len(t.text.replace('\n','')) >= 8
                    and abs(t.bbox[0]-x) <= h*.65 and -.2*h <= t.bbox[1]-(y+h) <= h
                    and t.bbox[2] >= w*.8 for t in graph.text.texts)
        if role == 'section_header' and not below:
            continue
        style = styles[text.style]
        color = np.array(list(bytes.fromhex(style.color)),dtype=int)
        group = next((g for g in groups if role == g['role'] and g['family'] == style.ea
                      and abs(y - g['y']) <= .27*h and abs(h-g['height']) <= .20*h
                      and np.abs(g['color']-color).max() <= 24),None)
        if group is None:
            group = dict(role=role, y=y, height=h, color=color, family=style.ea, texts=[])
            groups.append(group)
        group['texts'].append(text)

    reports = []
    for group in groups:
        members = group['texts']
        if len(members) < 2:
            continue
        samples = []
        for t in members:
            s = styles[t.style]
            if any((s.ea,b) not in faces for b in (False,True)):
                continue
            a = measure_text_appearance(source,t.bbox)
            if a is None or a['backing_error'] > 6 or a['agreement'] < .72:
                continue
            x,y,w,h = map(round,t.bbox)
            bg = a['backing'].astype(float)
            vector = bg - a['color_rgb']
            energy = (vector*vector).sum(2)
            if float(np.median(energy)) < 40**2:
                continue
            alpha = np.clip(((bg-source[y:y+h,x:x+w])*vector).sum(2) / np.maximum(1,energy),0,1)
            density = _ink_density(alpha)
            alternatives = []
            for bold in (False,True):
                face = faces[s.ea,bold]
                try:
                    font = ImageFont.truetype(face['file'],max(1,round(s.size_px)),index=face['index'])
                    l,top,r,bot = font.getbbox(t.text)
                    if r <= l or bot <= top:
                        break
                    glyph = Image.new('L',(r-l,bot-top))
                    ImageDraw.Draw(glyph).text((-l,-top),t.text,font=font,fill=255)
                    alternatives.append(_ink_density(np.asarray(glyph)/255.))
                except (ValueError,OSError):
                    break
            if density is not None and len(alternatives)==2 and None not in alternatives and alternatives[1]-alternatives[0] >= .06:
                samples.append({'id':t.id, 'source_density':density,'regular_density':alternatives[0],
                                'bold_density':alternatives[1],
                                'bold_advantage':abs(density-alternatives[0])-abs(density-alternatives[1])})
        votes = [sample['bold_advantage'] for sample in samples]
        if len(votes) < 2 or max(votes)*min(votes) <= 0 or abs(float(np.median(votes))) < .04:
            continue
        weight = min(votes) > 0
        identifiers = {t.id for t in members}
        by_style = defaultdict(list)
        for t in members:
            by_style[t.style].append(t)
        for sid, subset in by_style.items():
            style = styles[sid]
            if style.bold == weight or (style.latin,weight) not in faces:
                continue
            if any((t.style==sid and t.id not in identifiers) or any(sp.style==sid for sp in t.spans)
                   for t in graph.text.texts):
                if len(graph.text.styles) >= 100:
                    continue
                style = style.model_copy(update={'id': f'{sid[:30]}-heading-{subset[0].id[:20]}'})
                styles[style.id] = style
                graph.text.styles.append(style)
                for t in subset:
                    t.style = style.id
            style.bold = weight
        reports.append({'role':group['role'],'members':[t.id for t in members], 'measured':samples,
                        'preferred_bold':weight, 'applied_bold':{t.id:styles[t.style].bold for t in members}})
    return reports
