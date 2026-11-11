"""Source-reviewed page styles. Groups, not individual text boxes, are atomic.

This module does not infer roles or call a model. The conversation authors the
group map against the source. It is deliberately separate from public SlideIR.
"""
from __future__ import annotations
import copy
import math


def paragraphs(ir):
    return {f"{n['id']}/p{i}": (n, p) for n in ir['nodes'] if n['type']=='text'
            for i,p in enumerate(n.get('paragraphs') or [n])}


def immutable(ir):
    value=copy.deepcopy(ir)
    for n in value['nodes']:
        if n['type']!='text': continue
        for p in [n,*n.get('paragraphs',[])]:
            for obj in [p,*p.get('runs',[])]:
                for field in ('font','bold','line_spacing_pt'): obj.pop(field,None)
    return value


def validate_groups(ir, groups):
    available=paragraphs(ir); seen=set(); ids=set()
    for g in groups:
        if not g['id'] or g['id'] in ids: raise ValueError('Duplicate style group')
        ids.add(g['id'])
        if not g['members'] or not g['candidates']: raise ValueError('Empty group')
        for key in g['members']:
            if key not in available or key in seen: raise ValueError('Missing/duplicate member: '+key)
            seen.add(key)
        for key,style in g.get('exceptions',{}).items():
            parent,sep,run=key.rpartition('/r')
            if not sep or parent not in g['members'] or not run.isdigit(): raise ValueError('Unbound emphasis')
            if int(run)>=len(available[parent][1].get('runs',[])): raise ValueError('Unknown emphasis run')
            validate_style(style, partial=True)
        for style in g['candidates']: validate_style(style)
    if seen!=set(available): raise ValueError('Every paragraph must belong to one group')


def validate_style(style, partial=False):
    if set(style)-{'ea','latin','bold','size_pt','line_spacing_pt'}: raise ValueError('Unexpected style field')
    if partial and 'line_spacing_pt' in style: raise ValueError('Runs cannot override paragraph spacing')
    if not partial and not {'ea','latin','bold','size_pt'}<=set(style): raise ValueError('Incomplete style')
    for key in ('size_pt','line_spacing_pt'):
        if key in style and (type(style[key]) not in (int,float) or not math.isfinite(style[key]) or style[key]<=0):
            raise ValueError('Invalid typography size')
    if 'bold' in style and type(style['bold']) is not bool: raise ValueError('Invalid weight')
    for key in ('ea','latin'):
        if key in style and (not isinstance(style[key],str) or not style[key]): raise ValueError('Invalid face')


def font_available(manifest, family, bold, text):
    faces=[f for f in manifest['fonts'] if f['family']==family and f['bold']==bold]
    return any(not (set(text)-set('\n\r\t')) & set(f['missing_characters']) for f in faces)


def apply_group(ir, group, style, manifest):
    validate_style(style)
    result=copy.deepcopy(ir); mapping=paragraphs(result)
    def apply(obj, effective, text):
        for script in ('ea','latin'):
            chars=''.join(c for c in text if (ord(c)<128)==(script=='latin'))
            if chars and not font_available(manifest,effective[script],effective['bold'],chars):
                raise ValueError('Unavailable face/weight or missing glyph: '+effective[script])
        obj['font']={k:effective[k] for k in ('ea','latin','size_pt')}
        obj['bold']=effective['bold']
        if 'line_spacing_pt' in effective: obj['line_spacing_pt']=effective['line_spacing_pt']
    for key in group['members']:
        node,p=mapping[key]; apply(p,style,p['text'])
        for i,r in enumerate(p.get('runs',[])):
            effective={**style,**group.get('exceptions',{}).get(key+f'/r{i}',{})}
            # Explicitly replace stale overrides while keeping content/color.
            apply(r,effective,r['text'])
        first=(node.get('paragraphs') or [node])[0]
        node['font']=copy.deepcopy(first['font']); node['bold']=first['bold']
    if immutable(ir)!=immutable(result): raise ValueError('Typography altered protected content')
    return result


def verify_groups(ir, groups, accepted):
    mapping=paragraphs(ir)
    for g in groups:
        if g['id'] not in accepted: continue
        style=accepted[g['id']]
        for key in g['members']:
            _,p=mapping[key]
            for i,obj in [(-1,p),*enumerate(p.get('runs',[]))]:
                effective={**style,**g.get('exceptions',{}).get(key+f'/r{i}',{})}
                if obj['font']!={k:effective[k] for k in ('ea','latin','size_pt')} or obj['bold']!=effective['bold']:
                    raise ValueError('Group style was partially applied: '+key)


def choose_groups(ir, groups, manifest, check):
    """Check the complete accumulated scene. Failed candidates leave it intact.

    check(before, candidate, group) must explicitly return True after its hard
    gates. False is a rejection; exceptions propagate (never accept missing QA).
    Render-based callers may cache a full-page render, but not a member decision.
    """
    validate_groups(ir,groups)
    selected=copy.deepcopy(ir); accepted={}; rejected=[]
    for group in groups:
        for style in group['candidates']:
            try: candidate=apply_group(selected,group,style,manifest)
            except ValueError: continue
            if candidate==selected or check(selected,candidate,group) is True:
                selected=candidate; accepted[group['id']]=copy.deepcopy(style); break
        else: rejected.append(group['id'])
    verify_groups(selected,groups,accepted)
    return selected,accepted,rejected
