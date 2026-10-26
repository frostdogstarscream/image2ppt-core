"""Actual-file evidence, separated visual/deletion reviews, and scoped repairs."""
from copy import deepcopy
from pathlib import Path
import math
import re

from PIL import Image
import numpy as np

from image2ppt.quality import inspect_pptx
from image2ppt.storage import digest, read_json
from .auto_provider import fingerprint, save_new
from .auto_protocol import transform_model_boxes, inside
from .auto_region_protocol import require
from .supervised_elements import _intersection
from .typography import _candidates, _font


def crop_box(box, size):
    x,y,w,h=box
    return (max(0,math.floor(x)),max(0,math.floor(y)),min(size[0],math.ceil(x+w)),min(size[1],math.ceil(y+h)))


def crop_source(source, box, output):
    output=Path(output)
    with Image.open(source) as image:
        crop=image.convert('RGB').crop(crop_box(box,image.size))
        if output.exists():
            with Image.open(output) as old:
                require(old.size==crop.size and old.convert('RGB').tobytes()==crop.tobytes(), 'saved source crop changed')
        else:
            output.parent.mkdir(parents=True,exist_ok=True); crop.save(output)
    return output


def region_packets(build_dir, layout):
    build_dir=Path(build_dir); root=build_dir/'region-evidence'; root.mkdir(exist_ok=True)
    source=Image.open(build_dir/'evidence/source.png').convert('RGB')
    preview=Image.open(build_dir/'evidence/preview.png').convert('RGB').resize(source.size,Image.Resampling.LANCZOS)
    deleted=Image.open(build_dir/'evidence/delete-probe.png').convert('RGB').resize(source.size,Image.Resampling.LANCZOS)
    result={}
    for region in layout['regions']:
        path=root/region['id']; path.mkdir(exist_ok=True); box=crop_box(region['bbox'],source.size)
        for name,image in [('source.png',source),('preview.png',preview),('delete-probe.png',deleted)]:
            target=path/name
            if not target.exists(): image.crop(box).save(target)
        result[region['id']]={'directory':path,'files':{name:digest(path/name) for name in ['source.png','preview.png','delete-probe.png']}}
    return result


def verify_deletion(before_file, after_file, targets):
    before=inspect_pptx(before_file)['slides'][0]['objects']
    after=inspect_pptx(after_file)['slides'][0]['objects']
    original={o['name']:o for o in before}; edited={o['name']:o for o in after}
    expected=[o['name'] for o in before if o['name'] not in targets]
    passed=(set(targets)<=set(original) and not set(targets)&set(edited)
            and expected==[o['name'] for o in after])
    for name in expected:
        if name not in edited or {k:v for k,v in original[name].items() if k!='z'} != {k:v for k,v in edited[name].items() if k!='z'}:
            passed=False
    return {'schema':'ActualDeletionCheck/1','passed':bool(passed),'targets':targets,
            'before_sha256':digest(before_file),'after_sha256':digest(after_file),
            'other_objects_and_relative_order_unchanged':bool(passed)}


def issue(oid, kind, box, description, *, blocking=True):
    return {'id':'check-'+fingerprint([oid,kind,description])[:12],'kind':kind,
            'severity':'blocking' if blocking else 'minor','object_ids':[oid] if oid else [],
            'bbox':box,'description':description,'evidence_origin':'deterministic_check'}


def _luminance(color):
    rgb=[int(color[i:i+2],16)/255 for i in (0,2,4)]
    rgb=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb]
    return sum(a*b for a,b in zip(rgb,(.2126,.7152,.0722)))


def typography_checks(plan, region, elements, actual_objects):
    """Screen exported positions and font estimates; visual review must confirm appearance."""
    issues=[]; size=plan['image_size']; byname={o['name']:o for o in actual_objects['slides'][0]['objects']}
    slide_size=actual_objects.get('slide_size_emu') or actual_objects.get('size_emu')
    fonts=_candidates(None)
    audits={a['id']:a for a in elements['text_audit']}
    shapes=sorted(plan['objects'],key=lambda o:o['z'])
    native=[t for t in elements['text_decisions'] if t['action']=='native']
    for text in native:
        tid=text['id']; box=text['bbox']; actual=byname.get(tid)
        if actual is None or actual.get('text')!=text['text']:
            issues.append(issue(tid,'text',box,'Native text is missing or differs in the actual PPTX.'))
        if audits[tid]['status']=='unresolved':
            issues.append(issue(tid,'style',box,'Typography remains explicitly unresolved; calibrate from the source.'))
        # Measure the exported-plan typeface, not the OCR box height. This is an
        # estimate with a margin, not a claim of exact rendered glyph bounds.
        font=next((f for f in fonts if f['family']==text['font_family'] and f['bold']==text['bold']),None)
        if font:
            face=_font(font['path'],max(1,round(text['font_size_px'])),font.get('index',0))
            width=max((face.getlength(line) for line in text['text'].splitlines()),default=0)
            if width>box[2]*1.14+4 or box[0]+width>min(size[0],region['bbox'][0]+region['bbox'][2])+4:
                issues.append(issue(tid,'layout',box,'Font-metric width exceeds its exported box or region; confirm clipping/overlap in the actual preview.'))
        background=plan['background_color']; known=True
        for shape in shapes:
            b=shape.get('bbox',shape.get('dest_bbox'))
            if _intersection(box,b)<box[2]*box[3]*.8: continue
            if shape['kind']=='raster_asset': known=False
            elif shape.get('fill') not in (None,'none'):
                background=shape['fill']; known=not bool(shape.get('fill_gradient'))
        if known:
            lum=sorted([_luminance(text['color']),_luminance(background)])
            if (lum[1]+.05)/(lum[0]+.05)<1.5:
                issues.append(issue(tid,'style',box,'Text and underlying native background have near-identical colors; verify visibility against the source.'))
    groups={}
    for text in native:
        group=audits[text['id']]['style_group']
        if re.fullmatch(r'\d+(?:\.\d+)?%',text['text'].strip()): group='__percentage_labels__'
        if group: groups.setdefault(group,[]).append(text)
    for values in groups.values():
        if len(values)>1 and max(t['font_size_px'] for t in values)>min(t['font_size_px'] for t in values)*1.18:
            for text in values:
                issues.append(issue(text['id'],'style',text['bbox'],'Same-level text group has inconsistent font sizes; verify the source hierarchy.'))
    return issues


def validate_visual(raw, *, region, elements, binding, decision_binding, size):
    required={'schema','region_id','artifact_binding','decision_binding','coordinate_space','quality_passed','issues','redecision_required','redecision_reason','summary'}
    require(set(raw)==required and raw['schema']=='RegionVisualReviewDraft/1', 'invalid visual review fields')
    require(raw['coordinate_space']=='normalized_1000_xyxy', 'explicit review coordinates required')
    require(raw['region_id']==region['id'] and raw['artifact_binding']==binding and raw['decision_binding']==decision_binding, 'stale visual review')
    require(type(raw['quality_passed']) is bool and type(raw['redecision_required']) is bool, 'review flags must be booleans')
    require(isinstance(raw['summary'],str) and len(raw['summary'])<=3000, 'invalid visual summary')
    require(isinstance(raw['redecision_reason'],str) and len(raw['redecision_reason'])<=1000, 'invalid redecision reason')
    require(not raw['redecision_required'] or len(raw['redecision_reason'])>=4, 'redecision needs concrete evidence')
    require(isinstance(raw['issues'],list) and len(raw['issues'])<=60, 'invalid visual issues')
    review=transform_model_boxes(raw,size,to_model=False)
    ids={o['id'] for o in elements['objects']+elements['text_decisions']}; seen=set()
    for item in review['issues']:
        require(set(item)=={'id','kind','severity','object_ids','bbox','description'}, 'invalid visual issue fields')
        require(isinstance(item['id'],str) and re.fullmatch(r'[A-Za-z0-9_-]{1,70}',item['id']) and item['id'] not in seen,'invalid issue ID')
        seen.add(item['id'])
        require(item['kind'] in ('text','numeric','missing','duplicate','position','style','crop','layout'), 'visual review cannot infer editability or deletion')
        require(item['severity'] in ('blocking','minor') and isinstance(item['object_ids'],list) and set(item['object_ids'])<=ids,'invalid visual issue targets')
        require(inside(item['bbox'],region['bbox']), 'visual issue lies outside its fixed region')
        require(isinstance(item['description'],str) and 1<=len(item['description'])<=2000,'invalid visual issue description')
        if item['kind'] in ('text','numeric','missing','duplicate'): require(item['severity']=='blocking','content failure is blocking')
    require(not review['quality_passed'] or (not review['redecision_required'] and not any(i['severity']=='blocking' for i in review['issues'])), 'visual pass contradicts findings')
    require(review['quality_passed'] or review['issues'] or review['redecision_required'],'failed review requires findings')
    review['schema']='RegionVisualReview/1'; review.pop('coordinate_space')
    return review


def validate_probe(raw, *, region_id, binding, targets):
    require(set(raw)=={'schema','artifact_binding','region_id','ghost_free','findings'} and raw['schema']=='RegionProbeReview/1','invalid probe review')
    require(raw['artifact_binding']==binding and raw['region_id']==region_id and type(raw['ghost_free']) is bool,'probe binding mismatch')
    require(isinstance(raw['findings'],list),'probe findings required')
    for item in raw['findings']:
        require(set(item)=={'target_id','description'} and item['target_id'] in targets and isinstance(item['description'],str) and 1<=len(item['description'])<=1000,'invalid ghost finding')
    require(raw['ghost_free']==(not raw['findings']),'probe verdict contradicts findings')
    return deepcopy(raw)


def check_transitions(previous, current, changed_ids):
    """Stable issue identity uses target/kind, not the model's chosen issue label."""
    def key(item):
        return (item['kind'],tuple(sorted(item['object_ids'])),None if item['object_ids'] else tuple(round(v,1) for v in item['bbox']))
    before={key(i):i for i in previous['issues']}; after={key(i):i for i in current['issues']}
    return {'schema':'RegionIssueTransitions/1','changed_object_ids':changed_ids,
            'resolved':[before[k] for k in before.keys()-after.keys()],
            'unresolved':[after[k] for k in before.keys()&after.keys()],
            'introduced':[after[k] for k in after.keys()-before.keys()]}
