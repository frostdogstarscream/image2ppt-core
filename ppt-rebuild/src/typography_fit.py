"""Local font evidence and final-render gates; never infer success from font count."""
from __future__ import annotations
import copy
from functools import lru_cache
import math
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typography_plan import targets,read,write,digest

STYLE_FIELDS={'font','bold','color','line_spacing_pt'}


def protected(ir):
    ir=copy.deepcopy(ir)
    for n in ir['nodes']:
        if n['type']!='text': continue
        for x in [n,*n.get('paragraphs',[])]:
            for field in STYLE_FIELDS: x.pop(field,None)
            for r in x.get('runs',[]):
                for field in STYLE_FIELDS: r.pop(field,None)
    for field in ['review_notes','review_status','source_image']: ir.pop(field,None)
    return ir


def assert_frozen(before,after):
    if protected(before)!=protected(after):
        raise ValueError('Typography changed frozen wording, geometry, grouping or evidence')


def revision(irfile,ir,directory,record):
    before=read(irfile); assert_frozen(before,ir)
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    for folder in ('assets','source'):
        if (Path(irfile).parent/folder).exists(): shutil.copytree(Path(irfile).parent/folder,directory/folder)
    source=Path(irfile).parent/before['source_image']
    if not (directory/before['source_image']).exists():
        (directory/'source').mkdir(exist_ok=True); shutil.copy2(source,directory/'source/page.png'); ir['source_image']='source/page.png'
    ir['review_status']='needs_review'
    ir.setdefault('review_notes',[]).append('MoMA typography candidate with local evidence; human acceptance pending')
    target=directory/'slideir.json'; write(target,ir)
    write(directory/'authoring.json',{'tool':'image2ppt deterministic typography refinement',
          'model_id':'qwen/qwen3.8-max','ir_sha256':digest(target),
          'input_ir_sha256':digest(irfile),'decisions':['Fixed wording and object positions','Local fit and actual PPTX gates'],
          'typography_record':record})
    return target


@lru_cache(maxsize=1024)
def font(file,index,size):
    return ImageFont.truetype(file,max(1,round(size)),index=index)


def face(manifest,family,bold,text):
    choices=[f for f in manifest['fonts'] if f['family']==family and f['bold']==bold]
    if not choices: raise ValueError('Missing exact font/weight: '+family)
    found=choices[0]
    if set(text)&set(found['missing_characters']): raise ValueError('Font lacks required characters')
    return found


def glyph(text,style,manifest,size):
    """Render two scripts with their explicit fonts; preserve advances and baseline."""
    parts=[]; pen=0.; ascent=0; descent=0
    for ch in text:
        family=style['font_latin'] if ord(ch)<128 else style['font_family']
        f=face(manifest,family,style['bold'],ch)
        fo=font(f['file'],f['index'],size)
        a,d=fo.getmetrics(); ascent=max(ascent,a); descent=max(descent,d)
        parts.append((pen,ch,fo)); pen+=fo.getlength(ch)
    canvas=Image.new('L',(max(1,math.ceil(pen)+4),max(1,ascent+descent+4)),0)
    draw=ImageDraw.Draw(canvas)
    for x,ch,fo in parts: draw.text((x+2,ascent+2),ch,font=fo,fill=255,anchor='ls')
    box=canvas.getbbox()
    if not box: raise ValueError('Empty glyph')
    return np.array(canvas.crop(box)),pen


def source_boxes(ir,node):
    evidence={e['id']:e['bbox'] for e in ir['evidence'] if 'bbox' in e}
    boxes=[evidence[k] for k in node.get('evidence_ids',[]) if k in evidence]
    return boxes if len(boxes)==len(node.get('paragraphs',[])) else [node['bbox']]*len(node.get('paragraphs') or [node])


def style_from_target(t):
    return {'font_family':t['font']['ea'],'font_latin':t['font']['latin'],
            'font_size_px':t['size_px'],'bold':t['bold'],'color':t['color'],
            'line_spacing_px':t['line_spacing_px']}


def fit_line(source,background,box,text,proposal,baseline,manifest):
    x,y,w,h=map(int,box); crop=source[y:y+h,x:x+w]; bg=background[y:y+h,x:x+w]
    delta=np.abs(crop.astype(float)-bg).max(axis=2); ink=delta>40
    if ink.sum()<8: return baseline,{'status':'REVIEW_REQUIRED','reason':'no_source_ink'}
    yy,xx=np.nonzero(ink); width=int(xx.max()-xx.min()+1)
    core=delta>=max(45,float(np.percentile(delta[ink],65)))
    color=''.join(f'{v:02X}' for v in np.median(crop[core],axis=0).round().astype(int))
    results=[]
    for style in [proposal,baseline]:
        try:
            reference,advance=glyph(text,style,manifest,64)
        except ValueError: continue
        estimate=64*width/max(1,reference.shape[1])
        lo=max(5,baseline['font_size_px']*.7); hi=min(400,baseline['font_size_px']*1.35)
        sizes={round(v) for v in [estimate-1,estimate,estimate+1,style['font_size_px'],baseline['font_size_px']] if lo<=v<=hi}
        for size in sorted(sizes):
            rendered,advance=glyph(text,style,manifest,size)
            gh,gw=rendered.shape
            if gw>w+1 or gh>h+2: continue
            alpha=rendered.astype(float)/255
            for rgbhex in {style['color'],color}:
                rgb=np.array([int(rgbhex[i:i+2],16) for i in (0,2,4)])
                for dx,dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
                    left=int(xx.min())+dx; top=int(yy.min())+dy
                    if left<0 or top<0 or left+gw>w or top+gh>h: continue
                    out=bg.astype(float).copy()
                    patch=out[top:top+gh,left:left+gw]
                    out[top:top+gh,left:left+gw]=rgb*alpha[:,:,None]+patch*(1-alpha[:,:,None])
                    mask=np.abs(out-bg).max(axis=2)>40
                    recall=float((ink&cv2.dilate(mask.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)).sum()/ink.sum())
                    weight=abs(math.log(max(1,mask.sum())/ink.sum()))
                    score=float(np.abs(out-crop).mean())+2*(1-recall)+weight
                    results.append((score,{**style,'font_size_px':size,'color':rgbhex},
                                    {'score':round(score,4),'source_recall':round(recall,4),'weight_error':round(weight,4),
                                     'advance':round(advance,3),'local_alignment_px':[dx,dy]}))
    if not results: return baseline,{'status':'REVIEW_REQUIRED','reason':'no_unclipped_exact_font_candidate'}
    _,style,note=min(results,key=lambda r:r[0]); note['status']='LOCAL_CANDIDATE'
    return style,note


def propose(ir,plan,manifest,source,background,canonical=None):
    result=copy.deepcopy(ir); table={s['id']:s for s in plan['styles']}
    binding={b['target_id']:b['style_id'] for b in plan['bindings']}
    targetmap={t['id']:t for t in targets(ir)}; scale=ir['canvas']['target_px'][0]/ir['canvas']['source_px'][0]
    records=[]
    for node in result['nodes']:
        if node['type']!='text': continue
        boxes=source_boxes(ir,node)
        for pi,p in enumerate(node.get('paragraphs') or [node]):
            key=f"{node['id']}/p{pi}"; requested=table[binding[key]]
            shared=(canonical or {}).get(binding[key],requested)
            old=style_from_target(targetmap[key])
            if p.get('runs'):
                chosen=shared; note={'status':'MODEL_MIXED_RUN_CANDIDATE'}
            else:
                chosen,note=fit_line(source,background,boxes[pi],p['text'],shared,old,manifest)
            def assign(obj,s):
                obj['font']={'ea':s['font_family'],'latin':s['font_latin'],
                             'size_pt':round(s['font_size_px']*.75*scale*4)/4}
                obj['bold']=s['bold'];obj['color']=s['color'].upper()
            assign(p,chosen)
            # Estimate line baseline intervals from frozen source evidence, not
            # font height. Keep existing paragraph/line structure unchanged.
            if pi<len(boxes)-1:
                distance=boxes[pi+1][1]-boxes[pi][1]
                if distance>0: p['line_spacing_pt']=round(distance*.75*scale,3)
            for ri,r in enumerate(p.get('runs') or []):
                rk=key+f'/r{ri}'; rs=table[binding[rk]]
                rs=(canonical or {}).get(binding[rk],rs)
                assign(r,rs)
            records.append({'target_id':key,'style_id':binding[key],'original':old,'model':requested,
                            'shared':shared,'local':chosen,'fit':note,
                            'style_exception':any(chosen.get(k)!=shared.get(k) for k in ['font_family','font_latin','font_size_px','bold','color'])})
        first=(node.get('paragraphs') or [node])[0]
        for key in ['font','bold','color']: node[key]=copy.deepcopy(first[key])
    assert_frozen(ir,result)
    return result,records


def envelope(node,size,extra=8):
    x,y,w,h=node['bbox']
    return [max(0,math.floor(x)-extra),max(0,math.floor(y)-extra),
            min(size[0],math.ceil(x+w)+extra),min(size[1],math.ceil(y+h)+extra)]


def metrics(source,rendered,background,box):
    x,y,r,b=box; src=source[y:b,x:r].astype(float); out=rendered[y:b,x:r].astype(float); bg=background[y:b,x:r].astype(float)
    truth=np.abs(src-bg).max(axis=2)>45; actual=np.abs(out-bg).max(axis=2)>45
    kernel=np.ones((3,3),np.uint8)
    cover=cv2.dilate(actual.astype('uint8'),kernel).astype(bool)
    nearby=cv2.dilate(truth.astype('uint8'),kernel).astype(bool)
    return {'mae':float(np.abs(src-out).mean()),
            'recall':float((truth&cover).sum()/max(1,truth.sum())),
            'precision':float((actual&nearby).sum()/max(1,actual.sum())),
            'weight_error':abs(math.log(max(1,actual.sum())/max(1,truth.sum()))),
            'source_ink':int(truth.sum()),'rendered_ink':int(actual.sum())}


def fits(node,ir,manifest):
    scale=ir['canvas']['target_px'][0]/ir['canvas']['source_px'][0]
    height=0.
    for pi,p in enumerate(node.get('paragraphs') or [node]):
        width=0.;inkheight=0.
        for r in p.get('runs') or [{'text':p['text']}]:
            style={**p,**r}; f=style['font']
            description={'font_family':f['ea'],'font_latin':f['latin'],'bold':style.get('bold',False)}
            try: mask,advance=glyph(r['text'],description,manifest,f['size_pt']/.75/scale)
            except ValueError: return False
            width+=advance;inkheight=max(inkheight,mask.shape[0])
        if width+p.get('left_indent_px',0)>node['bbox'][2]+1: return False
        if pi<len(node.get('paragraphs',[node]))-1: height+=p.get('line_spacing_pt',p['font']['size_pt']*1.2)/.75/scale
        else: height+=inkheight
    return height<=node['bbox'][3]+2


def choose(before,proposed,source,old,new,background,manifest):
    selected=copy.deepcopy(before); decisions=[]
    proposed_nodes={n['id']:n for n in proposed['nodes']}
    size=before['canvas']['source_px']
    for index,a in enumerate(before['nodes']):
        if a['type']!='text': continue
        b=proposed_nodes[a['id']]; box=envelope(a,size)
        oldm=metrics(source,old,background,box); newm=metrics(source,new,background,box)
        fit=fits(b,before,manifest)
        reasons=[]
        if not fit: reasons.append('font_extent_or_glyph_coverage_failed')
        if oldm['mae']-newm['mae']<.20: reasons.append('insufficient_region_improvement')
        if newm['recall']<max(.70,oldm['recall']-.02): reasons.append('stroke_coverage_regressed')
        if newm['precision']<max(.60,oldm['precision']-.02): reasons.append('extra_or_displaced_strokes')
        if newm['weight_error']>oldm['weight_error']+.03: reasons.append('stroke_weight_regressed')
        if oldm['source_ink']<8: reasons.append('no_source_stroke_evidence')
        if a==b: reasons.append('unchanged')
        if not reasons: selected['nodes'][index]=copy.deepcopy(b)
        decisions.append({'id':a['id'],'bbox_xyxy':box,'selected':not reasons,'baseline':oldm,'candidate':newm,
                          'fits':fit,'reasons':reasons,'status':'IMPROVED_CANDIDATE' if not reasons else 'BASELINE_RETAINED'})
    assert_frozen(before,selected)
    return selected,decisions


def verify_selected(before,after,source,old,final,background,decisions):
    assert_frozen(before,after)
    allowed=np.zeros(old.shape[:2],bool); reports=[]
    by_id={n['id']:n for n in after['nodes']}
    for n in before['nodes']:
        if n['type']!='text': continue
        d=next(x for x in decisions if x['id']==n['id'])
        if not d['selected']:
            if n!=by_id[n['id']]: raise ValueError('Retained baseline text changed')
            continue
        x,y,r,b=d['bbox_xyxy']; allowed[y:b,x:r]=True
        m=metrics(source,final,background,d['bbox_xyxy'])
        if m['mae']>d['candidate']['mae']+.05: raise ValueError('Mixed final render changed selected text')
        reports.append({'id':n['id'],'final':m})
    outside=int((np.any(old!=final,axis=2)&~allowed).sum())
    if outside: raise ValueError('Typography modified pixels outside selected text envelopes')
    oldmae=float(np.abs(source.astype(float)-old).mean()); newmae=float(np.abs(source.astype(float)-final).mean())
    if newmae>oldmae+.00001: raise ValueError('Full page typography regressed')
    return {'status':'PASS_COMPUTATIONAL_GATES','improved':len(reports),'baseline_mae':oldmae,'final_mae':newmae,
            'outside_changed_pixels':outside,'objects':reports,'human_acceptance':False}
