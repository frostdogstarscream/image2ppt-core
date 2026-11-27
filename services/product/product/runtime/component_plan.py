"""MoMA's source-bound editing plan for the explicit native synthesis route."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, model_validator

from ..contracts import Shape, Span
from ..materialize import digest, write
from ..parallel_store import Store, key
from ..recovery.adaptive_policy import AdaptiveBudget
from .provider import planning_actor


class Component(BaseModel):
    role: Literal['panel','ribbon','pedestal','divider','connector','scenery','artwork']
    label: str
    geometry: Literal['rect','roundRect','pentagon','ellipse','rightArrow','downArrow','leftArrow','upArrow','path'] = 'rect'
    # Absolute source pixels. A closed path uses a native filled shape.
    bounds: tuple[int,int,int,int]
    points: list[tuple[int,int]] = Field(default_factory=list,max_length=32)
    arrow_at_end: bool = False
    fill: str = Field(default='none',pattern=r'^(none|[0-9A-Fa-f]{6})$')
    stroke: str = Field(default='none',pattern=r'^(none|[0-9A-Fa-f]{6})$')
    target_ids: list[str] = Field(default_factory=list)
    parent_id: str = ''

    @model_validator(mode='before')
    @classmethod
    def derive_path_bounds(cls, value):
        # An ordered polyline completely determines its source bounds.
        if isinstance(value,dict) and value.get('points') and not value.get('bounds'):
            points=value['points']
            if all(isinstance(p,(tuple,list)) and len(p)==2 and
                   all(type(n) is int for n in p) for p in points):
                xs,ys=zip(*points)
                return {**value,'bounds':(min(xs),min(ys),max(1,max(xs)-min(xs)),max(1,max(ys)-min(ys)))}
        return value


class Emphasis(BaseModel):
    source: str = Field(min_length=1)
    color: str = Field(pattern=r'^[0-9A-Fa-f]{6}$')
    bold: bool


class TextRule(BaseModel):
    target_ids: list[str] = Field(min_length=1)
    bold: bool | None = None
    align: Literal['left','right','center'] | None = None
    keep_source_width: bool = False
    emphasis: list[Emphasis] = Field(default_factory=list)
    normal_color: str | None = Field(default=None,pattern=r'^[0-9A-Fa-f]{6}$')


class PagePlan(BaseModel):
    components: list[Component] = Field(max_length=220)
    text_rules: list[TextRule] = Field(max_length=140)
    uncertainties: list[str] = Field(default_factory=list)


class ComponentCorrection(BaseModel):
    index: int = Field(ge=0)
    component: Component


class PlanCorrection(BaseModel):
    replacements: list[ComponentCorrection] = Field(default_factory=list)
    additions: list[Component] = Field(default_factory=list)
    text_rules: list[TextRule] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


def apply_correction(plan, correction):
    plan=plan.model_copy(deep=True)
    for change in correction.replacements:
        if change.index>=len(plan.components):
            raise ValueError('unknown component in page correction')
        plan.components[change.index]=change.component
    plan.components.extend(correction.additions)
    plan.text_rules.extend(correction.text_rules)
    plan.uncertainties.extend(correction.uncertainties)
    return plan


PROMPT = """为上传的原始幻灯片定可编辑组件稿。第一张是原图，第二张带每100源像素的坐标网格。
输入 objects 是初步恢复的文字、框和素材，给出实际 id、[x,y,width,height]。你决定成品要如何拆分。
components 把源图可辨认的标题带、卡片、段落底板、编号胶囊、图标圆托、横竖分隔、
所有业务连接线(包括折返回路及纵向箭头)和页边云浪装饰列出。已正确识别的普通卡片可仅引用其 id。
bounds 必须是源图绝对像素 xywh。箭头折线路径 points 使用绝对像素，从起点到终点；业务关系必须保留方向。
使用 target_ids 引用已有对象：替换同一底框、指出圆托所承载的图标。相同样式且各自有圆托的图标可在一条
pedestal 的 target_ids 列出多个 image id，程序逐一测量。parent_id 可指向宿主卡片 shape id。
scenery 给出应保留原始纹理的装饰带区域；不得将整页或文字、照片当装饰。标明实测可见的 fill/stroke 颜色，透明用 none。
text_rules 按源图指出同组字重与对齐、正文段宽，明确强调的短语逐字引用已识别的 text，跨框分别引用。
普通正文 normal_color 与 bold 指明基础样式，局部强调单列。并列小标题共享字重。
逐个核对同排图标，确实遗漏或被识别为文字时，用 artwork 给出完整图案的局部范围。流程边线分开追踪每个出入口与箭头方向，避免把外框当作反馈线。
不编造缺失的内容、对象及坐标。若有歧义写入 uncertainties。仅输出符合 schema 的 JSON。"""


def request_plan(state, graph, folder, directory, *, feedback=None, preview=None):
    source = folder/'source.png'
    if digest(source) != graph.source_sha256:
        raise ValueError('native plan source changed')
    request = json.loads((state.job/'request.json').read_text('utf8'))
    objects = [{'id':k, 'kind':r['kind'], 'box':r['bbox'],
                **({'text':r['payload']['text']} if r['kind']=='text' else {})}
               for k,r in graph.selected.items()]
    model = planning_actor(state.job)
    model_name = model.cfg.model if hasattr(model,'cfg') else model.model
    inventory=json.loads((folder/'page-structure.json').read_text('utf8'))['elements']
    contract=PlanCorrection if feedback else PagePlan
    prompt=PROMPT
    if feedback:
        prompt=prompt.replace('第二张带每100源像素的坐标网格','第二张是刚才的 PowerPoint 实际渲染')
        prompt+=' 只返回与具体缺陷有关的原组件索引及替换值、新增组件或文字规则，其余组件直接复用。'
    binding = key([graph.source_sha256, objects, inventory, prompt, contract.model_json_schema(), model_name,
                   feedback, digest(preview) if preview else None, digest(__file__)])
    directory.mkdir(exist_ok=True, parents=True)
    cache = state.job/'cache/native-component-plan'/binding
    cache.mkdir(exist_ok=True, parents=True)
    board = Image.open(source).convert('RGB')
    draw = ImageDraw.Draw(board)
    for x in range(0,board.width,100):
        draw.line((x,0,x,board.height),fill='#a1a8b3')
        draw.text((x+2,2),str(x),fill='black')
    for y in range(0,board.height,100):
        draw.line((0,y,board.width,y),fill='#a1a8b3')
        draw.text((2,y+2),str(y),fill='black')
    grid=cache/'grid.png'
    board.save(grid)
    receipt = cache/'response.json'
    if not receipt.exists():
        budget=AdaptiveBudget(Store(state.job/'state'),request.get('adaptive_budget'),page_count=len(request['sources']))
        if not budget.reserve('model_calls',{'tool':'native_component_plan','binding':binding}):
            raise RuntimeError('native component plan response unknown')
        model.ask_receipted(prompt,{'image_size':graph.image_size,'objects':objects,
                            'inventory':inventory,'feedback':feedback},
                            [source,preview if feedback else grid],contract,receipt)
    from ..parallel_provider import parse_reply
    reply=contract.model_validate(parse_reply(json.loads(receipt.read_text('utf8'))))
    if feedback:
        plan=apply_correction(PagePlan.model_validate(feedback['prior_plan']),reply)
    else:
        plan=reply
    write(directory/'moma-plan.json',{'binding':binding,'source_sha256':graph.source_sha256,
          'response_sha256':digest(receipt),'grid_sha256':digest(grid),'model':model_name,**plan.model_dump()})
    return plan


def style_text(graph, rules):
    texts={t.id:t for t in graph.text.texts}
    styles={s.id:s for s in graph.text.styles}
    report=[]
    for rule in rules:
        # A phrase may start on one visual line and finish in the next OCR field.
        ordered=[texts[k] for k in rule.target_ids if k in texts]
        joined=''.join(t.text for t in ordered)
        parts={}
        for mark in rule.emphasis:
            begin=joined.find(mark.source)
            if begin<0 or joined.count(mark.source)!=1:
                continue
            offset=0
            for field in ordered:
                left=max(0,begin-offset)
                right=min(len(field.text),begin+len(mark.source)-offset)
                if right>left:
                    parts.setdefault(field.id,[]).append((left,right,mark))
                offset+=len(field.text)
        for oid in rule.target_ids:
            t=texts.get(oid)
            if not t or oid not in graph.selected:
                report.append({'id':oid,'status':'UNKNOWN_TEXT'})
                continue
            if rule.align: t.align=rule.align
            if rule.keep_source_width or rule.align=='center':
                graph.selected[oid]['source_width']=True
            original=styles[t.style]
            changes={}
            if rule.bold is not None: changes['bold']=rule.bold
            if rule.normal_color: changes['color']=rule.normal_color
            def clone(style, overrides):
                value=style.model_copy(update=overrides)
                value.id='sty-'+key([value.ea,value.latin,value.size_px,value.color,value.bold,value.role])[:18]
                if value.id not in styles:
                    styles[value.id]=value
                    graph.text.styles.append(value)
                return value.id
            if changes:
                t.style=clone(original,changes)
                for span in t.spans:
                    span.style=clone(styles[span.style],changes)
            # Map actual source substrings. Missing spans never change the text.
            source_styles=[sid for span in t.spans for sid in [span.style]*len(span.text)] if t.spans else [t.style]*len(t.text)
            if len(source_styles)!=len(t.text):
                source_styles=[t.style]*len(t.text)
            spans=list(parts.get(oid,[]))
            for mark in rule.emphasis:
                if any(item==mark for _,_,item in spans):
                    continue
                if t.text.count(mark.source)==1:
                    start=t.text.index(mark.source)
                    spans.append((start,start+len(mark.source),mark))
                elif joined.count(mark.source)!=1:
                    report.append({'id':oid,'status':'UNRESOLVED_EMPHASIS','source':mark.source})
            for start,end,mark in spans:
                for i in range(start,end):
                    source_styles[i]=clone(styles[source_styles[i]],{'color':mark.color,'bold':mark.bold})
            if rule.emphasis:
                from itertools import groupby
                t.spans=[Span(text=''.join(letter for letter,_ in group),style=sid)
                         for sid,group in groupby(zip(t.text,source_styles),lambda p:p[1])]
            graph.selected[oid]['payload']=t.model_dump()
            report.append({'id':oid,'status':'NATIVE_TEXT'})
    return report


def matte_symbol(file):
    """Unmix only simple opaque symbols on measurably uniform local paper."""
    from reconstruction import extract_rgba
    img=Image.open(file).convert('RGBA')
    rgba=np.array(img)
    if rgba[:,:,3].min()<250:
        return 'existing-alpha'
    tile=rgba[:,:,:3].copy()
    h,w=tile.shape[:2]
    if min(w,h)<6:
        return 'retained-local-image'
    edge=np.vstack([tile[0],tile[-1],tile[:,0],tile[:,-1]])
    paper=np.median(edge,axis=0)
    if np.percentile(abs(edge-paper).max(1),75)>14:
        return 'uncertain-edge'
    distance=abs(tile.astype(float)-paper).max(2)
    ink=distance>13
    if not .04<ink.mean()<.82:
        return 'uncertain-coverage'
    bg=np.broadcast_to(paper,tile.shape).astype(np.uint8)
    result=extract_rgba(tile,bg,ink)
    alpha=result[:,:,3:4]/255.
    error=np.abs(tile-(result[:,:,:3]*alpha+bg*(1-alpha))).max(2)
    if np.percentile(error,98)>12:
        return 'uncertain-recomposition'
    Image.fromarray(result).save(file)
    return 'unmixed-on-measured-paper'


def apply_components(graph, rgb, plan, folder):
    from .native_synthesis import area, inside, iou, color
    width,height=graph.image_size
    report=style_text(graph,plan.text_rules)
    shapes={s.id:s for s in graph.surface.shapes if s.id in graph.selected}
    source_pixels=None
    ocr_file=folder/'runtime-ocr-protection.json'
    ocr_rows={r['id']:r for r in json.loads(ocr_file.read_text('utf8'))['lines']} if ocr_file.exists() else {}
    for index,c in sorted(enumerate(plan.components),key=lambda row:row[1].role!='artwork'):
        targets=[k for k in c.target_ids if k in graph.selected]
        record={'plan_index':index,'role':c.role,'label':c.label,'source_bbox':c.bounds,'object_ids':[]}
        report.append(record)
        if c.role=='artwork':
            from ..object_contracts import ClosedObject
            x,y,w,h=c.bounds
            if min(x,y)<0 or min(w,h)<5 or x+w>width or y+h>height or w*h>.045*width*height:
                record['reason']='unverified standalone symbol extent'
                continue
            from ..contracts import intersection
            if source_pixels is None:
                source_pixels=np.asarray(Image.open(folder/'source.png').convert('RGB'))
            existing=[k for k,r in graph.selected.items() if r['kind']=='asset' and iou(c.bounds,r['bbox'])>.6]
            text_boxes=[t.bbox for t in graph.text.texts if t.id in graph.selected]
            dx,dy=round(w*1.5),round(h*.6)
            x0,y0=max(0,x-dx),max(0,y-dy)
            search=source_pixels[y0:min(height,y+h+dy),x0:min(width,x+w+dx)]
            ink=(np.ptp(search,axis=2)>85).astype(np.uint8)
            for tx,ty,tw,th in text_boxes:
                left,top=max(0,round(tx-x0)),max(0,round(ty-y0))
                right,bottom=round(tx+tw-x0),round(ty+th-y0)
                if right>left and bottom>top: ink[top:bottom,left:right]=0
            size=max(3,round(min(w,h)*.14))
            joined=cv2.dilate(ink,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size)))
            count,labels,_,_=cv2.connectedComponentsWithStats(joined)
            choices=[]
            for k in range(1,count):
                ys,xs=np.where((labels==k)&(ink>0))
                if not len(xs): continue
                bw,bh=int(np.ptp(xs))+1,int(np.ptp(ys))+1
                if not (.4*w<=bw<=1.8*w and .4*h<=bh<=1.8*h): continue
                bx,by=x0+int(xs.min()),y0+int(ys.min())
                distance=((bx+bw/2-x-w/2)/w)**2+((by+bh/2-y-h/2)/h)**2
                choices.append((distance,bx,by,bw,bh))
            if choices:
                _,bx,by,bw,bh=min(choices)
                pad=max(2,round(max(bw,bh)*.085))
                x,y=max(0,bx-pad),max(0,by-pad)
                w,h=min(width,bx+bw+pad)-x,min(height,by+bh+pad)-y
            box=[x,y,w,h]
            if any(intersection(box,t)>.12*area(t) for t in text_boxes):
                record['reason']='overlapping independently editable text'
                continue
            tile=source_pixels[y:y+h,x:x+w]
            if np.percentile(tile.max(2),95)-int(np.percentile(tile.min(2),5))<25:
                record['reason']='insufficient graphic contrast'
                continue
            oid=existing[0] if existing else f'plan-symbol-{index}'
            file=folder/(oid+'.png')
            Image.fromarray(tile).save(file)
            record['matting']=matte_symbol(file)
            if existing:
                graph.asset.objects[:]=[a for a in graph.asset.objects if a.id!=oid]
            graph.asset.objects.append(ClosedObject(id=oid,description=c.label,prompt_en='source graphic',bbox=box,
                layers=[file.name],components=['symbol'],kind='icon',text_policy='exclude_independent_text',
                parts=[{'name':'symbol','bbox':box,'layers':[file.name]}]))
            graph.selected[oid]={'id':oid,'kind':'asset','bbox':box,
                'asset_reference':{'path':file.name,'sha256':digest(file),'source_sha256':graph.source_sha256,'bbox':box},
                'payload':{'method':'MoMA source graphic on blank canvas'}}
            graph.ownership[oid]='RESOLVED'
            record['object_ids'].append(oid)
            record['measured_bbox']=box
            continue
        if c.role=='scenery':
            # Preserve only observed decoration in a local tile. All detected text,
            # native carriers and images occlude it and cannot be copied onto it.
            x,y,w,h=c.bounds
            if not (0<=x<x+w<=width and 0<=y<y+h<=height and w*h < .4*width*height):
                record['reason']='decoration extent not locally bounded'
                continue
            from ..object_contracts import ClosedObject
            tile=rgb[y:y+h,x:x+w]
            allowed=np.ones((h,w),bool)
            for b in [t.bbox for t in graph.text.texts]+[r['bbox'] for r in graph.selected.values()]:
                bx,by,bw,bh=map(round,b)
                left,top=max(0,bx-x-2),max(0,by-y-2)
                right,bottom=min(w,bx+bw-x+2),min(h,by+bh-y+2)
                if right>left and bottom>top: allowed[top:bottom,left:right]=False
            ground=np.frombuffer(bytes.fromhex(graph.surface.background),dtype=np.uint8)
            # Only pale continuous decorative ink on a near-white page is supported
            # here. Detailed scenery needs an independent clean LayerD backing.
            allowed &= (tile.min(2)>154) & (np.ptp(tile,axis=2)>7)
            if ground.min()<210 or allowed.sum()<48:
                record['reason']='no verified light decorative surface'
                continue
            oid=f'plan-bg-{index}'
            file=folder/(oid+'.png')
            rgba=np.dstack([tile,allowed.astype('uint8')*255])
            Image.fromarray(rgba).save(file)
            bbox=[x,y,w,h]
            graph.asset.objects.append(ClosedObject(id=oid,description=c.label,prompt_en='page decoration',bbox=bbox,
                layers=[file.name],components=['decoration'],kind='illustration',text_policy='exclude_independent_text',
                parts=[{'name':'decoration','bbox':bbox,'layers':[file.name]}]))
            graph.selected[oid]={'id':oid,'kind':'asset','bbox':bbox,'container_layer':True,'page_decoration':True,
                'asset_reference':{'path':file.name,'sha256':digest(file),'source_sha256':graph.source_sha256,'bbox':bbox},
                'payload':{'method':'visible decoration outside measured components'}}
            graph.ownership[oid]='RESOLVED'
            record['object_ids'].append(oid)
            continue
        image_targets=[k for k in targets if graph.selected[k]['kind']=='asset']
        boxes=[list(c.bounds)]
        if c.role=='pedestal' and image_targets:
            boxes=[]
            record['matting']={}
            for target in image_targets:
                ref=graph.selected[target]['asset_reference']
                path=folder/ref['path']
                record['matting'][target]=matte_symbol(path)
                ref['sha256']=digest(path)
            for oid in image_targets:
                x,y,w,h=graph.selected[oid]['bbox']
                diameter=round(max(w,h)*1.08)
                boxes.append([round(x+(w-diameter)/2),round(y+(h-diameter)/2),diameter,diameter])
        for part,b in enumerate(boxes):
            custom=None
            if c.role=='pedestal' and c.geometry=='ellipse' and not image_targets and max(b[2:])<=12:
                x,y,w,h=b
                radius=max(12,round(.011*width))
                left,top=max(0,x-radius),max(0,y-radius)
                if source_pixels is None:
                    source_pixels=np.asarray(Image.open(folder/'source.png').convert('RGB'))
                crop=source_pixels[top:min(height,y+h+radius),left:min(width,x+w+radius)]
                if not crop.size:
                    record['reason']='list marker outside source'
                    continue
                ink=(np.ptp(crop,axis=2)>55).astype(np.uint8)
                count,labels,stats,centers=cv2.connectedComponentsWithStats(ink)
                hits=[]
                for k in range(1,count):
                    bx,by,bw,bh,n=stats[k]
                    if 2<=bw<=w+5 and 2<=bh<=h+5 and .65<=bw/bh<=1.5 and n>=.5*bw*bh:
                        ax,ay=centers[k]
                        hits.append(((left+ax-x-w/2)**2+(top+ay-y-h/2)**2,k))
                if not hits:
                    record['reason']='no compact list marker observed near proposal'
                    continue
                _,k=min(hits)
                bx,by,bw,bh,_=map(int,stats[k])
                b=[left+bx,top+by,bw,bh]
                # Bind a source dot to its OCR line. A separately drawn dot
                # would stay at the old baseline when editable text reflows.
                cx,cy=b[0]+bw/2,b[1]+bh/2
                nearest=[]
                for text in graph.text.texts:
                    if text.id not in graph.selected:
                        continue
                    lines=text.text.split('\n')
                    members=graph.text.members.get(text.id,[])
                    if len(lines)!=len(members): continue
                    for row_index,member in enumerate(members):
                        if member not in ocr_rows: continue
                        tx,ty,tw,th=ocr_rows[member]['bbox']
                        if tx-1.7*th<=cx<=tx+.45*th and ty-1<=cy<=ty+th+1:
                            nearest.append((abs(cy-ty-th/2)+.1*abs(cx-tx),text.id,row_index))
                if nearest:
                    _,tid,row_index=min(nearest)
                    text=next(t for t in graph.text.texts if t.id==tid)
                    lines=text.text.split('\n')
                    if not lines[row_index].lstrip().startswith(('·','•','●','▪')):
                        if text.spans:
                            record['reason']='list marker requires a styled paragraph edit'
                            continue
                        lines[row_index]='·'+lines[row_index]
                        text.text='\n'.join(lines)
                        graph.selected[tid]['payload']=text.model_dump()
                    record.update(object_ids=[tid],measured_bbox=b,method='marker incorporated in editable paragraph')
                else:
                    record['reason']='source dot has no verified paragraph baseline'
                continue
            if c.geometry=='path' and len(c.points)>1:
                pts=np.array(c.points)
                if (pts<0).any() or (pts[:,0]>=width).any() or (pts[:,1]>=height).any():
                    record['reason']='path outside page'
                    continue
                x,y=pts.min(0).tolist()
                w,h=np.maximum(1,np.ptp(pts,axis=0)).tolist()
                b=[x,y,w,h]
                custom=[{'width':float(w),'height':float(h),'commands':[
                    {'type':'moveTo' if i==0 else 'lineTo','x':float(px-x),'y':float(py-y)}
                    for i,(px,py) in enumerate(c.points)]}]
            x,y,w,h=b
            if w<=0 or h<=0 or min(x,y)<0 or x+w>width or y+h>height or w*h>.94*width*height:
                record['reason']='geometry outside local source'
                continue
            near=[k for k,s in shapes.items() if iou(s.bbox,b)>.74]
            old=next((k for k in targets if k in shapes and iou(shapes[k].bbox,b)>.4),next(iter(near),None))
            if (c.geometry=='path' and not custom) or (c.fill=='none' and c.stroke=='none'):
                record['reason']='missing drawable geometry or paint'
                continue
            oid=old or f'plan-{index}-{part}'
            if old and c.role in ('panel','ribbon') and iou(shapes[old].bbox,b)>.78:
                b=list(shapes[old].bbox)
            options={'id':oid,'bbox':b,'geometry':'custom' if custom else c.geometry,
                     'fill':c.fill,'line_color':c.stroke if c.stroke!='none' else 'FFFFFF',
                     'line_width_pt':.85 if c.stroke!='none' else 0,'z':shapes[old].z if old else 300+index}
            if c.arrow_at_end: options['tail_end']='triangle'
            if c.role=='pedestal' and image_targets:
                # The ink inside a symbol often differs from its supporting disc.
                bx,by,bw,bh=map(round,b)
                crop=rgb[by:by+bh,bx:bx+bw]
                yy,xx=np.indices(crop.shape[:2])
                radius=((xx+.5-bw/2)/(bw/2))**2+((yy+.5-bh/2)/(bh/2))**2
                rim=crop[(radius>.72)&(radius<.96)]
                if rim.size:
                    options.update(fill=color(np.median(rim,axis=0)),line_width_pt=0)
                graphic=graph.selected[image_targets[part]]
                if .82<graphic['bbox'][2]/graphic['bbox'][3]<1.22 and c.geometry=='ellipse':
                    graphic['payload']['clip_geometry']='ellipse'
            if custom: options['custom_paths']=custom
            value=Shape(**options)
            if old: graph.surface.shapes.remove(shapes[old])
            graph.surface.shapes.append(value)
            shapes[oid]=value
            graph.ownership[oid]='RESOLVED'
            graph.selected[oid]={**graph.selected.get(oid,{}),'id':oid,'kind':'surface','bbox':b}
            if c.role=='panel':
                graph.selected[oid]['planned_panel']=True
                for target in targets:
                    if target!=oid and inside(graph.selected[target]['bbox'],b):
                        graph.selected[target]['planned_parent']=oid
            if c.parent_id in shapes and c.parent_id!=oid and inside(b,shapes[c.parent_id].bbox):
                graph.selected[oid]['planned_parent']=c.parent_id
            record['object_ids'].append(oid)
    return report
