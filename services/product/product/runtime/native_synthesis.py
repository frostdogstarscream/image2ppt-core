"""Project measured page components onto a blank canvas for the direct runner."""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..contracts import Shape, intersection
from ..materialize import digest, write
from ..object_contracts import ClosedObject
from ..parallel_contracts import SceneGraph
from ..parallel_rebuild import adapt


def area(box):
    return box[2] * box[3]


def iou(a, b):
    overlap = intersection(a, b)
    return overlap / max(1, area(a) + area(b) - overlap)


def color(rgb):
    return ''.join(f'{int(np.clip(round(v),0,255)):02X}' for v in rgb)


def inside(inner, outer):
    return intersection(inner, outer) >= .84 * area(inner)


def component_outlines(rgb):
    """Measure closed rectilinear carriers at several source contrast levels."""
    height, width = rgb.shape[:2]
    value = rgb.min(axis=2)
    chroma = np.ptp(rgb, axis=2)
    masks = [value < threshold for threshold in (248,237,220,185)]
    masks += [chroma > threshold for threshold in (28,85)]
    masks.append(cv2.Canny(rgb, 18, 56) > 0)
    # Bridge small anti-aliased gaps in pale extended panel outlines.
    masks.append(cv2.morphologyEx((chroma>16).astype('uint8'),cv2.MORPH_CLOSE,
                                  np.ones((7,7),np.uint8)) > 0)
    boxes = []
    for binary in masks:
        contours, _ = cv2.findContours(binary.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            # Decorative edge strokes can connect unrelated cards to the page border.
            if w > .91*width and h > .5*height and (x < 2 or y+h >= height-1):
                continue
            if w < width*.035 or h < 5 or area((x,y,w,h)) > width*height*.91:
                continue
            if cv2.contourArea(contour) < w*h*.79:
                continue
            if w < h*.75 or (h < height*.018 and w < width*.09):
                continue
            boxes.append([x,y,w,h])
    unique = []
    for box in sorted(boxes,key=area,reverse=True):
        if not any(iou(box,other)>.88 for other in unique):
            unique.append(box)
    return unique


def observed_box(item, width, height, anchors):
    """Resolve model xywh/xyxy ambiguity only against measured outlines or OCR/layout."""
    x,y,w,h = map(round,item['bbox'])
    candidates = [[x,y,w,h]]
    if item['id'].startswith('e') and w>x and h>y:
        candidates.append([x,y,w-x,h-y])
    legal = [b for b in candidates if min(b)>=0 and min(b[2:])>0 and b[0]+b[2]<=width and b[1]+b[3]<=height]
    if not legal:
        return None
    return max(legal,key=lambda b:max((iou(b,o) for o in anchors),default=0))


def repeated_panels(boxes, text_boxes, photos, canvas):
    """Recover repeated photo cards from closed outlines and contained content."""
    width, height = canvas
    eligible = [b for b in boxes if .09 * width < b[2] < .65 * width
                and .16 * height < b[3] < .85 * height
                and sum(inside(t, b) for t in text_boxes) >= 2]
    accepted = []
    for box in sorted(eligible, key=area, reverse=True):
        if any(inside(box, larger) for larger in accepted):
            continue
        peers = [b for b in eligible if abs(b[1] - box[1]) <= .015 * height
                 and abs(b[3] - box[3]) <= .08 * box[3] and .86 < b[2] / box[2] < 1.16]
        if len([p for p in peers if intersection(p, box) == 0]) >= 2 and any(
                inside(photo, peer) for photo in photos for peer in peers):
            accepted.append(box)
    return accepted


def measured_icon_box(rgb, box):
    """Complete a clipped high-chroma symbol using its connected source ink."""
    x, y, w, h = box
    height, width = rgb.shape[:2]
    dx, dy = round(.6 * w), round(.7 * h)
    left, top = max(0, x-dx), max(0, y-dy)
    tile = rgb[top:min(height, y+h+dy), left:min(width, x+w+dx)]
    # Include separated strokes (such as both halves of a refresh symbol).
    ink = (np.ptp(tile, axis=2) > 95).astype('uint8')
    radius = max(1, round(min(w, h) * .065))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    joined = cv2.dilate(ink, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    choices = []
    for index in range(1, count):
        ys, xs = np.where((labels == index) & (ink != 0))
        if not len(xs):
            continue
        bounds = [left+int(xs.min()), top+int(ys.min()), int(np.ptp(xs))+1, int(np.ptp(ys))+1]
        if (.45*w <= bounds[2] <= 1.45*w and .45*h <= bounds[3] <= 1.45*h
                and intersection(bounds, box) >= .18 * area(box)):
            choices.append(bounds)
    if not choices:
        return box
    bx, by, bw, bh = max(choices, key=lambda b: iou(b, box))
    if inside([bx, by, bw, bh], box):
        return box
    pad = max(2, round(.06*max(bw, bh)))
    left, top = max(0, bx-pad), max(0, by-pad)
    return [left, top, min(width, bx+bw+pad)-left, min(height, by+bh+pad)-top]


def clean_source(graph, folder):
    rgb = np.array(Image.open(folder/'source.png').convert('RGB'))
    unverified = []
    for obj in graph.selected.values():
        if obj['kind'] != 'text':
            continue
        ref = obj.get('owner_evidence',{})
        path = (folder/ref.get('file','')).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or digest(path)!=ref.get('sha256'):
            unverified.append(obj['bbox'])
            continue
        with np.load(path,allow_pickle=False) as proof:
            if 'backing_patch' not in proof:
                unverified.append(obj['bbox'])
                continue
            x,y,w,h = map(int,proof['restore_bbox'])
            patch = proof['backing_patch']
            mask = proof['mask']
            if mask.dtype!=bool or mask.shape!=rgb.shape[:2] or patch.shape!=(h,w,3):
                raise ValueError('invalid source-bound text restoration evidence')
            local = mask[y:y+h,x:x+w]
            rgb[y:y+h,x:x+w][local] = patch[local]
    return rgb, unverified


def carrier_style(rgb, box, child_boxes):
    x,y,w,h = box
    tile = rgb[y:y+h,x:x+w]
    allowed = np.ones((h,w),bool)
    for child in child_boxes:
        if area(child)>=.95*w*h or not inside(child,box):
            continue
        xx,yy,ww,hh = map(round,child)
        allowed[max(0,yy-y):min(h,yy+hh-y),max(0,xx-x):min(w,xx+ww-x)] = False
    margin = min(4,max(1,min(w,h)//7))
    interior = allowed.copy()
    interior[:margin] = interior[-margin:] = False
    interior[:,:margin] = interior[:,-margin:] = False
    samples = tile[interior] if interior.any() else tile.reshape(-1,3)
    fill = np.median(samples,axis=0)
    line = np.median(tile[0,w//4:max(w//4+1,3*w//4)],axis=0)
    result = {'fill':color(fill),'line_color':color(line),'line_width_pt':.65 if np.abs(line-fill).max()>10 else 0}
    # Fit gradients from both sides on the same carrier. Reject textured bands.
    for axis,angle in ((0,90),(1,0)):
        length = (h,w)[axis]
        index = np.indices((h,w))[axis]/max(1,length-1)
        stops = []
        for a,b in ((.05,.25),(.4,.6),(.75,.95)):
            band = tile[interior & (index>=a) & (index<=b)]
            if len(band)<8:
                break
            mid = np.median(band,axis=0)
            if np.percentile(abs(band-mid).max(1),65)>22:
                break
            stops.append(mid)
        if len(stops)==3 and np.max(abs(stops[0]-stops[2]))>12 and np.max(abs((stops[0]+stops[2])/2-stops[1]))<17:
            result['fill_gradient'] = {'angle_deg':angle,'stops':[{'offset':0,'color':color(stops[0])},
                                                                  {'offset':100000,'color':color(stops[2])}]}
            break
    return result


def text_backplate(rgb, box):
    """Measure a saturated ribbon from orthogonal scans through its restored paper."""
    x,y,w,h=map(round,box)
    height,width=rgb.shape[:2]
    cx,cy=min(width-1,x+w//2),min(height-1,y+h//2)
    saturated=np.ptp(rgb,axis=2)>62
    if not saturated[cy,cx]:
        return None
    def bounds(values, center):
        off=np.flatnonzero(~values)
        left=off[off<center]
        right=off[off>center]
        return (int(left[-1])+1 if len(left) else 0,int(right[0]) if len(right) else len(values))
    lx,rx=bounds(saturated[cy],cx)
    ty,by=bounds(saturated[:,cx],cy)
    if rx-lx<.9*w or not .65*h<=by-ty<3*h or rx-lx>width*.85:
        return None
    return [lx,ty,rx-lx,by-ty]


def fit_editing_space(ir, panels):
    """Use free space inside the measured card or row for naturally wrapping prose."""
    texts=[n for n in ir['nodes'] if n['type']=='text']
    width,height=ir['canvas']['source_px']
    for n in texts:
        if len(n['text'].strip())<8:
            continue
        x,y,w,h=n['bbox']
        # The smallest carrier permitting the present text is its natural boundary.
        carriers=[p for p in panels if inside(n['bbox'],p) and p[2]>=w and p[3]>=h]
        outer=min(carriers,key=area,default=[0,0,width,height])
        right,bottom=outer[0]+outer[2]-3,outer[1]+outer[3]-2
        for peer in texts:
            if peer is n:
                continue
            px,py,pw,ph=peer['bbox']
            if y<py+ph and py<y+h and px>=x+w-1:
                right=min(right,px-2)
            if x<px+pw and px<max(x+w,right) and py>=y+h-2:
                bottom=min(bottom,py-1)
        n['bbox']=[x,y,max(w,min(right-x,w*1.6)),max(h,min(bottom-y,h*1.65))]
        n['wrap']=True


def order_components(ir, groups):
    ordered = sorted(ir['nodes'],key=lambda node:node['z'])
    member = {oid:i for i,group in enumerate(groups) for oid in group}
    # Ungrouped foreground must remain above carriers. Native groups form contiguous stacks.
    def order(node):
        if node['id'] in member:
            return 1,member[node['id']],node['z']
        return (0 if node['type']=='shape' else 2),0,node['z']
    for z,node in enumerate(sorted(ordered,key=order)):
        node['z'] = z


def synthesize_page(page, destination):
    base = Path(page['ir']).parent
    graph = SceneGraph.model_validate_json((base/'scene.json').read_text('utf8'))
    structure = json.loads((base/'page-structure.json').read_text('utf8'))
    if structure['source_sha256']!=graph.source_sha256 or digest(base/'source.png')!=graph.source_sha256:
        raise ValueError('native synthesis source or component inventory changed')
    if not structure.get('elements'):
        raise ValueError('native synthesis requires an observed component inventory')
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('source.png','runtime-ocr-protection.json','page-structure.json','embedded-ui-frames.json','fonts.json'):
        if (base/name).is_file():
            shutil.copyfile(base/name,destination/name)
    rgb,unverified = clean_source(graph,base)
    height,width = rgb.shape[:2]
    boxes = component_outlines(rgb)
    elements = structure['elements']
    anchors = boxes + [e['bbox'] for e in elements if e['id'].startswith('inv-')]
    logical = [(e,observed_box(e,width,height,anchors)) for e in elements]
    logical = [(e,b) for e,b in logical if b]
    original = graph.selected
    graph.selected = {k:v.copy() for k,v in original.items() if v['kind'] in ('text','chart','table')}
    for rec in graph.selected.values():
        rec.pop('move_members',None)
        ref = rec.get('owner_evidence')
        if ref and (base/ref['file']).is_file():
            shutil.copyfile(base/ref['file'],destination/ref['file'])
    graph.raster_backing = {'method':'native_synthesis','source_sha256':graph.source_sha256}
    graph.surface.background = color(np.median(np.vstack((rgb[0],rgb[-1],rgb[:,0],rgb[:,-1])),axis=0))
    # Use a native filled shape for a measured continuous page gradient. Opposite
    # edges must agree across the unused canvas; a decorative corner is insufficient.
    for near, far, mid, angle in ((rgb[0], rgb[-1], rgb[height//2], 90),
                                  (rgb[:,0], rgb[:,-1], rgb[:,width//2], 0)):
        start,end = np.median(near,axis=0),np.median(far,axis=0)
        if np.max(abs(start-end)) < 12:
            continue
        if (np.mean(np.max(abs(near-start),axis=1)<8) < .70
                or np.mean(np.max(abs(far-end),axis=1)<8) < .70
                or np.mean(np.max(abs(mid-(start+end)/2),axis=1)<10) < .38):
            continue
        oid = 'syn-page-gradient'
        background = Shape(id=oid,bbox=[0,0,width,height],z=0,geometry='rect',fill=color(start),
            line_width_pt=0,fill_gradient={'angle_deg':angle,'stops':[
            {'offset':0,'color':color(start)},{'offset':100000,'color':color(end)}]})
        graph.surface.shapes.append(background)
        graph.ownership[oid]='RESOLVED'
        graph.selected[oid]={'id':oid,'kind':'surface','bbox':background.bbox}
        break
    photos = [b for e,b in logical if e['category']=='photo' and area(b)<.9*width*height]
    panels = []
    for e,b in logical:
        if e['category']!='card_shape' or not boxes:
            continue
        nearest = max(boxes,key=lambda other:iou(b,other))
        if iou(b,nearest)>.55 and nearest not in panels:
            panels.append(nearest)
    if not panels:
        panels = repeated_panels(boxes, [t.bbox for t in graph.text.texts], photos, (width,height))
    # A missing semantic card can share the measured size, alignment and spacing of its siblings.
    for b in boxes:
        if any(abs(b[1]-p[1])<height*.016 and abs(b[3]-p[3])<max(8,p[3]*.1)
               and .78<b[2]/p[2]<1.26 and intersection(b,p)==0 for p in panels):
            if all(iou(b,p)<.2 for p in panels):
                panels.append(b)
    photo_boxes = []
    for photo in photos:
        parent = min((p for p in panels if inside(photo,p)),key=area,default=None)
        if parent and photo[2]>.79*parent[2]:
            bottoms = [b[1] for b in boxes if inside(b,parent) and .86<b[2]/parent[2]<1.05
                       and abs(b[1]-(photo[1]+photo[3]))<.15*parent[3]]
            if bottoms:
                bottom = min(bottoms,key=lambda n:abs(n-photo[1]-photo[3]))
                photo = [parent[0]+1,parent[1]+1,parent[2]-2,bottom-parent[1]-1]
        if not any(iou(photo,p)>.68 for p in photo_boxes):
            photo_boxes.append(photo)
    # Recover a missing photo only when equivalent siblings have independently
    # measured photo boundaries and this site also contains photographic texture.
    for panel in panels:
        if any(inside(photo, panel) for photo in photo_boxes):
            continue
        peers = [p for p in panels if p != panel and abs(p[1]-panel[1]) < height*.015
                 and abs(p[3]-panel[3]) < panel[3]*.1 and .8 < p[2]/panel[2] < 1.2]
        ratios = [(b[1]+b[3]-p[1])/p[3] for p in peers for b in photo_boxes
                  if inside(b, p) and b[2] > .78*p[2]]
        if len(ratios) < 2 or np.ptp(ratios) > .08:
            continue
        x,y,w,h = panel
        end = y+round(h*np.median(ratios))
        tile = rgb[y+2:end,x+2:x+w-2]
        if tile.size and np.mean(cv2.Canny(tile,45,110)>0) > .024 and np.std(tile,axis=(0,1)).max() > 32:
            photo_boxes.append([x+1,y+1,w-2,end-y-1])
    icon_boxes = []
    for e,b in logical:
        if e['category'] not in ('icon','logo') or area(b)>width*height*.06 or any(inside(b,p) for p in photo_boxes):
            continue
        if e['category']=='icon' and e['id'].startswith('e'):
            b=measured_icon_box(rgb,b)
        if any(iou(b,other)>.45 or inside(b,other) for other in icon_boxes):
            continue
        icon_boxes.append(b)
    # Include complete small graphics with independent owner evidence that the
    # layout service missed. For repeated flow arrows use an editable shape.
    arrows=[]
    for rec in original.values():
        if rec['kind']!='asset' or rec.get('region_crop'):
            continue
        b=list(rec['bbox'])
        x,y,w,h=map(round,b)
        if max(w,h)>width*.086 or any(iou(b,p)>.23 or inside(b,p) for p in photo_boxes+icon_boxes):
            continue
        if rec['payload'].get('method')=='cleaned-plate-container-unmix/1':
            if .8<w/max(1,h)<1.8 and any(abs(y-other['bbox'][1])<4 and abs(w-other['bbox'][2])<4
                      and other['id']!=rec['id'] and other['kind']=='asset' for other in original.values()):
                if w<width*.035 and np.ptp(rgb[y:y+h,x:x+w],axis=2).max()>50:
                    arrows.append(b)
            continue
        icon_boxes.append(b)
    picture_boxes = photo_boxes+icon_boxes
    for t in graph.text.texts:
        if t.id in graph.selected or graph.ownership.get(t.id)!='RESOLVED' or not graph.text.members.get(t.id):
            continue
        # A native field on an empty surface needs no destructive erase. Text
        # inside a retained picture still requires an independently cleaned site.
        if any(intersection(t.bbox,p)>0 for p in picture_boxes):
            continue
        graph.selected[t.id]={'id':t.id,'kind':'text','bbox':list(t.bbox),'payload':t.model_dump(),
                              'method':'source-ocr-on-native-surface'}
    decorative = [b for b in boxes if b not in panels and not any(inside(b,p) or iou(b,p)>.72 for p in picture_boxes)
                  and not any(.1<iou(b,p)<.84 and inside(p,b) for p in panels)
                  and (b[2]/b[3]>1.25 or b[2]>width*.14)]
    shapes = panels + [b for b in decorative if not any(iou(b,p)>.78 for p in panels)]
    styles={s.id:s for s in graph.text.styles}
    for t in graph.text.texts:
        if t.id not in graph.selected or min(bytes.fromhex(styles[t.style].color))<218:
            continue
        if any(inside(t.bbox,p) for p in photo_boxes):
            continue
        b=text_backplate(rgb,t.bbox)
        if b and not any(iou(b,other)>.74 for other in shapes):
            shapes.append(b)
    shapes += [b for b in arrows if not any(iou(b,other)>.78 for other in shapes)]
    parent_shapes = {}
    for i,b in enumerate(sorted(shapes,key=area,reverse=True)):
        oid = f'syn-s-{i:03d}'
        style = carrier_style(rgb,b,shapes+picture_boxes)
        geometry = 'rightArrow' if b in arrows else ('roundRect' if b[3]>9 else 'rect')
        if b in arrows:
            x,y,w,h=b
            pixels=rgb[y:y+h,x:x+w].reshape(-1,3)
            style={'fill':color(np.median(pixels[np.ptp(pixels,axis=1)>=np.percentile(np.ptp(pixels,axis=1),85)],axis=0))}
        shape = Shape(id=oid,bbox=b,z=i+1,geometry=geometry,
                      corner_radius_px=min(11,b[3]*.14) if geometry=='roundRect' else None,**style)
        graph.surface.shapes.append(shape)
        graph.ownership[oid]='RESOLVED'
        graph.selected[oid]={'id':oid,'kind':'surface','bbox':b}
        if b in panels:
            parent_shapes[oid]=b
    skipped = []
    for i,b in enumerate(picture_boxes):
        if any(intersection(b,r)>0 for r in unverified):
            skipped.append({'bbox':b,'reason':'overlaid text lacks verified clean backing'})
            continue
        oid=f'syn-img-{i:03d}'
        x,y,w,h=b
        file=destination/(oid+'.png')
        # Match a complete, measured transparent source asset before cropping.
        matches=[v for v in original.values() if v['kind']=='asset' and not v.get('region_crop')
                 and not v.get('container_layer') and iou(v['bbox'],b)>.57 and b not in photo_boxes]
        if matches:
            match=max(matches,key=lambda v:iou(v['bbox'],b))
            path=base/match['asset_reference']['path']
            if digest(path)!=match['asset_reference']['sha256']:
                raise ValueError('native graphic binding changed')
            shutil.copyfile(path,file)
            b=list(match['bbox'])
        else:
            Image.fromarray(rgb[y:y+h,x:x+w]).save(file)
        ref={'path':file.name,'bbox':b,'sha256':digest(file),'source_sha256':graph.source_sha256}
        label='photo' if i<len(photo_boxes) else 'graphic'
        part={'name':label,'bbox':b,'layers':[file.name]}
        graph.asset.objects.append(ClosedObject(id=oid,description=label,prompt_en=label,bbox=b,layers=[file.name],
            text_policy='exclude_independent_text',components=[label],kind='illustration',parts=[part]))
        graph.ownership[oid]='RESOLVED'
        graph.selected[oid]={'id':oid,'kind':'asset','bbox':b,'asset_reference':ref,
                             'payload':{'method':'verified-layer' if matches else 'text-cleared-local-crop'}}
    for oid,rec in graph.selected.items():
        if oid in parent_shapes:
            continue
        b = rec['bbox']
        parent = min((p for p in parent_shapes if inside(b,parent_shapes[p])),key=lambda p:area(parent_shapes[p]),default=None)
        if parent:
            graph.selected[parent].setdefault('move_members',[parent]).append(oid)
    ir=adapt(graph,destination)
    write(destination/'scene.json',graph.model_dump(by_alias=True))
    write(destination/'slideir.json',ir)
    write(destination/'synthesis.json',{'source_sha256':graph.source_sha256,'observed_panels':panels,
          'pictures':picture_boxes,'skipped':skipped,'groups':ir['groups'],'background':ir['background']})
    return {**page,'ir':str(destination/'slideir.json'),'scene':str(destination/'scene.json')}
