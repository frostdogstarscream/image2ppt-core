"""Source-bound candidate evidence for the six-model comparison."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw

from .materialize import digest, write
from .parallel_store import key

PROFILES = ('baseline','marks','pp-l','doclayout','omni','vitmatte','local-layerd','vtracer','combined')
ALIASES = {'figure':'image','picture':'image','plain_text':'text','figure_caption':'figure_caption',
           'table_caption':'table_caption', 'abandon':'other'}


def invoke(op, image, directory, **fields):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    raw = buf.getvalue()
    receipt = {'model':op, 'input_sha256':hashlib.sha256(raw).hexdigest(), 'size':list(image.size),
               'status':'unavailable','regions':[], 'peak_device_bytes':None}
    started = time.monotonic()
    try:
        cfg = json.loads(Path(os.environ['IMAGE2PPT_EXPERIMENT_CONFIG']).read_text(encoding='utf-8-sig'))
        request = json.dumps({'op':op, 'png':base64.b64encode(raw).decode(), **fields}).encode()
        command = cfg[op]
        if not isinstance(command,list) or Path(command[0]).name.lower() not in ('ssh','ssh.exe'):
            raise ValueError('research model workers require OpenSSH transport')
        proc = subprocess.run(command,input=request,capture_output=True,timeout=cfg.get('timeout',300),check=True)
        line = next(line for line in reversed(proc.stdout.splitlines()) if line.startswith(b'IMAGE2PPT_EXPERIMENT='))
        out = json.loads(line.split(b'=',1)[1])
        if out.get('error'):
            raise RuntimeError(out['error'] + ': ' + out['detail'])
        if out['op'] != op or out['input_sha256'] != receipt['input_sha256']:
            raise ValueError('worker input binding mismatch')
        for field in ('rgba','svg'):
            if field in out:
                file = directory / ('candidate.png' if field=='rgba' else 'candidate.svg')
                data = out.pop(field)
                file.write_bytes(base64.b64decode(data) if field=='rgba' else data.encode('utf8'))
                receipt[field] = str(file.resolve())
                receipt[field+'_sha256'] = digest(file)
        if op=='local-layerd':
            layers=[]
            for n,layer in enumerate(out.pop('layers',[])):
                data=base64.b64decode(layer['png'],validate=True)
                if hashlib.sha256(data).hexdigest()!=layer['sha256']:
                    raise ValueError('local LayerD layer binding mismatch')
                file=directory/f'layer-{n:02d}.png'
                file.write_bytes(data)
                layers.append({'z':layer['z'],'sha256':layer['sha256'],'path':str(file.resolve())})
            receipt['layers']=layers
        receipt.update(out)
        receipt['status'] = 'candidate' if receipt.get('regions') or receipt.get('layers') or 'rgba' in receipt or 'svg' in receipt else 'no_candidates'
    except (KeyError,StopIteration,OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
        receipt['reason'] = type(exc).__name__ + ': ' + str(exc)[:500]
    receipt['elapsed_seconds'] = time.monotonic()-started
    write(directory/'receipt.json',receipt)
    return receipt


def overlap(a,b):
    x,y,w,h=a
    u,v,p,q=b
    return max(0,min(x+w,u+p)-max(x,u))*max(0,min(y+h,v+q)-max(y,v))


def detect(source, profile, directory, window=None, *, detectors=None):
    img = Image.open(source).convert('RGB')
    W,H = img.size
    models = ['pp-l','doclayout'] if profile=='combined' else [profile if profile in ('pp-l','doclayout') else 'v3']
    rows, receipts = [], []
    for op in detectors if detectors is not None else models + (['omni'] if profile in ('combined','omni') else []):
        windows = [[0,0,W,H]]
        if window:
            windows = [window]
        elif op=='omni':
            windows += [r['bbox'] for r in rows if r['label'] in ('image','table','chart') and min(r['bbox'][2:])>=64 and r['bbox'][2]*r['bbox'][3]<W*H*.8][:24]
        # The detector returns original coordinates for each search window.
        if op!='omni' and window:
            x,y,w,h=window
            view=img.crop((x,y,x+w,y+h))
        else:
            x=y=0
            view=img
        result=invoke(op,view,directory/op,windows=windows)
        receipts.append({k:v for k,v in result.items() if k!='regions'})
        for pred in sorted(result['regions'],key=lambda r:-r['score']):
            l,t,r,b=pred['xyxy']
            l,t,r,b=max(0,round(l)+x),max(0,round(t)+y),min(W,round(r)+x),min(H,round(b)+y)
            box=[l,t,r-l,b-t]
            if min(box[2:])<=0: continue
            label=pred['label'].lower().replace(' ','_')
            label=ALIASES.get(label,label)
            provenance={'model':op,'label':pred['label'],'score':pred['score'],'window':pred.get('window',window)}
            for row in rows:
                inter=overlap(row['bbox'],box)
                union=row['bbox'][2]*row['bbox'][3]+box[2]*box[3]-inter
                if row['label']==label and inter/union>=.8:
                    row['sources'].append(provenance)
                    break
            else:
                rows.append({'label':label,'bbox':box,'score':pred['score'],'sources':[provenance]})
    sha=digest(source)
    for row in rows:row['id']='m-'+key([sha,row['label'],row['bbox']])[:16]
    return {'schema':'LayoutRegions/1','source_sha256':sha,'regions':rows,'models':receipts}


def mark_page(source, layout, objects, directory):
    rows=list(layout['regions'])
    for obj in objects:
        rows.append({'id':obj['id'],'label':obj.get('kind','text'),'bbox':list(obj['bbox']),
                     'text':obj.get('text',''),'ocr_ids':obj.get('ocr_line_ids',[])})
    rows.sort(key=lambda r:(r['bbox'][1],r['bbox'][0],r['id']))
    img=Image.open(source).convert('RGB')
    draw=ImageDraw.Draw(img)
    def can_contain(p):
        if p['label']!='icon':
            return p['label']!='text'
        x,y,w,h=p['bbox']
        return w>=60 and h>=22 and w>=1.4*h and any(
            t.get('ocr_line_ids') and t['bbox'][2]*t['bbox'][3]<=.55*w*h
            and overlap(p['bbox'],t['bbox'])>=.95*t['bbox'][2]*t['bbox'][3] for t in objects)
    containers=[r for r in rows if can_contain(r)]
    for n,row in enumerate(rows,1):
        x,y,w,h=row['bbox']
        row['mark']=n
        parents=[p for p in containers if p['id']!=row['id'] and
                 p['bbox'][2]*p['bbox'][3]>w*h*1.1 and overlap(row['bbox'],p['bbox'])>=w*h*.95]
        row['parent']=min(parents,key=lambda p:p['bbox'][2]*p['bbox'][3])['id'] if parents else None
        row['intersecting_ocr_ids']=sorted({lid for t in objects if overlap(t['bbox'],row['bbox'])>0 for lid in t.get('ocr_line_ids',[])})
        color='#b51977' if row.get('ocr_ids') else '#00659d'
        draw.rectangle([x,y,x+w,y+h],outline=color)
        label=str(n)
        rect=draw.textbbox((x,y),label)
        draw.rectangle(rect,fill=color)
        draw.text((x,y),label,fill='white')
    img.save(directory/'marks.png')
    write(directory/'marks.json',{'source_sha256':digest(source),'rows':rows})
    return rows


def mark_blocks(source, rows, directory):
    """Partition spatially without renumbering, dropping, or duplicating marks."""
    with Image.open(source) as im:
        original=im.convert('RGB')
    def partition(part):
        if len(part)<=30:
            yield part
            return
        centers=np.array([[r['bbox'][k]+.5*r['bbox'][k+2] for k in (0,1)] for r in part])
        axis=int(np.argmax(np.ptp(centers,axis=0)))
        ordered=[part[i] for i in np.argsort(centers[:,axis],kind='stable')]
        mid=len(ordered)//2
        yield from partition(ordered[:mid])
        yield from partition(ordered[mid:])
    for n,part in enumerate(partition(rows)):
        if not part:continue
        boxes=[r['bbox'] for r in part]
        l,t=[max(0,int(min(b[k] for b in boxes))-4) for k in (0,1)]
        r,b=[min(original.size[k],int(np.ceil(max(b[k]+b[k+2] for b in boxes)))+5) for k in (0,1)]
        tile=original.crop((l,t,r,b))
        pen=ImageDraw.Draw(tile)
        for row in part:
            x,y,w,h=row['bbox']
            pos=(x-l,y-t)
            pen.rectangle([*pos,x+w-l,y+h-t],outline='#00659d')
            label=str(row['mark'])
            pen.rectangle(pen.textbbox(pos,label),fill='#00659d')
            pen.text(pos,label,fill='white')
        pic=directory/f'marks-{n}.png'
        info=directory/f'marks-{n}.json'
        tile.save(pic)
        write(info,[{k:row[k] for k in ('id','mark','label','bbox','parent','text') if k in row} for row in part])
        yield {'bbox':[l,t,r-l,b-t],'count':len(part)},pic,info

