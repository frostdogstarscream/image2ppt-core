"""Local-only disputed text evidence. No OCR over an entire page, no downloads."""
import json
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .materialize import digest,write
from .object_registry import fingerprint


@lru_cache(maxsize=1)
def installed_fonts():
    from .fonts import font_manifest
    return font_manifest([])


def native_glyph(node,size):
    font=node.get('font',{});family=font.get('ea')
    if not family or not font.get('size_pt'):return None
    face=next((f for f in installed_fonts()['fonts'] if f['family']==family and f['bold']==node.get('bold',False)),None)
    if face is None:return None
    image=Image.new('L',size);draw=ImageDraw.Draw(image);x,y,w,h=node['bbox']
    actual=ImageFont.truetype(face['file'],max(1,round(font['size_pt']/.75)),index=face['index'])
    for index,line in enumerate(node.get('text','').split('\n')):
        width=draw.textlength(line,font=actual);align=node.get('align','left')
        ox=x+(w-width)/2 if align=='center' else x+w-width if align=='right' else x
        spacing=node.get('paragraphs',[{}])[0].get('line_spacing_pt',font['size_pt']*1.2)/.75
        draw.text((ox,y+index*spacing),line,font=actual,fill=255,anchor='lt')
    return cv2.dilate(np.array(image),np.ones((5,5),np.uint8))>0


def local_disputes(source,ir,support,directory,*,recognizer=None,fonts=None):
    source=np.asarray(source);h,w=source.shape[:2];directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    hard=np.zeros((h,w),bool);unknown=np.zeros((h,w),bool);records=[]
    if recognizer is None:
        try:
            from image2ppt_vision.models import recognize,verified_models
            models=verified_models();recognizer=recognize
            model_key=fingerprint(models)
        except (OSError,ValueError,ImportError,KeyError):recognizer=None;model_key='UNAVAILABLE'
    else:model_key='injected-local-recognizer'
    styles=fonts or (installed_fonts() if recognizer else {'fonts':[]})
    for node in ir['nodes']:
        if node['type']!='text':continue
        x,y,bw,bh=[round(v) for v in node['bbox']];x=max(0,x);y=max(0,y);x1=min(w,x+bw);y1=min(h,y+bh)
        if not support[y:y1,x:x1].any():continue
        if x1-x>=w*.9 and y1-y>=h*.9:raise ValueError('whole-page OCR forbidden')
        rgb=source[y:y1,x:x1];key=fingerprint({'pixels':__import__('hashlib').sha256(rgb.tobytes()).hexdigest(),'models':model_key})
        cache=directory/(key+'.json')
        if cache.exists():rows=json.loads(cache.read_text(encoding='utf-8'))
        elif recognizer:
            rows=recognizer(rgb);write(cache,rows)
        else:rows=[]
        evidence=np.zeros((y1-y,x1-x),np.uint8)
        expected=''.join(node.get('text','').split())
        for row in rows:
            text=''.join(row.get('text','').split())
            if row.get('score',0)<.9 or not text or (text not in expected and expected not in text):continue
            poly=np.rint(row['polygon']).astype('int32');cv2.fillPoly(evidence,[poly],1)
        # Native glyph rendering corroborates OCR polygons; it never invents
        # source ink or replaces text/font decisions.
        font=node.get('font',{});family=font.get('ea');bold=node.get('bold',False)
        face=next((f for f in styles['fonts'] if f['family']==family and f['bold']==bold),None)
        glyph=np.zeros_like(evidence,dtype=bool)
        if face:
            image=Image.fromarray(glyph);draw=ImageDraw.Draw(image)
            font_file=ImageFont.truetype(face['file'],max(1,round(font['size_pt']/0.75)),index=face['index'])
            draw.multiline_text((0,0),node.get('text',''),font=font_file,fill=255,spacing=0)
            glyph=cv2.dilate(np.array(image),np.ones((5,5),np.uint8))>0
        colors=[node.get('color')]+[r.get('color') for p in node.get('paragraphs',[]) for r in p.get('runs',[])]
        ink=np.zeros_like(evidence,dtype=bool)
        for c in colors:
            if isinstance(c,str) and len(c)==6:
                colour=np.array(tuple(bytes.fromhex(c)),dtype='int16')
                ink|=(np.abs(rgb.astype('int16')-colour).max(2)<12)
        verified=(evidence>0)&glyph&ink
        hard[y:y1,x:x1]|=verified
        unknown[y:y1,x:x1]|=support[y:y1,x:x1]&~verified
        records.append({'id':node['id'],'ocr_cache':key,'ocr_available':recognizer is not None,'native_font_available':face is not None,
            'corroborated_pixels':int(verified.sum()),'human_acceptance':False})
    write(directory/'disputes.json',records)
    return hard,unknown,records
