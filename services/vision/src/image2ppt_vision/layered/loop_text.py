"""Ordered regional content checks and final-file bindings for v3.1."""
import difflib
import re
import unicodedata

from ..storage import digest
from .loop_review import intersect


def normalized(text):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',text))


def numbers(text):
    return re.findall(r'[+\-−]?\d+(?:[.,]\d+)*(?:%|‰)?',normalized(text))


def compare_text(expected, observed):
    a,b=normalized(expected),normalized(observed)
    opcodes=difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes()
    changes=[{'kind':tag,'expected':a[i:j],'observed':b[k:l],'expected_range':[i,j],'observed_range':[k,l]}
             for tag,i,j,k,l in opcodes if tag!='equal']
    return {'exact':a==b,'numbers_exact':numbers(a)==numbers(b),'expected_numbers':numbers(a),
            'observed_numbers':numbers(b),'changes':changes,
            'reordered':a!=b and sorted(a)==sorted(b)}


def regional_text(perception, rendered, *, render_size, retained_ids=(), annotations=()):
    W,H=perception['image_size'];rw,rh=render_size
    expected={t['id']:t for t in perception['ocr'] if t['id'] not in retained_ids}
    assignments={tid:[] for tid in expected};unmatched=[]
    retained=[t['bbox'] for t in perception['ocr'] if t['id'] in retained_ids]
    for hit in rendered:
        x,y,w,h=hit['bbox'];box=[x*W/rw,y*H/rh,w*W/rw,h*H/rh]
        matches=[]
        for tid,line in expected.items():
            bx,by,bw,bh=line['bbox']
            expanded=[bx-bh*.2,by-bh*.3,bw+bh*.4,bh*1.6]
            overlap=intersect(box,expanded)/max(box[2]*box[3],1)
            if overlap>=.35:matches.append((overlap,tid))
        if matches:
            assignments[max(matches)[1]].append({**hit,'bbox':box})
        elif not any(intersect(box,b)/max(box[2]*box[3],1)>=.4 for b in retained):
            unmatched.append({'text':hit['text'],'bbox':box,'score':hit.get('score')})
    rows=[]
    for tid,line in expected.items():
        hits=sorted(assignments[tid],key=lambda h:(round(h['bbox'][1]/max(h['bbox'][3],1)),h['bbox'][0]))
        found=''.join(h['text'] for h in hits)
        rows.append({'id':tid,'bbox':line['bbox'],'expected':line['text'],'observed':found,
                     'reference':'source_ocr_not_ground_truth',**compare_text(line['text'],found)})
    labelled=[]
    for reference in annotations:
        if reference.get('kind','text')!='text':continue
        hits=[h for h in rendered if intersect([h['bbox'][0]*W/rw,h['bbox'][1]*H/rh,h['bbox'][2]*W/rw,h['bbox'][3]*H/rh],reference['bbox'])>0]
        hits.sort(key=lambda h:(h['bbox'][1],h['bbox'][0]))
        labelled.append({'id':reference['id'],'expected':reference['text'],'observed':''.join(h['text'] for h in hits),
                         'reference':'source_visual_annotation',**compare_text(reference['text'],''.join(h['text'] for h in hits))})
    return {'schema':'LayeredRegionalText/1','regions':rows,'unexpected_text':unmatched,'labelled_regions':labelled,
            'mismatches':[r for r in rows if not r['exact']], 'numeric_mismatches':[r for r in rows if not r['numbers_exact']],
            'source_ground_truth_complete':False}
