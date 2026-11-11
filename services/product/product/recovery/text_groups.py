"""Join source-backed wrapped body lines before ownership is adjudicated."""
import re

from ..parallel_contracts import TextEvidence


def coalesce(branches, ocr, store):
    evidence=branches['text'].model_copy(deep=True)
    styles={s.id:s for s in evidence.styles}
    lines={r['id']:r for r in ocr.get('lines',[])}
    original={t.id:t for t in evidence.texts}
    merged=[];removed=set();changes=[]
    def body(text):
        role=styles[text.style].role.lower()
        ids=evidence.members.get(text.id,[])
        return (not text.spans and '\n' not in text.text and text.align=='left'
                and re.search(r'body|paragraph|description|正文|段落',role)
                and len(ids)==1 and ids[0] in lines
                and lines[ids[0]]['text'].replace(' ','')==text.text.replace(' ',''))
    def container(a,b):
        left=min(a.bbox[0],b.bbox[0]);top=min(a.bbox[1],b.bbox[1])
        right=max(a.bbox[0]+a.bbox[2],b.bbox[0]+b.bbox[2])
        bottom=max(a.bbox[1]+a.bbox[3],b.bbox[1]+b.bbox[3])
        for shape in branches['surface'].shapes:
            x,y,w,h=shape.bbox
            if (w*h<evidence.image_size[0]*evidence.image_size[1]*.2
                    and x<=left and y<=top and x+w>=right and y+h>=bottom):
                return shape.id
        return None
    for first in sorted(evidence.texts,key=lambda t:(t.bbox[1],t.bbox[0])):
        if first.id in removed:continue
        group=[first]
        if body(first):
            for other in sorted(evidence.texts,key=lambda t:t.bbox[1]):
                last=group[-1];size=styles[first.style].size_px
                pitch=other.bbox[1]-last.bbox[1]
                if (other.id==first.id or other.id in removed or len(group)>=4
                        or other.style!=first.style or not body(other)
                        or abs(first.bbox[0]-other.bbox[0])>max(4,size*.25)
                        or not .8*size<=pitch<=1.8*size):continue
                backing=container(first,other)
                if not backing:continue
                group.append(other);removed.add(other.id)
        if len(group)>1:
            x=min(t.bbox[0] for t in group);y=min(t.bbox[1] for t in group)
            right=max(t.bbox[0]+t.bbox[2] for t in group);bottom=max(t.bbox[1]+t.bbox[3] for t in group)
            members=[m for t in group for m in evidence.members[t.id]]
            if len(set(members))!=len(members):
                raise ValueError('wrapped lines cannot share ambiguous OCR ownership')
            payload=first.model_copy(update={'text':'\n'.join(t.text for t in group),'bbox':[x,y,right-x,bottom-y],
                                            'line_spacing_px':group[1].bbox[1]-first.bbox[1]})
            evidence.members[first.id]=members
            changes.append({'retained_id':first.id,'source_texts':[t.model_dump() for t in group],
                            'ocr_members':members,'container':container(first,group[-1]),'result':payload.model_dump()})
            merged.append(payload)
        else:merged.append(first)
    if not changes:return branches
    evidence.texts=sorted(merged,key=lambda t:t.z)
    evidence.members={k:v for k,v in evidence.members.items() if k not in removed}
    # Only explicit line separators may be introduced. Every source token and
    # emphasis-bearing object remains bound to its untouched branch response.
    for change in changes:
        if ''.join(original[t['id']].text for t in change['source_texts']) != change['result']['text'].replace('\n',''):
            raise ValueError('wrapped line grouping changed source wording')
    evidence.references.append(store.evidence({'schema':'WrappedBodyGroups/1','source_sha256':evidence.source_sha256,
                                               'changes':changes,'wording_changed':False}))
    result=dict(branches);result['text']=TextEvidence.model_validate(evidence.model_dump())
    return result
