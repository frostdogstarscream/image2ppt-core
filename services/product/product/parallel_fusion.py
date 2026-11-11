"""First barrier: explicit ownership and relationships, never rectangle erasure."""
import numpy as np
from PIL import Image
from .parallel_contracts import SceneGraph,Relation
from .contracts import intersection
from .object_extract import rectangle


def fuse(snap,branches,ocr):
    text,asset,surface=(branches[k] for k in ('text','asset','surface'))
    ownership={n.id:'RESOLVED' for n in [*text.texts,*asset.objects,*surface.shapes]};relations=[]
    lines={n['id']:n for n in ocr.get('lines',[])}
    users={i:sorted([t for t in text.texts if i in text.members.get(t.id,[])],key=lambda t:(t.bbox[0],t.bbox[1])) for i in lines}
    styles={s.id for s in text.styles};w,h=snap['image_size']
    for n in [*text.texts,*asset.objects,*surface.shapes]:
        x,y,bw,bh=n.bbox
        if min(x,y)<0 or min(bw,bh)<=0 or x+bw>w or y+bh>h:ownership[n.id]='UNOWNED'
    for a in asset.objects:
        for p in [*a.parts,*a.exclusions]:
            x,y,bw,bh=p.bbox
            if min(x,y)<0 or min(bw,bh)<=0 or x+bw>w or y+bh>h:ownership[a.id]='UNOWNED'
        if any('/' in name or '\\' in name or not name.endswith('.png') for name in a.layers):ownership[a.id]='UNOWNED'
    for t in text.texts:
        ids=text.members.get(t.id,[])
        # OCR contradictions stay source raster. Spaces may differ in grouping.
        norm=lambda s:s.replace(' ','').replace('\n','')
        if not ids or any(i not in lines or lines[i]['score']<.85 for i in ids):ownership[t.id]='AMBIGUOUS'
        elif len(ids)==1 and len(users[ids[0]])>1:
            # One OCR line may contain a separate sequence number and heading.
            # Joint exact coverage is required; overlapping repeated text fails.
            if norm(''.join(x.text for x in users[ids[0]]))!=norm(lines[ids[0]]['text']):ownership[t.id]='AMBIGUOUS'
        elif norm(''.join(lines[i]['text'] for i in ids))!=norm(t.text):ownership[t.id]='AMBIGUOUS'
        if t.style not in styles or any(s.style not in styles for s in t.spans):ownership[t.id]='UNOWNED'
        if t.spans and ''.join(s.text for s in t.spans)!=t.text:ownership[t.id]='UNOWNED'
        for a in asset.objects:
            overlap=intersection(t.bbox,a.bbox)
            if not overlap:continue
            intrinsic=a.text_policy=='intrinsic_label' and t.text.replace('\n','').replace(' ','') in a.intrinsic_text.replace(' ','')
            kind='intrinsic_text' if intrinsic else 'label_for'
            relations.append(Relation(source=t.id,target=a.id,kind=kind,status='RESOLVED' if intrinsic else 'AMBIGUOUS',evidence=['source','ocr','asset-intent']))
            if intrinsic:ownership[t.id]='UNOWNED'  # lettering remains with the raster logo
    for b in surface.shapes:
        for n in [*text.texts,*asset.objects]:
            if intersection(b.bbox,n.bbox)>=n.bbox[2]*n.bbox[3]*.99:
                relations.append(Relation(source=b.id,target=n.id,kind='contains',status='RESOLVED',evidence=['surface-geometry','source']))
    for a in asset.objects:
        description=a.description.lower()
        background_intent=a.kind in ('decoration','shadow','glow') and any(word in description for word in
            ('background','card','banner','plate','底板','底色','背景','卡片'))
        if not background_intent:continue
        for b in surface.shapes:
            if ownership[b.id]!='RESOLVED':continue
            if b.geometry not in ('rect','roundRect'):continue
            overlap=intersection(a.bbox,b.bbox);union=a.bbox[2]*a.bbox[3]+b.bbox[2]*b.bbox[3]-overlap
            if overlap/max(1,union)>.75:
                # Two independent semantic branches agree on a backing role.
                # It is not a PNG allowed to erase text or neighboring icons.
                ownership[a.id]='UNOWNED'
                relations.append(Relation(source=a.id,target=b.id,kind='part_of',status='RESOLVED',evidence=['asset-background-intent','surface-geometry','source']))
                break
        if ownership[a.id]=='UNOWNED' or len(a.parts)<2:continue
        # A grouped set of backing panels can correspond to several native
        # Surface proposals. Each declared part needs its own corroboration.
        matches=[]
        for part in a.parts:
            candidates=[]
            for b in surface.shapes:
                if ownership[b.id]!='RESOLVED' or b.geometry not in ('rect','roundRect'):continue
                overlap=intersection(part.bbox,b.bbox)
                union=part.bbox[2]*part.bbox[3]+b.bbox[2]*b.bbox[3]-overlap
                ratios=[part.bbox[i]/b.bbox[i] for i in (2,3)]
                if overlap/max(1,union)>=.6 and all(.8<=ratio<=1.25 for ratio in ratios):candidates.append((overlap/union,b))
            if not candidates:break
            matches.append(max(candidates,key=lambda pair:pair[0])[1])
        if len(matches)==len(a.parts) and len({b.id for b in matches})==len(matches):
            ownership[a.id]='UNOWNED'
            for b in matches:
                relations.append(Relation(source=a.id,target=b.id,kind='part_of',status='RESOLVED',
                    evidence=['asset-background-intent','surface-geometry','source','complete-background-parts']))
    # Connectivity and 2D overlap alone are not semantic contradictions.
    return SceneGraph.model_validate({'schema':'SceneGraph/1',**{k:snap[k] for k in ('source_sha256','image_size','transform')},
        'method_version':'r10-fusion/1','text':text.model_dump(),'asset':asset.model_dump(),'surface':surface.model_dump(),
        'ownership':ownership,'relations':[r.model_dump() for r in relations]})


def glyph_evidence(source,box,foreground,backing):
    """Cross-check source ink against color and a separately sampled backing.

    Return certain/uncertain pixels, NOT the OCR rectangle. No morphology can
    manufacture visible strokes. In ambiguous regions callers must keep source.
    """
    x,y,w,h=[round(v) for v in box];crop=source[y:y+h,x:x+w].astype(float)
    bg=backing.astype(float);fg=np.array([int(foreground[i:i+2],16) for i in (0,2,4)],float)
    direction=fg-bg;delta=crop-bg;den=(direction*direction).sum(2)
    alpha=np.clip((delta*direction).sum(2)/np.maximum(den,1),0,1)
    error=np.abs(bg+alpha[...,None]*direction-crop).max(2)
    # Low-amplitude compression/background uncertainty is retained, not erased.
    # Source screenshots contain color ringing; bounded color residual accepts
    # these only along the same foreground-to-backing color direction.
    visible=np.abs(delta).max(2)>6
    certain=visible & (error<=32) & (alpha>.005) & (den>100)
    uncertain=visible & ~certain
    return certain,uncertain
