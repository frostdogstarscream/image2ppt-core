"""Bounded dependency-directed repairs, with logical identities preserved."""
from pathlib import Path
import numpy as np
from PIL import Image
from .parallel_branches import perceive,local_ocr,layer_evidence
from .parallel_fusion import fuse
from .parallel_typography import calibrate
from .parallel_text_anchor import anchor
from .parallel_rebuild import build
from .parallel_store import key
from .object_extract import load_page
from .resource_lease import lease


def local_findings(graph,report,stable):
    """Technical/semantic failures are actionable even when raster fallback is faithful."""
    result=[];anchors={a['id']:a for a in stable}
    for obj in graph.asset.objects:
        old=anchors.get(obj.id)
        if not old:continue
        bx,by,bw,bh=old['bbox'];x,y,w,h=obj.bbox
        previous=old['definition']
        changed_policy=previous.get('text_policy','exclude')!=obj.text_policy
        if w*h>bw*bh*1.6 or changed_policy:
            result.append({'category':'graphics','severity':'major','node_ids':[obj.id],
                'description':'Stable object grain/boundary conflicts with the new proposal. Check the source; retain the persistent symbol ID. '+
                    'Do not absorb an independent label, plate, platform or card into it. Existing boundary evidence: '+str({k:previous.get(k) for k in ('description','text_policy','intrinsic_text')}),
                'suggested_fix':'Use source and layer evidence to correct parts/exclusions; report ambiguity explicitly.'})
    kinds={n.id:'text' for n in graph.text.texts}|{n.id:'asset' for n in graph.asset.objects}
    for r in report:
        if r['state']!='FALLBACK' or r['id'] not in kinds:continue
        reason=r.get('reason','')
        if reason=='source raster retained':continue
        result.append({'category':'graphics' if kinds[r['id']]=='asset' else 'font','severity':'major',
            'node_ids':[r['id']],'description':'Local reconstruction could not safely replace this object: '+reason,
            'suggested_fix':'Recheck only affected source geometry, role, layers and exclusions; do not invent pixels or text. Keep safe raster if evidence remains insufficient.'})
    return result


def route(graph,findings):
    ids={n.id:b for b,items in [('text',graph.text.texts),('asset',graph.asset.objects),('surface',graph.surface.shapes)] for n in items}
    targets=set()
    for f in findings:
        named={ids[i] for i in f.get('node_ids',[]) if i in ids}
        if named:targets|=named
        elif f['category'] in ('wording','font','emphasis'):targets.add('text')
        elif f['category']=='graphics':targets|={'asset','surface'}
        else:targets|={'text','asset','surface'}
    return targets


def repair(graph,snap,findings,store,provider,fonts,cache,models,directory,baseline,strategy):
    targets=route(graph,findings)
    if not targets:raise ValueError('no actionable repair evidence')
    existing={'text':graph.text,'asset':graph.asset,'surface':graph.surface};feedback={}
    for branch in targets:
        previous=existing[branch].model_dump()
        # File locations are receipts, not visual planning input. Leaving them
        # in feedback makes a resumed no-op look like a new paid request.
        previous['references']=[]
        store.consume(snap['source_sha256']+':'+branch,'planning-repair',key({'findings':findings,'previous_evidence':previous}),2)
        feedback[branch]={'findings':findings,'previous_evidence':previous,
            'identity_rule':'Preserve all previous logical IDs; do not rename existing objects. Fix only the affected evidence.'}
    saved=next((p['layers'] for p in baseline['pages'] if p['source']['sha256']==snap['source_sha256']),None)
    stable=[a for a in baseline['stable_assets'] if a['source_sha256']==snap['source_sha256']]
    branches,failures=perceive(snap,store,provider,fonts,cache,models,saved_layers=saved,existing=existing,targets=targets,repair=feedback,stable=stable)
    if failures:raise ValueError('targeted repair branch failed')
    for name in targets:
        collection='texts' if name=='text' else 'objects' if name=='asset' else 'shapes'
        old={n.id for n in getattr(existing[name],collection)};new={n.id for n in getattr(branches[name],collection)}
        if not old<=new:raise ValueError('repair lost persistent identities')
    ocr=local_ocr(snap,cache,models);new=fuse(snap,branches,ocr)
    pixels=np.array(Image.open(snap['path']).convert('RGB'))
    with lease('cpu'):
        new=anchor(new,ocr,pixels,fonts,store);new=calibrate(new,pixels,fonts,store,ocr=ocr,stable=stable);new=anchor(new,ocr,pixels,fonts,store,fitted=True)
    try:layers=load_page(layer_evidence(snap,cache,saved)[0])
    except Exception:layers=None
    with lease('cpu'):ir,report=build(new,snap,fonts,Path(directory),layers=layers,stable=stable,strategy=strategy,pixel_cache=Path(cache)/'pixels')
    return new,ir,report
