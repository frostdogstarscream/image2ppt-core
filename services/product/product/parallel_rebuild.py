"""Pure candidates and the second barrier. One SceneGraph -> SlideIR adapter."""
import copy
import hashlib
import json
import re
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
from .materialize import digest,write,font_gate
from .object_extract import rectangle,load_page
from .object_closure import assemble
from .parallel_fusion import glyph_evidence
from .parallel_store import key
from .resource_lease import lease

ROOT=Path(__file__).resolve().parents[3]
CODE_SHA=key({str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),
    *[Path(__file__).with_name(n+'.py') for n in ('parallel_fusion','parallel_typography','parallel_font_match','parallel_text_anchor','parallel_pixels','object_closure','parallel_regions','parallel_region_gate')],
    ROOT/'ppt-rebuild/src/reconstruction.py']})
sys.path.insert(0,str(ROOT/'ppt-rebuild/src'))
from reconstruction import background_model,extract_rgba

def array_sha(value):return hashlib.sha256(value.tobytes()).hexdigest()


def bounds(mask):
    yy,xx=np.nonzero(mask)
    if len(xx)==0:raise ValueError('empty candidate')
    return [int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]


class OwnerConflict(ValueError):
    def __init__(self,component,protected,box,label,source,rgba):
        super().__init__('source component belongs to protected peer')
        self.component=component.copy();self.conflict=(component&protected).copy();self.box=box
        self.report={'schema':'OwnerConflictReport/1','component_id':int(label),'bbox':box,
            'component_mask_sha256':array_sha(component),'protected_mask_sha256':array_sha(protected),
            'candidate_rgba_sha256':array_sha(rgba),'source_pixels_sha256':array_sha(source),
            'overlap_pixels':int(self.conflict.sum()),'reason_code':'PROTECTED_PEER_COMPONENT',
            'status':'BLOCKED','semantic_attribution':'UNKNOWN'}


def source_owner(obj,rgba,source,protected,surface):
    """Prove source-ink closure independently of the proposed PNG's Alpha.

    The source components are retained whole, never clipped to the PNG. This is
    the distinction between a good-looking matte and removing all old ink.
    """
    size=(source.shape[1],source.shape[0]);scope=rectangle(obj.bbox,size)
    for p in obj.parts:scope|=rectangle(p.bbox,size)
    scope|=rgba[...,3]>0
    box=bounds(scope);window=rectangle(box,size,pad=8);x,y,w,h=bounds(window)
    backing,error,evidence=backing_from_evidence(source,[x,y,w,h],protected,surface)
    if not np.isfinite(error) or error>6:raise ValueError('source owner backing uncertain')
    crop=source[y:y+h,x:x+w];observed=np.abs(crop.astype(float)-backing).max(2)>max(6,error*2)
    support=(rgba[y:y+h,x:x+w,3]>0);seeds=observed&support
    _,labels,_,_=cv2.connectedComponentsWithStats(observed.astype('uint8'),8)
    owned=np.zeros((h,w),bool)
    for label in np.unique(labels[seeds]):
        if not label:continue
        component=labels==label
        if (component&protected[y:y+h,x:x+w]).any():
            raise OwnerConflict(component,protected[y:y+h,x:x+w],[x,y,w,h],label,source,rgba)
        sides=[(x>0,component[:,0]),(x+w<size[0],component[:,-1]),(y>0,component[0]),(y+h<size[1],component[-1])]
        if any(active and pixels.any() for active,pixels in sides):raise ValueError('source owner component escapes localization')
        owned|=component
    independent=observed&scope[y:y+h,x:x+w]&~protected[y:y+h,x:x+w]&~owned
    if independent.any():raise ValueError('source ink has no complete candidate owner')
    strong_missing=owned&~support&(np.abs(crop.astype(float)-backing).max(2)>40)
    if strong_missing.any():raise ValueError('candidate is missing independently observed strong strokes')
    if not owned.any():raise ValueError('no independently separated source ink')
    full=np.zeros(source.shape[:2],bool);full[y:y+h,x:x+w]=owned
    return [x,y,w,h],backing,full,{'source_owner_method':'source-components+semantic-parts+protected-peers/1',
        'source_owner_mask_sha256':array_sha(full),'png_visible_mask_sha256':array_sha(rgba[...,3]>0),
        'background_evidence':evidence,'source_background_error':error,'unowned_source_ink_pixels':int(independent.sum())}


def text_candidate(t,graph,source,fonts,blocked):
    styles={s.id:s for s in graph.text.styles};style=styles[t.style]
    # Reuse only the font validator, never the serial page contract/planner.
    from types import SimpleNamespace
    used={t.style,*[s.style for s in t.spans]}
    font_gate(SimpleNamespace(styles=[styles[i] for i in used],texts=[t]),fonts)
    box=[round(v) for v in t.bbox];x,y,w,h=box
    backing,error=background_model(source,box,blocked)
    if not np.isfinite(error) or error>4:raise ValueError('text backing uncertain')
    ink=np.zeros((h,w),bool);unknown=np.ones((h,w),bool)
    colors={style.color,*[styles[s.style].color for s in t.spans]}
    for color in colors:
        certain,uncertain=glyph_evidence(source,box,color,backing);ink|=certain;unknown&=uncertain
    # Unknown visible pixels are not background. Preserve whole text transaction.
    if unknown.any() or ink.sum()<2:raise ValueError('text source glyph ownership uncertain')
    full=np.zeros(source.shape[:2],bool);full[y:y+h,x:x+w]=ink
    if (full & blocked).any():raise ValueError('text intersects protected graphic ink')
    patch=source[y:y+h,x:x+w].copy();patch[ink]=backing[ink]
    return {'id':t.id,'kind':'text','bbox':box,'mask':full,'patch':patch,
            'payload':t.model_dump(),'background_error':error,'method':'source-color-glyph+local-background/1'}


from .object_unmix import candidate_from_evidence as matte_from_evidence,backing_from_evidence,CODE_SHA as UNMIX_CODE_SHA


def asset_candidates(obj,source,layers,protected,stable,strategy,sam=None,cache=None,surface=None):
    lock=key({'source':array_sha(source),'bbox':obj.bbox,'layers':obj.layers,
              'parts':[p.bbox for p in obj.parts],'exclusions':[p.bbox for p in obj.exclusions]})
    with lease('pixel-'+lock,capacity=1):return _asset_candidates(obj,source,layers,protected,stable,strategy,sam,cache,surface)


def _asset_candidates(obj,source,layers,protected,stable,strategy,sam=None,cache=None,surface=None):
    size=(source.shape[1],source.shape[0]);results=[]
    # A stable source-bound PNG wins without resampling or a new segmentation.
    for r in stable:
        bx=rectangle(r['bbox'],size);ob=rectangle(obj.bbox,size)
        if obj.id!=r['id'] and ((bx&ob).sum()/max(1,bx.sum())<.9 or (bx&ob).sum()/max(1,ob.sum())<.5):continue
        if obj.id==r['id'] and (bx&ob).sum()/max(1,bx.sum())<.85:raise ValueError('stable identity boundary conflict requires adjudication')
        if digest(r['path'])!=r['sha256']:raise ValueError('stable asset hash mismatch')
        png=np.array(Image.open(r['path']).convert('RGBA'));x,y,w,h=[round(v) for v in r['bbox']]
        if png.shape!=(h,w,4):raise ValueError('stable asset dimensions mismatch')
        full=np.zeros((*source.shape[:2],4),np.uint8);full[y:y+h,x:x+w]=png
        if (protected & (full[...,3]>0)).any():continue
        results.append((full,{'method':'stable-reuse','original_file':r['path'],'original_sha256':r['sha256'],
                              'stable_id':r['id'],'stable_bbox':r['bbox']}));return results
    # Pixel inputs exclude names, descriptions, page numbers and semantic IDs.
    identity=key({'source_pixels':array_sha(source),'bbox':obj.bbox,'layers':obj.layers,
        'parts':[{'bbox':p.bbox,'layers':p.layers} for p in obj.parts],
        'exclusions':[{'bbox':p.bbox,'layers':p.layers} for p in obj.exclusions],
        'layer_hashes':[l.sha256 for l in layers.layers] if layers else [],
        'protection':array_sha(protected),'strategy':strategy,
        'sam_masks':[array_sha(c['mask']) for c in sam or []],
        'surfaces':[{k:v for k,v in s.model_dump().items() if k!='id'} for s in surface.shapes] if surface is not None else [],
        'implementation':[CODE_SHA,UNMIX_CODE_SHA]})
    cache=Path(cache)/identity if cache else None
    if cache and (cache/'manifest.json').exists():
        rows=json.loads((cache/'manifest.json').read_text(encoding='utf-8'));restored=[]
        for r in rows:
            if digest(cache/r['file'])!=r['sha256']:raise ValueError('pixel cache hash mismatch')
            with np.load(cache/r['file'],allow_pickle=False) as value:rgba=value['rgba']
            restored.append((rgba,r['metadata']))
        return restored
    prior=None
    if layers:
        arrays=[np.array(Image.open(l.path).convert('RGBA')) for l in layers.layers];names=[l.file for l in layers.layers]
        rgba,ambiguous,missing=assemble(obj,arrays,names,size)
        if not missing and not ambiguous and (rgba[...,3]>0).any() and not ((rgba[...,3]>0)&protected).any():
            results.append((rgba,{'method':'alpha','ambiguous':ambiguous,'missing':missing}));prior=rgba
    if strategy not in ('alpha','off'):
        try:results.append(matte_from_evidence(obj,source,protected,original_alpha=prior,surface=surface))
        except ValueError:pass
    if sam and layers:
        # Each model mask remains a separate assembly hypothesis. Good stable
        # assets above return early, so SAM cannot replace their bytes.
        for candidate in sam:
            rgba,ambiguous,missing=assemble(obj,arrays,names,size,sam=candidate['mask'],reserved=protected)
            if not missing and not ((rgba[...,3]>0)&protected).any():
                results.append((rgba,{'method':'sam3','mask_index':candidate['index'],'mask_score':candidate['score'],'ambiguous':ambiguous}))
    results=results[:6]
    if cache:
        cache.mkdir(parents=True,exist_ok=True);rows=[]
        for index,(rgba,metadata) in enumerate(results):
            file=cache/f'{index:02d}.npz';np.savez_compressed(file,rgba=rgba)
            rows.append({'file':file.name,'sha256':digest(file),'metadata':metadata,'quality':'UNKNOWN'})
        write(cache/'manifest.json',rows)
    return results


def build(graph,snap,fonts,directory,*,layers=None,stable=(),strategy='auto',sam_candidates=None,pixel_cache=None):
    # A legacy build is itself an isolated trial, never a mutation of barrier 1.
    graph=graph.model_copy(deep=True)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    source=np.array(Image.open(snap['path']).convert('RGB'));size=tuple(graph.image_size);plate=source.copy()
    resolved_surface=graph.surface.model_copy(deep=True)
    resolved_surface.shapes=[s for s in resolved_surface.shapes if graph.ownership[s.id]=='RESOLVED']
    chosen=[];report=[];claimed=np.zeros(source.shape[:2],bool)
    from .parallel_pixels import graphic_ink
    blocked=graphic_ink(stable,size,graph.source_sha256)
    plate_alpha=np.full(source.shape[:2],255,np.uint8)
    # Asset search rectangles guard text reconstruction; true glyph masks guard
    # asset pixels. Empty text-box pixels remain available to graphic candidates.
    from .parallel_regions import regions,retain_atomic,finish_regions
    from .parallel_region_gate import region_ready
    region_records,region_by_id=regions(graph)
    text_masks={}
    for t in graph.text.texts:
        if graph.ownership[t.id]!='RESOLVED':
            report.append({'id':t.id,'state':'FALLBACK','reason':'text ownership is not resolved'});continue
        try:
            c=text_candidate(t,graph,source,fonts,blocked);chosen.append(c);text_masks[t.id]=c['mask'];claimed|=c['mask']
        except (ValueError,KeyError) as exc:
            report.append({'id':t.id,'state':'FALLBACK','reason':str(exc)})
    # Shared typography remains calibrated, but is not pixel ownership.
    # Only explicit indivisible text relationships can cascade a rollback.
    text_ids={t.id for t in graph.text.texts}
    chosen,rejected=retain_atomic(chosen,[r for r in region_records if set(r['member_ids'])<=text_ids])
    for identifier,code in rejected.items():
        report.append({'id':identifier,'state':'FALLBACK','reason':'indivisible text region member failed','reason_code':code})
    claimed[:]=False
    for c in chosen:claimed|=c['mask']
    text_protection=np.zeros_like(claimed);uncertain_text=np.zeros_like(claimed)
    ocr=None
    for ref in graph.text.references:
        file=Path(ref).resolve()
        if file.parent.name!='evidence' or not re.fullmatch(r'[a-f0-9]{64}',file.stem):continue
        if not file.is_relative_to(ROOT) or not file.is_file():continue
        record=json.loads(file.read_text(encoding='utf-8'))
        if record.get('schema')=='OCRLines/1' and record.get('source_sha256')==graph.source_sha256:
            if key(record)!=file.stem:raise ValueError('OCR evidence content hash mismatch')
            ocr=record;break
    if ocr:
        for line in ocr['lines']:
            if any(a.text_policy=='intrinsic_label' and line['text'].replace(' ','') in a.intrinsic_text.replace(' ','')
                   and rectangle(a.bbox,size)[round(line['bbox'][1]),round(line['bbox'][0])] for a in graph.asset.objects):continue
            poly=np.zeros_like(claimed,dtype='uint8');cv2.fillPoly(poly,[np.rint(line['polygon']).astype('int32')],1)
            try:
                x,y,w,h=[round(v) for v in line['bbox']];bg,error=background_model(source,[x,y,w,h],blocked)
                ink=np.abs(source[y:y+h,x:x+w].astype(float)-bg).max(2)>max(6,error*3)
                local=poly[y:y+h,x:x+w]>0
                target=text_protection if line['score']>=.85 and error<=4 else uncertain_text
                target[y:y+h,x:x+w]|=ink&local
            except ValueError:uncertain_text|=poly>0
    else:
        # No OCR cross-evidence: conservatively keep source, never claim the box
        # itself as a confirmed glyph mask.
        for t in graph.text.texts:uncertain_text|=rectangle(t.bbox,size)
    protected=text_protection|uncertain_text
    locked={}
    for r in stable:
        if digest(r['path'])!=r['sha256']:continue
        x,y,w,h=[round(v) for v in r['bbox']];png=np.array(Image.open(r['path']).convert('RGBA'))
        if png.shape!=(h,w,4):continue
        mask=np.zeros_like(claimed);mask[y:y+h,x:x+w]=png[...,3]>0;locked[r['id']]=mask
    for obj in graph.asset.objects:
        if strategy=='off' or graph.ownership[obj.id]!='RESOLVED':
            report.append({'id':obj.id,'state':'FALLBACK','reason':'source raster retained'});continue
        reserved=protected.copy()
        for identifier,mask in locked.items():
            if identifier!=obj.id:reserved|=mask
        try:candidates=asset_candidates(obj,source,layers,reserved,stable,strategy,(sam_candidates or {}).get(obj.id),pixel_cache,resolved_surface)
        except (ValueError,OSError) as exc:
            report.append({'id':obj.id,'state':'FALLBACK','reason':str(exc)});continue
        accepted=False;candidate_failures=[]
        def alternatives():
            yield from candidates
            if any(meta['method']=='stable-reuse' for _,meta in candidates):
                # Only reached if the source/ownership checks rejected the old
                # PNG. Retain its bytes/history and try equivalent candidates;
                # a successfully verified stable object never enters this path.
                for rgba,meta in asset_candidates(obj,source,layers,reserved,(),strategy,
                        (sam_candidates or {}).get(obj.id),pixel_cache,resolved_surface)[:max(0,6-len(candidates))]:
                    yield rgba,{**meta,'equivalent_of':[m['original_sha256'] for _,m in candidates if m['method']=='stable-reuse']}
        for index,(rgba,meta) in enumerate(alternatives()):
            mask=rgba[...,3]>0
            if not mask.any() or (mask & claimed).any():continue
            box=meta.get('stable_bbox',bounds(mask));x,y,w,h=[round(v) for v in box]
            try:
                backing,error,background_evidence=backing_from_evidence(source,box,reserved,resolved_surface)
                if not np.isfinite(error) or error>6:continue
                local=rgba[y:y+h,x:x+w];alpha=local[...,3:4]/255
                reconstructed=local[...,:3]*alpha+backing*(1-alpha)
                # This only rejects poor candidates; it cannot prove semantics.
                err=np.abs(reconstructed-source[y:y+h,x:x+w]).max(2)
                source_p95=float(np.percentile(err[mask[y:y+h,x:x+w]],95))
                if source_p95>24 and meta['method']!='stable-reuse':continue
                meta={**meta,'source_recomposition_p95':source_p95,'background_evidence':background_evidence,
                      'source_difference_requires_review':source_p95>24}
                restore_box,restore_backing,owner_mask,proof=source_owner(obj,rgba,source,reserved,resolved_surface)
                rx,ry,rw,rh=restore_box;patch=source[ry:ry+rh,rx:rx+rw].copy();patch[owner_mask[ry:ry+rh,rx:rx+rw]]=restore_backing[owner_mask[ry:ry+rh,rx:rx+rw]]
                chosen.append({'id':obj.id,'kind':'asset','bbox':box,'restore_bbox':restore_box,'mask':owner_mask,'patch':patch,'rgba':local,'payload':{**meta,**proof},
                               'candidate_index':index,'background_error':error})
                claimed|=owner_mask;accepted=True;break
            except ValueError as exc:candidate_failures.append({'candidate_index':index,'method':meta.get('method'),'reason':str(exc)})
        if not accepted:report.append({'id':obj.id,'state':'FALLBACK','reason':'no safe candidate; full source pixels retained','candidate_failures':candidate_failures})
    foreground_ids={t.id for t in graph.text.texts}|{a.id for a in graph.asset.objects}
    chosen,rejected=retain_atomic(chosen,[r for r in region_records if set(r['member_ids'])<=foreground_ids])
    for identifier,code in rejected.items():
        report.append({'id':identifier,'state':'FALLBACK','reason':'indivisible region member failed','reason_code':code})
    claimed[:]=False
    for c in chosen:claimed|=c['mask']
    # No branch mutates the parent: all source changes happen at this barrier.
    for c in chosen:
        x,y,w,h=c.get('restore_bbox',c['bbox']);mask=c['mask'][y:y+h,x:x+w];plate[y:y+h,x:x+w][mask]=c['patch'][mask]
        record={k:v for k,v in c.items() if k not in ('mask','patch','rgba')}
        record.update(state='STAGED',quality='REVIEW_REQUIRED',human_acceptance=False,source_sha256=graph.source_sha256,
            original_owner='source-plate',complete_owner_scope=True,mask_sha256=array_sha(c['mask'].astype('uint8')))
        proof_file=directory/(c['id']+'-owner.npz')
        np.savez_compressed(proof_file,mask=c['mask'],restore_bbox=c.get('restore_bbox',c['bbox']),backing_patch=c['patch'])
        record['owner_evidence']={'file':proof_file.name,'sha256':digest(proof_file),'method':'source-derived-backing-and-ink/1'}
        graph.transactions.append(record);graph.selected[c['id']]=record
        report.append(record)
    # Native surfaces sit behind a soft source-backed plate. Only independently
    # color-confirmed flat backing pixels become transparent; unknown lettering,
    # neighboring graphics and edges retain their original raster contribution.
    surface_claimed=np.zeros_like(claimed)
    accepted_ids={c['id'] for c in chosen}
    intrinsic_ids={r.source for r in graph.relations if r.kind=='intrinsic_text' and r.target in accepted_ids and r.status=='RESOLVED'}
    for s in graph.surface.shapes:
        if not region_ready(region_by_id[s.id],region_records,accepted_ids|intrinsic_ids):
            report.append({'id':s.id,'state':'FALLBACK','reason':'local surface foreground closure is incomplete','reason_code':'SURFACE_REGION_UNPROVEN'});continue
        if graph.ownership[s.id]!='RESOLVED' or s.fill=='none' or s.fill_gradient or s.geometry!='rect' or s.line_width_pt:
            report.append({'id':s.id,'state':'FALLBACK','reason':'complex surface remains source raster'});continue
        area=rectangle(s.bbox,size);color=np.array([int(s.fill[i:i+2],16) for i in (0,2,4)])
        # MoMA supplies geometry/intent; actual source samples determine fill.
        values,counts=np.unique(source[area],axis=0,return_counts=True)
        dominant=values[counts.argmax()]
        if counts.max()/max(1,area.sum())>=.7 and np.abs(dominant.astype(float)-color).max()<=16:
            color=dominant;s.fill=''.join(f'{int(c):02X}' for c in color)
        matching=area & (np.abs(source.astype(float)-color).max(2)<=1)
        if (area&~matching&~claimed).any() or matching.sum()<area.sum()*.7 or (area&surface_claimed).any():
            report.append({'id':s.id,'state':'FALLBACK','reason':'surface geometry/color or overlap uncertain'});continue
        parents=graph.surface.model_copy(deep=True);parents.shapes=[b for b in parents.shapes if b.id!=s.id]
        x,y,w,h=[round(v) for v in s.bbox]
        try:
            parent,error,evidence=backing_from_evidence(source,[x,y,w,h],claimed,parents)
            if error>4:raise ValueError('parent backing uncertain')
        except ValueError:
            report.append({'id':s.id,'state':'FALLBACK','reason':'surface parent cannot be safely restored'});continue
        # Native backing is above the source plate and below its reconstructed
        # foreground. It remains visible when moved away from its old location.
        plate[y:y+h,x:x+w]=parent;claimed|=area;surface_claimed|=area
        c={'id':s.id,'kind':'surface','bbox':s.bbox,'payload':s.model_dump(),'method':'source-color-confirmed-flat/1'}
        chosen.append(c);record={**c,'state':'STAGED','quality':'REVIEW_REQUIRED','human_acceptance':False,
            'mask_sha256':array_sha(area.astype('uint8'))}
        proof_file=directory/(s.id+'-owner.npz')
        np.savez_compressed(proof_file,mask=area,restore_bbox=[x,y,w,h],backing_patch=parent)
        record['owner_evidence']={'file':proof_file.name,'sha256':digest(proof_file),'method':'source-sampled-parent-backing/1'}
        graph.transactions.append(record);graph.selected[s.id]=record;report.append(record)
    outside=np.any(plate!=source,2)&~claimed
    if outside.any():raise ValueError('transaction changed unauthorized source pixels')
    Image.fromarray(np.dstack([plate,plate_alpha])).save(directory/'source-plate.png')
    graph.raster_backing={'path':'source-plate.png','sha256':digest(directory/'source-plate.png'),
        'source_sha256':graph.source_sha256,'bbox':[0,0,*graph.image_size]}
    for c in chosen:
        if c['kind']!='asset':continue
        file=directory/(c['id']+'.png');meta=c['payload']
        if meta['method']=='stable-reuse':file.write_bytes(Path(meta['original_file']).read_bytes())
        else:Image.fromarray(c['rgba']).save(file)
        graph.selected[c['id']]['asset_reference']={'path':file.name,'sha256':digest(file),
            'source_sha256':graph.source_sha256,'bbox':c['bbox'],'crop_offset':c['bbox'][:2]}
    ownership=np.full(source.shape[:2],4,np.uint8)  # uncertain
    ownership[surface_claimed]=2
    for a in stable:
        if digest(a['path'])!=a['sha256']:continue
        x,y,w,h=[round(v) for v in a['bbox']];old=np.array(Image.open(a['path']).convert('RGBA'))
        ownership[y:y+h,x:x+w][old[...,3]>0]=3  # another known graphic's contribution
    for c in chosen:
        if c['kind']=='asset':ownership[c['mask']]=0
    ownership[text_protection]=1
    for c in chosen:
        if c['kind']=='text':ownership[c['mask']]=1
    np.savez_compressed(directory/'ownership.npz',ownership=ownership,uncertain_text=uncertain_text)
    write(directory/'ownership.json',{'schema':'PixelOwnership/1','source_sha256':graph.source_sha256,
        'map_sha256':digest(directory/'ownership.npz'),'classes':{'0':'graphic','1':'text','2':'background','3':'other_object','4':'uncertain'},
        'uncertain_is_not_background':True})
    shutil_source=Path(snap['path']);(directory/'source.png').write_bytes(shutil_source.read_bytes())
    ir=adapt(graph,directory)
    for r in report:r['region_id']=region_by_id[r['id']]['region_id']
    region_records=finish_regions(region_records,report)
    write(directory/'region-transactions.json',region_records)
    np.savez_compressed(directory/'editable-mask.npz',mask=claimed)
    write(directory/'scene.json',graph.model_dump());write(directory/'transactions.json',report)
    write(directory/'slideir.json',ir)
    write(directory/'build-binding.json',{'rebuild_sha256':CODE_SHA,'unmix_sha256':UNMIX_CODE_SHA,
        'scene_sha256':digest(directory/'scene.json'),'ir_sha256':digest(directory/'slideir.json')})
    return directory/'slideir.json',report


def adapt(graph,directory):
    """The sole r10 SceneGraph-to-existing-SlideIR conversion."""
    directory=Path(directory).resolve();chosen=list(graph.selected.values())
    if digest(directory/'source.png')!=graph.source_sha256:raise ValueError('SceneGraph source snapshot mismatch')
    def reference(record):
        file=(directory/record['path']).resolve()
        if not file.is_relative_to(directory) or record['source_sha256']!=graph.source_sha256 or digest(file)!=record['sha256']:
            raise ValueError('SceneGraph asset reference binding mismatch')
        with Image.open(file) as image:
            if list(image.size)!=[round(v) for v in record['bbox'][2:]]:raise ValueError('SceneGraph asset dimensions mismatch')
        return file
    if graph.raster_backing is None:raise ValueError('SceneGraph source backing missing')
    backing=reference(graph.raster_backing)
    ir={'schema_version':'0.1','source_image':'source.png','source_sha256':graph.source_sha256,
        'canvas':{'source_px':graph.image_size,'target_px':graph.image_size},'background':{'color':graph.surface.background},
        'review_status':'needs_review','review_notes':['r10 parallel branches; human acceptance pending'],
        'nodes':[],'assets':[],'evidence':[],'groups':[]}
    def image(id,bbox,file,description,z):
        ev='ev-'+id;ir['evidence'].append({'id':ev,'kind':'image','bbox':bbox,'description':description})
        ir['assets'].append({'id':id,'path':file.name,'purpose':'module','sha256':digest(file),'contains_evidence_ids':[ev],'description':description})
        ir['nodes'].append({'id':id,'type':'image','bbox':bbox,'z':z,'asset_id':id,'evidence_ids':[]})
    surfaces=[c for c in chosen if c['kind']=='surface']
    image('source-plate',[0,0,*graph.image_size],backing,'Safe source backing with only committed replacement ink cleared',0)
    for z,c in enumerate(surfaces,1):
        s=next(s for s in graph.surface.shapes if s.id==c['id']);ev='ev-'+s.id
        ir['evidence'].append({'id':ev,'kind':'shape','bbox':s.bbox,'description':'source-confirmed flat backing'})
        ir['nodes'].append({'id':s.id,'type':'shape','geometry':s.geometry,'bbox':s.bbox,'z':z,'fill':s.fill,
                            'line':{'color':s.line_color,'width_pt':s.line_width_pt},'evidence_ids':[ev]})
    styles={s.id:s for s in graph.text.styles}
    def attrs(s):return {'font':{'ea':s.ea,'latin':s.latin,'size_pt':s.size_px*.75},'bold':s.bold,'color':s.color}
    for z,c in enumerate([c for c in chosen if c['kind']!='surface'],len(surfaces)+1):
        if c['kind']=='asset':
            file=reference(c['asset_reference']);meta=c['payload']
            image(c['id'],c['bbox'],file,'Complete transparent graphic candidate; '+meta['method'],z)
        elif c['kind']=='text':
            t=next(t for t in graph.text.texts if t.id==c['id']);s=styles[t.style];ev='ev-'+t.id
            ir['evidence'].append({'id':ev,'kind':'text','bbox':t.bbox,'text':t.text})
            node={'id':t.id,'type':'text','bbox':t.bbox,'z':z,'evidence_ids':[ev],'text':t.text,**attrs(s),
                  'align':t.align,'wrap':False,'insets':[0,0,0,0],'vertical_anchor':'top','paragraphs':[]}
            lines=[[]]
            for content,style in ([(x.text,styles[x.style]) for x in t.spans] or [(t.text,s)]):
                for i,part in enumerate(content.split('\n')):
                    if i:lines.append([])
                    if part:lines[-1].append({'text':part,**attrs(style)})
            for runs in lines:
                p={'text':''.join(r['text'] for r in runs),**attrs(s),'align':t.align,'runs':runs}
                if t.line_spacing_px:p['line_spacing_pt']=t.line_spacing_px*.75
                node['paragraphs'].append(p)
            ir['nodes'].append(node)
    return ir
