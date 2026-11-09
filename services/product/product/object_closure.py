"""Shadow-safe complete-object assembly. Candidates are not accepted assets."""
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import cv2
from PIL import Image
from .materialize import digest,write
from .object_contracts import read_plan
from .object_extract import rectangle,visible_rgba,load_sam,load_page,recompose
from .object_ownership import text_evidence,replacement_proof
from .object_registry import Registry,pixel_key,scene_key,SAM_BINDING


def components(alpha,parts,exclusions,size):
    support=alpha>0;_,labels,stats,_=cv2.connectedComponentsWithStats(support.astype('uint8'),8)
    selected=np.zeros_like(support);ambiguous=[]
    forbidden=np.zeros_like(support)
    for part in exclusions:forbidden|=rectangle(part.bbox,size)
    for part in parts:
        roi=rectangle(part.bbox,size)
        for label in np.unique(labels[roi & support]):
            full=labels==label
            # Keep entire components. An exclusion vetoes a component; it never
            # punches holes/cuts visible strokes out of it.
            if (full & forbidden).any() or (full & ~rectangle(part.bbox,size,pad=32)).any():
                ambiguous.append(int(label));continue
            selected|=full
    return selected,sorted(set(ambiguous))


def enclosed_ownership(mask):
    """A separate hypothesis, NOT a corrected model mask or opacity fill.
    Only existing source-layer pixels can contribute. Independent QA must reject
    this hypothesis for real transparent holes and decide white-fill ownership.
    """
    padded=np.pad(mask.astype('uint8'),1)
    exterior=(1-padded).copy();cv2.floodFill(exterior,None,(0,0),2)
    holes=(exterior[1:-1,1:-1]==1)
    return mask|holes


def assemble(obj,layers,names,size,sam=None,reserved=None,allowed_layers=None):
    masks=[];ambiguities={}
    for index,rgba in enumerate(layers):
        name=names[index]
        if allowed_layers is not None and name not in allowed_layers:
            masks.append(np.zeros(rgba.shape[:2],bool));continue
        parts=[p for p in obj.parts if name in p.layers]
        exclusions=[p for p in obj.exclusions if name in p.layers]
        mask,bad=components(rgba[...,3],parts,exclusions,size)
        if bad:ambiguities[name]=bad
        if sam is not None and name in obj.layers:
            # SAM supplies only uncertain source pixels. Preserve original soft
            # Alpha, including white content. Exclusion BOXES are not masks: a
            # platform box may overlap a legitimate cloud curve. Semantic/text
            # gates must reject a bad model mask instead of cutting that curve.
            addition=sam & (rgba[...,3]>0)
            if reserved is not None:addition&=~reserved
            mask|=addition
        masks.append(mask)
    rgba=visible_rgba(layers,masks)
    missing=[p.name for p in obj.parts if not (rectangle(p.bbox,size)&(rgba[...,3]>0)).any()]
    return rgba,ambiguities,missing


def replace(ir,obj,row):
    candidate=copy.deepcopy(ir);old={n['id']:n for n in ir['nodes']};owners=set(obj.replaces)
    candidate['nodes']=[n for n in candidate['nodes'] if n['id'] not in owners]
    used={n.get('asset_id') for n in candidate['nodes']}
    removed={old[i].get('asset_id') for i in owners}-used
    oldev={e for i in owners for e in old[i].get('evidence_ids',[])}
    for a in ir['assets']:
        if a['id'] in removed:oldev.update(a.get('contains_evidence_ids',[]))
    candidate['assets']=[a for a in candidate['assets'] if a['id'] not in removed]
    candidate['evidence']=[e for e in candidate['evidence'] if e['id'] not in oldev]
    nodeid='extracted-'+obj.id;ev='ev-'+nodeid
    if any(n['id']==nodeid for n in candidate['nodes']):raise ValueError('object identity collision')
    z=min([old[i]['z'] for i in owners] or [max([n['z'] for n in old.values()] or [0])+1])
    candidate['nodes'].append({'id':nodeid,'type':'image','bbox':row['bbox'],'z':z,'asset_id':nodeid,'evidence_ids':[]})
    candidate['assets'].append({'id':nodeid,'path':row['asset'],'purpose':'module','sha256':row['asset_sha256'],
        'description':obj.description,'contains_evidence_ids':[ev]})
    candidate['evidence'].append({'id':ev,'kind':'image','bbox':row['bbox'],'description':obj.description})
    if row.get('background_assets') and not row.get('background_reuse'):
        for node in candidate['nodes']:node['z']*=2
        candidate['nodes'][-1]['z']=max(old[i]['z'] for i in owners)*2+1
        for bg in row['background_assets']:
            candidate['nodes'].append({'id':bg['id'],'type':'image','bbox':bg['bbox'],'z':bg['old_z']*2,'asset_id':bg['id'],'evidence_ids':[]})
            candidate['assets'].append({'id':bg['id'],'path':bg['file'],'purpose':'module','sha256':bg['sha256'],
                'contains_evidence_ids':[],'description':'Background restoration; independent of extracted object'})
        candidate['nodes'].sort(key=lambda n:n['z'])
        for z,node in enumerate(candidate['nodes']):node['z']=z
    candidate['nodes'].sort(key=lambda n:n['z'])
    return candidate


def extract_closed(planfile,layerdir,baseline_ir,output,*,sam_dir=None,registry=None,rejected=None,candidate_offsets=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    baseline_ir=Path(baseline_ir);planfile=Path(planfile)
    raw=json.loads(planfile.read_text(encoding='utf-8'));plan,legacy=read_plan(raw)
    page=load_page(layerdir);ir=json.loads(baseline_ir.read_text(encoding='utf-8'))
    if plan.source_sha256!=digest(page.source_path) or plan.image_size!=list(page.size):raise ValueError('object source mismatch')
    source_path=(baseline_ir.parent/ir['source_image']).resolve()
    if not source_path.is_relative_to(baseline_ir.parent.resolve()):raise ValueError('source path escape')
    source=np.array(Image.open(page.source_path).convert('RGB'))
    if not np.array_equal(source,np.array(Image.open(source_path).convert('RGB'))):raise ValueError('baseline source mismatch')
    for a in ir['assets']:
        file=(baseline_ir.parent/a['path']).resolve()
        if not file.is_relative_to(baseline_ir.parent.resolve()):raise ValueError('asset path escape')
        if a.get('sha256') and digest(file)!=a['sha256']:raise ValueError('baseline asset hash mismatch')
        dst=output/a['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dst)
    shutil.copyfile(source_path,output/ir['source_image'])
    layers=[np.array(Image.open(l.path).convert('RGBA')) for l in page.layers];names=[l.file for l in page.layers]
    layer_hashes={l.file:{'sha256':digest(l.path),'z':l.z,'role':l.role} for l in page.layers}
    composite=recompose(page).astype('int16')
    all_source_rgba=None
    hard,unknown,text_records=text_evidence(ir,source,page.size)
    write(output/'text-ownership.json',text_records)
    Image.fromarray(np.where(hard,1,np.where(unknown,4,0)).astype('uint8')).save(output/'text-ownership.png')
    original=copy.deepcopy(ir);records=[];prepared={};reserved=np.zeros_like(hard);prep_owners={}
    registry=Registry(registry) if registry else None
    # Compute good Alpha ownership for ALL objects before any SAM candidate.
    for obj in plan.objects:
        if not set(obj.layers)<=set(names) or any(not set(p.layers)<=set(names) for p in obj.exclusions):continue
        if registry:
            stable=registry.lookup(pixel_key(plan.source_sha256,layer_hashes,obj)) or registry.lookup(pixel_key(plan.source_sha256,layer_hashes,obj,SAM_BINDING))
            if stable:
                locked=np.zeros_like(layers[0]);x,y,w,h=stable['bbox']
                locked[y:y+h,x:x+w]=np.array(Image.open(registry.asset(stable)).convert('RGBA'))
                prepared[obj.id]=locked;reserved|=locked[...,3]>0;continue
        rgba,bad,missing=assemble(obj,layers,names,page.size)
        support=rgba[...,3]>0
        if support.any() and not bad and not missing and not (support&(hard|unknown)).any():
            complete,_,_=replacement_proof(ir,obj,support,output,source,page.size,prep_owners)
            if complete:prepared[obj.id]=rgba;reserved|=support
    occupied=np.zeros_like(hard);owner_cache={}
    for obj in plan.objects:
        row={'id':obj.id,'kind':obj.kind,'replaces':obj.replaces,'status':'FALLBACK','quality':'REVIEW_REQUIRED',
            'human_acceptance':False,'candidates':[],'reasons':[],'legacy_adapter':legacy}
        row['definition']=obj.model_dump()
        records.append(row)
        if obj.id in (rejected or {}):row['reasons']=list(rejected[obj.id]);continue
        if not set(obj.layers)<=set(names) or any(not set(p.layers)<=set(names) for p in obj.exclusions):row['reasons']=['wrong_layer_selected'];continue
        # Even an unselected higher layer attenuates visible source contribution.
        key=pixel_key(plan.source_sha256,layer_hashes,obj)
        sam_key=pixel_key(plan.source_sha256,layer_hashes,obj,SAM_BINDING)
        row['pixel_key']=key;row['scene_key']=scene_key(original)
        choices=[]
        cached=(registry.lookup(key) or registry.lookup(sam_key)) if registry else None
        if cached:
            if cached.get('background_assets'):
                row['background_assets']=[];row['background_reuse']=True
                for bg in cached['background_assets']:
                    n=next((n for n in ir['nodes'] if n['id']==bg['id']),None)
                    if n is None:raise ValueError('stable background missing; scene revalidation required')
                    asset=next(a for a in ir['assets'] if a['id']==n['asset_id'])
                    if asset['sha256']!=bg['sha256']:raise ValueError('stable background changed')
                    row['background_assets'].append({**bg,'file':asset['path'],'old_z':n['z'],'bbox':n['bbox']})
            stable_node='extracted-'+cached['object_id']
            if any(n['id']==stable_node for n in ir['nodes']):
                row['owner_adapter']={'from':obj.replaces,'to':[stable_node],'protocol':'StableOwner/1'}
                obj=obj.model_copy(update={'replaces':[stable_node],'required_owners':[stable_node]})
                row['replaces']=[stable_node]
            rgba=np.zeros_like(layers[0]);x,y,w,h=cached['bbox'];rgba[y:y+h,x:x+w]=np.array(Image.open(registry.asset(cached)).convert('RGBA'))
            choices.append(('stable-reuse',rgba,{},[]))
        rgba,bad,missing=assemble(obj,layers,names,page.size)
        choices.append(('alpha',rgba,bad,missing))
        def white_assembly(base):
            nonlocal all_source_rgba
            owned=base[...,3]>0;closed=enclosed_ownership(owned);holes=closed & ~owned & ~reserved
            if not holes.any():return None
            from .object_ownership import background
            z=min([n['z'] for n in ir['nodes'] if n['id'] in obj.replaces] or [10**9])
            bg,known=background(ir,page.size,z)
            white='white fill' in obj.prompt_en.lower()
            if not white:
                _,labels=cv2.connectedComponents(holes.astype('uint8'),8);qualified=np.zeros_like(holes)
                for label in range(1,int(labels.max())+1):
                    region=labels==label
                    pixels=source[region]
                    if known[region].all() and np.mean(pixels.min(1)>=245)>.6 and np.median(np.abs(pixels.astype('int16')-bg[region].astype('int16')).max(1))>=6:
                        qualified|=region
                holes=qualified
            if not holes.any():return None
            if all_source_rgba is None:all_source_rgba=visible_rgba(layers,[np.ones_like(hard) for _ in layers])
            result=base.copy();result[holes]=all_source_rgba[holes]
            filled=owned|holes
            interior=cv2.distanceTransform(filled.astype('uint8'),cv2.DIST_L2,5)>2
            interior&=~reserved
            result[interior]=all_source_rgba[interior]
            return result
        if not cached and not bad and not missing:
            good,reason,_=replacement_proof(ir,obj,rgba[...,3]>0,output,source,page.size,owner_cache)
            if reason=='replacement_contains_other_content':
                choices.append(('alpha-native-bg',rgba,bad,missing))
                choices.append(('alpha-native-owner-bg',rgba,bad,missing))
            white=white_assembly(rgba)
            if white is not None:choices.append(('alpha-source-alpha-holes-native-owner-bg',white,bad,missing))
        if not cached and len(obj.layers)>1:
            for name in obj.layers[:2]:
                rgba,bad,missing=assemble(obj,layers,names,page.size,allowed_layers={name})
                choices.append(('alpha-layer-'+name,rgba,bad,missing))
        # A locked Alpha object is never downgraded/replaced by SAM.
        if not cached and sam_dir:
            masks,error=load_sam(sam_dir,obj,plan.source_sha256,digest(planfile),page.size,'sam3',individual=True)
            row['sam_error']=error
            for value in (masks or []):
                if len(choices)>=6:break
                rgba,bad,missing=assemble(obj,layers,names,page.size,value['mask'],reserved)
                choices.append(('sam3-mask-'+str(value['index']),rgba,bad,missing))
                if len(choices)<6:choices.append(('sam3-mask-'+str(value['index'])+'-native-owner-bg',rgba,bad,missing))
                rgba_support=rgba[...,3]>0
                filled=enclosed_ownership(rgba_support)
                holes=filled & ~rgba_support & ~reserved
                if len(choices)<6 and holes.any():
                    rgba=rgba.copy()
                    # White fill can have been assigned to a lower LayerD layer
                    # rather than the icon layer. Propose that observed source
                    # contribution ONLY in enclosed, still-unowned holes. Keep
                    # the raw-hole alternative and require an explicit white-
                    # ownership verdict; this is never a silent mask repair.
                    if holes.any():
                        if all_source_rgba is None:all_source_rgba=visible_rgba(layers,[np.ones_like(hard) for _ in layers])
                        rgba[holes]=all_source_rgba[holes]
                        # When planning explicitly identifies a white fill, its
                        # backing also belongs behind INTERNAL antialias pixels.
                        # Keep the complete outer two-pixel soft edge unchanged.
                        # This remains a separate, independently reviewed claim.
                        if 'white fill' in obj.prompt_en.lower():
                            interior=cv2.distanceTransform(filled.astype('uint8'),cv2.DIST_L2,5)>2
                            interior&=~reserved
                            rgba[interior]=all_source_rgba[interior]
                    choices.append(('sam3-mask-'+str(value['index'])+'-source-alpha-holes-native-owner-bg',rgba,bad,missing))
        seen=set()
        for index,(method,rgba,bad,missing) in enumerate(choices[:6]):
            support=rgba[...,3]>0;problems=[]
            import hashlib
            pixels_hash=hashlib.sha256(rgba.tobytes()).hexdigest()
            identity=(pixels_hash,method.endswith('-native-bg'),method.endswith('-native-owner-bg'),method.startswith('sam'))
            if identity in seen:continue
            seen.add(identity)
            if not support.any():problems.append('no_visible_pixels')
            if bad and method.startswith('alpha'):problems.append('ambiguous_components')
            if missing:problems.append('missing_required_parts')
            if (support & hard).any():problems.append('protected_text_ink')
            if (support & unknown).any():problems.append('uncertain_text_ownership')
            if (support & (hard|unknown)).any():
                from .object_text import local_disputes
                _,_,text_check=local_disputes(source,ir,support & (hard|unknown),output/'text-disputes'/obj.id)
                row['text_disputes']=text_check
            if (support & occupied).any():problems.append('object_ownership_overlap')
            if method.startswith('sam') and (support & (reserved & ~(prepared.get(obj.id,np.zeros_like(rgba))[...,3]>0))).any():
                problems.append('locked_alpha_overlap')
            ok,reason,proof=replacement_proof(ir,obj,support,output,source,page.size,owner_cache,
                {b['id']:b['sha256'] for b in row.get('background_assets',[])})
            backgrounds=None
            if not ok and reason=='replacement_contains_other_content':
                from .object_background import restoration
                backgrounds=restoration(ir,obj,rgba,layers,names,output,source,hard,unknown,
                    native=method.endswith(('-native-bg','-native-owner-bg')),owner_scope=method.endswith('-native-owner-bg'))
            if not ok and not backgrounds:problems.append(reason)
            mae=float(np.abs(composite-source.astype('int16'))[support].mean()) if support.any() else None
            if mae is None or mae>10:problems.append('source_fidelity')
            stem=f'candidate-{obj.id}-{index:02d}'
            # Retain full RGBA/ownership, including failed hypotheses, privately.
            np.savez_compressed(output/(stem+'.npz'),rgba=rgba,support=support)
            record={'index':index,'method':method,'raw':stem+'.npz','raw_sha256':digest(output/(stem+'.npz')),
                'ambiguous_components':bad,'missing_parts':missing,'source_mae':mae,'reasons':problems,'owner_proof':proof}
            record['pending_background_review']=bool(backgrounds)
            row['candidates'].append(record)
            if problems or index<(candidate_offsets or {}).get(obj.id,0):continue
            ys,xs=np.nonzero(support);box=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)]
            x,y,w,h=box;file=output/('object-'+obj.id+'.png');Image.fromarray(rgba[y:y+h,x:x+w]).save(file)
            if method=='stable-reuse':shutil.copyfile(registry.asset(cached),file)
            row.update(status='STAGED',method=method,bbox=box,asset=file.name,asset_sha256=digest(file),selected_index=index)
            row['source_alpha_hole_hypothesis']=cached.get('source_alpha_hole_hypothesis',False) if method=='stable-reuse' else '-source-alpha-holes' in method
            if row['source_alpha_hole_hypothesis']:row['additional_layer_ownership']='Enclosed source pixels only; independent white-content review required'
            if backgrounds:
                row['background_assets']=[]
                for bg in backgrounds:
                    bgfile=output/(bg['id']+'.png');Image.fromarray(bg.pop('rgba')).save(bgfile)
                    row['background_assets'].append({**bg,'file':bgfile.name,'sha256':digest(bgfile)})
            if method.startswith('sam'):row['pixel_key']=sam_key
            if method=='stable-reuse' and cached:row['pixel_key']=key if registry.lookup(key) else sam_key
            ir=replace(ir,obj,row);occupied|=support;break
        if row['status']=='FALLBACK':row['reasons']=sorted({p for c in row['candidates'] for p in c['reasons']}) or ['no_candidate']
    text_fields=lambda value:[{k:v for k,v in n.items() if k!='z'} for n in value['nodes'] if n['type']=='text']
    if text_fields(ir)!=text_fields(original):raise ValueError('text changed')
    file=output/'slideir.json';write(file,ir)
    from .object_ownership import background
    bg,known=background(original,page.size,10**9)
    ownership=np.full(hard.shape,4,np.uint8)
    ownership[known & (np.abs(source.astype('int16')-bg.astype('int16')).max(2)<=1)]=0
    for mask,_ in owner_cache.values():ownership[mask]=3
    ownership[hard]=2;ownership[occupied]=1
    Image.fromarray(ownership).save(output/'pixel-ownership.png')
    write(output/'pixel-ownership.json',{'classes':{'0':'certain_background','1':'staged_icon','2':'source_native_text_evidence','3':'other_object','4':'uncertain'},
        'semantic_truth':False,'scene_sha256':digest(file),'counts':{str(i):int((ownership==i).sum()) for i in range(5)}})
    write(output/'authoring.json',{'tool':'object closure shadow stage','model_id':'versioned-object-plan','ir_sha256':digest(file),
        'decisions':['Staged only; no automatic semantic or human acceptance']})
    report={'schema':'ObjectClosure/2','source_sha256':plan.source_sha256,'plan_sha256':digest(planfile),
        'layer_manifest_sha256':digest(Path(layerdir)/'manifest.json'),'layer_bindings':layer_hashes,
        'parent_ir_sha256':digest(baseline_ir),'scene_sha256':digest(file),'objects':records,
        'plan_file':str(planfile.resolve()),'layer_directory':str(Path(layerdir).resolve()),'baseline_ir':str(baseline_ir.resolve()),'sam_directory':str(Path(sam_dir).resolve()) if sam_dir else None,
        'staged':sum(r['status']=='STAGED' for r in records),'fallback':sum(r['status']=='FALLBACK' for r in records),
        'rejected_objects':rejected or {},
        'human_acceptance':False,'automatic_quality':'REVIEW_REQUIRED'}
    write(output/'closure.json',report)
    return file,report
