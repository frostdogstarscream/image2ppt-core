"""Source-evidenced Alpha completion for a single failed icon transaction."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..materialize import digest, write
from ..object_extract import load_page, rectangle
from ..object_unmix import backing_from_evidence
from ..parallel_candidate_contracts import ReconstructionCandidate, PreflightResult, transition
from ..parallel_candidate_engine import CHECKS
from ..parallel_rebuild import asset_candidates, array_sha, bounds, source_owner
from ..parallel_regions import regions
from ..parallel_store import key


def protected_for_asset(graph, baseline_directory, object_id=None, *, background_context=False, evidence=None):
    """Reconstruct conservative peer protection from hash-bound host evidence."""
    baseline=Path(baseline_directory).resolve()
    if digest(baseline/'source.png')!=graph.source_sha256:
        raise ValueError('baseline source binding mismatch')
    ownership=json.loads((baseline/'ownership.json').read_text('utf8'))
    if digest(baseline/'ownership.npz')!=ownership['map_sha256']:
        raise ValueError('baseline ownership map changed')
    with np.load(baseline/'ownership.npz',allow_pickle=False) as data:
        protected=(data['ownership']==1)|data['uncertain_text']
    if protected.shape!=tuple(reversed(graph.image_size)):
        raise ValueError('invalid baseline ownership dimensions')
    baseline_scene=json.loads((baseline/'scene.json').read_text('utf8'))
    if baseline_scene['source_sha256']!=graph.source_sha256:
        raise ValueError('baseline scene source changed')
    for record in baseline_scene['selected'].values():
        proof=record['owner_evidence']
        file=(baseline/proof['file']).resolve()
        if not file.is_relative_to(baseline) or digest(file)!=proof['sha256']:
            raise ValueError('stable owner proof changed')
        with np.load(file,allow_pickle=False) as data:
            mask=data['mask']
        if mask.dtype!=bool or mask.shape!=protected.shape:
            raise ValueError('invalid stable owner mask')
        protected|=mask
    for text in graph.text.texts:
        protected|=rectangle(text.bbox,tuple(graph.image_size))
    classified={};source=None;surfaces={}
    if background_context:
        from ..peer_evidence import roles
        classified=roles(graph);source=np.array(Image.open(baseline/'source.png').convert('RGB'))
        surfaces={s.id:s for s in graph.surface.shapes}
    target=next((a for a in graph.asset.objects if a.id==object_id),None)
    if object_id is not None:
        for peer in graph.asset.objects:
            if peer.id==object_id:continue
            verified=False
            if background_context and classified[peer.id]['role']=='BACKGROUND_PANEL_CANDIDATE':
                from ..peer_evidence import panel_evidence
                from ..contracts import intersection
                for surface_id in classified[peer.id]['surface_ids']:
                    surface=surfaces[surface_id]
                    if not intersection(surface.bbox,target.bbox):continue
                    backing,proof=panel_evidence(peer,surface,graph,source)
                    if backing is not None:
                        verified=True
                        if evidence is not None:evidence.append({'object_id':peer.id,'surface_id':surface_id,**proof,
                            'authority':'read-only backing context; PNG and source owner independently rechecked'})
                        break
            if not verified:protected|=rectangle(peer.bbox,tuple(graph.image_size))
    return protected


def resolved_surface(graph):
    surface=graph.surface.model_copy(deep=True)
    surface.shapes=[shape for shape in surface.shapes if graph.ownership[shape.id]=='RESOLVED']
    return surface


def complete_source_halo(obj, seed_rgba, source, protected, surface):
    """Add bounded, measured source components, keeping the seed bytes intact.

    Only small low-contrast source components inside semantic parts and within
    four pixels of existing Alpha can supply new segmentation evidence. Strong
    strokes, large components, independent neighbors and unknown context remain
    blocked by the original source_owner gate.
    """
    if (source.dtype!=np.uint8 or seed_rgba.dtype!=np.uint8
            or seed_rgba.shape!=(*source.shape[:2],4)
            or protected.dtype!=bool or protected.shape!=source.shape[:2]):
        raise ValueError('invalid source-halo evidence arrays')
    if obj.kind!='icon' or obj.text_policy!='exclude_independent_text':
        raise ValueError('bounded source-halo evidence requires an independent icon')
    size=source.shape[1::-1]
    semantic=rectangle(obj.bbox,size)
    for part in obj.parts:
        semantic|=rectangle(part.bbox,size)
    support=seed_rgba[...,3]>0
    if not support.any() or (support&protected).any():
        raise ValueError('seed Alpha is empty or conflicts with a peer')
    window=rectangle(bounds(semantic|support),size,pad=8)
    x,y,width,height=bounds(window)
    if width*height>256*256:
        raise ValueError('source-halo evidence scope is too large')
    backing,error,background=backing_from_evidence(source,[x,y,width,height],protected,surface)
    if not np.isfinite(error) or error>6:
        raise ValueError('source-halo context is unproven')
    crop=source[y:y+height,x:x+width]
    signal=np.abs(crop.astype(float)-backing).max(2)
    observed=signal>max(6,error*2)
    local_support=support[y:y+height,x:x+width]
    _,labels,_,_=cv2.connectedComponentsWithStats(observed.astype('uint8'),8)
    seeded=np.unique(labels[observed&local_support])
    owned=np.isin(labels,seeded)&(labels>0)
    local_semantic=semantic[y:y+height,x:x+width]
    local_protected=protected[y:y+height,x:x+width]
    missing=observed&~owned
    distance=cv2.distanceTransform((~local_support).astype('uint8'),cv2.DIST_L2,5)
    count,components,stats,_=cv2.connectedComponentsWithStats(missing.astype('uint8'),8)
    additions=np.zeros_like(missing)
    rows=[]
    for component in range(1,count):
        mask=components==component
        if not (mask&local_semantic).any():continue
        pixels=int(stats[component,cv2.CC_STAT_AREA])
        peak=float(signal[mask].max())
        extent=float(distance[mask].max())
        # Each entire source component gets a decision. No clipped component or
        # bounding rectangle may authorize source deletion.
        accepted=(pixels<=32 and peak<=12 and extent<=4
                  and not (mask&local_protected).any() and not (mask&~local_semantic).any())
        rows.append({'pixels':pixels,'max_source_contrast':peak,'max_alpha_distance':extent,
                     'bbox':[int(stats[component,0]+x),int(stats[component,1]+y),
                             int(stats[component,2]),int(stats[component,3])],
                     'accepted':bool(accepted)})
        if accepted:additions|=mask
    if additions.sum()>64:
        raise ValueError('source-halo additions exceed the bounded evidence budget')
    from reconstruction import extract_rgba
    supplemental=extract_rgba(crop,backing,additions)
    result=seed_rgba.copy()
    result[y:y+height,x:x+width][additions]=supplemental[additions]
    report={'schema':'SourceHaloSegmentation/1','source_pixels_sha256':array_sha(source),
        'seed_rgba_sha256':array_sha(seed_rgba),'protected_sha256':array_sha(protected),
        'result_rgba_sha256':array_sha(result),'scope_bbox':[x,y,width,height],
        'background_error':float(error),'background_evidence':background,'components':rows,
        'added_pixels':int(additions.sum()),'source_owner_gate_relaxed':False,
        'implementation_sha256':digest(__file__)}
    return result,report


def sam_extent_unmix(obj, source, protected, surface, candidate):
    """Use whole source components to propose an Unmix scope, never an erase mask."""
    from ..object_unmix import candidate_from_evidence
    if len(obj.parts) != 1:
        raise ValueError('independent multi-part scope needs separate semantic evidence')
    size = source.shape[1::-1]
    anchor = rectangle(obj.bbox, size) | candidate['mask']
    failures = []
    for padding in (8, 24, 48):
        window = rectangle(bounds(anchor), size, pad=padding)
        x, y, width, height = bounds(window)
        if width > 256 or height > 256:
            failures.append('bounded component search exceeded 256 pixels')
            break
        backing, error, evidence = backing_from_evidence(source, [x, y, width, height], protected, surface)
        if not np.isfinite(error) or error > 6:
            failures.append('source component background uncertain')
            continue
        # Use the existing strong-stroke contrast for localization only. The
        # unchanged Unmix and source_owner still recover/check soft edges.
        observed = np.abs(source[y:y+height, x:x+width].astype(float)-backing).max(2) > 40
        _, labels, _, _ = cv2.connectedComponentsWithStats(observed.astype('uint8'), 8)
        local_protected = protected[y:y+height, x:x+width]
        seed = candidate['mask'][y:y+height, x:x+width] & observed
        selected = np.zeros_like(observed)
        escaped = False
        for label in np.unique(labels[seed]):
            if not label:continue
            component = labels == label
            if (component & local_protected).any():
                if (component & ~local_protected).any():
                    raise ValueError('SAM source component crosses protected peer')
                continue
            if component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any():
                escaped = True
                break
            selected |= component
        if escaped or not selected.any():
            failures.append('source component escapes search window or no independent component')
            continue
        local_box = bounds(selected)
        box = [local_box[0]+x, local_box[1]+y, local_box[2], local_box[3]]
        # Only this estimator's search hypothesis changes. The original object,
        # all protected pixels and source_owner's semantic checks stay intact.
        scoped = obj.model_copy(update={'bbox': box,
            'parts': [obj.parts[0].model_copy(update={'bbox': box})]})
        try:
            rgba, metadata = candidate_from_evidence(scoped, source, protected, surface=surface)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        metadata['source_extent_evidence'] = {'padding': padding, 'search_bbox': [x, y, width, height],
            'source_component_bbox': box, 'source_component_pixels': int(selected.sum()),
            'sam_mask_sha256': array_sha(candidate['mask']), 'background': evidence,
            'search_signal_threshold': 40,
            'scope_only': True, 'source_owner_gate_relaxed': False, 'failed_scales': failures}
        return rgba, metadata
    raise ValueError('source/SAM extent unresolved: '+'; '.join(failures))


def sam_source_candidates(obj,source,protected,surface,masks,source_seeds):
    """SAM selects entire source-derived components, retaining their soft Alpha."""
    source_seed=next(((rgba,meta) for rgba,meta in source_seeds if meta['method']=='source-unmix'),None)
    results=[]
    failures=[]
    for candidate in masks:
        selected_seed=source_seed
        if selected_seed is None:
            # An ambiguous LayerD assembly cannot supply the Unmix extent.
            # A real SAM-supported assembly can supply that prior to the SAME
            # source estimator. It still grants no writeback authority.
            prior=next((rgba for rgba,meta in source_seeds
                        if meta['method']=='sam3' and meta['mask_index']==candidate['index']),None)
            from ..object_unmix import candidate_from_evidence
            try:
                if prior is not None:
                    selected_seed=candidate_from_evidence(obj,source,protected,original_alpha=prior,surface=surface)
            except ValueError as exc:
                failures.append('mask '+str(candidate['index'])+': '+str(exc))
            if selected_seed is None:
                try:selected_seed=sam_extent_unmix(obj,source,protected,surface,candidate)
                except ValueError as exc:
                    failures.append('mask '+str(candidate['index'])+': '+str(exc))
                    continue
        rgba,metadata=selected_seed
        support=rgba[...,3]>0
        _,labels,_,_=cv2.connectedComponentsWithStats(support.astype('uint8'),8)
        touched=np.unique(labels[support&candidate['mask']]);touched=touched[touched>0]
        keep=support if metadata.get('source_extent_evidence') else np.isin(labels,touched)&support
        if not keep.any():continue
        selected=rgba.copy();selected[~keep]=0
        results.append((selected,{'method':'sam-source-unmix','mask_index':candidate['index'],
            'mask_score':candidate['score'],'source_unmix':metadata,
            'component_selection':'whole source components intersecting real SAM evidence',
            'selected_source_pixels':int(keep.sum())}))
    if not results:raise ValueError('SAM has no independently derived complete source matte; '+'; '.join(failures))
    return results


def localized_owner_object(obj, metadata):
    """Recomputed source scope corrects localization without changing scene content."""
    extent=metadata.get('source_unmix',{}).get('source_extent_evidence')
    if not extent:return obj
    box=extent['source_component_bbox']
    return obj.model_copy(update={'bbox':box,'parts':[obj.parts[0].model_copy(update={'bbox':box})]})


def asset_background(source, box, mask, protected, layers, fitted):
    """Reuse the existing lower-background adapter after source ownership is proved."""
    from .layerd_adapter import lower_backgrounds
    from .background_adapter import generate_background_candidates
    x,y,width,height=box
    reports=[]
    for lower in lower_backgrounds(source,layers,mask):
        candidates,issues=generate_background_candidates(source,mask,protected,
            lower_rgb=lower.image,lower_provenance_verified=True)
        reports.append({'provenance':lower.provenance,'issues':issues})
        if candidates:
            return candidates[0].image[y:y+height,x:x+width].copy(),{
                'method':'layerd_lower','metrics':candidates[0].diagnostics,'lower_hypotheses':reports},lower.image[y:y+height,x:x+width]
    patch=source[y:y+height,x:x+width].copy()
    local=mask[y:y+height,x:x+width]
    patch[local]=fitted[local]
    return patch,{'method':'source_verified_continuous_fit','lower_hypotheses':reports,
                  'big_lama_calls':0},fitted


def retry_asset_extraction(graph,object_id,source_file,layer_directory,baseline_directory,output,*,strategy='source-halo',sam_directory=None,sam_plan=None):
    """Return one private candidate package, or retain explicit blocking evidence."""
    if strategy not in ('source-halo','verified-backing','sam'):raise ValueError('unsupported asset repair strategy')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    baseline=Path(baseline_directory).resolve();source_file=Path(source_file).resolve()
    if digest(source_file)!=graph.source_sha256:
        raise ValueError('asset retry source binding mismatch')
    original_scene=key(graph.model_dump())
    write(output/'scene.json',graph.model_dump())
    result={'status':'BLOCKED','object_id':object_id,'source_sha256':graph.source_sha256,
            'scene_file':str(output/'scene.json'),'baseline_directory':str(baseline),
            'attempts':[],'published':False,'office':'NOT_RUN','go':'NOT_RUN'}
    baseline_scene=json.loads((baseline/'scene.json').read_text('utf8'))
    if object_id in baseline_scene['selected']:
        result.update(status='STABLE_NOT_RETRIED',reason='verified or staged object retained')
        write(output/'result.json',result)
        return result
    obj=next(item for item in graph.asset.objects if item.id==object_id)
    region=regions(graph)[1][object_id]
    if graph.ownership[object_id]!='RESOLVED' or region['member_ids']!=[object_id]:
        result['reason']='source ownership or dependency closure is unresolved'
        write(output/'result.json',result)
        return result
    source=np.array(Image.open(source_file).convert('RGB'))
    protection=[]
    protected=protected_for_asset(graph,baseline,object_id,background_context=strategy!='source-halo',evidence=protection)
    write(output/'protection.json',{'strategy':strategy,'relations':protection,'protected_pixels':int(protected.sum()),'mask_sha256':array_sha(protected)})
    np.savez_compressed(output/'protection.npz',mask=protected)
    surface=resolved_surface(graph)
    layers=load_page(layer_directory)
    sam=None;sam_binding=None
    if strategy=='sam':
        from ..object_extract import load_sam
        if sam_directory is None or sam_plan is None:raise ValueError('SAM strategy requires bound model evidence and plan')
        sam,error=load_sam(sam_directory,obj,graph.source_sha256,digest(sam_plan),tuple(graph.image_size),'sam3',individual=True)
        if error:raise ValueError(error)
        sam_binding={'directory':str(Path(sam_directory).resolve()),'plan':str(Path(sam_plan).resolve()),
            'files':{str(p.resolve()):digest(p) for p in [Path(sam_plan),*[p for p in Path(sam_directory).iterdir() if p.is_file()]]}}
    seeds=asset_candidates(obj,source,layers,protected,(),'auto',sam=sam,surface=surface)
    if strategy=='sam':
        source_seeds=list(seeds)
        seeds=[(seed,meta) for seed,meta in seeds if meta['method']=='sam3']
        try:seeds.extend(sam_source_candidates(obj,source,protected,surface,sam,source_seeds))
        except ValueError as exc:result['sam_source_failure']=str(exc)
    for index,(seed,metadata) in enumerate(seeds):
        folder=output/('candidate-'+str(index));folder.mkdir()
        attempt={'seed_method':metadata['method'],'seed_rgba_sha256':array_sha(seed)}
        try:
            owner_obj=localized_owner_object(obj,metadata)
            rgba,segmentation=complete_source_halo(owner_obj,seed,source,protected,surface)
            write(folder/'segmentation.json',segmentation)
            restore_box,backing,mask,proof=source_owner(owner_obj,rgba,source,protected,surface)
            proof['owner_localization_bbox']=owner_obj.bbox
            bx,by,bw,bh=bounds(rgba[...,3]>0)
            x,y,width,height=restore_box
            patch,background_selection,chosen_background=asset_background(source,restore_box,mask,protected,layer_directory,backing)
            crop=rgba[by:by+bh,bx:bx+bw]
            checked_backing,error,_=backing_from_evidence(source,[bx,by,bw,bh],protected,surface)
            if background_selection['method']=='layerd_lower':
                checked_backing=chosen_background[by-y:by-y+bh,bx-x:bx-x+bw]
            alpha=crop[...,3:4]/255;visible=crop[...,3]>0
            recomposed=crop[...,:3]*alpha+checked_backing*(1-alpha)
            p95=float(np.percentile(np.abs(recomposed-source[by:by+bh,bx:bx+bw]).max(2)[visible],95))
            attempt.update(background_error=float(error),source_recomposition_p95=p95)
            if not np.isfinite(error) or error>6 or p95>24:raise ValueError('existing background/recomposition gates failed')
            Image.fromarray(crop).save(folder/'asset.png')
            Image.fromarray(source).save(folder/'source.png')
            Image.fromarray(source[y:y+height,x:x+width]).save(folder/'source_crop.png')
            Image.fromarray(patch).save(folder/'written_crop.png')
            Image.fromarray(chosen_background).save(folder/'background.png')
            Image.fromarray((mask[y:y+height,x:x+width]*255).astype('uint8')).save(folder/'mask.png')
            Image.fromarray(crop[...,3]).save(folder/'alpha.png')
            Image.fromarray(np.clip(np.rint(recomposed),0,255).astype('uint8')).save(folder/'recomposed.png')
            np.savez_compressed(folder/'pixels.npz',mask=mask,patch=patch,bbox=restore_box,
                                rgba=crop,rgba_bbox=[bx,by,bw,bh],seed_rgba=seed,protected=protected)
            payload={'id':object_id,'kind':'asset','bbox':[bx,by,bw,bh],'restore_bbox':restore_box,
                'payload':{**metadata,**proof,'method':'source-halo-completion','segmentation':segmentation},
                'protection_strategy':strategy,'protection_evidence':protection,'sam_evidence':sam_binding,
                'background_selection':background_selection,
                'source_asset_definition':obj.model_dump(),'proposed_asset_definition':obj.model_dump(),
                'baseline_directory':str(baseline),'baseline_scene_sha256':digest(baseline/'scene.json'),
                'layer_directory':str(Path(layer_directory).resolve()),
                'layer_manifest_sha256':digest(Path(layer_directory)/'manifest.json'),
                'asset_reference':{'path':'asset.png','sha256':digest(folder/'asset.png'),
                    'source_sha256':graph.source_sha256,'bbox':[bx,by,bw,bh],'crop_offset':[bx,by]},
                'owner_method':'source-components+bounded-source-halo/1'}
            write(folder/'payload.json',payload)
            references={name:digest(folder/name) for name in ('pixels.npz','payload.json')}
            identity={'source':graph.source_sha256,'scene':original_scene,'object':object_id,
                      'references':references,'implementation_sha256':digest(__file__)}
            candidate=ReconstructionCandidate(candidate_id='c-'+key(identity),source_sha256=graph.source_sha256,
                scene_sha256=original_scene,region_id=region['region_id'],object_ids=[object_id],
                representation='TRANSPARENT_PNG',strategy='source-halo-completion' if strategy=='source-halo' else strategy+'-'+metadata['method'],references=references,
                provenance={'version':'r12-05-asset-retry/1','input_binding':key(identity)},hard_gate='PASS')
            candidate=transition(candidate,'PRECHECK_PASS')
            preflight=PreflightResult(candidate_id=candidate.candidate_id,status='PASS',
                checks={name:'PASS' for name in CHECKS},unknown_pixels=0,evidence_hashes=list(references.values()))
            write(folder/'candidate.json',candidate.model_dump())
            write(folder/'preflight.json',preflight.model_dump())
            attempt.update(status='CANDIDATE_READY',added_source_halo_pixels=segmentation['added_pixels'])
            result['attempts'].append(attempt)
            result.update(status='CANDIDATE_READY',candidate_directory=str(folder),
                          candidate_id=candidate.candidate_id)
            break
        except (ValueError,OSError) as exc:
            attempt.update(status='BLOCKED',reason=str(exc))
            write(folder/'blocked.json',attempt)
            result['attempts'].append(attempt)
    if not seeds:
        result['reason']='no safe source Alpha or unmix seed; segmentation evidence is missing'
        from ..object_closure import assemble
        from ..object_unmix import candidate_from_evidence
        arrays=[np.array(Image.open(layer.path).convert('RGBA')) for layer in layers.layers]
        assembled,ambiguous,missing=assemble(obj,arrays,[layer.file for layer in layers.layers],tuple(graph.image_size))
        result['seed_diagnostics']={'alpha_ambiguities':ambiguous,'missing_parts':missing,
            'alpha_visible_pixels':int((assembled[...,3]>0).sum()),'sam':'NOT_SUPPLIED'}
        try:candidate_from_evidence(obj,source,protected,surface=surface)
        except ValueError as exc:result['seed_diagnostics']['source_unmix_failure']=str(exc)
    if key(graph.model_dump())!=original_scene or digest(source_file)!=graph.source_sha256:
        raise ValueError('asset retry modified its input')
    write(output/'result.json',result)
    write(output/'hashes.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    return result
