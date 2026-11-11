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


def protected_for_asset(graph, baseline_directory, object_id=None):
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
    if object_id is not None:
        for peer in graph.asset.objects:
            if peer.id!=object_id:
                protected|=rectangle(peer.bbox,tuple(graph.image_size))
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


def retry_asset_extraction(graph,object_id,source_file,layer_directory,baseline_directory,output):
    """Return one private candidate package, or retain explicit blocking evidence."""
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
    protected=protected_for_asset(graph,baseline,object_id)
    surface=resolved_surface(graph)
    layers=load_page(layer_directory)
    seeds=asset_candidates(obj,source,layers,protected,(),'auto',surface=surface)
    for index,(seed,metadata) in enumerate(seeds):
        folder=output/('candidate-'+str(index));folder.mkdir()
        attempt={'seed_method':metadata['method'],'seed_rgba_sha256':array_sha(seed)}
        try:
            rgba,segmentation=complete_source_halo(obj,seed,source,protected,surface)
            write(folder/'segmentation.json',segmentation)
            restore_box,backing,mask,proof=source_owner(obj,rgba,source,protected,surface)
            bx,by,bw,bh=bounds(rgba[...,3]>0)
            x,y,width,height=restore_box
            patch=source[y:y+height,x:x+width].copy()
            patch[mask[y:y+height,x:x+width]]=backing[mask[y:y+height,x:x+width]]
            crop=rgba[by:by+bh,bx:bx+bw]
            Image.fromarray(crop).save(folder/'asset.png')
            Image.fromarray(source).save(folder/'source.png')
            Image.fromarray(source[y:y+height,x:x+width]).save(folder/'source_crop.png')
            Image.fromarray(patch).save(folder/'written_crop.png')
            np.savez_compressed(folder/'pixels.npz',mask=mask,patch=patch,bbox=restore_box,
                                rgba=crop,rgba_bbox=[bx,by,bw,bh],seed_rgba=seed,protected=protected)
            payload={'id':object_id,'kind':'asset','bbox':[bx,by,bw,bh],'restore_bbox':restore_box,
                'payload':{**metadata,**proof,'method':'source-halo-completion','segmentation':segmentation},
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
                representation='TRANSPARENT_PNG',strategy='source-halo-completion',references=references,
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
