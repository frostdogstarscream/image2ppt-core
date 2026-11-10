"""Copy-on-write isolated trial. No selection or generation publication here."""
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from .materialize import digest, write
from .parallel_store import key
from .parallel_contracts import SceneGraph
from .parallel_candidate_contracts import ReconstructionCandidate, PreflightResult, transition
from .parallel_rebuild import adapt, array_sha


def stage(candidate_dir, graph, baseline_dir, output):
    candidate_dir,baseline_dir,output=map(Path,(candidate_dir,baseline_dir,output))
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    candidate=ReconstructionCandidate.model_validate(load(candidate_dir/'candidate.json'))
    preflight=PreflightResult.model_validate(load(candidate_dir/'preflight.json'))
    if candidate.status!='PRECHECK_PASS' or candidate.hard_gate!='PASS' or preflight.status!='PASS' or preflight.candidate_id!=candidate.candidate_id:
        raise ValueError('candidate has not passed preflight')
    if key(graph.model_dump())!=candidate.scene_sha256 or digest(baseline_dir/'source.png')!=candidate.source_sha256:
        raise ValueError('stale candidate binding')
    if set(candidate.references)!={'pixels.npz','payload.json'}:
        raise ValueError('incomplete candidate evidence')
    for file,sha in candidate.references.items():
        if digest(candidate_dir/file)!=sha:raise ValueError('candidate evidence tampered')
    baseline=SceneGraph.model_validate(load(baseline_dir/'scene.json'))
    if baseline.source_sha256!=graph.source_sha256 or baseline.text!=graph.text:
        raise ValueError('baseline scene/content mismatch')
    with np.load(candidate_dir/'pixels.npz',allow_pickle=False) as data:
        mask=data['mask'];patch=data['patch'];box=list(map(int,data['bbox']))
    if mask.dtype!=bool or mask.shape!=tuple(reversed(graph.image_size)):
        raise ValueError('invalid owner mask')
    if any(i in baseline.selected for i in candidate.object_ids):raise ValueError('trial would replace existing stable object')
    for record in baseline.selected.values():
        proof=record.get('owner_evidence')
        if not proof:raise ValueError('baseline owner evidence missing')
        file=(baseline_dir/proof['file']).resolve()
        if not file.is_relative_to(baseline_dir.resolve()) or digest(file)!=proof['sha256']:raise ValueError('baseline owner binding mismatch')
        with np.load(file,allow_pickle=False) as data:
            if (mask & data['mask'].astype(bool)).any():raise ValueError('candidate conflicts with stable owner')
    shutil.copytree(baseline_dir,output)  # fails if destination exists
    plate=np.array(Image.open(output/'source-plate.png').convert('RGBA'))
    before=plate.copy();x,y,w,h=box;local=mask[y:y+h,x:x+w]
    plate[y:y+h,x:x+w,:3][local]=patch[local]
    if ((plate!=before).any(2)&~mask).any():raise ValueError('unauthorized plate mutation')
    Image.fromarray(plate).save(output/'source-plate.png')
    record=load(candidate_dir/'payload.json');identifier=record['id']
    if candidate.object_ids!=[identifier]:raise ValueError('region identity mismatch')
    proof=output/(identifier+'-owner.npz')
    np.savez_compressed(proof,mask=mask,restore_bbox=box,backing_patch=patch)
    record.update(state='STAGED',quality='REVIEW_REQUIRED',candidate_id=candidate.candidate_id,
        region_id=candidate.region_id,source_sha256=candidate.source_sha256,original_owner='source-plate',
        complete_owner_scope=True,human_acceptance=False,mask_sha256=array_sha(mask.astype('uint8')),
        owner_evidence={'file':proof.name,'sha256':digest(proof),'method':'source-derived-four-sided-backing-and-ink/1'})
    baseline.selected[identifier]=record;baseline.transactions.append(record)
    # Layout proposals are private SceneGraph changes; the existing adapter is
    # still the only writer of public SlideIR.
    for text in baseline.text.texts:
        if text.id==identifier:
            if record['payload']['text']!=text.text:raise ValueError('candidate rewrote content')
            text.bbox=record['payload']['bbox']
    baseline.raster_backing['sha256']=digest(output/'source-plate.png')
    write(output/'scene.json',baseline.model_dump())
    reports=[r for r in load(output/'transactions.json') if r['id']!=identifier]+[record]
    write(output/'transactions.json',reports)
    write(output/'slideir.json',adapt(baseline,output))
    # Old build bindings and derived ownership maps cannot describe this trial.
    with np.load(output/'editable-mask.npz') as data:editable=data['mask'].copy()
    np.savez_compressed(output/'editable-mask.npz',mask=editable|mask)
    with np.load(output/'ownership.npz') as data:maps={k:data[k].copy() for k in data.files}
    maps['ownership'][mask]=1;np.savez_compressed(output/'ownership.npz',**maps)
    ownership=load(output/'ownership.json');ownership['map_sha256']=digest(output/'ownership.npz');write(output/'ownership.json',ownership)
    from .parallel_regions import regions,finish_regions
    write(output/'region-transactions.json',finish_regions(regions(baseline)[0],reports))
    write(output/'build-binding.json',{'candidate_id':candidate.candidate_id,'parent_scene_sha256':digest(baseline_dir/'scene.json'),
        'scene_sha256':digest(output/'scene.json'),'ir_sha256':digest(output/'slideir.json'),'trial_only':True})
    write(output/'candidate-stage.json',transition(candidate,'STAGED').model_dump(by_alias=True))
    return output/'slideir.json'
