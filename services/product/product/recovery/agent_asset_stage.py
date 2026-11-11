"""Stage an independently checked PNG through the existing private adapter."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from ..materialize import digest, write
from ..parallel_candidate_contracts import ReconstructionCandidate, PreflightResult
from ..parallel_candidate_stage import stage
from ..parallel_contracts import SceneGraph
from ..parallel_store import key
from .agent_decision import ToolCandidate, verify_candidate


def stage_asset_candidate(candidate_directory, graph, baseline_directory, output, *, layer_directory):
    """Validate complete ownership, then copy-on-write append one independent PNG.

    Original candidate, old image bytes and parent scene remain unchanged. The
    existing SceneGraph adapter is the only producer of the resulting SlideIR.
    """
    directory=Path(candidate_directory).resolve();baseline=Path(baseline_directory).resolve()
    output=Path(output).resolve()
    if output.exists():raise ValueError('asset staging output already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    parent={str(p.relative_to(baseline)):digest(p) for p in baseline.rglob('*') if p.is_file()}
    original={str(p.relative_to(directory)):digest(p) for p in directory.rglob('*') if p.is_file()}
    with TemporaryDirectory(prefix='a-',dir=output.parent) as temporary:
        root=Path(temporary);candidate_dir=root/'c'
        shutil.copytree(directory,candidate_dir)
        candidate=ReconstructionCandidate.model_validate_json((candidate_dir/'candidate.json').read_text('utf8'))
        if len(candidate.object_ids)!=1:raise ValueError('one complete independent asset required')
        identifier=candidate.object_ids[0]
        result=verify_candidate(ToolCandidate(graph,candidate_dir,'TRANSPARENT_PNG',baseline,Path(layer_directory)),
                                graph,identifier,baseline/'source.png',root)
        if result['status']!='CANDIDATE_READY':raise ValueError('asset candidate is not ready for staging')
        payload=json.loads((candidate_dir/'payload.json').read_text('utf8'))
        basename=identifier+'.png'
        baseline_copy=root/'b';shutil.copytree(baseline,baseline_copy)
        image=candidate_dir/payload['asset_reference']['path'];target=baseline_copy/basename
        if target.exists() and digest(target)!=digest(image):
            raise ValueError('asset filename conflicts with retained baseline bytes')
        shutil.copyfile(image,target)
        payload['asset_reference']['path']=basename
        write(candidate_dir/'payload.json',payload)
        references={name:digest(candidate_dir/name) for name in ('pixels.npz','payload.json')}
        cid='c-'+key({'parent_candidate':candidate.candidate_id,'references':references,'stage_code':digest(__file__)})
        derived=candidate.model_copy(update={'candidate_id':cid,'references':references,
            'provenance':{**candidate.provenance,'original_candidate_id':candidate.candidate_id}})
        preflight=PreflightResult.model_validate_json((candidate_dir/'preflight.json').read_text('utf8'))
        preflight=preflight.model_copy(update={'candidate_id':cid,'evidence_hashes':list(references.values())})
        write(candidate_dir/'candidate.json',derived.model_dump());write(candidate_dir/'preflight.json',preflight.model_dump())
        staged=root/'s'
        stage(candidate_dir,graph,baseline_copy,staged)
        # Existing private text staging marks ownership as text. An asset uses
        # the established graphic class without changing the public IR format.
        with np.load(candidate_dir/'pixels.npz',allow_pickle=False) as data:mask=data['mask']
        with np.load(staged/'ownership.npz',allow_pickle=False) as data:maps={name:data[name].copy() for name in data.files}
        maps['ownership'][mask]=0;np.savez_compressed(staged/'ownership.npz',**maps)
        ownership=json.loads((staged/'ownership.json').read_text('utf8'))
        ownership['map_sha256']=digest(staged/'ownership.npz');write(staged/'ownership.json',ownership)
        composed=SceneGraph.model_validate_json((staged/'scene.json').read_text('utf8'))
        if composed.selected[identifier]['asset_reference']['sha256']!=digest(image):
            raise ValueError('staged asset bytes changed')
        retained_scene=identifier+'-candidate-scene.json'
        shutil.copyfile(root/'candidate-scene.json',staged/retained_scene)
        retained_checks=identifier+'-independent-checks.json'
        shutil.copyfile(root/'asset-independent-checks.json',staged/retained_checks)
        previews=[]
        for entry in result['preview_files']:
            filename=identifier+'-'+Path(entry['path']).name
            shutil.copyfile(entry['path'],staged/filename)
            previews.append({'path':str(output/filename),'role':entry['role']})
        result.update(candidate_directory=str(directory),candidate_scene=str(output/retained_scene),
                      preview_files=previews,independent_checks=str(output/retained_checks))
        write(staged/'asset-stage.json',{'schema':'AgentAssetStage/1','source_candidate_id':candidate.candidate_id,
            'staged_candidate_id':cid,'candidate_source':str(directory),'source_sha256':graph.source_sha256,
            'asset_file':basename,'asset_sha256':digest(image),'parent_scene_sha256':digest(baseline/'scene.json'),
            'independent_checks':result,'office':'NOT_RUN','go':'NOT_RUN','published':False})
        if parent!={str(p.relative_to(baseline)):digest(p) for p in baseline.rglob('*') if p.is_file()}:
            raise ValueError('asset staging modified its parent')
        if original!={str(p.relative_to(directory)):digest(p) for p in directory.rglob('*') if p.is_file()}:
            raise ValueError('asset staging modified its source candidate')
        staged.rename(output)
    return output/'slideir.json'
