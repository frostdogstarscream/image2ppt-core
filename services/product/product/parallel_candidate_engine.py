"""Bounded r11 text trials; read-only inputs and immutable per-attempt receipts."""
from pathlib import Path
import numpy as np
from PIL import Image
from .materialize import digest, write
from .parallel_store import key
from .parallel_regions import regions
from .parallel_candidate_contracts import ReconstructionCandidate, PreflightResult, transition, candidate_path
from .parallel_text_candidates import generate, TextUncertainty, peer_protection

VERSION='r11-text-candidates/1'
CHECKS=('source_binding','content_verification','owner_coverage','protected_peer_integrity','background_proof','region_atomicity')


def engine_mode(value='off'):
    if value not in ('off','r11-shadow','r12-shadow','r12-regional'):raise ValueError('unsupported candidate engine')
    return value


def font_bindings(graph,fonts):
    used={(name,s.bold) for s in graph.text.styles for name in (s.ea,s.latin)}
    files={str(Path(f['file']).resolve()) for f in fonts.get('fonts',[]) if (f['family'],f['bold']) in used}
    return {file:digest(file) if Path(file).is_file() else 'MISSING' for file in sorted(files)}


def shadow_discover(graph,snap,fonts,directory,store,stable=()):
    """Upload integration, disabled by default; no output choice or model call."""
    import json
    from .parallel_pixels import graphic_ink
    directory=Path(directory)
    # Code and source changes get a separate directory; resume reuses only a
    # complete, hash-verified immutable candidate index.
    binding=key({'scene':graph.model_dump(),'fonts':fonts,'font_files':font_bindings(graph,fonts),'engine':digest(Path(__file__)),
                 'text':digest(Path(__file__).with_name('parallel_text_candidates.py')),
                 'stable':[a.get('sha256') for a in stable]})
    target=directory/('r11-shadow-'+binding[:24])
    index=target/'candidate-index.json'
    if index.exists():
        data=json.loads(index.read_text(encoding='utf-8'))
        for row in data['candidates']:
            candidate=ReconstructionCandidate.model_validate(row)
            for name,sha in candidate.references.items():
                if name not in ('pixels.npz','payload.json') or digest(candidate_path(target,candidate.candidate_id)/name)!=sha:
                    raise ValueError('shadow candidate cache binding mismatch')
        return data['candidates']
    if target.exists():
        # An interrupted attempt is retained. It made no external calls, so a
        # new immutable local attempt is safe and does not duplicate charges.
        from uuid import uuid4
        target=target.with_name(target.name+'-'+uuid4().hex[:8])
    protected=graphic_ink(stable,tuple(graph.image_size),graph.source_sha256)
    return discover(graph,snap,fonts,target,store,blocked=protected)


def discover(graph, snap, fonts, directory, store, *, blocked=None, object_ids=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    if digest(snap['path']) != graph.source_sha256:
        raise ValueError('candidate source binding mismatch')
    source=np.array(Image.open(snap['path']).convert('RGB'))
    if list(source.shape[1::-1]) != graph.image_size:
        raise ValueError('candidate source dimensions mismatch')
    original=key(graph.model_dump());source.setflags(write=False)
    font_files=font_bindings(graph,fonts)
    from .parallel_editability_policy import profiles
    write(directory/'editability-profiles.json',profiles(graph))
    blocked=np.zeros(source.shape[:2],bool) if blocked is None else blocked.copy()
    _,by_id=regions(graph);receipts=[]
    algorithm=key({p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('parallel_text_candidates.py'),
        Path(__file__).with_name('parallel_candidate_contracts.py'),Path(__file__).with_name('parallel_rebuild.py')]})
    for text in graph.text.texts:
        if object_ids is not None and text.id not in object_ids:continue
        for strategy in ('T1-exact','T2-soft-alpha-perimeter','T3-joint-backing','T4-source'):
            identity={'source':graph.source_sha256,'scene':original,'object':text.id,'strategy':strategy,
                      'version':VERSION,'algorithm':algorithm,'fonts':key(fonts),'font_files':font_files,
                      'protected':__import__('hashlib').sha256(blocked.tobytes()).hexdigest()}
            cid='c-'+key(identity);folder=candidate_path(directory,cid);folder.mkdir()
            x,y,w,h=map(round,text.bbox)
            if min(x,y)>=0 and min(w,h)>0 and x+w<=source.shape[1] and y+h<=source.shape[0]:
                Image.fromarray(source[y:y+h,x:x+w]).save(folder/'source-crop.png')
            candidate=ReconstructionCandidate(candidate_id=cid,source_sha256=graph.source_sha256,
                scene_sha256=original,region_id=by_id[text.id]['region_id'],object_ids=[text.id],
                representation='RASTER_FALLBACK' if strategy=='T4-source' else 'HYBRID' if strategy.startswith('T3') else 'NATIVE_TEXT',
                strategy=strategy,provenance={'algorithm_version':VERSION,'algorithm_sha256':algorithm,'input_binding':key(identity)})
            checks={c:'UNKNOWN' for c in CHECKS};checks['source_binding']='PASS'
            issues=[];unknown=None;refs={};result=None
            try:
                if strategy=='T4-source':raise TextUncertainty('RASTER_NOT_EDITABLE','safe source fallback has zero editable gain')
                if graph.ownership[text.id]!='RESOLVED':
                    checks['content_verification']='FAIL'
                    raise TextUncertainty('CONTENT_CONTRADICTION','text ownership/content unresolved')
                checks['content_verification']='PASS'
                if len(by_id[text.id]['member_ids'])!=1:
                    raise TextUncertainty('REGION_CLOSURE_UNPROVEN','single text trial cannot split indivisible region')
                checks['region_atomicity']='PASS'
                result=generate(text,graph,source,fonts,blocked,strategy)
                protection=peer_protection(text,graph,source,blocked)
                if (result['mask']&protection).any():
                    checks['protected_peer_integrity']='FAIL'
                    raise TextUncertainty('PROTECTED_NEIGHBOR','candidate overlaps peer protection')
                checks.update(owner_coverage='PASS',protected_peer_integrity='PASS',background_proof='PASS')
                unknown=0
                np.savez_compressed(folder/'pixels.npz',mask=result['mask'],patch=result['patch'],bbox=result['bbox'])
                payload={k:v for k,v in result.items() if k not in ('mask','patch')}
                write(folder/'payload.json',payload)
                refs={n:digest(folder/n) for n in ('pixels.npz','payload.json')}
                x,y,w,h=result['bbox']
                Image.fromarray(source[y:y+h,x:x+w]).save(folder/'source-crop.png')
                Image.fromarray(result['mask'][y:y+h,x:x+w].astype('uint8')*255).save(folder/'candidate-mask.png')
                Image.fromarray(result['patch']).save(folder/'proposed-backing.png')
            except (ValueError,KeyError) as exc:
                issues=[getattr(exc,'code','LEGACY_TEXT_UNCERTAINTY')]
                diagnostic_files={}
                for name,array in getattr(exc,'artifacts',{}).items():
                    file=folder/(name+'.png');Image.fromarray(array.astype('uint8')*255 if array.dtype==bool else array).save(file)
                    diagnostic_files[file.name]=digest(file)
                write(folder/'diagnosis.json',{'reason':str(exc),'code':issues[0],'details':getattr(exc,'evidence',{}),
                                              'evidence_files':diagnostic_files})
            status='FAIL' if 'FAIL' in checks.values() else 'UNKNOWN' if 'UNKNOWN' in checks.values() else 'PASS'
            preflight=PreflightResult(candidate_id=cid,status=status,checks=checks,unknown_pixels=unknown,
                reason_codes=issues,evidence_hashes=list(refs.values()))
            candidate=candidate.model_copy(update={'hard_gate':status,'references':refs,'issues':issues})
            candidate=transition(candidate,{'PASS':'PRECHECK_PASS','FAIL':'PRECHECK_BLOCKED','UNKNOWN':'PRECHECK_UNKNOWN'}[status])
            write(folder/'preflight.json',preflight.model_dump(by_alias=True))
            payload=candidate.model_dump(by_alias=True);write(folder/'candidate.json',payload)
            store.candidate(text.id,payload);receipts.append(payload)
    if key(graph.model_dump())!=original or digest(snap['path'])!=graph.source_sha256:
        raise ValueError('candidate discovery modified input')
    write(directory/'candidate-index.json',{'schema':'CandidateIndex/1','candidates':receipts,
          'model_calls':0,'published':False,'scene_sha256':original,'font_files':font_files})
    return receipts
