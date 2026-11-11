"""Reuse immutable, actually edited delivery objects before local expansion."""
import json
from pathlib import Path
import shutil

from ..materialize import digest,write
from ..parallel_candidate_stage import compatible_text
from ..parallel_contracts import SceneGraph
from ..parallel_store import Store,key


def load(job,graph,fonts):
    prior=Path(job).resolve()
    commit=Store(prior/'state').committed()
    if not commit:raise ValueError('verified seed has no committed generation')
    result=commit['result']
    if not result['office_pass'] or not all(r.get('passed') and not r.get('findings') for r in result['independent_review']):
        raise ValueError('verified seed requires actual Office and independent review')
    source=next((Path(p) for p in result['slide_irs'] if json.loads(Path(p).read_text('utf8'))['source_sha256']==graph.source_sha256),None)
    if source is None:raise ValueError('verified seed source differs')
    baseline=SceneGraph.model_validate_json((source.parent/'scene.json').read_text('utf8'))
    if not compatible_text(graph,baseline):raise ValueError('verified seed confirmed text or typography differs')
    objects={o.id:o.model_dump() for o in graph.asset.objects}
    old={o.id:o.model_dump() for o in baseline.asset.objects}
    for identifier,record in baseline.selected.items():
        if graph.ownership.get(identifier)!='RESOLVED':raise ValueError('verified seed owner no longer resolved')
        if record['kind']=='asset' and objects.get(identifier)!=old.get(identifier):
            raise ValueError('verified seed asset definition differs')
    def relations(scene):
        ids=set(baseline.selected)
        return {(r.source,r.target,r.kind,r.status) for r in scene.relations if r.source in ids or r.target in ids}
    if relations(graph)!=relations(baseline):raise ValueError('verified seed dependency closure differs')
    old_fonts=json.loads((prior/'fonts.json').read_text('utf8'))
    if key(fonts)!=key(old_fonts):raise ValueError('verified seed font inventory differs')
    styles={s.id:s for s in baseline.text.styles}
    used=set()
    for text in baseline.text.texts:
        if baseline.selected.get(text.id,{}).get('kind')!='text':continue
        for sid in {text.style,*[span.style for span in text.spans]}:
            style=styles[sid];used.update([(style.ea,style.bold),(style.latin,style.bold)])
    for row in fonts['fonts']:
        if (row['family'],row['bold']) in used and digest(row['file'])!=row['sha256']:
            raise ValueError('verified seed font bytes changed')
    regions=[dict(row) for page in result['regions'] for row in page
             if row['state']=='VERIFIED_LOCAL' and row['object_id'] in baseline.selected]
    return {'source_job':str(prior),'source_ir':str(source),'source_sha256':graph.source_sha256,
        'semantics':{'ownership':graph.ownership,'relations':[r.model_dump() for r in graph.relations]},
        'selected_ids':list(baseline.selected),'regions':regions,'office_pptx_sha256':result['artifacts']['result.pptx'],
        'files':{path:sha for path,sha in commit['files'].items() if Path(path).is_relative_to(source.parent)}}


def materialize(seed,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    source=Path(seed['source_ir']).parent
    if not seed['files'] or any(digest(p)!=sha for p,sha in seed['files'].items()):
        raise ValueError('verified seed bytes changed before reuse')
    target=folder/'seed';shutil.copytree(source,target)
    semantics=seed.get('semantics')
    composed=SceneGraph.model_validate_json((target/'scene.json').read_text('utf8'))
    if semantics and (composed.ownership!=semantics['ownership'] or [r.model_dump() for r in composed.relations]!=semantics['relations']):
        from ..parallel_contracts import Relation
        from ..parallel_regions import regions,finish_regions
        composed.ownership=dict(semantics['ownership'])
        composed.relations=[Relation.model_validate(r) for r in semantics['relations']]
        write(target/'scene.json',composed.model_dump())
        reports=json.loads((target/'transactions.json').read_text('utf8'))
        write(target/'region-transactions.json',finish_regions(regions(composed)[0],reports))
        write(target/'build-binding.json',{'trial_only':True,'scene_sha256':digest(target/'scene.json'),
            'ir_sha256':digest(target/'slideir.json'),'parent_scene_sha256':digest(source/'scene.json'),
            'semantic_update_only':True,'selected_owner_closures_unchanged':True})
    write(folder/'verified-seed.json',seed)
    return target/'slideir.json'
