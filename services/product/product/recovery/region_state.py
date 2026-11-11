"""Copy-on-write dependency-local rollback of verified source-owner closures."""
import json
import shutil

import numpy as np
from PIL import Image

from ..materialize import digest,write
from ..parallel_contracts import SceneGraph
from ..parallel_regions import regions,finish_regions
from ..parallel_rebuild import adapt


def closure(graph, identifiers):
    records,by_id=regions(graph)
    affected={by_id[i]['region_id'] for i in identifiers if i in by_id}
    while True:
        expanded=affected|{r['region_id'] for r in records if affected.intersection(r['dependencies'])}
        if expanded==affected:break
        affected=expanded
    return {identifier for r in records if r['region_id'] in affected for identifier in r['member_ids']}


def rollback(original, baseline, identifiers, output, reason):
    graph=SceneGraph.model_validate_json((baseline/'scene.json').read_text('utf8'))
    remove=closure(original,identifiers)&graph.selected.keys()
    if not remove:return baseline/'slideir.json'
    source=np.array(Image.open(baseline/'source.png').convert('RGB'))
    if digest(baseline/'source.png')!=original.source_sha256:raise ValueError('rollback source binding mismatch')
    masks={}
    for identifier,record in graph.selected.items():
        ref=record['owner_evidence'];file=(baseline/ref['file']).resolve()
        if not file.is_relative_to(baseline.resolve()) or digest(file)!=ref['sha256']:
            raise ValueError('rollback owner proof mismatch')
        with np.load(file,allow_pickle=False) as data:masks[identifier]=data['mask'].astype(bool)
    restored=np.logical_or.reduce([masks[i] for i in remove])
    retained=np.zeros(source.shape[:2],bool)
    for identifier,mask in masks.items():
        if identifier not in remove:retained|=mask
    if (restored&retained).any():raise ValueError('rollback intersects retained owner')
    shutil.copytree(baseline,output)
    plate=np.array(Image.open(output/'source-plate.png').convert('RGBA'))
    plate[restored,:3]=source[restored];plate[restored,3]=255
    Image.fromarray(plate).save(output/'source-plate.png')
    for identifier in remove:graph.selected.pop(identifier)
    originals={t.id:t for t in original.text.texts}
    graph.text.texts=[originals[t.id].model_copy(deep=True) if t.id in remove else t for t in graph.text.texts]
    used_styles={t.style for t in graph.text.texts}|{span.style for t in graph.text.texts for span in t.spans}
    original_styles={s.id for s in original.text.styles}
    graph.text.styles=[s for s in graph.text.styles if s.id in used_styles|original_styles]
    graph.transactions=[r for r in graph.transactions if r['id'] not in remove]
    report=json.loads((output/'transactions.json').read_text('utf8'))
    report=[r for r in report if r['id'] not in remove]+[
        {'id':i,'state':'FALLBACK','reason_code':reason,'reason':'dependency-local rollback'} for i in sorted(remove)]
    graph.raster_backing['sha256']=digest(output/'source-plate.png')
    write(output/'scene.json',graph.model_dump());write(output/'transactions.json',report)
    write(output/'slideir.json',adapt(graph,output))
    with np.load(output/'editable-mask.npz') as data:mask=data['mask']&~restored
    np.savez_compressed(output/'editable-mask.npz',mask=mask)
    with np.load(output/'ownership.npz') as data:maps={k:data[k] for k in data.files}
    maps['ownership'][restored]=4;np.savez_compressed(output/'ownership.npz',**maps)
    ownership=json.loads((output/'ownership.json').read_text('utf8'))
    ownership['map_sha256']=digest(output/'ownership.npz');write(output/'ownership.json',ownership)
    write(output/'region-transactions.json',finish_regions(regions(original)[0],report))
    write(output/'rollback.json',{'removed':sorted(remove),'retained':sorted(graph.selected),'reason':reason,
          'parent_scene_sha256':digest(baseline/'scene.json'),'source_sha256':original.source_sha256})
    write(output/'build-binding.json',{'trial_only':True,'scene_sha256':digest(output/'scene.json'),
          'ir_sha256':digest(output/'slideir.json'),'parent_ir_sha256':digest(baseline/'slideir.json')})
    return output/'slideir.json'
