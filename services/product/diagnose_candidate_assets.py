"""Replay complete asset candidates and preserve protected-owner conflicts."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from product.parallel_contracts import SceneGraph
from product.parallel_asset_candidates import diagnose
from product.object_extract import load_page,load_sam
from product.materialize import digest,write


def run(previous,perception,baseline,page,output,objects):
    previous,perception,baseline,output=map(lambda p:Path(p).resolve(),(previous,perception,baseline,output))
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    graph=SceneGraph.model_validate(load(previous/f'page-{page:02d}/input-scene.json'))
    source_file=previous/f'page-{page:02d}/after/source.png'
    if digest(source_file)!=graph.source_sha256:raise ValueError('source binding mismatch')
    source=np.array(Image.open(source_file).convert('RGB'));saved=load(baseline)
    layers=load_page(saved['pages'][page-1]['layers']);stable=[a for a in saved['stable_assets'] if a['page']==page]
    masks={};plan=perception/f'page-{page:02d}/sam-plan.json'
    if plan.exists():
        for obj in graph.asset.objects:
            if obj.id not in objects:continue
            candidates,error=load_sam(plan.parent/'sam',obj,graph.source_sha256,digest(plan),tuple(graph.image_size),'sam3',individual=True)
            if not error:masks[obj.id]=candidates
    # The original build records these exact protections. Final ownership also
    # includes staged text; that only adds conservative vetoes in this diagnostic.
    ownership=previous/f'page-{page:02d}/after/ownership.npz'
    with np.load(ownership,allow_pickle=False) as data:
        protection=(data['ownership']==1)|data['uncertain_text']
    result=diagnose(graph,source,layers,stable,masks,protection,output,set(objects))
    write(output/'input-binding.json',{'source_sha256':graph.source_sha256,'ownership_sha256':digest(ownership),
        'baseline_sha256':digest(baseline),'sam_plan_sha256':digest(plan) if plan.exists() else None,
        'model_calls':0,'fresh_upload':False})
    print(json.dumps([{'object':r['object_id'],'method':r.get('method'),'status':r['status'],'overlap_pixels':r.get('overlap_pixels')} for r in result]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('previous');p.add_argument('perception');p.add_argument('baseline');p.add_argument('output')
    p.add_argument('--page',type=int,default=5);p.add_argument('--objects',nargs='+',default=['cloud_server']);a=p.parse_args()
    run(a.previous,a.perception,a.baseline,a.page,a.output,a.objects)
