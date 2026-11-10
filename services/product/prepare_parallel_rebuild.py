"""Replay valid three-branch evidence through changed LOCAL reconstruction only.

No old VisualPage planner, no model call, no re-OCR or segmentation when cached.
This is a regression driver; the normal upload worker produces its own evidence.
"""
import argparse,json,shutil
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from product.materialize import write,digest
from product.parallel_contracts import SceneGraph
from product.parallel_branches import snapshot,local_ocr
from product.parallel_fusion import fuse
from product.parallel_typography import calibrate
from product.parallel_text_anchor import anchor
from product.parallel_store import Store
from product.parallel_rebuild import build
from product.object_extract import load_page,load_sam
from product.resource_lease import lease

def prepare(baseline,previous,output):
    baseline=Path(baseline).resolve();previous=Path(previous).resolve();output=Path(output).resolve()
    b=json.loads(baseline.read_text(encoding='utf-8'));fonts=json.loads((previous/'fonts.json').read_text(encoding='utf-8'))
    (output/'inputs').mkdir(parents=True,exist_ok=False);store=Store(output/'state');cache=Path(__file__).resolve().parents[2]/'.runtime/parallel-cache'
    names=[];summary=[]
    for i,p in enumerate(b['pages'],1):
        name=f'{i:02d}.png';names.append(name);shutil.copyfile(p['source']['path'],output/'inputs'/name)
        snap=snapshot(output/'inputs'/name,output/'snapshots')
        prior=SceneGraph.model_validate(json.loads((previous/f'page-{i:02d}/barrier-1.json').read_text(encoding='utf-8')))
        if prior.source_sha256!=snap['source_sha256']:raise ValueError('replay source mismatch')
        branches={'text':prior.text,'asset':prior.asset,'surface':prior.surface}
        ocr=local_ocr(snap,cache,Path(__file__).resolve().parents[2]/'.runtime/ocr-models.json')
        graph=fuse(snap,branches,ocr);pixels=np.array(Image.open(snap['path']).convert('RGB'))
        stable=[a for a in b['stable_assets'] if a['page']==i]
        with lease('cpu'):
            graph=anchor(graph,ocr,pixels,fonts,store);graph=calibrate(graph,pixels,fonts,store,ocr=ocr,stable=stable);graph=anchor(graph,ocr,pixels,fonts,store,fitted=True)
        page=output/f'page-{i:02d}';page.mkdir();write(page/'barrier-1.json',graph.model_dump())
        masks={};samplan=previous/f'page-{i:02d}/sam-plan.json';samdir=previous/f'page-{i:02d}/sam'
        if samplan.exists():
            for obj in graph.asset.objects:
                values,error=load_sam(samdir,obj,snap['source_sha256'],digest(samplan),tuple(snap['image_size']),'sam3',individual=True)
                if not error:masks[obj.id]=values
        with lease('cpu'):
            ir,report=build(graph,snap,fonts,page/'staged',layers=load_page(p['layers']),
                stable=[a for a in b['stable_assets'] if a['page']==i],strategy='auto',sam_candidates=masks,pixel_cache=cache/'pixels')
        counts=Counter((r['state'],r.get('kind','other')) for r in report)
        row={'page':i,'counts':{'/'.join(k):v for k,v in counts.items()},'ir':str(ir)};summary.append(row);print(json.dumps(row),flush=True)
    write(output/'request.json',{'sources':names,'pipeline_mode':'parallel_review','object_extraction':'auto','frozen_baseline':str(baseline),
        'reuse_perception_from':str(previous),'reuse_assets':True})
    write(output/'reconstruction-replay.json',{'pages':summary,'model_calls':0,'semiautomatic_page_plans_used':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('previous');p.add_argument('output');a=p.parse_args();prepare(a.baseline,a.previous,a.output)
