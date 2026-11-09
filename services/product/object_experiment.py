"""Frozen-input object-planning experiment. No baseline plan can be modified."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from product.object_contracts import ObjectPlan, OBJECT_PROMPT
from product.materialize import digest, write


def prepare(root):
    records=json.loads((root/'inputs.json').read_text(encoding='utf-8'))
    for record in records:
        directory=Path(record['directory']);layerdir=directory/'layerd'
        meta=json.loads((layerdir/'manifest.json').read_text(encoding='utf-8'))
        visual=json.loads(Path(record['baseline_page']).read_text(encoding='utf-8'))
        source=Image.open(record['source']).convert('RGB')
        baseline=Image.open(directory/'baseline/source.png').convert('RGB')
        if not np.array_equal(np.array(source),np.array(baseline)):raise ValueError('source pixels differ from text baseline')
        images=[record['source']];layers=[]
        yy,xx=np.indices((source.height,source.width));checker=np.where(((xx//20+yy//20)%2)[...,None],215,245).astype(np.uint8)
        checker=np.repeat(checker,3,axis=2)
        for layer in meta['layers']:
            p=layerdir/layer['file']
            if digest(p)!=layer['sha256']:raise ValueError('layer hash mismatch')
            image=Image.open(p).convert('RGBA');background=Image.fromarray(checker).convert('RGBA');background.alpha_composite(image)
            preview=directory/(layer['file'].replace('.png','-checker.png'));background.convert('RGB').save(preview)
            images.append(str(preview))
            layers.append({'file':layer['file'],'z':layer['z'],'role':layer.get('role'),'sha256':digest(p)})
        context={'source_sha256':digest(record['source']),'image_size':list(source.size),
                 'layers':layers,'image_order':['source']+[x['file'] for x in layers],
                 'protected_texts':[{'id':t['id'],'bbox':t['bbox'],'text':t['text']} for t in visual['texts']],
                 'baseline_graphics':visual['crops']+visual['shapes']}
        request={'system':OBJECT_PROMPT,'context':context,'images':images,'output_schema':ObjectPlan.model_json_schema()}
        write(directory/'object-request.json',request)
    write(root/'experiment.json',{'schema':'ObjectExperiment/1','pages':len(records),
        'prompt_sha256':hashlib.sha256(OBJECT_PROMPT.encode()).hexdigest(),
        'text_baseline':'preserved Terra VisualPage; not human accepted',
        'arms':['alpha','sam3','sam3.1'],'human_acceptance':False,
        'source_hashes':[digest(r['source']) for r in records]})


def plan(root,planner):
    from object_review import start_monitor
    start_monitor(root)
    from product.pipeline import Provider, settings
    actor=Provider(settings()) if planner=='moma' else None
    records=json.loads((root/'inputs.json').read_text(encoding='utf-8'))
    for record in records:
        d=Path(record['directory']);request=json.loads((d/'object-request.json').read_text(encoding='utf-8'))
        out=root/'plans'/planner/f"{record['page']:02d}.json"
        binding={'request_sha256':digest(d/'object-request.json'),
                 'image_sha256':[digest(p) for p in request['images']],
                 'model':actor.model if actor else 'gpt-6.1-terra'}
        receipt=root/'bindings'/planner/f"{record['page']:02d}.json"
        if out.exists():
            value=ObjectPlan.model_validate(json.loads(out.read_text(encoding='utf-8')))
            if value.source_sha256!=request['context']['source_sha256']:raise ValueError('stale plan')
            if not receipt.exists():raise ValueError('cached plan lacks a frozen-context receipt')
            if json.loads(receipt.read_text(encoding='utf-8'))!={**binding,'plan_sha256':digest(out)}:
                raise ValueError('cached plan context or images changed')
            continue
        if actor is None:continue
        call=root/'calls'/planner/f"{record['page']:02d}.json"
        for attempt in range(3):
            started=time.time();write(call,{'status':'RESERVED','attempt':attempt+1,'started_at':started,'model':actor.model})
            try:
                raw=actor.ask(request['system'],request['context'],request['images'],ObjectPlan)
                write(root/'raw'/planner/f"{record['page']:02d}-{attempt+1}.json",raw)
                value=ObjectPlan.model_validate(raw)
                if value.source_sha256!=request['context']['source_sha256'] or value.image_size!=request['context']['image_size']:raise ValueError('source binding mismatch')
                if any(not set(o.layers).issubset({x['file'] for x in request['context']['layers']}) for o in value.objects):raise ValueError('unknown layer')
                write(out,value.model_dump());write(receipt,{**binding,'plan_sha256':digest(out)})
                write(call,{'status':'RECEIVED','attempt':attempt+1,'model':actor.model,'elapsed_s':time.time()-started})
                print(json.dumps({'page':record['page'],'planner':planner,'objects':len(value.objects)}),flush=True);break
            except ValueError as exc:
                write(call,{'status':'INVALID','attempt':attempt+1,'error':str(exc)[:2000]})
                request['context']['validation_feedback']=str(exc)[:3000]
                if attempt==2:raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','plan']);ap.add_argument('--root',required=True,type=Path);ap.add_argument('--planner',choices=['moma','terra'],default='moma');ap.add_argument('--prompt-key',action='store_true');args=ap.parse_args()
    if args.prompt_key:
        import getpass,os
        key=getpass.getpass('MoMA key (hidden, memory only): ')
        os.environ['MOMA_API_KEY']=key;os.environ['PLANNER_API_KEY']=key
    if args.action=='prepare':prepare(args.root)
    else:plan(args.root,args.planner)
