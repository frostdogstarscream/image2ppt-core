"""Reproduce a single real icon failure without rerunning perception or OCR."""
import argparse,json
from pathlib import Path
from product.parallel_contracts import SceneGraph
from product.recovery.single_asset import probe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('job','layers','object-id','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--office',action='store_true')
    p.add_argument('--strategy',choices=['source-halo','verified-backing','sam'],default='sam')
    args=p.parse_args();job=Path(args.job).resolve()
    result=json.loads((job/'result.json').read_text('utf8'));base=Path(result['slide_irs'][0]).parent
    graph=SceneGraph.model_validate_json((base/'scene.json').read_text('utf8'))
    row=probe(graph,args.object_id,base/'source.png',args.layers,base,args.output,strategy=args.strategy)
    if args.office and row['status']=='CANDIDATE_READY':
        from product.recovery.single_asset import verify_office
        row['l2']=verify_office(row,graph,base/'source.png',args.layers,base,json.loads((job/'fonts.json').read_text('utf8')),args.output)
        print(json.dumps({'l2_status':row['l2']['status'],'elapsed_seconds':row['l2']['elapsed_seconds']}),flush=True)
    print(json.dumps({k:row.get(k) for k in ('object_id','strategy','status','failure','reason','cache_hit','elapsed_seconds','receipt','candidate_directory')}),flush=True)


if __name__=='__main__':main()
