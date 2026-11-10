"""Explicit 12-page live regression; prompts for keys without writing them."""
import argparse,getpass,json,os,shutil
from pathlib import Path
from product.materialize import write

def main():
    p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('output');p.add_argument('--prompt-keys',action='store_true');p.add_argument('--reuse-perception-from');p.add_argument('--reuse-assets',action='store_true');args=p.parse_args()
    if args.prompt_keys:
        secret=getpass.getpass('MoMA key (hidden): ');os.environ['MOMA_API_KEY']=secret;os.environ['PLANNER_API_KEY']=secret
        os.environ['OPENCODE_GO_API_KEY']=getpass.getpass('Go key (hidden): ')
    from product.parallel_pipeline import run
    baseline=Path(args.baseline).resolve();b=json.loads(baseline.read_text(encoding='utf-8'));output=Path(args.output).resolve()
    if not (output/'request.json').exists():
        (output/'inputs').mkdir(parents=True,exist_ok=False)
        for i,row in enumerate(b['pages'],1):shutil.copyfile(row['source']['path'],output/'inputs'/f'{i:02d}.png')
        write(output/'request.json',{'sources':[f'{i:02d}.png' for i in range(1,len(b['pages'])+1)],
            'pipeline_mode':'parallel_review','object_extraction':'auto','frozen_baseline':str(baseline),
            **({'reuse_perception_from':str(Path(args.reuse_perception_from).resolve()),'reuse_assets':args.reuse_assets} if args.reuse_perception_from else {})})
    result=run(output,progress=lambda **v:print(json.dumps(v),flush=True))
    print(json.dumps({k:result[k] for k in ('status','pages','reconstructed_pages','generation')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
