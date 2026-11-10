"""Run the same product pipeline locally; credentials remain in process memory."""
import argparse
import getpass
import json
import os
from pathlib import Path
from PIL import Image
from product.pipeline import ROOT, PipelineError
from product.parallel_pipeline import run
from product.materialize import write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',action='append',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--prompt-key',action='store_true')
    parser.add_argument('--opencode-profile')
    parser.add_argument('--pipeline-mode',choices=['parallel_review','parallel_shadow'],default='parallel_review')
    parser.add_argument('--object-extraction',choices=['default','auto','off','alpha','sam3','sam3.1'],default='default')
    args=parser.parse_args()
    if args.opencode_profile:
        from product.go_provider import load_profile
        load_profile(args.opencode_profile)
    output=Path(args.output).resolve()
    if not output.is_relative_to(ROOT):parser.error('output must stay under compiler workspace')
    if args.prompt_key:
        key=getpass.getpass('MoMA API key (hidden): ')
        os.environ['MOMA_API_KEY']=key
        os.environ['PLANNER_API_KEY']=key
        if not os.environ.get('OPENCODE_GO_API_KEY'):
            os.environ['OPENCODE_GO_API_KEY']=getpass.getpass('OpenCode Go API key (hidden): ')
    (output/'inputs').mkdir(parents=True,exist_ok=False)
    names=[]
    for i,path in enumerate(args.image,1):
        name=f'{i:02d}.png'
        with Image.open(path) as image:image.convert('RGB').save(output/'inputs'/name)
        names.append(name)
    write(output/'request.json',{'sources':names,'max_calls':None,'italic_difference_ignored':True,
                               'pipeline_mode':args.pipeline_mode,'object_extraction':args.object_extraction})
    def progress(**kw):
        write(output/'progress.json',kw)
        print(json.dumps(kw),flush=True)
    try:result=run(output,progress=progress)
    except Exception as exc:
        result={'status':'FAILED','error':str(exc) if isinstance(exc,PipelineError) else type(exc).__name__,'artifacts':{}}
        write(output/'result.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
