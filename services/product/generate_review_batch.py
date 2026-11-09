"""Explicit human-review batch; independent MoMA proposals and Go reviews."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import getpass
import json
import os
from pathlib import Path
import shutil
import threading
from PIL import Image
from product.contracts import Page
from product.fonts import font_manifest
from product.materialize import write, digest
from product.pipeline import run, compile_pages, ROOT, PipelineError


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sources',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--opencode-profile')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--pages',type=int,nargs='+',default=list(range(1,13)))
    parser.add_argument('--assemble-only',action='store_true')
    parser.add_argument('--planner',choices=['moma','terra-bridge'],default='moma')
    args=parser.parse_args()
    if any(i<1 or i>12 for i in args.pages):parser.error('pages must be 1 through 12')
    output=Path(args.output).resolve()
    if not output.is_relative_to(ROOT):parser.error('output must stay under compiler workspace')
    source_root=Path(args.sources).resolve()
    sources=[source_root/f'{i:02d}'/'source.png' for i in range(1,13)]
    if not all(p.is_file() for p in sources):parser.error('all twelve source pages are required')
    output.mkdir(parents=True,exist_ok=args.resume)
    source_records=[{'page':i,'path':str(p),'sha256':digest(p)} for i,p in enumerate(sources,1)]
    if args.resume and json.loads((output/'sources.json').read_text(encoding='utf-8'))!=source_records:
        parser.error('resume source set does not match the preserved batch')
    if not args.assemble_only:
        if not args.opencode_profile:parser.error('--opencode-profile is required for model calls')
        from product.go_provider import load_profile
        load_profile(args.opencode_profile)
        if args.planner=='moma':
            key=getpass.getpass('MoMA API key (hidden): ')
            os.environ['MOMA_API_KEY']=key;os.environ['PLANNER_API_KEY']=key
        manifest=font_manifest([])
    write(output/'sources.json',source_records)
    compile_lock=threading.Lock()
    def compiler(*a):
        with compile_lock:return compile_pages(*a)
    def page_job(index,source):
        job=output/f'page-{index:02d}'
        initial={}
        if job.exists():
            attempts=sorted(output.glob(f'page-{index:02d}*'))
            previous=attempts[-1]
            proposals=sorted(previous.glob('round-*/page-01-proposal.json'))
            validations=sorted(previous.glob('round-*/page-01-validation.json'))
            if proposals:
                initial['initial_plan']=json.loads(proposals[-1].read_text(encoding='utf-8'))
                initial['initial_feedback']={
                    'validation':json.loads(validations[-1].read_text(encoding='utf-8')) if validations else None,
                    'repair_instruction':'Keep each text box inside its original source card. If a line is too wide, restore the source paragraph line breaks using actual newline characters in text. The reported unbroken width is diagnostic, not a request to widen the box across cards or past the page. Preserve all wording and font sizes; adjust line spacing and box height only within the source region.'}
            job=output/f'page-{index:02d}-retry-{len(attempts):02d}'
        (job/'inputs').mkdir(parents=True)
        with Image.open(source) as image:image.convert('RGB').save(job/'inputs/01.png')
        write(job/'request.json',{'sources':['01.png'],'delivery_for_human_review':True,'max_calls':None,**initial})
        def progress(**kw):
            write(job/'progress.json',kw)
            print(json.dumps({'batch_page':index,**kw}),flush=True)
        provider=None
        if args.planner=='terra-bridge':
            from product.agent_bridge import FilePlanner
            provider=FilePlanner(output/'bridge'/job.name)
        try:result=run(job,provider=provider,compiler=compiler,manifest=manifest,progress=progress)
        except Exception as exc:
            result={'status':'FAILED','error':str(exc) if isinstance(exc,PipelineError) else type(exc).__name__}
            write(job/'result.json',result)
        print(json.dumps({'batch_page':index,'status':result['status']}),flush=True)
        return result
    results={}
    if args.resume:
        for i in range(1,13):
            for candidate in sorted(output.glob(f'page-{i:02d}*')):
                file=candidate/'result.json'
                if file.exists():
                    result=json.loads(file.read_text(encoding='utf-8'))
                    if result['status']=='REVIEW_REQUIRED':results[i]=result
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(page_job,i,p):i for i,p in enumerate(sources,1)
                 if not args.assemble_only and i not in results and i in args.pages}
        for future in as_completed(futures):
            results[futures[future]]=future.result()
            write(output/'batch-progress.json',{str(k):v['status'] for k,v in sorted(results.items())})
    if len(results)!=12 or any(r['status']!='REVIEW_REQUIRED' for r in results.values()):
        write(output/'result.json',{'status':'INCOMPLETE','pages':{str(k):v['status'] for k,v in sorted(results.items())}})
        return
    ordered=[results[i] for i in range(1,13)]
    write(output/'selected-pages.json',ordered)
    pages=[Page.model_validate(json.loads(Path(r['effective_plans'][0]).read_text(encoding='utf-8'))) for r in ordered]
    artifacts=compile_pages([r['slide_irs'][0] for r in ordered],output/'combined-build',pages)
    delivery=output/'delivery';delivery.mkdir()
    shutil.copyfile(artifacts['pptx'],delivery/'Image2PPT-12pages-human-review.pptx')
    for i,p in enumerate(sorted(Path(artifacts['preview']).glob('*.png')),1):
        shutil.copyfile(p,delivery/f'page-{i:02d}.png')
    write(output/'result.json',{'status':'REVIEW_REQUIRED','pages':12,'human_acceptance':False,
        'model_calls':sum(len(json.loads(p.read_text(encoding='utf-8'))) for p in output.glob('page-*/calls.json')),
        'selected_attempt_model_calls':sum(r['model_calls'] for r in ordered),
        'artifacts':{p.name:digest(p) for p in delivery.iterdir()}})
    print('BATCH_READY_FOR_HUMAN_REVIEW',flush=True)


if __name__=='__main__':main()
