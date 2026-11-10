"""Private twelve-page regression delivery using the product's real gate path."""
import argparse
import getpass
import json
import os
from pathlib import Path
import shutil
import time
from PIL import Image
from product.contracts import Page,Review
from product.materialize import digest,write,verify_package
from product.object_contracts import read_plan,OBJECT_REVIEW_PROMPT
from product.object_closure import extract_closed
from product.object_parent import adapt_owners,rebind_sam
from product.object_stage import effective_page
from product.object_finalize import finalize,review_objects
from product.pipeline import compile_pages,effective_findings
from product.go_provider import GoProvider
from product import prompts
from product.object_registry import fingerprint


def cached_page_review(context,images,cache_root,model,ask,*,previous_raw=None,previous_model=None):
    key=fingerprint({'context':context,'images':[digest(p) for p in images],
        'model':model,'prompt':prompts.REVIEW,'schema':Review.model_json_schema()})
    cache=Path(cache_root)/key;cache.mkdir(parents=True,exist_ok=True)
    rawfile=cache/'raw.json'
    if not rawfile.exists() and previous_model==model and previous_raw and Path(previous_raw).exists():
        raw=json.loads(Path(previous_raw).read_text(encoding='utf-8'))
        binding=('source_sha256','plan_sha256','preview_sha256')
        if all(raw.get(k)==context[k] for k in binding):
            write(rawfile,raw)
            write(cache/'historical-reuse.json',{'raw_sha256':digest(previous_raw),'binding':{k:context[k] for k in binding},'model':model})
    if rawfile.exists():return json.loads(rawfile.read_text(encoding='utf-8'))
    if (cache/'reserved.json').exists():raise RuntimeError('identical page review already reserved')
    write(cache/'reserved.json',{'state':'RESERVED','model':model,'key':key})
    raw=ask(context,images,Review);write(rawfile,raw);return raw


def export_office(pptx,directory):
    import win32com.client
    from office_objects import close_copy
    directory.mkdir(parents=True,exist_ok=False);original=digest(pptx)
    app=win32com.client.Dispatch('PowerPoint.Application');deck=None
    try:
        deck=app.Presentations.Open(str(Path(pptx).resolve()),ReadOnly=True,Untitled=False,WithWindow=False)
        width=round(float(deck.PageSetup.SlideWidth)/.75);height=round(float(deck.PageSetup.SlideHeight)/.75)
        for i in range(1,deck.Slides.Count+1):deck.Slides(i).Export(str((directory/f'page-{i:02d}.png').resolve()),'PNG',width,height)
    finally:
        if deck is not None:close_copy(deck)
    if digest(pptx)!=original:raise ValueError('original deck changed')


def run(args):
    if args.go_key_stdin:os.environ['OPENCODE_GO_API_KEY']=getpass.getpass('Go key (process memory): ')
    frozen=args.input.resolve();live=args.live.resolve();parent=args.parent.resolve();output=args.output.resolve()
    rows=json.loads((frozen/'inputs.json').read_text(encoding='utf-8'))
    if args.wait:
        while any(not (live/f"page-{r['page']:02d}/result.json").exists() and not list((live/f"page-{r['page']:02d}").glob('failure-*.json')) for r in rows):time.sleep(10)
    output.mkdir(parents=True,exist_ok=False);irs=[];parents=[];plans=[];reports=[];sources=[];ledger=[]
    prior=None
    if args.resume_final:
        prior=json.loads((args.resume_final.resolve()/'result.json').read_text(encoding='utf-8'))
        for name,sha in prior['artifacts'].items():
            if digest(Path(prior['delivery'])/name)!=sha:raise ValueError('previous delivery changed')
        if len(prior['slide_irs'])!=len(rows):raise ValueError('previous delivery page mismatch')
    offsets=json.loads(args.candidate_offsets.read_text(encoding='utf-8')) if args.candidate_offsets else {}
    focus=json.loads(args.focus_objects.read_text(encoding='utf-8')) if args.focus_objects else {}
    actor=None if args.stage_only else GoProvider(output)
    prior_models={r['model'] for r in json.loads((args.resume_final.resolve()/'calls.json').read_text(encoding='utf-8')) if r['kind']=='final_page_review'} if prior else set()
    def ask(kind,system,context,images,contract):
        call={'kind':kind,'state':'RESERVED','model':actor.model,'started_at':time.time()};ledger.append(call);write(output/'calls.json',ledger)
        try:value=actor.ask(system,context,images,contract)
        except Exception:
            call.update(state='FAILED',finished_at=time.time());write(output/'calls.json',ledger);raise
        call.update(state='RECEIVED',finished_at=time.time());write(output/'calls.json',ledger);return value
    for row in rows:
        number=row['page'];directory=output/f'page-{number:02d}';directory.mkdir();state=live/f'page-{number:02d}/state'
        baseline=parent/f'page-{number:02d}/slideir.json';history=parent/f'page-{number:02d}/extraction.json'
        page=Page.model_validate(json.loads(Path(row['baseline_page']).read_text(encoding='utf-8')))
        sources.append(Path(row['source']));parents.append(baseline)
        if prior:
            previous=Path(prior['slide_irs'][number-1]);previous_report=prior['reports'][number-1]
            if previous_report.get('publication')=='STABLE' and digest(previous)!=previous_report['scene_sha256']:
                raise ValueError('previous stable scene changed')
            parents[-1]=previous
            if number not in args.refresh_pages:
                report=json.loads(json.dumps(previous_report))
                for obj in report['objects']:
                    if obj['status']=='STAGED':obj['status']='STABLE_REUSED'
                report['reuse_provenance']={'result_sha256':digest(args.resume_final.resolve()/'result.json'),'scene_sha256':digest(previous)}
                irs.append(previous);plans.append(effective_page(page,previous));reports.append(report)
                print(json.dumps({'reused_final_page':number}),flush=True);continue
            baseline=previous
        if args.resume_stage:
            old=args.resume_stage.resolve()/f'page-{number:02d}/stage'
            report=json.loads((old/'closure.json').read_text(encoding='utf-8'));ir=old/'slideir.json'
            if digest(ir)!=report['scene_sha256']:raise ValueError('resume staged scene changed')
            report.update(state_directory=str(state),publication='STAGED_ONLY')
            irs.append(ir);plans.append(effective_page(page,ir));reports.append(report)
            print(json.dumps({'reused_stage_page':number,'staged':report['staged']}),flush=True);continue
        report={'objects':[],'publication':'RETAINED_PARENT','automatic_quality':'REVIEW_REQUIRED'};ir=baseline
        try:
            active=state/'active-plan.json';planfile=state/'object-plan.json'
            if active.exists():
                pointer=json.loads(active.read_text(encoding='utf-8'));planfile=(state/pointer['file']).resolve()
                if not planfile.is_relative_to(state) or digest(planfile)!=pointer['sha256']:raise ValueError('cached plan changed')
            plan,_=read_plan(json.loads(planfile.read_text(encoding='utf-8')))
            parent_ir=json.loads(baseline.read_text(encoding='utf-8'));old=json.loads(history.read_text(encoding='utf-8'))
            if plan.source_sha256!=old['source_sha256'] or parent_ir['source_sha256']!=plan.source_sha256:raise ValueError('historical ledger source mismatch')
            adapted,changes=adapt_owners(plan,parent_ir,old,closure=previous_report if prior else None)
            adaptedfile=directory/'adapted-plan.json';write(adaptedfile,adapted.model_dump())
            write(directory/'owner-adapter.json',{'schema':'HistoricalOwnerAdapter/1','raw_plan_sha256':digest(planfile),
                'parent_ir_sha256':digest(baseline),'history_sha256':digest(history),'changes':changes})
            samdir=None
            if (state/'sam').exists():
                candidates=[state/'sam',*[p for p in (state/'sam').iterdir() if p.is_dir()]]
                for candidate in candidates:
                    receipts=[p for p in candidate.glob('*.json') if not p.name.startswith(('raw-','remote-'))]
                    if any(json.loads(p.read_text(encoding='utf-8')).get('plan_sha256')==digest(planfile) for p in receipts):
                        samdir=directory/'sam-rebound';rebind_sam(planfile,adaptedfile,candidate,samdir);break
            rejected={o.id:['retained_for_focused_regression'] for o in adapted.objects if str(number) in focus and o.id not in focus[str(number)]}
            retained={r['id']:r for r in previous_report['objects'] if r['status'] in ('STAGED','STABLE_REUSED')} if prior else {}
            ir,report=extract_closed(adaptedfile,Path(row['directory'])/'layerd',baseline,directory/'stage',sam_dir=samdir,registry=state/'registry',candidate_offsets=offsets.get(str(number)),rejected=rejected,
                source_unmix_only=args.source_unmix_only,retained=retained)
            report.update(state_directory=str(state),publication='STAGED_ONLY')
        except (ValueError,KeyError,OSError) as exc:
            write(directory/'fallback.json',{'error_type':type(exc).__name__,'message':str(exc)[:800],'parent_preserved':True})
        irs.append(ir);plans.append(effective_page(page,ir));reports.append(report)
        print(json.dumps({'stage_page':number,'staged':sum(r['status']=='STAGED' for r in report['objects'])}),flush=True)
    if args.stage_only:
        write(output/'stage-result.json',{'reports':reports,'slide_irs':[str(p) for p in irs],
            'source_unmix_only':args.source_unmix_only,'human_acceptance':False,'publication':'STAGED_ONLY'})
        return
    compile_deck=compile_pages
    if args.resume_stage:
        previous=args.resume_stage.resolve()/'gates/build-0'
        old_irs=json.loads((previous/'input-irs.json').read_text(encoding='utf-8'))
        old_bindings=[digest(p) for p in old_irs]
        from image2ppt_vision.conversion import _final_json,_compiled_artifacts
        def compile_deck(current,path,current_pages):
            if [digest(p) for p in current]==old_bindings:
                artifact=_compiled_artifacts(_final_json((previous/'compiler.stdout').read_bytes()),previous,len(current_pages))
                verify_package(artifact['pptx'],current_pages)
                path.mkdir(parents=True,exist_ok=False);write(path/'compile-reuse.json',{'pptx_sha256':digest(artifact['pptx']),'input_ir_sha256s':old_bindings})
                return artifact
            return compile_pages(current,path,current_pages)
    artifacts,irs,plans=finalize(sources,irs,parents,reports,plans,output/'gates',compile_deck,
        lambda i,source,preview,ir,report,path:review_objects(source,preview,ir,report,path,
            lambda context,images,contract:ask('object_review',OBJECT_REVIEW_PROMPT,context,images,contract),review_model=actor.model),
        reuse_office=args.resume_stage.resolve()/'gates/office-0' if args.resume_stage else None,
        parent_previews=[Path(prior['delivery'])/f'page-{i:02d}.png' for i in range(1,len(rows)+1)] if prior else None)
    verify_package(artifacts['pptx'],plans)
    actual=output/'actual-office';export_office(artifacts['pptx'],actual)
    reviews=[]
    for i,(source,page,preview) in enumerate(zip(sources,plans,sorted(actual.glob('*.png'))),1):
        planfile=output/f'effective-{i:02d}.json';write(planfile,page.model_dump())
        binding={'source_sha256':digest(source),'plan_sha256':digest(planfile),'preview_sha256':digest(preview)}
        try:
            raw=cached_page_review({**binding,'plan':page.model_dump()},[source,preview],
                live/f'page-{i:02d}/state/final-review-cache',actor.model,
                lambda context,images,contract:ask('final_page_review',prompts.REVIEW,context,images,contract),
                previous_raw=args.resume_final.resolve()/f'page-{i:02d}/final-review-raw.json' if args.resume_final else None,
                previous_model=actor.model if prior_models=={actor.model} else None)
            write(output/f'page-{i:02d}/final-review-raw.json',raw);review=Review.model_validate(raw)
            if any(getattr(review,k)!=v for k,v in binding.items()):raise ValueError('final review binding mismatch')
            ids={n.id for n in [*page.texts,*page.shapes,*page.crops]}
            if any(not set(f.node_ids)<=ids for f in review.findings):raise ValueError('final review unknown object')
            reviews.append({'page':i,'status':'REVIEW_REQUIRED' if effective_findings(review) else 'AUTOMATED_PASS','findings':effective_findings(review)})
        except (RuntimeError,ValueError,OSError) as exc:reviews.append({'page':i,'status':'REVIEW_REQUIRED','error_type':type(exc).__name__})
        print(json.dumps({'review_page':i,'status':reviews[-1]['status']}),flush=True)
    delivery=output/'delivery';delivery.mkdir();shutil.copyfile(artifacts['pptx'],delivery/'Image2PPT-12pages-object-closure.pptx')
    for preview in actual.glob('*.png'):shutil.copyfile(preview,delivery/preview.name)
    gallery=''.join(f'<figure><figcaption>第 {i} 页</figcaption><img src="page-{i:02d}.png"></figure>' for i in range(1,len(plans)+1))
    (delivery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>12 页整页预览</title><style>body{font:18px system-ui;background:#eee;margin:24px}figure{margin:24px 0}img{display:block;max-width:100%}</style><h1>12 页整页预览</h1>'+gallery,encoding='utf-8')
    (delivery/'结果说明.txt').write_text('已生成 12 页可编辑 PPTX，并通过实际 PowerPoint 导出整页预览。未通过对象门禁的部分保留历史表示；尚未全部独立透明化，因此自动路线继续保持默认关闭。本轮保留既有文字和卡片。此版本待人工复核，不代表人工视觉验收通过。',encoding='utf-8-sig')
    write(output/'result.json',{'status':'REVIEW_REQUIRED','pages':len(plans),'human_acceptance':False,'powerpoint_rendered':True,
        'automatic_default':'off','reports':reports,'final_reviews':reviews,'slide_irs':[str(p) for p in irs],
        'delivery':str(delivery),'artifacts':{p.name:digest(p) for p in delivery.iterdir()}})
    print(json.dumps({'delivery':str(delivery),'status':'REVIEW_REQUIRED'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('input','live','parent','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--go-key-stdin',action='store_true');p.add_argument('--wait',action='store_true');p.add_argument('--resume-stage',type=Path)
    p.add_argument('--resume-final',type=Path);p.add_argument('--refresh-pages',type=int,nargs='+',default=[])
    p.add_argument('--candidate-offsets',type=Path);p.add_argument('--focus-objects',type=Path);p.add_argument('--source-unmix-only',action='store_true');p.add_argument('--stage-only',action='store_true');run(p.parse_args())
