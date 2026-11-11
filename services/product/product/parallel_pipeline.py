"""Independent r10 worker. Never invokes the serial page planner or its run()."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from .materialize import digest,write
from .parallel_contracts import mode,strategy,SceneGraph
from .parallel_store import Store,key
from .parallel_branches import snapshot,perceive,local_ocr,layer_evidence
from .parallel_fusion import fuse
from .parallel_rebuild import build
from .resource_lease import lease

ROOT=Path(__file__).resolve().parents[3]


def effective_texts(graph,irfile):
    ir=json.loads(Path(irfile).read_text(encoding='utf-8'));ids={n['id'] for n in ir['nodes'] if n['type']=='text'}
    return SimpleNamespace(texts=[t for t in graph.text.texts if t.id in ids],styles=graph.text.styles)


def safe_graph(graph):
    value=graph.model_copy(deep=True);value.selected={};value.transactions=[]
    value.ownership={k:'UNOWNED' for k in value.ownership};return value


def run(job,*,provider=None,review_provider=None,compiler=None,fonts=None,progress=None,office=None):
    from .parallel_config import configure
    configure()
    from .pipeline import Provider,settings,compile_pages,effective_findings
    from .fonts import font_manifest
    from .contracts import Review
    from .parallel_review import PROMPT as REVIEW,panels
    from .go_provider import GoProvider
    job=Path(job).resolve();request=json.loads((job/'request.json').read_text(encoding='utf-8'))
    from .parallel_candidate_engine import engine_mode,shadow_discover
    candidate_engine=engine_mode(request.get('candidate_engine',os.environ.get('IMAGE2PPT_CANDIDATE_ENGINE','off')))
    pipeline_mode=mode(request.get('pipeline_mode','parallel_review'));route=strategy(request.get('object_extraction','default'))
    cache=Path(os.environ.get('IMAGE2PPT_PARALLEL_CACHE',str(ROOT/'.runtime/parallel-cache')));cache.mkdir(parents=True,exist_ok=True)
    store=Store(job/'state');call_store=Store(cache/'model-calls')
    from .parallel_catalog import Catalog
    catalog=Catalog(cache)
    # Branch tasks live in this job; model-call receipts are reusable across jobs.
    store.call=call_store.call
    expected=store.head()
    if store.committed() is not None:
        try:catalog.register(job)
        except Exception as exc:write(job/'catalog-recovery-required.json',{'reason':type(exc).__name__})
        result=store.committed()['result'];write(job/'result.json',result);return result
    from .parallel_provider import Planner,Reviewer
    provider=provider or Planner(settings())
    if review_provider is None:
        try:review_provider=Reviewer(job)
        except RuntimeError:
            if not candidate_engine.startswith('r12-'):raise
            from .recovery.product_pipeline import UnavailableReviewer
            review_provider=UnavailableReviewer()
    compiler=compiler or compile_pages;fonts=fonts or font_manifest([]);write(job/'fonts.json',fonts)
    models=os.environ.get('IMAGE2PPT_OCR_MANIFEST',str(ROOT/'.runtime/ocr-models.json'))
    baseline=json.loads(Path(request['frozen_baseline']).read_text(encoding='utf-8')) if request.get('frozen_baseline') else {'pages':[],'stable_assets':[]}
    snaps=[snapshot(job/'inputs'/name,job/'snapshots') for name in request['sources']]
    for snap in snaps:
        known={a['id'] for a in baseline['stable_assets'] if a['source_sha256']==snap['source_sha256']}
        baseline['stable_assets'].extend(a for a in catalog.lookup(snap['source_sha256']) if a['id'] not in known)
    def update(**value):
        write(job/'progress.json',value)
        if progress:progress(**value)
    write(job/'provenance.json',{'pipeline_mode':pipeline_mode,'architecture':'text-asset-surface/'+('r12' if candidate_engine.startswith('r12-') else 'r10'),'serial_executed':False,
        'perception_mode':'hash-bound-recorded-branches' if request.get('perception_replay') else 'live-or-cache',
        'source_sha256s':[s['source_sha256'] for s in snaps],'moma_model':provider.model,'review_model':review_provider.model,
        'human_acceptance':False,'stable_baseline_is_pixel_truth':False,'candidate_engine':candidate_engine})
    def _page(i):
        with lease('page'):
            snap=snaps[i];folder=job/f'page-{i+1:02d}';folder.mkdir(parents=True,exist_ok=True)
            for existing in sorted((p for p in folder.iterdir() if p.is_dir()),key=lambda p:p.stat().st_mtime,reverse=True):
                if all((existing/f).is_file() for f in ('slideir.json','scene.json','transactions.json')):
                    ir=json.loads((existing/'slideir.json').read_text(encoding='utf-8'))
                    if ir['source_sha256']!=snap['source_sha256']:raise ValueError('resume source binding mismatch')
                    for a in ir['assets']:
                        if digest(existing/a['path'])!=a['sha256']:raise ValueError('resume candidate hash mismatch')
                    graph=SceneGraph.model_validate(json.loads((existing/'scene.json').read_text(encoding='utf-8')))
                    from .parallel_rebuild import CODE_SHA,UNMIX_CODE_SHA
                    receipt=existing/'build-binding.json'
                    if not receipt.exists() or graph.raster_backing is None:continue
                    binding=json.loads(receipt.read_text(encoding='utf-8'))
                    if binding.get('text_candidates',True)!=(candidate_engine!='r12-regional'):continue
                    if binding['rebuild_sha256']!=CODE_SHA or binding['unmix_sha256']!=UNMIX_CODE_SHA or binding['scene_sha256']!=digest(existing/'scene.json') or binding['ir_sha256']!=digest(existing/'slideir.json'):continue
                    if candidate_engine.startswith('r12-') and (folder/'barrier-1.json').is_file():
                        graph=SceneGraph.model_validate_json((folder/'barrier-1.json').read_text('utf8'))
                        if graph.source_sha256!=snap['source_sha256']:raise ValueError('resume source scene mismatch')
                    return graph,existing/'slideir.json',json.loads((existing/'transactions.json').read_text(encoding='utf-8')),{}
            saved=next((p['layers'] for p in baseline['pages'] if p['source']['sha256']==snap['source_sha256']),None)
            stable=[a for a in baseline['stable_assets'] if a['source_sha256']==snap['source_sha256']]
            update(stage='perceiving',page=i+1,branches=['text','asset','surface'])
            existing={};replay_ocr=None
            if request.get('perception_replay'):
                from .recovery.evidence_replay import load as replay_evidence
                existing,replay_ocr=replay_evidence(request['perception_replay'],snap)
                write(folder/'perception-replay.json',{'manifest_sha256':digest(request['perception_replay']),
                    'source_sha256':snap['source_sha256'],'fresh_perception':False,'branches':sorted(existing)})
            if request.get('reuse_perception_from'):
                previous=Path(request['reuse_perception_from']).resolve()
                import sqlite3
                from .parallel_contracts import TextEvidence,SurfaceEvidence,AssetEvidence
                prior=json.loads((previous/'provenance.json').read_text(encoding='utf-8'))
                if prior['moma_model']!=provider.model:raise ValueError('cached evidence model mismatch')
                with sqlite3.connect(previous/'state/parallel.sqlite') as db:
                    rows=db.execute("SELECT branch,payload FROM tasks WHERE state='COMPLETED' AND branch IN ('text','surface','asset')").fetchall()
                types={'text':TextEvidence,'surface':SurfaceEvidence,**({'asset':AssetEvidence} if request.get('reuse_assets') else {})}
                for branch,payload in rows:
                    if branch not in types:continue
                    ref=Path(json.loads(payload)['evidence']);value=types[branch].model_validate(json.loads(ref.read_text(encoding='utf-8')))
                    if value.source_sha256==snap['source_sha256'] and value.image_size==snap['image_size']:existing[branch]=value
                for file in previous.glob('page-*/barrier-1.json'):
                    cached=SceneGraph.model_validate(json.loads(file.read_text(encoding='utf-8')))
                    if cached.source_sha256==snap['source_sha256']:
                        for branch,value in [('text',cached.text),('surface',cached.surface),*([('asset',cached.asset)] if request.get('reuse_assets') else [])]:
                            if value.method_version!='r10-fallback/1':existing[branch]=value
                        break
            branches,failures=perceive(snap,store,provider,fonts,cache,models,saved_layers=saved,stable=stable,
                                      existing=existing,targets=set(('text','asset','surface'))-set(existing))
            try:ocr=replay_ocr if replay_ocr is not None else local_ocr(snap,cache,models)
            except Exception:ocr={'lines':[]}
            graph=fuse(snap,branches,ocr)
            from .parallel_typography import calibrate
            from .parallel_text_anchor import anchor
            import numpy as np
            from PIL import Image
            pixels=np.array(Image.open(snap['path']).convert('RGB'))
            with lease('cpu'):
                graph=anchor(graph,ocr,pixels,fonts,store)
                graph=calibrate(graph,pixels,fonts,store,ocr=ocr,stable=stable)
                graph=anchor(graph,ocr,pixels,fonts,store,fitted=True)
            write(folder/'barrier-1.json',graph.model_dump())
            try:
                directory,_=layer_evidence(snap,cache,saved)
                from .object_extract import load_page
                layers=load_page(directory)
            except Exception:layers=None
            stage=folder/'staged'
            if stage.exists():
                from uuid import uuid4
                stage=folder/('staged-'+uuid4().hex[:8])
            with lease('cpu'):ir,report=build(graph,snap,fonts,stage,layers=layers,stable=stable,strategy=route,pixel_cache=cache/'pixels',
                text_candidates=candidate_engine!='r12-regional')
            difficult=[r['id'] for r in report if r['state']=='FALLBACK' and r['id'] in {o.id for o in graph.asset.objects}]
            if difficult and route in ('auto','sam3','sam3.1'):
                from .object_stage import infer_sam
                from .object_extract import load_sam
                samplan=folder/'sam-plan.json'
                write(samplan,{'schema':'LogicalObjects/2','source_sha256':snap['source_sha256'],'image_size':snap['image_size'],
                    'objects':[o.model_dump() for o in graph.asset.objects],'limitations':[]})
                try:
                    samdir=infer_sam(snap['path'],samplan,folder/'sam','sam3' if route=='auto' else route,difficult)
                    masks={}
                    for obj in graph.asset.objects:
                        if obj.id not in difficult:continue
                        candidates,error=load_sam(samdir,obj,snap['source_sha256'],digest(samplan),tuple(snap['image_size']),'sam3' if route=='auto' else route,individual=True)
                        if not error:masks[obj.id]=candidates
                    if masks:
                        graph.selected={};graph.transactions=[]
                        with lease('cpu'):ir,report=build(graph,snap,fonts,folder/'staged-sam',layers=layers,stable=stable,strategy=route,sam_candidates=masks,pixel_cache=cache/'pixels',
                            text_candidates=candidate_engine!='r12-regional')
                except Exception as exc:write(folder/'sam-failure.json',{'reason':type(exc).__name__+':'+str(exc)[:300]})
            # A visually faithful raster fallback must not suppress directed
            # repairs of known extraction/planning failures before Go reviews.
            from .parallel_repair import local_findings,repair
            findings=local_findings(graph,report,stable)
            if findings and route!='off' and candidate_engine not in ('r12-regional','r12-shadow'):
                try:
                    graph,ir,report=repair(graph,snap,findings,store,provider,fonts,cache,models,
                        folder/'local-repair-1',baseline,route)
                except Exception as exc:
                    write(folder/'local-repair-failure.json',{'reason':type(exc).__name__+':'+str(exc)[:300]})
            store.task('scene-'+snap['source_sha256'],'scene','STAGED',{'ir':str(ir),'failures':failures},
                       [key({'source':snap['source_sha256'],'branch':b,'version':'r10/1','repair':None}) for b in ('text','asset','surface')])
            return graph,ir,report,failures
    def page(i):
        try:return _page(i)
        except Exception as exc:
            from .parallel_contracts import TextEvidence,AssetEvidence,SurfaceEvidence
            snap=snaps[i];base={k:snap[k] for k in ('source_sha256','image_size','transform')};base['method_version']='r10-fallback/1'
            reason=type(exc).__name__+':'+str(exc)[:500]
            branches={'text':TextEvidence.model_validate({'schema':'TextEvidence/1',**base,'limitations':[reason]}),
                'asset':AssetEvidence.model_validate({'schema':'AssetEvidence/1',**base,'limitations':[reason]}),
                'surface':SurfaceEvidence.model_validate({'schema':'SurfaceEvidence/1',**base,'background':'FFFFFF','limitations':[reason]})}
            graph=fuse(snap,branches,{'lines':[]})
            from uuid import uuid4
            ir,report=build(graph,snap,fonts,job/f'page-{i+1:02d}'/('branch-fallback-'+uuid4().hex[:8]),strategy='off')
            store.task('failed-page-'+snap['source_sha256'],'scene','FALLBACK',{'reason':reason,'ir':str(ir)})
            return graph,ir,report,{'page':reason}
    with ThreadPoolExecutor(max_workers=2) as pool:pages=list(pool.map(page,range(len(snaps))))
    graphs=[p[0] for p in pages];irs=[p[1] for p in pages];reports=[p[2] for p in pages]
    if candidate_engine in ('r12-regional','r12-shadow'):
        from .recovery.product_pipeline import run as regional_run
        return regional_run(job,request,snaps,graphs,irs,baseline,fonts,store,expected,
            compiler=compiler,reviewer=review_provider,office=office,progress=update,shadow=candidate_engine=='r12-shadow')
    if candidate_engine=='r11-shadow':
        for i,(graph,snap) in enumerate(zip(graphs,snaps)):
            try:
                with lease('cpu'):
                    shadow_discover(graph,snap,fonts,job/f'page-{i+1:02d}',store,
                        [a for a in baseline['stable_assets'] if a['source_sha256']==snap['source_sha256']])
            except Exception as exc:
                write(job/f'page-{i+1:02d}'/'r11-shadow-failure.json',{'status':'UNKNOWN','reason':type(exc).__name__})
    # Persist candidate-region state independently of final quality/publication.
    for _,ir,_,_ in pages:
        receipt=Path(ir).parent/'region-transactions.json'
        if receipt.exists():
            for region in json.loads(receipt.read_text(encoding='utf-8')):
                store.task(region['region_id'],'region',region['status'],
                    {**region,'ir':str(ir),'quality':'REVIEW_REQUIRED','human_acceptance':False},region['dependencies'])

    def compile_set(paths,name):
        update(stage='compiling')
        output=job/name
        if output.exists():
            from uuid import uuid4
            output=job/(name+'-resume-'+uuid4().hex[:8])
        with lease('compiler'):return compiler(paths,output,[effective_texts(g,p) for g,p in zip(graphs,paths)])
    reviews=[None]*len(snaps);failed=set();review_history=[]
    for round_no in range(3):
        try:artifacts=compile_set(irs,f'compile-{round_no}')
        except Exception as exc:
            write(job/f'compile-failure-{round_no}.json',{'type':type(exc).__name__,'reason':str(exc)[:300]})
            for i,g in enumerate(graphs):
                graphs[i]=safe_graph(g)
                irs[i],reports[i]=build(graphs[i],snaps[i],fonts,job/f'page-{i+1:02d}'/f'compile-fallback-{round_no}',strategy='off')
            artifacts=compile_set(irs,f'compile-safe-{round_no}')
        previews=sorted(Path(artifacts['preview']).glob('*.png'));office_pass=False;office_receipt=None
        office_dir=job/f'office-{round_no}'
        if office_dir.exists():
            from uuid import uuid4
            office_dir=job/(f'office-{round_no}-resume-'+uuid4().hex[:8])
        try:
            from .parallel_office import inspect
            with lease('office'):office_receipt=(office or inspect)(artifacts['pptx'],irs,office_dir)
            if office_receipt.get('status')!='PASS_APP_AUTOMATED':raise ValueError('Office probe failed')
            previews=[office_dir/f'page-{i+1:02d}-before.png' for i in range(len(snaps))];office_pass=True
        except Exception as exc:write(job/f'office-failure-{round_no}.json',{'type':type(exc).__name__,'reason':str(exc)[:300]})
        failed=set()
        for i,(snap,g,ir,preview) in enumerate(zip(snaps,graphs,irs,previews)):
            update(stage='reviewing',page=i+1,round=round_no)
            ctx={'source_sha256':snap['source_sha256'],'plan_sha256':digest(ir),'preview_sha256':digest(preview),
                'scene':json.loads(Path(ir).read_text(encoding='utf-8')),'transactions':reports[i],
                'raster_remainder_is_explicit_fallback':True,'office_pass':office_pass}
            try:
                details,refs=panels(snap['path'],preview,ir,office_dir,i+1,job/f'review-panels-{round_no}-{i+1:02d}') if office_pass else ([],[])
                ctx['actual_edit_evidence']=refs
                r=store.call('final-review',review_provider,REVIEW,ctx,[snap['path'],str(preview),*details],Review,lease('go'))
                if any(getattr(r,k)!=ctx[k] for k in ('source_sha256','plan_sha256','preview_sha256')):raise ValueError('review binding mismatch')
                reviews[i]=r.model_dump()
                if effective_findings(r):failed.add(i)
            except Exception as exc:
                reviews[i]={'quality':'UNKNOWN','reason':type(exc).__name__+':'+str(exc)[:200]};failed.add(i)
        if not office_pass:failed.update(range(len(snaps)))
        review_history.append({'round':round_no,'reviews':list(reviews),'failed_pages':[i+1 for i in sorted(failed)]})
        write(job/f'quality-round-{round_no}.json',review_history[-1])
        if not failed or round_no==2:break
        repaired=False
        for i in sorted(failed):
            findings=reviews[i].get('findings',[])
            if not findings:continue  # Unknown transport/Office is not a planning defect.
            try:
                from .parallel_repair import repair
                repaired_graph,repaired_ir,repaired_report=repair(graphs[i],snaps[i],findings,store,provider,fonts,cache,models,
                    job/f'page-{i+1:02d}'/f'repair-{round_no+1}',baseline,route)
                if digest(repaired_ir)==digest(irs[i]):
                    write(job/f'repair-no-change-{round_no}-{i+1:02d}.json',{'ir_sha256':digest(repaired_ir),'parent_retained':True})
                    continue
                graphs[i],irs[i],reports[i]=repaired_graph,repaired_ir,repaired_report
                repaired=True
            except Exception as exc:write(job/f'repair-failure-{round_no}-{i+1:02d}.json',{'reason':type(exc).__name__+':'+str(exc)[:300]})
        if not repaired:break
    # Reject a whole staged page transaction rather than mixing unverified halves.
    # Original source is the parent; no serial planner is invoked on this path.
    if failed:
        for i in failed:
            irs[i],_=build(safe_graph(graphs[i]),snaps[i],fonts,job/f'page-{i+1:02d}'/'qa-fallback',strategy='off')
        artifacts=compile_set(irs,'compile-accepted-and-safe')
        previews=sorted(Path(artifacts['preview']).glob('*.png'))
        office_receipt=None
        try:
            from .parallel_office import inspect
            with lease('office'):office_receipt=(office or inspect)(artifacts['pptx'],irs,job/'office-final')
            previews=[job/'office-final'/f'page-{i+1:02d}-before.png' for i in range(len(snaps))]
        except Exception as exc:
            write(job/'office-final-failure.json',{'reason':type(exc).__name__+':'+str(exc)[:300]})
        valid_final=office_receipt and office_receipt.get('status')=='PASS_APP_AUTOMATED'
        preview_changed=valid_final and any(i not in failed and
            reviews[i].get('preview_sha256')!=digest(previews[i]) for i in range(len(snaps)))
        if not valid_final or preview_changed:
            # No old QA receipt may certify a changed/unknown final package.
            # Revert the whole task to its immutable original representations.
            failed=set(range(len(snaps)));office_receipt=None
            for i,g in enumerate(graphs):
                irs[i],_=build(safe_graph(g),snaps[i],fonts,job/f'page-{i+1:02d}'/'final-safe-source',strategy='off')
            artifacts=compile_set(irs,'compile-final-safe-source')
            previews=sorted(Path(artifacts['preview']).glob('*.png'))
            try:
                from .parallel_office import inspect
                with lease('office'):office_receipt=(office or inspect)(artifacts['pptx'],irs,job/'office-final-safe-source')
                if office_receipt.get('status')=='PASS_APP_AUTOMATED':
                    previews=[job/'office-final-safe-source'/f'page-{i+1:02d}-before.png' for i in range(len(snaps))]
            except Exception as exc:write(job/'office-safe-source-failure.json',{'reason':type(exc).__name__})
    delivery=job/'delivery';delivery.mkdir(exist_ok=True)
    shutil.copyfile(artifacts['pptx'],delivery/'result.pptx')
    for i,p in enumerate(previews,1):shutil.copyfile(p,delivery/f'page-{i:02d}.png')
    reconstructed=[]
    for i,ir in enumerate(irs):
        nodes=json.loads(Path(ir).read_text(encoding='utf-8'))['nodes']
        if i not in failed and any(n['id']!='source-plate' for n in nodes):reconstructed.append(i+1)
    accepted=len(reconstructed)
    write(job/'qa.json',{'reviews':reviews,'history':review_history,'reverted_pages':[i+1 for i in sorted(failed)],'office':office_receipt,
        'status':'REVIEW_REQUIRED','human_acceptance':False})
    (delivery/'结果说明.txt').write_text(f'三分支独立重建 r10；共 {len(snaps)} 页，自动门禁保留重建 {accepted} 页，其余保持完整原图安全表示。\n人工验收未确认；尚未开放 parallel_auto。详细证据保存在后台。\n',encoding='utf-8-sig')
    result={'status':'REVIEW_REQUIRED','pages':len(snaps),'pipeline_mode':pipeline_mode,'human_acceptance':False,'generation':expected+1,
        'reconstructed_pages':accepted,'reconstructed_page_numbers':reconstructed,
        'raster_fallback_pages':[i+1 for i in range(len(snaps)) if i+1 not in reconstructed],
        'artifacts':{p.name:digest(p) for p in delivery.iterdir() if p.is_file()},'slide_irs':[str(p) for p in irs]}
    # All pages, exact assets, QA and delivery enter one immutable manifest.
    files={str(p.resolve()):digest(p) for p in [*irs,job/'qa.json',*delivery.iterdir()] if p.is_file()}
    for irfile in irs:
        ir=json.loads(Path(irfile).read_text(encoding='utf-8'))
        for a in ir['assets']:
            p=(Path(irfile).parent/a['path']).resolve();files[str(p)]=digest(p)
        for name in ('scene.json','transactions.json','ownership.json','ownership.npz','build-binding.json','source.png','region-transactions.json','editable-mask.npz'):
            p=Path(irfile).parent/name
            if p.is_file():files[str(p.resolve())]=digest(p)
    for irfile in irs:
        for p in Path(irfile).parent.glob('*-owner.npz'):files[str(p.resolve())]=digest(p)
    for p in [job/'request.json',job/'provenance.json',job/'fonts.json',*job.glob('quality-round-*.json'),
              *job.glob('office*/report.json'),*job.glob('snapshots/*.png')]:
        if p.is_file():files[str(p.resolve())]=digest(p)
    for g in graphs:
        for evidence in (g.text,g.asset,g.surface):
            for ref in evidence.references:
                p=Path(ref).resolve()
                if p.is_file() and (p.is_relative_to(job) or p.is_relative_to(cache)):
                    files[str(p)]=digest(p)
    stable_assets=[]
    for i,irfile in enumerate(irs):
        if i+1 not in reconstructed:continue
        ir=json.loads(Path(irfile).read_text(encoding='utf-8'));assets={a['id']:a for a in ir['assets']}
        definitions={o.id:o for o in graphs[i].asset.objects}
        for n in ir['nodes']:
            if n['type']!='image' or n['id'] not in definitions:continue
            a=assets[n['asset_id']];file=(Path(irfile).parent/a['path']).resolve()
            stable_assets.append({'source_sha256':snaps[i]['source_sha256'],'id':n['id'],'bbox':n['bbox'],
                'path':str(file),'sha256':a['sha256'],'definition':definitions[n['id']].model_dump(),
                'automatic_quality':'PASS','human_acceptance':False,'pixel_reference':None})
    generation=store.publish({'schema':'SceneCommit/1','files':files,'result':result,'stable_assets':stable_assets},expected)
    # This is a recoverable index, never a second publishing pointer.
    try:catalog.register(job)
    except Exception as exc:write(job/'catalog-recovery-required.json',{'reason':type(exc).__name__})
    result['generation']=generation;write(job/'result.json',result);update(stage='complete',status=result['status'])
    return result


if __name__=='__main__':
    job=Path(sys.argv[1])
    try:run(job)
    except Exception as exc:
        write(job/'result.json',{'status':'FAILED','error':type(exc).__name__+':'+str(exc)[:300],'human_acceptance':False});raise
