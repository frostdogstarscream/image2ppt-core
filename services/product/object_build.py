"""Run frozen plans through Alpha/SAM arms; export only explicitly selected arms."""
import argparse
from collections import Counter
from datetime import datetime
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
from PIL import Image
from product.object_extract import extract
from product.object_stage import effective_page
from product.materialize import write,digest,verify_package
from product.pipeline import ROOT
from image2ppt_vision.conversion import _compiler,_final_json,_compiled_artifacts


def review_rejections(run,records,root,planner):
    run=Path(run).resolve();result={}
    for record in records:
        i=record['page'];folder=run/f'page-{i:02d}'
        report=json.loads((folder/'extraction.json').read_text(encoding='utf-8'))
        review=json.loads((run/'go-review'/f'page-{i:02d}.json').read_text(encoding='utf-8'))
        binding=review['binding']
        if binding!={'source_sha256':digest(record['source']),'plan_sha256':digest(folder/'slideir.json'),'preview_sha256':digest(run/'delivery'/f'page-{i:02d}.png')}:
            raise ValueError('visual rollback review binding mismatch')
        if report['plan_sha256']!=digest(root/'plans'/planner/f'{i:02d}.json'):raise ValueError('visual rollback plan changed')
        rejected={}
        for obj in report['objects']:
            if obj['status']!='EXTRACTED':continue
            ids={obj['id'],'extracted-'+obj['id'],*obj['replaces']}
            findings=[f['description'] for f in review['findings'] if f['category']=='graphics' and ids.intersection(f.get('node_ids',[]))]
            if findings:rejected[obj['id']]=findings
        result[i]=rejected
    return result


def build(root,planner,route,compile_deck=True,reviewed_run=None):
    records=json.loads((root/'inputs.json').read_text(encoding='utf-8'))
    rejected=review_rejections(reviewed_run,records,root,planner) if reviewed_run else {}
    run=root/'runs'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+planner+'-'+route.replace('.','_'))
    run.mkdir(parents=True,exist_ok=False);reports=[];irs=[];pages=[]
    if reviewed_run:write(run/'visual-rollback.json',{'reviewed_run':str(reviewed_run),'objects':rejected})
    from product.contracts import Page
    for r in records:
        d=Path(r['directory']);planfile=root/'plans'/planner/f"{r['page']:02d}.json"
        if not planfile.is_file():raise RuntimeError(f"missing frozen {planner} plan page {r['page']}")
        sam=root/'sam'/planner/route/f"{r['page']:02d}"
        ir,report=extract(planfile,d/'layerd',r['baseline_ir'],run/f"page-{r['page']:02d}",route=route,sam_dir=sam if sam.is_dir() else None,rejected=rejected.get(r['page']))
        page=Page.model_validate(json.loads(Path(r['baseline_page']).read_text(encoding='utf-8')))
        pages.append(effective_page(page,ir));irs.append(str(ir));reports.append(report)
        print(json.dumps({'page':r['page'],'planner':planner,'route':route,'extracted':report['extracted'],'fallback':report['fallback']}),flush=True)
    summary={'planner':planner,'route':route,'status':'STRUCTURAL_CHECK_ONLY','human_acceptance':False,
        'pages':len(records),'objects':sum(len(r['objects']) for r in reports),'extracted':sum(r['extracted'] for r in reports),
        'fallback':sum(r['fallback'] for r in reports),
        'fallback_reasons':dict(Counter(reason for r in reports for o in r['objects'] for reason in o['reasons'])),
        'sam_objects':sum(o['status']=='EXTRACTED' and o['method'] in ('sam3','sam3.1') for r in reports for o in r['objects']),
        'sam_proposals':sum(o['method'] in ('sam3','sam3.1') for r in reports for o in r['objects']),
        'independent_context_per_page':planner=='moma','codex_required_at_runtime':planner=='terra',
        'semantic_completeness_measured':False,'powerpoint_checked':False,'opencode_review':'NOT_RUN'}
    if route!='alpha' and summary['sam_objects']==0:summary['sam_comparison_status']='NOT_RUN_NO_VALID_SAM_RESULTS'
    write(run/'summary.json',summary);write(run/'irs.json',irs)
    if compile_deck:
        compiler,node=_compiler(ROOT);builddir=run/'build';builddir.mkdir()
        with (builddir/'stdout.log').open('w',encoding='utf-8') as out,(builddir/'stderr.log').open('w',encoding='utf-8') as err:
            result=subprocess.run([str(node),str(ROOT/'services/product/compose_objects.mjs'),str(run/'irs.json'),str(builddir)],cwd=ROOT,stdout=out,stderr=err,timeout=1800,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if result.returncode:raise RuntimeError('fixed_compiler_failed: '+str(builddir/'stderr.log'))
        artifacts=_compiled_artifacts(_final_json((builddir/'stdout.log').read_bytes()),builddir,len(pages))
        verify_package(artifacts['pptx'],pages)
        delivery=run/'delivery';delivery.mkdir();shutil.copy2(artifacts['pptx'],delivery/f'{planner}-{route}-12pages.pptx')
        previews=sorted(Path(artifacts['preview']).glob('*.png'))
        if len(previews)!=12:raise ValueError('expected twelve actual-file previews')
        for i,p in enumerate(previews,1):shutil.copy2(p,delivery/f'page-{i:02d}.png')
        summary.update(status='REVIEW_REQUIRED',pptx=str(delivery/f'{planner}-{route}-12pages.pptx'),preview_origin='reimported_final_pptx')
        write(run/'summary.json',summary)
        write(run/'effective-pages.json',[p.model_dump() for p in pages])
        write(run/'artifacts.json',artifacts)
        rows=[]
        for i,r in enumerate(records,1):
            shutil.copy2(r['source'],delivery/f'source-{i:02d}.png')
            rows.append(f'<h2>第 {i} 页</h2><div class="row"><img src="source-{i:02d}.png"><img src="page-{i:02d}.png"></div>')
        (delivery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>对象提取复核</title><style>body{font:16px system-ui;background:#eee;margin:20px}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%}</style><h1>'+html.escape(planner+' / '+route)+'：源图与实际 PPTX 预览</h1><p>这是待复核实验稿，不代表 SAM 版本比较或人工验收通过。</p>'+''.join(rows),encoding='utf-8')
    write(root/f'latest-{planner}-{route}.json',{'run':str(run),'summary':summary})
    return run


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--planner',choices=['moma','terra'],required=True);p.add_argument('--route',choices=['alpha','sam3','sam3.1'],default='alpha');p.add_argument('--no-compile',action='store_true');p.add_argument('--reviewed-run',type=Path);a=p.parse_args()
    print(build(a.root.resolve(),a.planner,a.route,not a.no_compile,a.reviewed_run))
