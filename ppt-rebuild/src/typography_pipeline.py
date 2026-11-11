"""Standalone, resumable MoMA -> typography -> fixed PPTX compiler workflow."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from PIL import Image,ImageDraw,ImageEnhance
from typography_plan import read,write,digest,targets,font_manifest,page_plan,global_plan
from typography_fit import propose,revision,choose,verify_selected,assert_frozen
from rebuild_pipeline import ArtifactBackend,REPO,ROOT,compile_ir,pixels,stamp
from surface_rebuild import rebuild_page
from surface_acceptance import verify_native_shape_properties,check_page
from text_repair_acceptance import verify_styles,delete_deck_text,edit_text_copy
from reconstruction import intersect
from ocr_engine import detect

PILOT=['01','03','04','10']


def verify_inputs(batch):
    for page in batch['pages']:
        irfile=Path(page['ir']); ir=read(irfile)
        if digest(irfile)!=page['ir_sha256']: raise ValueError('Baseline IR changed')
        if read(irfile.parent/'authoring.json')['ir_sha256']!=digest(irfile): raise ValueError('Authoring binding changed')
        if digest(page['pptx'])!=page['pptx_sha256']: raise ValueError('Baseline PPTX changed')
        if digest(irfile.parent/ir['source_image'])!=ir['source_sha256']: raise ValueError('Source changed')
        for asset in ir['assets']:
            if digest(irfile.parent/asset['path'])!=asset['sha256']: raise ValueError('Asset changed')


def checkpoint_valid(page):
    return digest(page['ir'])==page['ir_sha256'] and digest(page['pptx'])==page['pptx_sha256']


def surface_baseline(r5,r3,reviews,output):
    """Rerun background/card reconstruction before fitting against its clean plate.

    Surface fitting consumes frozen source text evidence, never new font colors.
    Doing it first lets final glyph comparisons share exactly the same background.
    """
    file=Path(output)/'surface-baseline.json'
    reviewfile=Path(reviews)/(r5['page'][:2]+'.json')
    inputs={'r3_ir_sha256':digest(r3['ir']),'review_sha256':digest(reviewfile)}
    if file.exists():
        record=read(file)
        if record['inputs']!=inputs or not checkpoint_valid(record['page']): raise ValueError('Surface checkpoint changed')
        return record['page']
    result=rebuild_page(r3,read(reviewfile),Path(output)/'surface-baseline')
    result.update(ir_sha256=digest(result['ir']),pptx_sha256=digest(result['pptx']))
    write(file,{'inputs':inputs,'page':result})
    return result


def shared_styles(plans,global_result,page):
    lookup={f'{pid}:{s["id"]}':s for pid,p in plans.items() for s in p['styles']}
    mapping={}
    for group in global_result['groups']:
        s=lookup[group['canonical']]
        for member in group['members']:
            pid,sid=member.split(':',1)
            if pid!=page: continue
            original=lookup[member]
            # Preserve clear source hierarchy; model grouping cannot silently
            # erase real emphasis or unite materially different source sizes.
            compatible=(abs(original['font_size_px']-s['font_size_px'])<=max(1,original['font_size_px']*.06) and
                        max(abs(int(original['color'][i:i+2],16)-int(s['color'][i:i+2],16)) for i in (0,2,4))<=16)
            mapping[sid]=s if compatible else original
    return mapping


def process_page(r5,r3,plan,manifest,reviews,output,canonical=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    plan_hash=__import__('hashlib').sha256(json.dumps({'plan':plan,'canonical':canonical},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    done=output/'result.json'
    if done.exists():
        record=read(done)
        if record['plan_hash']!=plan_hash or not checkpoint_valid(record): raise ValueError('Typography checkpoint changed')
        return record
    baseline=surface_baseline(r5,r3,reviews,output)
    ir=read(baseline['ir']); size=ir['canvas']['source_px']
    # PagePlan binds r5. Rerunning the deterministic background stage may change
    # masks/shapes, but must not alter the frozen text or paragraph identifiers.
    if targets(ir)!=targets(read(r5['ir'])): raise ValueError('Surface stage changed frozen typography')
    source=pixels(Path(baseline['ir']).parent/ir['source_image'],size)
    naked=pixels(baseline['deleted_text_preview'],size)
    old=pixels(baseline['preview'],size)
    attempt=output/('fit-'+stamp()); attempt.mkdir()
    candidate,fit_records=propose(ir,plan,manifest,source,naked,canonical)
    write(attempt/'local-fit.json',fit_records)
    proposed=revision(baseline['ir'],candidate,attempt/'candidate',{'stage':'local_fit','plan_hash':plan_hash})
    backend=ArtifactBackend(REPO,600)
    print('[typography] candidate',r5['page'],flush=True)
    compiled=compile_ir(backend,proposed)
    verify_styles(compiled['pptx'],candidate)
    new=pixels(compiled['preview'],size)
    selected,decisions=choose(ir,candidate,source,old,new,naked,manifest)
    final_ir=revision(baseline['ir'],selected,attempt/'selected',{'stage':'final_render_selection','decisions':decisions})
    final=compile_ir(backend,final_ir)
    verified=verify_selected(ir,selected,source,old,pixels(final['preview'],size),naked,decisions)
    verify_styles(final['pptx'],selected)
    verify_native_shape_properties(final['pptx'],selected)
    write(attempt/'selection.json',{'objects':decisions,'verification':verified})
    # Store complete candidates, actual previews and explicit baseline exceptions.
    report={**baseline,**final,'ir':str(final_ir.resolve()),'directory':str(Path(baseline['directory']).resolve()),
            'typography_directory':str(attempt.resolve()),'plan_hash':plan_hash,
            'ir_sha256':digest(final_ir),'pptx_sha256':digest(final['pptx']),
            'baseline_ir':baseline['ir'],'baseline_preview':baseline['preview'],
            'r5_ir':r5['ir'],'r5_preview':r5['preview'],'typography':verified,
            'decisions':decisions,'structure_review':plan['structure_review'],
            'status':'REVIEW_REQUIRED','powerpoint_checked':False,'wps_checked':False,'human_acceptance':False}
    write(done,report)
    print('[typography] checked',r5['page'],'improved',verified['improved'],flush=True)
    return report


def audit(batch,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    backend=ArtifactBackend(REPO,600)
    deleted=output/'all-native-text-deleted.pptx'
    delete_deck_text(batch['deck']['pptx'],deleted,batch['pages'])
    rendered=backend.render(deleted,output/'deleted-render')
    reports=[]
    for number,p in enumerate(batch['pages'],1):
        ir=read(p['ir']); size=ir['canvas']['source_px']
        styles=verify_styles(batch['deck']['pptx'],ir,number)
        shapes=verify_native_shape_properties(batch['deck']['pptx'],ir,number)
        image=rendered['pages'][number-1]['path']
        same=float(np.abs(pixels(image,size).astype(float)-pixels(p['deleted_text_preview'],size)).mean())
        if same>.1: raise ValueError('Final deck deletion background changed')
        contrast=output/(p['page'][:2]+'-contrast.png')
        ImageEnhance.Contrast(Image.open(image).convert('RGB')).enhance(3).save(contrast)
        hits=[]
        boxes=[e['bbox'] for e in ir['evidence'] if e.get('kind')=='text']
        if not boxes: boxes=[n['bbox'] for n in ir['nodes'] if n['type']=='text']
        for check in [image,contrast]:
            for hit in detect(check,min_score=.45):
                if any(intersect(hit.bbox,b)>.5*hit.bbox[2]*hit.bbox[3] for b in boxes):
                    hits.append({'text':hit.text,'bbox':hit.bbox})
        if hits:
            write(output/(p['page'][:2]+'-ocr-failure.json'),hits)
            raise ValueError('OCR detected old text after final deletion')
        row={'page':p['page'],'style_checks':len(styles),'shape_checks':len(shapes),
             'deleted_preview':image,'deleted_background_mae':same,'ocr_hits':hits}
        if p['page'][:2] in PILOT:
            choices=[n for n in ir['nodes'] if n['type']=='text']
            node=(max(choices,key=lambda n:n['font']['size_pt']) if p['page'][:2]=='01'
                  else max(choices,key=lambda n:len(n.get('paragraphs',[]))))
            edited=output/(p['page'][:2]+'-text-edited.pptx')
            edit_text_copy(p['pptx'],edited,node['id'])
            er=backend.render(edited,output/(p['page'][:2]+'-edit-render'))
            delta=float(np.abs(pixels(er['pages'][0]['path'],size).astype(float)-pixels(p['preview'],size)).mean())
            if delta<=0: raise ValueError('Actual text edit produced no rendering change')
            row['text_edit']={'id':node['id'],'preview':er['pages'][0]['path'],'difference':delta}
            if p['page'][:2]=='03':
                emphasized=next(n for n in choices if any(q.get('runs') for q in n.get('paragraphs',[])))
                target=output/'03-emphasis-edited.pptx'
                edit_text_copy(p['pptx'],target,emphasized['id'])
                proof=backend.render(target,output/'03-emphasis-edit-render')
                changed=float(np.abs(pixels(proof['pages'][0]['path'],size).astype(float)-pixels(p['preview'],size)).mean())
                if changed<=0: raise ValueError('Emphasized number edit did not render')
                row['emphasis_edit']={'id':emphasized['id'],'preview':proof['pages'][0]['path'],'difference':changed}
        if p['page'][:2] in ['03','04','10']:
            row['card_edit']=check_page(p,output/(p['page'][:2]+'-card'),backend)
        reports.append(row)
    result={'status':'PASS_COMPUTATIONAL_GATES','pages':reports,'deleted_pptx':str(deleted),
            'powerpoint_checked':False,'wps_checked':False,'human_acceptance':False}
    write(output/'report.json',result);return result


def package(batch,audited,run):
    run=Path(run); out=run/'delivery';out.mkdir(exist_ok=False)
    name='徐州高新区AI交流_字体修正_r6_12页_待人工审阅.pptx'
    shutil.copy2(batch['deck']['pptx'],out/name)
    shutil.copy2(audited['deleted_pptx'],out/'删除全部原生文字_r6.pptx')
    for f in ['font-manifest.json','api-budget.json','run-manifest.json']:
        shutil.copy2(run/f,out/f)
    shutil.copytree(run/'plans',out/'plans',ignore=shutil.ignore_patterns('messages.json'))
    shutil.copytree(run/'global-plan',out/'global-plan',ignore=shutil.ignore_patterns('messages.json'))
    all_rows=[]; gallery=[]
    for index,(p,a) in enumerate(zip(batch['pages'],audited['pages']),1):
        directory=out/'pages'/f'{index:02d}'; directory.mkdir(parents=True)
        ir=read(p['ir'])
        for file,label in [(Path(p['ir']).parent/ir['source_image'],'source.png'),(p['r5_preview'],'r5.png'),
                           (p['preview'],'r6.png'),(a['deleted_preview'],'deleted.png'),(p['pptx'],'page.pptx')]: shutil.copy2(file,directory/label)
        shutil.copytree(Path(p['ir']).parent,directory/'revision',ignore=shutil.ignore_patterns('outputs'))
        shutil.copy2(Path(p['typography_directory'])/'local-fit.json',directory/'local-fit.json')
        write(directory/'selection.json',p['decisions'])
        write(directory/'structure-review.json',p['structure_review'])
        crops=directory/'text-comparisons';crops.mkdir()
        for j,d in enumerate(p['decisions']):
            all_rows.append({'page':index,'id':d['id'],'status':'已改善' if d['selected'] else '保留基线_待复核',
                             'baseline_mae':round(d['baseline']['mae'],4),'candidate_mae':round(d['candidate']['mae'],4),
                             'reason':','.join(d['reasons'])})
            box=d['bbox_xyxy']; pieces=[Image.open(directory/f).convert('RGB').crop(box) for f in ['source.png','r5.png','r6.png']]
            w=max(i.width for i in pieces);h=max(i.height for i in pieces)
            sheet=Image.new('RGB',(w*3,h+22),'white');draw=ImageDraw.Draw(sheet)
            for n,(im,label) in enumerate(zip(pieces,['source','r5','r6'])): sheet.paste(im,(n*w,22));draw.text((n*w+2,2),label,fill='black')
            sheet.save(crops/f'{j:03d}.png')
        gallery.append(f'<h2>第{index}页</h2><div class="row">'+''.join(f'<figure><figcaption>{label}</figcaption><img src="pages/{index:02d}/{file}"></figure>' for file,label in [('source.png','原图'),('r5.png','r5'),('r6.png','r6'),('deleted.png','删除文字')])+'</div>')
    with (out/'逐对象字体审阅.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    (out/'逐页对照.html').write_text('<meta charset="utf-8"><style>body{font-family:system-ui}.row{display:grid;grid-template-columns:1fr 1fr}figure{margin:8px}img{width:100%}</style><h1>原图 / r5 / r6 / 去字检查</h1><p>自动检查不等于PowerPoint/WPS或人工验收。</p>'+''.join(gallery),encoding='utf8')
    totals=Counter(r['status'] for r in all_rows)
    summary={'schema':'TypographyDelivery/1','status':'REVIEW_REQUIRED','deck':name,'deck_sha256':digest(out/name),
             'counts':dict(totals),'pages':len(batch['pages']),'calls':len(read(run/'api-budget.json')['calls']),
             'powerpoint_checked':False,'wps_checked':False,'human_acceptance':False,
             'source_layerd_rerun':False,'model':'qwen/qwen3.8-max'}
    write(out/'delivery-index.json',summary); write(out/'actual-pptx-checks.json',audited)
    (out/'审阅说明.md').write_text('# r6 字体专项审阅版\n\n使用MoMA千问逐页规划及跨页样式归并，本地渲染校准并通过固定编译器生成。原文、位置、分组保持冻结；结构建议未自动应用。\n\n'+
        f'对象统计：{dict(totals)}。保留基线不算已修复；详见逐对象字体审阅.csv和局部三栏对照。\n\n'+
        '背景与卡片重新运行后再校准字体，使候选使用相同干净背景。PNG复杂图标继续保留。字体候选与统计不能证明原字体身份。PowerPoint、WPS及人工验收均待完成。\n',encoding='utf8')
    archive=run/'4090_字体修正_r6_人工审阅包.zip'
    with ZipFile(archive,'x',compression=ZIP_DEFLATED) as z:
        for f in out.rglob('*'):
            if f.is_file(): z.write(f,str(f.relative_to(out.parent)))
    return summary


def run(args):
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=True)
    batch=read(args.baseline);verify_inputs(batch)
    source_batch=read(batch['input']);r3={p['page']:p for p in source_batch['pages']}
    baseline_hash=digest(args.baseline)
    inputs={'baseline':str(Path(args.baseline).resolve()),'baseline_sha256':baseline_hash,
            'pages':[{ 'page':p['page'],'ir_sha256':digest(p['ir']),'source_sha256':p['source_sha256']} for p in batch['pages']]}
    existing=output/'run-manifest.json'
    if existing.exists() and read(existing)['inputs']!=inputs: raise ValueError('Resume inputs changed')
    if not existing.exists():
        write(existing,{'schema':'TypographyRun/1','inputs':inputs,'code':{p.name:digest(p) for p in Path(__file__).parent.glob('typography_*.py')},
                        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                        'policy':'source fidelity; frozen content and geometry; no skill dependency'})
    texts=[t['text'] for p in batch['pages'] for t in targets(read(p['ir']))]
    manifest=font_manifest(texts)
    catalog=output/'font-manifest.json'
    if catalog.exists() and read(catalog)!=manifest: raise ValueError('Font environment changed')
    write(catalog,manifest)
    order=PILOT+[p['page'][:2] for p in batch['pages'] if p['page'][:2] not in PILOT]
    plans={};pilot=[]
    for prefix in order:
        p=next(p for p in batch['pages'] if p['page'].startswith(prefix))
        print('[typography] plan',prefix,flush=True)
        plans[prefix]=page_plan(p['ir'],manifest,output/'plans'/prefix,output/'api-budget.json')
        if prefix in PILOT and args.stage!='plan':
            pilot.append(process_page(p,r3[p['page']],plans[prefix],manifest,args.reviews,output/'pilot'/prefix))
        if args.stage=='pilot' and len(pilot)==len(PILOT): return {'pilot_pages':len(pilot),'status':'REVIEW_REQUIRED'}
    global_result=global_plan(plans,output/'global-plan',output/'api-budget.json')
    if args.stage=='plan': return {'planned_pages':len(plans),'global':True}
    results=[]
    for p in batch['pages']:
        prefix=p['page'][:2]
        results.append(process_page(p,r3[p['page']],plans[prefix],manifest,args.reviews,output/'pages'/prefix,shared_styles(plans,global_result,prefix)))
    from image2ppt.compiler import execute
    backend=ArtifactBackend(REPO,600)
    composed=execute([backend.node,ROOT/'compiler_bridge.mjs','compose',output/'deck',*[p['ir'] for p in results]],REPO,600)
    final={'schema':'TypographyBatch/1','directory':str(output),'input':str(Path(args.baseline).resolve()),'pages':results,'deck':composed,'status':'REVIEW_REQUIRED'}
    write(output/'batch.json',final)
    audited=audit(final,output/('acceptance-'+stamp()))
    return package(final,audited,output)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--reviews',default=str(ROOT/'configs/surface-reviews'))
    parser.add_argument('--stage',choices=['pilot','plan','all'],default='all')
    result=run(parser.parse_args());print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
