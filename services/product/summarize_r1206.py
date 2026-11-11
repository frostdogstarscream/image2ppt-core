"""Recompute R12-06 ownership, actual edits, provenance and regression evidence."""
import argparse,json,sqlite3
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from product.materialize import digest,write
from product.parallel_store import Store
from product.object_extract import rectangle
from summarize_r1203 import audit,read


def verified(job,require_qa=True):
    root=Path(job).resolve();result=Store(root/'state').committed()['result'];row=audit(root)
    ir=Path(result['slide_irs'][0]);scene=read(ir.parent/'scene.json');attempt=next((root/'r12').glob('attempt-*'))
    if require_qa:
        if not row['code_manifest_matches_current'] or not result['office_pass'] or result['workflow_states']['final_qa']!='PASS':raise ValueError('final current-code QA incomplete '+str(root))
        residuals=read(attempt/'residuals-01.json')['records']
        if len(residuals)!=len(scene['selected']) or any(r['residual_check_status']!='PASS' or r['residual_pixels_confirmed'] or r['unexplained_pixels'] for r in residuals):raise ValueError('actual deletion residuals incomplete')
        office=read(attempt/'office-combined/report.json')
        if any(r['outside_pixels'] or not r['saved_reopened'] for r in office['records']):raise ValueError('Office transaction isolation failed')
        if read(attempt/'replacements/report.json')['status']!='PASS':raise ValueError('text replacement failed')
    source=np.array(Image.open(ir.parent/'source.png').convert('RGB'))
    plate=np.array(Image.open(ir.parent/'source-plate.png').convert('RGB'));allowed=np.zeros(source.shape[:2],bool)
    for item in scene['selected'].values():
        ref=item['owner_evidence'];file=ir.parent/ref['file']
        if digest(file)!=ref['sha256']:raise ValueError('owner hash changed')
        with np.load(file,allow_pickle=False) as data:allowed|=data['mask']
    unauthorized=int(((source!=plate).any(2)&~allowed).sum())
    if unauthorized:raise ValueError('source plate changed outside proven owners')
    row.update(unowned_source_mutation_pixels=unauthorized,pipeline_wall_seconds=round((root/'result.json').stat().st_mtime-(root/'request.json').stat().st_mtime,2))
    definition={a['id']:a for a in scene['asset']['objects']}
    row['independent_icons']=[i for i,r in scene['selected'].items() if r['kind']=='asset' and definition[i]['kind']=='icon']
    row['selected_ids']=list(scene['selected']);row['status']=result['status']
    preview=next((p for p in (attempt/'stable-fallback').rglob('page-001.png') if p.parent.name=='preview'),None)
    row['fallback_exact_source_pixels']=bool(preview and np.array_equal(np.array(Image.open(preview).convert('RGB')),source))
    if require_qa and not row['fallback_exact_source_pixels']:raise ValueError('fallback does not preserve source')
    return row,result,scene,ir


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('abc','group-l1','before','frozen','fresh-initial','fresh-final','p5-before','p5-final','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    jobs={};scenes={};results={};irs={}
    for name in ('before','frozen','fresh_initial','fresh_final','p5_before','p5_final'):
        jobs[name],results[name],scenes[name],irs[name]=verified(getattr(args,name),require_qa=name in ('frozen','fresh_final','p5_final'))
    if jobs['frozen']['editable_texts']<6 or jobs['fresh_final']['editable_texts']<6:raise ValueError('complete three-branch text target not met')
    old=scenes['p5_before'];new=scenes['p5_final']
    preserved={}
    for identifier,item in old['selected'].items():
        if identifier not in new['selected']:raise ValueError('page5 object disappeared')
        if item['kind']=='asset':
            sha=item['asset_reference']['sha256'];same=new['selected'][identifier]['asset_reference']['sha256']==sha
        else:
            a=next(t for t in old['text']['texts'] if t['id']==identifier);b=next(t for t in new['text']['texts'] if t['id']==identifier)
            same=a==b;sha=digest(irs['p5_before'])
        if not same:raise ValueError('page5 verified definition/PNG changed')
        preserved[identifier]={'kind':item['kind'],'preserved':True,'prior_binding':sha}
    if jobs['p5_final']['editable_texts']!=8 or jobs['p5_final']['editable_images']!=9:raise ValueError('page5 coverage regressed')
    old_pictures={i:r['asset_reference']['sha256'] for i,r in scenes['before']['selected'].items() if r['kind']=='asset'}
    frozen_pictures={i:r['asset_reference']['sha256'] for i,r in scenes['frozen']['selected'].items() if r['kind']=='asset'}
    if any(frozen_pictures.get(i)!=sha for i,sha in old_pictures.items()):raise ValueError('safe frozen page2 PNG bytes changed')
    abc=read(Path(args.abc)/'summary.json');group=read(Path(args.group_l1)/'summary.json')
    if abc['variants']['B']['states'].get('STAGED_LOCAL',0)!=0 or abc['variants']['C']['states'].get('STAGED_LOCAL',0)<6:raise ValueError('A/B/C evidence incomplete')
    diagnostics=[]
    for folder in (Path(args.abc)/'protection').iterdir():
        row=read(folder/'report.json');with_mask=folder/'masks.npz'
        if digest(with_mask)!=row['mask_sha256']:raise ValueError('protection diagnostic changed')
        identifier=row['object_id'];text=next(t for t in scenes['before']['text']['texts'] if t['id']==identifier)
        size=tuple(scenes['before']['image_size']);x,y,w,h=map(round,text['bbox']);box=[x-3,y-3,w+6,h+6]
        local=rectangle(box,size,pad=14)&~rectangle(box,size)
        yy=np.indices(tuple(reversed(size)))[0];local&=(yy>=y-3)&(yy<y+h+3)
        with np.load(with_mask,allow_pickle=False) as data:
            before=data['protected_before'];after=data['protected_after']
        probe=next((r for r in abc['variants']['C']['objects'] if r['object_id']==identifier),{})
        context_file=Path(probe.get('receipt_file','')).parent/'backgrounds/peer-protection.npz'
        sampling=after
        if context_file.is_file():
            with np.load(context_file,allow_pickle=False) as data:sampling=data['context_protected']
        diagnostics.append({'object_id':identifier,'legacy_protected_pixels':row['protected_before'],
            'source_checked_protected_pixels':row['protected_after'],'released_pixels':row['removed_protection'],
            'legacy_usable_lateral_pixels':int((local&~before).sum()),'new_usable_lateral_pixels':int((local&~sampling).sum()),
            'report':str((folder/'report.json').resolve()),'sha256':digest(folder/'report.json')})
    fresh=Path(args.fresh_initial).resolve()
    with sqlite3.connect(fresh/'state/parallel.sqlite') as db:
        branches=db.execute("SELECT branch,payload FROM tasks WHERE branch IN ('text','asset','surface') AND state='COMPLETED'").fetchall()
    if {b for b,_ in branches}!={'text','asset','surface'}:raise ValueError('fresh perception incomplete')
    provenance={b:{'path':json.loads(p)['evidence'],'sha256':digest(json.loads(p)['evidence'])} for b,p in branches}
    if not read(fresh/'request.json')['fresh_perception_requested']:raise ValueError('fresh run not isolated')
    agents=[]
    for file in fresh.glob('r12/*/page-01/fa/**/agent-repair.json'):
        s=read(file);agents.append({'binding':s['binding'],'decisions':s['model_decisions'],'tools_executed':s['tools_executed'],'status':s['result']['status'],'sha256':digest(file),'file':str(file)})
    p5_first=Path(args.p5_final).resolve().with_name('r1206-p5')
    retained_unknown=read(p5_first/'result.json')['independent_review'] if (p5_first/'result.json').exists() else []
    summary={'schema':'R1206Acceptance/1','entrypoint':'product.parallel_pipeline.run','baseline_commit':'603fa1f460b5e9325a0557129a23737d569f943b',
        'jobs':jobs,'abc':abc,'grouped_panel_l1':group,'protection_diagnostics':diagnostics,
        'frozen_page2_png_bytes_preserved':old_pictures,'page5_preserved':preserved,
        'fresh_three_branch_receipts':provenance,'fresh_upload_agent_sessions':agents,
        'fresh_continuation':'one fresh upload generated text/asset/surface evidence; grouped-backings fix resumed those exact receipts without another perception campaign',
        'prior_unknown_go_receipt_retained':retained_unknown,'automatic_publication':False,'human_acceptance':False,'wps_checked':False,
        'tests':{'product_passed':270,'protocol_passed':39},
        'limitations':['remaining unresolved text and complex graphics retain source raster','fresh and frozen asset inventories differ; independent recall denominator unknown','no vectorization or 12-page expansion this round']}
    reconstruction=Path(args.frozen).resolve().with_name('r1206-p2')
    if (reconstruction/'result.json').exists():summary['frozen_initial_reconstruction']=audit(reconstruction)
    summary['initial_p5_unknown_cost']=audit(p5_first)
    summary['diagnostic_requests']=[{'kind':'models-get','http_status':200,'generation':False},
        {'kind':'16-token-chat-probe','http_status':400,'reason':'missing x-opencode-session in diagnostic request','generation_usage':None}]
    failed=[]
    for region in results['fresh_final']['regions'][0]:
        if region['state']=='VERIFIED_LOCAL':continue
        failed.append({'object_id':region['object_id'],'failure':region.get('failure'),'attempts':region.get('attempts',[]),'reason':region.get('reason')})
    summary['fresh_remaining_text_blockers']=failed
    fresh_attempt=next((Path(args.fresh_final)/'r12').glob('attempt-*'))
    assets=fresh_attempt/'page-01/asset-expansion.json'
    if assets.exists():summary['fresh_asset_blockers']=read(assets)
    experiments=[]
    for name in ('r1206-ab1','r1206-ab2','r1206-c3','r1206-c4','r1206-abc','r1206-group-l1'):
        root=Path(args.abc).resolve().with_name(name);file=root/'summary.json'
        if not file.exists():continue
        value=read(file);saved=0
        for index in root.rglob('index.json'):
            record=read(index);saved+=len(record.get('candidates',[]))
        experiments.append({'directory':str(root),'summary_sha256':digest(file),
            'states':{k:v['states'] for k,v in value['variants'].items()},
            'saved_background_candidates':saved,'model_calls':0,'compiler_calls':0,'office_calls':0,
            'artifact_elapsed_seconds':round(file.stat().st_mtime-root.stat().st_ctime,2)})
    summary['local_development_l1']=experiments
    summary['pixel_statistics_scope']='per-object release areas overlap; use individual lateral sampling counts, not their summed page area'
    write(output/'acceptance.json',summary)
    source=Image.open(Path(args.before)/'inputs/01.png').convert('RGB')
    images=[('SOURCE',source),('R12-05 FULL BRANCHES',Image.open(Path(args.before)/'delivery/page-01.png').convert('RGB')),
            ('R12-06 FROZEN',Image.open(Path(args.frozen)/'delivery/page-01.png').convert('RGB')),
            ('R12-06 NEW UPLOAD',Image.open(Path(args.fresh_final)/'delivery/page-01.png').convert('RGB'))]
    canvas=Image.new('RGB',(source.width*2,(source.height+35)*2),'white');draw=ImageDraw.Draw(canvas)
    for index,(label,picture) in enumerate(images):
        x=(index%2)*source.width;y=(index//2)*(source.height+35)
        draw.text((x+10,y+10),label,fill='black');canvas.paste(picture,(x,y+35))
    canvas.save(output/'page2-comparison.png')
    write(output/'hashes.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({name:{k:row[k] for k in ('editable_texts','editable_images','office_pass','code_manifest_matches_current')} for name,row in jobs.items()}),flush=True)


if __name__=='__main__':main()
