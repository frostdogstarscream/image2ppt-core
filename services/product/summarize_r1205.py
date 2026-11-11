"""Audit R12-05 coverage, live tool evidence and actual Office packages."""
import argparse
import json
from pathlib import Path
import sqlite3
from PIL import Image,ImageDraw
from product.materialize import digest,write
from product.parallel_store import Store
from summarize_r1203 import audit,read


def checked(job, *, require_qa=True):
    job=Path(job).resolve()
    row=audit(job)
    result=Store(job/'state').committed()['result']
    scene=read(Path(result['slide_irs'][0]).parent/'scene.json')
    attempt=next((job/'r12').glob('attempt-*'))
    residual=read(attempt/'residuals-01.json')['records']
    if require_qa and (not row['office_pass'] or not all(r.get('passed') and not r.get('findings') for r in row['visual'])):
        raise ValueError('final Office and independent visual QA required')
    if len(residual)!=len(scene['selected']) or any(r['residual_check_status']!='PASS' or r['residual_pixels_confirmed'] or r['unexplained_pixels'] for r in residual):
        raise ValueError('incomplete actual deletion evidence')
    office=read(attempt/'office-combined/report.json')
    row['actual_final_office_actions']=len(office['records'])
    row['text_replacement_status']=read(attempt/'replacements/report.json')['status']
    if any(not r['saved_reopened'] or r['outside_pixels'] for r in office['records']):
        raise ValueError('real edits did not preserve siblings/outside pixels')
    reports=[read(p) for p in attempt.rglob('report.json')]
    row['actual_office_packages']=sum(r.get('status')=='PASS_APP_AUTOMATED' for r in reports)
    row['actual_replacement_packages']=sum(r.get('schema')=='TextEditProbe/1' for r in reports)
    row['pipeline_wall_seconds']=round((job/'result.json').stat().st_mtime-(job/'request.json').stat().st_mtime,2)
    row['independent_icons']=[i for i,v in scene['selected'].items() if v['kind']=='asset' and next(a for a in scene['asset']['objects'] if a['id']==i)['kind']=='icon']
    row['residual_records']=residual
    row['source_text_objects']=len(scene['text']['texts'])
    row['source_asset_definitions']=len(scene['asset']['objects'])
    row['independent_reference_recall']=None
    row['cost_ledger_scope']='compiler and Office ledger includes new asset batch; background_candidates counts text backgrounds; PNG candidates are separate'
    row['png_candidates']=len(list(attempt.glob('page-*/ax/*/c/candidate-*/candidate.json')))
    source=Image.open(job/'inputs/01.png').convert('RGB')
    fallback=next(p for p in (attempt/'stable-fallback').rglob('page-001.png') if p.parent.name=='preview')
    import numpy as np
    row['fallback_exact_source_pixels']=bool(np.array_equal(np.array(source),np.array(Image.open(fallback).convert('RGB'))))
    if not row['fallback_exact_source_pixels']:raise ValueError('stable raster fallback differs from source')
    return row,result,scene


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','expanded','final','smoke','smoke-final','text-agent','asset-agent','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    jobs={};results={};scenes={}
    for name in ('baseline','expanded','final','smoke','smoke_final'):
        jobs[name],results[name],scenes[name]=checked(getattr(args,name),require_qa=name!='smoke_final')
    if not all(jobs[k]['code_manifest_matches_current'] for k in ('final','smoke_final')):
        raise ValueError('final tested code differs from current product')
    before=scenes['baseline'];after=scenes['final']
    old_assets={i:r['asset_reference']['sha256'] for i,r in before['selected'].items() if r['kind']=='asset'}
    if any(after['selected'].get(i,{}).get('asset_reference',{}).get('sha256')!=sha for i,sha in old_assets.items()):
        raise ValueError('previous verified PNG bytes changed')
    old_texts={t['id']:t for t in before['text']['texts'] if before['selected'].get(t['id'],{}).get('kind')=='text'}
    if any(next(t for t in after['text']['texts'] if t['id']==i)!=definition for i,definition in old_texts.items()):
        raise ValueError('previous native text definitions changed')
    if jobs['final']['editable_texts']<8 or jobs['final']['editable_images']<9:
        raise ValueError('expanded coverage goal not met')
    agents={}
    for kind in ('text','asset'):
        folder=Path(getattr(args,kind+'_agent')).resolve();record=read(folder/'result.json');session=read(folder/'agent/agent-repair.json')
        if record['result']['status']!='CANDIDATE_READY' or session['tools_executed']!=1 or not record['live_vlm_called']:
            raise ValueError('real VLM repair loop incomplete')
        identifier=record['object_id'];candidate=Path(record['result']['selected']['candidate_directory'])
        if identifier not in after['selected']:raise ValueError('Agent target absent from delivered package')
        if kind=='text':
            final_trial=Path(after['selected'][identifier]['recovery_evidence']['trial_path'])
            matched={name:digest(candidate/name)==digest(final_trial/name) for name in ('source.png','mask.png','raw_inpaint.png','written_back.png')}
        else:
            matched={'asset_png':digest(candidate/'asset.png')==after['selected'][identifier]['asset_reference']['sha256']}
        if not all(matched.values()):raise ValueError('Agent candidate bytes differ from Office-verified final object')
        agents[kind]={'id':identifier,'tools':[e.get('tool') for e in session['events'] if e.get('tool')],
            'decisions':session['model_decisions'],'status':record['result']['status'],'bytes_match_final':matched,
            'receipt_sha256':digest(folder/'agent/agent-repair.json'),'budget':record['budget']}
    smoke=Path(args.smoke).resolve()
    branches=[]
    for name in ('smoke','smoke_final'):
        with sqlite3.connect(Path(getattr(args,name))/'state/parallel.sqlite') as db:
            branches.extend(db.execute("SELECT branch,payload FROM tasks WHERE branch IN ('text','asset','surface') AND state='COMPLETED'").fetchall())
    if {b for b,_ in branches}!={'text','asset','surface'}:raise ValueError('fresh three-branch run incomplete')
    fresh={b:{'path':json.loads(p)['evidence'],'sha256':digest(json.loads(p)['evidence'])} for b,p in branches}
    if read(smoke/'request.json').get('perception_replay') or not read(smoke/'request.json')['fresh_perception_requested']:
        raise ValueError('held-out page must have fresh perception')
    expanded=Path(args.expanded).resolve();attempt=next((expanded/'r12').glob('attempt-*'))
    assets=read(attempt/'page-01/asset-expansion.json')
    scan=next(Path('../r1204-text/outputs/r1205-scan-final').glob('scan-*.json'))
    blockers=read(scan)
    switch=Path('../r1204-text/outputs/r1205-background-switch-case/summary.json').resolve()
    summary={'background_switch_l1':{'path':str(switch),'sha256':digest(switch),'report':read(switch)},'schema':'R1205Acceptance/1','entrypoint':'product.parallel_pipeline.run',
        'baseline_commit':'878a14d98510a9138e87df6d59378ce3ec3beab5','jobs':jobs,
        'old_png_sha256_preserved':old_assets,'old_native_text_definitions_preserved':list(old_texts),
        'asset_expansion':assets,'full_text_scan':{'path':str(scan.resolve()),'sha256':digest(scan),'report':blockers},
        'agent_loops':agents,'agent_trigger_scope':'new tools verified through bounded run_r1205_agent host entry; upload automatic Agent trigger retains existing coordinate-diagnosis policy','fresh_heldout_branches':fresh,'heldout_original_page':2,
        'unresolved_issues':['page5 pc-gear/cloud-sync/database-gear complete foreground crosses semantic scope; no new SAM evidence','page5 30 text regions blocked in full L1 scan; one further L1-ready body deferred at goal8','heldout page2 asset/surface overlap expands peer protection and blocks all native text recovery','heldout full run has no native text replacement records; TextEditProbe FAIL; final Go NOT_RUN'],
        'source_boundaries':'page5 expanded from frozen evidence; page2 initial fresh text/surface plus recovered asset branch from the same upload; no prior-page scene or OCR used',
        'final_recheck_reason':'fault-injection found missing/UNKNOWN asset QA receipts; fail-closed fix retested, existing candidates reused for final combination QA',
        'tests':{'product_passed':257,'protocol_passed':39},'human_acceptance':False,'wps_checked':False,'automatic_publication':False,
        'scope':'8 native texts and 9 independent PNGs on page5;heldout coverage is recorded under jobs.smoke_final;remaining regions retain source raster;independent recall unknown'}
    failed=Path(args.smoke).resolve().with_name('r1205-smoke-full')
    if (failed/'result.json').exists():
        summary['failed_resume']=audit(failed)
        with sqlite3.connect(failed/'state/parallel.sqlite') as db:
            summary['failed_resume']['failure_receipts']=[{'branch':b,'state':st,'receipt':json.loads(p)} for b,st,p in db.execute("SELECT branch,state,payload FROM tasks WHERE state='FALLBACK'")]
        summary['failed_resume']['cause']='initial local forward reached legacy LayerD without expected model_binding; seed rejected unresolved ownership; corrected local target to existing remote port18847; no server deployment changed'
    write(output/'acceptance.json',summary)
    source=Image.open(expanded/'inputs/01.png').convert('RGB');old=Image.open(Path(args.baseline)/'delivery/page-01.png').convert('RGB');final=Image.open(Path(args.final)/'delivery/page-01.png').convert('RGB')
    canvas=Image.new('RGB',(source.width*3,source.height+35),'white');draw=ImageDraw.Draw(canvas)
    for n,(label,picture) in enumerate((('SOURCE',source),('R12-04',old),('R12-05',final))):
        draw.text((n*source.width+10,10),label,fill='black');canvas.paste(picture,(n*source.width,35))
    canvas.save(output/'whole-page-comparison.png')
    final_job=Path(args.final).resolve();final_attempt=next((final_job/'r12').glob('attempt-*'))
    rows=[]
    for identifier in after['selected']:
        if identifier in before['selected']:continue
        x,y,w,h=after['selected'][identifier]['bbox'];box=(max(0,int(x)-8),max(0,int(y)-8),min(source.width,int(x+w)+8),min(source.height,int(y+h)+8))
        deletion=Image.open(final_attempt/'office-combined'/f'01-{identifier}-delete.png').convert('RGB')
        row=Image.new('RGB',(1200,190),'white');d=ImageDraw.Draw(row);d.text((8,5),identifier+'    SOURCE / FINAL / ACTUAL OFFICE DELETION',fill='black')
        for n,picture in enumerate((source,final,deletion)):
            crop=picture.crop(box);crop.thumbnail((380,155));row.paste(crop,(n*400+8,30))
        rows.append(row)
    contact=Image.new('RGB',(1200,190*len(rows)),'white')
    for n,row in enumerate(rows):contact.paste(row,(0,n*190))
    contact.save(output/'new-object-edit-proof.png')
    write(output/'hashes.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'texts':jobs['final']['editable_texts'],'images':jobs['final']['editable_images'],'heldout_texts':jobs['smoke_final']['editable_texts'],'agents':agents},ensure_ascii=True))


if __name__=='__main__':main()
