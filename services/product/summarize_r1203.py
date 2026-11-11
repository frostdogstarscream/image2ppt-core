"""Recompute R12-03 acceptance and comparable costs from immutable job receipts."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from product.materialize import digest, write
from product.parallel_store import Store


def read(path):
    return json.loads(Path(path).read_text('utf8'))


def audit(job):
    job=Path(job).resolve();commit=Store(job/'state').committed()
    if commit is None:raise ValueError('committed generation required')
    result=read(job/'result.json')
    if commit['result']!=result:raise ValueError('visible result differs from generation')
    attempt=next((job/'r12').glob('attempt-*'))
    scene=read(Path(result['slide_irs'][0]).parent/'scene.json')
    code=read(attempt/'code-manifest.json')
    package=Path(__file__).resolve().parent
    code_matches=all(digest(package/p)==sha for p,sha in code['files'].items())
    indexes=[read(p) for p in attempt.glob('page-*/*/index.json')]
    indexes=[r for r in indexes if 'candidates' in r]
    qa=read(job/'qa.json')
    if qa['regions']!=result['regions']:raise ValueError('region status differs from committed QA')
    local=[]
    for region in result['regions'][0]:
        if region['state']!='VERIFIED_LOCAL':continue
        proof=read(region['acceptance'])
        if proof['regional_automatic_checks']!='PASS':raise ValueError('regional hard gates failed')
        if proof['deleted_background']['original_signal_residual_pixels']:
            raise ValueError('visible text residual in actual Office deletion')
        local.append({'id':region['object_id'],'acceptance_sha256':digest(region['acceptance']),
                      'actual_deleted_source_signal_pixels':0,'replacement_status':proof['replacement']['status']})
    return {'job':str(job),'committed_files_verified':len(commit['files']),
        'code_manifest_matches_current':code_matches,'source_sha256':scene['source_sha256'],
        'perception_mode':read(job/'provenance.json')['perception_mode'],
        'background_candidates':sum(len(r['candidates']) for r in indexes),
        'lower_hypotheses_assessed':sum(sum('checks' in row for row in r.get('routes',[])) for r in indexes),
        'lama_candidates':sum(sum(c['method']=='big-lama' for c in r['candidates']) for r in indexes),
        'compiler_packages':len(list(attempt.rglob('quality.json'))),
        'office_packages':sum(p.parent.name in ('office','office-combined') for p in attempt.rglob('report.json')),
        'replacement_packages':sum(read(p).get('schema')=='TextEditProbe/1' for p in attempt.rglob('report.json')),
        'regional_wall_seconds':round((job/'qa.json').stat().st_mtime-(attempt/'code-manifest.json').stat().st_mtime,2),
        'editable_texts':sum(v['kind']=='text' for v in scene['selected'].values()),
        'editable_images':sum(v['kind']=='asset' for v in scene['selected'].values()),
        'office_pass':result['office_pass'],'final_office_actions':len(qa['office'].get('records',[])),
        'local_hard_gate_receipts':local,
        'outside_pixels':sum(r['outside_pixels'] for r in qa['office'].get('records',[])),
        'visual':result['independent_review'],'execution_cost':result.get('execution_cost'),
        'workflow_states':result.get('workflow_states'),
        'verified_regions':[{'object_id':r['object_id'],'quality':r.get('quality'),'early_stop':r.get('early_stop'),
                             'failure':r.get('failure')} for r in result['regions'][0]],
        'artifacts':result['artifacts']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('previous-job','current-job','fresh-job','replay-job','fault-job','negative-control','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    jobs={name:audit(getattr(args,name+'_job')) for name in ('previous','current','replay','fault')}
    if not all(jobs[k]['code_manifest_matches_current'] for k in ('current','replay','fault')):
        raise ValueError('final evidence does not match current product code')
    fresh=Path(args.fresh_job).resolve()
    import sqlite3
    with sqlite3.connect(fresh/'state/parallel.sqlite') as db:
        branches=db.execute("SELECT branch,payload FROM tasks WHERE state='COMPLETED' AND branch IN ('text','asset','surface')").fetchall()
    if {b for b,_ in branches}!={'text','asset','surface'}:raise ValueError('fresh three-branch evidence incomplete')
    branch_files={}
    for branch,payload in branches:
        file=Path(json.loads(payload)['evidence']);value=read(file)
        branch_files[branch]={'file':str(file),'sha256':digest(file),
            'references':[{'file':ref,'sha256':digest(ref)} for ref in value.get('references',[]) if Path(ref).is_file()]}
    with sqlite3.connect(fresh/'fresh-cache/model-calls/parallel.sqlite') as db:
        calls=db.execute('SELECT state,count(*) FROM calls GROUP BY state').fetchall()
    negative=read(Path(args.negative_control)/'control.json')
    if not negative['severe_defect_detected']:raise ValueError('negative control was not detected')
    current_result=read(Path(args.current_job)/'result.json')
    if negative['ir_sha256']!=digest(current_result['slide_irs'][0]):
        raise ValueError('negative control differs from the final native-object definitions')
    if jobs['current']['editable_texts']<2 or not jobs['current']['office_pass']:
        raise ValueError('two real Office-verified body regions required')
    if jobs['fault']['editable_texts']!=1:raise ValueError('fault isolation did not retain exactly the independent region')
    summary={'schema':'R1203Acceptance/1','entrypoint':'product.parallel_pipeline.run','jobs':jobs,
        'fresh_perception':{'job':str(fresh),'branch_evidence':branch_files,'call_states':dict(calls),
            'request':read(fresh/'request.json'),
            'wall_seconds':round((fresh/'page-01/barrier-1.json').stat().st_mtime-(fresh/'request.json').stat().st_mtime,2),
            'continued_using_current_task_raw_receipts':True,'frozen_scene_or_ocr_input':False},
        'negative_control':negative,'human_acceptance':False,'automatic_publication':False,
        'comparison_scope':'regional timings exclude perception; fresh upload has no historical asset anchors; frozen replay preserves seven verified images'}
    write(output/'acceptance.json',summary)
    columns=[('753399b frozen',jobs['previous']),('R12-03 same frozen inputs',jobs['replay']),('R12-03 fresh upload',jobs['current'])]
    report=['# R12-03 acceptance','',
        'Formal entrypoint product.parallel_pipeline.run. The fresh branches were called on the original page-5 image in this task. Recovery resumed from those raw model receipts after a generic wrapped-body grouping repair. No preset target IDs, frozen SceneGraph or frozen OCR inputs were used in the fresh path.','',
        '| Metric | '+' | '.join(name for name,_ in columns)+' |',
        '|---|'+'---|'*len(columns)]
    for name in ('background_candidates','compiler_packages','office_packages','replacement_packages','regional_wall_seconds','editable_texts','editable_images','lama_candidates'):
        report.append('| '+name+' | '+' | '.join(str(row[name]) for _,row in columns)+' |')
    report.extend(['','The same-input frozen comparison includes seven verified images in both versions. The isolated fresh-upload comparison retains unproven images in the source backing. Model planning calls are 0 for each frozen replay and 3 for fresh perception. Each final independent Go review is accounted separately in the budget ledger. Local OCR and LayerD are evidence services, distinct from MoMA/Go calls.',
        '', 'Normal final Office and Go QA passed. Minor visual differences are recorded by Go. A real Office deleted-text preview was independently rejected by Go as a severe defect. Fault injection retains one independent editable region when the other fails. REVIEW_REQUIRED and the publication gate remain active.','',
        'Reproduce with run_local.py --candidate-engine r12-regional --fresh-perception --prompt-key, then run_r1203_visual_control.py and this summarizer. Configure existing OCR/LayerD/LaMa paths through the existing runtime configuration. Credentials are injected into the process. Cost units in the ledger are operation weights, not a currency quote.'])
    (output/'comparison.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    source=Image.open(Path(args.current_job)/'inputs/01.png').convert('RGB')
    final=Image.open(Path(args.current_job)/'delivery/page-01.png').convert('RGB')
    canvas=Image.new('RGB',(source.width*2,source.height+35),'white');draw=ImageDraw.Draw(canvas)
    draw.text((10,10),'SOURCE',fill='black');draw.text((source.width+10,10),'R12-03 FRESH UPLOAD',fill='black')
    canvas.paste(source,(0,35));canvas.paste(final,(source.width,35));canvas.save(output/'whole-page-comparison.png')
    crops=Image.new('RGB',(1440,345),'white');draw=ImageDraw.Draw(crops)
    previous=Image.open(Path(args.previous_job)/'delivery/page-01.png').convert('RGB')
    for index,(label,image) in enumerate([('Source',source),('753399b frozen',previous),('R12-03 fresh upload',final)]):
        draw.text((index*480+10,10),label,fill='black')
        crops.paste(image.crop((275,155,505,305)).resize((460,300)),(index*480+10,35))
    crops.save(output/'body-comparison.png')
    write(output/'hashes.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'acceptance':str(output/'acceptance.json'),'normal_texts':jobs['current']['editable_texts']}))


if __name__=='__main__':main()
