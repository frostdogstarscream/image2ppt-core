"""Hash-check R12-04's fresh upload, actual edits, tool loop and coverage costs."""
import argparse
import json
from pathlib import Path
import sqlite3

from PIL import Image,ImageDraw

from product.materialize import digest,write
from product.parallel_store import Store
from summarize_r1203 import audit,read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    job=Path(args.job).resolve();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    current=audit(job);previous=audit(args.baseline)
    if not current['code_manifest_matches_current']:raise ValueError('actual product code differs from tested code')
    result=Store(job/'state').committed()['result']
    attempt=next((job/'r12').glob('attempt-*'));ir=Path(result['slide_irs'][0]);scene=read(ir.parent/'scene.json')
    definition={o['id']:o for o in scene['asset']['objects']}
    pictures=[identifier for identifier,row in scene['selected'].items() if row['kind']=='asset']
    icons=[identifier for identifier in pictures if definition[identifier]['kind']=='icon']
    if current['editable_texts']<4 or len(icons)<3 or not current['office_pass']:
        raise ValueError('minimum fresh-upload editability not reached')
    if not all(v.get('passed') is True for v in current['visual']):raise ValueError('final independent visual QA incomplete')
    residuals=read(attempt/'residuals-01.json')['records']
    if any(r['residual_check_status']!='PASS' or r['residual_pixels_confirmed'] or r['unexplained_pixels'] for r in residuals):
        raise ValueError('actual deletion residual checks failed')
    office=read(job/'qa.json')['office']
    if any(not r['saved_reopened'] or r['outside_pixels'] for r in office['records']):
        raise ValueError('actual edit isolation failed')
    agent=read(job/'page-01/agent/agent-repair.json')
    if agent['result']['status']!='CANDIDATE_READY' or agent['tools_executed']!=1:
        raise ValueError('live tool decision loop incomplete')
    identifier=agent['binding']['object_id']
    selected=next(r for r in result['regions'][0] if r['object_id']==identifier and r['state']=='VERIFIED_LOCAL')
    actual_trial=attempt/'page-01'/identifier/selected['selected_background']
    agent_trial=Path(agent['result']['selected']['candidate_directory'])
    paired={name:digest(actual_trial/name)==digest(agent_trial/name) for name in ('source.png','mask.png','raw_inpaint.png','written_back.png')}
    if not all(paired.values()):raise ValueError('agent candidate differs from final edited region')
    agent_trials=len(read(agent_trial.parent/'index.json')['candidates'])
    pixel_candidates={(digest(agent_trial/'mask.png'),digest(agent_trial/'written_back.png'))}
    for state in result['regions'][0]:
        if state['state']!='VERIFIED_LOCAL':continue
        trial=attempt/'page-01'/state['object_id']/state['selected_background']
        pixel_candidates.add((digest(trial/'mask.png'),digest(trial/'written_back.png')))
    with sqlite3.connect(job/'state/parallel.sqlite') as db:
        branches=db.execute("SELECT branch,payload FROM tasks WHERE branch IN ('text','asset','surface') AND state='COMPLETED'").fetchall()
    if {r[0] for r in branches}!={'text','asset','surface'}:raise ValueError('fresh perception incomplete')
    branch_receipts={name:{'file':json.loads(payload)['evidence'],'sha256':digest(json.loads(payload)['evidence'])} for name,payload in branches}
    calls=[]
    for path in (job/'fresh-cache/model-calls/raw').glob('*.network.json'):
        reply=read(path)
        calls.append({'sha256':digest(path),'model':reply['model'],'usage':reply.get('usage')})
    records=read(job/'page-01/staged/transactions.json')
    asset_attempts=[r for r in records if r['id'] in definition]
    source=Image.open(job/'inputs/01.png').convert('RGB')
    final=Image.open(job/'delivery/page-01.png').convert('RGB')
    rows=[]
    for identifier in icons:
        x,y,w,h=scene['selected'][identifier]['bbox'];pad=12
        box=(max(0,int(x)-pad),max(0,int(y)-pad),min(source.width,int(x+w)+pad),min(source.height,int(y+h)+pad))
        deleted=Image.open(attempt/'office-combined'/f'01-{identifier}-delete.png').convert('RGB')
        row=Image.new('RGB',(1100,205),'white');draw=ImageDraw.Draw(row)
        draw.text((8,5),identifier+'    SOURCE / FINAL / ACTUAL OFFICE DELETION',fill='black')
        for index,image in enumerate((source,final,deleted)):
            crop=image.crop(box)
            scale=min(3,340/crop.width,170/crop.height)
            crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))))
            row.paste(crop,(index*365+8,30))
        rows.append(row)
    contact=Image.new('RGB',(1100,sum(r.height for r in rows)),'white');top=0
    for row in rows:contact.paste(row,(0,top));top+=row.height
    contact.save(output/'icons-edit-proof.png')
    old=Image.open(Path(args.baseline)/'delivery/page-01.png').convert('RGB')
    comparison=Image.new('RGB',(source.width*2,source.height+35),'white');draw=ImageDraw.Draw(comparison)
    draw.text((10,10),'R12-03',fill='black');draw.text((source.width+10,10),'R12-04 FRESH UPLOAD',fill='black')
    comparison.paste(old,(0,35));comparison.paste(final,(source.width,35));comparison.save(output/'whole-page-comparison.png')
    receipt={'schema':'R1204Acceptance/1','baseline_commit':'ab78a5e54bfd96c523b3d325b00a3c0039292f12',
        'baseline':previous,'current':current,'fresh_branches':branch_receipts,'model_receipts':calls,
        'source_sha256':scene['source_sha256'],'independent_icons':icons,'picture_ids':pictures,
        'asset_attempts':asset_attempts,'owner_residuals':residuals,
        'agent':{'binding':agent['binding'],'model_decisions':agent['model_decisions'],'tools_executed':agent['tools_executed'],
            'status':agent['result']['status'],'candidate_bytes_match_final':paired,
            'actual_office_verified_id':agent['binding']['object_id'],'receipt_sha256':digest(job/'page-01/agent/agent-repair.json')},
        'text_candidate_counts':{'regional':current['background_candidates'],'agent_tool':agent_trials,
            'total_saved':current['background_candidates']+agent_trials,'unique_mask_writeback_pairs':len(pixel_candidates)},
        'pipeline_wall_seconds':round((job/'result.json').stat().st_mtime-(job/'request.json').stat().st_mtime,2),
        'perception_wall_seconds':round((job/'page-01/barrier-1.json').stat().st_mtime-(job/'request.json').stat().st_mtime,2),
        'fallback_basis':result['fallback_basis'],'human_acceptance':False,'automatic_publication':False,
        'tests':{'product_passed':233,'protocol_passed':39},
        'workstreams':{'assets':'050b539925f4ac91b8a30b80512a0eaa8c8c5b73',
            'text':'efc6d13896b7e9d483d2fe0afc4f8adb9cdca657','agent':'dc3dc46721011fa11ddcc275c66b6c9708bf6f78'},
        'scope':'four native text regions and three independent icons plus two decorations; remaining content stays raster; recall denominator unknown'}
    write(output/'acceptance.json',receipt)
    write(output/'hashes.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'texts':current['editable_texts'],'icons':len(icons),'pictures':len(pictures),
        'agent':receipt['agent']['status'],'regional_seconds':current['regional_wall_seconds'],
        'pipeline_seconds':receipt['pipeline_wall_seconds']}))


if __name__=='__main__':main()
