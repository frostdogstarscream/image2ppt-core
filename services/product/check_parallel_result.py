"""Private real-run qualification, kept separate from the delivery bundle."""
import argparse,json
from pathlib import Path
from collections import Counter
from product.parallel_store import Store
from product.materialize import digest,write
from product.parallel_office import verify_pictures


def check(baseline,job,output):
    baseline=Path(baseline).resolve();job=Path(job).resolve()
    b=json.loads(baseline.read_text(encoding='utf-8'));commit=Store(job/'state').committed()
    if commit is None:raise ValueError('job is not committed')
    result=commit['result'];irs=result['slide_irs']
    if len(irs)!=len(b['pages']):raise ValueError('baseline page count changed')
    verify_pictures(job/'delivery/result.pptx',irs)
    frozen={'total':len(b['files']),'unchanged':sum(digest(p)==sha for p,sha in b['files'].items())}
    rows=[]
    for a in b['stable_assets']:
        irfile=Path(irs[a['page']-1]);ir=json.loads(irfile.read_text(encoding='utf-8'))
        if ir['source_sha256']!=a['source_sha256']:raise ValueError('page/source binding changed')
        node=next((n for n in ir['nodes'] if n['id']==a['id'] and n['type']=='image'),None)
        status='SAFE_RASTER_FALLBACK';sha=None
        if node:
            asset=next(r for r in ir['assets'] if r['id']==node['asset_id']);sha=asset['sha256']
            status='REUSED_EXACT' if sha==a['sha256'] else 'EQUIVALENT_CANDIDATE'
        rows.append({'page':a['page'],'id':a['id'],'status':status,'original_sha256':a['sha256'],
            'final_sha256':sha,'human_acceptance':False})
    qa=json.loads((job/'qa.json').read_text(encoding='utf-8'));office=qa.get('office') or {}
    counts=dict(Counter(r['status'] for r in rows))
    qualifies=frozen['total']==frozen['unchanged'] and counts.get('SAFE_RASTER_FALLBACK',0)==0 and office.get('status')=='PASS_APP_AUTOMATED'
    record={'schema':'ParallelRunQualification/1','status':'PASS' if qualifies else 'REVIEW_REQUIRED',
        'parallel_auto_qualified':False,'qualification_note':'Human/WPS acceptance and all required gates remain independent.',
        'generation':Store(job/'state').head(),'committed_files_verified':len(commit['files']),
        'pages':result['pages'],'reconstructed_pages':result['reconstructed_pages'],'frozen':frozen,
        'stable_objects':rows,'stable_counts':counts,'embedded_alpha_and_visible_rgb':'PASS',
        'office_status':office.get('status','UNKNOWN'),'actual_edit_actions':len(office.get('records',[])),
        'outside_pixels':sum(r['outside_pixels'] for r in office.get('records',[])),
        'human_acceptance':False,'wps_acceptance':'UNTESTED'}
    write(output,record)
    print(json.dumps({k:v for k,v in record.items() if k!='stable_objects'},ensure_ascii=False),flush=True)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('job');p.add_argument('output');a=p.parse_args()
    check(a.baseline,a.job,a.output)
