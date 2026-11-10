"""Freeze regression references; never import old page plans into new planning."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).parent))
from product.materialize import digest,write

def freeze(old,output):
    old=Path(old);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    inputs=json.loads((old/'outputs/closure-regression-03/inputs.json').read_text(encoding='utf-8'))
    result=json.loads((old/'outputs/source-unmix-delivery-02/result.json').read_text(encoding='utf-8'))
    release=json.loads((old/'outputs/source-unmix-delivery-02/release-check.json').read_text(encoding='utf-8'))
    if not release.get('embedded_117_objects_checked') or release.get('new_stable_objects')!=78:
        raise ValueError('missing historical automatic release evidence')
    files={};assets=[];pages=[]
    def ref(p):
        p=Path(p).resolve();sha=digest(p);files[str(p)]=sha;return {'path':str(p),'sha256':sha}
    historical=old/'outputs/closure-regression-03/frozen-evidence.json'
    if historical.is_file():
        for row in json.loads(historical.read_text(encoding='utf-8'))['files']:
            p=(historical.parent/row['file']).resolve()
            if not p.is_relative_to(historical.parent.resolve()) or digest(p)!=row['sha256']:
                raise ValueError('historical frozen evidence changed')
            ref(p)
        ref(historical)
    for row,irfile,report in zip(inputs,result['slide_irs'],result['reports']):
        irfile=Path(irfile);ir=json.loads(irfile.read_text(encoding='utf-8'))
        bynode={n['id']:n for n in ir['nodes']};byasset={a['id']:a for a in ir['assets']}
        source=ref(row['source']);layers=Path(row['directory'])/'layerd'
        manifest=json.loads((layers/'manifest.json').read_text(encoding='utf-8'))
        ref(layers/'manifest.json');ref(layers/'request.json') if (layers/'request.json').exists() else None
        for layer in manifest['layers']:ref(layers/layer['file'])
        pages.append({'page':row['page'],'source':source,'layers':str(layers),'regression_ir':ref(irfile),
                      'quality':'REVIEW_REQUIRED','human_acceptance':False})
        for r in report['objects']:
            if r['status'] not in ('STABLE_REUSED','STAGED'):continue
            n=bynode['extracted-'+r['id']];a=byasset[n['asset_id']];asset=ref(irfile.parent/a['path'])
            assets.append({'page':row['page'],'source_sha256':source['sha256'],'id':r['id'],
                'definition':r['definition'],'bbox':n['bbox'],**asset,'automatic_quality':'PASS',
                'human_acceptance':False,'pixel_reference':None,'original_status':r['status'],
                'automatic_receipt':ref(old/'outputs/source-unmix-delivery-02/release-check.json')})
    if len(assets)!=117:raise ValueError(f'expected 117 stable automatic assets, found {len(assets)}')
    write(output/'baseline.json',{'schema':'ParallelFrozenBaseline/1','base_commit':'066bc9e44cdcdd5e8375b64b048a864d9280d405',
        'pages':pages,'stable_assets':assets,'files':files,'human_acceptance':False,'pixel_reference':None})
    print(json.dumps({'pages':len(pages),'stable_assets':len(assets),'frozen_files':len(files)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('old');p.add_argument('output');a=p.parse_args();freeze(a.old,a.output)
