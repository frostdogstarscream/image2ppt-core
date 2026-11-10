"""Regression ONLY: frozen old scene round-trip, never new-task generation."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from product.materialize import digest,write
from product.pipeline import compile_pages
from product.object_stage import effective_page
from product.contracts import Page
from product.resource_lease import lease
from product.parallel_office import inspect

def check(baseline,old,output):
    b=json.loads(Path(baseline).read_text(encoding='utf-8'));old=Path(old);output=Path(output).resolve();output.mkdir(exist_ok=False)
    inputs=json.loads((old/'outputs/closure-regression-03/inputs.json').read_text(encoding='utf-8'))
    pages=[];irs=[]
    for p,original in zip(b['pages'],inputs):
        ir=Path(p['regression_ir']['path'])
        if digest(ir)!=p['regression_ir']['sha256']:raise ValueError('frozen reference changed')
        # Read the frozen expected native text ONLY for this regression oracle.
        expected=Page.model_validate(json.loads(Path(original['baseline_page']).read_text(encoding='utf-8')))
        pages.append(effective_page(expected,ir));irs.append(ir)
    with lease('compiler'):artifacts=compile_pages(irs,output/'compile',pages)
    with lease('office'):office=inspect(artifacts['pptx'],irs,output/'office',render_only=True)
    rows=[]
    previous=old/'outputs/source-unmix-delivery-02/actual-office'
    for i in range(1,13):
        before=np.array(Image.open(previous/f'page-{i:02d}.png').convert('RGB'))
        after=np.array(Image.open(output/'office'/f'page-{i:02d}-before.png').convert('RGB'))
        rows.append({'page':i,'changed_pixels':int((before!=after).any(2).sum()),'same_size':before.shape==after.shape})
    unchanged=sum(digest(p)==sha for p,sha in b['files'].items())
    result={'schema':'FrozenSceneRegression/1','status':'PASS' if all(r['changed_pixels']==0 for r in rows) else 'REVIEW_REQUIRED',
        'pages':rows,'frozen_files_unchanged':unchanged,'frozen_files':len(b['files']),'stable_asset_hashes_checked':len(b['stable_assets']),
        'office':office,'new_reconstruction_accuracy_claimed':False,'human_acceptance':False}
    write(output/'report.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('old');p.add_argument('output');a=p.parse_args();check(a.baseline,a.old,a.output)
