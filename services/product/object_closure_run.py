"""Freeze private real-image evidence and replay the shadow route, never overwrite."""
import argparse
import json
from pathlib import Path
import shutil
from product.materialize import digest,write
from product.object_closure import extract_closed


def freeze(previous,output):
    previous=Path(previous).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    inputs=json.loads((previous/'inputs.json').read_text(encoding='utf-8'));records=[]
    receipt_file=previous.parents[1]/'ppt-rebuild/configs/layerd-baseline.json'
    receipts=json.loads(receipt_file.read_text(encoding='utf-8'))
    shutil.copyfile(receipt_file,output/'historical-provenance.json')
    for row in inputs:
        target=output/'inputs'/f"{row['page']:02d}"
        shutil.copytree(row['directory'],target)
        layerdir=target/'layerd';manifest=json.loads((layerdir/'manifest.json').read_text(encoding='utf-8'))
        if not (layerdir/'request.json').exists():
            receipt=receipts['pages'].get(manifest['source_sha256'],{})
            if receipt.get('manifest_sha256')!=digest(layerdir/'manifest.json') or receipt.get('source_png_sha256')!=digest(layerdir/'source.png'):
                raise ValueError('historical provenance mismatch')
            write(layerdir/'request.json',{'source_sha256':manifest['source_sha256'],'normalized_sha256':receipt['source_png_sha256'],
                'adapter':'HistoricalProvenance/1','receipt_sha256':digest(receipt_file)})
        records.append({'page':row['page'],'directory':str(target),'source':str(target/'layerd/source.png'),
            'baseline_ir':str(target/'baseline/slideir.json'),'baseline_page':str(target/'visual-page.json')})
    shutil.copytree(previous/'plans',output/'plans')
    evidence=[]
    for file in output.rglob('*'):
        if file.is_file():evidence.append({'file':str(file.relative_to(output)),'sha256':digest(file)})
    history=[]
    for pattern in ('runs/*/pages/05/extraction.json','runs/*/build/*.pptx'):
        for file in previous.glob(pattern):history.append({'path':str(file),'sha256':digest(file)})
    write(output/'inputs.json',records)
    write(output/'frozen-evidence.json',{'schema':'FrozenObjectRegression/1','baseline_commit':'25ebee15391d979b416eb8b55d536cd2546f8af2',
        'files':evidence,'history':history,'golden_mask':None,'human_acceptance':False,
        'page05_boundary':'Cloud contour + server bars/dots + own edges; caption, plate, platform and external connectors excluded.'})


def replay(root,output,pages):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False);results=[]
    for row in json.loads((root/'inputs.json').read_text(encoding='utf-8')):
        if row['page'] not in pages:continue
        number=f"{row['page']:02d}"
        file,report=extract_closed(root/'plans/moma'/(number+'.json'),Path(row['directory'])/'layerd',row['baseline_ir'],output/number)
        results.append({'page':row['page'],'ir':str(file),'staged':report['staged'],'fallback':report['fallback']})
        print(json.dumps(results[-1]),flush=True)
    write(output/'summary.json',{'schema':'ObjectClosureShadow/1','pages':results,'human_acceptance':False,'publication':'DISABLED'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','replay']);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pages',type=int,nargs='+',default=list(range(1,13)));a=p.parse_args()
    if a.action=='freeze':freeze(a.input,a.output)
    else:replay(a.input,a.output,a.pages)
