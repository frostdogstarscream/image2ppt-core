"""Freeze cached GPU/asset evidence separately, never rerun perception."""
import argparse,json,shutil
from pathlib import Path
from product.materialize import digest,write


def freeze(baseline,perception,output,pages=(2,5)):
    baseline,perception,output=map(lambda p:Path(p).resolve(),(baseline,perception,output))
    output.mkdir(parents=True,exist_ok=False);saved=json.loads(baseline.read_text(encoding='utf-8'));files={}
    def copy(source,target):
        source=Path(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        sha=digest(source)
        if digest(target)!=sha:raise ValueError('frozen byte mismatch')
        files[str(source)]={'copy':str(target),'sha256':sha}
    copy(baseline,output/'original-baseline.json')
    for page in pages:
        layerdir=Path(saved['pages'][page-1]['layers']);samdir=perception/f'page-{page:02d}/sam'
        for name,root in [('layerd',layerdir),('sam',samdir)]:
            for file in root.rglob('*'):
                if file.is_file():copy(file,output/f'page-{page:02d}'/name/file.relative_to(root))
        plan=samdir.parent/'sam-plan.json'
        if plan.exists():copy(plan,output/f'page-{page:02d}/sam-plan.json')
        for asset in saved['stable_assets']:
            if asset['page']!=page:continue
            if digest(asset['path'])!=asset['sha256']:raise ValueError('stable input hash mismatch')
            copy(asset['path'],output/f'page-{page:02d}'/'stable'/f"{asset['id']}-{asset['sha256']}.png")
    write(output/'frozen-inputs.json',{'schema':'R11FrozenGPUEvidence/1','files':files,'pages':list(pages),'model_calls':0})
    print(json.dumps({'frozen_files':len(files),'output':str(output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline');p.add_argument('perception');p.add_argument('output');a=p.parse_args()
    freeze(a.baseline,a.perception,a.output)
