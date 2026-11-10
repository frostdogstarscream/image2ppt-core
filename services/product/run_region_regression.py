"""Two-page, frozen automatic-evidence A/B. Never authors SlideIR by hand."""
import argparse,json,subprocess,types,shutil
from pathlib import Path
from collections import Counter
from product.materialize import digest,write
from product.parallel_contracts import SceneGraph
from product.parallel_store import Store,key
from product.parallel_rebuild import build
from product.object_extract import load_page,load_sam
from product.parallel_pipeline import effective_texts
from product.resource_lease import lease
from diagnose_region_baseline import pptx_inventory
ROOT=Path(__file__).resolve().parents[2]
COMMIT='3e97fac448c864138b8c657702ebd00ab0e6597f'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def rebuild(previous,output):
    previous=Path(previous).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    req=load(previous/'request.json');baseline=load(req['frozen_baseline']);fonts=load(previous/'fonts.json');write(output/'fonts.json',fonts)
    # Reuse exact automatically planned/calibrated inputs to isolate code effects.
    original_irs=load(previous/'compile-0/input-irs.json')
    oldsource=subprocess.check_output(['git','show',COMMIT+':services/product/product/parallel_rebuild.py'],cwd=ROOT).decode('utf-8')
    (output/'baseline-rebuild-3e97fac.py.txt').write_text(oldsource,encoding='utf-8')
    old=types.ModuleType('product._region_baseline');old.__package__='product';old.__file__=str(ROOT/'services/product/product/parallel_rebuild.py')
    exec(compile(oldsource,old.__file__,'exec'),old.__dict__)
    rows=[];store=Store(output/'evidence-store')
    for page in (2,5):
        folder=output/f'page-{page:02d}';folder.mkdir()
        graphfile=Path(original_irs[page-1]).parent/'scene.json';graph=SceneGraph.model_validate(load(graphfile))
        graph.selected={};graph.transactions=[];graph.raster_backing=None
        bindings={str(graphfile):digest(graphfile)}
        # Relocate verified local receipts into this worktree, preserving content.
        for evidence in [graph,graph.text,graph.asset,graph.surface]:
            relocated=[]
            for ref in evidence.references:
                file=Path(ref)
                if not file.is_file():continue
                value=load(file)
                if file.parent.name=='evidence' and len(file.stem)==64:
                    if key(value)!=file.stem:raise ValueError('cached evidence hash mismatch')
                    relocated.append(store.evidence(value));bindings[str(file)]=digest(file)
                else:relocated.append(ref)
            evidence.references=relocated
        snap={'path':str(previous/'inputs'/f'{page:02d}.png'),'source_sha256':graph.source_sha256,'image_size':graph.image_size,'transform':graph.transform}
        if digest(snap['path'])!=graph.source_sha256:raise ValueError('source mismatch')
        saved=baseline['pages'][page-1];layers=load_page(saved['layers']);stable=[a for a in baseline['stable_assets'] if a['page']==page]
        masks={};samplan=previous/f'page-{page:02d}/sam-plan.json'
        if samplan.exists():
            for obj in graph.asset.objects:
                values,error=load_sam(samplan.parent/'sam',obj,graph.source_sha256,digest(samplan),tuple(graph.image_size),'sam3',individual=True)
                if not error:masks[obj.id]=values
        write(folder/'input-scene.json',graph.model_dump());write(folder/'input-bindings.json',bindings)
        row={'page':page,'source_sha256':graph.source_sha256,'proposed':{'text':len(graph.text.texts),'asset':len(graph.asset.objects),'surface':len(graph.surface.shapes)}}
        for label,operation in [('before',old.build),('after',build)]:
            g=graph.model_copy(deep=True)
            with lease('cpu'):ir,report=operation(g,snap,fonts,folder/label,layers=layers,stable=stable,strategy='auto',sam_candidates=masks,pixel_cache=output/'pixel-cache'/label)
            counts=Counter(n['type'] for n in load(ir)['nodes'] if n['id']!='source-plate')
            row[label]={'ir':str(ir),'candidate_counts':dict(counts),'staged_ids':[r['id'] for r in report if r['state']=='STAGED'],
                'fallback_reasons':dict(Counter(r.get('reason','unknown') for r in report if r['state']=='FALLBACK')),
                'fallback_objects':[r for r in report if r['state']=='FALLBACK']}
        row['added_candidates']=sorted(set(row['after']['staged_ids'])-set(row['before']['staged_ids']))
        row['removed_candidates']=sorted(set(row['before']['staged_ids'])-set(row['after']['staged_ids']))
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('before','after')},ensure_ascii=False),flush=True)
    write(output/'comparison.json',{'code_baseline':COMMIT,'mode':'frozen-automatic-evidence-replay','manual_slideir':False,'model_calls':0,'pages':rows,'status':'REVIEW_REQUIRED'})
    diff=subprocess.check_output(['git','diff','--','services/product'],cwd=ROOT)
    (output/'code.diff').write_bytes(diff)
    for name in ('parallel_regions.py','parallel_region_gate.py'):
        shutil.copyfile(ROOT/'services/product/product'/name,output/name)


def compile_and_check(output,office=False):
    from product.pipeline import compile_pages
    from product.parallel_office import inspect
    output=Path(output).resolve();comparison=load(output/'comparison.json')
    for label in ('before','after'):
        irs=[Path(r[label]['ir']) for r in comparison['pages']]
        graphs=[SceneGraph.model_validate(load(p.parent/'scene.json')) for p in irs]
        manifest=output/(label+'-artifacts.json')
        if manifest.exists():artifacts=load(manifest)
        else:
            compile_dir=output/(label+'-compile')
            if compile_dir.exists():compile_dir=output/(label+'-compile-'+__import__('uuid').uuid4().hex[:8])
            with lease('compiler'):artifacts=compile_pages(irs,compile_dir,[effective_texts(g,p) for g,p in zip(graphs,irs)])
            write(manifest,artifacts)
        inventory=pptx_inventory(artifacts['pptx']);write(output/(label+'-inventory.json'),inventory)
        print(json.dumps({'phase':'compiled','variant':label,'inventory':inventory}),flush=True)
        if office:
            target=output/(label+'-office')
            if (target/'report.json').exists():continue
            with lease('office'):receipt=inspect(artifacts['pptx'],irs,target,render_only=label=='before')
            print(json.dumps({'phase':'office','variant':label,'status':receipt['status'],'actions':len(receipt['records'])}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['rebuild','compile','office']);p.add_argument('output');p.add_argument('--previous');a=p.parse_args()
    if a.phase=='rebuild':rebuild(a.previous,a.output)
    else:compile_and_check(a.output,a.phase=='office')
