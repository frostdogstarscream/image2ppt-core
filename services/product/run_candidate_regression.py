"""r11 first-batch frozen evidence trials. Explicit opt-in, never publication.

The previous directory is a run_region_regression output, not a fresh upload.
Each safe text candidate is compiled independently over the unchanged baseline.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path
from product.materialize import digest,write
from product.parallel_contracts import SceneGraph
from product.parallel_store import Store
from product.parallel_candidate_engine import discover
from product.parallel_candidate_stage import stage
from product.parallel_candidate_contracts import candidate_path
from product.parallel_pipeline import effective_texts
from product.pipeline import compile_pages
from product.parallel_office import inspect
from product.parallel_edit_probe import inspect_text
from product.parallel_residual_gate import inspect_residuals
from product.resource_lease import lease
from diagnose_region_baseline import pptx_inventory

BASELINE='43d6833a293bda10e470173651f4ae65b5739676'


def load(file):return json.loads(Path(file).read_text(encoding='utf-8'))


def freeze(previous,output):
    baseline=output/'baseline';baseline.mkdir()
    manifest={};targets=[]
    def copy(file,target):
        file=Path(file);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
        sha=digest(file)
        if digest(target)!=sha:raise ValueError('baseline copy changed bytes')
        manifest[str(file.resolve())]={'copy':str(target.resolve()),'sha256':sha}
    copy(previous/'fonts.json',baseline/'fonts.json')
    artifacts=load(previous/'after-artifacts.json')
    if digest(artifacts['pptx'])!=artifacts['pptx_sha256']:raise ValueError('baseline PPTX changed')
    copy(artifacts['pptx'],baseline/'result.pptx')
    inventory=pptx_inventory(baseline/'result.pptx')
    for index,page in enumerate((2,5)):
        original=previous/f'page-{page:02d}'
        for file in (original/'after').rglob('*'):
            if file.is_file():copy(file,baseline/f'page-{page:02d}'/'after'/file.relative_to(original/'after'))
        copy(original/'input-scene.json',baseline/f'page-{page:02d}'/'input-scene.json')
        copy(original/'input-bindings.json',baseline/f'page-{page:02d}'/'input-bindings.json')
        for file,sha in load(original/'input-bindings.json').items():
            if digest(file)!=sha:raise ValueError('frozen automatic evidence changed')
            copy(file,baseline/'evidence'/sha/Path(file).name)
        graph=load(original/'input-scene.json');stage_graph=load(original/'after/scene.json')
        for t in graph['text']['texts']:
            targets.append({'page':page,'object_id':t['id'],'text':t['text'],
                'previously_native':t['id'] in stage_graph['selected'],'source_sha256':graph['source_sha256']})
        inventory[index]['source_page']=page
    write(baseline/'baseline.json',{'schema':'R11FrozenBaseline/1','requested_code_baseline':BASELINE,
          'provenance':'existing automatic-evidence replay; original model run SHA not inferred',
          'files':manifest,'inventory':inventory,'fresh_upload':False})
    write(baseline/'candidate-test-targets.json',targets)
    with (baseline/'baseline-inventory.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['source_page','native_text','native_shapes','images','transparent_png']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(inventory)
    return baseline


def run(previous,output,office=False):
    previous=Path(previous).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    baseline=freeze(previous,output);store=Store(output/'state');fonts=load(baseline/'fonts.json');trials=[]
    report={'schema':'R11CandidateRegression/1','mode':'frozen-evidence-replay','fresh_upload':False,
            'model_calls':0,'parallel_auto':False,'published':False,'trials':trials,
            'human_acceptance':False,'text_recall':None,'go_review':'UNKNOWN'}
    write(output/'run-manifest.json',report)
    for page in (2,5):
        folder=output/f'page-{page:02d}';folder.mkdir()
        graph=SceneGraph.model_validate(load(baseline/f'page-{page:02d}/input-scene.json'))
        base=baseline/f'page-{page:02d}/after'
        # Existing final/source-bound stable ownership is independently checked
        # again in stage. Discovery cannot modify any of these stable files.
        candidates=discover(graph,{'path':str(base/'source.png')},fonts,folder/'candidates',store)
        selected=load(base/'scene.json')['selected']
        for candidate in candidates:
            if candidate['hard_gate']!='PASS' or any(i in selected for i in candidate['object_ids']):continue
            cid=candidate['candidate_id'];trial=candidate_path(folder,cid)
            row={'source_page':page,'candidate_id':cid,'object_ids':candidate['object_ids'],
                 'status':'REVIEW_UNKNOWN','office':'UNKNOWN','text_edit':'UNKNOWN','residual':'UNKNOWN'}
            trials.append(row)
            try:
                ir=stage(candidate_path(folder/'candidates',cid),graph,base,trial/'staged')
                staged=SceneGraph.model_validate(load(ir.parent/'scene.json'))
                with lease('compiler'):artifacts=compile_pages([ir],trial/'compile',[effective_texts(staged,ir)])
                write(trial/'artifacts.json',artifacts);row['artifacts']=artifacts;row['ir']=str(ir)
                row['actual_inventory']=pptx_inventory(artifacts['pptx'])
                if office:
                    with lease('office'):
                        check=inspect(artifacts['pptx'],[ir],trial/'office')
                        text=inspect_text(artifacts['pptx'],[ir],trial/'text-office',set(candidate['object_ids']))
                    residual=inspect_residuals(ir,trial/'office',1,trial/'residuals.json')
                    target=[r for r in residual if r['id'] in candidate['object_ids']]
                    row.update(office=check['status'],text_edit=text['status'],
                        residual='PASS' if target and all(r['residual_check_status']=='PASS' for r in target) else 'UNKNOWN',
                        long_text_supported=text.get('long_text_supported',False))
                    if check['status']=='PASS_APP_AUTOMATED' and text['status']=='PASS' and row['residual']=='PASS':
                        row['status']='EDIT_PROBES_PASS_REVIEW_UNKNOWN'
            except Exception as exc:
                row['error']=type(exc).__name__+': '+str(exc)
            write(output/'run-manifest.json',report)
            print(json.dumps({k:v for k,v in row.items() if k not in ('artifacts','actual_inventory')},ensure_ascii=False),flush=True)
    report['page05_edit_probe_gain']=sum(len(t['object_ids']) for t in trials if t['source_page']==5 and t['status']=='EDIT_PROBES_PASS_REVIEW_UNKNOWN')
    # This is intentionally not an automatic quality/pass counter.
    report['automatic_verified_gain']=0
    for original,entry in load(baseline/'baseline.json')['files'].items():
        if digest(original)!=entry['sha256'] or digest(entry['copy'])!=entry['sha256']:
            raise ValueError('baseline mutated during candidate trials')
    report['baseline_preserved']=True
    write(output/'run-manifest.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous');parser.add_argument('output')
    parser.add_argument('--candidate-engine',choices=['r11-text'],required=True)
    parser.add_argument('--office',action='store_true')
    args=parser.parse_args();run(args.previous,args.output,args.office)
