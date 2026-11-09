"""Independent Go review of a frozen object experiment; never edits its plans."""
import json
import os
import threading
import time
from pathlib import Path
from product.contracts import Page,Review
from product.go_provider import GoProvider
from product.materialize import digest,write
from product.pipeline import effective_findings
from product.prompts import REVIEW
from product.visual_checks import compare_graphics

_monitors=set()
_monitor_lock=threading.Lock()


def start_monitor(root):
    root=Path(root).resolve()
    with _monitor_lock:
        if str(root) in _monitors:return
        _monitors.add(str(root))
    def monitor():
        while True:
            time.sleep(2);queue=root/'review-queue.json'
            if not queue.exists() or not os.environ.get('OPENCODE_GO_API_KEY'):continue
            try:
                for item in json.loads(queue.read_text(encoding='utf-8')):
                    run=Path(item).resolve()
                    if not run.is_relative_to(root/'runs'):raise ValueError('review outside experiment')
                    receipt=run/'review-dispatch.json'
                    try:
                        with receipt.open('x',encoding='utf-8') as stream:json.dump({'status':'RESERVED'},stream)
                    except FileExistsError:continue
                    try:review_run(run);write(receipt,{'status':'DONE'})
                    except Exception as exc:write(receipt,{'status':'FAILED','error_type':type(exc).__name__})
            except (OSError,ValueError):pass
    threading.Thread(target=monitor,daemon=True).start()


def review_run(run,preview_kind='artifact',baseline_previews=None,office_directory=None):
    run=Path(run).resolve();root=run.parent.parent
    inputs=json.loads((root/'inputs.json').read_text(encoding='utf-8'))
    from product.object_stage import effective_page
    pages=[effective_page(Page.model_validate(json.loads(Path(r['baseline_page']).read_text(encoding='utf-8'))),
                          run/f'page-{i:02d}'/'slideir.json') for i,r in enumerate(inputs,1)]
    actor=GoProvider(run);reports=[];directory=run/('go-review' if preview_kind=='artifact' else 'go-review-powerpoint');directory.mkdir(exist_ok=True)
    for i,(record,page) in enumerate(zip(inputs,pages),1):
        source=Path(record['source']);preview=(run/'delivery'/f'page-{i:02d}.png' if preview_kind=='artifact' else (Path(office_directory) if office_directory else run/'office')/f'page-{i:02d}-before.png');plan=run/f'page-{i:02d}'/'slideir.json'
        binding={'source_sha256':digest(source),'plan_sha256':digest(plan),'preview_sha256':digest(preview)}
        resultfile=directory/f'page-{i:02d}.json'
        if resultfile.exists():
            previous=json.loads(resultfile.read_text(encoding='utf-8'))
            if previous.get('binding')==binding:reports.append(previous);continue
        callfile=directory/f'page-{i:02d}-call.json'
        rawfile=directory/f'page-{i:02d}-raw.json'
        if callfile.exists() and not rawfile.exists():raise RuntimeError('Review request already reserved; inspect before explicit retry')
        extraction=json.loads((run/f'page-{i:02d}'/'extraction.json').read_text(encoding='utf-8'))
        if rawfile.exists():
            reserved=json.loads(callfile.read_text(encoding='utf-8'))
            if reserved.get('binding')!=binding:raise ValueError('stale raw review')
            raw=json.loads(rawfile.read_text(encoding='utf-8'))
        else:
            write(callfile,{'status':'RESERVED','model':actor.model,'binding':binding})
            detail=compare_graphics(source,preview,page,directory/f'page-{i:02d}-detail.png')
            baseline=Path(baseline_previews)/f'page-{i:02d}.png' if baseline_previews else None
            extra={'baseline_preview_sha256':digest(baseline),'image_order':['source','candidate','frozen_baseline','detail_if_present']} if baseline else {'image_order':['source','candidate','detail_if_present']}
            raw=actor.ask(REVIEW+'\nThis is a frozen object extraction experiment. Focus on complete icons/illustrations, thin connectors, text leakage and residual duplicate graphics. Text and layout are a shared fixed baseline. If a baseline image is supplied, distinguish inherited differences from NEW extraction defects explicitly in descriptions. Do not infer a successful extraction just from consistent fields. Report defects; never rewrite the plan.',
                          {**binding,**extra,'plan':page.model_dump(),'object_extraction':extraction},[source,preview,*([baseline] if baseline else []),*([detail] if detail else [])],Review)
            write(rawfile,raw)
        checked=Review.model_validate(raw)
        if any(getattr(checked,k)!=v for k,v in binding.items()):raise ValueError('Go review binding mismatch')
        ids=({n.id for n in [*page.texts,*page.shapes,*page.crops]}|{o['id'] for o in extraction['objects']}
             |{identifier for o in extraction['objects'] for identifier in o['replaces']})
        if any(not set(f.node_ids)<=ids for f in checked.findings):raise ValueError('Go review unknown node')
        row={'binding':binding,'model':actor.model,'findings':effective_findings(checked),'passed':not effective_findings(checked),'human_acceptance':False}
        write(resultfile,row);write(callfile,{'status':'RECEIVED','model':actor.model,'binding':binding});reports.append(row)
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    summary['opencode_review']='PASSED' if all(r['passed'] for r in reports) else 'FINDINGS_REMAIN'
    summary['review_findings']=sum(len(r['findings']) for r in reports);write(run/'summary.json',summary)
    return summary


if __name__=='__main__':
    import argparse,getpass
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--prompt-key',action='store_true');p.add_argument('--preview-kind',choices=['artifact','powerpoint'],default='artifact');p.add_argument('--baseline-previews',type=Path);a=p.parse_args()
    if a.prompt_key:os.environ['OPENCODE_GO_API_KEY']=getpass.getpass('OpenCode Go key (hidden, memory only): ')
    result=review_run(a.run,a.preview_kind,a.baseline_previews);print(json.dumps({'status':result['opencode_review'],'findings':result['review_findings']}))
