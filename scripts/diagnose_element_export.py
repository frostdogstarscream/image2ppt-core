"""Bounded delta-debugging of a local Artifact Tool export; never edits source IR."""
import concurrent.futures,json,math,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];irfile=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(exist_ok=False)
data=json.loads(irfile.read_text(encoding='utf8'));ids=[n['id'] for n in sorted(data['nodes'],key=lambda n:n['z'])]
node=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe';history=[];counter=0
def trial(entry):
    index,keep=entry;folder=out/str(index);folder.mkdir();(folder/'ids.json').write_text(json.dumps(keep),encoding='utf8')
    with (folder/'trace.log').open('w',encoding='utf8') as log:
        try:
            p=subprocess.run([str(node),str(root/'scripts/trace_element_build.mjs'),str(irfile),'all',str(folder/'diagnostic.pptx'),'0','selected',str(folder/'ids.json')],stdout=log,stderr=subprocess.STDOUT,timeout=5)
            failed=False;reason='exit_'+str(p.returncode)
        except subprocess.TimeoutExpired:failed=True;reason='timeout_5s'
    return {'trial':index,'ids':keep,'timeout':failed,'reason':reason}
granularity=2
while len(ids)>=2 and counter<44:
    size=math.ceil(len(ids)/granularity);candidates=[ids[:start]+ids[start+size:] for start in range(0,len(ids),size)]
    candidates=[c for c in candidates if c]
    entries=[]
    for keep in candidates:counter+=1;entries.append((counter,keep))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(trial,entries))
    history.extend(results);failed=next((r for r in results if r['timeout']),None)
    if failed:ids=failed['ids'];granularity=max(2,granularity-1)
    elif granularity>=len(ids):break
    else:granularity=min(len(ids),granularity*2)
    (out/'progress.json').write_text(json.dumps({'remaining':ids,'trials':counter}),encoding='utf8')
report={'reproducer_ids':ids,'trials':history,'note':'Diagnostic timeout is not semantic validation; each trial retains all selected object properties.'}
(out/'diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'reproducer_ids':ids,'trials':counter}))
