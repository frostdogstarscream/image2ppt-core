"""Run frozen single-frame jobs after explicit environment/weight readiness."""
import argparse,json,subprocess,sys,time
from pathlib import Path

CHECKPOINTS={
 'sam3':('sam3.pt','9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'),
 'sam3.1':('sam3.1_multiplex.pt','0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6')}

def main(root,output,versions):
 output.mkdir(parents=True,exist_ok=False)
 def state(status,**extra):
  (output/'status.json').write_text(json.dumps({'status':status,'human_acceptance':False,**extra},indent=2))
 deadline=time.monotonic()+6*3600
 jobs=json.loads((root/'gpu-jobs.json').read_text())
 records=[]
 for version in versions:
  filename,sha=CHECKPOINTS[version]
  while not (root/'environment-ready.json').exists() or not (root/'weights'/filename).is_file():
   state('WAITING_DEPENDENCIES',version=version,checkpoint_present=(root/'weights'/filename).is_file(),records=records)
   if time.monotonic()>deadline:raise TimeoutError('Environment/weights not ready within six hours')
   time.sleep(20)
  tasks=[{'planner':'synthetic','page':0,'objects':[]}]+jobs
  for job in tasks:
   planner,page=job['planner'],job['page']
   target=output/version/planner/f'{page:02d}';target.parent.mkdir(parents=True,exist_ok=True)
   source=root/'inputs'/('synthetic/source.png' if planner=='synthetic' else f'sources/{page:02d}.png')
   plan=root/'inputs'/('synthetic/plan.json' if planner=='synthetic' else f'plans/{planner}/{page:02d}.json')
   command=[sys.executable,str(root/'single_frame.py'),'--version',version,'--source',str(source),'--plan',str(plan),
    '--sam-source',str(root/'sam3-official'),'--code-revision','f1647e8d0c1d280991f4d1b7d974c2704537fe66',
    '--checkpoint',str(root/'weights'/filename),'--checkpoint-sha256',sha,'--output',str(target)]
   if job['objects']:command+=['--objects',*job['objects']]
   state('RUNNING',version=version,planner=planner,page=page,records=records)
   with target.with_suffix('.stdout.log').open('w') as out,target.with_suffix('.stderr.log').open('w') as err:
    result=subprocess.run(command,stdout=out,stderr=err,timeout=1800)
   records.append({'version':version,'planner':planner,'page':page,'returncode':result.returncode})
   if result.returncode:
    state('FAILED',records=records,version=version,planner=planner,page=page)
    return 1
 state('INFERENCE_COMPLETE_REVIEW_REQUIRED',records=records)
 return 0

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--versions',nargs='+',choices=list(CHECKPOINTS),default=list(CHECKPOINTS));a=p.parse_args()
 raise SystemExit(main(a.root.resolve(),a.output.resolve(),a.versions))
