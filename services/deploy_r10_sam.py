"""Deploy a NEW private SAM instance on the authorized Windows GPU host.

Never stops/reconfigures an old task. Reads only this project's existing model
configuration; its private token file is referenced, never copied or displayed.
"""
import argparse,json,os,shutil,subprocess,sys,time,urllib.request,zipfile
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

def deploy(archive,target,old_config):
    archive=Path(archive).resolve();target=Path(target).resolve();old_config=Path(old_config).resolve()
    workspace=Path('F:/code/image2ppt/outputs').resolve()
    if not target.is_relative_to(workspace) or target==workspace:raise ValueError('deployment outside named project outputs')
    if target.exists():raise ValueError('use a new deployment directory; historical deployments remain immutable')
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (target/name).resolve().is_relative_to(target):raise ValueError('archive traversal')
        target.mkdir();z.extractall(target)
    cfg=json.loads(old_config.read_text(encoding='utf-8-sig'))
    cfg['jobs']=str(target/'jobs');Path(cfg['jobs']).mkdir()
    (target/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    # GPU workers share this physical-device namespace. No other app is killed.
    boot='''import os,runpy,sys
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ['IMAGE2PPT_RESOURCE_ROOT']=str(root.parent/'r10-gpu-shared-leases')
os.environ['IMAGE2PPT_GPU_ID']='gpu-6617fc79-6516-2e9f-6d7b-76b1e9ec2f6d'
log=(root/'worker.log').open('a',encoding='utf-8');sys.stdout=log;sys.stderr=log
sys.argv=[str(root/'segmentation/service.py'),'--config',str(root/'config.json'),'--host','127.0.0.1','--port','18844']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    (target/'boot.py').write_text(boot,encoding='utf-8')
    task='Image2PPT-Parallel-R10-SAM-20261110';pythonw=Path(cfg['python']).with_name('pythonw.exe')
    if not pythonw.is_file():raise ValueError('existing windowless Python missing')
    query=subprocess.run(['schtasks','/Query','/TN',task],capture_output=True)
    if query.returncode==0:raise ValueError('task already exists; will not overwrite it')
    action=f'"{pythonw}" "{target/"boot.py"}"'
    result=subprocess.run(['schtasks','/Create','/TN',task,'/SC','ONLOGON','/RL','LIMITED','/IT','/TR',action],capture_output=True)
    if result.returncode:raise RuntimeError('new scheduled task registration failed: '+result.stderr.decode('gbk',errors='replace')[:300])
    subprocess.run(['schtasks','/Run','/TN',task],capture_output=True,check=True)
    token=Path(cfg['token_file']).read_text(encoding='utf-8').strip()
    ready=False
    for _ in range(50):
        try:
            req=urllib.request.Request('http://127.0.0.1:18844/v1/capabilities',headers={'Authorization':'Bearer '+token})
            with urllib.request.urlopen(req,timeout=2) as r:cap=json.loads(r.read())
            if cap['worker_alive']:ready=True;break
        except OSError:pass
        time.sleep(.2)
    result={'ready':ready,'task':task,'port':18844,'old_task_untouched':True,'gpu_resource_root':str(target.parent/'r10-gpu-shared-leases')}
    (target/'deployment.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('target');p.add_argument('old_config');a=p.parse_args();deploy(a.archive,a.target,a.old_config)
