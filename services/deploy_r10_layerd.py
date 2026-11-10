"""Add r10 LayerD on a new port, using the existing readonly GPU environment."""
import json,subprocess,sys,time,urllib.request,argparse,zipfile
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

def deploy(root,*,archive=None,port=18845,task='Image2PPT-Parallel-R10-LayerD-20261110'):
    root=Path(root).resolve();project=Path('F:/code/image2ppt').resolve()
    if not root.is_relative_to(project/'outputs'):raise ValueError('outside project outputs')
    if not 1024<=port<=65535 or not task.startswith('Image2PPT-Parallel-R10-'):raise ValueError('invalid dedicated deployment')
    if subprocess.run(['schtasks','/Query','/TN',task],capture_output=True).returncode==0:raise ValueError('will not overwrite existing task')
    if archive:
        if root.exists():raise ValueError('deployment directory already exists')
        with zipfile.ZipFile(archive) as z:
            if any(not (root/n).resolve().is_relative_to(root) for n in z.namelist()):raise ValueError('archive path escape')
            root.mkdir();z.extractall(root)
    boot='''import os,sys
from pathlib import Path
root=Path(__file__).resolve().parent;project=Path('F:/code/image2ppt')
cache=root/'layerd-cache';cache.mkdir(exist_ok=True)
models=project/'.runtime/server38-snapshot/image2ppt-layerd/models'
os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1',
'HF_HOME':str(cache/'huggingface'),'TORCH_HOME':str(cache/'torch'),'TEMP':str(cache),'TMP':str(cache),
'LAYERD_MODELS':str(models),'LAMA_MODEL':str(models/'big-lama.pt'),
'LAYERD_DATA':str(root/'layerd-data'),'LAYERD_TOKEN_FILE':str(project/'.runtime/layerd/client-token'),
'LAYERD_PROCESS_SIZE':'1024','LAYERD_MAX_ITERATIONS':'12','LAYERD_REVISION':'21aef937a0371614adb4d961f52d02409cb8ecc7',
'IMAGE2PPT_RESOURCE_ROOT':str(root.parent/'r10-gpu-shared-leases'),
'IMAGE2PPT_GPU_ID':'gpu-6617fc79-6516-2e9f-6d7b-76b1e9ec2f6d','OMP_NUM_THREADS':'4'})
log=(root/'layerd-worker.log').open('a',encoding='utf-8');sys.stdout=log;sys.stderr=log
sys.path.insert(0,str(root/'layerd'));sys.path.insert(0,str(root))
import uvicorn
from app import app
uvicorn.run(app,host='127.0.0.1',port=__R10_PORT__,workers=1,access_log=False)
'''
    file=root/'layerd-boot.py';file.write_text(boot.replace('__R10_PORT__',str(port)),encoding='utf-8')
    pythonw=project/'.runtime/layerd-venv/Scripts/pythonw.exe'
    if not pythonw.is_file():raise ValueError('existing LayerD environment missing')
    action=f'"{pythonw}" "{file}"'
    result=subprocess.run(['schtasks','/Create','/TN',task,'/SC','ONLOGON','/RL','LIMITED','/IT','/TR',action],capture_output=True)
    if result.returncode:raise RuntimeError('LayerD task registration failed')
    subprocess.run(['schtasks','/Run','/TN',task],capture_output=True,check=True)
    ready=False
    for _ in range(80):
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/ready',timeout=2) as r:ready=r.status==200
            if ready:break
        except OSError:pass
        time.sleep(.5)
    record={'ready':ready,'task':task,'port':port,'old_services_untouched':True,
        'gpu_resource_root':str(root.parent/'r10-gpu-shared-leases'),'physical_gpu':'GPU-6617fc79-6516-2e9f-6d7b-76b1e9ec2f6d'}
    (root/'layerd-deployment.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--archive');p.add_argument('--port',type=int,default=18845)
    p.add_argument('--task-name',default='Image2PPT-Parallel-R10-LayerD-20261110');a=p.parse_args()
    deploy(a.root,archive=a.archive,port=a.port,task=a.task_name)
