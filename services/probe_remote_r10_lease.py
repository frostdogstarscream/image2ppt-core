"""Exercise the deployed host/device lease across independent processes."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root))
os.environ['IMAGE2PPT_RESOURCE_ROOT']=str(root.parent/'r10-gpu-shared-leases')
os.environ['IMAGE2PPT_GPU_ID']='gpu-6617fc79-6516-2e9f-6d7b-76b1e9ec2f6d'
from resource_lease import lease
name='gpu-'+os.environ['IMAGE2PPT_GPU_ID']
code='import os,sys;sys.path.insert(0,sys.argv[1]);from resource_lease import lease\nwith lease(sys.argv[2],timeout=.3):print("unexpected overlap")'
with lease(name,timeout=300):
    p=subprocess.run([sys.executable,'-c',code,str(root),name],capture_output=True,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW)
result={'host_cross_process_exclusion_passed':p.returncode!=0 and b'resource wait exceeded' in p.stderr,
    'shared_lease_sha256':hashlib.sha256((root/'resource_lease.py').read_bytes()).hexdigest(),
    'device':os.environ['IMAGE2PPT_GPU_ID'],'old_services_modified':False,'model_accuracy_claimed':False}
(root/'lease-probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
