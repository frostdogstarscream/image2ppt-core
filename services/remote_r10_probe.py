"""Read-only, allowlisted diagnostics of THIS project's GPU runtime."""
import json,subprocess,sys,csv,io,re,xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
root=Path('F:/code/image2ppt/outputs/auto-object-closure-service-20261109')
cfg=json.loads((root/'config.json').read_text(encoding='utf-8-sig'))
print(json.dumps({k:cfg.get(k) for k in ('jobs','python','source','code_revision')},ensure_ascii=False))
print(json.dumps({'sam3':{k:cfg['models']['sam3'].get(k) for k in ('checkpoint','sha256')}}))
for cmd in [['schtasks','/Query','/TN','Image2PPT-Private-SAM-Closure-20261109','/FO','CSV'],
            ['nvidia-smi','--query-gpu=name,uuid,memory.used,memory.total','--format=csv']]:
    p=subprocess.run(cmd,capture_output=True)
    print(p.stdout.decode('gbk',errors='replace')[:3000],flush=True)
tasks=subprocess.run(['schtasks','/Query','/FO','CSV','/NH'],capture_output=True)
print(json.dumps({'project_tasks':[r for r in csv.reader(io.StringIO(tasks.stdout.decode('gbk',errors='replace')))
    if r and r[0].lstrip('\\').lower().startswith('image2ppt')]}),flush=True)
print(json.dumps({'layerd_directories':[str(p) for p in root.parent.iterdir() if p.is_dir() and 'layerd' in p.name.lower()]}),flush=True)
for name in ('image2ppt-layerd-4090','image2ppt-lama-4090'):
    p=subprocess.run(['schtasks','/Query','/TN',name,'/XML'],capture_output=True)
    if p.returncode:continue
    tree=ET.fromstring(p.stdout.decode('utf-16') if p.stdout.startswith(b'\xff\xfe') else p.stdout.decode('gbk'))
    ns={'t':'http://schemas.microsoft.com/windows/2004/02/mit/task'}
    scripts=[]
    for node in tree.findall('.//t:Exec',ns):
        args=node.findtext('t:Arguments','',ns)
        scripts.extend(re.findall(r'[A-Za-z]:[/\\][^\s\"]+\.(?:py|ps1)',args))
        print(json.dumps({'task':name,'executable':node.findtext('t:Command','',ns),'working_directory':node.findtext('t:WorkingDirectory','',ns),'scripts':scripts}),flush=True)
