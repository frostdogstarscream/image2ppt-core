"""Hold this project's existing Windows GPU lock through an SSH pipe."""
from contextlib import contextmanager
import base64
import json
from pathlib import Path
import queue
import re
import subprocess
import threading


@contextmanager
def remote_gpu_lease(config,timeout=300):
    if not config:
        yield {'status':'UNCONFIGURED'}
        return
    required={'host','user','identity_file','known_hosts_file','host_key_alias','python','lock_file'}
    if set(config)!=required:raise ValueError('incomplete remote GPU lease configuration')
    for name in ('host','user','host_key_alias'):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',config[name]):raise ValueError('invalid SSH target')
    if not re.fullmatch(r'[A-Za-z]:/[A-Za-z0-9_./-]+',config['python']):raise ValueError('invalid remote Python path')
    for field in ('identity_file','known_hosts_file'):
        if not Path(config[field]).is_file():raise ValueError('missing SSH configuration file')
    script='''import pathlib,msvcrt,time,sys
p=pathlib.Path(LOCK_FILE)
assert p.parent.is_dir(), 'shared GPU lease directory missing'
f=p.open('a+b')
if f.tell()==0:f.write(b'0');f.flush()
deadline=time.monotonic()+WAIT_SECONDS
while True:
 f.seek(0)
 try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
 except OSError:
  if time.monotonic()>=deadline:raise TimeoutError('GPU lease busy')
  time.sleep(.1)
print('GPU_LEASE_ACQUIRED',flush=True)
try:sys.stdin.readline()
finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1);f.close()
'''.replace('LOCK_FILE',repr(config['lock_file'])).replace('WAIT_SECONDS',str(int(timeout)))
    encoded=base64.b64encode(script.encode()).decode()
    command=config['python']+' -u -c "import base64;exec(base64.b64decode(\''+encoded+'\'))"'
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','StrictHostKeyChecking=yes',
          '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2',
          '-o','HostKeyAlias='+config['host_key_alias'],'-o','UserKnownHostsFile='+config['known_hosts_file'],
          '-o','IdentitiesOnly=yes','-i',config['identity_file'],config['user']+'@'+config['host'],command]
    proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    signal=queue.Queue()
    threading.Thread(target=lambda:signal.put(proc.stdout.readline()),daemon=True).start()
    try:
        try:line=signal.get(timeout=timeout+15)
        except queue.Empty:raise TimeoutError('remote GPU lease acquisition timed out') from None
        if line.strip()!=b'GPU_LEASE_ACQUIRED':raise ValueError('remote GPU lease unavailable')
        yield {'status':'ACQUIRED','host':config['host'],'lock_file':config['lock_file']}
    finally:
        try:proc.communicate(input=b'\n',timeout=10)
        except (subprocess.TimeoutExpired,BrokenPipeError):proc.kill();proc.communicate()
