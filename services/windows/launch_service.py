"""Scheduled-task foreground supervisor, with persistent local logs."""
from pathlib import Path
import os, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[2]
kind=sys.argv[1]
assert kind in ('layerd','lama')
env=os.environ.copy()
snapshot=ROOT/'.runtime/server38-snapshot'
models=snapshot/'image2ppt-layerd/models'
cache=ROOT/'.runtime/service-cache'
cache.mkdir(parents=True,exist_ok=True)
env.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1',
    'HF_HOME':str(cache/'huggingface'),'TORCH_HOME':str(cache/'torch'),
    'TEMP':str(cache),'TMP':str(cache),'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8',
    'LAYERD_MODELS':str(models),'LAMA_MODEL':str(models/'big-lama.pt'),
    'LAYERD_DATA':str(ROOT/'.runtime/layerd/data'),
    'LAYERD_TOKEN_FILE':str(ROOT/'.runtime/layerd/client-token'),
    'LAYERD_PROCESS_SIZE':'1024','LAYERD_MAX_ITERATIONS':'12',
    'LAYERD_REVISION':'21aef937a0371614adb4d961f52d02409cb8ecc7','OMP_NUM_THREADS':'4'})
python=ROOT/'.runtime/layerd-venv/Scripts/python.exe'
if kind=='layerd':
    args=[str(python),'-u','-m','uvicorn','app:app','--host','127.0.0.1','--port','18840','--workers','1','--no-access-log']
    cwd=ROOT/'services/layerd'
else:
    args=[str(python),'-u',str(ROOT/'services/lama/server.py'),'--port','18841','--model',str(snapshot/'image2ppt-lama/models/big-lama.pt')]
    cwd=ROOT/'services/lama'
logs=ROOT/'.runtime/logs'
logs.mkdir(exist_ok=True)
with (logs/f'{kind}.log').open('ab',buffering=0) as log:
    log.write(f'\nStarting {kind}: {time.ctime()}\n'.encode())
    result=subprocess.run(args,cwd=cwd,env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
raise SystemExit(result.returncode)
