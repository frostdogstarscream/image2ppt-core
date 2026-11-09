"""Private serialized SAM jobs. Model paths come exclusively from server config."""
import argparse
import contextlib
import hashlib
import hmac
import io
import ipaddress
import json
import os
from pathlib import Path
import queue
import subprocess
import shutil
import sys
import threading
import time
from uuid import uuid4
from fastapi import FastAPI,Depends,File,Form,Header,HTTPException,UploadFile
from fastapi.responses import FileResponse
from PIL import Image


def sha(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(file,value):
    tmp=file.with_name(file.name+'.tmp-'+uuid4().hex);tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    try:
        for attempt in range(50):
            try:tmp.replace(file);break
            except PermissionError:
                if attempt==49:raise
                time.sleep(.02)
    finally:tmp.unlink(missing_ok=True)


def canonical(plan,source_hash,model):
    if plan.get('source_sha256')!=source_hash:raise ValueError('source mismatch')
    size=plan['image_size'];definitions={};mapping={}
    if len(size)!=2 or not all(type(v)==int and 0<v<=4000 for v in size) or size[0]*size[1]>12_000_000:raise ValueError('invalid size')
    if not 1<=len(plan['objects'])<=100:raise ValueError('invalid object count')
    for obj in plan['objects']:
        bbox=obj['bbox'];prompt=obj['prompt_en'];identifier=obj['id']
        if not isinstance(identifier,str) or identifier in mapping:raise ValueError('duplicate object')
        if len(bbox)!=4 or not all(type(v)==int for v in bbox):raise ValueError('invalid bbox')
        x,y,w,h=bbox
        if min(x,y)<0 or min(w,h)<=0 or x+w>size[0] or y+h>size[1]:raise ValueError('bbox outside canvas')
        if not isinstance(prompt,str) or not 1<=len(prompt)<=300:raise ValueError('invalid prompt')
        value={'bbox':bbox,'prompt_en':prompt};key=identity(value);definitions[key]=value;mapping[identifier]=key
    ordered=sorted(definitions);names={key:f'obj-{i:03d}' for i,key in enumerate(ordered)}
    output={'source_sha256':source_hash,'image_size':size,'objects':[{'id':names[k],**definitions[k]} for k in ordered]}
    key=identity({'source':source_hash,'objects':output['objects'],'size':size,'model':model,'adapter':'private-sam/1'})
    return key,output,{k:names[v] for k,v in mapping.items()}


def create_app(config,*,runner=None):
    root=Path(config['jobs']).resolve();root.mkdir(parents=True,exist_ok=True)
    token=Path(config['token_file']).read_text(encoding='utf-8').strip()
    if len(token)<32:raise ValueError('SAM token too short')
    tasks=queue.Queue();lock=threading.Lock();stop=threading.Event()
    def read(job):return json.loads((job/'status.json').read_text(encoding='utf-8'))
    def object_cache(job):
        plan=json.loads((job/'plan.json').read_text(encoding='utf-8'));request=json.loads((job/'request.json').read_text(encoding='utf-8'))
        model={'version':request['version'],'sha256':config['models'][request['version']]['sha256'],'revision':config['code_revision']}
        return [(obj,identity({'source':plan['source_sha256'],'size':plan['image_size'],'bbox':obj['bbox'],'prompt':obj['prompt_en'],'model':model})) for obj in plan['objects']]
    def restore_objects(job):
        result=job/'result';hits=0
        for obj,key in object_cache(job):
            cached=root/'object-cache'/key
            if not (cached/'binding.json').exists():continue
            binding=json.loads((cached/'binding.json').read_text(encoding='utf-8'))
            if sha(cached/'mask.npz')!=binding['mask_sha256'] or sha(cached/'raw.json')!=binding['receipt_sha256']:raise ValueError('object cache corrupted')
            result.mkdir(exist_ok=True)
            name=obj['id'];target=result/(name+'.npz');receipt=result/(name+'.json')
            if receipt.exists():continue  # single_frame --resume verifies it
            shutil.copyfile(cached/'mask.npz',target);shutil.copyfile(cached/'raw.json',result/('cache-raw-'+name+'.json'))
            raw=json.loads((cached/'raw.json').read_text(encoding='utf-8'))
            save(receipt,{**raw,'plan_sha256':sha(job/'plan.json'),'mask_file':target.name,
                'adapter':'SamObjectCache/1','cache_key':key,'raw_receipt_sha256':binding['receipt_sha256']})
            hits+=1
        return hits
    def retain_objects(job):
        for obj,key in object_cache(job):
            cached=root/'object-cache'/key
            if (cached/'binding.json').exists():continue
            cached.mkdir(parents=True,exist_ok=True);result=job/'result';receipt=result/(obj['id']+'.json')
            meta=json.loads(receipt.read_text(encoding='utf-8'));mask=(result/meta['mask_file']).resolve()
            if not mask.is_relative_to(result.resolve()) or sha(mask)!=meta['mask_sha256']:raise ValueError('result cache binding mismatch')
            shutil.copyfile(mask,cached/'mask.npz');shutil.copyfile(receipt,cached/'raw.json')
            save(cached/'binding.json',{'mask_sha256':sha(mask),'receipt_sha256':sha(receipt),'producer_job':job.name})
    def verify_results(job):
        plan=json.loads((job/'plan.json').read_text(encoding='utf-8'));request=json.loads((job/'request.json').read_text(encoding='utf-8'))
        result=(job/'result').resolve();version=request['version']
        expected={'status':'SUCCEEDED','source_sha256':sha(job/'source.png'),'plan_sha256':sha(job/'plan.json'),'image_size':plan['image_size'],
            'version':version,'checkpoint_sha256':config['models'][version]['sha256'],'code_revision':config['code_revision']}
        for obj in plan['objects']:
            meta=json.loads((result/(obj['id']+'.json')).read_text(encoding='utf-8'));file=(result/meta['mask_file']).resolve()
            if any(meta.get(k)!=v for k,v in expected.items()) or not file.is_relative_to(result) or sha(file)!=meta.get('mask_sha256'):
                raise ValueError('completed object receipt invalid')
    def execute(job):
        request=json.loads((job/'request.json').read_text(encoding='utf-8'));version=request['version'];model=config['models'][version]
        if runner:return runner(job,request)
        hits=restore_objects(job)
        # If every object was already completed, no GPU subprocess is launched.
        plan=json.loads((job/'plan.json').read_text(encoding='utf-8'))
        if all((job/'result'/(o['id']+'.json')).exists() for o in plan['objects']):
            verify_results(job)
            save(job/'result/run.json',{'status':'SUCCEEDED','cache_only':True,'object_cache_hits':hits});return
        args=[config.get('python',sys.executable),str(Path(__file__).with_name('single_frame.py')),'--version',version,
            '--source',str(job/'source.png'),'--plan',str(job/'plan.json'),'--output',str(job/'result'),
            '--sam-source',config['source'],'--code-revision',config['code_revision'],
            '--checkpoint',model['checkpoint'],'--checkpoint-sha256',model['sha256'],'--resume']
        with (job/'worker.log').open('ab') as log:
            subprocess.run(args,stdout=log,stderr=log,check=True,timeout=1900,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    def work():
        while not stop.is_set():
            try:job=tasks.get(timeout=.2)
            except queue.Empty:continue
            state=read(job);state.update(status='RUNNING');save(job/'status.json',state)
            try:
                result=job/'result'
                if (result/'run.json').exists():
                    shutil.copyfile(result/'run.json',job/('run-history-'+uuid4().hex+'.json'))
                execute(job)
                artifacts={p.name:sha(p) for p in (job/'result').iterdir() if p.is_file() and p.suffix in ('.json','.npz')}
                if not artifacts:raise ValueError('missing SAM results')
                meta=json.loads((job/'result/run.json').read_text(encoding='utf-8'))
                if meta.get('status')!='SUCCEEDED':raise ValueError('SAM incomplete')
                if not runner:verify_results(job);retain_objects(job)
                state.update(status='SUCCEEDED',artifacts=artifacts)
            except Exception as exc:state.update(status='FAILED',error_type=type(exc).__name__)
            save(job/'status.json',state);tasks.task_done()
    @contextlib.asynccontextmanager
    async def lifespan(app):
        # OS-held lease releases on process death, so recovery does not require
        # deleting an ambiguous stale lock file or permitting a second worker.
        lease=(root/'service.lock').open('a+b');lease.seek(0);lease.write(b'1');lease.flush();lease.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lease.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for job in list(root.iterdir()):
            if job.is_dir() and len(job.name)==64 and all(c in '0123456789abcdef' for c in job.name) and (job/'status.json').exists():
                state=read(job)
                if state['status'] in ('QUEUED','RUNNING'):tasks.put(job)
                elif state['status']=='SUCCEEDED' and not runner:
                    try:verify_results(job);retain_objects(job)
                    except (ValueError,KeyError,OSError):save(job/'cache-invalid.json',{'reusable':False})
        thread=threading.Thread(target=work,daemon=True);app.state.worker=thread;thread.start()
        try:yield
        finally:
            stop.set();thread.join(timeout=2)
            if not thread.is_alive():lease.close()
    app=FastAPI(lifespan=lifespan)
    def auth(authorization:str=Header(default='')):
        if not hmac.compare_digest(authorization,'Bearer '+token):raise HTTPException(401,'unauthorized')
    @app.get('/v1/capabilities',dependencies=[Depends(auth)])
    def capabilities():
        return {'schema':'PrivateSam/1','serial_gpu':True,'worker_alive':getattr(app.state,'worker',None) is not None and app.state.worker.is_alive(),
            'models':{k:{'sha256':v['sha256'],'code_revision':config['code_revision']} for k,v in config['models'].items()}}
    @app.post('/v1/jobs',dependencies=[Depends(auth)],status_code=202)
    def submit(file:UploadFile=File(...),plan:str=Form(...),version:str=Form('sam3')):
        if version not in config['models']:raise HTTPException(422,'model unavailable')
        data=file.file.read(20*1024*1024+1)
        if len(data)>20*1024*1024 or len(plan)>2_000_000:raise HTTPException(413,'input too large')
        try:
            model={**config['models'][version],'version':version,'code_revision':config['code_revision']};model.pop('checkpoint',None)
            key,canonical_plan,mapping=canonical(json.loads(plan),hashlib.sha256(data).hexdigest(),model)
            with Image.open(io.BytesIO(data)) as image:
                if list(image.size)!=canonical_plan['image_size'] or image.format!='PNG' or getattr(image,'n_frames',1)!=1:raise ValueError()
        except (ValueError,KeyError,TypeError,OSError):raise HTTPException(422,'invalid bound PNG/plan') from None
        with lock:
            job=root/key
            if not job.exists():
                if tasks.qsize()>=24:raise HTTPException(429,'queue full')
                staged=root/('.incoming-'+uuid4().hex);staged.mkdir();(staged/'source.png').write_bytes(data);save(staged/'plan.json',canonical_plan)
                save(staged/'request.json',{'version':version,'created':time.time()})
                save(staged/'status.json',{'id':key,'status':'QUEUED','artifacts':{}});staged.rename(job);tasks.put(job)
        return {**read(job),'mapping':mapping}
    def job_dir(identifier):
        if len(identifier)!=64 or any(c not in '0123456789abcdef' for c in identifier):raise HTTPException(404,'unknown job')
        job=root/identifier
        if not (job/'status.json').exists():raise HTTPException(404,'unknown job')
        return job
    @app.get('/v1/jobs/{identifier}',dependencies=[Depends(auth)])
    def status(identifier:str):return read(job_dir(identifier))
    @app.get('/v1/jobs/{identifier}/artifacts/{filename}',dependencies=[Depends(auth)])
    def artifact(identifier:str,filename:str):
        job=job_dir(identifier);state=read(job)
        if state['status']!='SUCCEEDED' or filename not in state['artifacts'] or Path(filename).name!=filename:raise HTTPException(404,'unknown artifact')
        path=job/'result'/filename
        if sha(path)!=state['artifacts'][filename]:raise HTTPException(409,'artifact corrupted')
        return FileResponse(path)
    return app


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=18843);args=p.parse_args()
    address=ipaddress.ip_address(args.host)
    if not address.is_private or address.is_unspecified:raise ValueError('bind only an explicit private/loopback address')
    import uvicorn
    config=json.loads(args.config.read_text(encoding='utf-8'))
    logging={'version':1,'disable_existing_loggers':False,'formatters':{'plain':{'format':'%(asctime)s %(levelname)s %(message)s'}},
        'handlers':{'file':{'class':'logging.FileHandler','filename':str(args.config.with_suffix('.log')),'encoding':'utf-8','formatter':'plain'}},
        'loggers':{'uvicorn':{'handlers':['file'],'level':'INFO','propagate':False},'uvicorn.error':{'level':'INFO'},'uvicorn.access':{'level':'WARNING'}}}
    uvicorn.run(create_app(config),host=args.host,port=args.port,access_log=False,log_config=logging)
