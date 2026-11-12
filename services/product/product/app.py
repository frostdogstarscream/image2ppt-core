"""Loopback product UI and durable single-worker job queue. No Codex process needed."""
from __future__ import annotations
from contextlib import asynccontextmanager
import hashlib
import asyncio
import hmac
import io
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
from uuid import uuid4
from fastapi import FastAPI, Depends, File, Header, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from .materialize import write, digest

ROOT=Path(__file__).resolve().parents[3]
TERMINAL={'SUCCEEDED','REVIEW_REQUIRED','QUALITY_FAILED','FAILED','INTERRUPTED','CANCELLED'}


class Queue:
    def __init__(self, root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock();self.stop_event=threading.Event();self.process=None;self.thread=None
        self.processes={};self.threads=[]
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, state TEXT, created REAL)')

    def connect(self):
        return sqlite3.connect(self.root/'jobs.sqlite',timeout=10)

    def set(self,id,state):
        with self.connect() as db:db.execute('UPDATE jobs SET state=? WHERE id=?',(state,id))

    def start(self):
        # Never duplicate or replay paid requests after restart.
        self.lease=(self.root/'worker.lock').open('a+b')
        if self.lease.tell()==0:self.lease.write(b'0');self.lease.flush()
        self.lease.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(self.lease.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with self.connect() as db:
            # r10 resumes immutable artifacts and call receipts; UNKNOWN calls
            # are never repeated by the worker's durable reservation ledger.
            for (id,) in db.execute("SELECT id FROM jobs WHERE state='RUNNING'").fetchall():
                resume=False
                if re.fullmatch(r'[a-f0-9]{32}',id):
                    try:
                        request=json.loads((self.root/id/'request.json').read_text(encoding='utf-8'))
                        resume=request.get('pipeline_mode') in ('parallel_review','parallel_shadow') and (self.root/id/'state/parallel.sqlite').is_file()
                        resume=resume or (request.get('orchestrator')=='opencode' and (self.root/id/'runtime/session.json').is_file())
                    except (OSError,ValueError):pass
                db.execute('UPDATE jobs SET state=? WHERE id=?',('QUEUED' if resume else 'INTERRUPTED',id))
        count=int(os.environ.get('IMAGE2PPT_JOB_WORKERS','1'))
        if count not in (1,2):raise ValueError('job workers must be 1 or 2')
        self.threads=[threading.Thread(target=self.loop,daemon=True) for _ in range(count)]
        self.thread=self.threads[0]
        for thread in self.threads:thread.start()

    def take(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute("SELECT id FROM jobs WHERE state='QUEUED' ORDER BY created LIMIT 1").fetchone()
            if row:db.execute("UPDATE jobs SET state='RUNNING' WHERE id=?",(row[0],))
        return row[0] if row else None

    def loop(self):
        while not self.stop_event.wait(.2):
            id=self.take()
            if not id:continue
            job=self.root/id
            with (job/'worker.log').open('w',encoding='utf-8') as log:
                env=os.environ.copy()
                env['PYTHONPATH']=os.pathsep.join(dict.fromkeys([str(Path(__file__).resolve().parents[1]),
                    *filter(None,env.get('PYTHONPATH','').split(os.pathsep))]))
                # Fixed module only; uploaded names and model output never become commands.
                process=subprocess.Popen([sys.executable,'-m','product.parallel_pipeline',str(job)],cwd=ROOT,
                    env=env,stdout=log,stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                with self.lock:self.processes[id]=process
                request=json.loads((job/'request.json').read_text())
                deadline=time.monotonic()+(request.get('runtime_seconds',7200)+120 if request.get('orchestrator')=='opencode'
                                           else max(3600,1800*len(request['sources'])))
                while process.poll() is None and not self.stop_event.wait(.2) and time.monotonic()<deadline:
                    pass
                if process.poll() is None:
                    self.kill(process)
                    self.set(id,'INTERRUPTED' if self.stop_event.is_set() else 'FAILED')
                else:
                    try:result=json.loads((job/'result.json').read_text(encoding='utf-8'))
                    except (OSError,ValueError):result={'status':'FAILED'}
                    self.set(id,result['status'] if result.get('status') in TERMINAL else 'FAILED')
                with self.lock:self.processes.pop(id,None)

    def kill(self,process=None):
        if process is None:
            with self.lock:processes=list(self.processes.values())
            for process in processes:self.kill(process)
            return
        if process.poll() is None:
            if os.name=='nt':
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:process.kill()
            process.wait(timeout=10)

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:thread.join(15)
        self.kill()
        if hasattr(self,'lease'):self.lease.close()

    def status(self,id):
        if not re.fullmatch(r'[a-f0-9]{32}',id):raise HTTPException(404)
        with self.connect() as db:row=db.execute('SELECT state FROM jobs WHERE id=?',(id,)).fetchone()
        if not row:raise HTTPException(404)
        result={'id':id,'status':row[0]}
        for name in ('progress','result'):
            file=self.root/id/(name+'.json')
            if file.exists():result[name]=json.loads(file.read_text(encoding='utf-8'))
        runtime=self.root/id/'runtime/state.json'
        if runtime.exists():
            state=json.loads(runtime.read_text('utf8'))
            committed=[]
            for page in state['pages']:
                scene=Path(page['scene']) if page else None
                if scene and scene.resolve().is_relative_to(self.root/id) and scene.exists():
                    committed.append(sorted(json.loads(scene.read_text('utf8')).get('selected',{})))
                else:committed.append([])
            result['runtime']={'version':state['version'],'candidates':len(state['candidates']),
                'candidate_states':{name:sum(c['status']==name for c in state['candidates'].values())
                                    for name in sorted({c['status'] for c in state['candidates'].values()})},
                'committed_objects':committed,'recall':None,
                'visual_qa':state.get('visual_review',{}).get('status','UNKNOWN')}
        return result


def create_app(root=None, token=None, run_worker=True):
    root=Path(root or ROOT/'outputs/product-jobs').resolve()
    configured_root=os.environ.get('IMAGE2PPT_RUNTIME_JOBS_ROOT')
    if not root.is_relative_to(ROOT) and (not configured_root or root!=Path(configured_root).resolve()):
        raise ValueError('jobs must stay under compiler workspace or explicit runtime job root')
    token=token or os.environ.get('IMAGE2PPT_PRODUCT_TOKEN')
    if not token or len(token)<24:raise ValueError('IMAGE2PPT_PRODUCT_TOKEN needs at least 24 characters')
    queue=Queue(root)
    @asynccontextmanager
    async def lifespan(app):
        if run_worker:queue.start()
        yield
        if run_worker:queue.stop()
    app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
    @app.middleware('http')
    async def bound_upload(request,call_next):
        if request.method=='POST':
            from fastapi.responses import JSONResponse
            length=request.headers.get('content-length','')
            if not length.isdigit():return JSONResponse({'detail':'需要 Content-Length'},status_code=411)
            limit=101*1024*1024 if request.url.path in ('/api/jobs','/api/v1/jobs') else 8192
            if request.url.path=='/api/v1/office/result':limit=512*1024*1024
            if int(length)>limit:return JSONResponse({'detail':'请求体积超限'},status_code=413)
        return await call_next(request)
    def auth(authorization: str=Header(default='')):
        if not hmac.compare_digest(authorization,'Bearer '+token):raise HTTPException(401)
    def office_auth(authorization: str=Header(default='')):
        worker_token=os.environ.get('IMAGE2PPT_OFFICE_WORKER_TOKEN','')
        if len(worker_token)<24 or not hmac.compare_digest(authorization,'Bearer '+worker_token):raise HTTPException(401)
    from .runtime.office_bridge import OfficeQueue
    office_queue=OfficeQueue(root)
    @app.post('/api/v1/office/claim',dependencies=[Depends(office_auth)])
    def office_claim(x_image2ppt_worker_binding:str=Header(default='')):
        if x_image2ppt_worker_binding:
            if not re.fullmatch('[a-f0-9]{64}',x_image2ppt_worker_binding):raise HTTPException(422,'Invalid worker binding')
            write(root/'office-worker-binding.json',{'binding':x_image2ppt_worker_binding})
        return office_queue.claim()
    @app.post('/api/v1/office/input',dependencies=[Depends(office_auth)])
    async def office_input(request:Request):
        from fastapi.responses import Response
        try:
            data=await request.json();directory=office_queue.directory(data['ticket'],data['token'])
        except (ValueError,KeyError,OSError):raise HTTPException(409,'Office lease unavailable') from None
        return Response((directory/'input.zip').read_bytes(),media_type='application/zip')
    @app.post('/api/v1/office/result',dependencies=[Depends(office_auth)])
    async def office_result(request:Request,ticket:str,token:str):
        try:directory=office_queue.directory(ticket,token)
        except (ValueError,OSError):raise HTTPException(409,'Office lease unavailable') from None
        data=await request.body()
        temporary=directory/'output-upload.zip'
        temporary.write_bytes(data)
        temporary.replace(directory/'output.zip')
        return {'status':'RECEIVED'}
    @app.get('/',response_class=HTMLResponse)
    def home():
        page=(Path(__file__).parent/'index.html').read_text(encoding='utf-8')
        if os.environ.get('IMAGE2PPT_DEFAULT_ORCHESTRATOR')=='opencode':
            page=page.replace('MoMA 分别判断文字、素材与背景，合并后生成 PPT；OpenCode Go 对照原图复核。无需 Codex 对话。',
                              'OpenCode 理解图片，自主选择重建工具，检查实际编辑结果并生成 PPT。')
            page=page.replace('<option value="opencode_go">OpenCode Go · 复核</option>','')
            page=page.replace('OpenCode Go','OpenCode 自检').replace('每批 1–12 页','每批 1–32 页')
        return page
    @app.get('/api/health',dependencies=[Depends(auth)])
    def health():
        from .pipeline import settings
        try:settings();ready=True
        except (RuntimeError,OSError):ready=False
        runtime_ready=ready and bool(os.environ.get('IMAGE2PPT_OPENCODE_URL'))
        agent_provider=os.environ.get('IMAGE2PPT_AGENT_PROVIDER','moma')
        return {'model_ready':ready and (bool(os.environ.get('OPENCODE_GO_API_KEY')) or runtime_ready),'moma_configured':ready and agent_provider=='moma','max_pages':12,'opencode_max_pages':32,'max_calls_per_page':None,'independent_worker':True,
                'agent_inference_provider':agent_provider,
                'opencode_runtime_configured':runtime_ready,
                'opencode_go_configured':bool(os.environ.get('OPENCODE_GO_API_KEY')),'italic_difference_ignored':True}
    @app.post('/api/provider-key',dependencies=[Depends(auth)])
    async def provider_key(request:Request):
        try:
            data=await request.json();key=data.get('key','')
            if not isinstance(key,str) or not 8<=len(key)<=4096 or any(ord(c)<33 or ord(c)>126 for c in key):
                raise ValueError()
        except (ValueError,AttributeError):raise HTTPException(422,'密钥格式无效') from None
        provider=data.get('provider','moma')
        if provider=='moma':
            os.environ['MOMA_API_KEY']=key
            os.environ['PLANNER_API_KEY']=key
        elif provider=='opencode_go':os.environ['OPENCODE_GO_API_KEY']=key
        else:raise HTTPException(422,'未知模型供应商')
        return {'configured':True,'storage':'process_memory'}
    @app.post('/api/jobs',status_code=202,dependencies=[Depends(auth)])
    @app.post('/api/v1/jobs',status_code=202,dependencies=[Depends(auth)])
    def submit(request:Request,files:list[UploadFile]=File(...),object_extraction:str='default',pipeline_mode:str='parallel_review',
               orchestrator:str='pipeline',goal:str='Restore these images as an editable PPTX.',fresh_perception:bool=True,
               agent_steps:int=64,runtime_seconds:int=7200,candidate_limit:int=4,model_calls:int=12,office_validations:int=60,tool_profile:str='all',agent_thinking:bool=True):
        if request.url.path=='/api/v1/jobs' and orchestrator=='pipeline':orchestrator='opencode'
        if os.environ.get('IMAGE2PPT_DEFAULT_ORCHESTRATOR')=='opencode' and orchestrator=='pipeline':orchestrator='opencode'
        if orchestrator not in ('pipeline','opencode'):raise HTTPException(422,'未知执行器')
        if tool_profile not in ('all','basic'):raise HTTPException(422,'未知工具目录')
        from .parallel_contracts import strategy,mode
        try:object_route=strategy(object_extraction);selected_mode=mode(pipeline_mode)
        except ValueError:raise HTTPException(422,'未知对象提取模式') from None
        max_pages=32 if orchestrator=='opencode' else 12
        if not 1<=len(files)<=max_pages:raise HTTPException(422,f'请上传 1–{max_pages} 张图片')
        if not (1<=agent_steps<=128 and 60<=runtime_seconds<=21600 and 1<=candidate_limit<=12
                and 0<=model_calls<=200 and 1<=office_validations<=200):raise HTTPException(422,'执行预算超出支持范围')
        from .pipeline import settings
        try:settings()
        except (RuntimeError,OSError):raise HTTPException(503,'未配置 MoMA 密钥，请使用启动器配置') from None
        if orchestrator!='opencode' and not os.environ.get('OPENCODE_GO_API_KEY'):raise HTTPException(503,'未配置 OpenCode Go 复核密钥')
        if not 1<=len(goal)<=4000:raise HTTPException(422,'交付目标长度无效')
        # Serialize admission, bounded decode and queue insertion.
        with queue.lock:
            with queue.connect() as db:
                if db.execute("SELECT count(*) FROM jobs WHERE state IN ('QUEUED','RUNNING')").fetchone()[0]>=2:
                    raise HTTPException(429,'已有两项任务，请等待完成')
            images=[];originals=[];total=0;size=None
            for upload in files:
                data=upload.file.read(20*1024*1024+1);total+=len(data)
                if len(data)>20*1024*1024 or total>100*1024*1024:raise HTTPException(413,'图片体积超限')
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        if image.format not in ('PNG','JPEG','WEBP') or getattr(image,'n_frames',1)!=1:
                            raise ValueError()
                        format=image.format.lower();image=ImageOps.exif_transpose(image)
                        w,h=image.size
                        if max(w,h)>4000 or w*h>12_000_000:raise ValueError()
                        if size is not None and image.size!=size:raise HTTPException(422,'同一批页面需相同尺寸')
                        size=image.size;images.append(image.convert('RGB'));originals.append((data,format))
                except (OSError,ValueError,UnidentifiedImageError,Image.DecompressionBombError):
                    raise HTTPException(422,'图片无效，支持尺寸不超过 4000px 的单帧 PNG/JPEG/WebP') from None
            id=uuid4().hex;job=root/id;(job/'inputs').mkdir(parents=True)
            names=[];original_refs=[];(job/'originals').mkdir()
            for i,image in enumerate(images,1):
                name=f'{i:02d}.png';image.save(job/'inputs'/name);names.append(name)
                data,format=originals[i-1];original=job/'originals'/f'{i:02d}.{format}';original.write_bytes(data)
                original_refs.append({'path':str(original.relative_to(job)),'sha256':digest(original),
                                      'canonical_file':name,'canonical_sha256':digest(job/'inputs'/name)})
            write(job/'request.json',{'sources':names,'max_calls':None,'italic_difference_ignored':True,
                                     'agent_inference_provider':os.environ.get('IMAGE2PPT_AGENT_PROVIDER','moma'),
                                     'agent_model':os.environ.get('IMAGE2PPT_AGENT_MODEL'),
                                     'object_extraction':object_route,'pipeline_mode':selected_mode,'original_inputs':original_refs,
                                     'orchestrator':orchestrator,'visual_review_backend':'opencode' if orchestrator=='opencode' else 'go',
                                     'goal':goal,'fresh_perception':fresh_perception,'candidate_engine':'r12-regional' if orchestrator=='opencode' else 'off',
                                     'agent_steps':agent_steps,'runtime_seconds':runtime_seconds,
                                     'tool_profile':tool_profile,
                                     'agent_thinking':agent_thinking,
                                     **({'adaptive_budget':{'region_candidates':min(2,candidate_limit),'complex_region_candidates':candidate_limit,
                                         'model_calls':model_calls,'office_validations':office_validations,'max_seconds':runtime_seconds}} if orchestrator=='opencode' else {})})
            with queue.connect() as db:db.execute('INSERT INTO jobs VALUES(?,?,?)',(id,'QUEUED',time.time()))
        return {'id':id,'status':'QUEUED','max_model_calls':None}
    @app.get('/api/jobs',dependencies=[Depends(auth)])
    def jobs():
        with queue.connect() as db:
            rows=db.execute('SELECT id,state,created FROM jobs ORDER BY created DESC LIMIT 20').fetchall()
        return {'jobs':[{'id':id,'status':state,'created':created} for id,state,created in rows]}
    @app.get('/api/jobs/{id}',dependencies=[Depends(auth)])
    @app.get('/api/v1/jobs/{id}',dependencies=[Depends(auth)])
    def status(id:str):return queue.status(id)
    @app.get('/api/v1/jobs/{id}/events',dependencies=[Depends(auth)])
    async def events(id:str,request:Request,after:int=0):
        queue.status(id)
        async def stream():
            cursor=max(0,after)
            while not await request.is_disconnected():
                file=root/id/'runtime/runtime.sqlite'
                rows=[]
                if file.exists():
                    with sqlite3.connect(file,timeout=10) as db:
                        rows=db.execute('SELECT id,payload FROM events WHERE id>? ORDER BY id LIMIT 100',(cursor,)).fetchall()
                for cursor,payload in rows:yield f'id: {cursor}\ndata: {payload}\n\n'
                current=queue.status(id)
                if current['status'] in TERMINAL:
                    yield 'event: complete\ndata: '+json.dumps(current,ensure_ascii=False)+'\n\n'
                    break
                if not rows:yield ': heartbeat\n\n'
                await asyncio.sleep(1)
        return StreamingResponse(stream(),media_type='text/event-stream')
    @app.post('/api/v1/jobs/{id}/cancel',dependencies=[Depends(auth)])
    def cancel(id:str):
        current=queue.status(id)
        if current['status'] in TERMINAL:return current
        write(root/id/'cancel.requested',{'requested':time.time()})
        if current['status']=='QUEUED':queue.set(id,'CANCELLED')
        return {'id':id,'status':'CANCEL_REQUESTED'}
    @app.get('/api/v1/jobs/{id}/artifacts',dependencies=[Depends(auth)])
    def artifacts(id:str):
        current=queue.status(id)
        return {'id':id,'status':current['status'],'artifacts':[
            {'name':name,'sha256':sha,'url':f'/api/jobs/{id}/files/{name}'}
            for name,sha in current.get('result',{}).get('artifacts',{}).items()]}
    @app.get('/api/jobs/{id}/files/{name}',dependencies=[Depends(auth)])
    def artifact(id:str,name:str):
        status=queue.status(id)
        if status['status'] not in ('SUCCEEDED','REVIEW_REQUIRED'):raise HTTPException(409,'该任务尚无可交付文件')
        if status.get('result',{}).get('pipeline_mode','').startswith('parallel_') and status.get('result',{}).get('visual_review_backend')!='opencode':
            from .parallel_store import Store
            try:
                committed=Store(root/id/'state').committed()
                if committed is None or committed['result']!=status['result']:raise ValueError()
            except (ValueError,OSError):raise HTTPException(409,'可交付版本校验失败') from None
        manifest=status.get('result',{}).get('artifacts',{})
        if name not in manifest or Path(name).name!=name:raise HTTPException(404)
        path=root/id/'delivery'/name
        if not path.is_file() or digest(path)!=manifest[name]:raise HTTPException(409,'文件校验失败')
        return FileResponse(path,filename=name)
    return app
