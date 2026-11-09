"""Loopback product UI and durable single-worker job queue. No Codex process needed."""
from __future__ import annotations
from contextlib import asynccontextmanager
import hashlib
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
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError
from .materialize import write, digest

ROOT=Path(__file__).resolve().parents[3]
TERMINAL={'SUCCEEDED','REVIEW_REQUIRED','QUALITY_FAILED','FAILED','INTERRUPTED'}


class Queue:
    def __init__(self, root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock();self.stop_event=threading.Event();self.process=None;self.thread=None
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
            db.execute("UPDATE jobs SET state='INTERRUPTED' WHERE state IN ('RUNNING','QUEUED')")
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()

    def loop(self):
        while not self.stop_event.wait(.2):
            with self.connect() as db:
                row=db.execute("SELECT id FROM jobs WHERE state='QUEUED' ORDER BY created LIMIT 1").fetchone()
            if not row:continue
            id=row[0];self.set(id,'RUNNING');job=self.root/id
            with (job/'worker.log').open('w',encoding='utf-8') as log:
                env=os.environ.copy()
                env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
                # Fixed module only; uploaded names and model output never become commands.
                self.process=subprocess.Popen([sys.executable,'-m','product.pipeline',str(job)],cwd=ROOT,
                    env=env,stdout=log,stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                deadline=time.monotonic()+max(3600,1800*len(json.loads((job/'request.json').read_text())['sources']))
                while self.process.poll() is None and not self.stop_event.wait(.2) and time.monotonic()<deadline:
                    pass
                if self.process.poll() is None:
                    self.kill()
                    self.set(id,'INTERRUPTED' if self.stop_event.is_set() else 'FAILED')
                else:
                    try:result=json.loads((job/'result.json').read_text(encoding='utf-8'))
                    except (OSError,ValueError):result={'status':'FAILED'}
                    self.set(id,result['status'] if result.get('status') in TERMINAL else 'FAILED')
                self.process=None

    def kill(self):
        if self.process and self.process.poll() is None:
            if os.name=='nt':
                subprocess.run(['taskkill','/PID',str(self.process.pid),'/T','/F'],capture_output=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:self.process.kill()
            self.process.wait(timeout=10)

    def stop(self):
        self.stop_event.set()
        if self.thread:self.thread.join(15)
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
        return result


def create_app(root=None, token=None, run_worker=True):
    root=Path(root or ROOT/'outputs/product-jobs').resolve()
    if not root.is_relative_to(ROOT):raise ValueError('jobs must stay under compiler workspace')
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
            limit=101*1024*1024 if request.url.path=='/api/jobs' else 8192
            if int(length)>limit:return JSONResponse({'detail':'请求体积超限'},status_code=413)
        return await call_next(request)
    def auth(authorization: str=Header(default='')):
        if not hmac.compare_digest(authorization,'Bearer '+token):raise HTTPException(401)
    @app.get('/',response_class=HTMLResponse)
    def home():return (Path(__file__).parent/'index.html').read_text(encoding='utf-8')
    @app.get('/api/health',dependencies=[Depends(auth)])
    def health():
        from .pipeline import settings
        try:settings();ready=True
        except (RuntimeError,OSError):ready=False
        return {'model_ready':ready and bool(os.environ.get('OPENCODE_GO_API_KEY')),'moma_configured':ready,'max_pages':12,'max_calls_per_page':None,'independent_worker':True,
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
    def submit(files:list[UploadFile]=File(...),object_extraction:str='default'):
        from .object_stage import route_setting
        try:object_route=route_setting(object_extraction)
        except ValueError:raise HTTPException(422,'未知对象提取模式') from None
        if object_route!='off' and not os.environ.get('LAYERD_TOKEN_FILE'):
            raise HTTPException(503,'对象提取需要已配置的 LayerD 服务')
        if not 1<=len(files)<=12:raise HTTPException(422,'请上传 1–12 张图片')
        from .pipeline import settings
        try:settings()
        except (RuntimeError,OSError):raise HTTPException(503,'未配置 MoMA 密钥，请使用启动器配置') from None
        if not os.environ.get('OPENCODE_GO_API_KEY'):raise HTTPException(503,'未配置 OpenCode Go 复核密钥')
        # Serialize admission, bounded decode and queue insertion.
        with queue.lock:
            with queue.connect() as db:
                if db.execute("SELECT count(*) FROM jobs WHERE state IN ('QUEUED','RUNNING')").fetchone()[0]>=2:
                    raise HTTPException(429,'已有两项任务，请等待完成')
            images=[];total=0;size=None
            for upload in files:
                data=upload.file.read(20*1024*1024+1);total+=len(data)
                if len(data)>20*1024*1024 or total>100*1024*1024:raise HTTPException(413,'图片体积超限')
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        if image.format not in ('PNG','JPEG','WEBP') or getattr(image,'n_frames',1)!=1:
                            raise ValueError()
                        w,h=image.size
                        if max(w,h)>4000 or w*h>12_000_000:raise ValueError()
                        if size is not None and image.size!=size:raise HTTPException(422,'同一批页面需相同尺寸')
                        size=image.size;images.append(image.convert('RGB'))
                except (OSError,ValueError,UnidentifiedImageError,Image.DecompressionBombError):
                    raise HTTPException(422,'图片无效，支持尺寸不超过 4000px 的单帧 PNG/JPEG/WebP') from None
            id=uuid4().hex;job=root/id;(job/'inputs').mkdir(parents=True)
            names=[]
            for i,image in enumerate(images,1):
                name=f'{i:02d}.png';image.save(job/'inputs'/name);names.append(name)
            write(job/'request.json',{'sources':names,'max_calls':None,'italic_difference_ignored':True,'object_extraction':object_route})
            with queue.connect() as db:db.execute('INSERT INTO jobs VALUES(?,?,?)',(id,'QUEUED',time.time()))
        return {'id':id,'status':'QUEUED','max_model_calls':None}
    @app.get('/api/jobs',dependencies=[Depends(auth)])
    def jobs():
        with queue.connect() as db:
            rows=db.execute('SELECT id,state,created FROM jobs ORDER BY created DESC LIMIT 20').fetchall()
        return {'jobs':[{'id':id,'status':state,'created':created} for id,state,created in rows]}
    @app.get('/api/jobs/{id}',dependencies=[Depends(auth)])
    def status(id:str):return queue.status(id)
    @app.get('/api/jobs/{id}/files/{name}',dependencies=[Depends(auth)])
    def artifact(id:str,name:str):
        status=queue.status(id)
        if status['status'] not in ('SUCCEEDED','REVIEW_REQUIRED'):raise HTTPException(409,'该任务尚无可交付文件')
        manifest=status.get('result',{}).get('artifacts',{})
        if name not in manifest or Path(name).name!=name:raise HTTPException(404)
        path=root/id/'delivery'/name
        if not path.is_file() or digest(path)!=manifest[name]:raise HTTPException(409,'文件校验失败')
        return FileResponse(path,filename=name)
    return app
