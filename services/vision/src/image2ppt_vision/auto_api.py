"""Loopback-only upload/status/download UI for opt-in automatic validation."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
import secrets
from uuid import uuid4

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from image2ppt.storage import read_json, digest
from .auto_transport import load_config
from .auto_v2 import create_job, run_job


def create_app(root, config_file, project_root, *, runner=run_job, cached_preflight=None, resume_pending=True):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    config = load_config(config_file)
    secret = secrets.token_urlsafe(32)
    executor = None
    active = set()

    def submit(job):
        if job.name in active:
            return
        active.add(job.name)
        def task():
            try:
                runner(job, project_root=project_root, preflight_directory=cached_preflight)
            finally:
                active.discard(job.name)
        executor.submit(task)

    @asynccontextmanager
    async def lifespan(app):
        nonlocal executor
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='image2ppt-auto')
        for job in root.glob('auto-*') if resume_pending else []:
            if ((job/'state.json').is_file() and (job/'job.json').is_file()
                and read_json(job/'job.json').get('pipeline_version') == 2
                and read_json(job/'state.json').get('status') in ('QUEUED', 'RUNNING')):
                submit(job)
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title='image2PPT region-routed automatic validation', version='0.2.0', lifespan=lifespan)

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        host = request.url.hostname
        if host not in ('127.0.0.1', 'localhost', '::1'):
            return HTMLResponse('Loopback host required', status_code=403)
        origin = request.headers.get('origin')
        if origin and origin != str(request.base_url).rstrip('/'):
            return HTMLResponse('Same-origin requests required', status_code=403)
        return await call_next(request)

    def authorize(token):
        if not secrets.compare_digest(token or '', secret):
            raise HTTPException(403, 'local session token required')

    def job_path(identifier):
        if not re.fullmatch(r'auto-[a-f0-9]{12}', identifier):
            raise HTTPException(404, 'job not found')
        job = root/identifier
        if not (job/'state.json').is_file():
            raise HTTPException(404, 'job not found')
        return job

    @app.get('/', response_class=HTMLResponse)
    def index():
        page = Path(__file__).with_name('auto_ui.html').read_text(encoding='utf-8')
        return page.replace('__SESSION_TOKEN__', secret)

    @app.get('/health')
    def health():
        return {'status': 'alive', 'execution': 'server_models_local_compiler', 'pipeline_version': 2}

    @app.get('/v1/auto-jobs')
    def jobs(x_local_token: str = Header(default='')):
        authorize(x_local_token)
        return [{'job_id': p.name, **read_json(p/'state.json')} for p in sorted(root.glob('auto-*'), reverse=True)
                if (p/'state.json').is_file()][:100]

    @app.post('/v1/auto-jobs', status_code=202)
    async def create(files: list[UploadFile] = File(...), preference: str = Form('faithful_editable'),
                     gate_pages: str = Form(''), x_local_token: str = Header(default='')):
        authorize(x_local_token)
        if not 1 <= len(files) <= 200:
            raise HTTPException(422, 'expected 1 to 200 images')
        incoming = root/'incoming'/uuid4().hex; incoming.mkdir(parents=True)
        paths = []
        try:
            for i, file in enumerate(files):
                name = re.sub(r'[^\w.\-]', '_', Path(file.filename or 'page.png').name)[:100]
                target = incoming/(f'{i+1:03d}-'+name)
                size = 0
                with target.open('xb') as out:
                    while chunk := await file.read(1024*1024):
                        size += len(chunk)
                        if size > 20_000_000:
                            raise HTTPException(413, 'image exceeds 20 MB')
                        out.write(chunk)
                paths.append(target)
            gate = [int(v) for v in gate_pages.split(',')] if gate_pages else None
            job = create_job(paths, root, config, gate_pages=gate, preference=preference)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, 'invalid source images or job settings') from exc
        finally:
            for file in files:
                await file.close()
        submit(job)
        return {'job_id': job.name, 'status': 'QUEUED', 'status_url': '/v1/auto-jobs/'+job.name}

    @app.get('/v1/auto-jobs/{identifier}')
    def status(identifier: str, x_local_token: str = Header(default='')):
        authorize(x_local_token)
        job = job_path(identifier)
        result = read_json(job/'state.json')
        verified = job/'delivery/verified-summary.json'
        if verified.is_file():
            summary = read_json(verified)
            if summary.get('schema') == 'AutomaticEvidenceVerification/1':
                for row in summary['pages']:
                    page = result.get('pages', {}).get(row['page'])
                    if page is not None:
                        page['recorded_execution_status'] = page['execution_status']
                        page['execution_status'] = row['pipeline_status']
                        page['pipeline_completed'] = row['validated_model_review']
        current = result.get('current_page')
        if current and (job/current/'progress.json').is_file():
            result['page_progress'] = read_json(job/current/'progress.json')
        if (job/'delivery').is_dir():
            result['artifacts'] = [{'name': p.name, 'sha256': digest(p)} for p in (job/'delivery').iterdir() if p.is_file()]
        return {'job_id': identifier, **result}

    @app.get('/v1/auto-jobs/{identifier}/artifacts/{name}')
    def artifact(identifier: str, name: str, x_local_token: str = Header(default='')):
        authorize(x_local_token)
        job = job_path(identifier)
        if Path(name).name != name or '\\' in name or ':' in name:
            raise HTTPException(404, 'artifact not found')
        target = (job/'delivery'/name).resolve()
        if not target.is_relative_to((job/'delivery').resolve()) or not target.is_file():
            raise HTTPException(404, 'artifact not found')
        return FileResponse(target, filename=name)

    return app
