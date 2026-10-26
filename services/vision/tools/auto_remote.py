"""SSH-only validation worker. Credentials never leave the server.

Run with a private JSON config path and one JSON request on stdin. The worker
does not listen on a port, edit Dify configuration, or restart any service.
Completed requests are immutable and replayable by request_id + payload hash.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

MAX_INPUT = 45_000_000
MAX_OUTPUT = 8_000_000
WORKER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# Executed in the existing Dify environment, where its tenant key provider is
# already configured. Only the explicitly selected credential row is read.
DIFY_CALL = r'''
import contextlib,io,json,logging,sys,time,urllib.request,urllib.error
payload=json.load(sys.stdin)
logging.disable(logging.CRITICAL)
out={"ok":False,"error":"credential_initialization_failed"}
phase='imports'
try:
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  from app_factory import create_app
  from extensions.ext_database import db
  from models.provider import ProviderModelCredential
  from core.helper.encrypter import decrypt_token
  from sqlalchemy import select
  phase='application'
  app=create_app()
  if isinstance(app,tuple):app=next(x for x in app if hasattr(x,'app_context'))
  with app.app_context():
   phase='credential_query'
   rows=db.session.scalars(select(ProviderModelCredential).where(
    ProviderModelCredential.provider_name==payload['credential_provider'],
    ProviderModelCredential.model_name==payload['credential_model'])).all()
   if len(rows)!=1:raise ValueError('credential_selection_ambiguous')
   row=rows[0];cfg=json.loads(row.encrypted_config)
   endpoint=cfg.get('endpoint_url','').rstrip('/')
   if endpoint!=payload['base_url'].rstrip('/'):raise ValueError('credential_endpoint_mismatch')
   phase='credential_decrypt'
   secret=decrypt_token(row.tenant_id,cfg['api_key'])
  phase='request'
  class NoRedirect(urllib.request.HTTPRedirectHandler):
   def redirect_request(self,*args,**kwargs):return None
  body_data=dict(payload['body']);body_data.update(stream=True,stream_options={'include_usage':True})
  body=json.dumps(body_data,ensure_ascii=False,allow_nan=False).encode()
  req=urllib.request.Request(endpoint+'/chat/completions',data=body,headers={
   'Authorization':'Bearer '+secret,'Content-Type':'application/json'})
  start=time.monotonic()
  with urllib.request.build_opener(NoRedirect).open(req,timeout=min(60,payload['timeout'])) as response:
   if 'text/event-stream' in response.headers.get('Content-Type',''):
    pieces=[];usage={};finish=None;length=0;done=False;model_name=payload['body']['model'];tools_seen=False
    for rawline in response:
     if time.monotonic()-start>payload['timeout']:raise TimeoutError()
     length+=len(rawline)
     if length>8000000:raise ValueError('model_response_too_large')
     if not rawline.startswith(b'data:'):continue
     chunk=rawline[5:].strip()
     if chunk==b'[DONE]':done=True;break
     if not chunk:continue
     event=json.loads(chunk)
     if event.get('usage'):usage=event['usage']
     model_name=event.get('model',model_name)
     for choice in event.get('choices',[]):
      delta=choice.get('delta',{})
      if delta.get('tool_calls') or delta.get('function_call') or delta.get('refusal'):tools_seen=True
      if delta.get('content'):pieces.append(delta['content'])
      if choice.get('finish_reason') is not None:finish=choice['finish_reason']
    if not done or finish!='stop':
     out={'ok':False,'error':'model_response_incomplete','finish_reason':finish,'usage':usage,
          'partial_content':''.join(pieces).replace(secret,'[REDACTED]')}
     raise ValueError('stream_incomplete')
    data={'model':model_name,'usage':usage,'choices':[{'finish_reason':finish,'message':{'content':''.join(pieces),'tool_calls':tools_seen}}]}
   else:
    raw=response.read(8000001)
    if len(raw)>8000000:raise ValueError('model_response_too_large')
    data=json.loads(raw)
  choice=data['choices'][0];message=choice['message']
  if choice.get('finish_reason') not in ('stop',None):
   out={'ok':False,'error':'model_response_incomplete','finish_reason':choice.get('finish_reason'),'usage':data.get('usage',{})}
  elif not isinstance(message.get('content'),str) or message.get('tool_calls') or message.get('refusal'):
   out={'ok':False,'error':'invalid_model_message','usage':data.get('usage',{})}
  else:
   out={'ok':True,'content':message['content'].replace(secret,'[REDACTED]'),
        'model':data.get('model',payload['body']['model']),'usage':data.get('usage',{}),
        'elapsed_seconds':round(time.monotonic()-start,3)}
except urllib.error.HTTPError as exc:
 out={'ok':False,'error':'model_http_'+str(exc.code),'http_status':exc.code}
except (TimeoutError, __import__('socket').timeout):out={'ok':False,'error':'model_timeout'}
except urllib.error.URLError:out={'ok':False,'error':'model_transport_failed'}
except ValueError as exc:
 code=str(exc)
 if code!='stream_incomplete':out={'ok':False,'error':code if code in ('credential_selection_ambiguous','credential_endpoint_mismatch','model_response_too_large') else 'invalid_model_response'}
except Exception as exc:
 out={'ok':False,'error':phase+'_'+type(exc).__name__}
print('I2P_RESULT='+json.dumps(out,ensure_ascii=False))
'''


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8')


def save(path, value):
    temp = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temp.open('xb') as f:
        f.write(encode(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def model_call(config, payload):
    model = payload['body']['model']
    if model not in config['allowed_models']:
        return {'ok': False, 'error': 'model_not_allowed'}
    body = payload['body']
    if set(body)-{'model', 'messages', 'temperature', 'max_tokens', 'response_format', 'enable_thinking'}:
        return {'ok': False, 'error': 'unsupported_model_request'}
    request = {k: config[k] for k in ('credential_provider', 'credential_model', 'base_url')}
    request.update(body=body, timeout=min(480, max(1, int(payload.get('timeout', 180)))))
    start = time.monotonic()
    try:
        result = subprocess.run(['docker', 'exec', '-i', config['dify_container'],
            '/app/api/.venv/bin/python', '-c', DIFY_CALL], input=encode(request),
            capture_output=True, timeout=request['timeout']+60)
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': 'model_worker_timeout'}
    lines = [x[len('I2P_RESULT='):] for x in result.stdout.decode('utf-8', errors='replace').splitlines()
             if x.startswith('I2P_RESULT=')]
    if result.returncode or not lines:
        return {'ok': False, 'error': 'model_worker_failed'}
    try:
        output = json.loads(lines[-1])
    except ValueError:
        return {'ok': False, 'error': 'model_worker_invalid_json'}
    output.update(requested_model=model, total_seconds=round(time.monotonic()-start, 3))
    return output


def analyze(config, payload, directory):
    image = base64.b64decode(payload['image_base64'], validate=True)
    sha = hashlib.sha256(image).hexdigest()
    if sha != payload['source_sha256'] or len(image) > 20_000_000:
        return {'ok': False, 'error': 'source_binding_mismatch'}
    secret = Path(config['vision_token_file']).read_text().strip()
    base = config['vision_url'].rstrip('/')

    def http(path, data=None, content_type='application/json'):
        req = urllib.request.Request(base+path, data=data, headers={
            'Authorization': 'Bearer '+secret, 'Content-Type': content_type})
        with urllib.request.urlopen(req, timeout=40) as response:
            raw = response.read(MAX_OUTPUT+1)
        if len(raw) > MAX_OUTPUT:
            raise ValueError('analysis_response_too_large')
        return json.loads(raw)

    receipt = directory/'analysis-job.json'
    if receipt.exists():
        job_id = json.loads(receipt.read_text())['id']
    else:
        boundary = 'image2ppt-'+uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="page.png"\r\n'
                'Content-Type: image/png\r\n\r\n').encode()+image+f'\r\n--{boundary}--\r\n'.encode()
        asset = http('/v1/assets', body, 'multipart/form-data; boundary='+boundary)
        request = {'operation': 'analyze', 'asset_id': asset['id'], 'ocr': True,
                   'segment': False, 'timeout_seconds': 300}
        job = http('/v1/jobs', encode(request))
        job_id = job['id']
        save(receipt, {'id': job_id, 'asset_id': asset['id'], 'source_sha256': sha})
    deadline = time.monotonic()+360
    while time.monotonic() < deadline:
        job = http('/v1/jobs/'+job_id)
        if job['status'] not in ('QUEUED', 'RUNNING'):
            break
        time.sleep(1)
    if job['status'] != 'SUCCEEDED':
        return {'ok': False, 'error': 'analysis_'+job['status'].lower(), 'job_id': job_id}
    result = http('/v1/jobs/'+job_id+'/artifacts/result.json')
    meta = job.get('artifacts', {}).get('result.json', {})
    # The service verifies the original bytes before returning the artifact;
    # the parsed result is additionally bound to this exact submitted source.
    hashes = {item.get(k) for item in result.get('inputs', {}).values() for k in ('original_sha256', 'sha256')}
    if sha not in hashes:
        return {'ok': False, 'error': 'analysis_source_mismatch'}
    return {'ok': True, 'analysis': result, 'job_id': job_id,
            'server_result_sha256': meta.get('sha256'), 'source_sha256': sha}


def analyze_isolated(config, payload, directory):
    """Reuse installed OCR code/models, with a separate Store and worker process.

    Explicit --data prevents touching the production queue/database. The host
    and container roots refer to a new subdirectory in the existing data mount.
    """
    image = base64.b64decode(payload['image_base64'], validate=True)
    sha = hashlib.sha256(image).hexdigest()
    if sha != payload['source_sha256'] or len(image) > 20_000_000:
        return {'ok': False, 'error': 'source_binding_mismatch'}
    work = Path(config['isolated_host_root'])/directory.name
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    inside = config['isolated_container_root'].rstrip('/')+'/'+directory.name
    source = work/'source.png'
    if source.exists():
        if hashlib.sha256(source.read_bytes()).hexdigest() != sha:
            return {'ok': False, 'error': 'isolated_source_changed'}
    else:
        source.write_bytes(image)
    jobs = list((work/'store/jobs').glob('*/status.json'))
    if not jobs:
        save(work/'options.json', {'ocr': True, 'segment': False, 'timeout_seconds': 300})
        try:
            process = subprocess.run(['docker', 'exec', config['vision_container'],
                'image2ppt-vision', '--data', inside+'/store', 'analyze',
                '--input', inside+'/source.png', '--options', inside+'/options.json'],
                capture_output=True, timeout=340)
        except subprocess.TimeoutExpired:
            return {'ok': False, 'error': 'isolated_ocr_timeout'}
        jobs = list((work/'store/jobs').glob('*/status.json'))
    if len(jobs) != 1:
        return {'ok': False, 'error': 'isolated_ocr_job_missing'}
    job_dir = jobs[0].parent
    status = json.loads(jobs[0].read_text())
    if status['status'] != 'SUCCEEDED':
        return {'ok': False, 'error': 'isolated_ocr_'+status['status'].lower(), 'job_id': status['id']}
    raw = (job_dir/'result.json').read_bytes()
    manifest = json.loads((job_dir/'manifest.json').read_text())
    expected = manifest['result.json']['sha256']
    if hashlib.sha256(raw).hexdigest() != expected:
        return {'ok': False, 'error': 'isolated_ocr_manifest_mismatch'}
    result = json.loads(raw)
    hashes = {item.get(k) for item in result.get('inputs', {}).values() for k in ('original_sha256', 'sha256')}
    if sha not in hashes:
        return {'ok': False, 'error': 'analysis_source_mismatch'}
    return {'ok': True, 'analysis': result, 'job_id': status['id'], 'server_result_sha256': expected,
            'source_sha256': sha, 'analysis_execution': 'isolated_cli_existing_vision_container'}


def dispatch(config, request):
    identifier = request.get('request_id', '')
    if not re.fullmatch(r'[a-f0-9]{32}', identifier):
        return {'ok': False, 'error': 'invalid_request_id'}
    root = Path(config['data_root']).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory = root/identifier
    directory.mkdir(exist_ok=True, mode=0o700)
    fingerprint = hashlib.sha256(encode(request)).hexdigest()
    output = directory/'result.json'
    binding = directory/'binding.json'
    if binding.exists():
        if json.loads(binding.read_text())['request_sha256'] != fingerprint:
            return {'ok': False, 'error': 'request_id_collision'}
        if output.exists():
            return json.loads(output.read_text())
        # A disconnected caller must not accidentally spend a second inference.
        if request.get('operation') == 'model':
            return {'ok': False, 'error': 'remote_request_incomplete', 'request_id': identifier}
    else:
        try:
            with binding.open('xb') as f:
                f.write(encode({'request_sha256': fingerprint, 'created_at': time.time()}))
        except FileExistsError:
            return {'ok': False, 'error': 'request_busy'}
    try:
        operation = request.get('operation')
        if operation == 'model':
            result = model_call(config, request)
        elif operation == 'analyze':
            if config.get('analysis_mode') == 'isolated_cli':
                result = analyze_isolated(config, request, directory)
            else:
                result = analyze(config, request, directory)
        elif operation == 'catalog':
            with urllib.request.urlopen(config['base_url'].rstrip('/')+'/models', timeout=20) as response:
                value = json.loads(response.read(MAX_OUTPUT))
            result = {'ok': True, 'base_url': config['base_url'],
                      'models': [x['id'] for x in value['data']], 'credential_source': 'server_dify'}
        else:
            result = {'ok': False, 'error': 'unsupported_operation'}
    except urllib.error.HTTPError as exc:
        result = {'ok': False, 'error': 'remote_http_'+str(exc.code)}
    except Exception as exc:
        result = {'ok': False, 'error': 'remote_'+type(exc).__name__}
    result.update(request_id=identifier, request_sha256=fingerprint,
                  worker_sha256=WORKER_SHA256,
                  configuration_sha256=hashlib.sha256(encode(config)).hexdigest())
    save(output, result)
    return result


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    raw = sys.stdin.buffer.read(MAX_INPUT+1)
    if len(raw) > MAX_INPUT:
        result = {'ok': False, 'error': 'request_too_large'}
    else:
        result = dispatch(config, json.loads(raw))
    sys.stdout.buffer.write(encode(result)+b'\n')


if __name__ == '__main__':
    main()
