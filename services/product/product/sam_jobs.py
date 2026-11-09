"""Source/model-bound private SAM client with resumable, authenticated download."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlparse
import uuid
from .materialize import digest,write


def infer_remote(source,planfile,output,object_ids,*,url,token_file,timeout=2000):
    parsed=urlparse(url)
    import ipaddress
    if parsed.scheme not in ('http','https') or not parsed.hostname:raise ValueError('invalid private SAM URL')
    if parsed.scheme=='http' and parsed.hostname!='localhost' and not ipaddress.ip_address(parsed.hostname).is_private:
        raise ValueError('unencrypted SAM endpoint must be private')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    token=Path(token_file).read_text().strip()
    def request(path,data=None,content_type=None):
        headers={'Authorization':'Bearer '+token}
        if content_type:headers['Content-Type']=content_type
        req=urllib.request.Request(url.rstrip('/')+path,data=data,headers=headers)
        with urllib.request.urlopen(req,timeout=45) as response:return response.read()
    capabilities=json.loads(request('/v1/capabilities'));model=capabilities['models']['sam3']
    if capabilities.get('worker_alive') is not True:raise RuntimeError('private SAM worker unavailable')
    from .object_registry import SAM_BINDING
    if any(model.get(k)!=SAM_BINDING[k] for k in ('sha256','code_revision') if k in SAM_BINDING):raise ValueError('SAM model version not pinned')
    if model.get('sha256')!=SAM_BINDING['checkpoint_sha256']:raise ValueError('SAM checkpoint not pinned')
    original=json.loads(Path(planfile).read_text(encoding='utf-8'))
    plan={**original,'objects':[o for o in original['objects'] if o['id'] in object_ids]}
    if not plan['objects']:return output
    # Re-submission uses the server's content key and never schedules duplicate
    # inference, even if the client crashed before recording the returned ID.
    boundary=uuid.uuid4().hex
    data=(f'--{boundary}\r\nContent-Disposition: form-data; name="plan"\r\n\r\n'+json.dumps(plan)+
          f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="source.png"\r\nContent-Type: image/png\r\n\r\n').encode()+Path(source).read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
    submitted=json.loads(request('/v1/jobs',data,'multipart/form-data; boundary='+boundary));write(output/'remote-job.json',submitted)
    identifier=submitted['id'];deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        state=json.loads(request('/v1/jobs/'+identifier));write(output/'remote-status.json',state)
        if state['status']=='FAILED':raise RuntimeError('private_sam_failed')
        if state['status']=='SUCCEEDED':break
        time.sleep(2)
    else:raise TimeoutError('private SAM pending; job ID retained for resume')
    for obj in plan['objects']:
        canonical=submitted['mapping'][obj['id']];receipt=canonical+'.json';file=canonical+'.npz'
        values={}
        for name in (receipt,file):
            blob=request(f'/v1/jobs/{identifier}/artifacts/{name}')
            if hashlib.sha256(blob).hexdigest()!=state['artifacts'].get(name):raise ValueError('SAM download hash mismatch')
            values[name]=blob
        meta=json.loads(values[receipt])
        expected={'status':'SUCCEEDED','source_sha256':digest(source),'image_size':plan['image_size'],'version':'sam3',
            'checkpoint_sha256':model['sha256'],'code_revision':model['code_revision']}
        if any(meta.get(k)!=v for k,v in expected.items()) or meta.get('mask_sha256')!=state['artifacts'][file]:raise ValueError('SAM receipt binding mismatch')
        (output/('raw-'+obj['id']+'.json')).write_bytes(values[receipt])
        localfile=obj['id']+'.npz';(output/localfile).write_bytes(values[file])
        # Explicit adapter receipt; original server response remains untouched.
        write(output/(obj['id']+'.json'),{**meta,'plan_sha256':digest(planfile),'mask_file':localfile,
            'adapter':'PrivateSamReceipt/1','raw_receipt_sha256':state['artifacts'][receipt],'remote_job':identifier})
    return output
