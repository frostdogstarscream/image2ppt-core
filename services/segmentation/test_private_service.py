import hashlib
import importlib.util
import io
import json
from pathlib import Path
import threading
import time
from fastapi.testclient import TestClient
from PIL import Image

spec=importlib.util.spec_from_file_location('private_sam',Path(__file__).with_name('service.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_status_read_retries_windows_sharing_violation(tmp_path,monkeypatch):
    file=tmp_path/'status.json';file.write_text('{"status":"RUNNING"}',encoding='utf-8')
    original=Path.read_text;attempts=[]
    def read(path,*args,**kwargs):
        if path==file:
            attempts.append(1)
            if len(attempts)<3:raise PermissionError('controlled sharing violation')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    assert module.read_json(file)=={'status':'RUNNING'} and len(attempts)==3


def test_private_service_auth_binding_dedup_serial_and_download(tmp_path):
    token=tmp_path/'token';token.write_text('x'*48);active=[];peak=[];calls=[]
    def runner(job,request):
        active.append(job.name);peak.append(len(active));calls.append(job.name);time.sleep(.05)
        out=job/'result';out.mkdir(exist_ok=True)
        (out/'obj-000.npz').write_bytes(b'controlled-synthetic-result')
        module.save(out/'run.json',{'status':'SUCCEEDED'})
        module.save(out/'obj-000.json',{'mask_sha256':module.sha(out/'obj-000.npz')})
        active.remove(job.name)
    config={'jobs':str(tmp_path/'jobs'),'token_file':str(token),'code_revision':'revision',
        'models':{'sam3':{'checkpoint':'server-controlled','sha256':'model'}}}
    app=module.create_app(config,runner=runner)
    image=io.BytesIO();Image.new('RGB',(40,30),'white').save(image,format='PNG');data=image.getvalue()
    p={'source_sha256':hashlib.sha256(data).hexdigest(),'image_size':[40,30],
        'objects':[{'id':'cloud','bbox':[1,1,20,20],'prompt_en':'cloud'}]}
    auth={'Authorization':'Bearer '+'x'*48}
    with TestClient(app) as client:
        assert client.get('/v1/capabilities').status_code==401
        assert client.get('/v1/capabilities',headers=auth).json()['serial_gpu']
        def submit(plan):return client.post('/v1/jobs',headers=auth,data={'plan':json.dumps(plan)},files={'file':('source.png',data,'image/png')})
        first=submit(p).json();p['objects'][0]['id']='renamed';same=submit(p).json()
        assert first['id']==same['id'] and same['mapping']=={'renamed':'obj-000'}
        p['objects'][0]['prompt_en']='a different icon';second=submit(p).json()
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            a=client.get('/v1/jobs/'+first['id'],headers=auth).json();b=client.get('/v1/jobs/'+second['id'],headers=auth).json()
            if a['status']==b['status']=='SUCCEEDED':break
            time.sleep(.02)
        assert a['status']==b['status']=='SUCCEEDED' and max(peak)==1 and len(calls)==2
        url=f"/v1/jobs/{first['id']}/artifacts/obj-000.npz"
        assert client.get(url,headers=auth).content==b'controlled-synthetic-result'
        (Path(config['jobs'])/first['id']/'result/obj-000.npz').write_bytes(b'corrupt')
        assert client.get(url,headers=auth).status_code==409
        p['source_sha256']='0'*64
        assert submit(p).status_code==422
        assert client.get('/v1/jobs/unknown',headers=auth).status_code==404
