"""Run independently of Codex: python services/product/launch.py."""
import argparse
import getpass
import os
import secrets
import sys
import subprocess
from pathlib import Path
import threading
import time
import webbrowser
from urllib.request import Request, urlopen

def main():
    from product.parallel_config import configure
    configure()
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--prompt-key',action='store_true')
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--opencode-profile')
    parser.add_argument('--smoke-image')
    parser.add_argument('--smoke-report')
    args=parser.parse_args()
    if args.opencode_profile:
        from product.go_provider import load_profile
        load_profile(args.opencode_profile)
    if args.prompt_key:
        key=getpass.getpass('MoMA API key (hidden): ')
        os.environ['MOMA_API_KEY']=key;os.environ['PLANNER_API_KEY']=key
        if not os.environ.get('OPENCODE_GO_API_KEY'):
            os.environ['OPENCODE_GO_API_KEY']=getpass.getpass('OpenCode Go API key (hidden): ')
    token=os.environ.get('IMAGE2PPT_PRODUCT_TOKEN') or secrets.token_urlsafe(32)
    os.environ['IMAGE2PPT_PRODUCT_TOKEN']=token
    if args.background:
        from product.pipeline import ROOT
        (ROOT/'outputs').mkdir(exist_ok=True)
        with (ROOT/'outputs/product-server.stdout.log').open('a',encoding='utf-8') as out, (ROOT/'outputs/product-server.stderr.log').open('a',encoding='utf-8') as err:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--no-browser','--port',str(args.port)],
                cwd=ROOT,env=os.environ.copy(),stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        ready=False
        for _ in range(50):
            if process.poll() is not None:break
            try:
                request=Request(f'http://127.0.0.1:{args.port}/api/health',headers={'Authorization':'Bearer '+token})
                with urlopen(request,timeout=1) as response:ready=response.status==200
                if ready:break
            except OSError:pass
            time.sleep(.2)
        if not ready:raise RuntimeError('Product server did not become ready; inspect outputs/product-server.stderr.log')
        if args.smoke_image:
            import httpx,json,hashlib
            source=Path(args.smoke_image).resolve()
            with httpx.Client(timeout=60) as client:
                with source.open('rb') as file:
                    reply=client.post(f'http://127.0.0.1:{args.port}/api/jobs',headers={'Authorization':'Bearer '+token},
                        files=[('files',(source.name,file,'image/jpeg'))])
                reply.raise_for_status();identifier=reply.json()['id']
                print('Fresh-upload job='+identifier,flush=True)
                while True:
                    response=client.get(f'http://127.0.0.1:{args.port}/api/jobs/{identifier}',headers={'Authorization':'Bearer '+token})
                    response.raise_for_status();state=response.json()
                    if state['status'] not in ('QUEUED','RUNNING'):break
                    time.sleep(2)
                downloaded={}
                if state['status'] in ('SUCCEEDED','REVIEW_REQUIRED'):
                    for name,sha in state['result']['artifacts'].items():
                        artifact=client.get(f'http://127.0.0.1:{args.port}/api/jobs/{identifier}/files/'+name,headers={'Authorization':'Bearer '+token})
                        artifact.raise_for_status()
                        if hashlib.sha256(artifact.content).hexdigest()!=sha:raise RuntimeError('download binding failed')
                        downloaded[name]=sha
                report={'id':identifier,'status':state['status'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                    'downloaded':downloaded,'independent_api_worker':True,'human_acceptance':False}
                if args.smoke_report:
                    from product.materialize import write
                    write(args.smoke_report,report)
                print(json.dumps(report),flush=True)
        if not args.no_browser:webbrowser.open(f'http://127.0.0.1:{args.port}/#token={token}')
        print('Started background product server PID='+str(process.pid))
        return
    from product.app import create_app
    import uvicorn
    if not args.no_browser:
        threading.Timer(2,lambda:webbrowser.open(f'http://127.0.0.1:{args.port}/#token={token}')).start()
    print(f'Image2PPT product: http://127.0.0.1:{args.port} (loopback only)')
    uvicorn.run(create_app(token=token),host='127.0.0.1',port=args.port,access_log=False)

if __name__=='__main__':main()
