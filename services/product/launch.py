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
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--prompt-key',action='store_true')
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--opencode-profile')
    args=parser.parse_args()
    if args.opencode_profile:
        from product.go_provider import load_profile
        load_profile(args.opencode_profile)
    if args.prompt_key:
        key=getpass.getpass('MoMA API key (hidden): ')
        os.environ['MOMA_API_KEY']=key;os.environ['PLANNER_API_KEY']=key
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
