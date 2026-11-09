"""Loopback-only, ephemeral credential handoff for the authorized experiment."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
import time
import webbrowser
from product.materialize import write


def serve(root,port=8766):
    capability=secrets.token_urlsafe(32);lock=threading.Lock();state={'status':'WAITING_FOR_KEYS'}
    def work():
        try:
            from object_experiment import plan
            state.update(status='PLANNING');write(root/'live-status.json',state)
            plan(root,'moma')
            state.update(status='PLANS_READY');write(root/'live-status.json',state)
        except Exception as exc:
            state.update(status='FAILED',error=type(exc).__name__);write(root/'live-status.json',state)
        finally:lock.release()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,code,body,kind='application/json'):
            self.send_response(code);self.send_header('Content-Type',kind+'; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Referrer-Policy','no-referrer');self.end_headers();self.wfile.write(body.encode('utf-8'))
        def do_GET(self):
            if self.path=='/'+capability+'/status':return self.send(200,json.dumps(state))
            if self.path!='/'+capability:return self.send(404,'{}')
            self.send(200,'''<!doctype html><meta charset="utf-8"><title>Image2PPT 对象提取实验</title><style>body{font:18px system-ui;max-width:700px;margin:70px auto;line-height:1.7}input{display:block;padding:12px;width:90%;margin:12px 0}button{padding:12px 24px}</style><h1>连接模型，继续 12 页对象对照</h1><p>密钥仅保留在这个本机进程内存中，关闭服务即失效。已有 PPT 和源图不会被覆盖。</p><form><label>MoMA 密钥<input id="m" type="password" autocomplete="off" required></label><label>OpenCode Go 密钥（可稍后补充，用于成品复核）<input id="g" type="password" autocomplete="off"></label><button>连接并继续</button></form><p id="s">等待配置</p><script>document.querySelector('form').onsubmit=async(e)=>{e.preventDefault();let r=await fetch(location.pathname,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({moma:m.value,go:g.value})});m.value='';g.value='';s.textContent=r.ok?'已配置，后台继续运行':'配置失败，请重试';};setInterval(async()=>{let r=await fetch(location.pathname+'/status');if(r.ok)s.textContent=JSON.stringify(await r.json());},3000);</script>''','text/html')
        def do_POST(self):
            if self.path!='/'+capability:return self.send(404,'{}')
            if self.headers.get('Origin')!=f'http://127.0.0.1:{port}':return self.send(403,'{}')
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8192:raise ValueError()
                payload=json.loads(self.rfile.read(length));moma=payload.get('moma','');go=payload.get('go','')
                if not isinstance(moma,str) or not 8<=len(moma)<=4096 or any(ord(c)<33 or ord(c)>126 for c in moma):raise ValueError()
                if not isinstance(go,str) or len(go)>4096 or any(ord(c)<33 or ord(c)>126 for c in go):raise ValueError()
                os.environ['MOMA_API_KEY']=moma;os.environ['PLANNER_API_KEY']=moma
                if go:os.environ['OPENCODE_GO_API_KEY']=go
                if lock.acquire(False):threading.Thread(target=work,daemon=True).start()
                self.send(200,'{"configured":true}')
            except (ValueError,TypeError):self.send(422,'{"error":"invalid_input"}')
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    write(root/'live-status.json',state)
    url=f'http://127.0.0.1:{port}/{capability}'
    webbrowser.open(url);print('Opened local credential page on port '+str(port),flush=True)
    server.serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=8766);a=p.parse_args();serve(a.root.resolve(),a.port)
