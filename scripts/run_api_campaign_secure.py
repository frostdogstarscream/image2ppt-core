"""Interactive in-memory credential entry for the explicitly authorized API run.

Never pass credentials through argv, files, reports, or echoed terminal input.
Generation still runs the ordinary frozen, budgeted evaluate-api subprocess.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib import request,error

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'services/vision/src'))
from image2ppt_vision.model_provider import _NoRedirect
from image2ppt_vision.storage import read_json,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',required=True)
    parser.add_argument('--config',default='generation-config.json')
    parser.add_argument('--campaign',default='campaign-verified')
    args=parser.parse_args()
    folder=Path(args.experiment).resolve();config_path=(folder/args.config).resolve()
    campaign=(folder/args.campaign).resolve()
    if not config_path.is_relative_to(folder) or not campaign.is_relative_to(folder):raise SystemExit('Invalid experiment path')
    config=read_json(config_path)
    key=getpass.getpass('API credential (hidden): ')
    if not key or any(ord(c)<32 or ord(c)>126 for c in key):raise SystemExit('Invalid credential format')
    os.environ['VISION_LLM_API_KEY']=key
    try:
        url=config['base_url'].rstrip('/')
        if url.endswith('/chat/completions'):url=url[:-len('/chat/completions')]
        status={'requested_model':config['model'],'status':'NOT_LISTED'}
        try:
            req=request.Request(url+'/models',headers={'Authorization':'Bearer '+key,'Accept':'application/json'})
            with request.build_opener(_NoRedirect).open(req,timeout=30) as response:
                raw=response.read(2_000_000)
            data=json.loads(raw)
            ids=[v['id'] for v in data.get('data',[]) if isinstance(v,dict) and isinstance(v.get('id'),str)]
            ids=[i for i in ids if key not in i and len(i)<=256 and all(ord(c)>=32 for c in i)]
            status.update(status='LISTED',model_ids=ids,requested_model_listed=config['model'] in ids)
        except error.HTTPError as exc:status.update(status='MODEL_LIST_UNAVAILABLE',http_status=exc.code)
        except Exception:status.update(status='MODEL_LIST_UNAVAILABLE')
        write_json(folder/'model-discovery.json',status)
        print(json.dumps(status,ensure_ascii=False),flush=True)
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'services/vision/src')
        while True:
            command=input('Command: round<N> / quit > ').strip()
            if command=='quit':break
            if not command.startswith('round') or not command[5:].isdigit():continue
            number=int(command[5:])
            if number<1:continue
            arguments=[sys.executable,'-m','image2ppt_vision.cli','evaluate-api',
                       '--config',str(config_path),'--output',str(campaign),
                       '--compiler-root',str(ROOT),'--round',str(number)]
            if (campaign/f'round-{number}').exists():arguments.append('--resume')
            result=subprocess.run(arguments,cwd=ROOT,env=env)
            print(json.dumps({'round':number,'process_exit_code':result.returncode}),flush=True)
    finally:
        os.environ.pop('VISION_LLM_API_KEY',None)
        key=None


if __name__=='__main__':main()
