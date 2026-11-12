"""Business API client. Windows profiles contain generated service tokens only."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import httpx


def save_profile(path, profile):
    import win32crypt
    allowed={'url','business_token','office_token','task','tunnel_pid','worker_pid'}
    if set(profile)-allowed:raise ValueError('runtime profiles only store generated service access configuration')
    blob=json.dumps(profile).encode('utf8')
    encrypted=win32crypt.CryptProtectData(blob,'Image2PPT service access',None,None,None,0)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encrypted)


def load_profile(path):
    import win32crypt
    data=win32crypt.CryptUnprotectData(Path(path).read_bytes(),None,None,None,0)[1]
    return json.loads(data.decode('utf8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile')
    parser.add_argument('--url',default='http://127.0.0.1:8771')
    commands=parser.add_subparsers(dest='command',required=True)
    submit=commands.add_parser('submit')
    submit.add_argument('images',nargs='+')
    submit.add_argument('--goal',default='Restore these images as an editable PPTX.')
    submit.add_argument('--tool-profile',choices=('all','basic'),default='all')
    submit.add_argument('--agent-steps',type=int,default=64)
    submit.add_argument('--seconds',type=int,default=7200)
    submit.add_argument('--agent-thinking',action=argparse.BooleanOptionalAction,default=True)
    for name in ('status','cancel','artifacts'):
        command=commands.add_parser(name);command.add_argument('job_id')
        if name=='artifacts':command.add_argument('--output')
    options=parser.parse_args()
    profile=load_profile(options.profile) if options.profile else {
        'url':options.url,'business_token':os.environ['IMAGE2PPT_PRODUCT_TOKEN']}
    with httpx.Client(base_url=profile['url'],headers={'Authorization':'Bearer '+profile['business_token']},
                      timeout=60,trust_env=False) as client:
        if options.command=='submit':
            from contextlib import ExitStack
            with ExitStack() as stack:
                files=[('files',(Path(name).name,stack.enter_context(open(name,'rb')),'image/png')) for name in options.images]
                response=client.post('/api/v1/jobs',params={'goal':options.goal,'tool_profile':options.tool_profile,
                    'agent_steps':options.agent_steps,'runtime_seconds':options.seconds,
                    'agent_thinking':options.agent_thinking},files=files)
        else:
            route='/api/v1/jobs/'+options.job_id
            if options.command=='status':response=client.get(route)
            elif options.command=='cancel':response=client.post(route+'/cancel')
            else:response=client.get(route+'/artifacts')
        response.raise_for_status()
        result=response.json()
        if options.command=='artifacts' and options.output:
            output=Path(options.output).resolve();output.mkdir(parents=True,exist_ok=True)
            for artifact in result.get('artifacts',[]):
                name=artifact['name'];route=artifact['url']
                if Path(name).name!=name or not route.startswith('/api/jobs/'+options.job_id+'/files/'):
                    raise ValueError('invalid artifact location')
                download=client.get(route);download.raise_for_status()
                if hashlib.sha256(download.content).hexdigest()!=artifact['sha256']:
                    raise ValueError('artifact hash mismatch')
                target=output/name
                if target.exists() and target.read_bytes()!=download.content:
                    raise ValueError('existing artifact differs; choose a new output directory')
                target.write_bytes(download.content)
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
