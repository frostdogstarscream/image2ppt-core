"""OpenCode Go reviewer using the OpenAI-compatible multimodal request format."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import uuid
import httpx


def load_profile(path):
    """Read an explicitly selected Windows DPAPI profile without copying its key."""
    path=Path(path).resolve();profile=json.loads(path.read_text(encoding='utf-8-sig'))
    if profile.get('provider')!='opencode_go' or profile.get('base_url')!='https://opencode.ai/inference/go/openai/v1':
        raise ValueError('unsupported OpenCode Go profile')
    file=(path.parent/profile['encrypted_key_file']).resolve()
    if not file.is_relative_to(path.parent):raise ValueError('credential path outside profile')
    env=os.environ.copy();env['IMAGE2PPT_DPAPI_FILE']=str(file)
    command="$ErrorActionPreference='Stop'; $s=ConvertTo-SecureString ([IO.File]::ReadAllText($env:IMAGE2PPT_DPAPI_FILE).Trim()); try { [Console]::Write([Net.NetworkCredential]::new('', $s).Password) } finally { $s.Dispose() }"
    shell=shutil.which('pwsh') or 'powershell.exe'
    result=subprocess.run([shell,'-NoProfile','-NonInteractive','-Command',command],env=env,
                          capture_output=True,timeout=20,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or not result.stdout:raise RuntimeError('OpenCode Go credential unavailable')
    os.environ['OPENCODE_GO_API_KEY']=result.stdout.decode('utf-8').strip()
    os.environ['OPENCODE_GO_BASE_URL']=profile['base_url'];os.environ['OPENCODE_GO_MODEL']=profile['model']


class GoProvider:
    def __init__(self, session):
        self.key=os.environ.get('OPENCODE_GO_API_KEY','')
        if not self.key:raise RuntimeError('OpenCode Go key absent')
        self.model=os.environ.get('OPENCODE_GO_MODEL','deepseek-v4.1-flash')
        self.base_url=os.environ.get('OPENCODE_GO_BASE_URL','https://opencode.ai/inference/go/openai/v1')
        if self.base_url!='https://opencode.ai/inference/go/openai/v1':raise ValueError('unsupported OpenCode Go endpoint')
        self.session=str(uuid.uuid5(uuid.NAMESPACE_URL,str(session)))

    def ask(self,system,context,images,contract):
        from .pipeline import PipelineError,image_part
        text=json.dumps({**context,'output_schema':contract.model_json_schema()},ensure_ascii=False)
        content=[{'type':'text','text':text},*[image_part(p) for p in images]] if images else text
        body={'model':self.model,'messages':[{'role':'system','content':system},
              {'role':'user','content':content}],
              'temperature':0,'stream':False,'max_tokens':32768,'response_format':{'type':'json_object'},
              'thinking':{'type':'disabled'}}
        try:
            with httpx.Client(proxy=os.environ.get('OPENCODE_GO_PROXY') or None,trust_env=False,follow_redirects=False,timeout=httpx.Timeout(600,connect=20)) as client:
                response=client.post(self.base_url+'/chat/completions',json=body,headers={
                    'Authorization':'Bearer '+self.key,'User-Agent':'image2ppt-product/1.0','x-opencode-session':self.session})
            if response.status_code!=200:raise PipelineError('opencode_http_'+str(response.status_code))
            data=response.json();choice=data['choices'][0];content=choice['message']['content']
            if self.key in content:raise PipelineError('credential_echo_rejected')
            if choice.get('finish_reason') not in ('stop',None):raise ValueError('OpenCode Go response truncated')
            from image2ppt_planner.client import strip_code_fence
            return json.loads(strip_code_fence(content))
        except httpx.HTTPError as exc:
            error=PipelineError('opencode_'+type(exc).__name__)
            detail=str(exc)
            error.transport_detail=detail[:500] if self.key not in detail else 'credential_redacted'
            raise error from None
