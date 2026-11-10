"""Persist original network replies BEFORE parsing; replay without a new call."""
import asyncio,json,re
from pathlib import Path
import httpx
from .pipeline import Provider,PipelineError,image_part
from .go_provider import GoProvider
from .materialize import write
from image2ppt_planner.client import MomaClient,strip_code_fence

def save_reply(file,data,key):
    encoded=json.dumps(data,ensure_ascii=False)
    if key and key in encoded:raise PipelineError('credential_echo_rejected')
    if len(encoded)>4_000_000:raise ValueError('model response too large')
    write(file,data)

def parse_reply(data):
    if data['finish_reason'] not in ('stop',None):raise ValueError('model response truncated; original receipt retained')
    try:return json.loads(strip_code_fence(data['content']))
    except (ValueError,RecursionError):raise ValueError('model response is not JSON; original receipt retained') from None


class Planner(Provider):
    def ask_receipted(self,system,context,images,contract,receipt):
        receipt=Path(receipt)
        if receipt.exists():return parse_reply(json.loads(receipt.read_text(encoding='utf-8')))
        messages=[{'role':'system','content':system},{'role':'user','content':[
            {'type':'text','text':json.dumps({**context,'output_schema':contract.model_json_schema()},ensure_ascii=False)},
            *[image_part(p) for p in images]]}]
        async def call():return await asyncio.wait_for(MomaClient(self.cfg).chat(messages),self.cfg.timeout_s+10)
        try:reply=asyncio.run(call())
        except Exception as exc:
            status=re.search(r'HTTP\s+(\d{3})',str(exc))
            raise PipelineError('model_http_'+status.group(1) if status else 'model_transport_response_unknown') from None
        data={'content':reply.text,'finish_reason':reply.finish_reason,'usage':reply.usage,'raw_response':reply.raw,
              'model':reply.model,'request_settings_sha256':reply.request_sha256}
        save_reply(receipt,data,self.cfg.api_key)
        return parse_reply(data)


class Reviewer(GoProvider):
    def ask_receipted(self,system,context,images,contract,receipt):
        receipt=Path(receipt)
        if receipt.exists():return parse_reply(json.loads(receipt.read_text(encoding='utf-8')))
        text=json.dumps({**context,'output_schema':contract.model_json_schema()},ensure_ascii=False)
        content=[{'type':'text','text':text},*[image_part(p) for p in images]] if images else text
        body={'model':self.model,'messages':[{'role':'system','content':system},{'role':'user','content':content}],
              'temperature':0,'stream':False,'max_tokens':32768,'response_format':{'type':'json_object'},'thinking':{'type':'disabled'}}
        try:
            with httpx.Client(proxy=__import__('os').environ.get('OPENCODE_GO_PROXY') or None,trust_env=False,follow_redirects=False,timeout=httpx.Timeout(600,connect=20)) as client:
                response=client.post(self.base_url+'/chat/completions',json=body,headers={'Authorization':'Bearer '+self.key,
                    'User-Agent':'image2ppt-product/1.0','x-opencode-session':self.session})
            if response.status_code!=200:raise PipelineError('opencode_http_'+str(response.status_code))
            raw=response.json();choice=raw['choices'][0]
            data={'content':choice['message']['content'],'finish_reason':choice.get('finish_reason'),
                  'usage':raw.get('usage'),'raw_response':raw,'model':raw.get('model',self.model)}
        except httpx.HTTPError as exc:raise PipelineError('opencode_'+type(exc).__name__) from None
        save_reply(receipt,data,self.key)
        return parse_reply(data)
