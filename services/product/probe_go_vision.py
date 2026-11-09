"""Capability probe with a synthetic image; never prints credentials."""
from pathlib import Path
import argparse
import json
import httpx
from pydantic import BaseModel
from PIL import Image,ImageDraw
from product.go_provider import GoProvider,load_profile
from product.materialize import write
from product.pipeline import PipelineError,ROOT

class Answer(BaseModel):
    left_shape:str
    left_color:str
    right_shape:str
    right_color:str

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',required=True);args=parser.parse_args()
    load_profile(args.profile)
    p=GoProvider('image2ppt-vision-capability-probe')
    out=ROOT/'outputs/go-vision-probe';out.mkdir(exist_ok=True)
    im=Image.new('RGB',(400,220),'white');d=ImageDraw.Draw(im)
    d.polygon([(35,170),(95,35),(155,170)],fill='#8C30CE')
    d.ellipse((240,60,355,175),fill='#F39A22');im.save(out/'probe.png')
    try:
        reply=p.ask('Inspect the attached image. Identify the shape and color of the two objects from left to right. Use English simple color and shape names. Return JSON only.',{},[out/'probe.png'],Answer)
        result={'model':p.model,'status':'RECEIVED','answer':reply,
                'expected':{'left_shape':'triangle','left_color':'purple','right_shape':'circle','right_color':'orange'}}
    except Exception as exc:result={'model':p.model,'status':'FAILED','code':str(exc) if isinstance(exc,PipelineError) else type(exc).__name__}
    write(out/'probe-result.json',result);print(json.dumps(result))
    try:
        with httpx.Client(trust_env=False,timeout=30,follow_redirects=False) as client:
            r=client.get(p.base_url+'/models',headers={'Authorization':'Bearer '+p.key,'User-Agent':'image2ppt-product/1.0','x-opencode-session':p.session})
        data=r.json() if r.status_code==200 else {}
        models=[{k:row[k] for k in ('id','name','modalities','capabilities') if k in row} for row in data.get('data',[])]
        write(out/'models.json',{'http_status':r.status_code,'models':models});print(json.dumps({'http_status':r.status_code,'models':models})[:6000])
    except Exception as exc:print('model listing failed: '+type(exc).__name__)

if __name__=='__main__':main()
