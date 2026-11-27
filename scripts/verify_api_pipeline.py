"""End-to-end offline loopback fixture, outside the real 14-page API budget."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'services/vision/src'))
from image2ppt_vision.api_evaluation import evaluate_api,verify_generation
from image2ppt_vision.api_scoring import score_api,compare_rounds
from image2ppt_vision.storage import write_json,read_json,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    source=out/'source.png';Image.new('RGB',(320,180),'white').save(source)
    analysis=out/'analysis.json';write_json(analysis,{'schema':'VisionAnalysis','version':'1.0','image_size':[320,180],
              'objects':[],'inputs':{'source':{'sha256':digest(source)}}})
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert payload['enable_thinking'] is False and payload['max_tokens']==8192
            calls.append(1);version='1.0' if len(calls)==1 else '1.1'
            plan={'schema':'ReconstructionPlan','version':version,'source_sha256':digest(source),'image_size':[320,180],
                  'author':{'tool':'vision-model-plan','model_id':'offline-fixture'},
                  'regions':[{'id':'body','bbox':[0,0,320,180],'kind':'panel'}]}
            body=json.dumps({'model':'offline-fixture','usage':{'prompt_tokens':1,'completion_tokens':1},
                  'choices':[{'finish_reason':'stop','message':{'content':json.dumps(plan)}}]}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    config=out/'suite.json';write_json(config,{'schema':'IndependentAPISuite','model':'offline-fixture',
        'base_url':f'http://127.0.0.1:{server.server_port}/v1','pages':[{'id':'fixture','source':str(source),'source_sha256':digest(source),'analysis':str(analysis),'analysis_sha256':digest(analysis)}]})
    old=os.environ.get('VISION_LLM_API_KEY');os.environ['VISION_LLM_API_KEY']='offline-fixture-not-a-credential'
    try:
        for n in (1,2):
            result=evaluate_api(config,out/'campaign',ROOT,n)
            assert result['status']=='GENERATED' and result['requests_reserved']==1
            assert evaluate_api(config,out/'campaign',ROOT,n,resume=True)==result
            verify_generation(out/'campaign'/f'round-{n}')
        assert len(calls)==2
    finally:
        server.shutdown();server.server_close();thread.join()
        if old is None:os.environ.pop('VISION_LLM_API_KEY',None)
        else:os.environ['VISION_LLM_API_KEY']=old
    r1=out/'campaign/round-1';pptx=r1/read_json(r1/'generation.json')['pages'][0]['artifacts']['pptx']
    reference=out/'evaluation-only-reference.json'
    write_json(reference,{'schema':'APIReferenceSuite','reference_pptx':str(pptx),'reference_sha256':digest(pptx),'mapping':{'fixture':1}})
    for n in (1,2):score_api(out/'campaign'/f'round-{n}',reference,out/f'score-{n}',ROOT)
    compare_rounds(out/'score-1/score.json',out/'score-2/score.json',out/'comparison.json')
    write_json(out/'verification.json',{'status':'PASS','scope':'synthetic loopback API only','http_requests':len(calls),
               'rounds':2,'resumes_reissued_requests':False,'actual_pptx_compiled_and_reimported':True,
               'real_model_requests':0,'human_acceptance':False})
    print(json.dumps({'status':'PASS','report':str(out/'verification.json')}))


if __name__=='__main__':main()
