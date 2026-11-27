"""Bound long source-only planning by separating geometry from text decisions."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'services/vision/src'))
from image2ppt_vision.supervised_elements import parse_response
from image2ppt_vision.storage import write_json
from prepare_supervised_element_jobs import img,PROMPT
from image2ppt_vision.typography import font_status

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['objects','texts','assemble']);p.add_argument('--page',required=True);a=p.parse_args()
    r=ROOT/'outputs/supervised-elements-20261019';invfolder=r/'inventory-v2'/a.page;inv=json.loads((invfolder/'inventory.json').read_text(encoding='utf8'))
    if a.stage=='assemble':
        obj=parse_response(r/('objects-kimi-'+a.page)/'response.body');txt=parse_response(r/('texts-kimi-'+a.page)/'response.body')
        if obj.get('source_sha256')!=inv['source_sha256'] or txt.get('source_sha256')!=inv['source_sha256'] or obj.get('image_size')!=inv['image_size']:raise ValueError('stage source mismatch')
        plan={**obj,'schema':'ElementPlan/2' if txt.get('schema')=='ElementTexts/2' else 'ElementPlan/1','text_decisions':txt['text_decisions'],'limitations':[*obj.get('limitations',[]),*txt.get('limitations',[])]}
        write_json(r/('texts-kimi-'+a.page)/'assembled-plan.json',plan)
        print('Assembled source-bound plan without modifying model content.');return
    job=r/(a.stage+'-kimi-'+a.page);job.mkdir(exist_ok=False)
    data={k:inv[k] for k in ('source_sha256','image_size','texts')}
    data['available_families']=font_status()['families']
    content=[img(invfolder/'source.png')]
    if a.stage=='objects':
        prompt=PROMPT+'\nTHIS REQUEST IS OBJECTS STAGE ONLY. Return schema ElementObjects/1, source_sha256, image_size, background_color, objects, limitations. DO NOT output text_decisions (next stage). Concentrate on complete independent non-text objects. Assign all objects z < 1000; next-stage text will have z >= 1000. Every localized raster asset that contains an OCR bbox must declare its id in contains_text_ids, including false OCR detections that are actually icons; never paint fake OCR characters. Use ellipse mask only for full circular badges, rect for other crops. Keep accurate complete icon boundaries. Compact JSON.'
        data['candidates']=[{k:c[k] for k in ('id','bbox','geometry_hint','ocr_inside','parent_candidate')} for c in inv['candidates']]
        content.append(img(invfolder/'candidate-contact.jpg'));tokens=16000
    else:
        obj=parse_response(r/('objects-kimi-'+a.page)/'response.body')
        prompt=PROMPT+'\nTHIS REQUEST IS TEXT STAGE ONLY. Return schema ElementTexts/2, source_sha256, text_decisions, limitations. DO NOT repeat objects or image_size. The prior object plan is provided. Cover every OCR id exactly once, keeping its original id. Any OCR id in a raster contains_text_ids MUST have action retained_in_asset. All other IDs must be native. Some OCR tokens are actually icons; if an unclaimed icon is recognized, record a limitation so ownership can be repaired, never silently omit the id. Read original source carefully and correct genuine OCR errors. Assign text z>=1000. Return compact JSON.'
        data['object_plan']=obj;tokens=18000
    content.insert(0,{'type':'text','text':json.dumps(data,ensure_ascii=False)})
    write_json(job/'request.json',{'model':'kimi/kimi-k2.6','thinking':{'type':'disabled'},'messages':[{'role':'system','content':prompt},{'role':'user','content':content}],'max_tokens':tokens})
    print('Prepared '+job.name)

if __name__=='__main__':main()
