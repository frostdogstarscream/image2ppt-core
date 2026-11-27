"""Local materialization and independent-review request preparation for three pages."""
import argparse,json,sys,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'services/vision/src'))
from image2ppt_vision.storage import digest,write_json
from image2ppt_vision.supervised_elements import parse_response,materialize_elements
from prepare_supervised_element_jobs import img

BASE=ROOT/'outputs/supervised-elements-20261019'
REVIEW='''You are an independent skeptical visual supervisor reviewing screenshot-to-editable-PPT reconstruction. Image 1 is original; image 2 was reimported/rendered from actual final PPTX; image 3 is a contact sheet of the raster assets. You also receive actual PPTX object counts/text plus proposed editable objects. Do not defer to planner confidence. Compare Chinese text character by character, reading order, bbox position, font/color/size, card backgrounds, icon boundaries, arrow directions, gradients, panel proportions, missing decorative layers. Identify raster assets hiding multiple independent objects or duplicated original text behind editable text. Native counts alone cannot establish accurate decomposition. Do not pass a bitmap-only page. A localized complex raster icon is allowed but never call its internals native editable. Return ONLY JSON: {schema:"ElementReview/1",verdict:"PASS" or "FAIL",issues:[{id,severity:"critical" or "major" or "minor",category:"text" or "decomposition" or "geometry" or "style" or "missing",object_ids:[],bbox:[x,y,w,h] in original pixel coordinates,evidence:string,required_change:string}],text_corrections:[{id,expected_text,reason}],missing_elements:[],summary:string,limitations:[]}. Separate directly visible problems from uncertainties. PASS requires source-faithful appearance and meaningful element decomposition. This is model review only, never human acceptance.'''

def actual_inventory(pptx):
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with zipfile.ZipFile(pptx) as z:
        root=ET.fromstring(z.read('ppt/slides/slide1.xml'))
        shapes=root.findall('.//p:sp',ns);pictures=root.findall('.//p:pic',ns)
        rows=[]
        for s in shapes:
            name=s.find('p:nvSpPr/p:cNvPr',ns)
            rows.append({'name':name.get('name') if name is not None else '', 'text':''.join(t.text or '' for t in s.findall('.//a:t',ns)), 'geometry':s.find('.//a:prstGeom',ns).get('prst') if s.find('.//a:prstGeom',ns) is not None else None})
        return {'pptx_sha256':digest(pptx),'native_text_objects':sum(bool(s['text']) for s in rows),'native_shape_objects':sum(not s['text'] for s in rows),'pictures':len(pictures),'groups':len(root.findall('.//p:grpSp',ns)),'shape_inventory':rows}

def contact(ir,folder,out):
    assets=ir['assets'];cw,ch=240,170;cols=5
    canvas=Image.new('RGB',(cols*cw,max(1,(len(assets)+cols-1)//cols)*ch),'#DADDE2');d=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    for i,a in enumerate(assets):
        im=Image.open(folder/a['path']).convert('RGBA');im.thumbnail((cw-10,ch-35));x=i%cols*cw;y=i//cols*ch
        canvas.paste(im,(x+5,y+30),im);d.text((x+5,y+5),a['id'][:32],fill='black',font=font)
    canvas.save(out,quality=94)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['materialize','review']);p.add_argument('--page',required=True);p.add_argument('--job');p.add_argument('--output');p.add_argument('--build');p.add_argument('--plan');p.add_argument('--blind',action='store_true');a=p.parse_args()
    page=a.page;invfolder=BASE/'inventory-v2'/page;inv=json.loads((invfolder/'inventory.json').read_text(encoding='utf8'))
    if a.action=='materialize':
        job=BASE/(a.job or 'plan-kimi-'+page);out=BASE/(a.output or 'reconstruction-kimi-'+page)
        try:
            plan=json.loads(Path(a.plan).read_text(encoding='utf8')) if a.plan else parse_response(job/'response.body');write_json(job/'parsed-plan.json',plan)
            report=materialize_elements(invfolder/'source.png',inv,plan,out,model_id=json.loads((job/'request.json').read_text(encoding='utf8'))['model'],response_sha256=digest(job/'response.body'))
            print(json.dumps(report,ensure_ascii=False))
        except Exception as e:
            write_json(job/'materialization-error.json',{'error':str(e)});print(json.dumps({'error':str(e)},ensure_ascii=False));raise
    else:
        folder=BASE/(a.output or 'reconstruction-kimi-'+page);ir=json.loads((folder/'slideir.json').read_text(encoding='utf8'));build=Path(a.build)
        job=BASE/(a.job or 'review-qwen-'+page);job.mkdir(exist_ok=False)
        pptx=build/'result.pptx';preview=build/'preview.png'
        inventory=actual_inventory(pptx);write_json(job/'actual-pptx-inventory.json',inventory);contact(ir,folder,job/'asset-contact.jpg')
        data={'image_size':inv['image_size'],'actual_pptx':inventory,'proposed_objects':ir['nodes'],'ocr':inv['texts'],'raster_ownership':ir['assets']}
        pictures=[img(invfolder/'source.png'),img(preview),img(job/'asset-contact.jpg')];prompt=REVIEW
        if a.blind:
            data={'image_size':inv['image_size'],'actual_counts':{k:v for k,v in inventory.items() if k!='shape_inventory'},'object_locations':[{k:n[k] for k in ('id','type','bbox','geometry') if k in n} for n in ir['nodes']]}
            source=Image.open(invfolder/'source.png').convert('RGB');actual=Image.open(preview).convert('RGB').resize(source.size);w,h=source.size;regions=[]
            for index,(l,t,r,b) in enumerate([(0,0,w,h//3),(0,h//3,w//2,h*2//3),(w//2,h//3,w,h*2//3),(0,h*2//3,w,h)]):
                s=source.crop((l,t,r,b));v=actual.crop((l,t,r,b));pair=Image.new('RGB',(s.width*2,s.height+24),'white');pair.paste(s,(0,24));pair.paste(v,(s.width,24));draw=ImageDraw.Draw(pair);draw.text((5,5),'SOURCE',fill='black');draw.text((s.width+5,5),'ACTUAL PPTX',fill='black');f=job/f'region-{index}.png';pair.save(f);pictures.append(img(f));regions.append([l,t,r-l,b-t])
            data['paired_region_boxes']=regions
            prompt=REVIEW+'\nThis is a BLIND visual review: no OCR transcript or expected text is supplied. Image 3 is asset contact sheet; images 4 onward are paired crops, SOURCE LEFT, ACTUAL PPTX RIGHT. ALL boxes are [x,y,width,height], never x1,y1,x2,y2. Example [0,313,836,314] means height 314, not 1. Atomic localized raster icons are explicitly allowed: separate background shape + icon image + native label is the GOAL, not a decomposition defect. A screenshot provides no evidence of the original editable grouping. Do not ask to merge independently movable icons into large screenshot regions. Only flag icon splitting if actual visible pieces are cropped, misplaced or duplicated. Prioritize directly visible missing logos, wrong arrow directions, panel boundaries, typography size/color and source pixel fidelity. Read source characters yourself; if too blurry, state uncertain. Never turn an icon glyph into a standalone text character. Return at most 8 concrete issues, concise evidence under 200 characters each, all strings on a single line with properly escaped quotes. No speculative effects or claims about original object types. The response must parse as valid JSON.'
        write_json(job/'request.json',{'model':'qwen/qwen3-vl-plus','messages':[{'role':'system','content':prompt},{'role':'user','content':[{'type':'text','text':json.dumps(data,ensure_ascii=False)},*pictures]}],'max_tokens':12000})
        print(json.dumps({'review_job':str(job),'pptx_sha256':inventory['pptx_sha256']}))

if __name__=='__main__':main()
