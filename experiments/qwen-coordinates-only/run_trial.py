"""One explicitly requested M06 coordinates-only Qwen trial; no automatic retries."""
import hashlib, json, math, pathlib, shutil, sys, time, uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'services/vision/src'))
from image2ppt_vision.auto_provider import image_content
from image2ppt_vision.auto_transport import SSHRemote

PROMPT = '''Locate the complete graphical objects in this slide screenshot.
Return ONLY a JSON array of bounding boxes: [[left, top, right, bottom], ...].
Use normalized 0..1000 xyxy coordinates relative to the entire supplied image, NOT width/height.
Include icons, logos, photos, illustrations, and important decorative graphics. Exclude text, text-containing card containers, and simple background fills or divider lines.
Keep a complex connected illustration or dense dashboard as one graphical object; do not duplicate its internal objects as separate boxes.
Each box must tightly enclose the WHOLE graphical object, including its badge/disc, shadow, glow, border, and connected parts. Do not cut off any part of an icon. Separate adjacent text from the graphic.
Inspect each object's actual visible edges individually. Do not assume identical box dimensions or equal spacing for repeated icons.
Do not return IDs, names, labels, categories, hierarchy, parent references, descriptions, explanations, markdown, or any other fields. Return coordinates only.'''

def digest(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p, v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    out=pathlib.Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'outputs/sam3-hardbox-ppt-20261103/ql-hardbox-20261103/input/M06.png'
    old=ROOT/'outputs/sam3-hardbox-ppt-20261103/ql-hardbox-20261103/plans/M06.json'
    shutil.copy2(source,out/'source.png');shutil.copy2(old,out/'old-plan.json')
    (out/'prompt.txt').write_text(PROMPT,encoding='utf-8')
    request={'request_id':uuid.uuid4().hex,'operation':'model','timeout':180,'body':{
        'model':'qwen/qwen3.8-max','temperature':0.1,'max_tokens':4000,'enable_thinking':False,
        'messages':[{'role':'system','content':PROMPT},{'role':'user','content':[
            {'type':'text','text':'Image dimensions: 1672 x 941 pixels. Return only normalized bounding box coordinates.'},image_content(out/'source.png')]}]}}
    write(out/'request.json',request)
    write(out/'binding.json',{'source_sha256':digest(out/'source.png'),'old_plan_sha256':digest(out/'old-plan.json'),
        'prompt_sha256':digest(out/'prompt.txt'),'runner_sha256':digest(__file__),'request_sha256':digest(out/'request.json'),
        'endpoint':'https://moma.cmecloud.cn/v1/chat/completions','model':request['body']['model'],
        'max_model_requests':1,'automatic_retries':0,'reserved_request_id':request['request_id']})
    config=json.loads((pathlib.Path('D:/Code/image2PPT/outputs/moma-vision-benchmark-20261029')/'client.json').read_text())
    print('Calling Qwen once for M06; no retries.',flush=True)
    response=SSHRemote(config).request(request,timeout=260);write(out/'response.json',response)
    result={'new_model_requests':1,'status':'FAILED','human_acceptance':False,
        'response_sha256':digest(out/'response.json'),'usage':response.get('usage',{}),'model':request['body']['model']}
    try:
        if not response.get('ok'):raise ValueError(str(response.get('error')))
        (out/'raw-output.txt').write_text(response['content'],encoding='utf-8')
        boxes=json.loads(response['content'])
        if not isinstance(boxes,list) or not 1<=len(boxes)<=100:raise ValueError('expected 1..100 boxes')
        for b in boxes:
            if not isinstance(b,list) or len(b)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in b):raise ValueError('invalid box')
            if not 0<=b[0]<b[2]<=1000 or not 0<=b[1]<b[3]<=1000:raise ValueError('invalid xyxy')
        write(out/'boxes.json',boxes)
        from PIL import Image,ImageDraw,ImageFont
        im=Image.open(out/'source.png').convert('RGB');w,h=im.size
        def draw_boxes(values):
            canvas=im.copy();d=ImageDraw.Draw(canvas)
            for b in values:d.rectangle([round(v*s/1000) for v,s in zip(b,[w,h,w,h])],outline='#A944DC',width=3)
            return canvas
        prior=json.loads((out/'old-plan.json').read_text())['plan']['regions'];parents={r.get('parent_id') for r in prior}
        oldboxes=[r['bbox'] for r in prior if r['action']=='keep_pixels' and r['id'] not in parents]
        oldim=draw_boxes(oldboxes);newim=draw_boxes(boxes);oldim.save(out/'old-boxes.png');newim.save(out/'new-boxes.png')
        panel=Image.new('RGB',(420,660),'white');d=ImageDraw.Draw(panel);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
        for i,(name,src) in enumerate([('原图',im),('旧提示词',oldim),('仅坐标',newim)]):
            d.text((i*140+10,8),name,font=font,fill='black');panel.paste(src.crop((45,245,145,825)),(i*140+15,45))
        panel.save(out/'icons-comparison.png')
        result.update(status='REVIEW_REQUIRED',old_boxes=len(oldboxes),new_boxes=len(boxes),coordinate_adjustment=False)
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Qwen 仅坐标试验</title><style>body{font-family:Microsoft YaHei;margin:28px;background:#eef2f6}img{max-width:100%}</style><h1>M06：原始框对照</h1><p>一次新 Qwen 调用；新框未经扩框、吸附、人工修正。仅坐标提示词同时取消了模块层级和其他分类任务，因此不能单独归因于删除编号。</p><p><a href="prompt.txt">新提示词</a> · <a href="boxes.json">新坐标</a> · <a href="raw-output.txt">模型原始输出</a></p><h2>左：原图；中：旧提示词；右：仅坐标</h2><img src="icons-comparison.png"><h2>旧框</h2><img src="old-boxes.png"><h2>新框</h2><img src="new-boxes.png">',encoding='utf-8')
    except Exception as error:result['error']=str(error)
    write(out/'result.json',result);print(json.dumps(result,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
