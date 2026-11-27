"""Render a schema-invalid response's unchanged bbox_2d values, without retrying."""
import hashlib,json,pathlib,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont

p=pathlib.Path(sys.argv[1])
def read(name):return json.loads((p/name).read_text(encoding='utf-8'))
def write(name,value):(p/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
response=read('response.json');raw=json.loads(response['content']);boxes=[r['bbox_2d'] for r in raw]
assert all(isinstance(b,list) and len(b)==4 and all(type(v) in (int,float) for v in b) and 0<=b[0]<b[2]<=1000 and 0<=b[1]<b[3]<=1000 for b in boxes)
im=Image.open(p/'source.png').convert('RGB');w,h=im.size
assert hashlib.sha256((p/'source.png').read_bytes()).hexdigest()==read('binding.json')['source_sha256']
prior=read('old-plan.json')['plan']['regions'];parents={r.get('parent_id') for r in prior}
oldboxes=[r['bbox'] for r in prior if r['action']=='keep_pixels' and r['id'] not in parents]
def pixel(b):return [round(v*s/1000) for v,s in zip(b,[w,h,w,h])]
def draw_boxes(bs):
    canvas=im.copy();draw=ImageDraw.Draw(canvas)
    for b in bs:draw.rectangle(pixel(b),outline='#A944DC',width=3)
    return canvas
oldim=draw_boxes(oldboxes);newim=draw_boxes(boxes)
oldim.save(p/'old-boxes.png');newim.save(p/'new-boxes-diagnostic.png')
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
panel=Image.new('RGB',(450,650),'white');draw=ImageDraw.Draw(panel)
for i,(title,source) in enumerate([('原图',im),('旧提示词',oldim),('简化提示词',newim)]):
    draw.text((i*150+8,8),title,font=font,fill='#183047');panel.paste(source.crop((45,245,145,825)),(i*150+20,48))
panel.save(p/'icons-comparison.png')
a=np.asarray(im).astype(int);measure=[]
for oldrow in [r for r in prior if r['id'] in ['G04','G05','G06','G07','G08']]:
    oldb=pixel(oldrow['bbox']);xx0,xx1=65,125;yy0,yy1=oldb[1]-20,oldb[3]+25
    q=a[yy0:yy1,xx0:xx1];mask=(q.max(2)-q.min(2)>100)&(q.min(2)<145)
    ys,xs=np.where(mask);xs=xs+xx0;ys=ys+yy0
    def coverage(b):
        l,t,r,bottom=b;return float(((xs>=l)&(xs<r)&(ys>=t)&(ys<bottom)).mean())
    candidates=[b for b in boxes if b[0]<100 and abs((b[1]+b[3])/2-(oldrow['bbox'][1]+oldrow['bbox'][3])/2)<40]
    nb=max(candidates,key=lambda b:coverage(pixel(b))) if candidates else None
    measure.append({'old_id':oldrow['id'],'name':oldrow['label'],'old_box_px':oldb,'new_box_normalized':nb,'new_box_px':pixel(nb) if nb else None,'old_colored_pixel_coverage':round(coverage(oldb),4),'new_colored_pixel_coverage':round(coverage(pixel(nb)),4) if nb else 0})
write('extracted-boxes-diagnostic.json',boxes);write('icon-diagnostics.json',measure)
result=read('result.json');result.update(diagnostic_render_only=True,coordinate_adjustment=False,raw_schema_compliant=False,raw_output_rows=len(raw),usable_bbox_rows=len(boxes),unsolicited_fields=['label'],actual_response_shape='objects with bbox_2d and label; expected coordinate arrays',metric_scope='saturated colored icon pixels only; excludes white badge, glow, shadow; not full icon accuracy',scope_caution='Task simplified beyond removing IDs; individual-edge instruction also added; single run not causal attribution',icon_metrics=measure)
write('diagnostic-result.json',result)
parts=['<!doctype html><meta charset="utf-8"><title>M06 简化提示词试验</title><style>body{font-family:Microsoft YaHei;margin:28px;background:#eef2f6;color:#183047}img{max-width:100%}section{background:white;padding:20px;margin:20px 0}td,th{border:1px solid #aaa;padding:8px}</style><h1>M06 简化提示词试验</h1><p>1 张图，1 次 Qwen3.8-Max 调用，无重试，无坐标修正。原始输出含 bbox_2d 和 label，未遵守仅数组契约；原运行状态保持 FAILED。这里只提取原始坐标做诊断绘图。</p><p>同时简化了任务范围与输出字段，并强调逐个观察边缘；一次结果不能单独归因于删除编号。覆盖指标只测彩色主体，不含白色底座与阴影，不等于整体分割精度。</p><p><a href="prompt.txt">新提示词</a> · <a href="raw-output.txt">模型原始输出</a> · <a href="response.json">完整响应</a> · <a href="diagnostic-result.json">诊断记录</a></p><section><h2>左：原图；中：旧框；右：简化提示词新框</h2><img src="icons-comparison.png"><table><tr><th>图标</th><th>旧框彩色主体覆盖</th><th>新框彩色主体覆盖</th></tr>']
for r in measure:parts.append(f'<tr><td>{r["name"]}</td><td>{r["old_colored_pixel_coverage"]:.1%}</td><td>{r["new_colored_pixel_coverage"]:.1%}</td></tr>')
parts.extend(['</table></section><section><h2>旧框：21 个图片目标</h2><img src="old-boxes.png"></section><section><h2>新框：31 个图形候选</h2><img src="new-boxes-diagnostic.png"></section>'])
(p/'index.html').write_text('\n'.join(parts),encoding='utf-8')
print(json.dumps({'new_model_requests':1,'boxes':len(boxes),'metrics':measure},ensure_ascii=True,indent=2))
