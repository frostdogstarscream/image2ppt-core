"""Synthetic real-compiler smoke. No external API, no image-fidelity claim."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from product.contracts import Page
from product.fonts import font_manifest
from product.materialize import materialize,digest,write
from product.pipeline import compile_pages

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output');args=parser.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    source=output/'source.png';image=Image.new('RGB',(960,540),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((20,20,940,90),fill='#EDF5FC');draw.ellipse((850,400,900,450),fill='#1780CD')
    rows=[('01','建设基础与升级方向'),('02','算网支撑与模型服务'),('03','系统协同与数据利用')]
    for i,(number,title) in enumerate(rows):
        y=110+i*90
        draw.text((40,y),number,font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',32),fill='#0078C8')
        draw.text((115,y),title,font=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',28),fill='#024481')
    image.save(source)
    styles=[{'id':'number','role':'编号','ea':'微软雅黑','latin':'Arial','size_px':32,'bold':True,'color':'0078C8'},
            {'id':'title','role':'并列标题','ea':'微软雅黑','latin':'Arial','size_px':28,'bold':True,'color':'024481'},
            {'id':'body','role':'说明','ea':'微软雅黑','latin':'Arial','size_px':24,'bold':False,'color':'333333'}]
    texts=[]
    for i,(n,t) in enumerate(rows):
        texts.extend([{'id':f'n{i}','text':n,'bbox':[40,110+i*90,60,48],'z':i*2+1,'style':'number'},
                      {'id':f't{i}','text':t,'bbox':[115,110+i*90,660,48],'z':i*2+2,'style':'title'}])
    texts.append({'id':'mixed','text':'业务应用 AI 2026','bbox':[40,400,600,50],'z':20,'style':'body',
                  'spans':[{'text':'业务应用 AI ','style':'body'},{'text':'2026','style':'number'}]})
    plan=Page.model_validate({'schema':'VisualPage/1','source_sha256':digest(source),'image_size':[960,540],
          'background':'FFFFFF','styles':styles,'texts':texts,
          'shapes':[{'id':'banner','bbox':[20,20,920,70],'z':0,'geometry':'roundRect','fill':'EDF5FC',
                     'corner_radius_px':12,'fill_gradient':{'angle_deg':0,'stops':[
                         {'offset':0,'color':'EDF5FC'},{'offset':100000,'color':'CDE6F8'}]}}],
          'crops':[{'id':'icon','bbox':[845,395,65,65],'z':21,'description':'synthetic circle crop'}],
          'repeated':[{'role':'编号','members':['n0','n1','n2'],'alignment':'left'},
                      {'role':'并列标题','members':['t0','t1','t2'],'alignment':'left'}],'limitations':[]})
    fonts=font_manifest([]);write(output/'plan.json',plan.model_dump())
    ir=materialize(plan,source,output/'scene',fonts,'synthetic-fixture')
    result=compile_pages([ir],output/'build',[plan]);write(output/'compiled.json',result)
    print(result['pptx'])

if __name__=='__main__':main()
