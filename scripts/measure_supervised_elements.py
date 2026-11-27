"""Measure the actual reimport preview, without treating low error as human acceptance."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'services/vision/src'))
from image2ppt_vision.storage import digest,write_json
from supervised_element_pilot import actual_inventory

def main():
    p=argparse.ArgumentParser();p.add_argument('--ir',required=True);p.add_argument('--build',required=True);a=p.parse_args()
    irpath=Path(a.ir);ir=json.loads(irpath.read_text(encoding='utf8'));build=Path(a.build)
    out=build/'element-quality'
    if out.exists():raise FileExistsError('quality output must be new')
    out.mkdir()
    original=Image.open(irpath.parent/ir['source_image']).convert('RGB');render=Image.open(build/'preview.png').convert('RGB').resize(original.size,Image.Resampling.LANCZOS)
    src=np.asarray(original).astype(float);dst=np.asarray(render).astype(float);delta=np.abs(src-dst).mean(axis=2)
    # Dark/saturated source pixels avoid hiding defects under large white areas.
    foreground=(src.min(axis=2)<190)&((src.max(axis=2)-src.min(axis=2)>25)|(src.mean(axis=2)<120))
    regions=[]
    for node in ir['nodes']:
        x,y,w,h=node['bbox'];x,y=int(x),int(y);r,b=min(original.width,int(x+w+1)),min(original.height,int(y+h+1));crop=delta[y:b,x:r];fg=foreground[y:b,x:r]
        regions.append({'id':node['id'],'type':node['type'],'bbox':node['bbox'],'mean_abs_rgb_error':round(float(crop.mean()),3),'foreground_abs_rgb_error':round(float(crop[fg].mean()),3) if fg.any() else None,'within_10_rgb_error':bool(crop.mean()<=10)})
    report={'schema':'ElementQuality/1','pptx':actual_inventory(build/'result.pptx'),'preview_sha256':digest(build/'preview.png'),'ir_sha256':digest(irpath),'whole_page_mean_abs_rgb_error':round(float(delta.mean()),3),'foreground_abs_rgb_error':round(float(delta[foreground].mean()),3),'regions_within_10':sum(r['within_10_rgb_error'] for r in regions),'regions_total':len(regions),'regions':regions,'human_acceptance':False,'notes':['Pixel comparison uses actual-file reimport preview resized to original source dimensions.','Region MAE <= 10 is diagnostic, not proof of source text accuracy or native semantic editability.']}
    write_json(out/'quality.json',report)
    heat=np.zeros((*delta.shape,3),dtype=np.uint8);heat[:,:,0]=np.minimum(255,delta*4).astype(np.uint8);Image.fromarray(heat).save(out/'difference.png')
    target=(1000,round(1000*original.height/original.width));board=Image.new('RGB',(2000,target[1]+30),'white');board.paste(original.resize(target),(0,30));board.paste(render.resize(target),(1000,30));draw=ImageDraw.Draw(board);draw.text((15,8),'Original source',fill='black');draw.text((1015,8),'Actual PPTX reimport preview',fill='black');board.save(out/'comparison.png')
    print(json.dumps({k:report[k] for k in ('whole_page_mean_abs_rgb_error','foreground_abs_rgb_error','regions_within_10','regions_total')},ensure_ascii=False))

if __name__=='__main__':main()
