"""Offline native-table integration fixture; never a real-image accuracy claim."""
import argparse
from pathlib import Path
import sys
import json
import subprocess
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'services/vision/src'))
from image2ppt_vision.storage import write_json,read_json,digest
from image2ppt_vision.reconstruction import reconstruct
from image2ppt_vision.api_scoring import inspect_objects,editing_probes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--protocol',choices=['1.1','2.0'],default='1.1');args=parser.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    source=out/'source.png';image=Image.new('RGB',(640,360),'white');draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    draw.rectangle([40,60,600,280],outline='black',width=2)
    draw.line([40,140,600,140],fill='black',width=2);draw.line([320,140,320,280],fill='black',width=2)
    items=[]
    for id,value,point in [('title','Synthetic table',(60,80)),('left','123',(60,170)),('right','456',(340,170))]:
        draw.text(point,value,font=font,fill='black');box=draw.textbbox(point,value,font=font)
        items.append({'id':id,'kind':'text','text':value,'score':1.,'bbox':[box[0],box[1],box[2]-box[0],box[3]-box[1]]})
    image.save(source)
    analysis={'schema':'VisionAnalysis','version':'1.0','image_size':[640,360],'objects':items,'inputs':{'source':{'sha256':digest(source)}}}
    plan={'schema':'ReconstructionPlan','version':'1.1','source_sha256':digest(source),'image_size':[640,360],
          'author':{'tool':'synthetic-fixture','model_id':'none'},'regions':[{'id':'table-region','bbox':[40,60,560,220],'kind':'panel'}],
          'tables':[{'id':'native-table','region_id':'table-region','bbox':[40,60,560,220],
                     'rows':[['Synthetic table',''],['123','456']],'column_widths':[280,280],'row_heights':[80,140],
                     'ocr_ids':[o['id'] for o in items],'merges':[{'row':0,'column':0,'row_span':1,'column_span':2}]}]}
    if args.protocol=='2.0':
        from image2ppt_vision.plans_v2 import parse_plan_v2,lower_plan_v2
        proposal={k:v for k,v in plan.items() if k not in ('regions','tables')};proposal['version']='2.0'
        proposal['tables']=[dict(id='native-table',bbox=[40,60,560,220],row_edges=[0,80/220,1],column_edges=[0,.5,1],
            cells=[dict(row=0,column=0,column_span=2,ocr_ids=['title']),
                   dict(row=1,column=0,ocr_ids=['left']),dict(row=1,column=1,ocr_ids=['right'])])]
        parsed=parse_plan_v2(json.dumps(proposal),source_sha256=digest(source),image_size=[640,360],texts=items)
        lowered,lowering=lower_plan_v2(parsed,items)
        write_json(out/'proposal-v2.json',proposal);write_json(out/'lowering.json',lowering)
        plan=lowered.to_dict()
    write_json(out/'analysis.json',analysis);write_json(out/'plan.json',plan)
    reconstruct(source,analysis,plan,out/'reconstruction',font_candidates=[{'path':'C:/Windows/Fonts/arial.ttf','family':'Arial','bold':False}])
    result=subprocess.run(['node',str(ROOT/'scripts/compose.mjs'),str(out/'reconstruction/slideir.json'),'--output',str(out/'build')],cwd=ROOT,capture_output=True)
    (out/'compile.stdout').write_bytes(result.stdout);(out/'compile.stderr').write_bytes(result.stderr)
    if result.returncode:raise RuntimeError('synthetic table compile failed; inspect retained logs')
    from image2ppt_vision.conversion import _final_json
    compiled=_final_json(result.stdout);actual=inspect_objects(compiled['pptx'])['slides'][0]['objects']
    tables=[o for o in actual if o['type']=='table'];assert len(tables)==1
    table=tables[0];assert table['cells']==plan['tables'][0]['rows']
    assert any(m.get('gridSpan')=='2' for m in table['merges'])
    probes=editing_probes(compiled['pptx'],actual,out/'probes',ROOT)
    assert any(p['action']=='edit_table_cell' and p['status']=='PASS' for p in probes['cases'])
    write_json(out/'verification.json',{'status':'PASS','scope':'synthetic table only','native_tables':1,'merged_cells_verified':True,
              'source_preserved':digest(source)==plan['source_sha256'],'probes':probes,'human_acceptance':False,'artifacts':compiled})
    print(json.dumps({'status':'PASS','report':str(out/'verification.json')}))


if __name__=='__main__':main()
