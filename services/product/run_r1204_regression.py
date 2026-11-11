"""L2 integration over real cached perception, with optional live regional VLM."""
import argparse
import getpass
import json
import os
from pathlib import Path
import shutil

from product.materialize import digest,write
from product.parallel_contracts import SceneGraph
from product.parallel_pipeline import run
from product.parallel_store import Store


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-job',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--layers',required=True)
    parser.add_argument('--text-region-goal',type=int,default=4)
    parser.add_argument('--regional-agent',action='store_true')
    parser.add_argument('--agent-skill')
    parser.add_argument('--prompt-key',action='store_true')
    args=parser.parse_args()
    prior=Path(args.from_job).resolve()
    if Store(prior/'state').committed() is None:raise ValueError('verified prior evidence required')
    scene=prior/'page-01/barrier-1.json'
    graph=SceneGraph.model_validate_json(scene.read_text('utf8'))
    source=prior/'inputs/01.png'
    if digest(source)!=graph.source_sha256:raise ValueError('source mismatch')
    ocr=None
    for reference in graph.text.references:
        path=Path(reference)
        if not path.is_file():continue
        try:row=json.loads(path.read_text('utf8'))
        except (ValueError,UnicodeError):continue
        if row.get('schema')=='OCRLines/1' and row.get('source_sha256')==graph.source_sha256:
            ocr=path;break
    if ocr is None:raise ValueError('independent OCR receipt required')
    output=Path(args.output).resolve();(output/'inputs').mkdir(parents=True,exist_ok=False)
    os.environ['IMAGE2PPT_PARALLEL_CACHE']=str(output/'cache')
    shutil.copyfile(source,output/'inputs/01.png')
    write(output/'replay.json',{'schema':'PerceptionReplay/1','pages':[{'source_sha256':graph.source_sha256,
        'scene':{'path':str(scene),'sha256':digest(scene)},'ocr':{'path':str(ocr),'sha256':digest(ocr)}}]})
    write(output/'baseline.json',{'pages':[{'source':{'path':str(source),'sha256':digest(source)},
        'layers':str(Path(args.layers).resolve())}],'stable_assets':[]})
    write(output/'request.json',{'sources':['01.png'],'pipeline_mode':'parallel_review','object_extraction':'auto',
        'candidate_engine':'r12-regional','regional_strategy':'adaptive','text_region_goal':args.text_region_goal,
        'regional_agent':args.regional_agent,'agent_skill':args.agent_skill,
        'perception_replay':str(output/'replay.json'),'frozen_baseline':str(output/'baseline.json')})
    if args.prompt_key:
        os.environ['MOMA_API_KEY']=getpass.getpass('MoMA credential (hidden): ')
        os.environ['OPENCODE_GO_API_KEY']=getpass.getpass('OpenCode Go credential (hidden): ')
    result=run(output,fonts=json.loads((prior/'fonts.json').read_text('utf8')),
        progress=lambda **value:print(json.dumps(value),flush=True))
    print(json.dumps({k:result[k] for k in ('status','locally_verified_text_regions','office_pass','workflow_states')},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
