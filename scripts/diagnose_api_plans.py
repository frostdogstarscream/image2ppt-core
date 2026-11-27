"""Validate saved responses using their frozen code; no calls and no repairs."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run=Path(args.run).resolve();output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('diagnostic output must be new')
    if output.is_relative_to(run):raise ValueError('diagnostics must be outside the frozen run')
    sys.path.insert(0,str(run/'frozen/services/vision/src'))
    from image2ppt_vision.model_provider import _completion_content
    from image2ppt_vision.plans import ReconstructionPlan,parse_plan_json
    from pydantic import ValidationError
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    round_metadata=read(run/'round.json')
    metadata={p['id']:p for p in round_metadata['pages']}
    v2=round_metadata['protocol']=='2.0'
    if v2:
        from image2ppt_vision.plans_v2 import ReconstructionPlanV2,parse_plan_v2
    rows=[]
    for row in read(run/'generation.json')['pages']:
        if row.get('stage')!='plan_validation':continue
        page=row['id'];d={'id':page,'recorded_error':row.get('error_code')}
        try:
            content=_completion_content((run/'attempts'/page/'response.body').read_bytes())
            plan=json.loads(content)
            try:(ReconstructionPlanV2 if v2 else ReconstructionPlan).model_validate(plan)
            except ValidationError as exc:
                d['schema_errors']=[{'location':list(e['loc']),'type':e['type'],'message':e['msg']}
                                    for e in exc.errors(include_input=False,include_url=False,include_context=False)]
            analysis=read(run/'inputs'/page/'analysis.json')
            try:
                bindings=dict(source_sha256=metadata[page]['source_sha256'],image_size=analysis['image_size'])
                if v2:parse_plan_v2(content,**bindings,texts=[t for t in analysis['objects'] if t['kind']=='text'])
                else:parse_plan_json(content,**bindings,ocr_ids={t['id'] for t in analysis['objects'] if t['kind']=='text'})
            except Exception as exc:d['binding_error']=str(exc)
        except Exception as exc:d['response_error_type']=type(exc).__name__
        rows.append(d)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'method':'read-only frozen validation; no repair and no model call','pages':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(output),'failed_plans':len(rows)},ensure_ascii=False))


if __name__=='__main__':main()
