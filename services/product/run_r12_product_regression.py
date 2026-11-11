"""Run the real product entrypoint with hash-bound recorded branch evidence."""
import argparse
import getpass
import json
import os
from pathlib import Path
import shutil

from product.materialize import digest,write
from product.parallel_contracts import SceneGraph
from product.parallel_pipeline import run


class RecordedPerception:
    model='recorded-three-branch-perception'
    def ask(self,*args):
        raise RuntimeError('recorded-perception regression cannot issue fresh planning calls')


class UnavailableReview:
    model='fault-injected-go-unavailable'
    def ask(self,*args):
        raise TimeoutError('fault-injected Go unavailable')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',required=True);parser.add_argument('--source',required=True)
    parser.add_argument('--layers',required=True);parser.add_argument('--stable-scene',required=True)
    parser.add_argument('--fonts',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--stable-acceptance')
    parser.add_argument('--object-id',action='append',required=True)
    parser.add_argument('--prompt-go-key',action='store_true')
    parser.add_argument('--go-unavailable',action='store_true')
    parser.add_argument('--inject-region-failure')
    parser.add_argument('--inject-lama-timeout',action='store_true')
    args=parser.parse_args()
    graph=SceneGraph.model_validate_json(Path(args.scene).read_text('utf8'))
    if digest(args.source)!=graph.source_sha256:raise ValueError('regression source mismatch')
    ocr_file=None
    for ref in graph.text.references:
        file=Path(ref)
        if not file.is_file():continue
        try:record=json.loads(file.read_text('utf8'))
        except (ValueError,UnicodeError):continue
        if record.get('schema')=='OCRLines/1' and record.get('source_sha256')==graph.source_sha256:
            ocr_file=file;break
    if ocr_file is None:raise ValueError('independent recorded OCR evidence missing')
    output=Path(args.output).resolve();(output/'inputs').mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.source,output/'inputs/01.png')
    write(output/'perception-replay-inputs.json',{'schema':'PerceptionReplay/1','pages':[{
        'source_sha256':graph.source_sha256,'scene':{'path':str(Path(args.scene).resolve()),'sha256':digest(args.scene)},
        'ocr':{'path':str(ocr_file.resolve()),'sha256':digest(ocr_file)}}]})
    old=SceneGraph.model_validate_json(Path(args.stable_scene).read_text('utf8'))
    definitions={obj.id:obj for obj in graph.asset.objects};assets=[]
    for identifier,record in old.selected.items():
        if record['kind']!='asset' or identifier not in definitions:continue
        ref=record['asset_reference'];file=Path(args.stable_scene).parent/ref['path']
        if digest(file)!=ref['sha256']:raise ValueError('stable bytes changed')
        stable={'source_sha256':graph.source_sha256,'id':identifier,'bbox':record['bbox'],
            'path':str(file.resolve()),'sha256':digest(file),'definition':definitions[identifier].model_dump()}
        if args.stable_acceptance:
            owner=Path(args.stable_scene).parent/record['owner_evidence']['file']
            stable['verified_replacement']={name:{'path':str(Path(path).resolve()),'sha256':digest(path)}
                for name,path in [('scene',args.stable_scene),('owner',owner),('acceptance',args.stable_acceptance)]}
        assets.append(stable)
    write(output/'stable-inputs.json',{'pages':[{'source':{'sha256':graph.source_sha256,'path':str(Path(args.source).resolve())},
        'layers':str(Path(args.layers).resolve())}],'stable_assets':assets})
    write(output/'request.json',{'sources':['01.png'],'pipeline_mode':'parallel_review','object_extraction':'alpha',
        'candidate_engine':'r12-regional','regional_object_ids':args.object_id,
        'frozen_baseline':str(output/'stable-inputs.json'),'perception_replay':str(output/'perception-replay-inputs.json')})
    if args.prompt_go_key:os.environ['OPENCODE_GO_API_KEY']=getpass.getpass('OpenCode Go credential (hidden): ')
    if args.inject_region_failure or args.inject_lama_timeout:
        from product.recovery import product_pipeline
        from product.recovery.background_adapter import RecoveryBlocked
        original=product_pipeline.generate
        def inject(graph,identifier,*positional,**keywords):
            if identifier==args.inject_region_failure:
                raise RecoveryBlocked('MASK_INCOMPLETE','fault-injected regional mask failure')
            if args.inject_lama_timeout:
                def timeout(*args):raise TimeoutError('fault-injected LaMa timeout before dispatch')
                keywords.update(inpaint=timeout,force_inpaint=True)
            return original(graph,identifier,*positional,**keywords)
        product_pipeline.generate=inject
        write(output/'fault-injection.json',{'region':args.inject_region_failure,'failure':'MASK_INCOMPLETE',
              'lama_timeout':args.inject_lama_timeout,'go_unavailable':args.go_unavailable,'external_lama_dispatch':False})
    from product.parallel_provider import Reviewer
    reviewer=UnavailableReview() if args.go_unavailable else Reviewer(output)
    result=run(output,provider=RecordedPerception(),review_provider=reviewer,
        fonts=json.loads(Path(args.fonts).read_text('utf8')),
        progress=lambda **value:print(json.dumps(value),flush=True))
    print(json.dumps({k:result[k] for k in ('status','locally_verified_text_regions','office_pass','independent_review')},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
