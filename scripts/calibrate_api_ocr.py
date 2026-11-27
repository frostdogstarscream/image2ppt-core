"""Post-generation OCR diagnostic against original and equally scaled sources."""
import argparse
import json
from pathlib import Path
import sys
import subprocess

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'services/vision/src'))
from image2ppt_vision.api_evaluation import verify_generation
from image2ppt_vision.api_scoring import content_metrics
from image2ppt_vision.storage import read_json,write_json,digest
from PIL import Image


def main():
    parser=argparse.ArgumentParser()
    for name in ('run','score','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--render-native-resolution',action='store_true')
    args=parser.parse_args();run=Path(args.run).resolve();score_dir=Path(args.score).resolve();out=Path(args.output).resolve()
    verify_generation(run);out.mkdir(parents=True,exist_ok=False)
    score=read_json(score_dir/'score.json');references=read_json(score_dir/'reference-objects.json')['slides']
    generation={p['id']:p for p in read_json(run/'generation.json')['pages']}
    results=[]
    for page in score['pages']:
        if 'content' not in page:continue
        id=page['id'];source=run/'inputs'/id/'source.png';analysis=read_json(run/'inputs'/id/'analysis.json')
        reference=references[page['reference_slide']-1]['objects']
        preview=run/generation[id]['artifacts']['preview']/'page-001.png'
        with Image.open(preview) as im:target=im.size
        resized=out/(id+'-source-at-preview-size.png')
        with Image.open(source) as im:im.convert('RGB').resize(target,Image.Resampling.LANCZOS).save(resized)
        metrics=lambda p:content_metrics(reference,[],analysis,p)['rendered_content']
        original=metrics(source);scaled=metrics(resized);generated=page['content']['rendered_content']
        result={'id':id,'source_sha256':digest(source),'source_original':original,
                        'source_at_preview_size':scaled,'generated_render':generated,
                        'generated_minus_same_size_source_accuracy':
                            generated['character_accuracy']-scaled['character_accuracy']
                            if generated.get('character_accuracy') is not None and scaled.get('character_accuracy') is not None else None}
        if args.render_native_resolution:
            from image2ppt_vision.conversion import _compiler,_final_json
            root,node=_compiler(Path(__file__).resolve().parents[1])
            with Image.open(source) as im:scale=im.width/target[0]
            native=out/(id+'-native-size-render')
            done=subprocess.run([str(node),str(root/'scripts/render_for_ocr.mjs'),
                                 str(run/generation[id]['artifacts']['pptx']),str(native),str(scale)],
                                cwd=root,capture_output=True,timeout=120)
            (out/(id+'-native-size-render.stderr.log')).write_bytes(done.stderr)
            if done.returncode:raise RuntimeError('native size render failed')
            rendered=_final_json(done.stdout);write_json(out/(id+'-native-size-render.json'),rendered)
            result['generated_at_source_resolution']=metrics(Path(rendered['pages'][0]['path']))
        results.append(result)
    write_json(out/'report.json',{'method':'same OCR and source-confirmed reference regions; diagnostic only, not a corrected content score',
                               'pages':results,'human_acceptance':False})
    verify_generation(run)
    print(json.dumps({'report':str(out/'report.json'),'pages':len(results)},ensure_ascii=False))


if __name__=='__main__':main()
