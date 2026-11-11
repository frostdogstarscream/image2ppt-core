"""L1 replay of actual asset evidence, with no model or Office dispatch."""
import argparse
import json
from pathlib import Path

from product.asset_geometry import normalize
from product.materialize import digest, write
from product.object_extract import load_page
from product.parallel_contracts import SceneGraph
from product.parallel_fusion import fuse
from product.parallel_rebuild import build
from product.parallel_store import Store


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True)
    parser.add_argument('--layers',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--strategy',choices=['alpha','auto'],default='auto')
    args=parser.parse_args()
    job=Path(args.job).resolve();output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    scene_file=job/'page-01/barrier-1.json'
    scene=SceneGraph.model_validate_json(scene_file.read_text('utf8'))
    source=job/'inputs/01.png'
    if digest(source)!=scene.source_sha256:raise ValueError('source differs from frozen evidence')
    ocr=None
    for ref in scene.text.references:
        file=Path(ref)
        if not file.is_file():continue
        try:record=json.loads(file.read_text('utf8'))
        except (ValueError,UnicodeError):continue
        if record.get('schema')=='OCRLines/1' and record.get('source_sha256')==scene.source_sha256:
            ocr=record;break
    if ocr is None:raise ValueError('source-bound OCR receipt required')
    store=Store(output/'state')
    asset,adapter=normalize(scene.asset,store)
    write(output/'coordinate-adapter.json',adapter)
    text=scene.text.model_copy(deep=True)
    text.references.append(store.evidence(ocr))
    snap={'path':str(source),'source_sha256':scene.source_sha256,
          'image_size':scene.image_size,'transform':scene.transform}
    graph=fuse(snap,{'text':text,'asset':asset,'surface':scene.surface},ocr)
    write(output/'scene-before-build.json',graph.model_dump())
    fonts=json.loads((job/'fonts.json').read_text('utf8'))
    ir,rows=build(graph,snap,fonts,output/'staged',layers=load_page(args.layers),
                  strategy=args.strategy,text_candidates=False)
    selected=[r for r in rows if r.get('kind')=='asset' and r['state']=='STAGED']
    report={'schema':'R1204AssetL1/1','source_sha256':scene.source_sha256,
        'input_scene_sha256':digest(scene_file),'input_layer_manifest_sha256':digest(Path(args.layers)/'manifest.json'),
        'adapter_sha256':digest(Path(__file__).parent/'product/asset_geometry.py'),
        'strategy':args.strategy,'attempted_assets':len(asset.objects),'selected_assets':[
            {'id':r['id'],'method':r['payload']['method'],'png':str(ir.parent/r['asset_reference']['path']),
             'png_sha256':r['asset_reference']['sha256'],'owner_evidence':r['owner_evidence']}
            for r in selected],
        'failed_assets':[r for r in rows if r['id'] in {o.id for o in asset.objects} and r['state']=='FALLBACK'],
        'slideir':str(ir),'model_calls':0,'office_status':'NOT_RUN_L1','go_status':'NOT_RUN_L1',
        'publication':'REVIEW_REQUIRED','actual_edit_acceptance':False}
    write(output/'result.json',report)
    write(output/'hashes.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    print(json.dumps({'attempted_assets':report['attempted_assets'],'selected_assets':[r['id'] for r in report['selected_assets']]}))


if __name__=='__main__':main()
