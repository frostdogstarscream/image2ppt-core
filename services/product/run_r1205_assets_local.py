"""Replay current real asset failures locally without model or Office calls."""
import argparse
import json
from pathlib import Path

from product.materialize import digest, write
from product.parallel_contracts import SceneGraph
from product.recovery.asset_retry import retry_asset_extraction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    job=Path(args.job).resolve();output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    result=json.loads((job/'result.json').read_text('utf8'))
    baseline=Path(result['slide_irs'][0]).parent
    graph=SceneGraph.model_validate_json((baseline/'scene.json').read_text('utf8'))
    layer=next((job/'fresh-cache/layers').glob('*/manifest.json')).parent
    originals={str(baseline/row['asset_reference']['path']):row['asset_reference']['sha256']
               for row in graph.selected.values() if row['kind']=='asset'}
    records=[]
    for obj in graph.asset.objects:
        if obj.kind!='icon' or obj.id in graph.selected:continue
        record=retry_asset_extraction(graph,obj.id,baseline/'source.png',layer,baseline,output/obj.id)
        records.append(record)
        print(json.dumps({'id':obj.id,'status':record['status']}),flush=True)
    if any(digest(path)!=sha for path,sha in originals.items()):
        raise ValueError('retry modified an existing verified asset')
    write(output/'result.json',{'schema':'R1205AssetLocalReplay/1','source_sha256':graph.source_sha256,
        'input_scene_sha256':digest(baseline/'scene.json'),'original_asset_bytes_preserved':originals,
        'attempts':records,'model_calls':0,'office':'NOT_RUN','go':'NOT_RUN','published':False})


if __name__=='__main__':main()
