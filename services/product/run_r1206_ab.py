"""L1 A/B/C over the same frozen upload, without model or Office calls."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from product.parallel_contracts import SceneGraph,AssetEvidence
from product.parallel_fusion import fuse
from product.parallel_rebuild import build
from product.materialize import digest,write
from product.object_extract import rectangle
from product.recovery.text_probe import probe_one
from product.recovery.text_actions import classify_failure
from product.parallel_text_candidates import peer_protection


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True);parser.add_argument('--layers',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--variants',default='ABC');args=parser.parse_args()
    job=Path(args.job).resolve();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    graph=SceneGraph.model_validate_json((job/'page-01/barrier-1.json').read_text('utf8'))
    source_file=job/'inputs/01.png';source=np.array(Image.open(source_file).convert('RGB'))
    fonts=json.loads((job/'fonts.json').read_text('utf8'));layers=Path(args.layers).resolve()
    result=json.loads((job/'result.json').read_text('utf8'));baseline=Path(result['slide_irs'][0]).parent
    ocr=next(json.loads(Path(p).read_text('utf8')) for p in graph.text.references if Path(p).is_file() and json.loads(Path(p).read_text('utf8')).get('schema')=='OCRLines/1')
    snap={'source_sha256':graph.source_sha256,'image_size':graph.image_size,'transform':graph.transform,'path':str(source_file)}
    empty=AssetEvidence.model_validate({'schema':'AssetEvidence/1','source_sha256':graph.source_sha256,'image_size':graph.image_size,'transform':graph.transform,'method_version':'diagnostic-without-asset/1'})
    c=fuse(snap,{'text':graph.text,'asset':graph.asset,'surface':graph.surface},ocr)
    a=fuse(snap,{'text':graph.text,'asset':empty,'surface':graph.surface},ocr)
    ir,_=build(a,snap,fonts,output/'a-base',strategy='off',text_candidates=False)
    def old_protection(text,graph,source,blocked,**kwargs):
        protected=blocked.copy()
        for peer in [*[t for t in graph.text.texts if t.id!=text.id],*graph.asset.objects]:protected|=rectangle(peer.bbox,source.shape[1::-1])
        return (protected,protected) if kwargs.get('purpose')=='both' else protected
    import product.parallel_text_candidates as text_candidates
    import product.recovery.region_candidates as candidates
    import product.recovery.candidate_stage as stage
    summary={'schema':'R1206FrozenABC/1','source_sha256':digest(source_file),'model_calls':0,'office_calls':0,'variants':{}}
    for name,current,base,protection in [('A',a,ir.parent,old_protection),('B',graph,baseline,old_protection),('C',c,baseline,peer_protection)]:
        if name not in args.variants:continue
        text_candidates.peer_protection=protection;candidates.peer_protection=protection;stage.peer_protection=protection
        rows=[]
        for text in current.text.texts:
            if current.ownership[text.id]!='RESOLVED':
                row={'object_id':text.id,'state':'BLOCKED','failure':'OWNER_CONFLICT','reason':current.ownership[text.id]}
            else:
                try:row=probe_one(current,text.id,source_file,layers,base,fonts,output/name)
                except Exception as exc:row={'object_id':text.id,'state':'BLOCKED','reason':str(exc),'error_type':type(exc).__name__}
            row['failure_category']=classify_failure(row);rows.append(row)
            print(json.dumps({'variant':name,'id':text.id,'state':row['state'],'failure':row['failure_category']}),flush=True)
            if name=='C':
                details=[];new=peer_protection(text,current,source,np.zeros(source.shape[:2],bool),diagnostics=details)
                old=old_protection(text,current,source,np.zeros(source.shape[:2],bool))
                directory=output/'protection'/text.id;directory.mkdir(parents=True)
                masks={'protected_before':old,'protected_after':new,'removed_protection':old&~new}
                for i,detail in enumerate(details):
                    mask=detail.pop('mask');detail['mask_key']='peer_'+str(i);masks[detail['mask_key']]=mask
                    detail['protected_pixels']=int(mask.sum())
                np.savez_compressed(directory/'masks.npz',**masks)
                write(directory/'report.json',{'source_sha256':current.source_sha256,'object_id':text.id,
                    'protected_before':int(old.sum()),'protected_after':int(new.sum()),'removed_protection':int((old&~new).sum()),
                    'mask_sha256':digest(directory/'masks.npz'),'peers':details})
        summary['variants'][name]={'states':dict(Counter(r['state'] for r in rows)),
            'ownership':dict(Counter(current.ownership[t.id] for t in current.text.texts)),
            'objects':rows,'actual_textboxes':'NOT_RUN_L1','images_in_baseline':len(json.loads((base/'scene.json').read_text('utf8'))['selected'])}
        write(output/'summary.json',summary)
    text_candidates.peer_protection=peer_protection;candidates.peer_protection=peer_protection;stage.peer_protection=peer_protection


if __name__=='__main__':main()
