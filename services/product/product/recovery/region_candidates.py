"""Product-layer local background trials, with optional demand-driven inpaint."""
import numpy as np
from PIL import Image

from ..materialize import digest,write
from ..parallel_store import key
from ..parallel_text_candidates import peer_protection
from ..parallel_rebuild import background_model
from ..parallel_fusion import glyph_evidence
from .gradient_backing import fit_gradient
from .text_mask_completion import TextMaskCompletion
from .layerd_adapter import lower_backgrounds
from .diagnostics import save_trial
from .background_adapter import RecoveryBlocked


def generate(graph,identifier,source_file,layer_directory,directory,*,inpaint=None,force_inpaint=False):
    directory.mkdir(parents=True,exist_ok=False)
    if digest(source_file)!=graph.source_sha256:raise RecoveryBlocked('SOURCE_BINDING','source changed')
    text=next(t for t in graph.text.texts if t.id==identifier)
    if graph.ownership[text.id]!='RESOLVED':raise RecoveryBlocked('OWNER_CONFLICT','unresolved source owner')
    source=np.array(Image.open(source_file).convert('RGB'))
    protected=peer_protection(text,graph,source,np.zeros(source.shape[:2],bool))
    x,y,w,h=map(round,text.bbox);box=[x-3,y-3,w+6,h+6];bx,by,bw,bh=box
    if min(bx,by)<0 or bx+bw>source.shape[1] or by+bh>source.shape[0]:
        raise RecoveryBlocked('MASK_INCOMPLETE','text scope leaves source')
    backing,_=background_model(source,box,protected)
    style=next(s for s in graph.text.styles if s.id==text.style)
    ink,_=glyph_evidence(source,box,style.color,backing)
    original=np.zeros(source.shape[:2],bool);original[by:by+bh,bx:bx+bw]=ink
    reference,fit=fit_gradient(source,box,protected)
    completion=TextMaskCompletion.analyze(source,original,protected,box,reference,glyph_height=style.size_px)
    write(directory/'mask-completion.json',completion.report)
    np.savez_compressed(directory/'mask-evidence.npz',**completion.evidence,**completion.masks)
    for name,mask in completion.evidence.items():
        Image.fromarray(mask[by:by+bh,bx:bx+bw].astype('uint8')*255).save(directory/(name+'.png'))
    lower=[];failures=[]
    if layer_directory:
        try:lower=lower_backgrounds(source,layer_directory,original)
        except (ValueError,OSError) as exc:failures.append({'stage':'layerd','code':getattr(exc,'code','BACKGROUND_REPAIR_FAILED')})
    records=[]
    def save(mask_name,method,raw,provenance):
        name=mask_name+'--'+method
        row=save_trial(directory/name,source,completion.masks[mask_name],raw,protected,box,reference,fit,
            {'candidate':name,'mask_variant':mask_name,'method':method,'object_id':identifier,
             'source_sha256':graph.source_sha256,'scene_graph_sha256':key(graph.model_dump()),
             'scene_sha256':key(graph.model_dump()),'provenance':provenance})
        records.append(row)
    for mask_name in completion.masks:
        save(mask_name,'gradient',reference,{'fit':fit})
        for index,bg in enumerate(lower):save(mask_name,'layerd-'+str(index),bg.image,bg.provenance)
    local_pass=any(row['status']=='PRECHECK_PASS' for row in records)
    if inpaint and (force_inpaint or not local_pass) and not completion.report['unassigned_contrast_pixels']:
        try:
            raw=inpaint(source,completion.masks['semantic_completion'])
            save('semantic_completion','big-lama',raw,getattr(inpaint,'receipt',None))
        except (ValueError,OSError,TimeoutError) as exc:
            failures.append({'stage':'lama','code':getattr(exc,'code','BACKGROUND_REPAIR_FAILED'),
                             'error_type':type(exc).__name__,'other_candidates_retained':True})
    passed=sorted([r for r in records if r['status']=='PRECHECK_PASS'],
                  key=lambda r:(r['scope_error_mae_rgb'],r['seam_max_rgb'],r['mask_pixels']))
    write(directory/'index.json',{'source_sha256':graph.source_sha256,'object_id':identifier,
        'candidates':records,'ranked_passes':[r['candidate'] for r in passed],'failures':failures,
        'lama_decision':'LOCAL_CANDIDATE_PASSED' if local_pass and not force_inpaint else 'LOCAL_REPAIR_UNPROVEN',
        'published':False})
    return passed,failures
