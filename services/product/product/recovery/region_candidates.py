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


def adaptive_candidates(graph, identifier, source_file, layer_directory, directory, *, inpaint=None, budget=None):
    """Lazy tool routing. The consumer advances only after a concrete defect.

    Incomplete masks are discarded before expensive background generation.
    A passing yielded candidate does not generate any later tool hypothesis.
    """
    directory.mkdir(parents=True, exist_ok=False)
    if digest(source_file) != graph.source_sha256:
        raise RecoveryBlocked('SOURCE_BINDING', 'source changed')
    text = next(t for t in graph.text.texts if t.id == identifier)
    if graph.ownership[text.id] != 'RESOLVED':
        raise RecoveryBlocked('OWNER_CONFLICT', 'unresolved source owner')
    source = np.array(Image.open(source_file).convert('RGB'))
    protected = peer_protection(text,graph,source,np.zeros(source.shape[:2],bool))
    x,y,w,h = map(round,text.bbox)
    box = [x-3,y-3,w+6,h+6]
    bx,by,bw,bh = box
    if min(bx,by)<0 or bx+bw>source.shape[1] or by+bh>source.shape[0]:
        raise RecoveryBlocked('MASK_INCOMPLETE','text scope leaves source')
    backing,_ = background_model(source,box,protected)
    style = next(s for s in graph.text.styles if s.id==text.style)
    ink,_ = glyph_evidence(source,box,style.color,backing)
    original = np.zeros(source.shape[:2],bool)
    original[by:by+bh,bx:bx+bw] = ink
    reference,fit = fit_gradient(source,box,protected)
    completion = TextMaskCompletion.analyze(source,original,protected,box,reference,glyph_height=style.size_px)
    write(directory/'mask-completion.json',completion.report)
    np.savez_compressed(directory/'mask-evidence.npz',**completion.evidence,**completion.masks)
    for name,mask in completion.evidence.items():
        Image.fromarray(mask[by:by+bh,bx:bx+bw].astype('uint8')*255).save(directory/(name+'.png'))
    if completion.report['unassigned_contrast_pixels']:
        raise RecoveryBlocked('MASK_INCOMPLETE','visible unassigned signal blocks repair')
    # Prefer the smallest complete legal mask. Do not spend a renderer trial on
    # the original mask when known AA/stroke pixels would necessarily remain.
    signal = (np.abs(source.astype(float)-reference.astype(float)).max(2)>6)
    scope = np.zeros(source.shape[:2],bool);scope[by:by+bh,bx:bx+bw]=True
    mask_name = next(name for name,mask in completion.masks.items()
                     if not (signal & scope & ~protected & ~mask).any())
    mask = completion.masks[mask_name]
    if fit['validation_p95_rgb']>3:
        raise RecoveryBlocked('INSUFFICIENT_CONTEXT','no trusted continuous background reference; expensive tools cannot repair that evidence')
    records=[];routes=[];failures=[]
    limit = int(budget.limits['region_candidates']) if budget else 2
    if completion.report['counts']['possible_text']:
        limit = int(budget.limits['complex_region_candidates']) if budget else 4
        routes.append({'reason':'bounded semantic stroke completion required','candidate_budget':limit})
    def receipt():
        write(directory/'index.json',{'source_sha256':graph.source_sha256,'object_id':identifier,
              'policy':'adaptive','routes':routes,'candidates':records,'failures':failures,
              'ranked_passes':[r['candidate'] for r in records if r['status']=='PRECHECK_PASS'],
              'generated_candidates':len(records),'published':False})
    def save(method,raw,provenance):
        if budget:budget.reserve('background_candidates',{'region':identifier,'scene':key(graph.model_dump()),'method':method,'mask':mask_name})
        name=mask_name+'--'+method
        row=save_trial(directory/name,source,mask,raw,protected,box,reference,fit,
            {'candidate':name,'mask_variant':mask_name,'method':method,'object_id':identifier,
             'source_sha256':graph.source_sha256,'scene_graph_sha256':key(graph.model_dump()),
             'scene_sha256':key(graph.model_dump()),'provenance':provenance})
        records.append(row);receipt()
        return row
    try:
        if layer_directory:
            try:
                from .diagnostics import assess_trial
                for index,bg in enumerate(lower_backgrounds(source,layer_directory,mask)):
                    assessment,_,_=assess_trial(source,mask,bg.image,protected,box,reference,fit,{})
                    routes.append({'tool':'layerd-'+str(index),'checks':assessment['checks'],
                                   'decision':'USE' if assessment['status']=='PRECHECK_PASS' else 'REJECT_LOWER_EVIDENCE'})
                    if assessment['status']=='PRECHECK_PASS':
                        yield save('layerd-'+str(index),bg.image,bg.provenance)
                        if len(records)>=limit:return
                        break
            except (ValueError,OSError) as exc:
                failures.append({'stage':'layerd','code':getattr(exc,'code','BACKGROUND_REPAIR_FAILED')})
        routes.append({'tool':'gradient','reason':'source-context continuous fit','validation_p95_rgb':fit['validation_p95_rgb']})
        yield save('gradient',reference,{'fit':fit})
        if len(records)>=limit:return
        if inpaint:
            routes.append({'tool':'big-lama','reason':'earlier candidate rejected by consumer or local gates'})
            try:
                raw=inpaint(source,mask)
                yield save('big-lama',raw,getattr(inpaint,'receipt',None))
            except (ValueError,OSError,TimeoutError) as exc:
                failures.append({'stage':'lama','status':'UNKNOWN' if isinstance(exc,(TimeoutError,OSError)) or getattr(exc,'code','')=='INPAINT_RESPONSE_UNKNOWN' else 'FAILED',
                                 'code':getattr(exc,'code','BACKGROUND_REPAIR_FAILED'),'error_type':type(exc).__name__})
    finally:
        receipt()


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
