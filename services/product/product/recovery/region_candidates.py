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


def _with_context_provenance(provenance, ctx_w):
    if ctx_w == 14:
        return provenance
    prov = dict(provenance) if isinstance(provenance, dict) else ({'receipt': provenance} if provenance is not None else {})
    prov.setdefault('background_context_width', ctx_w)
    return prov


def _safe_background_model(source, box, excluded):
    try:
        return background_model(source, box, excluded)
    except ValueError:
        height, width = source.shape[:2]
        x, y, w, h = [int(v) for v in box]
        strips = []
        for top, bot in ((max(0, y - 7), max(0, y - 1)), (min(height, y + h + 1), min(height, y + h + 7))):
            if bot <= top:
                continue
            sub = source[top:bot, x:x + w].astype(float)
            allow = ~excluded[top:bot, x:x + w]
            vals = sub[allow]
            if len(vals) >= max(8, w // 4):
                strips.append(np.median(vals, axis=0))
        if not strips:
            raise
        c0, c1 = strips[0], strips[-1]
        t = np.linspace(0.0, 1.0, h)[:, None, None]
        est = np.broadcast_to(c0[None, None, :] * (1.0 - t) + c1[None, None, :] * t, (h, w, 3))
        return np.clip(np.rint(est), 0, 255).astype(np.uint8), 4.0


def _fit_gradient_adaptive(source, box, context_protected):
    last_exc = None
    best = None
    for allow_vert, widths in ((False, (14, 28, 44)), (True, (14, 22, 34))):
        for ctx_w in widths:
            try:
                ref, fit = fit_gradient(source, box, context_protected, context=ctx_w, allow_vertical=allow_vert)
                if fit['validation_p95_rgb'] <= 3.0:
                    return ref, fit, ctx_w
                if best is None or fit['validation_p95_rgb'] < best[1]['validation_p95_rgb']:
                    best = (ref, fit, ctx_w)
            except RecoveryBlocked as exc:
                if exc.code != 'INSUFFICIENT_CONTEXT':
                    raise
                last_exc = exc
    if best is not None:
        return best
    raise last_exc


def _fallback_nonplanar_reference(source, box, backing, original, layer_directory):
    bx, by, bw, bh = box
    ref = source.copy()
    ref[by:by + bh, bx:bx + bw] = backing
    if layer_directory:
        try:
            lows = lower_backgrounds(source, layer_directory, original)
            if lows:
                ref = lows[0].image.copy()
        except Exception:
            pass
    ref.setflags(write=False)
    return ref, {
        'schema': 'NonPlanarFallbackReference/1',
        'bbox': list(box),
        'validation_p95_rgb': 99.0,
        'validation_mae_rgb': 99.0,
        'reference': 'nonplanar fallback reference; requires LayerD ring or Big-LaMa seam+ablation proof',
    }, 14


def adaptive_candidates(graph, identifier, source_file, layer_directory, directory, *, inpaint=None, budget=None, baseline_directory=None, marked_fields=False):
    """Lazy tool routing. The consumer advances only after a concrete defect.

    Incomplete masks are discarded before expensive background generation.
    A passing yielded candidate does not generate any later tool hypothesis.
    """
    directory.mkdir(parents=True, exist_ok=False)
    if digest(source_file) != graph.source_sha256:
        raise RecoveryBlocked('SOURCE_BINDING', 'source changed')
    text = next(t for t in graph.text.texts if t.id == identifier)
    source = np.array(Image.open(source_file).convert('RGB'))
    details=[]
    protected,context_protected=peer_protection(text,graph,source,np.zeros(source.shape[:2],bool),baseline_directory=baseline_directory,purpose='both',diagnostics=details)
    from ..peer_evidence import save_diagnostics
    save_diagnostics(directory,text,graph,source,protected,context_protected,details)
    if graph.ownership[text.id]!='RESOLVED':raise RecoveryBlocked('OWNER_CONFLICT','unresolved source owner')
    x,y,w,h = map(round,text.bbox)
    box = [x-3,y-3,w+6,h+6]
    bx,by,bw,bh = box
    if min(bx,by)<0 or bx+bw>source.shape[1] or by+bh>source.shape[0]:
        raise RecoveryBlocked('MASK_INCOMPLETE','text scope leaves source')
    try:
        backing,_ = _safe_background_model(source,box,context_protected)
    except ValueError as exc:
        raise RecoveryBlocked('INSUFFICIENT_CONTEXT', str(exc)) from None
    style = next(s for s in graph.text.styles if s.id==text.style)
    ink,_ = glyph_evidence(source,box,style.color,backing)
    original = np.zeros(source.shape[:2],bool)
    original[by:by+bh,bx:bx+bw] = ink
    try:
        reference,fit,ctx_w = _fit_gradient_adaptive(source,box,context_protected)
    except RecoveryBlocked as exc:
        if exc.code == 'INSUFFICIENT_CONTEXT' and (layer_directory or inpaint):
            reference,fit,ctx_w = _fallback_nonplanar_reference(source,box,backing,original,layer_directory)
        else:
            raise
    shared = graph.text.members.get(identifier)
    if marked_fields and shared and any(k!=identifier and ids==shared for k,ids in graph.text.members.items()):
        from .marked_fields import striped_backing
        stripe = striped_backing(source, box, context_protected)
        if stripe and stripe[1]['validation_p95_rgb'] < fit['validation_p95_rgb']:
            reference,fit = stripe
    completion = TextMaskCompletion.analyze(source,original,protected,box,reference,glyph_height=style.size_px)
    write(directory/'mask-completion.json',completion.report)
    np.savez_compressed(directory/'mask-evidence.npz',**completion.evidence,**completion.masks)
    for name,mask in completion.evidence.items():
        Image.fromarray(mask[by:by+bh,bx:bx+bw].astype('uint8')*255).save(directory/(name+'.png'))
    delta_ref = np.abs(source.astype(float)-reference.astype(float)).max(2)
    unassigned_mask = completion.evidence['unknown'] & (delta_ref > 6)
    unassigned_px = int(unassigned_mask.sum())
    unassigned_peak = float(delta_ref[unassigned_mask].max()) if unassigned_px else 0.0
    mild_interior_nonplanar = (fit.get('method')!='scanline-lateral' and
        unassigned_px > 0
        and bool(layer_directory or inpaint)
        and (unassigned_peak <= 22.0 or (unassigned_px <= 4 and unassigned_peak <= 16.0))
    )
    if fit['validation_p95_rgb'] <= 3 and unassigned_px and not mild_interior_nonplanar:
        raise RecoveryBlocked('MASK_INCOMPLETE','visible unassigned signal blocks repair')
    # Prefer the smallest complete legal mask. Do not spend a renderer trial on
    # the original mask when known AA/stroke pixels would necessarily remain.
    signal = (delta_ref > 6)
    scope = np.zeros(source.shape[:2],bool);scope[by:by+bh,bx:bx+bw]=True
    use_planar = (fit['validation_p95_rgb'] <= 3 and not mild_interior_nonplanar)
    if use_planar:
        mask_name = next(name for name,mask in completion.masks.items()
                         if not (signal & scope & ~protected & ~mask).any())
    else:
        if not layer_directory and not inpaint:
            raise RecoveryBlocked('INSUFFICIENT_CONTEXT','no trusted continuous background reference; expensive tools cannot repair that evidence')
        mask_name = 'semantic_completion'
    mask = completion.masks[mask_name]
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
             'scene_sha256':key(graph.model_dump()),'provenance':_with_context_provenance(provenance,ctx_w)})
        records.append(row);receipt()
        return row
    try:
        if layer_directory:
            try:
                from .diagnostics import assess_trial
                for index,bg in enumerate(lower_backgrounds(source,layer_directory,mask)):
                    method_name='layerd-'+str(index)
                    assessment,_,_=assess_trial(source,mask,bg.image,protected,box,reference,fit,{'method':method_name,'provenance':bg.provenance})
                    routes.append({'tool':method_name,'checks':assessment['checks'],
                                   'decision':'USE' if assessment['status']=='PRECHECK_PASS' else 'REJECT_LOWER_EVIDENCE'})
                    if assessment['status']=='PRECHECK_PASS':
                        yield save(method_name,bg.image,bg.provenance)
                        if len(records)>=limit:return
                        break
            except (ValueError,OSError) as exc:
                failures.append({'stage':'layerd','code':getattr(exc,'code','BACKGROUND_REPAIR_FAILED')})
        if use_planar:
            routes.append({'tool':'gradient','reason':'source-context continuous fit','validation_p95_rgb':fit['validation_p95_rgb']})
            yield save('gradient',reference,{'fit':fit})
            if len(records)>=limit:return
        else:
            routes.append({'tool':'gradient','decision':'SKIP_NONPLANAR_CONTEXT','validation_p95_rgb':fit['validation_p95_rgb']})
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


def generate(graph,identifier,source_file,layer_directory,directory,*,inpaint=None,force_inpaint=False,baseline_directory=None):
    directory.mkdir(parents=True,exist_ok=False)
    if digest(source_file)!=graph.source_sha256:raise RecoveryBlocked('SOURCE_BINDING','source changed')
    text=next(t for t in graph.text.texts if t.id==identifier)
    source=np.array(Image.open(source_file).convert('RGB'))
    details=[]
    protected,context_protected=peer_protection(text,graph,source,np.zeros(source.shape[:2],bool),baseline_directory=baseline_directory,purpose='both',diagnostics=details)
    from ..peer_evidence import save_diagnostics
    save_diagnostics(directory,text,graph,source,protected,context_protected,details)
    if graph.ownership[text.id]!='RESOLVED':raise RecoveryBlocked('OWNER_CONFLICT','unresolved source owner')
    x,y,w,h=map(round,text.bbox);box=[x-3,y-3,w+6,h+6];bx,by,bw,bh=box
    if min(bx,by)<0 or bx+bw>source.shape[1] or by+bh>source.shape[0]:
        raise RecoveryBlocked('MASK_INCOMPLETE','text scope leaves source')
    try:
        backing,_=_safe_background_model(source,box,context_protected)
    except ValueError as exc:
        raise RecoveryBlocked('INSUFFICIENT_CONTEXT', str(exc)) from None
    style=next(s for s in graph.text.styles if s.id==text.style)
    ink,_=glyph_evidence(source,box,style.color,backing)
    original=np.zeros(source.shape[:2],bool);original[by:by+bh,bx:bx+bw]=ink
    try:
        reference,fit,ctx_w=_fit_gradient_adaptive(source,box,context_protected)
    except RecoveryBlocked as exc:
        if exc.code == 'INSUFFICIENT_CONTEXT' and (layer_directory or inpaint):
            reference,fit,ctx_w=_fallback_nonplanar_reference(source,box,backing,original,layer_directory)
        else:
            raise
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
             'scene_sha256':key(graph.model_dump()),'provenance':_with_context_provenance(provenance,ctx_w)})
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
