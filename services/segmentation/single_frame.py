"""Offline SAM 3 / 3.1 single-frame adapter, identical public predictor protocol.

Run in a separate GPU environment. This module never downloads weights, replaces
LayerD dependencies, or falls back to another model. A successful synthetic smoke
proves interface compatibility only, not static-slide segmentation quality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import inspect
import numpy as np
from PIL import Image


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def normalize_box(box,size):
    x,y,w,h=box;sw,sh=size
    if min(x,y)<0 or min(w,h)<=0 or x+w>sw or y+h>sh:raise ValueError('invalid prompt box')
    return [x/sw,y/sh,w/sw,h/sh]


def arrays(outputs,size):
    def array(value):
        if hasattr(value,'detach'):value=value.detach().float().cpu().numpy()
        return np.asarray(value)
    masks=array(outputs['out_binary_masks']);scores=array(outputs['out_probs'])
    if masks.ndim!=3 or masks.shape[1:]!=(size[1],size[0]) or len(masks)!=len(scores):raise ValueError('invalid full-page output dimensions')
    if not np.isfinite(masks).all() or not np.isfinite(scores).all():raise FloatingPointError('nonfinite model output')
    return masks.astype(bool),scores.astype(np.float32)


def bridge_disabled_state_offload(model):
    original=model.init_state
    if 'offload_state_to_cpu' in inspect.signature(original).parameters:return False
    def compatible(*args,**kwargs):
        flag=kwargs.pop('offload_state_to_cpu',False)
        if flag is not False:raise ValueError('This model does not support state offload')
        return original(*args,**kwargs)
    model.init_state=compatible
    return True


def run(args):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=getattr(args,'resume',False))
    receipt={'status':'STARTED','version':args.version,'precision':'official_bfloat16','input_protocol':'one_frame_predictor','human_acceptance':False}
    try:
        if not args.checkpoint.is_file() or sha(args.checkpoint)!=args.checkpoint_sha256:raise ValueError('checkpoint hash mismatch')
        revision=subprocess.check_output(['git','-C',str(args.sam_source),'rev-parse','HEAD'],text=True).strip()
        if revision!=args.code_revision:raise ValueError('code revision mismatch')
        dirty=subprocess.check_output(['git','-C',str(args.sam_source),'status','--porcelain','--untracked-files=no'],text=True)
        if dirty.strip():raise ValueError('modified upstream code is not a controlled comparison')
        import sys
        sys.path.insert(0,str(args.sam_source.resolve()))
        import torch
        from sam3.model_builder import build_sam3_predictor
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.manual_seed(0);np.random.seed(0);torch.cuda.reset_peak_memory_stats()
        loads=[];original=torch.nn.Module.load_state_dict
        def checked(module,*a,**kw):
            result=original(module,*a,**kw)
            loads.append({'class':type(module).__name__,'missing':list(result.missing_keys),'unexpected':list(result.unexpected_keys)})
            return result
        torch.nn.Module.load_state_dict=checked
        try:
            predictor=build_sam3_predictor(checkpoint_path=str(args.checkpoint.resolve()),version=args.version,compile=False,warm_up=False,use_fa3=False,use_rope_real=False,async_loading_frames=False)
        finally:torch.nn.Module.load_state_dict=original
        # Intermediate tracker loads can deliberately be partial. The final
        # owning model must have a complete match, otherwise fail closed.
        owning=[r for r in loads if r['class']==type(predictor.model).__name__]
        receipt.update(code_revision=revision,checkpoint_sha256=args.checkpoint_sha256,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),state_dict_loads=loads)
        if not owning or owning[-1]['missing'] or owning[-1]['unexpected']:raise ValueError('final model checkpoint is incomplete')
        receipt['disabled_state_offload_bridge']=bridge_disabled_state_offload(predictor.model)
        plan=json.loads(args.plan.read_text(encoding='utf-8'));size=tuple(plan['image_size'])
        if sha(args.source)!=plan['source_sha256']:raise ValueError('source binding mismatch')
        with Image.open(args.source) as source:
            if source.size!=size:raise ValueError('source dimensions mismatch')
            # PNG avoids JPEG loss. Both versions receive exactly one frame.
            frames=root/'frames';frames.mkdir(exist_ok=getattr(args,'resume',False));source.convert('RGB').save(frames/'00000.png')
        session=None
        try:
            session=predictor.handle_request({'type':'start_session','resource_path':str(frames),'offload_video_to_cpu':True})['session_id']
            state=predictor._get_session(session)['state']
            if state.get('num_frames')!=1:raise ValueError('single-frame input not supported')
            for obj in plan['objects']:
                if args.objects and obj['id'] not in args.objects:continue
                old=root/(obj['id']+'.json')
                if getattr(args,'resume',False) and old.exists():
                    cached=json.loads(old.read_text(encoding='utf-8'))
                    expected={'source_sha256':sha(args.source),'plan_sha256':sha(args.plan),'version':args.version,
                        'checkpoint_sha256':args.checkpoint_sha256,'code_revision':revision,'status':'SUCCEEDED'}
                    file=(root/cached['mask_file']).resolve()
                    if not file.is_relative_to(root) or any(cached.get(k)!=v for k,v in expected.items()) or sha(file)!=cached.get('mask_sha256'):
                        raise ValueError('resume mask cache binding mismatch')
                    continue
                started=time.monotonic();predictor.handle_request({'type':'reset_session','session_id':session})
                result=predictor.handle_request({'type':'add_prompt','session_id':session,'frame_index':0,'text':obj['prompt_en'],
                    'bounding_boxes':[normalize_box(obj['bbox'],size)],'bounding_box_labels':[1],'rel_coordinates':True})
                masks,scores=arrays(result['outputs'],size)
                file=root/(obj['id']+'.npz');np.savez_compressed(file,masks=masks,scores=scores)
                write(root/(obj['id']+'.json'),{'status':'SUCCEEDED','source_sha256':sha(args.source),'plan_sha256':sha(args.plan),
                    'image_size':list(size),'version':args.version,'code_revision':revision,'checkpoint_sha256':args.checkpoint_sha256,
                    'mask_file':file.name,'mask_sha256':sha(file),'candidate_count':len(masks),'seconds':time.monotonic()-started,
                    'precision':'official_bfloat16','input_protocol':'one_frame_predictor'})
        finally:
            if session is not None:predictor.handle_request({'type':'close_session','session_id':session})
            predictor.shutdown()
        receipt.update(status='SUCCEEDED',single_frame_supported=True,gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
    except Exception as exc:
        receipt.update(status='FAILED',error_type=type(exc).__name__,error=str(exc)[:600],single_frame_supported=False)
        write(root/'run.json',receipt);raise
    write(root/'run.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',choices=['sam3','sam3.1'],required=True)
    for name in ('source','plan','checkpoint','sam-source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--code-revision',required=True)
    p.add_argument('--objects',nargs='*');p.add_argument('--resume',action='store_true')
    run(p.parse_args())
