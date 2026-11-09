"""Write readiness only after real CUDA computation and SAM import succeed."""
import argparse,importlib.metadata as metadata,json,subprocess,sys
from pathlib import Path

def main(root):
    sys.path.insert(0,str(root/'sam3-official'))
    import torch
    import triton
    from sam3.model_builder import build_sam3_predictor
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    value=torch.ones((32,32),device='cuda')
    if not torch.allclose(value@value,torch.full_like(value,32)):raise RuntimeError('CUDA computation mismatch')
    torch.cuda.synchronize()
    revision=subprocess.check_output(['git','-C',str(root/'sam3-official'),'rev-parse','HEAD'],text=True).strip()
    if revision!='f1647e8d0c1d280991f4d1b7d974c2704537fe66':raise RuntimeError('Unexpected source revision')
    result={'status':'CUDA_AND_IMPORT_VERIFIED','torch':torch.__version__,'triton':triton.__version__,
        'gpu':torch.cuda.get_device_name(),'source_revision':revision,'human_acceptance':False}
    (root/'environment-ready.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root.resolve())
