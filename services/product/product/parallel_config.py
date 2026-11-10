"""Own-project service paths only; model API keys are never stored here."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]

def configure():
    file=ROOT/'.runtime/parallel-services.json'
    if not file.exists():return
    cfg=json.loads(file.read_text(encoding='utf-8'))
    for branch,prefix in [('layerd','LAYERD'),('sam','IMAGE2PPT_SAM')]:
        data=cfg.get(branch,{})
        for field,suffix in [('url','URL'),('token_file','TOKEN_FILE')]:
            if data.get(field):os.environ.setdefault(prefix+'_'+suffix,data[field])
    if cfg.get('ocr_manifest'):os.environ.setdefault('IMAGE2PPT_OCR_MANIFEST',cfg['ocr_manifest'])
    if cfg.get('layerd',{}).get('model_binding'):os.environ.setdefault('LAYERD_MODEL_BINDING',cfg['layerd']['model_binding'])
    if cfg.get('ocr_python_path'):
        path=Path(cfg['ocr_python_path']).resolve()
        if not path.is_relative_to(ROOT/'.runtime') or not path.is_dir():raise ValueError('OCR dependencies must be in this workspace runtime')
        sys.path.insert(0,str(path));os.environ['PYTHONPATH']=os.pathsep.join(dict.fromkeys([str(path),*filter(None,os.environ.get('PYTHONPATH','').split(os.pathsep))]))
