"""Opt-in backend stage. Production default remains off until real-image acceptance."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from .contracts import Page,Crop
from .materialize import digest,write
from .object_contracts import ObjectPlan,OBJECT_PROMPT
from .object_extract import extract

ROOT=Path(__file__).resolve().parents[3]


def route_setting(value='default'):
    route=os.environ.get('IMAGE2PPT_OBJECT_EXTRACTION','off') if value=='default' else value
    if route not in ('off','alpha','sam3','sam3.1'):raise ValueError('unsupported object extraction route')
    return route


def obtain_layers(source,directory):
    """Fresh uploads use LayerD; experiment callers inject an immutable saved pack."""
    directory=Path(directory)
    if (directory/'manifest.json').is_file():return directory
    token=os.environ.get('LAYERD_TOKEN_FILE')
    if not token:raise RuntimeError('layerd_token_file_missing')
    directory.parent.mkdir(parents=True,exist_ok=True)
    log=directory.with_suffix('.log')
    args=[sys.executable,str(ROOT/'services/layerd/client.py'),'--input',str(source),'--output',str(directory),
          '--url',os.environ.get('LAYERD_URL','http://127.0.0.1:18840'),'--token-file',token]
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run(args,stdout=stream,stderr=stream,timeout=1300,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode:raise RuntimeError('layerd_decomposition_failed')
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['source_sha256']!=digest(source):raise ValueError('layerd_upload_binding_mismatch')
    if not np.array_equal(np.array(Image.open(source).convert('RGB')),np.array(Image.open(directory/'source.png').convert('RGB'))):raise ValueError('layerd_normalized_pixels_changed')
    write(directory/'request.json',{'source_sha256':digest(source),'normalized_sha256':digest(directory/'source.png')})
    return directory


def object_request(source,layerdir,page,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    meta=json.loads((Path(layerdir)/'manifest.json').read_text(encoding='utf-8'))
    w,h=page.image_size;yy,xx=np.indices((h,w));checker=np.repeat(np.where(((xx//20+yy//20)%2)[...,None],215,245).astype('uint8'),3,2)
    images=[Path(source)];layers=[]
    for row in meta['layers']:
        rgba=Image.open(Path(layerdir)/row['file']).convert('RGBA')
        preview=Image.fromarray(checker).convert('RGBA');preview.alpha_composite(rgba)
        path=directory/(row['file'].replace('.png','-checker.png'));preview.convert('RGB').save(path);images.append(path)
        layers.append({k:row[k] for k in ('file','z','sha256')})
    return {'source_sha256':digest(source),'image_size':page.image_size,'layers':layers,
            'image_order':['source']+[r['file'] for r in layers],
            'protected_texts':[{'id':t.id,'bbox':t.bbox,'text':t.text} for t in page.texts],
            'baseline_graphics':[n.model_dump() for n in [*page.crops,*page.shapes]]},images


def infer_sam(source,plan,directory,route,object_ids):
    config=os.environ.get('IMAGE2PPT_SAM_CONFIG')
    if not config:return None
    cfg=json.loads(Path(config).read_text(encoding='utf-8'));model=cfg['models'][route]
    args=[cfg['python'],str(ROOT/'services/segmentation/single_frame.py'),'--version',route,
        '--source',str(source),'--plan',str(plan),'--output',str(directory),'--sam-source',cfg['source'],
        '--code-revision',cfg['code_revision'],'--checkpoint',model['checkpoint'],'--checkpoint-sha256',model['sha256'],
        '--objects',*object_ids]
    with Path(directory).with_suffix('.log').open('w',encoding='utf-8') as stream:
        try:subprocess.run(args,stdout=stream,stderr=stream,timeout=1800,check=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except (subprocess.SubprocessError,OSError):return None
    return directory


def effective_page(page,irfile):
    ir=json.loads(Path(irfile).read_text(encoding='utf-8'));payload=page.model_dump();ids={n['id'] for n in ir['nodes']}
    payload['shapes']=[s for s in payload['shapes'] if s['id'] in ids]
    descriptions={a['id']:a.get('description','source graphic') for a in ir['assets']}
    payload['crops']=[{'id':n['id'],'bbox':n['bbox'],'z':n['z'],'description':descriptions.get(n['asset_id'],'source graphic')[:300]} for n in ir['nodes'] if n['type']=='image']
    # VisualPage's public crop validator correctly assumes an opaque source
    # rectangle. Extracted PNGs are different: their transparent bounding-box
    # corners may intersect text, but extract() has already rejected ANY actual
    # alpha/text intersection. Do not weaken the model-facing crop validator or
    # allow models to self-declare transparency to bypass it.
    return page.model_copy(update={
        'shapes':[s for s in page.shapes if s.id in ids],
        'crops':[Crop.model_validate(c) for c in payload['crops']]})


def process(source,page,baseline_ir,directory,route,ask,*,layerdir=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    layerdir=Path(layerdir) if layerdir else obtain_layers(source,directory/'layers')
    # Source may be reencoded by LayerD; normalized RGB equality is already gated.
    normalized=layerdir/'source.png'
    context,images=object_request(normalized,layerdir,page,directory/'previews')
    raw=ask(context,images,ObjectPlan)
    write(directory/'object-plan-raw.json',raw)
    plan=ObjectPlan.model_validate(raw)
    if plan.source_sha256!=digest(normalized) or plan.image_size!=page.image_size:raise ValueError('object plan source binding mismatch')
    planfile=directory/'object-plan.json';write(planfile,plan.model_dump())
    ir,report=extract(planfile,layerdir,baseline_ir,directory/'alpha',route='alpha')
    difficult=[r['id'] for r in report['objects'] if 'alpha_ambiguous_or_empty' in r['reasons']]
    if route!='alpha' and difficult:
        samdir=infer_sam(normalized,planfile,directory/'sam',route,difficult)
        ir,report=extract(planfile,layerdir,baseline_ir,directory/'applied',route=route,sam_dir=samdir)
    report['requested_route']=route
    proposal=effective_page(page,ir)
    # Any object fallback is an unresolved issue, never an automatic visual pass.
    if report['fallback']:
        proposal=proposal.model_copy(update={'limitations':[ *proposal.limitations, f"Object extraction retained {report['fallback']} original regions; incomplete independent extraction."]})
    return ir,proposal,report
