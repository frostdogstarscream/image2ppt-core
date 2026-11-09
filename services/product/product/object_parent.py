"""Explicit ledger adapter when replaying a plan against a newer parent scene."""
import copy
import json
from pathlib import Path
import shutil
from .materialize import digest,write
from .object_contracts import read_plan


def adapt_owners(plan,ir,history):
    present={n['id'] for n in ir['nodes']};payload=plan.model_dump();changes=[]
    for obj in payload['objects']:
        original=set(obj['replaces']);mapped=[]
        for owner in obj['replaces']:
            if owner in present:mapped.append(owner);continue
            matches=[r for r in history['objects'] if r['status']=='EXTRACTED' and owner in r['replaces']]
            if len(matches)!=1:raise ValueError('ambiguous historical owner')
            previous=matches[0];node='extracted-'+previous['id']
            if node not in present or not set(previous['replaces'])<=original:raise ValueError('cannot split historical complete owner')
            if node not in mapped:mapped.append(node)
        if mapped!=obj['replaces']:
            changes.append({'id':obj['id'],'from':obj['replaces'],'to':mapped})
            obj['replaces']=mapped;obj['required_owners']=mapped
    adapted,_=read_plan(payload)
    return adapted,changes


def rebind_sam(oldplan,newplan,source,destination):
    """Ownership-only receipt adapter; inference inputs must be exactly equal."""
    source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    old,_=read_plan(json.loads(Path(oldplan).read_text(encoding='utf-8')))
    new,_=read_plan(json.loads(Path(newplan).read_text(encoding='utf-8')))
    if old.source_sha256!=new.source_sha256 or old.image_size!=new.image_size:raise ValueError('SAM rebind source changed')
    prior={o.id:o for o in old.objects}
    for obj in new.objects:
        receipt=source/(obj.id+'.json')
        if not receipt.exists():continue
        previous=prior[obj.id]
        if previous.bbox!=obj.bbox or previous.prompt_en!=obj.prompt_en:raise ValueError('SAM rebind changed inference inputs')
        raw=json.loads(receipt.read_text(encoding='utf-8'))
        if raw['plan_sha256']!=digest(oldplan):raise ValueError('SAM original receipt unbound')
        mask=(source/raw['mask_file']).resolve()
        if not mask.is_relative_to(source.resolve()) or digest(mask)!=raw['mask_sha256']:raise ValueError('SAM original mask changed')
        shutil.copyfile(receipt,destination/('raw-'+obj.id+'.json'));shutil.copyfile(mask,destination/mask.name)
        write(destination/receipt.name,{**raw,'plan_sha256':digest(newplan),'adapter':'OwnerOnlySamReceipt/1',
            'raw_receipt_sha256':digest(receipt),'original_plan_sha256':digest(oldplan)})
