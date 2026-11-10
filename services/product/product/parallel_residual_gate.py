"""Source-owner residual checks against actual saved-and-reopened deletions.

This is separate from Office's outside-pixel gate and from semantic Go review.
The source-derived ownership/backing receipt is mandatory, never a PNG mask.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .materialize import digest,write


def inspect_residuals(irfile,office_dir,page,output):
    irfile=Path(irfile);scene=json.loads((irfile.parent/'scene.json').read_text(encoding='utf-8'))
    records=list(scene['selected'].values());source=np.array(Image.open(irfile.parent/'source.png').convert('RGB'));proofs={};results=[]
    for record in records:
        ref=record.get('owner_evidence')
        if not ref:continue
        file=(irfile.parent/ref['file']).resolve()
        if not file.is_relative_to(irfile.parent.resolve()) or digest(file)!=ref['sha256']:raise ValueError('source owner evidence binding mismatch')
        with np.load(file,allow_pickle=False) as data:
            proofs[record['id']]={k:data[k] for k in data.files}
    for record in records:
        identifier=record['id'];deleted=Path(office_dir)/f'{page:02d}-{identifier}-delete.png'
        base={'id':identifier,'residual_check_status':'UNKNOWN','residual_pixels_confirmed':None,'reference_sha256':record.get('owner_evidence',{}).get('sha256')}
        if identifier not in proofs or not deleted.exists():results.append({**base,'reason':'missing source-owner or actual-delete evidence'});continue
        data=proofs[identifier];mask=data['mask'].astype(bool).copy();x,y,w,h=map(int,data['restore_bbox'])
        for other,proof in proofs.items():
            if other!=identifier:mask&=~proof['mask'].astype(bool)
        local=mask[y:y+h,x:x+w];after=np.array(Image.open(deleted).convert('RGB'))
        if after.shape!=source.shape or not local.any():results.append({**base,'reason':'unverifiable size or occluded owner'});continue
        bg=data['backing_patch'].astype(float);src=source[y:y+h,x:x+w].astype(float);got=after[y:y+h,x:x+w].astype(float)
        distance=np.abs(got-bg).max(2);original=np.abs(src-bg).max(2)
        # Confirm a residual only where the independently observed old signal
        # remains closer to source than backing. Other discrepancies are unknown.
        residual=local&(original>12)&(distance>8)&(np.abs(got-src).max(2)<distance)
        unknown=local&(distance>8)&~residual
        status='FAIL' if residual.any() else 'UNKNOWN' if unknown.any() else 'PASS'
        results.append({**base,'residual_check_status':status,'residual_pixels_confirmed':int(residual.sum()),
            'unexplained_pixels':int(unknown.sum()),'checked_owner_pixels':int(local.sum()),'delete_sha256':digest(deleted),
            'source_sha256':scene['source_sha256'],'tolerance_rgb':8,'scope':'proven source-owner pixels excluding other staged owners; semantic completeness needs independent review'})
    write(output,{'schema':'SourceOwnerResiduals/1','records':results,'human_acceptance':False})
    return results
