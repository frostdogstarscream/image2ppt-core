"""Source-owner residual checks against actual saved-and-reopened deletions.

This is separate from Office's outside-pixel gate and from semantic Go review.
The source-derived ownership/backing receipt is mandatory, never a PNG mask.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .materialize import digest,write
from .tolerance_config import load_tolerance_config


def residual_accepted(row):
    """PASS, or an explicitly MOVE_UNSAFE region crop that erased nothing."""
    return row['residual_check_status']=='PASS' or (
        row['residual_check_status']=='NOT_APPLICABLE'
        and row.get('region_crop') is True
        and row.get('move_safety')=='MOVE_UNSAFE')


def inspect_residuals(irfile,office_dir,page,output):
    irfile=Path(irfile);scene=json.loads((irfile.parent/'scene.json').read_text(encoding='utf-8'))
    ir=json.loads(irfile.read_text(encoding='utf-8')) if irfile.is_file() else {'nodes':[]}
    node_z={n['id']:int(n.get('z',0)) for n in ir.get('nodes',[]) if isinstance(n,dict) and 'id' in n}
    node_bbox={n['id']:n.get('bbox') for n in ir.get('nodes',[]) if isinstance(n,dict) and 'id' in n}
    records=list(scene['selected'].values());source=np.array(Image.open(irfile.parent/'source.png').convert('RGB'));proofs={};results=[]
    rb=scene.get('raster_backing') or {}
    plate=None
    if rb.get('path'):
        plate_file=(irfile.parent/rb['path']).resolve()
        if plate_file.is_relative_to(irfile.parent.resolve()) and plate_file.is_file():
            if not rb.get('sha256') or digest(plate_file)==rb['sha256']:
                plate=np.array(Image.open(plate_file).convert('RGB'))
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
        if record.get('region_crop') and record.get('move_safety')=='MOVE_UNSAFE' and not record.get('owner_evidence'):
            bbox=record.get('bbox') or node_bbox.get(identifier)
            plate_ok=plate is not None and plate.shape==source.shape and bbox and len(bbox)==4
            if plate_ok:
                bx,by,bw,bh=[round(v) for v in bbox]
                bx0,by0=max(0,bx),max(0,by)
                bx1,by1=min(source.shape[1],bx+bw),min(source.shape[0],by+bh)
                check_mask=np.zeros(source.shape[:2],bool)
                if bx1>bx0 and by1>by0:
                    check_mask[by0:by1,bx0:bx1]=True
                z_self=node_z.get(identifier,0)
                for other_rec in records:
                    oid=other_rec['id']
                    if oid==identifier or oid not in proofs:continue
                    z_other=node_z.get(oid,10**9 if other_rec.get('kind')=='text' else 0)
                    if z_other>z_self:
                        check_mask&=~proofs[oid]['mask'].astype(bool)
                plate_ok=(not check_mask.any()) or np.array_equal(plate[check_mask],source[check_mask])
            if not plate_ok:
                results.append({**base,'residual_check_status':'FAIL','region_crop':True,'move_safety':'MOVE_UNSAFE',
                    'reason':'MOVE_UNSAFE crop underlying source-plate pixels were modified'});continue
            # Nothing was erased: the plate keeps identical pixels beneath this copy.
            results.append({**base,'residual_check_status':'NOT_APPLICABLE','region_crop':True,'move_safety':'MOVE_UNSAFE',
                'reason':'opaque region crop over unchanged source plate; deletion reveals identical source pixels by design'});continue
        if identifier not in proofs or not deleted.exists():results.append({**base,'reason':'missing source-owner or actual-delete evidence'});continue
        data=proofs[identifier];mask=data['mask'].astype(bool).copy();x,y,w,h=map(int,data['restore_bbox'])
        for other,proof in proofs.items():
            if other!=identifier:mask&=~proof['mask'].astype(bool)
        if record.get('kind') in ('asset','surface'):
            z_self=node_z.get(identifier,0)
            for other_rec in records:
                oid=other_rec['id']
                if oid==identifier:continue
                okind=other_rec.get('kind')
                z_other=node_z.get(oid,10**9 if okind=='text' else 0)
                if z_other<=z_self:continue
                obox=node_bbox.get(oid) or other_rec.get('bbox') or (other_rec.get('payload') or {}).get('bbox')
                if not obox or len(obox)!=4:continue
                tx,ty,tw,th=[round(v) for v in obox]
                tx0,ty0=max(0,tx-2),max(0,ty-2)
                tx1,ty1=min(source.shape[1],tx+tw+2),min(source.shape[0],ty+th+2)
                if tx1>tx0 and ty1>ty0 and mask[ty0:ty1,tx0:tx1].any():
                    # Higher-z sibling objects remain rendered above this object in PowerPoint when it is deleted.
                    mask[ty0:ty1,tx0:tx1]=False
        local=mask[y:y+h,x:x+w];after=np.array(Image.open(deleted).convert('RGB'))
        if after.shape!=source.shape or not local.any():results.append({**base,'reason':'unverifiable size or occluded owner'});continue
        tol_cfg=load_tolerance_config().get('residual_gate',{})
        plate_tol=float(tol_cfg.get('plate_backing_rgb',8.0))
        surf_tol=float(tol_cfg.get('native_surface_backing_rgb',26.0))
        ink_contrast=float(tol_cfg.get('source_ink_contrast_rgb',12.0))
        z_self=node_z.get(identifier,10**9 if record.get('kind')=='text' else 0)
        surface_under_local=np.zeros((h,w),bool)
        for other_rec in records:
            oid=other_rec['id']
            if oid==identifier or other_rec.get('kind')!='surface':continue
            z_other=node_z.get(oid,0)
            if z_other>=z_self:continue
            sbox=node_bbox.get(oid) or other_rec.get('bbox') or (other_rec.get('payload') or {}).get('bbox')
            if not sbox or len(sbox)!=4:continue
            sx0=max(0,round(sbox[0])-x);sy0=max(0,round(sbox[1])-y)
            sx1=min(w,round(sbox[0]+sbox[2])-x);sy1=min(h,round(sbox[1]+sbox[3])-y)
            if sx1>sx0 and sy1>sy0:
                surface_under_local[sy0:sy1,sx0:sx1]=True
        bg=data['backing_patch'].astype(float);src=source[y:y+h,x:x+w].astype(float);got=after[y:y+h,x:x+w].astype(float)
        distance=np.abs(got-bg).max(2);original=np.abs(src-bg).max(2)
        # Confirm a residual only where the independently observed old signal
        # remains closer to source than backing. Hard residual is never relaxed.
        residual=local&(original>ink_contrast)&(distance>plate_tol)&(np.abs(got-src).max(2)<distance)
        eff_tol=np.where(surface_under_local,surf_tol,plate_tol)
        minor_dev=local&(~residual)&(distance>plate_tol)&(distance<=eff_tol)
        unknown=local&(distance>eff_tol)&~residual
        status='FAIL' if residual.any() else 'UNKNOWN' if unknown.any() else 'PASS'
        results.append({**base,'residual_check_status':status,'residual_pixels_confirmed':int(residual.sum()),
            'minor_deviation_pixels':int(minor_dev.sum()),
            'unexplained_pixels':int(unknown.sum()),'checked_owner_pixels':int(local.sum()),'delete_sha256':digest(deleted),
            'source_sha256':scene['source_sha256'],'tolerance_rgb':plate_tol,'surface_tolerance_rgb':surf_tol,
            'scope':'proven source-owner pixels excluding other staged owners; semantic completeness needs independent review'})
    write(output,{'schema':'SourceOwnerResiduals/1','records':results,'human_acceptance':False})
    return results
