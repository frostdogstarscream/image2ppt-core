"""Source-owner residual checks against actual saved-and-reopened deletions.

This is separate from Office's outside-pixel gate and from semantic Go review.
The source-derived ownership/backing receipt is mandatory, never a PNG mask.
"""
import json
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile
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


def _visible_text_backing(image, mask, source_ink, noise_limit):
    """Measure the saved view against exposed, unowned neighbours of the former glyphs."""
    import cv2

    ring = cv2.dilate(mask.astype('uint8'), np.ones((3, 3), 'uint8')).astype(bool) & ~mask
    if not ring.any() or (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()):
        return None
    yy, xx = np.indices(mask.shape)
    basis = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    fit, _, rank, _ = np.linalg.lstsq(basis[ring], image[ring], rcond=None)
    error = np.abs(image - basis @ fit).max(2)
    # A textured/partially occluded ring cannot certify the erased field. The tiny
    # allowance on an otherwise smooth reference covers RGB raster quantization.
    if rank != 3 or error[ring].max() > 2:
        return None
    excess = mask & (error > 2)
    evidence = {'ring_pixels':int(ring.sum()), 'visible_peak_rgb':round(float(error[mask].max()), 3),
                'off_surface_pixels':int(excess.sum())}
    if not excess.any():
        return evidence
    if error[excess].max() > noise_limit:
        return None
    # Scale the test to the height of the original ink. A tiny but *complete*
    # source mark (decimal point, accent, etc.) is never an isolated speck.
    ink_rows = np.where(source_ink.any(axis=1))[0]
    if not ink_rows.size:
        return None
    glyph_height = int(ink_rows[-1] - ink_rows[0] + 1)
    _, _, components, _ = cv2.connectedComponentsWithStats(excess.astype('uint8'), connectivity=8)
    if components[1:, 2:4].max() > glyph_height * .16:
        return None
    density = cv2.boxFilter(excess.astype('float32'), -1, (glyph_height, glyph_height),
                            normalize=False, borderType=cv2.BORDER_CONSTANT)
    if density.max() > .02 * glyph_height ** 2:
        return None
    count, labels = cv2.connectedComponents(source_ink.astype('uint8'), connectivity=8)
    ink_area = np.bincount(labels.ravel(), minlength=count)[1:]
    signal = np.bincount(labels[excess], minlength=count)[1:]
    return None if (signal >= .35 * ink_area).any() else evidence


def _verify_text_specks(deleted, visible_plate, mask, source_ink, noise_limit, neighbours):
    import cv2
    if np.any(cv2.dilate(mask.astype('uint8'), np.ones((3, 3), 'uint8')) & neighbours):
        return None
    # The actual deletion and the PPTX's named plate must each be clean. A solid
    # native banner can otherwise hide old text baked into a lower picture.
    office = _visible_text_backing(deleted, mask, source_ink, noise_limit)
    if office is not None:
        plate = _visible_text_backing(visible_plate, mask, source_ink, noise_limit)
        if plate is not None:
            return {'after_delete':office, 'named_plate':plate}
    return None


def inspect_residuals(irfile,office_dir,page,output,*,owned_group_ids=(), text_edit_pptx=None):
    irfile=Path(irfile);scene=json.loads((irfile.parent/'scene.json').read_text(encoding='utf-8'))
    ir=json.loads(irfile.read_text(encoding='utf-8')) if irfile.is_file() else {'nodes':[]}
    node_z={n['id']:int(n.get('z',0)) for n in ir.get('nodes',[]) if isinstance(n,dict) and 'id' in n}
    node_bbox={n['id']:n.get('bbox') for n in ir.get('nodes',[]) if isinstance(n,dict) and 'id' in n}
    for group in ir.get('groups', []):
        if group['id'] in owned_group_ids:
            # A flat group occupies its members' contiguous z interval. A later
            # independent shape can occlude it, an earlier one cannot.
            node_z[group['id']] = min(node_z[member] for member in group['member_ids'])
    records=list(scene['selected'].values());source=np.array(Image.open(irfile.parent/'source.png').convert('RGB'));proofs={};results=[]
    rb=scene.get('raster_backing') or {}
    visible_plate = None
    plate_sha = None
    checked_text_deletes = set()
    receipt = Path(office_dir)/'report.json'
    if text_edit_pptx and Path(text_edit_pptx).is_file() and receipt.is_file():
        office = json.loads(receipt.read_text('utf8'))
        if (office.get('status') == 'PASS_APP_AUTOMATED' and office.get('embedded_alpha_and_visible_rgb_checked')
                and office.get('pptx_sha256') == digest(text_edit_pptx)):
            checked_text_deletes = {r['id'] for r in office.get('records', [])
                                    if r.get('action') == 'delete' and r.get('page') == page
                                    and r.get('saved_reopened') and r.get('outside_pixels') == 0}
    if (checked_text_deletes and ir.get('source_sha256') == scene['source_sha256']
            and digest(irfile.parent/'source.png') == scene['source_sha256']):
        ref = next((a for a in ir.get('assets', []) if a['id'] == 'source-plate'), None)
        node = next((n for n in ir.get('nodes', []) if n.get('asset_id') == 'source-plate'), None)
        if ref and node and node.get('type') == 'image' and node.get('bbox') == [0, 0, source.shape[1], source.shape[0]]:
            from .parallel_office import verify_pictures
            # Bind the current IR to the actual pictures in the saved presentation.
            try:
                verify_pictures(text_edit_pptx, [irfile], first_page=page)
            except (BadZipFile, KeyError, OSError, ParseError, ValueError):
                checked_text_deletes.clear()
            file = (irfile.parent / ref['path']).resolve()
            if (checked_text_deletes and file.is_relative_to(irfile.parent.resolve()) and file.is_file()
                    and digest(file) == ref.get('sha256')):
                with Image.open(file) as picture:
                    has_alpha = 'A' in picture.getbands() or 'transparency' in picture.info
                    if not has_alpha or picture.convert('RGBA').getchannel('A').getextrema() == (255, 255):
                        visible_plate = np.asarray(picture.convert('RGB'))
                        plate_sha = ref['sha256']
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
    # Each independently deleted footer field owns only its part of the OCR line.
    # The source-derived gutters partition every original mask pixel exactly once.
    from .recovery.marked_fields import footer_fields
    for record in list(records):
        parent=record['id']
        if parent+'-field-2' not in node_bbox or parent not in proofs:
            continue
        original=record.get('source_text_definition',{})
        parts=footer_fields(source,original.get('text',''),original.get('bbox',record['bbox']))
        if not parts:
            continue
        proof=proofs[parent]
        cuts=[(a[1][0]+a[1][2]+b[1][0])//2 for a,b in zip(parts,parts[1:])]
        for index,(left,right) in enumerate(zip([0,*cuts],[*cuts,source.shape[1]])):
            field_id=parent if index==0 else f'{parent}-field-{index+1}'
            mask=proof['mask'].copy()
            mask[:,:left]=False
            mask[:,right:]=False
            proofs[field_id]={**proof,'mask':mask}
            if index:
                records.append({**record,'id':field_id,'bbox':node_bbox[field_id]})
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
            if other!=identifier and other != record.get('payload', {}).get('background_parent_id'):
                mask&=~proof['mask'].astype(bool)
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
                    if other_rec.get('retained_card_artwork'):
                        ref = other_rec['asset_reference']
                        png = (irfile.parent / ref['path']).resolve()
                        if not png.is_relative_to(irfile.parent.resolve()) or digest(png) != ref['sha256']:
                            raise ValueError('card artwork binding mismatch')
                        with Image.open(png) as image:
                            visible = np.asarray(image.convert('RGBA'))[...,3] != 0
                        mask[ty:ty+th,tx:tx+tw] &= ~visible
                    else:
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
        # Flag old signal that is closer to the original than to its estimated backing.
        # Only weak differences can be reconsidered from the actual saved view below.
        residual=local&(original>ink_contrast)&(distance>plate_tol)&(np.abs(got-src).max(2)<distance)
        eff_tol=np.where(surface_under_local,surf_tol,plate_tol)
        minor_dev=local&(~residual)&(distance>plate_tol)&(distance<=eff_tol)
        unknown=local&(distance>eff_tol)&~residual
        flagged = residual | unknown
        if (identifier in checked_text_deletes and record.get('kind') == 'text' and flagged.any()
                and distance[flagged].max() <= ink_contrast and visible_plate is not None
                and visible_plate.shape == source.shape and _verified_single_line(record)
                and np.array_equal(local, data['mask'][y:y+h,x:x+w])):
            # Include real context beyond the restore bbox; do not extrapolate
            # across the slide boundary or grant relief for missing evidence.
            left, top = max(0, x - 1), max(0, y - 1)
            area = np.s_[top:y+h+1, left:x+w+1]
            field_mask = np.zeros(after[area].shape[:2], bool)
            inner = np.s_[y-top:y-top+h, x-left:x-left+w]
            field_mask[inner] = local
            strong = np.zeros_like(field_mask)
            strong[inner] = local & (original > max(ink_contrast, np.percentile(original[local], 90) * .4))
            check = None
            if strong.any():
                neighbours = np.zeros_like(field_mask)
                for other, proof in proofs.items():
                    if other != identifier:
                        neighbours |= proof['mask'][area].astype(bool)
                check = _verify_text_specks(after[area], visible_plate[area], field_mask, strong,
                                             ink_contrast, neighbours)
            if check is not None:
                base.update(text_backing_check=check, prior_residual_pixels=int(residual.sum()),
                            prior_unexplained_pixels=int(unknown.sum()), visible_plate_sha256=plate_sha)
                minor_dev |= flagged
                residual[:] = False
                unknown[:] = False
        status='FAIL' if residual.any() else 'UNKNOWN' if unknown.any() else 'PASS'
        results.append({**base,'residual_check_status':status,'residual_pixels_confirmed':int(residual.sum()),
            'minor_deviation_pixels':int(minor_dev.sum()),
            'unexplained_pixels':int(unknown.sum()),'checked_owner_pixels':int(local.sum()),'delete_sha256':digest(deleted),
            'source_sha256':scene['source_sha256'],'tolerance_rgb':plate_tol,'surface_tolerance_rgb':surf_tol,
            'scope':'proven source-owner pixels excluding other staged owners; semantic completeness needs independent review'})
    write(output,{'schema':'SourceOwnerResiduals/1','records':results,'human_acceptance':False})
    return results


def _verified_single_line(record):
    text = (record.get('source_text_definition') or {}).get('text')
    return isinstance(text, str) and bool(text.strip()) and len(text.splitlines()) == 1
