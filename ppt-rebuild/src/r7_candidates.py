"""Bounded r7 repairs of reviewed missing elements, preserving the r6 subscene."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

from object_resolver import registry_from_ir, mask_ref, resolve, authorize_repair
from relation_validator import digest, rgb, checked_mask, blue_mask, silhouette


REMOVED_SEPARATORS = {'s08-feedback-arrow-1': 'separator-h-025',
                      's08-feedback-arrow-2': 'separator-h-026',
                      's08-feedback-arrow-3': 'separator-h-027'}


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n', encoding='utf-8')


def color(value):
    return ''.join(f'{int(v):02X}' for v in np.clip(np.rint(value),0,255))


def _components(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'))
    return [(labels==i, list(map(int,stats[i,:4]))) for i in range(1,n) if stats[i,4]>=40]


def _plane(crop, sample):
    yy,xx=np.indices(sample.shape)
    design=np.stack([np.ones_like(xx),xx/max(1,xx.shape[1]-1),yy/max(1,yy.shape[0]-1)],axis=2)
    selected=sample.copy()
    for _ in range(3):
        if selected.sum()<8: raise ValueError('Insufficient independent background/color samples')
        coefficients=np.linalg.lstsq(design[selected],crop[selected].astype(float),rcond=None)[0]
        fitted=np.clip(design@coefficients,0,255)
        errors=np.abs(fitted-crop).max(axis=2)
        selected=sample & (errors<=max(5,float(np.percentile(errors[selected],85))))
    residual=np.abs(fitted-crop)[selected]
    return fitted,{'sample_pixels':int(selected.sum()),'mae':float(residual.mean()),
                   'p95':float(np.percentile(residual,95))}


def _label_rgba(crop, ink):
    """Fill every source letter hole using blue-only fitted color, never source white."""
    output=np.zeros((*ink.shape,4),np.uint8)
    records=[]
    for component, box in _components(silhouette(ink)):
        samples=cv2.erode(component.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)&ink
        fitted, quality=_plane(crop,samples)
        output[component,:3]=np.rint(fitted[component]).astype(np.uint8)
        output[component,3]=255
        records.append({'bbox_xywh':box,'fit':quality,'method':'blue_only_linear_plane_letters_excluded'})
    return output, records


def _stroke_rgba(crop, ink):
    """Unblend reviewed strokes against an independently fitted surrounding plate."""
    expanded=cv2.dilate(ink.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)
    background,quality=_plane(crop,~cv2.dilate(ink.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool))
    src=crop.astype(float);delta=src-background
    minimum=np.where(delta>=0,delta/np.maximum(255-background,1),-delta/np.maximum(background,1)).max(axis=2)
    alpha=np.where(expanded,np.clip(minimum+.02,1/255,1),0)
    foreground=np.clip(background+delta/np.maximum(alpha[:,:,None],1/255),0,255)
    rgba=np.dstack((np.rint(foreground).astype('uint8'),np.rint(alpha*255).astype('uint8')))
    rgba[~expanded]=0
    return rgba,[{'method':'source_stroke_unblend_local_plane','fit':quality}]


def build_candidate(irfile, spec, output, *, raster_ids=()):
    """Create a separate proposed revision; no input file or base image is altered."""
    irfile=Path(irfile).resolve();output=Path(output).resolve()
    if (output/'slideir.json').exists(): raise ValueError('Candidate revision already exists')
    original=json.loads(irfile.read_text(encoding='utf-8'))
    ir=copy.deepcopy(original); source_path=(irfile.parent/ir['source_image']).resolve()
    if digest(source_path)!=spec['source_sha256']: raise ValueError('Candidate source hash mismatch')
    if ir['canvas']['source_px']!=spec['canvas_px']: raise ValueError('Candidate canvas mismatch')
    unknown=set(raster_ids)-{e['id'] for e in spec['elements']}
    if unknown: raise ValueError('Unknown raster fallback element')
    output.mkdir(parents=True,exist_ok=True)
    for relative in [ir['source_image'],*[a['path'] for a in ir['assets']]]:
        src=(irfile.parent/relative).resolve();dest=(output/relative).resolve()
        if not src.is_relative_to(irfile.parent) or not dest.is_relative_to(output):
            raise ValueError('Candidate asset reference escapes its revision')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    size=spec['canvas_px'];source=rgb(source_path)
    old_nodes={n['id']:n for n in original['nodes']}
    removed=[];additions=[];masks={};raster_records=[]
    authorized=np.zeros((size[1],size[0]),bool)
    for element in spec['elements']:
        identifier=element['id'];x,y,right,bottom=element['bbox_xyxy']
        ink=checked_mask(element['masks']['source_ink'])
        footprint=checked_mask(element['masks']['silhouette'])
        crop=source[y:bottom,x:right]
        prior_id=REMOVED_SEPARATORS.get(identifier)
        migrated=[]
        if prior_id:
            if prior_id not in old_nodes: raise ValueError('Reviewed baseline separator is absent')
            removed.append(prior_id);migrated=old_nodes[prior_id].get('evidence_ids',[])
        localrepair=cv2.dilate(footprint.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)
        repair=np.zeros_like(authorized);repair[y:bottom,x:right]=localrepair
        if prior_id:
            ox,oy,ow,oh=old_nodes[prior_id]['bbox']
            repair[max(0,int(oy)-2):min(size[1],int(np.ceil(oy+oh))+2),max(0,int(ox)-2):min(size[0],int(np.ceil(ox+ow))+2)]=True
        authorized|=repair
        created=[]
        if identifier in raster_ids:
            rgba,records=(_label_rgba(crop,ink) if element['kind']=='label' else _stroke_rgba(crop,ink))
            relative='assets/'+identifier+'.png';Image.fromarray(rgba).save(output/relative)
            evid='ev-'+identifier
            ir['evidence'].append({'id':evid,'kind':'image','bbox':[x,y,right-x,bottom-y],
                                   'description':'Reviewed source element restored without embedded source text'})
            ir['assets'].append({'id':identifier,'path':relative,'purpose':'module','sha256':digest(output/relative),
                                 'contains_evidence_ids':[evid,*migrated]})
            created=[{'id':identifier,'type':'image','bbox':[x,y,right-x,bottom-y],
                      'asset_id':identifier,'evidence_ids':[],'z':0}]
            raster_records.append({'id':identifier,'records':records,'source_letters_copied':False})
        else:
            components=_components(footprint)
            for index,(component,box) in enumerate(components):
                lx,ly,w,h=box;node_id=identifier if len(components)==1 else identifier+'-'+str(index+1)
                selected=ink&component
                geometry='parallelogram' if element['kind']=='label' else ('rightArrow' if element['kind']=='arrow' else 'chevron')
                evid='ev-'+node_id
                ir['evidence'].append({'id':evid,'kind':'shape','bbox':[x+lx,y+ly,w,h],
                                       'description':'Reviewed missing '+element['kind']+' source contour'})
                created.append({'id':node_id,'type':'shape','bbox':[x+lx,y+ly,w,h],
                                'geometry':geometry,'fill':color(np.median(crop[selected],axis=0)),
                                'evidence_ids':[evid,*migrated] if index==0 else [evid],'z':0})
        for node in created:
            additions.append(node)
            visible=np.zeros_like(authorized);visible[y:bottom,x:right]=ink
            if node['type']=='shape':
                nx,ny,nw,nh=node['bbox'];part=np.zeros_like(authorized)
                part[ny:ny+nh,nx:nx+nw]=True;visible &= part
            masks[node['id']]={'visible':visible,'repair':repair,'inspection':repair}
    # Existing node order is stable; insert new backgrounds before text. Native
    # text retains its source fonts and is never copied into the repair assets.
    retained=[copy.deepcopy(n) for n in original['nodes'] if n['id'] not in removed]
    insertion=next((i for i,n in enumerate(retained) if n['type']=='text'),len(retained))
    ir['nodes']=retained[:insertion]+additions+retained[insertion:]
    for z,node in enumerate(ir['nodes']): node['z']=z
    for group in ir.get('groups',[]):
        if any(member in removed for member in group['member_ids']):
            raise ValueError('Reviewed separator unexpectedly belongs to an editing group')
    ir['review_status']='needs_review'
    ir.setdefault('review_notes',[]).append('r7 bounded candidate: frozen source relations; all native text retained')
    target=output/'slideir.json';write(target,ir)
    registry=registry_from_ir(ir,source_sha256=digest(source_path))
    protection=~authorized
    maskdir=output/'masks';maskdir.mkdir(exist_ok=True)
    protection_path=maskdir/'outside-protection.png';Image.fromarray(protection.astype('uint8')*255).save(protection_path)
    for obj in registry['objects']:
        if obj['id'] not in masks:continue
        obj.update(action='replace',transaction_id='r7-'+spec['page']+'-restoration',source_layers=['reviewed-source-composite'])
        for role,mask in {**masks[obj['id']],'protection':protection}.items():
            path=protection_path if role=='protection' else maskdir/(obj['id']+'-'+role+'.png')
            if role!='protection':Image.fromarray(mask.astype('uint8')*255).save(path)
            obj['masks'][role]=mask_ref(path,size,root=output,method='frozen_reviewed_source_'+role)
    registry=resolve(registry,output)
    if additions:
        checked=authorize_repair(registry,[n['id'] for n in additions],output)
        if not np.array_equal(authorized,checked): raise ValueError('Repair authorization changed')
    path=maskdir/'authorized.png';Image.fromarray(authorized.astype('uint8')*255).save(path)
    write(output/'registry.json',registry)
    write(output/'authoring.json',{'tool':'image2ppt r7 deterministic source restoration','model_id':'none',
                                   'ir_sha256':digest(target),'input_ir_sha256':digest(irfile),
                                   'decisions':['Keep original text, groups and unaffected relative z order',
                                                'No baseline pixel assets modified; restore reviewed missing contributions'],
                                   'raster_records':raster_records})
    return {'ir':str(target),'registry':registry,'registry_path':str(output/'registry.json'),
            'added_ids':[n['id'] for n in additions],'removed_ids':removed,'authorized_mask':str(path),
            'raster_ids':list(raster_ids),'raster_records':raster_records,
            'baseline_ir_sha256':digest(irfile),'source_sha256':digest(source_path)}
