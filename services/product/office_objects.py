"""Actual PowerPoint move/scale/delete checks on copies of final extracted objects."""
import argparse
import json
from pathlib import Path
import shutil
import io
import posixpath
import zipfile
import xml.etree.ElementTree as ET
import time
import numpy as np
from PIL import Image
import win32com.client
from product.materialize import digest,write


def texts(slide):
    return sorted(str(s.TextFrame.TextRange.Text) for s in slide.Shapes if s.HasTextFrame and s.TextFrame.HasText)


def close_copy(deck):
    for attempt in range(6):
        try:
            deck.Close()
            return
        except Exception:
            if attempt==5:raise
            time.sleep(.5)


def check(source,output,expected_scenes=None,per_object=False):
    if per_object:return check_individual(source,output,expected_scenes)
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    original=digest(source);app=win32com.client.Dispatch('PowerPoint.Application');deck=None;records=[]
    expected_scenes=expected_scenes or []
    try:
        for action in ('move-scale','delete'):
            file=output/(action+'.pptx');shutil.copy2(source,file)
            deck=app.Presentations.Open(str(file),ReadOnly=False,Untitled=False,WithWindow=False)
            pixel_width=round(float(deck.PageSetup.SlideWidth)/.75);pixel_height=round(float(deck.PageSetup.SlideHeight)/.75)
            expected=[]
            for i in range(1,deck.Slides.Count+1):
                slide=deck.Slides(i);targets=[s for s in slide.Shapes if str(s.Name).startswith('extracted-')]
                if expected_scenes:
                    scene=expected_scenes[i-1];names_all={str(s.Name) for s in slide.Shapes}
                    required={'extracted-'+r['id'] for r in scene['objects'] if r['status']=='STAGED'}
                    owners={owner for r in scene['objects'] if r['status']=='STAGED' for owner in r['replaces']}
                    if {str(s.Name) for s in targets}!=required:raise ValueError('final object target inventory mismatch')
                    if (owners-required) & names_all:raise ValueError('old object owner survived replacement')
                    if any(not c.get('owner_proof') and r['replaces'] for r in scene['objects'] if r['status']=='STAGED'
                           for c in r['candidates'] if c['index']==r['selected_index']):raise ValueError('owner evidence absent')
                baseline_text=texts(slide);names=[str(s.Name) for s in targets];positions={};bounds=[]
                sx=pixel_width/float(deck.PageSetup.SlideWidth);sy=pixel_height/float(deck.PageSetup.SlideHeight)
                if action=='move-scale':slide.Export(str(output/f'page-{i:02d}-before.png'),'PNG',pixel_width,pixel_height)
                for shape in targets:
                    bounds.append([float(shape.Left)*sx,float(shape.Top)*sy,float(shape.Width)*sx,float(shape.Height)*sy])
                    if action=='delete':shape.Delete()
                    else:
                        shape.Left+=12;shape.Top+=8;shape.Width*=.9;shape.Height*=.9
                        positions[str(shape.Name)]=[float(shape.Left),float(shape.Top),float(shape.Width),float(shape.Height)]
                        bounds.append([float(shape.Left)*sx,float(shape.Top)*sy,float(shape.Width)*sx,float(shape.Height)*sy])
                expected.append((baseline_text,names,positions,bounds))
            deck.Save();close_copy(deck);deck=None
            deck=app.Presentations.Open(str(file),ReadOnly=True,Untitled=False,WithWindow=False)
            for i,(baseline_text,names,positions,bounds) in enumerate(expected,1):
                slide=deck.Slides(i);actual={str(s.Name):s for s in slide.Shapes}
                if texts(slide)!=baseline_text:raise ValueError('native wording changed during image edit')
                if action=='delete' and any(n in actual for n in names):raise ValueError('object deletion did not persist')
                if action=='move-scale':
                    for name,values in positions.items():
                        s=actual[name];got=[float(s.Left),float(s.Top),float(s.Width),float(s.Height)]
                        if any(abs(a-b)>.1 for a,b in zip(values,got)):raise ValueError('object transform did not persist')
                slide.Export(str(output/f'page-{i:02d}-{action}.png'),'PNG',pixel_width,pixel_height)
                before=np.array(Image.open(output/f'page-{i:02d}-before.png').convert('RGB'))
                after=np.array(Image.open(output/f'page-{i:02d}-{action}.png').convert('RGB'))
                allowed=np.zeros((pixel_height,pixel_width),bool)
                for x,y,w,h in bounds:
                    allowed[max(0,int(y)-4):min(pixel_height,int(np.ceil(y+h))+4),max(0,int(x)-4):min(pixel_width,int(np.ceil(x+w))+4)]=True
                changed=(before!=after).any(2);outside=int((changed & ~allowed).sum())
                if outside:raise ValueError(f'non-target visual changes after {action}: page {i}, pixels={outside}')
                records.append({'page':i,'action':action,'object_count':len(names),'object_ids':names,'saved_reopened':True,'native_text_unchanged':True,'non_target_changed_pixels':outside})
            close_copy(deck);deck=None
        if digest(source)!=original:raise ValueError('original deck changed')
        report={'status':'PASS_APP_AUTOMATED','powerpoint_version':app.Version,'original_unchanged':True,'records':records,
            'owner_residual_checked':bool(expected_scenes),
            'pptx_sha256':original,'scene_sha256s':[s.get('scene_sha256') for s in expected_scenes],
            'visual_residual_check':'REQUIRES_IMAGE_INSPECTION','human_acceptance':False}
        write(output/'report.json',report);print(json.dumps(report))
    except Exception as exc:
        write(output/'report.json',{'status':'FAILED_APP_CHECK','records':records,
            'error_type':type(exc).__name__,'error':str(exc),'original_unchanged':digest(source)==original,
            'human_acceptance':False})
        raise
    finally:
        if deck is not None:close_copy(deck)
        # Never close other user presentations or quit their PowerPoint process.


def check_individual(source,output,scenes):
    """Move/scale/delete ONE object per copy; sibling changes cannot hide in a
    union of every object's bounding boxes. Originals and user windows stay open.
    """
    if not scenes:raise ValueError('candidate-bound scenes required')
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    original=digest(source);app=win32com.client.Dispatch('PowerPoint.Application');deck=None;records=[];inventory=[]
    def signature(slide):
        return {str(s.Name):[float(s.Left),float(s.Top),float(s.Width),float(s.Height),float(s.Rotation)] for s in slide.Shapes}
    try:
        verify_images(source,scenes)
        deck=app.Presentations.Open(str(source),ReadOnly=True,Untitled=False,WithWindow=False)
        pw=round(float(deck.PageSetup.SlideWidth)/.75);ph=round(float(deck.PageSetup.SlideHeight)/.75)
        sx=pw/float(deck.PageSetup.SlideWidth);sy=ph/float(deck.PageSetup.SlideHeight)
        for i,scene in enumerate(scenes,1):
            slide=deck.Slides(i);actual=signature(slide)
            targets=['extracted-'+r['id'] for r in scene['objects'] if r['status']=='STAGED']
            owners={n for r in scene['objects'] if r['status']=='STAGED' for n in r['replaces']}
            if not set(targets)<=set(actual) or (owners-set(targets)) & set(actual):raise ValueError('target/owner inventory mismatch')
            slide.Export(str(output/f'page-{i:02d}-before.png'),'PNG',pw,ph)
            inventory.append({'page':i,'targets':targets,'signature':actual,'texts':texts(slide)})
        close_copy(deck);deck=None
        for item in inventory:
            i=item['page']
            for name in item['targets']:
                for action in ('move-scale','delete'):
                    stem=f'{i:02d}-{name}-{action}';file=output/(stem+'.pptx');shutil.copyfile(source,file)
                    deck=app.Presentations.Open(str(file),ReadOnly=False,Untitled=False,WithWindow=False)
                    slide=deck.Slides(i);shape=next(s for s in slide.Shapes if str(s.Name)==name)
                    old=item['signature'][name];bounds=[[old[0]*sx,old[1]*sy,old[2]*sx,old[3]*sy]];wanted=None
                    if action=='delete':shape.Delete()
                    else:
                        shape.Left+=12;shape.Top+=8;shape.Width*=.9;shape.Height*=.9
                        wanted=signature(slide)[name];bounds.append([wanted[0]*sx,wanted[1]*sy,wanted[2]*sx,wanted[3]*sy])
                    deck.Save();close_copy(deck);deck=None
                    deck=app.Presentations.Open(str(file),ReadOnly=True,Untitled=False,WithWindow=False)
                    if action=='move-scale':verify_images(file,scenes)
                    slide=deck.Slides(i);actual=signature(slide)
                    expected={k:v for k,v in item['signature'].items() if k!=name}
                    siblings={k:v for k,v in actual.items() if k!=name}
                    if siblings!=expected or texts(slide)!=item['texts']:raise ValueError('non-target structure changed')
                    if action=='delete' and name in actual:raise ValueError('deletion did not persist')
                    if wanted and (name not in actual or any(abs(a-b)>.1 for a,b in zip(actual[name],wanted))):raise ValueError('transform did not persist')
                    png=output/(stem+'.png');slide.Export(str(png),'PNG',pw,ph)
                    before=np.array(Image.open(output/f'page-{i:02d}-before.png').convert('RGB'));after=np.array(Image.open(png).convert('RGB'))
                    allowed=np.zeros((ph,pw),bool)
                    for x,y,w,h in bounds:allowed[max(0,int(y)-4):min(ph,int(np.ceil(y+h))+4),max(0,int(x)-4):min(pw,int(np.ceil(x+w))+4)]=True
                    outside=int(((before!=after).any(2)&~allowed).sum())
                    if outside:raise ValueError('non-target rendered pixels changed')
                    records.append({'page':i,'action':action,'object_ids':[name],'object_count':1,'saved_reopened':True,
                        'sibling_transforms_unchanged':True,'native_text_unchanged':True,'non_target_changed_pixels':outside})
                    close_copy(deck);deck=None
        if digest(source)!=original:raise ValueError('original modified')
        report={'status':'PASS_APP_AUTOMATED','records':records,'owner_residual_checked':True,'per_object':True,
            'compiled_assets_checked':True,
            'saved_media_checked':True,
            'pptx_sha256':original,'scene_sha256s':[s.get('scene_sha256') for s in scenes],'human_acceptance':False,
            'original_unchanged':True,'powerpoint_version':app.Version}
        write(output/'report.json',report);return report
    except Exception as exc:
        write(output/'report.json',{'status':'FAILED_APP_CHECK','error_type':type(exc).__name__,'records':records,'human_acceptance':False});raise
    finally:
        if deck is not None:close_copy(deck)


def verify_images(pptx,scenes):
    """Check actual embedded pixels, including legitimate replacement ID reuse."""
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main',
        'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    with zipfile.ZipFile(pptx) as archive:
        for i,scene in enumerate(scenes,1):
            wanted=[r for r in scene['objects'] if r['status']=='STAGED']
            if not wanted:continue
            irfile=Path(scene['slide_ir']);ir=json.loads(irfile.read_text(encoding='utf-8'));assets={a['id']:a for a in ir['assets']}
            nodes={n['id']:n for n in ir['nodes']}
            xml=ET.fromstring(archive.read(f'ppt/slides/slide{i}.xml'));rels=ET.fromstring(archive.read(f'ppt/slides/_rels/slide{i}.xml.rels'))
            mapping={r.attrib['Id']:r.attrib['Target'] for r in rels};pictures={}
            for pic in xml.findall('.//p:pic',ns):
                prop=pic.find('p:nvPicPr/p:cNvPr',ns);name=prop.get('name')
                if name in pictures:raise ValueError('duplicate final picture name')
                blip=pic.find('.//a:blip',ns);target=mapping[blip.get('{'+ns['r']+'}embed')]
                pictures[name]=np.array(Image.open(io.BytesIO(archive.read(posixpath.normpath('ppt/slides/'+target)))).convert('RGBA'))
            for row in wanted:
                targets=[('extracted-'+row['id'],row['asset_sha256'])]+[(b['id'],b['sha256']) for b in row.get('background_assets',[])]
                for name,expected in targets:
                    asset=assets[nodes[name]['asset_id']];file=irfile.parent/asset['path']
                    if digest(file)!=expected or name not in pictures:raise ValueError('compiled asset binding mismatch')
                    wanted=np.array(Image.open(file).convert('RGBA'));actual=pictures[name]
                    if actual.shape!=wanted.shape or not np.array_equal(actual[...,3],wanted[...,3]) or not np.array_equal(actual[...,:3][wanted[...,3]>0],wanted[...,:3][wanted[...,3]>0]):
                        raise ValueError('embedded object pixels changed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('pptx');p.add_argument('output');p.add_argument('--scenes',type=Path)
    p.add_argument('--per-object',action='store_true')
    a=p.parse_args();check(a.pptx,a.output,json.loads(a.scenes.read_text(encoding='utf-8')) if a.scenes else None,a.per_object)
