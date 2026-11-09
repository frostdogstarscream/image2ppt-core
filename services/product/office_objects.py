"""Actual PowerPoint move/scale/delete checks on copies of final extracted objects."""
import argparse
import json
from pathlib import Path
import shutil
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


def check(source,output):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    original=digest(source);app=win32com.client.Dispatch('PowerPoint.Application');deck=None;records=[]
    try:
        for action in ('move-scale','delete'):
            file=output/(action+'.pptx');shutil.copy2(source,file)
            deck=app.Presentations.Open(str(file),ReadOnly=False,Untitled=False,WithWindow=False)
            expected=[]
            for i in range(1,deck.Slides.Count+1):
                slide=deck.Slides(i);targets=[s for s in slide.Shapes if str(s.Name).startswith('extracted-')]
                baseline_text=texts(slide);names=[str(s.Name) for s in targets];positions={};bounds=[]
                sx=1672/float(deck.PageSetup.SlideWidth);sy=941/float(deck.PageSetup.SlideHeight)
                if action=='move-scale':slide.Export(str(output/f'page-{i:02d}-before.png'),'PNG',1672,941)
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
                slide.Export(str(output/f'page-{i:02d}-{action}.png'),'PNG',1672,941)
                before=np.array(Image.open(output/f'page-{i:02d}-before.png').convert('RGB'))
                after=np.array(Image.open(output/f'page-{i:02d}-{action}.png').convert('RGB'))
                allowed=np.zeros((941,1672),bool)
                for x,y,w,h in bounds:
                    allowed[max(0,int(y)-4):min(941,int(np.ceil(y+h))+4),max(0,int(x)-4):min(1672,int(np.ceil(x+w))+4)]=True
                changed=(before!=after).any(2);outside=int((changed & ~allowed).sum())
                if outside:raise ValueError(f'non-target visual changes after {action}: page {i}, pixels={outside}')
                records.append({'page':i,'action':action,'object_count':len(names),'object_ids':names,'saved_reopened':True,'native_text_unchanged':True,'non_target_changed_pixels':outside})
            close_copy(deck);deck=None
        if digest(source)!=original:raise ValueError('original deck changed')
        report={'status':'PASS_APP_AUTOMATED','powerpoint_version':app.Version,'original_unchanged':True,'records':records,
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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('pptx');p.add_argument('output');a=p.parse_args();check(a.pptx,a.output)
