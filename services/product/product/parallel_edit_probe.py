"""Real PowerPoint content edits on copies, including saved/reopened overflow."""
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from .materialize import digest,write


def replacement(text, longer=False, shorter=False):
    phrase='编辑验证成功'
    value=''.join('\n' if c=='\n' else phrase[i%len(phrase)] if '\u4e00'<=c<='\u9fff' else '8' if c.isdigit() else 'X' if c.isalpha() else c
                  for i,c in enumerate(text))
    if shorter:return '\n'.join(line[:max(1,len(line)//2)] for line in value.split('\n'))
    return value*4 if longer else value


def layout_result(frame, bounds, lines, original_lines):
    left,top,width,height=frame;bl,bt,bw,bh=bounds
    overflow=bl<left-1 or bt<top-1 or bl+bw>left+width+1 or bt+bh>top+height+1
    return {'overflow':overflow,'line_count':lines,'unexpected_wrap':lines>original_lines,
            'frame':list(frame),'text_bounds':list(bounds)}


def inspect_text(pptx, irs, output, target_ids=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    pptx=Path(pptx).resolve();original=digest(pptx);records=[];deck=None;initialized=False
    result={'schema':'TextEditProbe/1','status':'UNKNOWN','pptx_sha256':original,'records':records,'human_acceptance':False}
    try:
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize();initialized=True
        app=win32com.client.Dispatch('PowerPoint.Application')
        def signature(slide):
            return {str(s.Name):(float(s.Left),float(s.Top),float(s.Width),float(s.Height),float(s.Rotation),
                str(s.TextFrame.TextRange.Text) if s.HasTextFrame and s.TextFrame.HasText else '') for s in slide.Shapes}
        for page,irfile in enumerate(irs,1):
            ir=json.loads(Path(irfile).read_text(encoding='utf-8'))
            for node in ir['nodes']:
                if node['type']!='text' or (target_ids is not None and node['id'] not in target_ids):continue
                for variant in ('short','equal','long'):
                    action='replace-'+variant;name=node['id'];longer=variant=='long'
                    file=output/f'{page:02d}-{name}-{action}.pptx';shutil.copyfile(pptx,file)
                    deck=app.Presentations.Open(str(file),ReadOnly=False,Untitled=False,WithWindow=False)
                    slide=deck.Slides(page);sig=signature(slide);s=next(s for s in slide.Shapes if str(s.Name)==name)
                    pw,ph=ir['canvas']['source_px'];sx=pw/float(deck.PageSetup.SlideWidth);sy=ph/float(deck.PageSetup.SlideHeight)
                    before=output/f'{page:02d}-{name}-{action}-before.png';slide.Export(str(before),'PNG',pw,ph)
                    count=int(s.TextFrame.TextRange.Lines().Count)
                    value=replacement(node['text'],longer,variant=='short').replace('\n','\r')
                    s.TextFrame.TextRange.Text=value
                    deck.Save();deck.Close();deck=None
                    deck=app.Presentations.Open(str(file),ReadOnly=True,Untitled=False,WithWindow=False)
                    slide=deck.Slides(page);got=signature(slide);s=next(s for s in slide.Shapes if str(s.Name)==name);tr=s.TextFrame.TextRange
                    layout=layout_result(got[name][:4],(float(tr.BoundLeft),float(tr.BoundTop),float(tr.BoundWidth),float(tr.BoundHeight)),int(tr.Lines().Count),count)
                    png=output/f'{page:02d}-{name}-{action}.png';slide.Export(str(png),'PNG',pw,ph)
                    a=np.array(Image.open(before).convert('RGB'));b=np.array(Image.open(png).convert('RGB'))
                    allowed=np.zeros((ph,pw),bool);x,y,w,h=sig[name][:4]
                    allowed[max(0,int(y*sy)-4):min(ph,int(np.ceil((y+h)*sy))+4),max(0,int(x*sx)-4):min(pw,int(np.ceil((x+w)*sx))+4)]=True
                    outside=int(((a!=b).any(2)&~allowed).sum())
                    siblings={k:v for k,v in got.items() if k!=name}=={k:v for k,v in sig.items() if k!=name}
                    content=str(tr.Text)==value;frame=all(abs(a-b)<.1 for a,b in zip(got[name][:5],sig[name][:5]))
                    valid=content and siblings and frame and outside==0 and not layout['overflow'] and not layout['unexpected_wrap']
                    row={'page':page,'id':name,'action':action,'status':'PASS' if valid else 'FAIL',
                         'saved_reopened':True,'content_matches':content,'siblings_unchanged':siblings,'frame_unchanged':frame,
                         'outside_pixels':outside,'layout':layout,'pptx_sha256':digest(file),'preview_sha256':digest(png),
                         'expected_text':value,'probe_pptx':str(file),'preview':str(png)}
                    records.append(row);write(output/'report.json',result)
                    deck.Close();deck=None
        required=[r for r in records if r['action'] in ('replace-equal','replace-short')]
        result.update(status='PASS' if required and all(r['status']=='PASS' for r in required) else 'FAIL',
                      powerpoint_version=app.Version,long_text_supported=bool(records) and all(r['status']=='PASS' for r in records))
    except Exception as exc:
        result.update(status='UNKNOWN',reason=type(exc).__name__+': '+str(exc))
    finally:
        if deck is not None:deck.Close()
        if initialized:pythoncom.CoUninitialize()
        if digest(pptx)!=original:result.update(status='FAIL',reason='original modified')
        write(output/'report.json',result)
    return result
