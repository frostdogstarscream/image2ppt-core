"""Real Office checks, scoped to this worker's copies; never quit user Office."""
import json
from pathlib import Path
import shutil
import io,posixpath,zipfile,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from .materialize import digest,write


class OfficeValidationError(ValueError):
    code='OFFICE_VALIDATION_FAILED'
    def __init__(self,object_id,reason):
        self.object_id=object_id
        super().__init__(reason)


def scale_text(tr, factor):
    """Scale absolute paragraph pitch together with glyph size in an edit probe."""
    for index in range(1, tr.Runs().Count+1):
        tr.Runs(index).Font.Size *= factor
    for index in range(1, tr.Paragraphs().Count+1):
        paragraph = tr.Paragraphs(index).ParagraphFormat
        for rule, spacing in (('LineRuleWithin', 'SpaceWithin'),
                              ('LineRuleBefore', 'SpaceBefore'), ('LineRuleAfter', 'SpaceAfter')):
            value = getattr(paragraph, spacing)
            if getattr(paragraph, rule) == 0 and value > 0:
                setattr(paragraph, spacing, value*factor)


def verify_pictures(pptx,irs,deleted=None):
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main',
        'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    with zipfile.ZipFile(pptx) as z:
        for i,irfile in enumerate(irs,1):
            irfile=Path(irfile);ir=json.loads(irfile.read_text(encoding='utf-8'));assets={a['id']:a for a in ir['assets']}
            xml=ET.fromstring(z.read(f'ppt/slides/slide{i}.xml'));rels=ET.fromstring(z.read(f'ppt/slides/_rels/slide{i}.xml.rels'))
            mapping={r.get('Id'):r.get('Target') for r in rels};pictures={}
            for pic in xml.findall('.//p:pic',ns):
                name=pic.find('p:nvPicPr/p:cNvPr',ns).get('name')
                if name in pictures:raise ValueError('duplicate final picture identity')
                target=mapping[pic.find('.//a:blip',ns).get('{'+ns['r']+'}embed')]
                media=target.lstrip('/') if target.startswith('/') else posixpath.normpath('ppt/slides/'+target)
                pictures[name]=np.array(Image.open(io.BytesIO(z.read(media))).convert('RGBA'))
            for node in ir['nodes']:
                if node['type']!='image' or deleted==(i,node['id']):continue
                a=assets[node['asset_id']];file=(irfile.parent/a['path']).resolve()
                if digest(file)!=a['sha256'] or node['id'] not in pictures:raise ValueError('final picture asset binding mismatch')
                wanted=np.array(Image.open(file).convert('RGBA'));got=pictures[node['id']]
                if got.shape!=wanted.shape or not np.array_equal(got[...,3],wanted[...,3]) or not np.array_equal(got[...,:3][wanted[...,3]>0],wanted[...,:3][wanted[...,3]>0]):
                    raise ValueError('embedded Alpha or visible RGB changed')


def inspect(pptx,irs,output,*,render_only=False,target_ids=None):
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);pptx=Path(pptx).resolve();original=digest(pptx)
    app=win32com.client.Dispatch('PowerPoint.Application');deck=None;records=[];current_id=None
    def text_styles(s):
        if not (s.HasTextFrame and s.TextFrame.HasText):return ()
        tr=s.TextFrame.TextRange
        return tuple((str(tr.Runs(j).Text),str(tr.Runs(j).Font.Name),str(tr.Runs(j).Font.NameFarEast),
            float(tr.Runs(j).Font.Size),int(tr.Runs(j).Font.Bold),int(tr.Runs(j).Font.Color.RGB)) for j in range(1,tr.Runs().Count+1))
    def signature(slide):
        return {str(s.Name):(float(s.Left),float(s.Top),float(s.Width),float(s.Height),float(s.Rotation),
            str(s.TextFrame.TextRange.Text) if s.HasTextFrame and s.TextFrame.HasText else '',text_styles(s)) for s in slide.Shapes}
    try:
        verify_pictures(pptx,irs)
        deck=app.Presentations.Open(str(pptx),ReadOnly=True,Untitled=False,WithWindow=False)
        pw=round(float(deck.PageSetup.SlideWidth)/.75);ph=round(float(deck.PageSetup.SlideHeight)/.75)
        sx=pw/float(deck.PageSetup.SlideWidth);sy=ph/float(deck.PageSetup.SlideHeight);inventory=[]
        for i,file in enumerate(irs,1):
            ir=json.loads(Path(file).read_text(encoding='utf-8'));slide=deck.Slides(i)
            slide.Export(str(output/f'page-{i:02d}-before.png'),'PNG',pw,ph)
            sig=signature(slide);targets=[] if render_only else [n['id'] for n in ir['nodes'] if n['id']!='source-plate'
                and (target_ids is None or n['id'] in target_ids)]
            if not set(targets)<=set(sig):raise ValueError('compiled object inventory mismatch')
            inventory.append((i,targets,sig))
        deck.Close();deck=None
        for i,targets,sig in inventory:
            for name in targets:
                current_id=name
                for action in ('move-scale','delete'):
                    file=output/f'{i:02d}-{name}-{action}.pptx';shutil.copyfile(pptx,file)
                    deck=app.Presentations.Open(str(file),ReadOnly=False,Untitled=False,WithWindow=False)
                    slide=deck.Slides(i);s=next(s for s in slide.Shapes if str(s.Name)==name);old=sig[name];wanted=None
                    if action=='delete':s.Delete()
                    else:
                        s.Left+=12;s.Top+=8;s.Width*=.9;s.Height*=.9
                        if s.HasTextFrame and s.TextFrame.HasText:
                            tr=s.TextFrame.TextRange
                            # A scale probe includes glyphs, not just the frame.
                            scale_text(tr,.9)
                        wanted=signature(slide)[name]
                    deck.Save();deck.Close();deck=None
                    verify_pictures(file,irs,(i,name) if action=='delete' else None)
                    deck=app.Presentations.Open(str(file),ReadOnly=True,Untitled=False,WithWindow=False)
                    slide=deck.Slides(i);got=signature(slide)
                    if {k:v for k,v in got.items() if k!=name}!={k:v for k,v in sig.items() if k!=name}:raise ValueError('Office changed sibling')
                    if action=='delete' and name in got:raise ValueError('Office deletion did not persist')
                    if wanted and (name not in got or any(abs(a-b)>.1 for a,b in zip(got[name][:5],wanted[:5]))):raise ValueError('Office transform did not persist')
                    if wanted and got[name][5:]!=wanted[5:]:raise ValueError('Office target text or effective run style did not persist')
                    png=output/f'{i:02d}-{name}-{action}.png';slide.Export(str(png),'PNG',pw,ph)
                    before=np.array(Image.open(output/f'page-{i:02d}-before.png').convert('RGB'));after=np.array(Image.open(png).convert('RGB'))
                    allowed=np.zeros((ph,pw),bool)
                    for b in [old]+([wanted] if wanted else []):
                        x,y,w,h=b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy
                        allowed[max(0,int(y)-4):min(ph,int(np.ceil(y+h))+4),max(0,int(x)-4):min(pw,int(np.ceil(x+w))+4)]=True
                    outside=int(((before!=after).any(2)&~allowed).sum())
                    if outside:raise ValueError('Office changed pixels outside transaction')
                    records.append({'page':i,'id':name,'action':action,'saved_reopened':True,'outside_pixels':outside})
                    deck.Close();deck=None
        if digest(pptx)!=original:raise ValueError('Office modified original')
        result={'status':'PASS_APP_AUTOMATED','pptx_sha256':original,'records':records,'human_acceptance':False,
                'edit_checked':not render_only,'embedded_alpha_and_visible_rgb_checked':True,'powerpoint_version':app.Version}
        write(output/'report.json',result);return result
    except Exception as exc:
        write(output/'report.json',{'status':'FAIL_APP_AUTOMATED','failed_object_id':current_id,
              'reason':str(exc)[:400],'error_type':type(exc).__name__,'records':records,
              'pptx_sha256':original,'human_acceptance':False})
        raise OfficeValidationError(current_id,str(exc)) from None
    finally:
        if deck is not None:deck.Close()
        pythoncom.CoUninitialize()
