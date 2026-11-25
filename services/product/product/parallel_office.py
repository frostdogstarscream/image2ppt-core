"""Real Office checks, scoped to this worker's copies; never quit user Office."""
import json
from pathlib import Path
import shutil
import time
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
            omitted = set()
            if deleted and deleted[0] == i:
                omitted = {deleted[1]} if isinstance(deleted[1], str) else set(deleted[1])
            xml=ET.fromstring(z.read(f'ppt/slides/slide{i}.xml'));rels=ET.fromstring(z.read(f'ppt/slides/_rels/slide{i}.xml.rels'))
            mapping={r.get('Id'):r.get('Target') for r in rels};pictures={}
            for pic in xml.findall('.//p:pic',ns):
                name=pic.find('p:nvPicPr/p:cNvPr',ns).get('name')
                if name in pictures:raise ValueError('duplicate final picture identity')
                target=mapping[pic.find('.//a:blip',ns).get('{'+ns['r']+'}embed')]
                media=target.lstrip('/') if target.startswith('/') else posixpath.normpath('ppt/slides/'+target)
                pictures[name]=np.array(Image.open(io.BytesIO(z.read(media))).convert('RGBA'))
            for node in ir['nodes']:
                if node['type']!='image' or node['id'] in omitted:continue
                a=assets[node['asset_id']];file=(irfile.parent/a['path']).resolve()
                if digest(file)!=a['sha256'] or node['id'] not in pictures:raise ValueError('final picture asset binding mismatch')
                wanted=np.array(Image.open(file).convert('RGBA'));got=pictures[node['id']]
                if got.shape!=wanted.shape or not np.array_equal(got[...,3],wanted[...,3]) or not np.array_equal(got[...,:3][wanted[...,3]>0],wanted[...,:3][wanted[...,3]>0]):
                    raise ValueError('embedded Alpha or visible RGB changed')


def leaf_shapes(shapes):
    for shape in shapes:
        if shape.Type==6:
            yield from leaf_shapes(shape.GroupItems)
        else:
            yield shape


def owned_edit_groups(irfile, ir, targets=None):
    """A group merits its own residual probe only when a selected source owner maps to it."""
    groups = [g for g in ir.get('groups', ()) if targets is None or g['id'] in targets]
    if not groups:
        return {}
    parent = Path(irfile).resolve().parent
    scene = json.loads((parent/'scene.json').read_text('utf8'))
    selected = scene.get('selected', {})
    verified = {}
    for group in groups:
        ref = selected.get(group['id'], {}).get('owner_evidence')
        if not ref:
            continue
        path = (parent/ref['file']).resolve()
        if not path.is_relative_to(parent) or not path.is_file() or digest(path) != ref['sha256']:
            raise ValueError('group source-owner evidence binding mismatch')
        verified[group['id']] = set(group['member_ids'])
    return verified


def _export_without_native_text(slide, ir, path, width, height):
    """Render the existing paint, pictures and groups with only editable glyphs removed."""
    expected = {n['id']: n['type'] for n in ir['nodes'] if n['type'] in ('text', 'table')}
    hidden = []
    for shape in leaf_shapes(slide.Shapes):
        name = str(shape.Name)
        if name not in expected:
            continue
        if expected[name] == 'text':
            shape.TextFrame.TextRange.Text = ''
        else:
            table = shape.Table
            for row in range(1, table.Rows.Count + 1):
                for col in range(1, table.Columns.Count + 1):
                    table.Cell(row, col).Shape.TextFrame.TextRange.Text = ''
        hidden.append(name)
    if set(hidden) != set(expected):
        raise ValueError('missing native text object in Office audit view')
    slide.Export(str(path), 'PNG', width, height)
    return {'sha256': digest(path), 'hidden_objects': sorted(hidden)}


class OfficeProbeUnavailable(RuntimeError):
    pass


def open_probe_document(app, path, *, read_only, checked):
    presentations = app.Presentations
    document = presentations.Open(str(path), ReadOnly=read_only, Untitled=False, WithWindow=False)
    if not checked:
        return document
    try:
        if Path(document.FullName) == path and document.Slides.Count:
            return document
    except Exception:
        pass  # Some Office builds return an incomplete late-bound Open result.
    # Retrieve the same open presentation. Never repeat an edit or select the active document.
    try:
        document = presentations.Item(path.name)
        if Path(document.FullName) == path and document.Slides.Count:
            return document
    except Exception as exc:
        raise OfficeProbeUnavailable('Office could not bind the opened probe to its expected file') from exc
    raise OfficeProbeUnavailable('Office returned a different or empty presentation for the probe')


def close_probe_document(app, document, path, *, checked):
    if not checked:
        document.Close()
        return
    # Close can report an automation error even after PowerPoint has closed the file.
    # Look up only our exact copy before retrying. Never repeat Save or an edit.
    error = None
    for attempt in range(16):
        if attempt:
            time.sleep(.1)
        try:
            if document is not None:
                document.Close()
        except Exception as exc:
            error = exc
        document = None
        try:
            presentations = app.Presentations
            for index in range(1, presentations.Count + 1):
                item = presentations.Item(index)
                if Path(item.FullName) == path:
                    document = item
                    break
            if document is None:
                return
        except Exception as exc:
            error = exc
    raise OfficeProbeUnavailable('Office could not confirm that the test copy closed') from error


def inspect(pptx,irs,output,*,render_only=False,target_ids=None,text_hidden_views=False,checked_open=False,
            check_owned_groups=False):
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);pptx=Path(pptx).resolve();original=digest(pptx)
    app=win32com.client.Dispatch('PowerPoint.Application');deck=None;records=[];current_id=None
    open_path = pptx
    def text_styles(s):
        if not (s.HasTextFrame and s.TextFrame.HasText):return ()
        tr=s.TextFrame.TextRange
        return tuple((str(tr.Runs(j).Text),str(tr.Runs(j).Font.Name),str(tr.Runs(j).Font.NameFarEast),
            float(tr.Runs(j).Font.Size),int(tr.Runs(j).Font.Bold),int(tr.Runs(j).Font.Color.RGB)) for j in range(1,tr.Runs().Count+1))
    def signature(slide):
        return {str(s.Name):(float(s.Left),float(s.Top),float(s.Width),float(s.Height),float(s.Rotation),
            str(s.TextFrame.TextRange.Text) if s.HasTextFrame and s.TextFrame.HasText else '',text_styles(s)) for s in leaf_shapes(slide.Shapes)}

    def group_geometry(shape):
        return tuple(float(v) for v in (shape.Left, shape.Top, shape.Width, shape.Height, shape.Rotation))

    def real_group(slide, name, members):
        shape = slide.Shapes(name)
        if shape.Type != 6 or {str(s.Name) for s in leaf_shapes(shape.GroupItems)} != members:
            raise ValueError('Office group membership differs from the claimed source owner')
        return shape

    def same_shape(actual, expected):
        return all(abs(a-b) <= .1 for a,b in zip(actual[:5],expected[:5])) and actual[5:] == expected[5:]
    try:
        verify_pictures(pptx,irs)
        deck=open_probe_document(app,pptx,read_only=True,checked=checked_open)
        pw=round(float(deck.PageSetup.SlideWidth)/.75);ph=round(float(deck.PageSetup.SlideHeight)/.75)
        sx=pw/float(deck.PageSetup.SlideWidth);sy=ph/float(deck.PageSetup.SlideHeight);inventory=[]
        hidden_views = {}
        for i,file in enumerate(irs,1):
            ir=json.loads(Path(file).read_text(encoding='utf-8'));slide=deck.Slides(i)
            slide.Export(str(output/f'page-{i:02d}-before.png'),'PNG',pw,ph)
            sig=signature(slide);targets=[] if render_only else [n['id'] for n in ir['nodes'] if n['id']!='source-plate'
                and (target_ids is None or n['id'] in target_ids)]
            if not set(targets)<=set(sig):raise ValueError('compiled object inventory mismatch')
            groups = owned_edit_groups(file,ir,target_ids) if check_owned_groups and not render_only else {}
            inventory.append((i, [*targets,*groups], sig, groups))
            if text_hidden_views:
                # This read-only presentation is discarded before opening any edit probe.
                try:
                    hidden_views[str(i)] = _export_without_native_text(
                        slide, ir, output/f'page-{i:02d}-text-hidden.png', pw, ph)
                except Exception as exc:
                    hidden_views[str(i)] = {'status': 'UNKNOWN', 'reason': type(exc).__name__}
        if text_hidden_views:
            deck.Saved = True
        close_probe_document(app,deck,open_path,checked=checked_open);deck=None
        for i,targets,sig,groups in inventory:
            for name in targets:
                current_id=name
                for action in ('move-scale','delete'):
                    file=output/f'{i:02d}-{name}-{action}.pptx';shutil.copyfile(pptx,file)
                    open_path = file
                    deck=open_probe_document(app,file,read_only=False,checked=checked_open)
                    slide=deck.Slides(i)
                    members = groups.get(name)
                    edited_ids = members or {name}
                    s = real_group(slide,name,members) if members else next(
                        s for s in leaf_shapes(slide.Shapes) if str(s.Name)==name)
                    old = group_geometry(s) if members else sig[name]
                    wanted = None
                    wanted_members = {}
                    if action=='delete':s.Delete()
                    else:
                        s.Left+=12;s.Top+=8
                        if members:
                            moved = signature(slide)
                            for member in members:
                                expected = (sig[member][0]+12, sig[member][1]+8, *sig[member][2:])
                                if not same_shape(moved[member],expected):
                                    raise ValueError('single group movement did not move every member')
                        if members:
                            s.Width=old[2]*.9;s.Height=old[3]*.9
                        else:
                            s.Width*=.9;s.Height*=.9
                        if not members and s.HasTextFrame and s.TextFrame.HasText:
                            tr=s.TextFrame.TextRange
                            # A scale probe includes glyphs, not just the frame.
                            scale_text(tr,.9)
                        if members:
                            wanted = group_geometry(s)
                            transformed = signature(slide)
                            wanted_members = {member:transformed[member] for member in members}
                            if abs(wanted[2]-old[2]*.9)>.1 or abs(wanted[3]-old[3]*.9)>.1:
                                raise ValueError('Office group did not accept requested scale')
                        else:
                            wanted=signature(slide)[name]
                    deck.Save()
                    close_probe_document(app,deck,file,checked=checked_open);deck=None
                    verify_pictures(file,irs,(i,edited_ids) if action=='delete' else None)
                    deck=open_probe_document(app,file,read_only=True,checked=checked_open)
                    slide=deck.Slides(i);got=signature(slide)
                    if {k:v for k,v in got.items() if k not in edited_ids}!={k:v for k,v in sig.items() if k not in edited_ids}:
                        raise ValueError('Office changed sibling')
                    if action=='delete' and (edited_ids.intersection(got) or
                            (members and name in {str(shape.Name) for shape in slide.Shapes})):
                        raise ValueError('Office deletion did not persist')
                    if wanted and members:
                        if not same_shape(group_geometry(real_group(slide,name,members)), wanted) or not all(
                                member in got and same_shape(got[member],expected) for member,expected in wanted_members.items()):
                            raise ValueError('Office group transform or member content did not persist')
                    elif wanted:
                        if name not in got or any(abs(a-b)>.1 for a,b in zip(got[name][:5],wanted[:5])):
                            raise ValueError('Office transform did not persist')
                        if got[name][5:]!=wanted[5:]:
                            raise ValueError('Office target text or effective run style did not persist')
                    png=output/f'{i:02d}-{name}-{action}.png';slide.Export(str(png),'PNG',pw,ph)
                    before=np.array(Image.open(output/f'page-{i:02d}-before.png').convert('RGB'));after=np.array(Image.open(png).convert('RGB'))
                    allowed=np.zeros((ph,pw),bool)
                    for b in [old]+([wanted] if wanted else []):
                        x,y,w,h=b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy
                        allowed[max(0,int(y)-4):min(ph,int(np.ceil(y+h))+4),max(0,int(x)-4):min(pw,int(np.ceil(x+w))+4)]=True
                    outside=int(((before!=after).any(2)&~allowed).sum())
                    if outside:raise ValueError('Office changed pixels outside transaction')
                    records.append({'page':i,'id':name,'action':action,'saved_reopened':True,'outside_pixels':outside,
                                    **({'group_members':sorted(members)} if members else {})})
                    close_probe_document(app,deck,file,checked=checked_open);deck=None
        if digest(pptx)!=original:raise ValueError('Office modified original')
        result={'status':'PASS_APP_AUTOMATED','pptx_sha256':original,'records':records,'human_acceptance':False,
                'edit_checked':not render_only,'embedded_alpha_and_visible_rgb_checked':True,'powerpoint_version':app.Version}
        if text_hidden_views:
            result['text_hidden_views'] = hidden_views
        write(output/'report.json',result);return result
    except Exception as exc:
        unavailable = isinstance(exc, OfficeProbeUnavailable)
        write(output/'report.json',{'status':'UNKNOWN' if unavailable else 'FAIL_APP_AUTOMATED','failed_object_id':current_id,
              'reason':str(exc)[:400],'error_type':type(exc).__name__,'records':records,
              'pptx_sha256':original,'human_acceptance':False})
        if unavailable:
            raise
        raise OfficeValidationError(current_id,str(exc)) from None
    finally:
        try:
            if deck is not None:close_probe_document(app,deck,open_path,checked=checked_open)
        finally:
            pythoncom.CoUninitialize()
