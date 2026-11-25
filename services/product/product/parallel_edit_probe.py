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


def _bounded_single_glyph(before, edited, deleted, pixel_frame):
    """All visible original and replacement ink must fit the unpadded, unclipped frame."""
    if before.shape != edited.shape or edited.shape != deleted.shape:
        return False
    height, width = edited.shape[:2]
    left, top, right, bottom = pixel_frame
    if not (0 < left < right < width and 0 < top < bottom < height):
        return False
    original_ink = (before != deleted).any(2)
    new_ink = (edited != deleted).any(2)
    crop = np.s_[top:bottom, left:right]
    return (original_ink.any() and new_ink.any() and np.any(before[crop] != edited[crop])
            and original_ink.sum() == original_ink[crop].sum() and new_ink.sum() == new_ink[crop].sum())


def _single_glyph_probe(shape, slide, deck, before, edited, deleted_file, pixel_frame, expanded_file, layout):
    """Check if an Office text bound overhang consists only of invisible spacing."""
    import pywintypes

    if not deleted_file.is_file():
        return
    try:
        deleted = np.asarray(Image.open(deleted_file).convert('RGB'))
        if not _bounded_single_glyph(before, edited, deleted, pixel_frame):
            return
        shape.Width *= 2
        shape.Height *= 2
        tr = shape.TextFrame.TextRange
        expanded = layout_result((shape.Left, shape.Top, shape.Width, shape.Height),
                                 (tr.BoundLeft, tr.BoundTop, tr.BoundWidth, tr.BoundHeight), tr.Lines().Count, 1)
        if expanded['overflow'] or expanded['line_count'] != 1:
            return
        slide.Export(str(expanded_file), 'PNG', edited.shape[1], edited.shape[0])
        if np.array_equal(edited, np.asarray(Image.open(expanded_file).convert('RGB'))):
            layout.update(overflow=False, reported_overflow=True, evidence='single_glyph_ink_within_original_frame',
                          delete_sha256=digest(deleted_file), expanded_preview_sha256=digest(expanded_file))
    except (OSError, pywintypes.com_error):
        return
    finally:
        # The comparison happens only in the reopened read-only probe copy.
        deck.Saved = True


def inspect_text(pptx, irs, output, target_ids=None, *, include_long=True, checked_open=False,
                 verified_deletions=None):
    from .parallel_office import close_probe_document, leaf_shapes, open_probe_document
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
                str(s.TextFrame.TextRange.Text) if s.HasTextFrame and s.TextFrame.HasText else '') for s in leaf_shapes(slide.Shapes)}
        for page,irfile in enumerate(irs,1):
            ir=json.loads(Path(irfile).read_text(encoding='utf-8'))
            for node in ir['nodes']:
                if node['type']!='text' or (target_ids is not None and node['id'] not in target_ids):continue
                for variant in (('short','equal','long') if include_long else ('short','equal')):
                    action='replace-'+variant;name=node['id'];longer=variant=='long'
                    file=output/f'{page:02d}-{name}-{action}.pptx';shutil.copyfile(pptx,file)
                    deck=open_probe_document(app,file,read_only=False,checked=checked_open)
                    slide=deck.Slides(page);sig=signature(slide);s=next(s for s in leaf_shapes(slide.Shapes) if str(s.Name)==name)
                    pw,ph=ir['canvas']['source_px'];sx=pw/float(deck.PageSetup.SlideWidth);sy=ph/float(deck.PageSetup.SlideHeight)
                    before=output/f'{page:02d}-{name}-{action}-before.png';slide.Export(str(before),'PNG',pw,ph)
                    count=int(s.TextFrame.TextRange.Lines().Count)
                    value=replacement(node['text'],longer,variant=='short').replace('\n','\r')
                    s.TextFrame.TextRange.Text=value
                    deck.Save()
                    close_probe_document(app,deck,file,checked=checked_open);deck=None
                    deck=open_probe_document(app,file,read_only=True,checked=checked_open)
                    slide=deck.Slides(page);got=signature(slide);s=next(s for s in leaf_shapes(slide.Shapes) if str(s.Name)==name);tr=s.TextFrame.TextRange
                    layout=layout_result(got[name][:4],(float(tr.BoundLeft),float(tr.BoundTop),float(tr.BoundWidth),float(tr.BoundHeight)),int(tr.Lines().Count),count)
                    png=output/f'{page:02d}-{name}-{action}.png';slide.Export(str(png),'PNG',pw,ph)
                    a=np.array(Image.open(before).convert('RGB'));b=np.array(Image.open(png).convert('RGB'))
                    allowed=np.zeros((ph,pw),bool);x,y,w,h=sig[name][:4]
                    allowed[max(0,int(y*sy)-4):min(ph,int(np.ceil((y+h)*sy))+4),max(0,int(x*sx)-4):min(pw,int(np.ceil((x+w)*sx))+4)]=True
                    outside=int(((a!=b).any(2)&~allowed).sum())
                    siblings={k:v for k,v in got.items() if k!=name}=={k:v for k,v in sig.items() if k!=name}
                    content=str(tr.Text)==value;frame=all(abs(a-b)<.1 for a,b in zip(got[name][:5],sig[name][:5]))
                    delete_file = (verified_deletions or {}).get((page, name))
                    if (delete_file and content and siblings and frame and outside == 0 and layout['overflow']
                            and layout['line_count'] == count == 1 and sig[name][4] == 0
                            and len(node['text']) == len(value) == 1 and node['text'].isalnum() and value.isalnum()):
                        pixel_frame = (int(x*sx), int(y*sy), int(np.ceil((x+w)*sx)), int(np.ceil((y+h)*sy)))
                        _single_glyph_probe(s, slide, deck, a, b, delete_file, pixel_frame,
                                            output/f'{page:02d}-{name}-{action}-expanded.png', layout)
                    valid=content and siblings and frame and outside==0 and not layout['overflow'] and not layout['unexpected_wrap']
                    row={'page':page,'id':name,'action':action,'status':'PASS' if valid else 'FAIL',
                         'saved_reopened':True,'content_matches':content,'siblings_unchanged':siblings,'frame_unchanged':frame,
                         'outside_pixels':outside,'layout':layout,'pptx_sha256':digest(file),'preview_sha256':digest(png),
                         'expected_text':value,'probe_pptx':str(file),'preview':str(png)}
                    records.append(row);write(output/'report.json',result)
                    close_probe_document(app,deck,file,checked=checked_open);deck=None
        required=[r for r in records if r['action'] in ('replace-equal','replace-short')]
        result.update(status='PASS' if required and all(r['status']=='PASS' for r in required) else 'FAIL',
                      powerpoint_version=app.Version,long_text_supported=(bool(records) and all(r['status']=='PASS' for r in records)) if include_long else None)
    except Exception as exc:
        result.update(status='UNKNOWN',reason=type(exc).__name__+': '+str(exc))
    finally:
        try:
            if deck is not None:close_probe_document(app,deck,file,checked=checked_open)
        finally:
            if initialized:pythoncom.CoUninitialize()
        if digest(pptx)!=original:result.update(status='FAIL',reason='original modified')
        write(output/'report.json',result)
    return result
