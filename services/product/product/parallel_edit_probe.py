"""Real PowerPoint content edits on copies, including saved/reopened overflow."""
import gc
import json
import shutil
import time
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


def _pixel_box(rect, scale):
    left, top, width, height = rect
    sx, sy = scale
    return (int(np.floor(left*sx)), int(np.floor(top*sy)),
            int(np.ceil((left+width)*sx)), int(np.ceil((top+height)*sy)))


def assess_text_footprint(before, after, backing, frame, glyphs, scale):
    """Compare visible replacement glyphs against the same slide with just its text cleared."""
    height, width = after.shape[:2]
    changed = np.any(before != after, axis=2)
    ink = np.max(np.abs(after.astype('int16') - backing.astype('int16')), axis=2) > 10
    allowed = np.zeros((height, width), dtype=bool)
    left, top, right, bottom = _pixel_box(frame, scale)
    allowed[max(0, top-4):bottom+4, max(0, left-4):right+4] = True
    raw_outside = int(np.count_nonzero(changed & ~allowed))
    issues = []
    visible = 0
    for _, rect in glyphs:
        x, y, r, b = _pixel_box(rect, scale)
        if not (0 <= x < r <= width and 0 <= y < b <= height):
            issues.append('glyph_outside_slide')
            continue
        # Ignore boundary rounding between adjacent Office character bounds.
        inner_x = min((r-x-1)//2, int((r-x)*0.12))
        region = ink[y:b, x+inner_x:r-inner_x]
        if not region.any():
            issues.append('glyph_without_visible_ink')
            continue
        visible += 1
        # A crisp edge in the text-cleared rendering can be another word, a card
        # boundary or artwork. Such cases need visual review, including inside the old frame.
        patch = backing[y:b, x:r].astype('int16')
        if any(np.any(np.abs(np.diff(patch, axis=axis)) > 18) for axis in (0, 1)):
            issues.append('nonuniform_backing_needs_review')
        # Beyond the original frame, only pixels actually painted by the target
        # text may change. No bounding rectangle grants permission to edit a peer.
        painted = np.any(after[y:b, x:r] != backing[y:b, x:r], axis=2)
        allowed[y:b, x:r] |= painted
    outside = int(np.count_nonzero(changed & ~allowed))
    if outside:
        issues.append('unexplained_pixel_change')
    if not glyphs:
        issues.append('no_visible_characters')
    return {'accepted': not issues, 'reasons': sorted(set(issues)),
            'glyph_count': len(glyphs), 'visible_glyph_count': visible,
            'frame_outside_pixels': raw_outside, 'outside_pixels': outside}


def _intersects(first, second):
    x, y, w, h = first
    px, py, pw, ph = second
    return x < px+pw and px < x+w and y < py+ph and py < y+h


def inspect_readable_layout(slide, shape, before, after, scale, blank_png):
    """Run on a reopened probe copy only, after recording its unmodified preview."""
    if abs(float(shape.Rotation)) > 0.1:
        return {'accepted': False, 'reasons': ['rotated_text_needs_review']}
    text_range = shape.TextFrame.TextRange
    glyphs = []
    for index in range(1, int(text_range.Length)+1):
        char = text_range.Characters(index, 1)
        if str(char.Text).strip():
            glyphs.append((str(char.Text), (char.BoundLeft, char.BoundTop, char.BoundWidth, char.BoundHeight)))
    # A topmost opaque shape can hide part of a character even on a white
    # background. Check the real stacking order, including group members.
    from .parallel_office import leaf_shapes
    shapes = list(leaf_shapes(slide.Shapes))
    position = next(i for i, peer in enumerate(shapes) if peer.Id == shape.Id)
    overlapping = []
    for peer in shapes[position+1:]:
        if not peer.Visible:
            continue
        px, py, pw, ph = peer.Left, peer.Top, peer.Width, peer.Height
        if any(_intersects(rect, (px, py, pw, ph)) for _, rect in glyphs):
            overlapping.append(peer)
    height, width = after.shape[:2]
    def render(path):
        slide.Export(str(path), 'PNG', width, height)
        with Image.open(path) as image:
            return np.asarray(image.convert('RGB'))
    if overlapping:
        for peer in overlapping:
            peer.Visible = 0
        clear_view = blank_png.with_suffix('.uncovered.png')
        uncovered = render(clear_view)
    frame = (shape.Left, shape.Top, shape.Width, shape.Height)
    text_range.Text = ''
    if overlapping:
        uncovered_blank = blank_png.with_suffix('.uncovered-blank.png')
        uncovered_ink = np.max(np.abs(uncovered.astype('int16') - render(uncovered_blank).astype('int16')), axis=2)
        for peer in overlapping:
            peer.Visible = -1
    backing = render(blank_png)
    proof = assess_text_footprint(before, after, backing, frame, glyphs, scale)
    if overlapping:
        visible_ink = np.max(np.abs(after.astype('int16') - backing.astype('int16')), axis=2)
        hidden = int(np.count_nonzero((uncovered_ink > 28) & (visible_ink <= 10)))
        proof.update(overlapping_shapes=[str(peer.Name) for peer in overlapping], hidden_ink_pixels=hidden,
                     uncovered_sha256=digest(clear_view), uncovered_blank_sha256=digest(uncovered_blank))
        if hidden:
            proof['accepted'] = False
            proof['reasons'].append('foreground_obscures_text')
    return {**proof, 'text_cleared_preview': Path(blank_png).name,
            'text_cleared_sha256': digest(blank_png)}


def source_wording_matches(office_text, expected):
    # PowerPoint distinguishes paragraph and soft breaks. Both are layout here.
    breaks = str.maketrans('', '', '\r\n\v')
    return office_text.translate(breaks) == expected.translate(breaks)


def replacement_accepted(record):
    layout = record.get('layout', {})
    fits = not layout.get('overflow') and not layout.get('unexpected_wrap')
    return (record.get('status', 'PASS') == 'PASS' and record.get('content_matches')
            and record.get('siblings_unchanged') and record.get('frame_unchanged')
            and record.get('outside_pixels') == 0
            and (fits or layout.get('readability', {}).get('accepted') is True))


def inspect_text(pptx, irs, output, target_ids=None, *, include_long=True, readable_text=False):
    from .parallel_office import leaf_shapes, _partition_disjoint_batches, _px_window
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    pptx=Path(pptx).resolve();original=digest(pptx);records=[];deck=None;app=None;initialized=False
    result={'schema':'TextEditProbe/1','status':'UNKNOWN','pptx_sha256':original,'records':records,'human_acceptance':False}
    try:
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize();initialized=True
        for attempt in range(3):
            try:
                app=win32com.client.Dispatch('PowerPoint.Application')
                _ = app.Version
                break
            except Exception as com_exc:
                app = None
                if attempt == 2:
                    raise
                time.sleep(0.4 * (attempt + 1))
        def signature(slide):
            return {str(s.Name):(float(s.Left),float(s.Top),float(s.Width),float(s.Height),float(s.Rotation),
                str(s.TextFrame.TextRange.Text) if s.HasTextFrame and s.TextFrame.HasText else '') for s in leaf_shapes(slide.Shapes)}
        variants=('short','equal','long') if include_long else ('short','equal')
        for page,irfile in enumerate(irs,1):
            ir=json.loads(Path(irfile).read_text(encoding='utf-8'))
            targets=[node for node in ir['nodes'] if node['type']=='text' and (target_ids is None or node['id'] in target_ids)]
            if not targets:continue
            deck=app.Presentations.Open(str(pptx),ReadOnly=True,Untitled=False,WithWindow=False)
            slide=deck.Slides(page);sig=signature(slide)
            pw,ph=ir['canvas']['source_px'];sx=pw/float(deck.PageSetup.SlideWidth);sy=ph/float(deck.PageSetup.SlideHeight)
            orig_line_counts={str(s.Name):int(s.TextFrame.TextRange.Lines().Count) for s in leaf_shapes(slide.Shapes) if s.HasTextFrame and s.TextFrame.HasText}
            before=output/f'{page:02d}-before.png';slide.Export(str(before),'PNG',pw,ph)
            a=np.array(Image.open(before).convert('RGB'))
            deck.Close();deck=None
            # Signature order follows the actual stacking order of the leaf shapes.
            peers_in_front = {}
            if readable_text:
                ordered = list(sig)
                for node in targets:
                    name = node['id']
                    peers_in_front[name] = [sig[k][:4] for k in ordered[ordered.index(name)+1:]]
            def needs_ink_check(name, layout, edited, expected):
                if not readable_text:
                    return False
                if any(_intersects(layout['text_bounds'], peer) for peer in peers_in_front[name]):
                    return True
                if sig[name][5] == expected:
                    return False
                x, y, r, b = _pixel_box(layout['text_bounds'], (sx, sy))
                window = np.s_[max(0, y):max(0, b), max(0, x):max(0, r)]
                return not np.any(a[window] != edited[window])
            def _run_single_probe(node,variant):
                nonlocal deck
                action='replace-'+variant;name=node['id'];longer=variant=='long'
                file=output/f'{page:02d}-{name}-{action}.pptx';shutil.copyfile(pptx,file)
                deck=app.Presentations.Open(str(file),ReadOnly=False,Untitled=False,WithWindow=False)
                slide=deck.Slides(page);s=next(s for s in leaf_shapes(slide.Shapes) if str(s.Name)==name)
                count=orig_line_counts[name]
                value=replacement(node['text'],longer,variant=='short').replace('\n','\r')
                s.TextFrame.TextRange.Text=value
                deck.Save();deck.Close();deck=None
                deck=app.Presentations.Open(str(file),ReadOnly=True,Untitled=False,WithWindow=False)
                slide=deck.Slides(page);got=signature(slide);s=next(s for s in leaf_shapes(slide.Shapes) if str(s.Name)==name);tr=s.TextFrame.TextRange
                layout=layout_result(got[name][:4],(float(tr.BoundLeft),float(tr.BoundTop),float(tr.BoundWidth),float(tr.BoundHeight)),int(tr.Lines().Count),count)
                png=output/f'{page:02d}-{name}-{action}.png';slide.Export(str(png),'PNG',pw,ph)
                b=np.array(Image.open(png).convert('RGB'))
                allowed=np.zeros((ph,pw),bool);x,y,w,h=sig[name][:4]
                allowed[max(0,int(y*sy)-4):min(ph,int(np.ceil((y+h)*sy))+4),max(0,int(x*sx)-4):min(pw,int(np.ceil((x+w)*sx))+4)]=True
                outside=int(((a!=b).any(2)&~allowed).sum())
                siblings={k:v for k,v in got.items() if k!=name}=={k:v for k,v in sig.items() if k!=name}
                content=str(tr.Text)==value;frame=all(abs(va-vb)<.1 for va,vb in zip(got[name][:5],sig[name][:5]))
                if readable_text:
                    content = content and source_wording_matches(sig[name][5], node['text'])
                valid=content and siblings and frame and outside==0 and not layout['overflow'] and not layout['unexpected_wrap']
                if readable_text and content and siblings and frame and (not valid or needs_ink_check(name, layout, b, value)):
                    proof = inspect_readable_layout(slide, s, a, b, (sx, sy), png.with_suffix('.blank.png'))
                    layout['readability'] = proof
                    outside = proof.get('outside_pixels', outside)
                    valid = proof['accepted']
                deck.Close();deck=None
                status = 'PASS' if valid else 'FAIL'
                review_reasons = layout.get('readability', {}).get('reasons', [])
                if review_reasons and all(reason in ('rotated_text_needs_review',
                        'nonuniform_backing_needs_review', 'unexplained_pixel_change') for reason in review_reasons):
                    status = 'REVIEW_REQUIRED'
                return {'page':page,'id':name,'action':action,'status':status,
                        'saved_reopened':True,'content_matches':content,'siblings_unchanged':siblings,'frame_unchanged':frame,
                        'outside_pixels':outside,'layout':layout,'pptx_sha256':digest(file),'preview_sha256':digest(png),
                        'expected_text':value,'probe_pptx':str(file),'preview':str(png)}
            def _try_batch_probe(b_idx,batch_nodes,variant):
                action='replace-'+variant;longer=variant=='long'
                file=output/f'{page:02d}-batch-{b_idx:02d}-{action}.pptx'
                png=output/f'{page:02d}-batch-{b_idx:02d}-{action}.png'
                batch_set={n['id'] for n in batch_nodes}
                values={}
                local_deck=None
                try:
                    shutil.copyfile(pptx,file)
                    local_deck=app.Presentations.Open(str(file),ReadOnly=False,Untitled=False,WithWindow=False)
                    slide=local_deck.Slides(page)
                    by_name={str(s.Name):s for s in leaf_shapes(slide.Shapes)}
                    for node in batch_nodes:
                        name=node['id']
                        val=replacement(node['text'],longer,variant=='short').replace('\n','\r')
                        values[name]=val
                        by_name[name].TextFrame.TextRange.Text=val
                    local_deck.Save();local_deck.Close();local_deck=None
                    local_deck=app.Presentations.Open(str(file),ReadOnly=True,Untitled=False,WithWindow=False)
                    slide=local_deck.Slides(page);got=signature(slide)
                    siblings={k:v for k,v in got.items() if k not in batch_set}=={k:v for k,v in sig.items() if k not in batch_set}
                    if not siblings:
                        return None
                    by_name={str(s.Name):s for s in leaf_shapes(slide.Shapes)}
                    slide.Export(str(png),'PNG',pw,ph)
                    b=np.array(Image.open(png).convert('RGB'))
                    union_allowed=np.zeros((ph,pw),bool)
                    for node in batch_nodes:
                        x,y,w,h=sig[node['id']][:4]
                        union_allowed[max(0,int(y*sy)-4):min(ph,int(np.ceil((y+h)*sy))+4),max(0,int(x*sx)-4):min(pw,int(np.ceil((x+w)*sx))+4)]=True
                    outside=int(((a!=b).any(2)&~union_allowed).sum())
                    if outside!=0:
                        return None
                    batch_rows={}
                    for node in batch_nodes:
                        name=node['id'];tr=by_name[name].TextFrame.TextRange
                        layout=layout_result(got[name][:4],(float(tr.BoundLeft),float(tr.BoundTop),float(tr.BoundWidth),float(tr.BoundHeight)),int(tr.Lines().Count),orig_line_counts[name])
                        content=str(tr.Text)==values[name]
                        if readable_text:
                            content = content and source_wording_matches(sig[name][5], node['text'])
                        frame=all(abs(va-vb)<.1 for va,vb in zip(got[name][:5],sig[name][:5]))
                        valid=content and frame and not layout['overflow'] and not layout['unexpected_wrap']
                        if not valid or needs_ink_check(name, layout, b, values[name]):
                            return None
                        batch_rows[name]={'page':page,'id':name,'action':action,'status':'PASS',
                                          'saved_reopened':True,'content_matches':True,'siblings_unchanged':True,'frame_unchanged':True,
                                          'outside_pixels':0,'layout':layout,'pptx_sha256':digest(file),'preview_sha256':digest(png),
                                          'expected_text':values[name],'probe_pptx':str(file),'preview':str(png)}
                    local_deck.Close();local_deck=None
                    return batch_rows
                except Exception:
                    return None
                finally:
                    if local_deck is not None:
                        try:local_deck.Close()
                        except Exception:pass
            win_map={node['id']:_px_window([sig[node['id']][:4]],sx,sy,pw,ph,pad_px=12) for node in targets}
            nodes_by_id={node['id']:node for node in targets}
            batches=_partition_disjoint_batches([node['id'] for node in targets],win_map,max_batch_size=16)
            rows_by_key={}
            for variant in variants:
                action='replace-'+variant
                for b_idx,batch_ids in enumerate(batches):
                    batch_nodes=[nodes_by_id[oid] for oid in batch_ids]
                    if len(batch_nodes)>1:
                        batch_rows=_try_batch_probe(b_idx,batch_nodes,variant)
                        if batch_rows is not None:
                            rows_by_key.update({(oid,variant):row for oid,row in batch_rows.items()})
                            continue
                        (output/f'{page:02d}-batch-{b_idx:02d}-{action}.pptx').unlink(missing_ok=True)
                        (output/f'{page:02d}-batch-{b_idx:02d}-{action}.png').unlink(missing_ok=True)
                    for node in batch_nodes:
                        rows_by_key[(node['id'],variant)]=_run_single_probe(node,variant)
            for node in targets:
                for variant in variants:
                    records.append(rows_by_key[(node['id'],variant)])
            write(output/'report.json',result)
        required=[r for r in records if r['action'] in ('replace-equal','replace-short')]
        status = 'FAIL'
        if required and all(r['status'] == 'PASS' for r in required):
            status = 'PASS'
        elif required and all(r['status'] != 'FAIL' for r in required):
            status = 'REVIEW_REQUIRED'
        result.update(status=status,
                      powerpoint_version=app.Version,long_text_supported=(bool(records) and all(r['status']=='PASS' for r in records)) if include_long else None)
    except Exception as exc:
        result.update(status='UNKNOWN',reason=type(exc).__name__+': '+str(exc))
    finally:
        if deck is not None:
            try:deck.Close()
            except Exception:pass
        deck=None
        app=None
        gc.collect()
        if initialized:pythoncom.CoUninitialize()
        if digest(pptx)!=original:result.update(status='FAIL',reason='original modified')
        write(output/'report.json',result)
    return result
