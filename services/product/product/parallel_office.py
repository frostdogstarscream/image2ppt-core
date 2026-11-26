"""Real Office checks, scoped to this worker's copies; never quit user Office."""
import gc
import json
import math
from pathlib import Path
import shutil
import time
import io, hashlib, posixpath, zipfile, xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from .materialize import digest,write
from .office_sites import clear_position, edit_sites, observation_window, record_site, site_geometry, visible_box


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


def verify_pictures(pptx,irs,deleted=None,*,cache=None):
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main',
        'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    if deleted is None:
        deleted_set=set()
    elif isinstance(deleted,tuple) and len(deleted)==2 and isinstance(deleted[0],int):
        deleted_set={deleted}
    else:
        deleted_set=set(deleted)
    if cache is not None and 'verified_media_sha' in cache:
        with zipfile.ZipFile(pptx) as z:
            for i,_ in enumerate(irs,1):
                xml=ET.fromstring(z.read(f'ppt/slides/slide{i}.xml'));rels=ET.fromstring(z.read(f'ppt/slides/_rels/slide{i}.xml.rels'))
                mapping={r.get('Id'):r.get('Target') for r in rels};pictures={}
                for pic in xml.findall('.//p:pic',ns):
                    name=pic.find('p:nvPicPr/p:cNvPr',ns).get('name')
                    if name in pictures:raise ValueError('duplicate final picture identity')
                    target=mapping[pic.find('.//a:blip',ns).get('{'+ns['r']+'}embed')]
                    media=target.lstrip('/') if target.startswith('/') else posixpath.normpath('ppt/slides/'+target)
                    pictures[name]=media
                for nid in cache['ir_image_ids'][i]:
                    if (i,nid) in deleted_set:continue
                    if nid not in pictures:raise ValueError('final picture asset binding mismatch')
                    raw_bytes=z.read(pictures[nid])
                    if hashlib.sha256(raw_bytes).hexdigest()==cache['verified_media_sha'][(i,nid)]:continue
                    wanted=np.array(Image.open(cache['wanted'][(i,nid)]).convert('RGBA'))
                    got=np.array(Image.open(io.BytesIO(raw_bytes)).convert('RGBA'))
                    if got.shape!=wanted.shape or not np.array_equal(got[...,3],wanted[...,3]) or not np.array_equal(got[...,:3][wanted[...,3]>0],wanted[...,:3][wanted[...,3]>0]):
                        raise ValueError('embedded Alpha or visible RGB changed')
        return
    cache_media_sha={};cache_wanted={};cache_ir_ids={}
    with zipfile.ZipFile(pptx) as z:
        for i,irfile in enumerate(irs,1):
            irfile=Path(irfile);ir=json.loads(irfile.read_text(encoding='utf-8'));assets={a['id']:a for a in ir['assets']}
            xml=ET.fromstring(z.read(f'ppt/slides/slide{i}.xml'));rels=ET.fromstring(z.read(f'ppt/slides/_rels/slide{i}.xml.rels'))
            mapping={r.get('Id'):r.get('Target') for r in rels};pictures={};media_sha={}
            for pic in xml.findall('.//p:pic',ns):
                name=pic.find('p:nvPicPr/p:cNvPr',ns).get('name')
                if name in pictures:raise ValueError('duplicate final picture identity')
                target=mapping[pic.find('.//a:blip',ns).get('{'+ns['r']+'}embed')]
                media=target.lstrip('/') if target.startswith('/') else posixpath.normpath('ppt/slides/'+target)
                pictures[name]=media
            cache_ir_ids[i]=[node['id'] for node in ir['nodes'] if node['type']=='image']
            for node in ir['nodes']:
                if node['type']!='image' or (i,node['id']) in deleted_set:continue
                a=assets[node['asset_id']];file=(irfile.parent/a['path']).resolve()
                if digest(file)!=a['sha256'] or node['id'] not in pictures:raise ValueError('final picture asset binding mismatch')
                raw_bytes=z.read(pictures[node['id']])
                media_sha[node['id']]=hashlib.sha256(raw_bytes).hexdigest()
                wanted=np.array(Image.open(file).convert('RGBA'));got=np.array(Image.open(io.BytesIO(raw_bytes)).convert('RGBA'))
                if got.shape!=wanted.shape or not np.array_equal(got[...,3],wanted[...,3]) or not np.array_equal(got[...,:3][wanted[...,3]>0],wanted[...,:3][wanted[...,3]>0]):
                    raise ValueError('embedded Alpha or visible RGB changed')
                if cache is not None:
                    cache_media_sha[(i,node['id'])]=media_sha[node['id']]
                    cache_wanted[(i,node['id'])]=str(file)
    if cache is not None and not deleted_set:
        cache['verified_media_sha']=cache_media_sha
        cache['wanted']=cache_wanted
        cache['ir_image_ids']=cache_ir_ids


def leaf_shapes(shapes):
    for shape in shapes:
        if shape.Type==6:
            yield from leaf_shapes(shape.GroupItems)
        else:
            yield shape


def _px_window(boxes_pt, sx, sy, pw, ph, pad_px, extra_px_boxes=()):
    """Compute clamped pixel window `(x0, y0, x1, y1)` covering point boxes and pixel boxes."""
    x0, y0, x1, y1 = pw, ph, 0, 0
    for b in boxes_pt:
        bx, by, bw, bh = b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy
        x0 = min(x0, max(0, int(bx) - pad_px))
        y0 = min(y0, max(0, int(by) - pad_px))
        x1 = max(x1, min(pw, int(np.ceil(bx + bw)) + pad_px))
        y1 = max(y1, min(ph, int(np.ceil(by + bh)) + pad_px))
    for pb in extra_px_boxes:
        if not pb or len(pb) != 4:
            continue
        bx, by, bw, bh = [float(v) for v in pb]
        x0 = min(x0, max(0, int(bx) - pad_px))
        y0 = min(y0, max(0, int(by) - pad_px))
        x1 = max(x1, min(pw, int(np.ceil(bx + bw)) + pad_px))
        y1 = max(y1, min(ph, int(np.ceil(by + bh)) + pad_px))
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0)
    return (x0, y0, x1, y1)


def _windows_disjoint(a, b):
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def _partition_disjoint_batches(names, windows, max_batch_size=16):
    """Partition object names into chromatic batches whose influence windows are pairwise disjoint."""
    batches = []
    batch_boxes = []
    for name in names:
        win = windows[name]
        placed = False
        for idx, b_names in enumerate(batches):
            if len(b_names) >= max_batch_size:
                continue
            if all(_windows_disjoint(win, other_win) for other_win in batch_boxes[idx]):
                b_names.append(name)
                batch_boxes[idx].append(win)
                placed = True
                break
        if not placed:
            batches.append([name])
            batch_boxes.append([win])
    return batches


def _node_bleed_margin_px(node: dict) -> int:
    """Compute safe pixel margin around a SlideIR node (accounts for outer shadows and thick/arrowed lines)."""
    margin = 4
    sh = node.get('shadow')
    if isinstance(sh, dict):
        blur = float(sh.get('blur_px', 0.0) or 0.0)
        dx = float(sh.get('offset_x_px', 0.0) or 0.0)
        dy = float(sh.get('offset_y_px', 0.0) or 0.0)
        margin = max(margin, math.ceil(blur * 2.5 + math.hypot(dx, dy)) + 6)
    ln = node.get('line')
    if isinstance(ln, dict):
        lw_px = float(ln.get('width_pt', 0.0) or 0.0) * 96.0 / 72.0
        has_end = ln.get('head_end') not in (None, 'none') or ln.get('tail_end') not in (None, 'none')
        if lw_px > 0 or has_end:
            margin = max(margin, math.ceil(lw_px * 1.5) + (10 if has_end else 0) + 4)
    return margin


def inspect(pptx,irs,output,*,render_only=False,target_ids=None,original_sites=False,text_layer_audit=False):
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);pptx=Path(pptx).resolve();original=digest(pptx)
    app=None
    for attempt in range(3):
        try:
            app=win32com.client.Dispatch('PowerPoint.Application')
            _ = app.Version
            break
        except Exception:
            app=None
            if attempt==2:raise
            time.sleep(0.4*(attempt+1))
    deck=None;records=[];site_records=[];unverified_site_groups=[];current_id=None
    text_layers = {}
    def text_styles(s):
        if not (s.HasTextFrame and s.TextFrame.HasText):return ()
        tr=s.TextFrame.TextRange
        return tuple((str(tr.Runs(j).Text),str(tr.Runs(j).Font.Name),str(tr.Runs(j).Font.NameFarEast),
            float(tr.Runs(j).Font.Size),int(tr.Runs(j).Font.Bold),int(tr.Runs(j).Font.Color.RGB)) for j in range(1,tr.Runs().Count+1))
    def shape_signature(s):
        return (float(s.Left),float(s.Top),float(s.Width),float(s.Height),float(s.Rotation),
            str(s.TextFrame.TextRange.Text) if s.HasTextFrame and s.TextFrame.HasText else '',text_styles(s))
    def signature(slide):
        return {str(s.Name):shape_signature(s) for s in leaf_shapes(slide.Shapes)}

    def transform(shape, name):
        if name not in destinations:
            shape.Left += 12
            shape.Top += 8
        shape.Width *= .9
        shape.Height *= .9
        if name in destinations:
            shape.Left, shape.Top = destinations[name]
        if shape.HasTextFrame and shape.TextFrame.HasText:
            scale_text(shape.TextFrame.TextRange, .9)
        return shape_signature(shape)

    def move_observation(name, reopened, after, render_file):
        geometry=reopened[name]
        new_bounds = visible_box(geometry, geometry[4])
        new_window = _px_window([new_bounds], sx, sy, pw, ph, margin_map.get(name, 4))
        if not _windows_disjoint(site_windows[name], new_window):
            raise ValueError('transformed picture still covers its original site')
        return record_site(output,i,[name],'move-clear',site_windows[name],after,render_file,sig,reopened)

    def _run_single(i,name,action,sig,before,pw,ph,sx,sy,pic_cache,margin_map,partners=()):
        nonlocal deck,current_id
        current_id=name
        members=(name,*partners)
        file=output/f'{i:02d}-{name}-{action}.pptx';shutil.copyfile(pptx,file)
        deck=app.Presentations.Open(str(file),ReadOnly=False,Untitled=False,WithWindow=False)
        slide=deck.Slides(i);s=next(s for s in leaf_shapes(slide.Shapes) if str(s.Name)==name);old=sig[name];wanted=None
        deleting = action in ('delete','delete-together')
        if deleting:
            for member in members:
                next(shape for shape in leaf_shapes(slide.Shapes) if str(shape.Name)==member).Delete()
        else:
            wanted=transform(s,name)
        deck.Save();deck.Close();deck=None
        verify_pictures(file,irs,{(i,member) for member in members} if deleting else None,cache=pic_cache)
        deck=app.Presentations.Open(str(file),ReadOnly=True,Untitled=False,WithWindow=False)
        slide=deck.Slides(i);got=signature(slide)
        if {k:v for k,v in got.items() if k not in members}!={k:v for k,v in sig.items() if k not in members}:raise ValueError('Office changed sibling')
        if deleting and any(member in got for member in members):raise ValueError('Office deletion did not persist')
        if wanted and (name not in got or any(abs(a-b)>.1 for a,b in zip(got[name][:5],wanted[:5]))):raise ValueError('Office transform did not persist')
        if wanted and got[name][5:]!=wanted[5:]:raise ValueError('Office target text or effective run style did not persist')
        png=output/f'{i:02d}-{name}-{action}.png';slide.Export(str(png),'PNG',pw,ph)
        after=np.array(Image.open(png).convert('RGB'))
        allowed=np.zeros((ph,pw),bool)
        bounds=[(sig[member],margin_map.get(member,4)) for member in members]
        if wanted:bounds.append((wanted,margin_map.get(name,4)))
        for b,pad in bounds:
            rect=visible_box(b,b[4]) if original_sites else b[:4]
            x0,y0,x1,y1=_px_window([rect],sx,sy,pw,ph,pad)
            allowed[y0:y1,x0:x1]=True
        outside=int(((before!=after).any(2)&~allowed).sum())
        if outside:raise ValueError('Office changed pixels outside transaction')
        deck.Close();deck=None
        if partners:
            return record_site(output,i,members,action,observation_window(site_boxes,members),after,png,sig,got)
        if wanted and name in site_windows:
            site_records.append(move_observation(name,got,after,png))
        return {'page':i,'id':name,'action':action,'saved_reopened':True,'outside_pixels':outside}
    def _try_run_batch(i,b_idx,batch_names,action,sig,before,influence_windows,pw,ph,sx,sy,pic_cache,margin_map):
        nonlocal deck
        batch_file=output/f'{i:02d}-batch-{b_idx:02d}-{action}.pptx'
        batch_png=output/f'{i:02d}-batch-{b_idx:02d}-{action}.png'
        shutil.copyfile(pptx,batch_file)
        batch_set=set(batch_names)
        wanted_map={}
        local_deck=None
        try:
            local_deck=app.Presentations.Open(str(batch_file),ReadOnly=False,Untitled=False,WithWindow=False)
            slide=local_deck.Slides(i)
            by_name={str(s.Name):s for s in leaf_shapes(slide.Shapes)}
            for name in batch_names:
                s=by_name[name]
                if action=='delete':s.Delete()
                else:
                    wanted_map[name]=transform(s,name)
            local_deck.Save();local_deck.Close();local_deck=None
            verify_pictures(batch_file,irs,{(i,n) for n in batch_names} if action=='delete' else None,cache=pic_cache)
            local_deck=app.Presentations.Open(str(batch_file),ReadOnly=True,Untitled=False,WithWindow=False)
            slide=local_deck.Slides(i);got=signature(slide)
            if {k:v for k,v in got.items() if k not in batch_set}!={k:v for k,v in sig.items() if k not in batch_set}:
                raise ValueError('Office changed sibling in batch')
            for name in batch_names:
                if action=='delete':
                    if name in got:raise ValueError('Office deletion did not persist in batch')
                else:
                    wanted=wanted_map[name]
                    if name not in got or any(abs(a-b)>.1 for a,b in zip(got[name][:5],wanted[:5])):
                        raise ValueError('Office transform did not persist in batch')
                    if got[name][5:]!=wanted[5:]:
                        raise ValueError('Office target text or effective run style did not persist in batch')
            if action=='move-scale' and any(name in site_windows for name in batch_names):
                # COM can preserve a picture's aspect ratio while applying a resize.
                # Only attribute real batch crop pixels to one object when the actual
                # reopened extents still isolate that observation from its batch peers.
                for owner in batch_names:
                    if owner not in site_windows:
                        continue
                    for peer in batch_set-{owner}:
                        extents=[visible_box(g,g[4]) for g in (sig[peer],got[peer])]
                        area=_px_window(extents,sx,sy,pw,ph,margin_map.get(peer,4))
                        if not _windows_disjoint(site_windows[owner],area):
                            raise ValueError('batch peer overlaps original-site observation')
            slide.Export(str(batch_png),'PNG',pw,ph)
            local_deck.Close();local_deck=None
            batch_after=np.array(Image.open(batch_png).convert('RGB'))
            union_allowed=np.zeros((ph,pw),bool)
            for name in batch_names:
                pad=margin_map.get(name,4)
                boxes=[sig[name]]+([wanted_map[name]] if name in wanted_map else [])
                for b in boxes:
                    rect=visible_box(b,b[4]) if original_sites else b[:4]
                    x0,y0,x1,y1=_px_window([rect],sx,sy,pw,ph,pad)
                    union_allowed[y0:y1,x0:x1]=True
            outside=int(((before!=batch_after).any(2)&~union_allowed).sum())
            if outside:
                raise ValueError('Office changed pixels outside batch transaction')
            if action=='delete':
                for name in batch_names:
                    ix0,iy0,ix1,iy1=influence_windows[name]
                    single=before.copy()
                    single[iy0:iy1,ix0:ix1]=batch_after[iy0:iy1,ix0:ix1]
                    Image.fromarray(single).save(output/f'{i:02d}-{name}-delete.png')
            else:
                observations = [move_observation(name,got,batch_after,batch_png)
                                for name in batch_names if name in site_windows]
                site_records.extend(observations)
            return {name:{'page':i,'id':name,'action':action,'saved_reopened':True,'outside_pixels':0} for name in batch_names}
        except Exception:
            return None
        finally:
            if local_deck is not None:
                try:local_deck.Close()
                except Exception:pass
            batch_file.unlink(missing_ok=True)
            batch_png.unlink(missing_ok=True)
    try:
        pic_cache={}
        verify_pictures(pptx,irs,cache=pic_cache)
        deck=app.Presentations.Open(str(pptx),ReadOnly=True,Untitled=False,WithWindow=False)
        pw=round(float(deck.PageSetup.SlideWidth)/.75);ph=round(float(deck.PageSetup.SlideHeight)/.75)
        sx=pw/float(deck.PageSetup.SlideWidth);sy=ph/float(deck.PageSetup.SlideHeight);inventory=[];before_arrays={};restore_maps={};margin_maps={}
        for i,file in enumerate(irs,1):
            irpath=Path(file)
            ir=json.loads(irpath.read_text(encoding='utf-8'));slide=deck.Slides(i)
            before_png=output/f'page-{i:02d}-before.png'
            slide.Export(str(before_png),'PNG',pw,ph)
            before_arrays[i]=np.array(Image.open(before_png).convert('RGB'))
            sig=signature(slide);targets=[] if render_only else [n['id'] for n in ir['nodes'] if n['id']!='source-plate'
                and (target_ids is None or n['id'] in target_ids)]
            if not set(targets)<=set(sig):raise ValueError('compiled object inventory mismatch')
            scene_file=irpath.parent/'scene.json'
            rmap={}
            if scene_file.is_file():
                sel=json.loads(scene_file.read_text(encoding='utf-8')).get('selected',{})
                rmap={oid:rec.get('restore_bbox') for oid,rec in sel.items() if rec.get('restore_bbox')}
            restore_maps[i]=rmap
            margin_maps[i]={n['id']:_node_bleed_margin_px(n) for n in ir['nodes']}
            moves,pairs,unverified=(edit_sites(irpath,target_ids,include_unverified=True)
                                    if original_sites and not render_only else ([],[],[]))
            unverified_site_groups.extend({'page':i,**row} for row in unverified)
            inventory.append((i,targets,sig,moves,pairs))
            if text_layer_audit:
                from .office_text_layer import export_text_layer
                text_layers[str(i)] = export_text_layer(slide, irpath, output/f'page-{i:02d}-text-cleared.png', pw, ph)
        deck.Close();deck=None
        for i,targets,sig,moves,pairs in inventory:
            before=before_arrays[i]
            rmap=restore_maps[i]
            margin_map=margin_maps[i]
            site_windows={}
            destinations={}
            if moves or pairs:
                site_boxes,canvas=site_geometry(irs[i-1])
                if tuple(canvas)!=(pw,ph):raise ValueError('render size does not match bound original-site canvas')
            for name in moves:
                geometry=sig[name]
                margin=margin_map.get(name,4)
                window=observation_window(site_boxes,[name])
                site_windows[name]=window
                site_pt=(window[0]/sx,window[1]/sy,window[2]/sx,window[3]/sy)
                destinations[name]=clear_position(geometry,site_pt,pw/sx,ph/sy,(margin+2)/min(sx,sy))
            records_by_key={}
            for action in ('move-scale','delete'):
                influence_windows={}
                for name in targets:
                    geometry=sig[name];old=geometry[:4]
                    rotation=geometry[4] if original_sites else 0
                    boxes=[visible_box(old,rotation)]
                    if action=='move-scale':
                        left,top=destinations.get(name,(old[0]+12,old[1]+8))
                        extent=1 if name in destinations else .9
                        boxes.append(visible_box((left,top,old[2]*extent,old[3]*extent),rotation))
                    extra=[rmap[name]] if name in rmap else ()
                    win_pad=max(16,margin_map.get(name,4)+12)
                    influence_windows[name]=_px_window(boxes,sx,sy,pw,ph,pad_px=win_pad,extra_px_boxes=extra)
                if action=='move-scale' and moves:
                    batches=_partition_disjoint_batches(moves,influence_windows,max_batch_size=16)
                    batches+=_partition_disjoint_batches([n for n in targets if n not in site_windows],influence_windows,max_batch_size=16)
                else:
                    batches=_partition_disjoint_batches(targets,influence_windows,max_batch_size=16)
                for b_idx,batch_names in enumerate(batches):
                    if len(batch_names)>1:
                        batch_recs=_try_run_batch(i,b_idx,batch_names,action,sig,before,influence_windows,pw,ph,sx,sy,pic_cache,margin_map)
                        if batch_recs is not None:
                            for name,rec in batch_recs.items():
                                records_by_key[(name,action)]=rec
                            continue
                    for name in batch_names:
                        records_by_key[(name,action)]=_run_single(i,name,action,sig,before,pw,ph,sx,sy,pic_cache,margin_map)
            for name in targets:
                for action in ('move-scale','delete'):
                    records.append(records_by_key[(name,action)])
            for backing,front in pairs:
                site_records.append(_run_single(i,backing,'delete-together',sig,before,pw,ph,sx,sy,
                                                pic_cache,margin_map,partners=(front,)))
            if original_sites and not render_only:
                from .office_groups import group_edit_targets, probe_group
                ir = json.loads(Path(irs[i-1]).read_text('utf8'))
                for group in group_edit_targets(ir, target_ids):
                    current_id = group[0]
                    geometry, _ = site_geometry(irs[i-1])
                    edits, sites = probe_group(app, pptx, output, i, group, before, sig, signature,
                        lambda path: verify_pictures(path, irs, cache=pic_cache), margin_map, geometry)
                    records.extend(edits)
                    site_records.extend(sites)
        if digest(pptx)!=original:raise ValueError('Office modified original')
        result={'status':'PASS_APP_AUTOMATED','pptx_sha256':original,'records':records,'human_acceptance':False,
                'edit_checked':not render_only,'embedded_alpha_and_visible_rgb_checked':True,'powerpoint_version':app.Version}
        if original_sites:
            result['original_sites']=site_records
            result['original_site_previews']={str(i):digest(output/f'page-{i:02d}-before.png') for i in before_arrays}
            result['unverified_original_site_groups']=unverified_site_groups
        if text_layer_audit:
            result['text_layers'] = text_layers
        write(output/'report.json',result);return result
    except Exception as exc:
        write(output/'report.json',{'status':'FAIL_APP_AUTOMATED','failed_object_id':current_id,
              'reason':str(exc)[:400],'error_type':type(exc).__name__,'records':records,
              'pptx_sha256':original,'human_acceptance':False})
        raise OfficeValidationError(current_id,str(exc)) from None
    finally:
        if deck is not None:
            try:deck.Close()
            except Exception:pass
        deck=None
        app=None
        gc.collect()
        pythoncom.CoUninitialize()
