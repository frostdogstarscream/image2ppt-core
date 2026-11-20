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


def inspect_text(pptx, irs, output, target_ids=None, *, include_long=True):
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
                valid=content and siblings and frame and outside==0 and not layout['overflow'] and not layout['unexpected_wrap']
                deck.Close();deck=None
                return {'page':page,'id':name,'action':action,'status':'PASS' if valid else 'FAIL',
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
                        frame=all(abs(va-vb)<.1 for va,vb in zip(got[name][:5],sig[name][:5]))
                        valid=content and frame and not layout['overflow'] and not layout['unexpected_wrap']
                        if not valid:
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
        result.update(status='PASS' if required and all(r['status']=='PASS' for r in required) else 'FAIL',
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
