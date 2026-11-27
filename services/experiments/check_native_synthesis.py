"""Exercise the actual grouped PowerPoint objects of a native synthesis composition."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from win32com.client import DispatchEx

from product.materialize import digest, write
from product.resource_lease import lease


def shapes(collection):
    return {str(s.Name):s for s in collection}


def check(composition, output, baseline=None):
    output.mkdir(parents=True,exist_ok=False)
    ppt = output/'原生重建检查稿.pptx'
    shutil.copyfile(composition['artifacts']['pptx'],ppt)
    changed = output/'编辑并保存的副本.pptx'
    shutil.copyfile(ppt,changed)
    edits = []
    references = list(baseline['pages'].values()) if baseline else []
    with lease('office'):
        app = DispatchEx('PowerPoint.Application')
        try:
            deck = app.Presentations.Open(str(changed),False,False,False)
            try:
                for index,page in enumerate(composition['pages'],1):
                    ir=json.loads(Path(page['ir']).read_text('utf8'))
                    slide=deck.Slides(index)
                    top=shapes(slide.Shapes)
                    nodes={n['id']:n for n in ir['nodes']}
                    groups=sorted(ir['groups'],key=lambda g:len(g['member_ids']),reverse=True)
                    # Prefer a card with a large photographic module when one exists.
                    def photo_size(group):
                        return max((n['bbox'][2]*n['bbox'][3] for k in group['member_ids']
                                    if (n:=nodes[k])['type']=='image'),default=0)
                    def edit_priority(group):
                        carrier = max((nodes[k]['bbox'] for k in group['member_ids']
                                       if nodes[k]['type']=='shape'), key=lambda b:b[2]*b[3])
                        peer_cards = sum(any(nodes[k]['type']=='shape' and
                            abs(nodes[k]['bbox'][1]-carrier[1]) < .03*ir['canvas']['source_px'][1]
                            and .75 < nodes[k]['bbox'][2]/carrier[2] < 1.30
                            and .75 < nodes[k]['bbox'][3]/carrier[3] < 1.30
                            for k in other['member_ids']) for other in groups if other!=group)
                        return (photo_size(group)>.025*np.prod(ir['canvas']['source_px']),
                                bool(peer_cards),any('text_override_reason' in nodes[k] for k in group['member_ids']),
                                len(group['member_ids']))
                    group=max(groups,key=edit_priority)
                    parent=top[group['id']]
                    members=shapes(parent.GroupItems)
                    if set(members)!=set(group['member_ids']):
                        raise ValueError('PowerPoint group inventory differs from the compiled scene')
                    width,height=ir['canvas']['source_px']
                    def render(label):
                        path=output/f'{index}-{label}.png'
                        slide.Export(str(path),'PNG',width,height)
                        return np.array(Image.open(path).convert('RGB'))
                    render('成品')
                    hidden=[s for s in top.values() if s.Visible]
                    for s in hidden:s.Visible=False
                    blank=render('仅页面背景')
                    for s in hidden:s.Visible=True
                    first_left=parent.Left
                    local={k:[s.Left,s.Top] for k,s in members.items()}
                    siblings={k:[s.Left,s.Top] for k,s in top.items() if k!=parent.Name}
                    parent.Left=first_left-deck.PageSetup.SlideWidth
                    delta=parent.Left-first_left
                    follows=all(abs((s.Left-local[k][0])-delta)<.2 and abs(s.Top-local[k][1])<.2 for k,s in members.items())
                    moved=render('移开整组')
                    parent.Left=first_left
                    shape_ids=[k for k in members if nodes[k]['type']=='shape']
                    carrier_id=max(shape_ids,key=lambda k:np.prod(nodes[k]['bbox'][2:]))
                    x,y,w,h=map(round,nodes[carrier_id]['bbox'])
                    cleared=moved[y+3:y+h-3,x+3:x+w-3]
                    ground=blank[y+3:y+h-3,x+3:x+w-3]
                    residual=int((abs(cleared.astype(int)-ground).max(2)>10).sum())
                    pic_ids=[k for k in members if nodes[k]['type']=='image']
                    picture=max(pic_ids,key=lambda k:np.prod(nodes[k]['bbox'][2:])) if pic_ids else None
                    photo_residual=None
                    if picture:
                        p=members[picture]
                        left=p.Left
                        p.Left=left-deck.PageSetup.SlideWidth
                        exposed=render('移开图片')
                        # Compare to the same source site with the picture actually hidden.
                        p.Left=left;p.Visible=False
                        expected=render('隐藏图片参照')
                        p.Visible=True
                        px,py,pw,ph=map(round,nodes[picture]['bbox'])
                        photo_residual=int((abs(exposed[py:py+ph,px:px+pw].astype(int)-expected[py:py+ph,px:px+pw]).max(2)>10).sum())
                    carrier=members[carrier_id]
                    carrier.Fill.Solid();carrier.Fill.ForeColor.RGB=0xAAEE99
                    texts={k:s for k,s in members.items() if nodes[k]['type']=='text'}
                    has_photo=photo_size(group)>.025*width*height
                    if has_photo:
                        target=min(texts,key=lambda k:nodes[k]['bbox'][1])
                    else:
                        target=max(texts,key=lambda k:('text_override_reason' in nodes[k],nodes[k].get('wrap',False),len(str(texts[k].TextFrame.TextRange.Text))))
                    s=texts[target]
                    original=str(s.TextFrame.TextRange.Text)
                    source_values={k:str(obj.TextFrame.TextRange.Text) for k,obj in texts.items()}
                    replacement= ('服务领域' if has_photo else original.rstrip('。？！？，')+'，并核对结果。')
                    s.TextFrame.TextRange.Text=replacement
                    render('组内改色改字')
                    other_texts=all(str(obj.TextFrame.TextRange.Text)==source_values[k] for k,obj in texts.items() if k!=target)
                    neighbors=all(abs(top[k].Left-xy[0])<.2 and abs(top[k].Top-xy[1])<.2 for k,xy in siblings.items())
                    inside_box=(s.TextFrame.TextRange.BoundHeight<=s.Height+2 and s.TextFrame.TextRange.BoundWidth<=s.Width+2)
                    content=None
                    if references:
                        norm=lambda t:''.join(t.split())
                        from product.parallel_office import leaf_shapes
                        native=Counter(norm(original if obj.Name==target else str(obj.TextFrame.TextRange.Text))
                                       for obj in leaf_shapes(slide.Shapes) if obj.Name in nodes and nodes[obj.Name]['type']=='text')
                        prior=Counter(norm(t) for t in references[index-1]['original_texts'].values())
                        content=list((prior-native).elements())
                    edits.append({'group':group['id'],'members':len(members),'target':target,'replaced':replacement,
                        'carrier':carrier_id,'picture':picture,'all_members_moved':follows,'siblings_stable':neighbors,
                        'other_text_unchanged':other_texts,'replacement_inside_frame':inside_box,
                        'original':original,'rendered_lines':s.TextFrame.TextRange.Lines().Count,
                        'frame_pt':[s.Width,s.Height],'edited_text_bounds_pt':[s.TextFrame.TextRange.BoundWidth,s.TextFrame.TextRange.BoundHeight],
                        'carrier_site_nonbackground_pixels':residual,'moved_vs_hidden_image_difference':photo_residual,
                        'baseline_text_missing':content,'objects':dict(Counter(n['type'] for n in nodes.values()))})
                deck.Save()
            finally:
                deck.Close()
            deck=app.Presentations.Open(str(changed),True,False,False)
            try:
                for i,r in enumerate(edits,1):
                    group=shapes(deck.Slides(i).Shapes)[r['group']]
                    children=shapes(group.GroupItems)
                    r['save_reopen']=str(children[r['target']].TextFrame.TextRange.Text)==r['replaced'] and children[r['carrier']].Fill.ForeColor.RGB==0xAAEE99
            finally:
                deck.Close()
        finally:
            if not app.Presentations.Count:app.Quit()
    write(output/'office.json',{'pptx_sha256':digest(ppt),'edit_copy_sha256':digest(changed),'pages':edits})
    return edits


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('composition',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    record=json.loads(args.composition.read_text('utf8'))
    baseline=json.loads(args.baseline.read_text('utf8')) if args.baseline else None
    print(json.dumps(check(record['composition'],args.output.resolve(),baseline),ensure_ascii=False))
