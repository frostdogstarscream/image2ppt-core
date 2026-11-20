"""Four development edits for workbench-08; coordinates are used only after delivery."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import win32com.client

from probe_card_caption import opened, render, select_target
from probe_region_text import HEADER, VERTICAL_RULE, DIVIDER, native_shape, region, changed_fraction
from probe_fields import box, intersects, name_text, named_shapes
from product.resource_lease import lease

SOURCE = '3ed7ccbde6a908d2200290556b116005459f1b0324ca4df7dcb9cbb31382d48e'
CASES = [
    ('footer', [1,'底部标题',[97,645,359,39],'text','本页已完成编辑'], [[78,646,12,35]]),
    ('department', [1,'覆盖部门',[283,489,87,30],'text','服务部门'], [[244,482,38,43]]),
    ('title', HEADER, [VERTICAL_RULE,DIVIDER]),
    ('diagram', [1,'趋势图',[1157,252,429,244],'move',None], [[1125,168,237,65],[1158,506,390,67]]),
]


def measure(app, source, deck, out, key, case, neighbors):
    copy = out/(key+'.pptx')
    shutil.copyfile(deck,copy)
    with opened(app,copy) as p:
        before = render(p,out/(key+'-before.png'))
        if key == 'diagram':
            candidates = []
            for name, shape in named_shapes(p.Slides(1).Shapes):
                bounds = box(shape,p.PageSetup.SlideWidth,p.PageSetup.SlideHeight)
                both = intersects(bounds,case[2])
                ratio = both / max(1,bounds[2]*bounds[3]+case[2][2]*case[2][3]-both)
                if shape.Type in (6,13) and ratio >= .65:
                    candidates.append((ratio,name,shape))
            found = max(candidates,key=lambda row:row[0],default=None)
        else:
            found = select_target(p, case)
        if not found:
            return {'status':'NO_TARGET_OBJECT'}
        _,name,shape = found
        moving = key in ('title','diagram')
        if moving:
            # Move across the slide, outside the old position and protected neighbors.
            shape.Left = p.PageSetup.SlideWidth * (.30 if key=='diagram' else .55)
            shape.Top = p.PageSetup.SlideHeight * .05
            expected = (shape.Left,shape.Top)
        else:
            shape.TextFrame.TextRange.Text = case[4]
        p.Save()
    with opened(app,copy) as p:
        shape = native_shape(p,name)
        saved = shape is not None and (max(abs(shape.Left-expected[0]),abs(shape.Top-expected[1]))<1
                                       if moving else name_text(shape)==case[4])
        changed = render(p,out/(key+'-edited.png'))
    bare = changed
    if not moving:
        erased_file = out/(key+'-without-text.pptx')
        shutil.copyfile(copy,erased_file)
        with opened(app,erased_file) as p:
            native_shape(p,name).Delete()
            p.Save()
        with opened(app,erased_file) as p:
            bare = render(p,out/(key+'-without-text.png'))
    patch = region(bare,case[2])
    reference = region(before,case[2])
    def blue(pixels):
        return (pixels[:,:,2]-pixels[:,:,0]>24) & (pixels[:,:,0]<180)
    old = blue(reference)
    retained = float((blue(patch)&old).sum()/max(1,old.sum()))
    adjacent = max(changed_fraction(before,step,b) for step in (changed,bare) for b in neighbors)
    # Keep a normal-size panel for independent review of faint neutral contours.
    from PIL import Image
    Image.fromarray(np.hstack((reference,patch)).astype('uint8')).save(out/(key+'-origin.png'))
    passed = saved and retained<.06 and adjacent<.02 and old.any()
    return {'status':'PASS_AUTOMATED' if passed else 'REVIEW_REQUIRED','saved_reopened':bool(saved),
            'remaining_blue_fraction':retained,'neighbor_change':adjacent,'object':name,
            'normal_size_review':'UNKNOWN'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('source','deck','out'):
        parser.add_argument(arg,type=Path)
    args=parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest()!=SOURCE:
        raise ValueError('Unexpected development image')
    args.out.mkdir(parents=True,exist_ok=False)
    with lease('office'):
        app=win32com.client.DispatchEx('PowerPoint.Application')
        try:
            result={key:measure(app,args.source,args.deck,args.out,key,case,neighbors) for key,case,neighbors in CASES}
        finally:
            if not app.Presentations.Count:
                app.Quit()
    (args.out/'receipts.json').write_text(json.dumps({'deck_sha256':hashlib.sha256(args.deck.read_bytes()).hexdigest(),
        'cases':result},ensure_ascii=False,indent=2),'utf8')
