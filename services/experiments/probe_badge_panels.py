"""Frozen Xuzhou-03 edit cases. Run on a copy of an uploaded job's final PPTX."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import win32com.client

from probe_card_caption import lease, opened, render
from probe_fields import box, intersects, named_shapes, name_text


PANELS = [(58,191,821,171), (58,385,821,174), (58,578,821,172)]
DISCS = [(89,222,113,114), (89,417,113,113), (86,607,119,119)]
COLORS = [(204,138,174), (127,197,149), (144,179,221)]


def office_rgb(color):
    return color[0] + 256*color[1] + 65536*color[2]


def locate(presentation, expected, kind, geometry=None, inset=False):
    choices = []
    for name, shape in named_shapes(presentation.Slides(1).Shapes):
        if shape.Type != kind or (geometry is not None and int(shape.AutoShapeType) != geometry):
            continue
        bounds = box(shape,presentation.PageSetup.SlideWidth,presentation.PageSetup.SlideHeight)
        overlap = intersects(bounds,expected)
        union = bounds[2]*bounds[3] + expected[2]*expected[3] - overlap
        iou = overlap/max(1,union)
        if inset:
            if not (.15 < iou < .83 and overlap >= .985*bounds[2]*bounds[3]):
                continue
        elif iou < .85:
            continue
        choices.append((iou,name))
    return max(choices)[1] if len(choices)==1 else None


def central_patch(image, disc):
    x,y,w,h = disc
    return image[y+h//4:y+3*h//4, x+w//4:x+3*w//4]


def probe(deck, out):
    out.mkdir(parents=True,exist_ok=False)
    rows = [{} for _ in PANELS]
    report = {'page':'Xuzhou-03','rows':rows, 'saved_reopened':[], 'footer':False}
    frames = {}
    previous = deck
    with lease('office'):
        app = win32com.client.DispatchEx('PowerPoint.Application')
        try:
            for action in ('original','recolor','delete-symbol','move-disc','edit-footer'):
                path = out/(action+'.pptx')
                shutil.copyfile(previous,path)
                with opened(app,path) as presentation:
                    shapes = dict(named_shapes(presentation.Slides(1).Shapes))
                    if action=='original':
                        for i,row in enumerate(rows):
                            row.update(panel=locate(presentation,PANELS[i],1,51),
                                       disc=locate(presentation,DISCS[i],1,9),
                                       symbol=locate(presentation,DISCS[i],13,inset=True))
                        pages = [name for name,shape in shapes.items()
                                 if shape.Top > .90*presentation.PageSetup.SlideHeight and name_text(shape).strip()=='03']
                        report['footer_shape'] = pages[0] if len(pages)==1 else None
                    for i,row in enumerate(rows):
                        if action=='recolor' and row['panel']:
                            fill = shapes[row['panel']].Fill
                            fill.Solid()
                            fill.ForeColor.RGB = office_rgb(COLORS[i])
                        elif action=='delete-symbol' and row['symbol']:
                            shapes[row['symbol']].Delete()
                        elif action=='move-disc' and row['disc']:
                            shapes[row['disc']].Left += presentation.PageSetup.SlideWidth*.31
                    if action=='edit-footer' and report['footer_shape']:
                        shapes[report['footer_shape']].TextFrame.TextRange.Text = '04'
                    presentation.Save()
                with opened(app,path) as presentation:
                    frames[action] = render(presentation,out/(action+'.png'))
                    shapes = dict(named_shapes(presentation.Slides(1).Shapes))
                    for i,row in enumerate(rows):
                        if action=='recolor':
                            row['persisted_fill'] = bool(row['panel'] and shapes[row['panel']].Fill.ForeColor.RGB==office_rgb(COLORS[i]))
                        elif action=='delete-symbol':
                            row['persisted_delete'] = bool(row['symbol'] and row['symbol'] not in shapes)
                        elif action=='move-disc':
                            row['persisted_move'] = bool(row['disc'] and shapes[row['disc']].Left > presentation.PageSetup.SlideWidth*.32)
                    if action=='edit-footer' and report['footer_shape']:
                        report['footer'] = name_text(shapes[report['footer_shape']]).strip()=='04'
                    report['saved_reopened'].append(action)
                previous = path
        finally:
            app.Quit()
    for i,row in enumerate(rows):
        x,y,w,h = PANELS[i]
        rgb = np.array(COLORS[i])
        fill_sample = frames['recolor'][y+int(h*.68):y+int(h*.83),x+int(w*.84):x+int(w*.9)]
        row['visible_fill'] = float((abs(fill_sample-rgb).max(2)<12).mean())>.95
        before = central_patch(frames['recolor'],DISCS[i])
        deleted = central_patch(frames['delete-symbol'],DISCS[i])
        face = np.median(deleted.reshape(-1,3),axis=0)
        row['clean_disc_after_delete'] = bool(np.percentile(abs(deleted-face).max(2),99)<12
                                               and (abs(before-deleted).max(2)>40).mean()>.045)
        moved = central_patch(frames['move-disc'],DISCS[i])
        row['new_fill_at_old_disc'] = float((abs(moved-rgb).max(2)<12).mean())>.97
    outside = np.ones(frames['original'].shape[:2], bool)
    for x,y,w,h in PANELS:
        outside[y-4:y+h+4,x-4:x+w+4] = False
    report['neighbor_changes'] = int(((abs(frames['original']-frames['move-disc']).max(2)>24)&outside).sum())
    report['passed_rows'] = sum(all(row[k] for k in ('persisted_fill','visible_fill','persisted_delete',
                 'clean_disc_after_delete','persisted_move','new_fill_at_old_disc')) for row in rows)
    (out/'receipt.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),'utf8')
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('deck',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    result = probe(args.deck.resolve(),args.output.resolve())
    print(json.dumps(result,ensure_ascii=False))
    raise SystemExit(0 if result['passed_rows']==3 and result['footer'] and result['neighbor_changes']==0 else 1)
