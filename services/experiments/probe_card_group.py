"""Exercise one independently specified card in the same PowerPoint document across four saved edits."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import json
import shutil
import sys

from PIL import Image
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'product'))
from product.materialize import digest, write
from product.resource_lease import lease


@contextmanager
def document(app, path, *, readonly):
    from product.parallel_office import close_probe_document, open_probe_document
    pres = open_probe_document(app,path,read_only=readonly,checked=True)
    try:
        yield pres
    finally:
        close_probe_document(app,pres,path,checked=True)


def run(pptx, out, *, page, face, caption, picture, replacement, source_kind):
    import win32com.client
    out.mkdir(parents=True,exist_ok=False)
    edited = out/'card-edits.pptx'
    shutil.copyfile(pptx,edited)
    report = {'source_kind':source_kind, 'source_sha256':digest(pptx), 'page':page,
              'expected_members':[face,caption,picture], 'edits':[]}
    actions = ('move_group','edit_caption','recolor_face','delete_picture')
    group_name = None
    delta = 8.0
    fill_rgb = 0x44DCFC

    def snapshot(pres, image, *, deleted=False):
        slide = pres.Slides(page)
        group = slide.Shapes(group_name)
        children = {s.Name:s for s in group.GroupItems}
        for key in [face,caption] + ([] if deleted else [picture]):
            if key not in children:
                raise ValueError(f'expected card member missing: {key}')
        if children[face].Type != 1 or not children[caption].HasTextFrame or (
                not deleted and children[picture].Type != 13):
            raise ValueError('card member type mismatch')
        width = 1440
        slide.Export(str(image), 'PNG', width, round(width*pres.PageSetup.SlideHeight/pres.PageSetup.SlideWidth))
        return {'children':{k:[float(v.Left),float(v.Top),float(v.Width),float(v.Height)]
                            for k,v in children.items()},
                'other_shapes':{s.Name:[float(s.Left),float(s.Top)] for s in slide.Shapes if s.Name!=group_name},
                'text':children[caption].TextFrame.TextRange.Text, 'fill':children[face].Fill.ForeColor.RGB,
                'preview':image, 'page_scale':width/float(pres.PageSetup.SlideWidth)}

    with lease('office'):
        app = win32com.client.DispatchEx('PowerPoint.Application')
        try:
            with document(app,edited,readonly=True) as pres:
                matches = [s.Name for s in pres.Slides(page).Shapes
                           if s.Type==6 and {face,caption,picture} <= {v.Name for v in s.GroupItems}]
                if len(matches)!=1:
                    raise ValueError('expected one already grouped native card')
                group_name = matches[0]
                report['group'] = group_name
                before = snapshot(pres,out/'0-before.png')
            for index,action in enumerate(actions,1):
                with document(app,edited,readonly=False) as pres:
                    group = pres.Slides(page).Shapes(group_name)
                    if action == 'move_group':
                        group.IncrementLeft(delta)
                    elif action == 'edit_caption':
                        group.GroupItems(caption).TextFrame.TextRange.Text = replacement
                    elif action == 'recolor_face':
                        group.GroupItems(face).Fill.Solid()
                        group.GroupItems(face).Fill.ForeColor.RGB = fill_rgb
                    else:
                        group.GroupItems(picture).Delete()
                    pres.Save()
                image = out/f'{index}-{action}.png'
                with document(app,edited,readonly=True) as pres:
                    after = snapshot(pres,image,deleted=action=='delete_picture')
                passed = before['other_shapes']==after['other_shapes']
                row = {'action':action,'saved_and_reopened':True,'neighbor_positions_preserved':passed,
                       'preview':image.name}
                if action == 'move_group':
                    passed &= all(abs(after['children'][k][0]-p[0]-delta)<.1 and
                                  np.allclose(after['children'][k][1:],p[1:],atol=.1)
                                  for k,p in before['children'].items())
                    row['member_offsets_pt'] = {k:round(p[0]-before['children'][k][0],2)
                                                for k,p in after['children'].items()}
                else:
                    passed &= all(np.allclose(before['children'][k],p,atol=.1) for k,p in after['children'].items())
                    passed &= after['text']==replacement
                    if action in ('recolor_face','delete_picture'):
                        passed &= after['fill']==fill_rgb
                    if action == 'delete_picture':
                        passed &= picture not in after['children']
                        x,y,w,h = before['children'][picture]
                        scale = after['page_scale']
                        inset = .15
                        roi = np.array(Image.open(image).convert('RGB').crop(tuple(round(p*scale) for p in
                            [x+w*inset,y+h*inset,x+w*(1-inset),y+h*(1-inset)])),dtype=int)
                        color = [fill_rgb & 255,(fill_rgb>>8)&255,(fill_rgb>>16)&255]
                        fraction = float(np.mean(np.max(np.abs(roi-color),axis=-1) <= 12)) if roi.size else 0
                        passed &= fraction>=.95
                        row['new_fill_visible_in_picture_interior'] = fraction
                # Compare exactly the same document before/after each operation. The crop
                # includes both positions of a moved object and the export antialias margin.
                a = np.asarray(Image.open(before['preview']).convert('RGB'),dtype=int)
                b = np.asarray(Image.open(image).convert('RGB'),dtype=int)
                changed = np.abs(a-b).max(axis=2)>12
                exclusions = ([*before['children'].values(), *after['children'].values()] if action=='move_group'
                              else [before['children'][{'edit_caption':caption,'recolor_face':face,
                                                         'delete_picture':picture}[action]]])
                for x,y,w,h in exclusions:
                    scale = after['page_scale']
                    changed[max(0,int(y*scale)-3):int((y+h)*scale)+4,
                            max(0,int(x*scale)-3):int((x+w)*scale)+4]=False
                row['outside_changed_pixels'] = int(changed.sum())
                passed &= not changed.any()
                row.update(passed=bool(passed), pptx_sha256=digest(edited))
                report['edits'].append(row)
                write(out/'report.json',report)
                before = after
        except Exception as exc:
            report['error'] = str(exc)
            raise
        finally:
            if app.Presentations.Count==0:
                app.Quit()
            write(out/'report.json',report)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pptx', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--page',type=int,default=1)
    for name in ('face','caption','picture','replacement'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--source-kind', choices=['fresh-upload','archived-scene-replay','synthetic'], required=True)
    args = parser.parse_args()
    result = run(args.pptx.resolve(),args.out.resolve(),**{k:v for k,v in vars(args).items() if k not in ('pptx','out')})
    print(json.dumps(result['edits'],ensure_ascii=False))
    sys.exit(0 if all(r['passed'] for r in result['edits']) else 1)
