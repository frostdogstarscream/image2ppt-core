"""Offline source-reviewed page typography, using the existing compiler.

Input is a conversation-authored page/group map. No OCR or provider calls.
The output directory is a build workspace; delivery copies only PPTX/previews.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import numpy as np
from PIL import Image
from typography_plan import read,write,digest,font_manifest
from typography_fit import rank_group_candidates,glyph
from typography_groups import paragraphs,choose_groups,verify_groups,immutable

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from image2ppt.compiler import ArtifactBackend,execute


def text_extent(n,ir,manifest):
    scale=ir['canvas']['target_px'][0]/ir['canvas']['source_px'][0]
    widths=[]; height=0
    paras=n.get('paragraphs') or [n]
    for i,p in enumerate(paras):
        width=0;h=0
        for r in p.get('runs') or [{'text':p['text']}]:
            s={**p,**r};f=s['font']
            ink,advance=glyph(r['text'],{'font_family':f['ea'],'font_latin':f['latin'],
                        'bold':s.get('bold',False)},manifest,f['size_pt']/.75/scale)
            width+=advance;h=max(h,ink.shape[0])
        widths.append(width+p.get('left_indent_px',0))
        height+=p.get('line_spacing_pt',p['font']['size_pt']*1.2)/.75/scale if i<len(paras)-1 else h
    return max(widths),height


def preflight(before,after,group,manifest):
    """Capacity filter only. Final actual-file validation remains mandatory."""
    old={n['id']:n for n in before['nodes']};new={n['id']:n for n in after['nodes']}
    for key in {k.rsplit('/p',1)[0] for k in group['members']}:
        try: w,h=text_extent(new[key],after,manifest)
        except ValueError:return False
        try: ow,oh=text_extent(old[key],before,manifest)
        except ValueError:ow,oh=old[key]['bbox'][2:]
        bw,bh=new[key]['bbox'][2:]
        if w>max(bw+3,ow+1) or h>max(bh+3,oh+1):return False
    return True


def verify_pptx(pptx,irs):
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
        'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    total=0
    with ZipFile(pptx) as z:
        for i,ir in enumerate(irs,1):
            root=ET.fromstring(z.read(f'ppt/slides/slide{i}.xml'))
            shapes={s.find('p:nvSpPr/p:cNvPr',ns).get('name'):s for s in root.findall('.//p:sp',ns)}
            for n in ir['nodes']:
                if n['type']!='text':continue
                paras=shapes[n['id']].findall('p:txBody/a:p',ns);planned=n.get('paragraphs') or [n]
                if len(paras)!=len(planned):raise ValueError('Paragraph count changed')
                for actual,p in zip(paras,planned):
                    runs=actual.findall('a:r',ns);spec=p.get('runs') or [{'text':p['text']}]
                    if len(runs)!=len(spec):raise ValueError('Run count changed')
                    for r,s in zip(runs,spec):
                        expected={**p,**s};rp=r.find('a:rPr',ns)
                        if r.find('a:t',ns).text!=s['text']:raise ValueError('Text changed')
                        if rp.get('b')!=str(int(expected['bold'])):raise ValueError('Weight changed')
                        if rp.get('sz')!=str(round(expected['font']['size_pt']*100)):raise ValueError('Size changed')
                        if rp.find('a:solidFill/a:srgbClr',ns).get('val')!=expected['color']:raise ValueError('Color changed')
                        for script in ('ea','latin'):
                            if rp.find('a:'+script,ns).get('typeface')!=expected['font'][script]:raise ValueError('Face changed')
                total+=1
    return total


def verify_rendered_change(before,after,old_background,new_background,ir):
    """Actual-file gate for a normal (not stress-edited) typography revision.

    Inputs must use the same renderer/size. Backgrounds are independent copies
    with all native text deleted. The 3 px raster tolerance matches r7 checks.
    This proves preservation/containment; it does not replace source viewing or
    independently establish absence of pre-existing in-frame collisions.
    """
    arrays=[np.asarray(a,dtype=np.int16) for a in (before,after,old_background,new_background)]
    a,b,old,new=arrays
    expected=tuple(reversed(ir['canvas']['source_px']))+(3,)
    if any(x.shape!=expected for x in arrays):raise ValueError('Mismatched actual-file previews')
    if (np.max(abs(old-new),axis=2)>1).any():raise ValueError('Non-text rendering changed')
    allowed=np.zeros(a.shape[:2],bool)
    for n in ir['nodes']:
        if n['type']!='text':continue
        x,y,w,h=n['bbox']
        allowed[max(0,math.floor(y)-3):min(a.shape[0],math.ceil(y+h)+3),
                max(0,math.floor(x)-3):min(a.shape[1],math.ceil(x+w)+3)]=True
    allowed|=np.max(abs(a-old),axis=2)>1
    if ((np.max(abs(b-a),axis=2)>1)&~allowed).any():raise ValueError('Typography changed protected pixels')
    return True


def prepare(baseline,plan,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    pages=sorted(plan['pages']); baseline=Path(baseline)
    originals=[read(baseline/'pages'/p/'scene/slideir.json') for p in pages]
    manifest=font_manifest([n['text'] for ir in originals for n in ir['nodes'] if n['type']=='text'])
    write(directory/'fonts.json',manifest)
    files=[];results=[]
    for page,ir in zip(pages,originals):
        origin=baseline/'pages'/page/'scene'
        groups=copy.deepcopy(plan['pages'][page])
        source=np.array(Image.open(baseline/'pages'/page/'source.png').convert('RGB'))
        background=np.array(Image.open(baseline/'pages'/page/'deleted.png').convert('RGB'))
        for g in groups:
            g['candidates']=rank_group_candidates(ir,g['members'],g.pop('families'),g.pop('bold'),
                                                  g.pop('size_pt'),manifest,source,background)
        selected,accepted,rejected=choose_groups(ir,groups,manifest,
            lambda a,b,g:preflight(a,b,g,manifest))
        if immutable(ir)!=immutable(selected):raise ValueError('Protected scene changed')
        target=directory/'pages'/page
        shutil.copytree(origin,target)
        file=target/'slideir.json';write(file,selected)
        write(target/'authoring.json',{'tool':'Codex source-reviewed page typography',
            'model_id':'current-conversation','ir_sha256':digest(file),
            'decisions':['Shared page roles and explicit emphasis; no content or geometry edits']})
        files.append(str(file.resolve()))
        results.append({'page':page,'groups':groups,'accepted':accepted,'rejected':rejected})
        print('prepared',page,'groups',len(groups),'accepted',len(accepted),'retained',rejected,flush=True)
    job={'files':files,'results':results,'baseline':str(baseline.resolve())}
    write(directory/'job.json',job)
    return job


def compose(directory):
    directory=Path(directory);job=read(directory/'job.json')
    backend=ArtifactBackend(REPO,1800)
    # The fixed finalizer requires the final build inside its checkout.
    result=execute([backend.node,REPO/'ppt-rebuild/compiler_bridge.mjs','compose',REPO/'outputs'/('typography-'+directory.name),*job['files']],REPO,1800)
    irs=[read(p) for p in job['files']]
    for ir,r in zip(irs,job['results']):verify_groups(ir,r['groups'],r['accepted'])
    result['checked_text_objects']=verify_pptx(result['pptx'],irs)
    write(directory/'compiled.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','compose'])
    parser.add_argument('--baseline');parser.add_argument('--plan');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.baseline,read(args.plan),args.output)
    else:compose(args.output)
