"""Freeze an existing automatic run and audit its actual final package."""
import argparse,json,shutil,zipfile,io,xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from PIL import Image
from product.materialize import digest,write


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def pptx_inventory(pptx):
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    import posixpath
    result=[]
    with zipfile.ZipFile(pptx) as z:
        slides=sorted((n for n in z.namelist() if __import__('re').fullmatch(r'ppt/slides/slide\d+.xml',n)),key=lambda n:int(n.split('slide')[-1].split('.')[0]))
        for i,name in enumerate(slides,1):
            tree=ET.fromstring(z.read(name));rels=ET.fromstring(z.read(f'ppt/slides/_rels/slide{i}.xml.rels'));mapping={r.get('Id'):r.get('Target') for r in rels}
            row={'page':i,'native_text':0,'native_shapes':0,'images':0,'transparent_png':0,'image_details':[]}
            for sp in tree.findall('.//p:sp',ns):
                if any((t.text or '').strip() for t in sp.findall('.//a:t',ns)):row['native_text']+=1
                else:row['native_shapes']+=1
            for pic in tree.findall('.//p:pic',ns):
                label=pic.find('p:nvPicPr/p:cNvPr',ns).get('name');target=mapping[pic.find('.//a:blip',ns).get('{'+ns['r']+'}embed')]
                media=target.lstrip('/') if target.startswith('/') else posixpath.normpath('ppt/slides/'+target)
                im=Image.open(io.BytesIO(z.read(media)));alpha=im.convert('RGBA').getchannel('A').getextrema()
                transparent=im.format=='PNG' and alpha[0]<255 and alpha[1]>0
                row['images']+=1;row['transparent_png']+=int(transparent)
                row['image_details'].append({'id':label,'transparent_png':transparent,'alpha_range':alpha})
            row['full_raster']=row['images']==1 and row['native_text']==row['native_shapes']==0
            result.append(row)
    return result


def freeze(previous,output,commit):
    previous=Path(previous).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    result=load(previous/'result.json');files={};pages=[]
    def save(p):
        p=Path(p).resolve()
        rel=p.relative_to(previous);dest=output/'frozen'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        files[str(p)]={'sha256':digest(p),'copy':str(dest),'copy_sha256':digest(dest)}
    for p in previous.rglob('*.json'):
        if any(part.startswith('compile') for part in p.relative_to(previous).parts):continue
        save(p)
    for p in (previous/'delivery').iterdir():
        if p.is_file():save(p)
    actual=pptx_inventory(previous/'delivery/result.pptx')
    for i,irpath in enumerate(result['slide_irs'],1):
        irpath=Path(irpath);g=load(previous/f'page-{i:02d}/barrier-1.json');ir=load(irpath)
        final=load(irpath.parent/'transactions.json');attempts=[]
        for p in (previous/f'page-{i:02d}').glob('*/transactions.json'):
            rows=load(p);attempts.append({'path':str(p),'counts':dict(Counter(r['state'] for r in rows)),'fallback_reasons':dict(Counter(r.get('reason','unknown') for r in rows if r['state']=='FALLBACK'))})
        for p in [irpath,*[irpath.parent/a['path'] for a in ir['assets']]]:
            if str(p.resolve()) not in files:save(p)
        pages.append({'page':i,'proposed':{'text':len(g['text']['texts']),'asset':len(g['asset']['objects']),'surface':len(g['surface']['shapes'])},
            'ownership':dict(Counter(g['ownership'].values())),'actual_pptx':actual[i-1],
            'final_fallback_reasons':dict(Counter(r.get('reason','unknown') for r in final if r['state']=='FALLBACK')),
            'attempts':attempts,'final_ir':str(irpath),'final_transactions':final})
    totals={k:sum(p['actual_pptx'][k] for p in pages) for k in ('native_text','native_shapes','images','transparent_png')}
    summary={'schema':'RegionFixBaseline/1','requested_code_baseline':commit,'artifact_job':str(previous),
        'provenance_note':'Frozen original delivered r10 run; its provenance has no exact generation git SHA. Do not infer it from requested code revision.',
        'pages':pages,'totals':totals,'full_raster_pages':[p['page'] for p in pages if p['actual_pptx']['full_raster']],
        'files':files,'human_acceptance':False}
    write(output/'baseline.json',summary);print(json.dumps({'totals':totals,'full_raster_pages':summary['full_raster_pages'],'frozen_files':len(files)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('previous');p.add_argument('output');p.add_argument('--commit',default='3e97fac448c864138b8c657702ebd00ab0e6597f');a=p.parse_args();freeze(a.previous,a.output,a.commit)
