import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

from lxml import etree
import pytest
from PIL import Image
from product import experiment
from product.experiment_svg import attach, A, P, REL, OFF, CT, SVG


def test_local_command_cannot_replace_ssh(tmp_path, monkeypatch):
    cfg=tmp_path/'worker.json'
    cfg.write_text(json.dumps({'omni':['python','worker.py']}))
    monkeypatch.setenv('IMAGE2PPT_EXPERIMENT_CONFIG',str(cfg))
    result=experiment.invoke('omni',Image.new('RGB',(20,20)),tmp_path/'call')
    assert result['status']=='unavailable'
    assert 'OpenSSH' in result['reason']


def test_marks_keep_caption_separate_from_parent(tmp_path):
    f=tmp_path/'source.png'
    Image.new('RGB',(100,100)).save(f)
    layout={'regions':[{'id':'picture','bbox':[3,4,90,70],'label':'image'}]}
    fields=[{'id':'t-001','bbox':[10,52,50,14],'text':'caption','ocr_line_ids':['l-1']}]
    rows=experiment.mark_page(f,layout,fields,tmp_path)
    by_id={r['id']:r for r in rows}
    assert by_id['t-001']['parent']=='picture'
    assert by_id['picture']['intersecting_ocr_ids']==['l-1']
    assert 'ocr_ids' not in by_id['picture']


def test_svg_uses_one_picture_and_verified_png(tmp_path):
    png=io.BytesIO()
    Image.new('RGBA',(3,3),'red').save(png,format='PNG')
    png=png.getvalue()
    svg=b'<svg xmlns="http://www.w3.org/2000/svg" width="3" height="3"><rect width="3" height="3" fill="red"/></svg>'
    (tmp_path/'test.svg').write_bytes(svg)
    manifest=tmp_path/'vector-assets.json'
    manifest.write_text(json.dumps({'graphic':{'path':'test.svg','sha256':hashlib.sha256(svg).hexdigest(),
                                              'png_sha256':hashlib.sha256(png).hexdigest()}}))
    deck=tmp_path/'test.pptx'
    with ZipFile(deck,'w') as z:
        z.writestr('[Content_Types].xml',f'<Types xmlns="{CT}"/>')
        z.writestr('ppt/slides/slide1.xml',f'<p:sld xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{OFF}"><p:pic><p:cNvPr name="graphic"/><a:blip r:embed="rId1"/></p:pic></p:sld>')
        z.writestr('ppt/slides/_rels/slide1.xml.rels',f'<Relationships xmlns="{REL}"><Relationship Id="rId1" Target="../media/asset.png"/></Relationships>')
        z.writestr('ppt/media/asset.png',png)
    attach(deck,[tmp_path/'slideir.json'])
    with ZipFile(deck) as z:
        root=etree.fromstring(z.read('ppt/slides/slide1.xml'))
        assert len(list(root.iter('{%s}pic'%P)))==1
        assert root.find('.//{%s}svgBlip'%SVG) is not None
        assert z.read('ppt/media/asset.png')==png

