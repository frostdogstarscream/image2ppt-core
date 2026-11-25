from copy import deepcopy

import numpy as np
from PIL import Image

from product.runtime import graphic_plate, host_baseline
from product.materialize import digest


def test_only_numbered_art_and_nontext_edges_contribute(tmp_path,monkeypatch):
    source = np.full((90,120,3),245,np.uint8)
    source[12:43,10:50] = [5,60,160]
    source[16:42,80:105] = [20,90,230]
    source[64:75,14:55] = [10,10,10]
    path = tmp_path/'source.png'
    Image.fromarray(source).save(path)
    cleaned = source.copy()
    cleaned[10:46,8:52] = 245
    monkeypatch.setattr(graphic_plate,'named_plate',lambda deck,page:Image.fromarray(cleaned))
    marks = [{'id':i,'label':'image','bbox':b} for i,b in (
        ('photo',[7,8,48,40]),('badge',[76,12,33,36]),('letters',[10,58,50,22]))]
    plans = [{'id':m['id'],'route':'asset'} for m in marks]
    audit = graphic_plate.measure('deck',1,{'path':path,'source_sha256':digest(path)},marks,plans,
        [{'bbox':[12,62,45,16]}],{'newphoto':{'mark_id':'photo','kind':'asset'}},{'newphoto'},'sha')
    rows = {r['id']:r for r in audit['rows']}
    assert rows['photo']['status']=='source_edges_removed' and rows['photo']['package_object_ids']==['newphoto']
    assert rows['badge']['status']=='retained' and rows['badge']['package_object_ids']==[]
    assert rows['letters']['status']=='unknown'


def test_retained_picture_can_be_downloaded_only_as_bound_partial():
    plate = {'status':'MEASURED','source_sha256':'source','pptx_sha256':'deck','rows':[]}
    graphic = {**plate,'rows':[{'id':'photo','status':'retained'}]}
    page = {'page':1,'source_sha256':'source','plate_audit':plate,'graphic_plate':graphic,
            'selected_editables':['title'],'passed_edits':['title/move-scale','title/delete']}
    score = {'verified':True,'office_status':'PASS','pptx_sha256':'deck','pages':[page]}
    defects = host_baseline.delivery_defects(score)
    assert defects[0]['reasons']==['GRAPHIC_STILL_ON_BACKGROUND']
    assert defects[0]['retained_graphics']==graphic['rows']
    assert host_baseline._verified_partial_copy(score,'deck',['source'],defects)
    changed = deepcopy(score)
    changed['pages'][0]['graphic_plate']['pptx_sha256']='other'
    assert not host_baseline._verified_partial_copy(changed,'deck',['source'],defects)
