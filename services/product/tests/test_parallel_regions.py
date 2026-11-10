import json
import numpy as np
import pytest
from PIL import Image
from test_parallel import branches
from product.contracts import Text,Style,Shape
from product.parallel_contracts import Relation
from product.parallel_fusion import fuse
from product.parallel_rebuild import build
from product.parallel_regions import regions,retain_atomic
from product.parallel_region_gate import region_ready


def test_thirty_same_style_texts_one_failure_is_local(tmp_path,monkeypatch):
    import product.parallel_rebuild as mod
    p=tmp_path/'s.png';Image.new('RGB',(300,80),'white').save(p);snap,b=branches(p)
    b['text'].styles=[Style(id='s',role='body',ea='Arial',latin='Arial',size_px=8.,bold=False,color='000000')]
    b['text'].texts=[Text(id=f't{i}',text='X',bbox=[float(i*10),2.,8.,12.],z=1,style='s') for i in range(30)]
    b['text'].members={t.id:[t.id] for t in b['text'].texts}
    g=fuse(snap,b,{'lines':[{'id':t.id,'text':t.text,'score':.99} for t in b['text'].texts]})
    def candidate(t,graph,source,fonts,blocked):
        if t.id=='t7':raise ValueError('unknown backing')
        x,y,w,h=map(round,t.bbox);mask=np.zeros(source.shape[:2],bool);mask[y,x]=True
        return {'id':t.id,'kind':'text','bbox':[x,y,w,h],'mask':mask,'patch':source[y:y+h,x:x+w].copy(),'payload':t.model_dump()}
    monkeypatch.setattr(mod,'text_candidate',candidate)
    ir,report=build(g,snap,{'fonts':[]},tmp_path/'scene')
    assert len([n for n in json.loads(ir.read_text())['nodes'] if n['type']=='text'])==29
    assert [r['id'] for r in report if r['state']=='FALLBACK']==['t7']
    assert all(n['bold'] is False for n in json.loads(ir.read_text())['nodes'] if n['type']=='text')


def surface_scene(tmp_path,inside=False,unknown_pixel=False):
    source=np.full((80,120,3),255,np.uint8);source[20:40,70:100]=[80,120,180]
    if unknown_pixel:source[25,80]=[200,0,0]
    p=tmp_path/'s.png';Image.fromarray(source).save(p);snap,b=branches(p)
    b['text'].styles=[Style(id='s',role='body',ea='Arial',latin='Arial',size_px=8.,bold=False,color='000000')]
    b['text'].texts=[Text(id='unresolved',text='?',bbox=[75. if inside else 3.,22.,10.,10.],z=2,style='s')]
    b['surface'].shapes=[Shape(id='card',bbox=[70.,20.,30.,20.],z=1,geometry='rect',fill='5078B4')]
    return snap,fuse(snap,b,{'lines':[]}),source


def test_unrelated_failure_does_not_block_card(tmp_path):
    snap,g,source=surface_scene(tmp_path)
    ir,report=build(g,snap,{'fonts':[]},tmp_path/'scene')
    assert any(n['id']=='card' and n['type']=='shape' for n in json.loads(ir.read_text())['nodes'])
    plate=np.array(Image.open(ir.parent/'source-plate.png').convert('RGB'))
    outside=np.ones(source.shape[:2],bool);outside[20:40,70:100]=False
    assert np.array_equal(source[outside],plate[outside])
    assert np.all(plate[20:40,70:100]==255)


@pytest.mark.parametrize('inside,unknown_pixel',[(True,False),(False,True)])
def test_local_unresolved_content_or_unknown_pixel_blocks_card(tmp_path,inside,unknown_pixel):
    snap,g,source=surface_scene(tmp_path,inside,unknown_pixel)
    ir,report=build(g,snap,{'fonts':[]},tmp_path/'scene')
    assert not any(n['type']=='shape' for n in json.loads(ir.read_text())['nodes'])
    assert np.array_equal(source,np.array(Image.open(ir.parent/'source-plate.png').convert('RGB')))


def test_overlap_is_dependency_not_semantic_merge(tmp_path):
    snap,g,_=surface_scene(tmp_path,True)
    rows,index=regions(g)
    assert index['card']['region_id']!=index['unresolved']['region_id']
    assert not region_ready(index['card'],rows,set())
    assert region_ready(index['card'],rows,{'unresolved'})


def test_explicit_cross_object_relation_is_atomic(tmp_path):
    snap,g,_=surface_scene(tmp_path)
    from product.contracts import Text
    g.text.texts.append(Text(id='other',text='x',bbox=[25.,22.,10.,10.],z=2,style='s'))
    g.ownership['other']='RESOLVED'
    g.relations.append(Relation(source='other',target='unresolved',kind='part_of',status='RESOLVED',evidence=['source-parts']))
    rows,index=regions(g)
    assert index['other']['region_id']==index['unresolved']['region_id']
    retained,rejected=retain_atomic([{'id':'other'}],rows)
    assert not retained and rejected=={'other':'REGION_MEMBER_FAILURE'}
