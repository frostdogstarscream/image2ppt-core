import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from product.object_contracts import ClosedObject
from product.contracts import Shape
from product.parallel_fusion import fuse
from product.peer_evidence import roles
from product.parallel_text_candidates import peer_protection
from test_r12_product_pipeline import product_fixture


def panel(f):
    g=f.graph.model_copy(deep=True)
    a=ClosedObject.model_validate({'id':'panel','description':'light background rounded card','prompt_en':'background card',
        'kind':'decoration','bbox':[5,5,165,105],'layers':['layer-00.png'],'components':['card'],
        'parts':[{'name':'backing','bbox':[5,5,165,105],'layers':['layer-00.png']}],'text_policy':'exclude_independent_text'})
    g.asset.objects=[a];g.surface.shapes=[Shape(id='surface',bbox=[5,5,165,105],z=1,geometry='roundRect',fill='EBEBEB')]
    ocr={'lines':[{'id':'l1','text':'正文甲','score':.99},{'id':'l2','text':'正文乙','score':.99}]}
    return fuse(f.snap,{'text':g.text,'asset':g.asset,'surface':g.surface},ocr)


def test_source_backing_relation_only_opens_read_context(product_fixture):
    f=product_fixture;g=panel(f);text=g.text.texts[0]
    assert roles(g)['panel']['role']=='BACKGROUND_PANEL_CANDIDATE'
    erase,context=peer_protection(text,g,f.source,np.zeros(f.source.shape[:2],bool),purpose='both')
    assert not context[20,60] and not erase[20,60]
    assert not erase[30,27]
    assert context[80,115] and erase[80,115]


def test_unknown_contrast_stays_protected(product_fixture):
    f=product_fixture;g=panel(f);source=f.source.copy();source[52:57,70:75]=[200,0,170]
    erase,context=peer_protection(g.text.texts[0],g,source,np.zeros(source.shape[:2],bool),purpose='both')
    assert erase[54,72] and not context[54,72]


def test_semantics_without_surface_or_source_proof_keeps_scope(product_fixture):
    f=product_fixture;g=panel(f);g.relations=[]
    mask=peer_protection(g.text.texts[0],g,f.source,np.zeros(f.source.shape[:2],bool))
    assert mask[30,27] and mask[20,60]
    g=panel(f);source=f.source.copy();source[:]=[20,80,150]
    mask=peer_protection(g.text.texts[0],g,source,np.zeros(source.shape[:2],bool))
    assert mask[30,27] and mask[20,60]


def test_real_foreground_bbox_stays_conservative_without_owner(product_fixture):
    f=product_fixture;g=panel(f)
    g.asset.objects[0].kind='icon';g.relations=[]
    assert peer_protection(g.text.texts[0],g,f.source,np.zeros(f.source.shape[:2],bool))[30,27]


def test_shared_ocr_number_heading_order_ignores_baseline_jitter(product_fixture):
    f=product_fixture;g=f.graph.model_copy(deep=True)
    from product.contracts import Text
    g.text.texts=[Text(id='num',text='03',bbox=[10,25,10,10],z=1,style='s'),Text(id='heading',text='标题',bbox=[30,23,30,10],z=2,style='s')]
    g.text.members={'num':['line'],'heading':['line']}
    ocr={'lines':[{'id':'line','text':'03标题','score':.99}]}
    fused=fuse(f.snap,{'text':g.text,'asset':g.asset,'surface':g.surface},ocr)
    assert fused.ownership['num']==fused.ownership['heading']=='RESOLVED'


def test_hash_bound_owner_protects_ink_and_preserves_holes(product_fixture,tmp_path):
    import hashlib,json
    from PIL import Image
    from product.materialize import write,digest
    f=product_fixture;g=panel(f);g.asset.objects[0].kind='icon';g.relations=[]
    base=tmp_path/'baseline';base.mkdir();Image.fromarray(f.source).save(base/'source.png')
    mask=np.zeros(f.source.shape[:2],bool);mask[25:40,25:30]=True
    np.savez_compressed(base/'owner.npz',mask=mask)
    record={'kind':'asset','complete_owner_scope':True,'source_sha256':g.source_sha256,
            'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),
            'owner_evidence':{'file':'owner.npz','sha256':digest(base/'owner.npz')}}
    write(base/'scene.json',{'source_sha256':g.source_sha256,'selected':{'panel':record}})
    protected=peer_protection(g.text.texts[0],g,f.source,np.zeros(mask.shape,bool),baseline_directory=base)
    assert protected[30,27] and not protected[20,60]
    np.savez_compressed(base/'owner.npz',mask=~mask)
    with pytest.raises(ValueError,match='proof mismatch'):
        peer_protection(g.text.texts[0],g,f.source,np.zeros(mask.shape,bool),baseline_directory=base)


def test_grouped_backings_match_each_surface_without_pixel_ownership(product_fixture):
    from product.object_contracts import ObjectPart
    from product.parallel_regions import regions
    f=product_fixture;g=panel(f)
    g.asset.objects[0].parts=[ObjectPart(name='left background',bbox=[5,5,70,105],layers=['layer-00.png']),
                              ObjectPart(name='right background',bbox=[85,5,85,105],layers=['layer-00.png'])]
    g.surface.shapes=[Shape(id='left',bbox=[5,5,70,105],z=1,geometry='roundRect',fill='EBEBEB'),
                      Shape(id='right',bbox=[85,5,85,105],z=2,geometry='roundRect',fill='EBEBEB')]
    ocr={'lines':[{'id':'l1','text':'正文甲','score':.99},{'id':'l2','text':'正文乙','score':.99}]}
    fused=fuse(f.snap,{'text':g.text,'asset':g.asset,'surface':g.surface},ocr)
    assert fused.ownership['panel']=='UNOWNED'
    assert set(roles(fused)['panel']['surface_ids'])=={'left','right'}
    _,mapping=regions(fused)
    assert mapping['panel']['region_id'] not in mapping['left']['dependencies']
    assert mapping['panel']['region_id'] not in mapping['right']['dependencies']
    protected=peer_protection(fused.text.texts[0],fused,f.source,np.zeros(f.source.shape[:2],bool))
    assert not protected[30,27]


def test_unmatched_part_does_not_become_backing_alias(product_fixture):
    from product.object_contracts import ObjectPart
    f=product_fixture;g=panel(f)
    g.asset.objects[0].parts=[ObjectPart(name='left background',bbox=[5,5,70,105],layers=['layer-00.png']),
                              ObjectPart(name='unproven distant component',bbox=[85,5,85,105],layers=['layer-00.png'])]
    g.surface.shapes=[Shape(id='left',bbox=[5,5,70,105],z=1,geometry='roundRect',fill='EBEBEB')]
    ocr={'lines':[{'id':'l1','text':'正文甲','score':.99},{'id':'l2','text':'正文乙','score':.99}]}
    fused=fuse(f.snap,{'text':g.text,'asset':g.asset,'surface':g.surface},ocr)
    assert roles(fused)['panel']['role']=='UNCERTAIN_COMPOSITE'
    assert peer_protection(fused.text.texts[0],fused,f.source,np.zeros(f.source.shape[:2],bool))[30,27]
