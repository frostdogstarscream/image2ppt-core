"""Synthetic coverage for the object-scoped upload path and isolation retries."""
import json

import cv2
import numpy as np
from PIL import Image

from product.parallel_contracts import SceneGraph
from product.parallel_fusion import fuse
from product.parallel_rebuild import build
from product.recovery.badge_stage import stage_badge_symbols
from product.recovery.circular_badge import select_badge_ink
from product.recovery.container_stage import stage_containers_and_surfaces
from product.recovery.region_crop import collect_crop_seeds
from test_parallel import branches
from test_region_crop import _icon


def panel_scene(root):
    source = np.full((340, 620, 3), 255, np.uint8)
    locators = {}
    for y in (30,165):
        cv2.fillPoly(source,[np.int32([(25,y),(480,y),(510,y+46),(480,y+92),(25,y+92)])],(240,246,252))
        cv2.circle(source,(93,y+46),32,(222,232,245),-1)
        cv2.circle(source,(93,y+46),14,(44,80,172),4)
        cv2.line(source,(83,y+50),(101,y+56),(97,56,130),3)
        locators[f'icon-{y}'] = [71,y+24,44,44]
    path = root/'source.png'
    Image.fromarray(source).save(path)
    snap, evidence = branches(path)
    for oid, bbox in locators.items():
        obj = _icon((oid,bbox)).model_copy(update={'kind':'icon'})
        evidence['asset'].objects.append(obj)
    baseline = root/'baseline'
    build(fuse(snap,evidence,{'lines':[]}),snap,{'fonts':[]},baseline,strategy='off',text_candidates=False)
    return baseline, locators


def test_badge_separates_disc_and_ink_and_deduplicates_verified_evidence(tmp_path):
    base, boxes = panel_scene(tmp_path)
    layout = {'regions':[{'id':'l-02','bbox':boxes['icon-30'],'label':'image','score':.99}]}
    out, selected = stage_badge_symbols(base,tmp_path/'ink',layout)
    graph = SceneGraph.model_validate_json((out/'scene.json').read_text('utf8'))
    assert len(selected) == 2
    top = next(graph.selected[i] for i in selected if 'icon-30' in graph.selected[i]['badge_evidence_ids'])
    assert top['badge_evidence_ids'] == ['icon-30','layout:l-02']
    alpha = np.asarray(Image.open(out/top['asset_reference']['path']))[...,3]
    assert alpha[alpha.shape[0]//3,alpha.shape[1]//2] == 0
    assert alpha.max() > 200
    assert not collect_crop_seeds(graph,layout=layout)[0]
    panel_ir, ids = stage_containers_and_surfaces(out,tmp_path/'all',stable_ids=True)
    nodes = {n['id']:n for n in json.loads(panel_ir.read_text('utf8'))['nodes'] if n['id'] in ids}
    arrows = sorted((n for n in nodes.values() if n.get('geometry')=='pentagon'),key=lambda n:n['bbox'][1])
    assert len(arrows)==2
    assert sum(n.get('geometry')=='ellipse' for n in nodes.values())==2
    _, retry = stage_containers_and_surfaces(out,tmp_path/'retry',stable_ids=True,skip_ids={arrows[0]['id']})
    assert arrows[1]['id'] in retry and arrows[0]['id'] not in retry


def test_text_protection_and_foreground_retries_are_local(tmp_path):
    base, boxes = panel_scene(tmp_path)
    (base/'runtime-ocr-protection.json').write_text(json.dumps({'lines':[{'id':'word','bbox':boxes['icon-30'],
                                                       'text':'A','score':.99}]}),'utf8')
    out, ids = stage_badge_symbols(base,tmp_path/'ink',None)
    assert len(ids)==1
    top = np.s_[30:123,25:511]
    assert np.array_equal(np.asarray(Image.open(base/'source-plate.png'))[top],
                          np.asarray(Image.open(out/'source-plate.png'))[top])
    measured = np.zeros((8,9,4),np.uint8)
    measured[2:6,3:6] = [42,74,140,255]
    source = np.where(measured[...,3:] > 0, measured[...,:3], 240)
    assert select_badge_ink(source,'f0f0f0',measured,measured,excluded={'color-unmix'})[1]['selected']=='LayerD'
    assert select_badge_ink(source,'f0f0f0',measured,None,excluded={'color-unmix'})[0] is None


def test_faint_antialias_is_foreground_and_open_holes_remain_transparent():
    measured = np.zeros((16,16,4), np.uint8)
    measured[2:14,2:14] = [253,253,253,14]
    measured[4:12,4:12] = [40,40,150,255]
    measured[6:10,6:10] = 0
    alpha = measured[...,3:] / 255
    source = np.rint(measured[...,:3]*alpha + 240*(1-alpha)).astype('uint8')
    _, accepted = select_badge_ink(source,'f0f0f0',measured,None,strict_holes=True)
    assert accepted['selected']=='color-unmix'
    dirty = measured.copy()
    dirty[7,8] = [240,240,240,255]
    _, rejected = select_badge_ink(source,'f0f0f0',measured,dirty,strict_holes=True)
    assert rejected['candidates'][1]['status']=='rejected'


def flat_card_scene(root, *, extra_art=False):
    image = np.full((250, 460, 3), 255, np.uint8)
    cv2.rectangle(image,(40,42),(355,155),(94,131,173),-1)
    cv2.rectangle(image,(43,45),(352,152),(228,239,250),-1)
    cv2.circle(image,(99,97),15,(39,72,153),4)
    if extra_art:
        cv2.line(image,(233,85),(256,110),(43,90,160),3)
    source = root/'source.png'
    Image.fromarray(image).save(source)
    snap, evidence = branches(source)
    box = [79,77,41,41]
    evidence['asset'].objects.append(_icon(('cloud',box)).model_copy(update={'kind':'icon'}))
    base = root/'baseline'
    build(fuse(snap,evidence,{'lines':[]}),snap,{'fonts':[]},base,strategy='off',text_candidates=False)
    return base, [40,42,316,114]


def test_scoped_panel_can_be_native_without_a_badge_or_any_asset_request(tmp_path):
    from product.runtime.batch_backbone import _chain_stage_from_spec
    base, box = flat_card_scene(tmp_path)
    # The same card without the uncommitted foreground has directly measurable paper.
    img = np.array(Image.open(base/'source.png').convert('RGB'))
    img[75:118,77:121] = [228,239,250]
    path = tmp_path/'no-icon.png'
    Image.fromarray(img).save(path)
    snap, ev = branches(path)
    plain = tmp_path/'plain'
    graph = fuse(snap,ev,{'lines':[]})
    build(graph,snap,{'fonts':[]},plain,strategy='off',text_candidates=False)
    out, ids, _ = _chain_stage_from_spec(initial_graph=graph, initial_baseline=plain, fonts={'fonts':[]},
        source_path=plain/'source.png', layers_dir=None, deferred_path=None, page_cache_dir=tmp_path,
        directory=tmp_path, tag='scoped', trials_spec=[], frame_overrides={}, skip_ids=set(), include_assets=False,
        include_containers=True, wanted_filter=None, sam_enabled=False, single_object_only=True, panel_regions=[box])
    result = SceneGraph.model_validate_json((out/'scene.json').read_text('utf8'))
    assert any(result.selected[i]['kind']=='surface' and result.selected[i]['bbox']==box for i in ids)


def test_flat_icon_before_or_after_its_card_keeps_one_symbol_and_clean_paper(tmp_path):
    from product.recovery.marked_surfaces import stage_marked_surfaces
    base, box = flat_card_scene(tmp_path, extra_art=True)
    first, ids = stage_badge_symbols(base,tmp_path/'symbol-first',None)
    assert len(ids) == 1
    ir, shapes = stage_containers_and_surfaces(first,tmp_path/'children-then-card',stable_ids=True)
    # Unidentified artwork keeps the native-only fitter from flattening the card.
    assert not any(r.get('bbox')==box for r in json.loads(ir.read_text('utf8'))['nodes'] if r['id'] in shapes)
    parent, pids = stage_marked_surfaces(base,tmp_path/'parent-first',[{'id':'panel','bbox':box}],set())
    assert len(pids)==2
    graph = SceneGraph.model_validate_json((parent.parent/'scene.json').read_text('utf8'))
    old_art = next(r for r in graph.selected.values() if r['kind']=='asset')
    png = np.array(Image.open(parent.parent/old_art['asset_reference']['path']))
    after, changed = stage_badge_symbols(parent.parent,tmp_path/'peeled',None)
    final = SceneGraph.model_validate_json((after/'scene.json').read_text('utf8'))
    assert set(changed)=={ids[0], old_art['id']}
    separated = final.selected[ids[0]]
    assert separated['payload']['background_parent_id'] in pids
    assert separated['asset_reference']['sha256'] == json.loads((first/'scene.json').read_text('utf8'))['selected'][ids[0]]['asset_reference']['sha256']
    assert (after/'source-plate.png').read_bytes()==(parent.parent/'source-plate.png').read_bytes()
    remaining = np.array(Image.open(after/final.selected[old_art['id']]['asset_reference']['path']))
    assert np.array_equal(remaining[:,-30:],png[:,-30:]) and not remaining[:,:38,3].any()
    assert not stage_badge_symbols(after,tmp_path/'repeat',None)[1]


def test_plain_card_keeps_the_same_editable_layers_in_both_orders(tmp_path):
    from product.recovery.marked_surfaces import stage_marked_surfaces
    base, box = flat_card_scene(tmp_path)
    child, first_ids = stage_badge_symbols(base,tmp_path/'icon-first',None)
    deck, _ = stage_containers_and_surfaces(child,tmp_path/'after-icon',stable_ids=True,native_only=True)
    first = SceneGraph.model_validate_json((deck.parent/'scene.json').read_text('utf8'))
    frame = next(r for r in first.selected.values() if r['kind']=='surface' and r['bbox']==box)
    assert first_ids[0] in frame['contained_children']

    deck, _ = stage_marked_surfaces(base,tmp_path/'frame-first',[{'id':'measured','bbox':box}],set())
    updated, ids = stage_badge_symbols(deck.parent,tmp_path/'after-frame',None)
    second = SceneGraph.model_validate_json((updated/'scene.json').read_text('utf8'))
    assert len(ids)==1 and set(first.selected[i]['kind'] for i in first.selected)=={'asset','surface'}
    assert len(second.selected)==len(first.selected)==2
    assert second.selected[ids[0]]['asset_reference']['sha256']==first.selected[first_ids[0]]['asset_reference']['sha256']
    assert second.selected[ids[0]]['payload']['background_parent_id']==frame['id']
    assert np.asarray(Image.open(updated/'source-plate.png'))[97,113,:3].tolist()==[255,255,255]


def test_flat_icon_touching_neighbor_or_resembling_a_photo_is_not_claimed(tmp_path):
    from product.recovery.circular_badge import measure_flat_symbol
    base, _ = flat_card_scene(tmp_path)
    image = np.array(Image.open(base/'source.png').convert('RGB'))
    image[75:118,79:83]=[40,70,155]
    assert measure_flat_symbol(image,[79,77,41,41]) is None
    image[77:118,79:120]=np.random.default_rng(72).integers(0,255,(41,41,3),dtype='uint8')
    assert measure_flat_symbol(image,[79,77,41,41]) is None
