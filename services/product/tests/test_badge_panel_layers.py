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
