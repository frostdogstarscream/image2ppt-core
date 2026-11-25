import json
import zipfile

import numpy as np
from PIL import Image

from product.materialize import digest, write
from product.parallel_residual_gate import inspect_residuals


def saved_deck(folder):
    with zipfile.ZipFile(folder/'candidate.pptx', 'w') as deck:
        deck.writestr('ppt/slides/slide1.xml', '''<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
            <p:pic><p:nvPicPr><p:cNvPr name="source-plate"/></p:nvPicPr>
            <p:blipFill><a:blip r:embed="rId2"/></p:blipFill></p:pic></p:sld>''')
        deck.writestr('ppt/slides/_rels/slide1.xml.rels',
                      '<Relationships><Relationship Id="rId2" Target="../media/image2.png"/></Relationships>')
        deck.write(folder/'visible.png','ppt/media/image2.png')


def text_scene(folder, *, plate_ghost=False, visible_stroke=False, decimal=False, speck=False):
    paper = np.full((48, 64, 3), 220, np.uint8)
    ink = np.zeros(paper.shape[:2], bool)
    ink[6:36, 15:20] = True
    ink[6:36, 31:36] = True
    ink[19:22, 15:36] = True
    ink[33:35, 41:43] = True
    source = paper.copy()
    source[ink] = 40
    Image.fromarray(source).save(folder/'source.png')
    plate = paper.copy()
    if plate_ghost:
        plate[ink] = 214
    Image.fromarray(plate).save(folder/'visible.png')
    after = paper.copy()
    if visible_stroke:
        after[6:36, 16] = 214
    if decimal:
        after[33:35, 41:43] = 214
    if speck:
        after[10, 17] = 214
    Image.fromarray(after).save(folder/'01-word-delete.png')
    # A small old reference error near an antialiased edge triggers the historical gate.
    backing = paper.copy()
    backing[9, 15] = 230
    backing[10, 15] = 230
    box = [8, 3, 42, 36]
    x, y, w, h = box
    np.savez(folder/'proof.npz', mask=ink, restore_bbox=box, backing_patch=backing[y:y+h, x:x+w])
    sha = digest(folder/'source.png')
    write(folder/'scene.json', {'source_sha256':sha, 'selected':{'word':{
        'id':'word', 'kind':'text', 'source_text_definition':{'text':'H.'},
        'owner_evidence':{'file':'proof.npz', 'sha256':digest(folder/'proof.npz')}}}})
    write(folder/'slideir.json', {'source_sha256':sha, 'assets':[{'id':'source-plate','path':'visible.png',
        'sha256':digest(folder/'visible.png')}], 'nodes':[{'id':'source-plate','asset_id':'source-plate',
        'type':'image','bbox':[0,0,64,48]}]})
    saved_deck(folder)
    write(folder/'report.json', {'pptx_sha256':digest(folder/'candidate.pptx'),
        'status':'PASS_APP_AUTOMATED','embedded_alpha_and_visible_rgb_checked':True,'records':[{
        'page':1, 'id':'word', 'action':'delete','saved_reopened':True,'outside_pixels':0}]})


def status(folder, *, verified=True):
    return inspect_residuals(folder/'slideir.json', folder, 1, folder/'result.json',
                             text_edit_pptx=folder/'candidate.pptx' if verified else None)[0]


def test_clean_render_and_plate_override_a_stale_patch_only_in_runtime(tmp_path):
    text_scene(tmp_path, speck=True)
    assert status(tmp_path, verified=False)['residual_check_status'] != 'PASS'
    row = status(tmp_path)
    assert row['residual_check_status'] == 'PASS'
    assert row['prior_unexplained_pixels'] == row['minor_deviation_pixels'] == 2
    assert row['text_backing_check']['after_delete']['off_surface_pixels'] == 1
    Image.new('RGB',(64,48),'white').save(tmp_path/'visible.png')
    assert status(tmp_path)['residual_check_status'] != 'PASS'
    # Even a self-consistent IR cannot substitute a different picture for the
    # media embedded under the same shape name in the original Office-checked deck.
    ir = json.loads((tmp_path/'slideir.json').read_bytes())
    ir['assets'][0]['sha256'] = digest(tmp_path/'visible.png')
    write(tmp_path/'slideir.json', ir)
    assert status(tmp_path)['residual_check_status'] != 'PASS'


def test_connected_faint_stroke_and_complete_decimal_stay_blocked(tmp_path):
    for options in ({'visible_stroke':True}, {'decimal':True}, {'plate_ghost':True}):
        text_scene(tmp_path, **options)
        assert status(tmp_path)['residual_check_status'] != 'PASS'


def test_adjacent_owner_cannot_supply_the_background_reference(tmp_path):
    text_scene(tmp_path)
    scene_file = tmp_path/'scene.json'
    scene = json.loads(scene_file.read_bytes())
    neighbour = np.zeros((48,64), bool)
    neighbour[6:36,14] = True
    np.savez(tmp_path/'neighbour.npz', mask=neighbour, restore_bbox=[14,6,1,30],
             backing_patch=np.full((30,1,3),220,np.uint8))
    scene['selected']['peer'] = {'id':'peer','owner_evidence':{
        'file':'neighbour.npz','sha256':digest(tmp_path/'neighbour.npz')}}
    write(scene_file,scene)
    assert status(tmp_path)['residual_check_status'] != 'PASS'


def test_outside_damage_and_missing_reopen_evidence_remain_blocked(tmp_path):
    text_scene(tmp_path)
    report = json.loads((tmp_path/'report.json').read_bytes())
    for saved, outside in [(False, 0), (True, 1)]:
        write(tmp_path/'report.json', {**report,'records':[{
            'id':'word','action':'delete','page':1,'saved_reopened':saved,'outside_pixels':outside}]})
        assert status(tmp_path)['residual_check_status'] != 'PASS'
    for key, value in (('pptx_sha256','some-other-deck'),('embedded_alpha_and_visible_rgb_checked',False)):
        write(tmp_path/'report.json', {**report,key:value})
        assert status(tmp_path)['residual_check_status'] != 'PASS'
