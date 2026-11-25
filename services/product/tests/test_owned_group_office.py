import numpy as np
from PIL import Image
import pytest

from product.materialize import digest, write
from product.parallel_office import owned_edit_groups
from product.parallel_residual_gate import inspect_residuals


def grouped_owner(tmp_path):
    h,w = 16,24
    paper = np.full((h,w,3), 239, np.uint8)
    source = paper.copy()
    mask = np.zeros((h,w), bool)
    mask[4:10, 3:15] = True
    source[mask] = (20,82,168)
    Image.fromarray(source).save(tmp_path/'source.png')
    proof = tmp_path/'group-owner.npz'
    np.savez_compressed(proof, mask=mask, restore_bbox=[0,0,w,h], backing_patch=paper)
    ir = {'nodes':[{'id':'paper','type':'shape','z':1,'bbox':[0,0,w,h]},
                   {'id':'tip','type':'shape','z':2,'bbox':[3,4,6,6]},
                   {'id':'tail','type':'image','z':3,'bbox':[9,4,6,6]}],
          'groups':[{'id':'drawing','member_ids':['tip','tail']},{'id':'unowned','member_ids':['paper']}]}
    write(tmp_path/'slideir.json', ir)
    write(tmp_path/'scene.json', {
        'source_sha256':digest(tmp_path/'source.png'), 'selected':{
            'paper':{'id':'paper','kind':'surface','bbox':[0,0,w,h]},
            'drawing':{'id':'drawing','kind':'asset','bbox':[3,4,12,6],
                       'owner_evidence':{'file':proof.name,'sha256':digest(proof)}}}})
    return ir, mask, paper, source


def test_only_group_with_its_own_bound_evidence_gets_a_whole_group_probe(tmp_path):
    ir,_,_,_=grouped_owner(tmp_path)
    path=tmp_path/'slideir.json'
    assert owned_edit_groups(path,ir)=={'drawing':{'tip','tail'}}
    assert owned_edit_groups(path,ir,{'tip'})=={}
    assert owned_edit_groups(path,ir,{'drawing','unowned'})=={'drawing':{'tip','tail'}}
    (tmp_path/'group-owner.npz').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='binding mismatch'):
        owned_edit_groups(path,ir)


def test_group_deletion_requires_group_preview_and_checks_pixels_on_lower_sibling(tmp_path):
    _,mask,paper,source=grouped_owner(tmp_path)
    def status():
        rows=inspect_residuals(tmp_path/'slideir.json',tmp_path,1,tmp_path/'result.json',owned_group_ids={'drawing'})
        return next(row for row in rows if row['id']=='drawing')
    for member in ('tip','tail'):
        Image.fromarray(paper).save(tmp_path/f'01-{member}-delete.png')
    assert status()['residual_check_status']=='UNKNOWN'
    Image.fromarray(source).save(tmp_path/'01-drawing-delete.png')
    bad=status()
    assert bad['residual_check_status']=='FAIL'
    assert bad['residual_pixels_confirmed']==int(mask.sum())
    Image.fromarray(paper).save(tmp_path/'01-drawing-delete.png')
    clean=status()
    assert clean['residual_check_status']=='PASS'
    assert clean['checked_owner_pixels']==int(mask.sum())


def test_research_worker_returns_group_move_and_delete_previews(tmp_path, monkeypatch):
    import io,zipfile
    from types import SimpleNamespace
    from product.runtime import office, office_bridge

    source=tmp_path/'packet'
    write(source/'request.json',{'irs':['page/slideir.json'],'check_owned_groups':True})
    write(source/'page/slideir.json',{'nodes':[]})
    (source/'input.pptx').write_bytes(b'ticket')
    packet=tmp_path/'input.zip'
    office_bridge.pack(source,packet)
    uploaded=[]

    class Client:
        def post(self,url,**kwargs):
            if url.endswith('/result'):
                uploaded.append(kwargs['content'])
            return SimpleNamespace(content=packet.read_bytes(),raise_for_status=lambda:None)

    def check(pptx,irs,directory,*,check_owned_groups):
        assert check_owned_groups
        move={'id':'drawing','action':'move-scale','page':1,'group_members':['left','right']}
        folder=directory/'office';folder.mkdir(parents=True)
        for name in ['01-drawing-move-scale','01-drawing-delete','01-left-move-scale','01-left-delete']:
            (folder/(name+'.png')).write_bytes(b'preview')
        replacement=directory/'replacement'
        replacement.mkdir()
        for name in ['01-glyph-replace-equal','01-glyph-replace-equal-expanded','01-other-replace-equal']:
            (replacement/(name+'.png')).write_bytes(b'preview')
        glyph={'id':'glyph','action':'replace-equal','page':1,'layout':{'evidence':'single_glyph_ink_within_original_frame'}}
        return {'status':'PASS','office':{'records':[move]},'replacements':{'records':[glyph]}}

    monkeypatch.setattr(office,'check_local',check)
    office_bridge.execute_ticket(Client(),{'ticket':'synthetic'},tmp_path/'run','test-worker')
    with zipfile.ZipFile(io.BytesIO(uploaded[0])) as result:
        assert 'check/office/01-drawing-move-scale.png' in result.namelist()
        assert 'check/office/01-drawing-delete.png' in result.namelist()
        assert 'check/office/01-left-move-scale.png' not in result.namelist()
        assert 'check/office/01-left-delete.png' in result.namelist()
        assert 'check/replacement/01-glyph-replace-equal.png' in result.namelist()
        assert 'check/replacement/01-glyph-replace-equal-expanded.png' in result.namelist()
        assert 'check/replacement/01-other-replace-equal.png' not in result.namelist()
