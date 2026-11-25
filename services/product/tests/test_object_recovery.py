import json
from types import SimpleNamespace

import pytest

from product.materialize import digest, write
from product.parallel_store import key
from product.runtime.card_edit_preservation import previous_edits_lost
from product.runtime.object_recovery import remember_edits, recover_page
from product.runtime.object_recovery import _matches
from product.runtime import host_baseline
from product.runtime.state import RuntimeState


def office_for(*ids):
    return {'pptx_sha256':'old-pptx', 'status':'REVIEW_REQUIRED', 'office':{
        'pptx_sha256':'old-pptx', 'status':'PASS_APP_AUTOMATED',
        'records':[{'page':1,'id':oid,'action':action,'saved_reopened':True,'outside_pixels':0}
                   for oid in ids for action in ('move-scale','delete')]}}


@pytest.mark.parametrize('outcome',['VALIDATED','REVIEW_REQUIRED','WRONG_FORM'])
def test_restage_group_from_failed_page_on_current_parent(tmp_path,outcome):
    state = SimpleNamespace(root=tmp_path/'runtime',remaining_seconds=lambda:120)
    source = tmp_path/'source.png'
    source.write_bytes(b'page')
    source_sha = digest(source)
    field = {'id':'mark-diagram', 'role':'illustration', 'route':'asset','move_with':[], 'bbox':[5,5,50,30]}
    write(tmp_path/'marks',{'source_sha256':source_sha,'rows':[{'id':field['id']}]})
    parent = tmp_path/'parent-ir'
    existing = {'id':'new-caption','type':'text'}
    old = {'nodes':[{'id':'pen','type':'image'},{'id':'tip','type':'shape'}],
           'groups':[{'id':'arrow','member_ids':['pen','tip']}]}
    write(parent, {'nodes':[existing]})
    old_ir = tmp_path/'rejected-ir'
    write(old_ir, old)
    graph = SimpleNamespace(source_sha256=source_sha, text=SimpleNamespace(texts=[]), asset=SimpleNamespace(objects=[]),
                            selected={'arrow':{'kind':'asset','mark_id':'mark-diagram','owner_evidence':{'sha256':'mask'}}})
    current = {'pages':[{'ir':str(parent),'snapshot':{'path':str(source)}}], 'marked_plans':{'1':[field]},
               'experiment_marks':{'1':str(tmp_path/'marks')}, 'candidates':{}}
    donor = {'id':'rejected-candidate','code':'code','kind':'batch','page':1,'object_ids':['arrow']}
    checked = office_for('pen','tip')
    remember_edits(state,current,donor,old_ir,graph,checked)
    old_ir.unlink()  # Recovery must not read the historic IR or copy nodes from it.
    calls = []
    def stage(state,trial,directory,*,page,object_ids,verify_office,**options):
        assert trial['pages'][0]['ir'] == str(parent) and object_ids == ['mark-diagram'] and verify_office
        calls.append(page)
        ir = directory/'restaged-ir'
        write(ir, {'nodes':[existing,*old['nodes']], 'groups':old['groups'] if outcome=='VALIDATED' else []})
        trial['candidates']['current'] = {'ir':str(ir), 'id':'current'}
        return {'candidate_id':'current','status':'REVIEW_REQUIRED' if outcome=='REVIEW_REQUIRED' else 'VALIDATED'}
    def commit(state,trial,directory,*,candidate_id):
        trial['pages'][0]['ir'] = trial['candidates'][candidate_id]['ir']
        return {'status':'COMMITTED'}
    service = SimpleNamespace(code_binding=lambda:'code',page=lambda c,p:c['pages'][p-1],
                              batch_reconstruct_page=stage,commit_region=commit)
    restored = recover_page(service,state,current,tmp_path/'recovery',1)
    assert restored == (['arrow'] if outcome=='VALIDATED' else [])
    assert (current['pages'][0]['ir'] != str(parent)) == (outcome=='VALIDATED')
    assert recover_page(service,state,current,tmp_path/'again',1) == []
    assert calls == [1]
    source.write_bytes(b'different image')
    assert recover_page(service,state,current,tmp_path/'other',1) == []


def test_movable_image_and_owned_group_guard_applies_to_noncard_edits(tmp_path):
    state = SimpleNamespace(root=tmp_path)
    ir, changed = tmp_path/'parent', tmp_path/'changed'
    write(ir,{'nodes':[{'id':'photo','type':'image'},{'id':'source-plate','type':'image'}],
              'groups':[{'id':'assembly','member_ids':['photo','label']}]})
    write(changed,{'nodes':[]})
    identity = {'irs':[digest(ir)],'pptx':'old-pptx'}
    receipt = office_for('photo','assembly','source-plate')
    for probe in receipt['office']['records']:
        if probe['id']=='assembly':
            probe['group_members']=['photo','label']
    write(tmp_path/'office-cache'/(key(identity)+'.json'),{'identity':identity,'result':receipt})
    current = {'pages':[{'ir':str(ir)}]}
    assert {r['object_id'] for r in previous_edits_lost(state,current,{'page':1},changed,all_edits=True)} == {'assembly','photo'}
    assert previous_edits_lost(state,current,{'page':1},changed) == []


def test_sequential_picture_ids_are_compared_by_source_owner():
    item = {'object_id':'r-001','crop_seeds':['left-photo'],'representation':'image'}
    shapes = {'r-001':'image', 'r-002':'image'}
    assert not _matches(item, shapes, {'r-001':{'seed_ids':['other-photo']}})
    assert _matches(item, shapes, {'r-002':{'seed_ids':['left-photo']}})


@pytest.mark.parametrize('regresses',[False,True])
def test_rejected_agent_can_deliver_rechecked_edits_without_changing_frozen_control(tmp_path, monkeypatch, regresses):
    job = tmp_path/'job'
    frozen = job/'host-baseline'
    frozen.mkdir(parents=True)
    base_file = frozen/'baseline.pptx'
    base_file.write_bytes(b'frozen control')
    rechecked = job/'rechecked.pptx'
    rechecked.write_bytes(b'independent region on the frozen control')
    state = RuntimeState(job)
    (state.root/'object-recovery').mkdir()
    write(job/'request.json',{'object_scoped_delivery':True})
    page = {'page':1,'editable_count':1,'native_field_count':0,'ocr_native_hits':[],'max_local_mae':0,
            'selected_editables':['photo'],'plate_audit':{'status':'MEASURED','rows':[]}}
    base = {'verified':True,'office_status':'PASS','pptx_sha256':digest(base_file),'pages':[page]}
    write(frozen/'scorecard.json',base)
    head = {'version':1,'status':'RUNNING','pages':[{'ir':'ir'}],
            'composition':{'artifacts':{'pptx':str(base_file)}, 'office':{}}}
    write(frozen/'state.json',head)
    write(state.root/'state.json',head)
    manifest = {'files':{'baseline.pptx':digest(base_file)},'sources':['source']}
    monkeypatch.setattr(host_baseline,'ensure_baseline',lambda _:manifest)
    monkeypatch.setattr('product.runtime.object_recovery.recover_page',lambda *a:['group'])
    def compose(self,state,current,directory, **opts):
        current['composition'] = {'artifacts':{'pptx':str(rechecked)}}
    monkeypatch.setattr(host_baseline.RuntimeService,'compose_delivery',compose)
    monkeypatch.setattr(host_baseline.RuntimeService,'composition',staticmethod(lambda c:c['composition']))
    monkeypatch.setattr(host_baseline.RuntimeService,'office_check',staticmethod(lambda *a:{'status':'PASS'}))
    measured = {**base, 'pptx_sha256':digest(rechecked),
                'pages':[{**page, 'editable_count':2, 'selected_editables':['photo','group']}]}
    monkeypatch.setattr(host_baseline,'assess',lambda *a:measured)
    monkeypatch.setattr(host_baseline,'regressions',lambda *a:['VISIBLE_OLD_CAPTION'] if regresses else [])
    result = host_baseline._publish(job, {'reason':'agent trial incomplete'},['CANDIDATE_REGRESSION'],None)
    assert result['status'] == 'REVIEW_REQUIRED'
    assert result['origin'] == (host_baseline.ORIGIN if regresses else 'object-incremental')
    assert digest(job/'delivery/result.pptx') == digest(base_file if regresses else rechecked)
    assert json.loads((frozen/'state.json').read_bytes()) == head
