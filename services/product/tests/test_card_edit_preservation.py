from types import SimpleNamespace

from product.materialize import digest, write
from product.parallel_store import key
from product.runtime.card_edit_preservation import previous_edits_lost


def test_card_keeps_verified_individual_shapes_and_group(tmp_path):
    original = {'nodes':[{'id':'band','type':'shape'}, {'id':'caption','type':'text'}],
                'groups':[{'id':'header','member_ids':['band','caption']}]}
    prior, current_ir = tmp_path/'prior.json', tmp_path/'candidate.json'
    write(prior, original)
    state = SimpleNamespace(root=tmp_path)
    current = {'pages':[{'ir':str(prior)}]}
    candidate = {'page':1, 'card_repair_targets':['mark-card']}
    def check():
        return [r['object_id'] for r in previous_edits_lost(state,current,candidate,current_ir)]
    identity = {'irs':[digest(prior)], 'pptx':'deck-hash'}
    records = []
    for oid in ('band','header'):
        for action in ('move-scale','delete'):
            records.append({'id':oid,'page':1,'action':action,'saved_reopened':True,'outside_pixels':0,
                            **({'group_members':['band','caption']} if oid=='header' else {})})
    result = {'pptx_sha256':'deck-hash','office':{'pptx_sha256':'deck-hash','status':'PASS_APP_AUTOMATED',
                                                'records':records}}
    path = state.root/'office-cache'/(key(identity)+'.json')
    def cache():
        write(path, {'identity':identity,'result':result})
    cache()
    write(current_ir, {'nodes':original['nodes'], 'groups':[{'id':'renamed','member_ids':['caption','band']}]})
    assert check() == []
    write(current_ir, {'nodes':[{'id':'band','type':'image'},original['nodes'][1]],'groups':[]})
    assert check() == ['band','header']
    records[1]['outside_pixels'] = 2
    records[3]['group_members'] = ['band']
    cache()
    assert check() == []
    records[1]['outside_pixels'] = 0
    result['office']['pptx_sha256'] = 'other-deck'
    cache()
    assert check() == []
    result['office']['pptx_sha256'] = 'deck-hash'
    cache()
    candidate.pop('card_repair_targets')
    assert check() == []
