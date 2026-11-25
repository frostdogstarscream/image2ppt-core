"""Independent audit findings must resolve against the reviewed source inventory."""
import pytest

from product.materialize import write
from product.runtime.host_evidence import normalize_arguments, read, scoped_ids
from product.runtime.review_targets import expose_review, locate, repair_inputs
from product.runtime.state import RuntimeState


ROWS = [
    {'mark':1, 'id':'field', 'label':'text', 'bbox':[20,10,60,20]},
    {'mark':2, 'id':'card', 'label':'card_frame', 'bbox':[100,20,50,60]},
]


def test_review_coordinate_spaces_and_ambiguous_marks():
    size = (200,100)
    for box, space in [([20,10,60,20],'source_px_xywh'), ([.1,.1,.3,.2],'normalized_xywh')]:
        result = locate({'bbox':box, 'bbox_space':space}, ROWS, size)
        assert result['mark_ids'] == ['field'] and result['bbox'] == ROWS[0]['bbox']
    assert locate({'bbox':ROWS[0]['bbox']},ROWS,size)['binding'] == 'unspecified_geometry'
    assert locate({'bbox':[-1,10,60,20], 'bbox_space':'source_px_xywh'},ROWS,size)['binding'] == 'invalid_geometry'
    ambiguous = locate({'bbox':ROWS[0]['bbox'], 'bbox_space':'source_px_xywh'},
                       ROWS + [{**ROWS[0], 'mark':3, 'id':'duplicate'}], size)
    assert set(ambiguous['candidate_ids']) == {'duplicate','field'} and not ambiguous['mark_ids']
    assert locate({'mark_numbers':[1,7], 'bbox':ROWS[0]['bbox']},ROWS,size)['binding'] == 'unknown_mark'
    selected = locate({'mark_numbers':[2,1,2]},ROWS,size)
    assert selected['mark_ids'] == ['card','field'] and selected['bbox'] is None


def test_findings_open_only_requested_objects_and_limit_retries(tmp_path,monkeypatch):
    state = RuntimeState(tmp_path)
    rows = ROWS + [{'id':'other','mark':3,'label':'text','bbox':[170,70,20,20]}]
    for file, value in [('marks',{'source_sha256':'s','rows':rows}), ('ocr',{'lines':[]}),
                        ('graph',{'selected':{r['id']:{} for r in rows}}), ('block',rows)]:
        write(tmp_path/(file+'.json'), value)
    block = state.register(tmp_path/'block.json','inventory')
    current = {'pages':[{'scene':str(tmp_path/'graph.json')}], 'ocr':{'1':str(tmp_path/'ocr.json')},
               'experiment_marks':{'1':str(tmp_path/'marks.json')},
               'host_evidence':{'1':{'blocks':[{'inventory':block,'image':{'id':'preview'}}]}},
               'marked_plans':{'1':[{'id':'card','route':'surface','role':'card'}]}}
    defects = [{'finding_id':'wrong-word','category':'text_corruption','mark_ids':['field'],'binding':'numbered'},
               {'finding_id':'apart','category':'incorrect_group','mark_ids':['card'],'binding':'numbered'},
               {'finding_id':'unbound','category':'missing_text','mark_ids':[],'binding':'new_region',
                'bbox':[1,1,15,9]}, {'finding_id':'guess','category':'other','mark_ids':[],'binding':'new_region',
                'bbox':[3,1,11,8]}]
    discoveries = []
    def discover(service,state,current,directory,*,page,findings):
        discoveries.append(findings)
        return [{'status':'pending','reason':'source_readings_unconfirmed','id':'t-late-sample'}]
    monkeypatch.setattr('product.runtime.late_fields.register',discover)
    ref = expose_review(None,state,current,tmp_path,{'pages':[{'page':1,'vlm_audit':{
        'status':'READY','critical_defects':defects}}]})
    assert discoveries == [[{'bbox':[1,1,15,9], 'text':''}]]
    assert read(state.evidence(ref)[0])[2]['discovery'][0]['status'] == 'pending'
    assert scoped_ids(state,current,1,1,None) == ['card','field']
    def eligible():
        return normalize_arguments(state,current,'batch_reconstruct_page',{'page':1,'block':1})['object_ids']
    first = eligible()
    inputs = repair_inputs(current,1,first)
    current['repair_rounds'] = {'1':[{'object_ids':first,'inputs':inputs,'committed_ids':['card']}]}
    with pytest.raises(ValueError,match='unchanged evidence'):
        eligible()
    current['marked_plans']['1'][0]['move_with'] = ['field']
    assert eligible() == ['card']
    current['repair_rounds']['1'].append({'object_ids':['card'],'inputs':repair_inputs(current,1,['card'])})
    current['marked_plans']['1'][0]['shape'] = 'roundRect'
    current['review_issues']['1']['field'].append({'finding_id':'new-evidence'})
    assert eligible() == ['field']
