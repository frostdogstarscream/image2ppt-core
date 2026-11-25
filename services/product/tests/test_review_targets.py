"""Independent audit findings must resolve against the reviewed source inventory."""
import pytest
from types import SimpleNamespace

from product.materialize import write
from product.runtime.host_evidence import compact, normalize_arguments, read, scoped_ids
from product.runtime.independent_review import VLMIndependentAuditReply, classify_audit
from product.runtime.review_targets import explain_empty_retry, expose_review, locate, repair_inputs
from product.runtime.state import RuntimeState


ROWS = [
    {'mark':1, 'id':'field', 'label':'text', 'bbox':[20,10,60,20]},
    {'mark':2, 'id':'card', 'label':'card_frame', 'bbox':[100,20,50,60]},
]


def test_scoped_noop_explains_only_observed_bindings_without_claiming_an_edit():
    graph = SimpleNamespace(selected={'field':{'kind':'text'}, 'icon':{'kind':'asset'}},
                            text=SimpleNamespace(texts=[SimpleNamespace(id='field'),SimpleNamespace(id='draft')]),
                            ownership={'field':'RESOLVED','draft':'AMBIGUOUS'})
    routes = [{'id':'alias','route':'text','target_ids':['field']},
              {'id':'layout','route':'text','target_ids':[]},
              {'id':'picture','route':'asset','target_ids':['candidate']}]
    failures = [{'object_id':'candidate','failure':'OWNER_CONFLICT','reason':'ownership is ambiguous'}]
    ids = ['field','alias','layout','draft','new','icon','picture']
    results = explain_empty_retry(ids,routes,graph,failures)
    assert [r['reason'] for r in results]==['SELECTED_TEXT_NOT_REOPENED_BY_BATCH',
        'ROUTE_BOUND_ONLY_TO_SELECTED_OBJECTS','TEXT_MARK_HAS_NO_BOUND_SCENE_TEXT','TEXT_OWNERSHIP_UNRESOLVED',
        'NO_CANDIDATE_RECORDED_CAUSE_UNKNOWN','SELECTED_OBJECT_NO_CHANGE_STAGED','EXECUTOR_REPORTED_FAILURE']
    assert results[1]['existing_selected_ids']==['field'] and results[2]['existing_selected_ids']==[]
    assert results[-1]['executor_failures'][0]['failure']=='OWNER_CONFLICT'
    outcome = compact('batch_reconstruct_page',{'status':'NO_CANDIDATES_STAGED','new_object_count':0,
                                               'target_outcomes':results})
    assert outcome['new_object_count']==0 and outcome['target_outcomes']==results


def test_font_severity_uses_explicit_visual_impact_and_preserves_uncertainty():
    base={'severity':'critical','category':'font_or_shade_deviation','description':'same wording in every test'}
    def classify(*rows, passed=False):
        return classify_audit(VLMIndependentAuditReply(passed=passed,defects=list(rows)))
    cosmetic={**base,'text_impact':'appearance_only'}
    critical,minor,passed=classify(cosmetic)
    assert passed and not critical and minor[0]['severity']=='minor'
    for impact in ('content_obscured','missing_strokes','wrong_or_missing_content'):
        critical,minor,passed=classify(cosmetic,{**base,'severity':'minor','text_impact':impact},passed=True)
        assert not passed and critical[0]['severity']=='critical' and len(minor)==1
    assert classify(base)[0][0]['text_impact']=='unknown'
    assert not classify()[2]
    assert not classify({**cosmetic,'category':'missing_content'},passed=True)[2]


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
