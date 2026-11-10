import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from product.parallel_store import Store,key
from product.parallel_quality_arbitrator import DefectReview,EditValueReview,validate_review,review_shadow
from product.parallel_candidate_selector import shadow_select


@pytest.fixture
def bundle():
    return {'source_sha256':'a'*64,'candidate_ids':['c-test'],'images':[],'files':{},
            'candidates':[{'candidate_id':'c-test','edit_probes':'PASS','residual':'PASS'}]}


def reviews(bundle,defects=()):
    binding={'candidate_ids':bundle['candidate_ids'],'source_sha256':bundle['source_sha256'],'review_bundle_sha256':key(bundle)}
    d=DefectReview.model_validate({'schema':'CandidateDefectReview/1',**binding,'defects':list(defects),'status':'COMPLETE'})
    c=EditValueReview.model_validate({'schema':'EditValueReview/1',**binding,'defects':[],
        'preferred_candidate_id':'c-test','edit_value_reason':'tested edit','confidence':'MEDIUM','recommendation':'ACCEPT_WITH_MINOR_DEVIATION'})
    return d,c


def test_shadow_never_selects_or_publishes(bundle):
    d,c=reviews(bundle);result=shadow_select(bundle,d,c)
    assert result['status']=='SHADOW_RECOMMENDATION' and result['selected_candidate_id'] is None
    assert result['publish_eligible'] is False


@pytest.mark.parametrize('kind',['wrong_number','ghosting','neighbor_damage','content_error','other'])
def test_minor_label_does_not_bypass_hard_defect(bundle,kind):
    d,c=reviews(bundle,[{'candidate_id':'c-test','type':kind,'severity':'MINOR','location':'title','description':'unsafe'}])
    assert shadow_select(bundle,d,c)['status']=='REJECTED'


def test_moderate_font_difference_rejected(bundle):
    d,c=reviews(bundle,[{'candidate_id':'c-test','type':'font_edge_difference','severity':'MODERATE','location':'title','description':'unsafe'}])
    assert shadow_select(bundle,d,c)['status']=='REJECTED'


@pytest.mark.parametrize('field,value',[('source_sha256','b'*64),('review_bundle_sha256','b'*64),('preferred_candidate_id','not-in-bundle'),('candidate_ids',['c-test','c-test'])])
def test_bad_review_bindings_fail_closed(bundle,field,value):
    _,c=reviews(bundle);c=c.model_copy(update={field:value})
    with pytest.raises(ValueError):validate_review(c,bundle)


def test_unknown_does_not_select(bundle):
    d,c=reviews(bundle);d=d.model_copy(update={'status':'UNKNOWN'})
    assert shadow_select(bundle,d,c)['status']=='REVIEW_UNKNOWN'


def test_no_actor_is_unknown_without_call(bundle,tmp_path):
    store=Store(tmp_path/'state');result=review_shadow(bundle,store,None,tmp_path/'shadow')
    assert result['status']=='REVIEW_UNKNOWN'
    with store.db() as db:assert db.execute('SELECT count(*) FROM calls').fetchone()[0]==0


def test_unknown_paid_response_is_not_resent(bundle,tmp_path,monkeypatch):
    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT',str(tmp_path/'leases'))
    class Actor:
        model='test';calls=0
        def ask(self,*args):self.calls+=1;raise TimeoutError('synthetic')
    actor=Actor();store=Store(tmp_path/'state')
    for name in ('first','second'):
        result=review_shadow(bundle,store,actor,tmp_path/name)
        assert result['status']=='REVIEW_UNKNOWN' and result['identical_paid_retry'] is False
    assert actor.calls==1


def test_two_distinct_protocols_and_cached_replay(bundle,tmp_path,monkeypatch):
    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT',str(tmp_path/'leases'))
    bundle['candidate_ids'].append('c-second');bundle['region_id']='r-test'
    bundle['candidates'].append({'candidate_id':'c-second','edit_probes':'PASS','residual':'PASS'})
    class Actor:
        model='test';calls=[]
        def ask(self,system,context,images,contract):
            self.calls.append(contract.__name__)
            d,c=reviews(bundle)
            return (d if contract is DefectReview else c).model_dump(by_alias=True)
    actor=Actor();store=Store(tmp_path/'state')
    for name in ('first','second'):
        assert review_shadow(bundle,store,actor,tmp_path/name)['status']=='SHADOW_RECOMMENDATION'
    assert actor.calls==['DefectReview','EditValueReview']


def test_single_verified_candidate_does_not_pay_for_comparison(bundle,tmp_path,monkeypatch):
    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT',str(tmp_path/'leases'))
    class Actor:
        model='test';calls=[]
        def ask(self,system,context,images,contract):
            self.calls.append(contract.__name__)
            return reviews(bundle)[0].model_dump(by_alias=True)
    actor=Actor()
    result=review_shadow(bundle,Store(tmp_path/'state'),actor,tmp_path/'shadow')
    assert result['status']=='DEFECT_REVIEW_ONLY' and actor.calls==['DefectReview']


def test_undefined_defect_enum_rejected(bundle):
    with pytest.raises(ValueError):reviews(bundle,[{'candidate_id':'c-test','type':'made_up','severity':'MINOR','location':'x','description':'x'}])
