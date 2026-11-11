"""Behavioral regressions for bounded lazy search and non-waivable hard gates."""
import json
from pathlib import Path

import numpy as np
import pytest

from test_r12_product_pipeline import product_fixture

from product.materialize import write
from product.parallel_pipeline import run
from product.parallel_store import Store, BudgetExceeded
from product.recovery import product_pipeline as regional
from product.recovery.adaptive_policy import AdaptiveBudget, QualityReview, quality_status
from product.recovery.region_candidates import adaptive_candidates


def test_first_acceptable_background_never_dispatches_lama(product_fixture):
    f=product_fixture
    def forbidden(*args):
        pytest.fail('accepted candidate must not request a later model')
    stream=adaptive_candidates(f.graph,'a',f.snap['path'],None,f.job/'early',inpaint=forbidden)
    first=next(stream)
    assert first['status']=='PRECHECK_PASS'
    stream.close()
    report=json.loads((f.job/'early/index.json').read_text('utf8'))
    assert report['generated_candidates']==1
    assert all(r.get('tool')!='big-lama' for r in report['routes'])


@pytest.mark.parametrize('lama_times_out',[False,True])
def test_explicit_background_failure_advances_to_second_valid_tool(product_fixture,monkeypatch,lama_times_out):
    f=product_fixture
    calls=[]
    def inpaint(source,mask):
        calls.append(1)
        if lama_times_out:raise TimeoutError('injected unavailable service')
        raw=source.copy();raw[mask]=235
        return raw
    monkeypatch.setattr(regional,'InpaintService',lambda *args:inpaint)
    from product.recovery import region_candidates
    original_stream=region_candidates.adaptive_candidates
    def with_inpaint(*args,**kwargs):
        kwargs['inpaint']=inpaint
        isolated=list(args);isolated[3]=None
        return original_stream(*isolated,**kwargs)
    monkeypatch.setattr(regional,'adaptive_candidates',with_inpaint)
    original=regional.stage_and_verify
    def reject_first(trial,*args,**kwargs):
        if Path(trial).name.endswith('--gradient') and Path(trial).parent.name=='a':
            output=args[3];output.mkdir()
            result={'regional_automatic_checks':'FAIL','failure':'BACKGROUND_REPAIR_FAILED'}
            write(output/'acceptance.json',result)
            return result
        return original(trial,*args,**kwargs)
    monkeypatch.setattr(regional,'stage_and_verify',reject_first)
    result=run(f.job,provider=f.actor,review_provider=f.actor,compiler=f.compiler,office=f.office,fonts={'fonts':[]})
    assert result['locally_verified_text_regions']==(1 if lama_times_out else 2)
    assert calls==[1]
    first=result['regions'][0][0]
    if lama_times_out:
        assert first['failure']=='QA_INFRASTRUCTURE_UNAVAILABLE' and first['quality']=='UNKNOWN'
        assert result['regions'][0][1]['state']=='VERIFIED_LOCAL'
        return
    assert first['selected_background'].endswith('--big-lama')
    assert first['early_stop']['background_candidates_consumed']==2


def test_minor_deviation_never_overrides_hard_failure():
    visual={'passed':True,'findings':[], 'minor_deviations':[{'category':'font','description':'slight weight difference'}]}
    assert quality_status(True,visual)=='ACCEPT_WITH_MINOR_DEVIATION'
    assert quality_status(False,visual)=='HARD_GATE_FAILED'
    assert quality_status(True,{'status':'UNKNOWN'})=='UNKNOWN'


def test_budget_is_durable_and_blocks_new_dispatch(tmp_path):
    store=Store(tmp_path/'state')
    first=AdaptiveBudget(store,{'model_calls':1})
    assert first.reserve('model_calls',{'input':'one'})
    restored=AdaptiveBudget(store,{'model_calls':1})
    assert not restored.reserve('model_calls',{'input':'one'})
    with pytest.raises(BudgetExceeded):
        restored.reserve('model_calls',{'input':'two'})
    assert restored.report()['counts']['model_calls']==1


def test_invalid_nonfinite_budget_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        AdaptiveBudget(Store(tmp_path/'state'),{'cost_units':float('inf')})


def test_budget_exhaustion_retains_previous_region(product_fixture):
    f=product_fixture
    request=json.loads((f.job/'request.json').read_text('utf8'))
    request['adaptive_budget']={'cost_units':.06}
    write(f.job/'request.json',request)
    result=run(f.job,provider=f.actor,review_provider=f.actor,compiler=f.compiler,office=f.office,fonts={'fonts':[]})
    assert result['locally_verified_text_regions']==1
    assert result['regions'][0][0]['state']=='VERIFIED_LOCAL'
    assert result['regions'][0][1]['failure']=='QA_INFRASTRUCTURE_UNAVAILABLE'
    assert not result['automatic_publication']


def test_visual_minor_contract_rejects_content_waiver():
    payload={'schema':'VisualReview/1','source_sha256':'a','plan_sha256':'b','preview_sha256':'c',
             'passed':True,'checked':['wording','structure','font','emphasis','layout','graphics'],'findings':[],
             'minor_deviations':[{'category':'wording','description':'missing punctuation','reason_to_accept':'small'}]}
    with pytest.raises(ValueError):
        QualityReview.model_validate(payload)


def test_severe_visual_failure_rolls_back_only_named_region(product_fixture,monkeypatch):
    f=product_fixture
    original=f.actor.ask
    def review(system,context,images,contract):
        result=original(system,context,images,contract)
        selected={node['id'] for node in context['scene']['nodes']}
        if 'b' in selected:
            result.update(passed=False,findings=[{'category':'layout','node_ids':['b'],'description':'severe clipping'}])
        else:
            result['minor_deviations']=[{'category':'font','node_ids':['a'],'description':'slightly different weight',
                                         'reason_to_accept':'readable, style remains close'}]
        return result
    monkeypatch.setattr(f.actor,'ask',review)
    result=run(f.job,provider=f.actor,review_provider=f.actor,compiler=f.compiler,office=f.office,fonts={'fonts':[]})
    assert result['locally_verified_text_regions']==1
    assert result['regions'][0][0]['quality']=='ACCEPT_WITH_MINOR_DEVIATION'
    assert result['regions'][0][1]['state']=='LOCAL_FALLBACK'


@pytest.mark.parametrize('same_card',[True,False])
def test_source_backed_wrapped_body_group_requires_shared_card(product_fixture,same_card):
    from product.contracts import Shape
    from product.recovery.text_groups import coalesce
    f=product_fixture
    text=f.graph.text.model_copy(deep=True)
    text.texts[0].bbox=[15.,15.,40.,15.]
    text.texts[1].bbox=[15.,32.,40.,15.]
    surface=f.graph.surface.model_copy(deep=True)
    surface.shapes=[Shape(id='card',bbox=[10.,10.,60.,50. if same_card else 20.],z=0,geometry='rect',fill='EBEBEB')]
    ocr={'lines':[{'id':'l1','text':text.texts[0].text},{'id':'l2','text':text.texts[1].text}]}
    result=coalesce({'text':text,'asset':f.graph.asset,'surface':surface},ocr,Store(f.job/'groups'))
    if same_card:
        assert len(result['text'].texts)==1
        assert result['text'].texts[0].text=='正文甲\n正文乙'
        assert result['text'].members['a']==['l1','l2']
    else:
        assert len(result['text'].texts)==2
    assert len(text.texts)==2


def test_restart_reuses_identical_failed_region_receipt(product_fixture,monkeypatch):
    f=product_fixture
    original=regional.adaptive_candidates
    calls=[]
    def fail_first(graph,identifier,*args,**kwargs):
        calls.append(identifier)
        if identifier=='a':
            from product.recovery.background_adapter import RecoveryBlocked
            raise RecoveryBlocked('MASK_INCOMPLETE','source evidence incomplete')
        return original(graph,identifier,*args,**kwargs)
    monkeypatch.setattr(regional,'adaptive_candidates',fail_first)
    def interrupted(*args):
        raise RuntimeError('composition interruption')
    with pytest.raises(RuntimeError,match='composition interruption'):
        run(f.job,provider=f.actor,review_provider=f.actor,compiler=interrupted,office=f.office,fonts={'fonts':[]})
    assert calls==['a','b']
    result=run(f.job,provider=f.actor,review_provider=f.actor,compiler=f.compiler,office=f.office,fonts={'fonts':[]})
    assert calls==['a','b']
    assert result['regions'][0][0]['resumed_from_failure_receipt']
    assert result['regions'][0][1]['resumed_from_verified_receipt']
