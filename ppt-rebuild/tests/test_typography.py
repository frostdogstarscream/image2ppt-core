"""Contracts, immutability, glyph safeguards and bounded provider calls."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from _helpers import SRC
import typography_plan as planning
from typography_fit import assert_frozen,choose,verify_selected


def sample():
    p={'text':'项目27项','font':{'ea':'微软雅黑','latin':'Arial','size_pt':18},'bold':False,'color':'112233',
       'runs':[{'text':'项目'},{'text':'27项','font':{'ea':'微软雅黑','latin':'Arial','size_pt':24},'color':'0062AA','bold':True}]}
    return {'canvas':{'source_px':[200,100],'target_px':[400,200]},'nodes':[{'id':'t1','type':'text','bbox':[10,20,160,40],
            'text':p['text'],'font':p['font'],'bold':False,'color':p['color'],'paragraphs':[p],'evidence_ids':['e1']}],
            'evidence':[{'id':'e1','bbox':[10,20,160,40]}],'assets':[],'groups':[]}


def contract():
    ir=sample(); evidence={'source_sha256':'source','input_ir_sha256':'ir','targets':planning.targets(ir)}
    manifest={'fonts':[{'family':f,'bold':b} for f in ['微软雅黑','Arial'] for b in [False,True]]}
    style={'id':'s','role':'body','font_family':'微软雅黑','font_latin':'Arial','font_size_px':12,
           'bold':False,'color':'112233','line_spacing_px':0,'confidence':.8,'evidence':'source glyphs'}
    plan={'schema':'FontPlan/2','source_sha256':'source','input_ir_sha256':'ir','styles':[style],
          'bindings':[{'target_id':t['id'],'style_id':'s'} for t in evidence['targets']],'structure_review':[]}
    return plan,evidence,manifest


def test_target_source_em_and_mixed_run_are_explicit():
    t=planning.targets(sample())
    assert [x['id'] for x in t]==['t1/p0','t1/p0/r0','t1/p0/r1']
    assert t[0]['size_px']==12 and t[-1]['size_px']==16
    assert t[-1]['text']=='27项' and t[-1]['bold']


@pytest.mark.parametrize('mutation',['missing_parent','duplicate','unknown_font','changed_source','nan','layout','unavailable_weight'])
def test_contract_rejects_invalid_model_without_repair(mutation):
    p,e,m=contract()
    if mutation=='missing_parent':p['bindings']=p['bindings'][1:]
    elif mutation=='duplicate':p['bindings'][0]=p['bindings'][1]
    elif mutation=='unknown_font':p['styles'][0]['font_family']='invented'
    elif mutation=='changed_source':p['source_sha256']='other'
    elif mutation=='nan':p['styles'][0]['font_size_px']=float('nan')
    elif mutation=='layout':p['bbox']=[0,0,1,1]
    else:m['fonts']=[f for f in m['fonts'] if f['family']!='微软雅黑' or f['bold']]
    frozen=copy.deepcopy(p)
    with pytest.raises(ValueError):planning.validate(p,e,m)
    if mutation!='nan':assert frozen==p


def test_valid_plan_and_frozen_run_text():
    p,e,m=contract(); assert planning.validate(p,e,m)==p
    before=sample(); after=copy.deepcopy(before)
    after['nodes'][0]['paragraphs'][0]['runs'][1]['font']['size_pt']=23
    assert_frozen(before,after)
    after['nodes'][0]['paragraphs'][0]['runs'][1]['text']='28项'
    with pytest.raises(ValueError):assert_frozen(before,after)


def test_frozen_geometry_even_when_font_changes():
    a=sample();b=copy.deepcopy(a);b['nodes'][0]['font']['size_pt']=19
    b['nodes'][0]['bbox'][0]+=1
    with pytest.raises(ValueError):assert_frozen(a,b)


def test_lower_mae_cannot_accept_missing_strokes(monkeypatch):
    import typography_fit
    monkeypatch.setattr(typography_fit,'fits',lambda *args:True)
    before=sample();after=copy.deepcopy(before);after['nodes'][0]['font']['size_pt']=12
    bg=np.full((100,200,3),255,np.uint8);src=bg.copy();src[30:45,25:85:3]=0
    old=src.copy();old[30:45,25:85:3]=100
    candidate=bg.copy();candidate[30:45,25:70:3]=0
    selected,rows=choose(before,after,src,old,candidate,bg,{})
    assert not rows[0]['selected']
    assert selected==before


def test_final_changes_outside_text_fail():
    before=sample();bg=np.full((100,200,3),255,np.uint8);final=bg.copy();final[0,0]=0
    with pytest.raises(ValueError,match='outside'):
        verify_selected(before,before,bg,bg,final,bg,[{'id':'t1','selected':False}])


def test_provider_attempts_cached_and_counted(tmp_path,monkeypatch):
    import credentials
    monkeypatch.setattr(credentials,'load',lambda p:'test-only-secret')
    plan,e,m=contract();calls=[]
    class Fake:
        def __init__(self,settings):assert settings.transport_attempts==1
        async def chat(self,messages):
            calls.append(messages)
            p=copy.deepcopy(plan)
            if len(calls)==1:p['bindings']=p['bindings'][1:]
            raw={'choices':[{'message':{'content':json.dumps(p)}}]}
            return SimpleNamespace(raw=raw,usage={'total_tokens':12},model='qwen3.8-max',finish_reason='stop',text=json.dumps(p))
    monkeypatch.setattr(planning,'MomaClient',Fake)
    directory=tmp_path/'01'; budget=tmp_path/'budget.json';messages=[{'role':'user','content':'fixed'}]
    value=planning.request_plan(directory,messages,lambda p:planning.validate(p,e,m),budget)
    assert value==plan and len(calls)==2
    assert 't1/p0' in calls[1][-1]['content']
    assert len(planning.read(budget)['calls'])==2
    planning.request_plan(directory,messages,lambda p:planning.validate(p,e,m),budget)
    assert len(calls)==2
    for f in tmp_path.rglob('*.json'):assert 'test-only-secret' not in f.read_text(encoding='utf8')
    with pytest.raises(ValueError,match='inputs changed'):
        planning.request_plan(directory,[{'role':'user','content':'different'}],lambda p:p,budget)


def test_global_styles_partition_is_enforced(tmp_path,monkeypatch):
    p,_,_=contract();invalid={'schema':'DeckStyles/1','groups':[{'id':'a','members':['01:s','01:s'],'canonical':'01:s','reason':'same'}],'notes':[]}
    def fake(directory,messages,validator,budget):return validator(invalid)
    monkeypatch.setattr(planning,'request_plan',fake)
    with pytest.raises(ValueError,match='partition'):
        planning.global_plan({'01':p},tmp_path/'global',tmp_path/'budget.json')


def test_real_client_single_attempt_does_not_hide_parameter_retry():
    import asyncio
    import httpx
    from image2ppt_planner.client import MomaClient,ModelError
    from image2ppt_planner.config import Settings
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(400,json={'error':'unsupported parameter'})
    client=MomaClient(Settings(api_key='test-only-secret',transport_attempts=1),httpx.MockTransport(handler))
    with pytest.raises(ModelError,match='HTTP 400'):
        asyncio.run(client.chat([{'role':'user','content':'synthetic'}]))
    assert len(calls)==1


def test_dpapi_roundtrip_does_not_store_plaintext():
    import os
    if os.name!='nt':pytest.skip('Windows DPAPI')
    from credentials import crypt
    original=b'{"moma":"synthetic-not-a-provider-key"}'
    protected=crypt(original)
    assert b'synthetic-not-a-provider-key' not in protected
    assert crypt(protected,decrypt=True)==original
