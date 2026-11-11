import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from product.materialize import write
from product.recovery.failure_agent import route
from test_r12_product_pipeline import product_fixture


def test_infrastructure_and_identical_asset_strategy_never_retry():
    assert route('QA_INFRASTRUCTURE_UNAVAILABLE','text',[])['status']=='UNKNOWN'
    assert route('ASSET_EXTRACTION_UNCERTAIN','asset',[],already_attempted=['halo'])['status']=='NO_CHANGED_EVIDENCE'
    assert route('ASSET_EXTRACTION_UNCERTAIN','asset',[])['tools']==['retry_asset_extraction']


def test_partial_mask_and_complete_mask_failure_choose_distinct_tools(tmp_path):
    file=tmp_path/'residual_report.json'
    write(file,{'status':'REVIEW_REQUIRED','mask_variant':'original','checks':{'complete_visible_scope':False}})
    assert route('MASK_INCOMPLETE','text',[file])['tools']==['retry_text_mask_completion']
    write(file,{'status':'REVIEW_REQUIRED','mask_variant':'semantic_completion','checks':{'complete_visible_scope':True}})
    assert route('BACKGROUND_REPAIR_FAILED','text',[file])['tools']==['switch_background_candidate']
    assert route('MASK_INCOMPLETE','text',[file])['status']=='NO_SAFE_TOOL'


def test_owner_and_role_conflict_use_source_reanalysis():
    assert route('OWNER_CONFLICT','text',[])['tools']==['recheck_text_ownership']
    assert route('SEMANTIC_ROLE_CONFLICT','text',[])['tools']==['recheck_text_ownership']


def test_formal_regional_entry_routes_failure_before_continuing(product_fixture,monkeypatch):
    import json
    from product.parallel_pipeline import run
    from product.recovery import product_pipeline as regional
    from product.recovery.background_adapter import RecoveryBlocked
    from product.recovery import failure_agent
    f=product_fixture
    request=json.loads((f.job/'request.json').read_text('utf8'));request.update(regional_agent=True,agent_skill='fixture-skill')
    write(f.job/'request.json',request)
    original=regional.adaptive_candidates
    def fail_one(graph,identifier,source,layers,directory,**kwargs):
        if identifier=='a':
            directory.mkdir();write(directory/'peer-protection.json',{'source_sha256':graph.source_sha256,'object_id':identifier})
            raise RecoveryBlocked('OWNER_CONFLICT','synthetic owner diagnosis')
        return original(graph,identifier,source,layers,directory,**kwargs)
    monkeypatch.setattr(regional,'adaptive_candidates',fail_one)
    calls=[]
    def repair(*args,**kwargs):calls.append((args[1],kwargs['failure']));return {'status':'LOCAL_FALLBACK'}
    monkeypatch.setattr(failure_agent,'run',repair)
    monkeypatch.setattr('product.recovery.agent_recovery.run',lambda *args:{'status':'NOT_NEEDED'})
    result=run(f.job,provider=f.actor,review_provider=f.actor,compiler=f.compiler,office=f.office,fonts={'fonts':[]})
    assert calls==[('a','OWNER_CONFLICT')]
    assert result['locally_verified_text_regions']==1


def test_rejected_seed_continues_regional_reanalysis(product_fixture,monkeypatch):
    import json
    from product.parallel_pipeline import run
    from product.recovery import verified_seed
    f=product_fixture
    request=json.loads((f.job/'request.json').read_text('utf8'));request['reuse_verified_from']='synthetic-rejected-seed'
    write(f.job/'request.json',request)
    def reject(*args):raise ValueError('new ownership evidence conflicts with prior seed')
    monkeypatch.setattr(verified_seed,'load',reject)
    result=run(f.job,provider=f.actor,review_provider=f.actor,compiler=f.compiler,office=f.office,fonts={'fonts':[]})
    assert result['locally_verified_text_regions']==2
    receipt=json.loads((f.job/'page-01/verified-seed-reanalysis.json').read_text('utf8'))
    assert receipt['whole_page_fallback'] is False
