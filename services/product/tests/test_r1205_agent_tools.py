"""Agent registration and whole-object dispatch safety regressions."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from product.materialize import digest,write
from product.object_contracts import ClosedObject
from product.parallel_store import Store
from product.recovery.agent_decision import RegionalRepairAgent,RegisteredTool
from product.recovery.agent_tool_catalog import register_repair_tools
from test_r12_product_pipeline import product_fixture
from test_r1204_agent_decision import Gateway


def test_only_real_matching_adapters_are_registered():
    callbacks={name:lambda output:None for name in ('retry_text_mask_completion','switch_background_candidate','retry_asset_extraction')}
    bindings={name:digest(__file__) for name in callbacks}
    text=register_repair_tools('text',callbacks,bindings)
    asset=register_repair_tools('asset',callbacks,bindings)
    assert [tool.name for tool in text]==['retry_text_mask_completion','switch_background_candidate']
    assert [tool.name for tool in asset]==['retry_asset_extraction']
    with pytest.raises(ValueError,match='must be executable'):
        register_repair_tools('text',{'retry_text_mask_completion':None},{'retry_text_mask_completion':digest(__file__)})


def test_registry_rejects_path_and_unbound_tool_code():
    with pytest.raises(ValueError,match='registration'):
        register_repair_tools('asset',{'C:/run.py':lambda output:None},{'C:/run.py':digest(__file__)})
    with pytest.raises(ValueError,match='hash'):
        register_repair_tools('asset',{'retry_asset_extraction':lambda output:None},{'retry_asset_extraction':'unbound'})


def test_asset_target_is_observed_as_a_whole_object(product_fixture,tmp_path):
    fixture=product_fixture
    graph=fixture.graph.model_copy(deep=True)
    obj=ClosedObject.model_validate({'id':'icon','description':'whole independent graphic','prompt_en':'icon',
        'kind':'icon','bbox':[60,50,20,20],'layers':['layer-01.png'],'components':['whole'],
        'text_policy':'exclude_independent_text','parts':[{'name':'whole','bbox':[60,50,20,20],'layers':['layer-01.png']}]})
    graph.asset.objects=[obj];graph.ownership[obj.id]='RESOLVED'
    failure=tmp_path/'owner-failure.json';write(failure,{'failure':'ASSET_EXTRACTION_UNCERTAIN'})
    gateway=Gateway([('stop_and_fallback',None)])
    def forbidden(output):pytest.fail('observation and stop must not execute an extraction')
    runtime=RegionalRepairAgent(graph,obj.id,fixture.snap['path'],[failure],
        tools=[RegisteredTool('retry_asset_extraction','whole source owner only',digest(__file__),forbidden)],
        gateway=gateway,policy=SimpleNamespace(version='policy'),store=Store(tmp_path/'state'),output=tmp_path/'agent')
    result=runtime.run()
    assert result['status']=='LOCAL_FALLBACK'
    assert gateway.requests[0]['object_kind']=='asset'
    assert gateway.requests[0]['object']==obj.model_dump()
