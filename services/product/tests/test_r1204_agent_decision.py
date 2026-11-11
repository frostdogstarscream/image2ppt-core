"""Synthetic control-flow tests. Real VLM/Office acceptance is recorded separately."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from product.materialize import digest,write
from product.parallel_store import Store
from product.recovery.agent_decision import RegionalRepairAgent,RegisteredTool,ToolCandidate
from product.recovery.region_candidates import adaptive_candidates
from test_r12_product_pipeline import product_fixture


class Gateway:
    def __init__(self, actions):
        self.actions=iter(actions);self.requests=[]

    def ask(self,purpose,policy,context,images,contract):
        self.requests.append(context)
        action,tool=next(self.actions)
        binding=context['binding']
        return contract.model_validate({'schema':'RegionalRepairDecision/1',
            **{k:binding[k] for k in ('source_sha256','scene_sha256','evidence_sha256','object_id')},
            'action':action,'tool':tool,'reason':'synthetic decision for behavioral regression'})


def agent(product_fixture,tmp_path,actions,*,execute=None,output='agent'):
    f=product_fixture
    failure=tmp_path/'failure.json';write(failure,{'failure':'MASK_INCOMPLETE'})
    calls=[]
    def recover(folder):
        calls.append(1)
        stream=adaptive_candidates(f.graph,'a',f.snap['path'],None,folder/'background')
        row=next(stream);stream.close()
        return ToolCandidate(f.graph,folder/'background'/row['candidate'])
    gateway=Gateway(actions)
    runtime=RegionalRepairAgent(f.graph,'a',f.snap['path'],[failure],
        tools=[RegisteredTool('recover_text','Source-bound background generation',digest(__file__),execute or recover)],
        gateway=gateway,policy=SimpleNamespace(version='policy-hash'),store=Store(tmp_path/'agent-store'),
        output=tmp_path/output)
    return runtime,calls,gateway


def test_real_tool_receipt_is_read_before_vlm_accepts(product_fixture,tmp_path):
    runtime,calls,gateway=agent(product_fixture,tmp_path,[('execute_tool','recover_text'),('accept_candidate',None)])
    result=runtime.run()
    assert result['status']=='CANDIDATE_READY' and calls==[1]
    assert result['selected']['office']=='NOT_RUN' and not result['published']
    assert gateway.requests[1]['events'][0]['result']['status']=='CANDIDATE_READY'
    assert gateway.requests[1]['image_roles']==['full_page_source','source_region','background_after_text_removal']
    receipt=json.loads((tmp_path/'agent/agent-repair.json').read_text('utf8'))
    assert receipt['model_decisions']==2 and receipt['tools_executed']==1


def test_model_cannot_invent_tool_or_grant_hard_gate(product_fixture,tmp_path):
    runtime,calls,_=agent(product_fixture,tmp_path,[('execute_tool','delete_source')])
    result=runtime.run()
    assert result['status']=='INVALID' and not calls


def test_candidate_cannot_be_accepted_before_execution(product_fixture,tmp_path):
    runtime,calls,_=agent(product_fixture,tmp_path,[('accept_candidate',None)])
    assert runtime.run()['status']=='INVALID' and not calls


def test_repeated_tool_is_not_executed_twice(product_fixture,tmp_path):
    runtime,calls,_=agent(product_fixture,tmp_path,[('execute_tool','recover_text'),('execute_tool','recover_text')])
    assert runtime.run()['status']=='LOCAL_FALLBACK' and calls==[1]


def test_timeout_remains_unknown_without_accepting_candidate(product_fixture,tmp_path):
    def unavailable(folder):raise TimeoutError('injected unavailable tool')
    runtime,_,gateway=agent(product_fixture,tmp_path,[('execute_tool','recover_text')],execute=unavailable)
    result=runtime.run()
    assert result['status']=='UNKNOWN' and len(gateway.requests)==1
    assert 'selected' not in result


def test_tampered_cached_tool_is_rejected(product_fixture,tmp_path):
    runtime,_,_=agent(product_fixture,tmp_path,[])
    result=runtime.execute('recover_text')
    file=Path(next(iter(result['files'])))
    file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='cached tool result changed'):
        runtime.execute('recover_text')


def test_scene_binding_uses_validated_numeric_representation(product_fixture,tmp_path):
    product_fixture.graph.text.texts[0].bbox=[15,15,40,35]
    runtime,_,_=agent(product_fixture,tmp_path,[])
    runtime.checked()
