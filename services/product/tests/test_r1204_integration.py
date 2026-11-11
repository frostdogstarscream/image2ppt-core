"""Regional coverage stopping is independent of unknown source remainder."""
import json

from test_r12_product_pipeline import product_fixture
from product.materialize import write
from product.parallel_pipeline import run
from product.recovery import product_pipeline as regional


def test_verified_coverage_goal_stops_before_next_region(product_fixture,monkeypatch):
    f=product_fixture
    request=json.loads((f.job/'request.json').read_text('utf8'))
    request['text_region_goal']=1
    write(f.job/'request.json',request)
    original=regional.adaptive_candidates;attempted=[]
    def generate(graph,identifier,*args,**kwargs):
        attempted.append(identifier)
        return original(graph,identifier,*args,**kwargs)
    monkeypatch.setattr(regional,'adaptive_candidates',generate)
    result=run(f.job,provider=f.actor,review_provider=f.actor,compiler=f.compiler,office=f.office,fonts={'fonts':[]})
    assert result['locally_verified_text_regions']==1
    assert attempted==['a']
    assert not result['automatic_publication']
