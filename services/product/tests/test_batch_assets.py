"""Independent batch transactions and strategy ordering on synthetic evidence."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.parallel_contracts import SceneGraph
from product.materialize import digest
from product.recovery import batch_assets
from test_r1205_agent_asset_stage import scene


def test_batch_reuses_local_evidence_and_keeps_first_png_when_second_fails(tmp_path, monkeypatch):
    graph, layers, baseline = scene(tmp_path)
    original_probe = batch_assets.probe
    def probe(current, identifier, *args, **kwargs):
        if identifier == 'second':
            raise ValueError('injected second-object failure')
        return original_probe(current, identifier, *args, **kwargs)
    monkeypatch.setattr(batch_assets, 'probe', probe)
    monkeypatch.setattr(batch_assets, 'verify_office', lambda *a: {'status': 'PASS'})
    monkeypatch.setattr('product.object_stage.infer_sam', lambda *a: (_ for _ in ()).throw(
        AssertionError('acceptable local candidate must stop before SAM')))
    result = batch_assets.recover(graph, layers/'source.png', layers, baseline, tmp_path/'job',
                                  tmp_path/'batch', {'fonts': []}, limit=2, office=True)
    assert [r['status'] for r in result['objects']] == ['VERIFIED_LOCAL', 'LOCAL_FALLBACK']
    assert result['objects'][0]['strategy'] == 'source-halo'
    assert len(result['objects'][0]['attempts']) == 1 and result['model_calls'] == 0
    selected = SceneGraph.model_validate_json((Path(result['combined_ir']).parent/'scene.json').read_text('utf8'))
    assert set(selected.selected) == {'first'}
    reference = selected.selected['first']['asset_reference']
    assert digest(Path(result['combined_ir']).parent/reference['path']) == reference['sha256']


def test_inventory_excludes_stable_assets_and_preserves_unknown_owners(tmp_path):
    graph, _, _ = scene(tmp_path)
    graph.ownership['second'] = 'AMBIGUOUS'
    rows = batch_assets.failure_inventory(graph, tmp_path/'missing-history')
    assert next(r for r in rows if r['object_id'] == 'first')['eligible']
    second = next(r for r in rows if r['object_id'] == 'second')
    assert not second['eligible'] and 'OWNER_UNRESOLVED' in second['failure_categories']
