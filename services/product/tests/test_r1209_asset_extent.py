"""Source/SAM localization remains separate from original ownership gates."""
from pathlib import Path
import sys
import json

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.recovery.asset_retry import sam_extent_unmix, localized_owner_object
from product.recovery.single_asset import probe
from product.object_contracts import ClosedObject


def object_definition():
    return ClosedObject(id='arbitrary-icon-id', description='complete outlined object',
        prompt_en='complete outlined object', kind='icon', bbox=[30, 42, 35, 25],
        layers=['foreground.png'], components=['whole'], text_policy='exclude_independent_text',
        parts=[{'name': 'whole', 'bbox': [30, 42, 35, 25], 'layers': ['foreground.png']}])


def pixels():
    source = np.full((100, 100, 3), 245, np.uint8)
    source[25:55, 30:60] = [20, 80, 150]
    source[29:51, 34:56] = 245
    sam = np.zeros((100, 100), bool)
    sam[45:55, 30:60] = True
    return source, sam


def test_partial_sam_traces_whole_source_component_and_preserves_scene_definition():
    obj = object_definition()
    before = obj.model_dump()
    source, sam = pixels()
    rgba, metadata = sam_extent_unmix(obj, source, np.zeros((100, 100), bool), None,
                                      {'mask': sam, 'index': 0, 'score': .9})
    assert rgba[25, 30, 3] > 0 and rgba[54, 59, 3] > 0
    assert not rgba[40, 45, 3]
    owner = localized_owner_object(obj, {'source_unmix': metadata})
    assert owner.bbox == [30, 25, 30, 30]
    assert obj.model_dump() == before


def test_component_crossing_protected_peer_stays_blocked():
    source, sam = pixels()
    protected = np.zeros((100, 100), bool)
    protected[35, 30] = True
    with pytest.raises(ValueError, match='protected peer'):
        sam_extent_unmix(object_definition(), source, protected, None,
                          {'mask': sam, 'index': 0, 'score': .9})


def test_missing_cached_sam_with_zero_budget_never_dispatches(tmp_path, monkeypatch):
    from test_r1205_agent_asset_stage import scene
    graph, layers, baseline = scene(tmp_path)
    monkeypatch.setattr('product.object_stage.infer_sam', lambda *a: pytest.fail('new SAM call prohibited'))
    result = probe(graph, 'first', layers/'source.png', layers, baseline, tmp_path/'out',
                   strategy='sam', sam_cache=tmp_path/'empty', allow_model_call=False)
    assert result['status'] == 'UNKNOWN' and result['model_calls'] == 0
