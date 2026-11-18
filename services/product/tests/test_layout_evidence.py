from types import SimpleNamespace

from product import layout_evidence


def test_layout_crop_seeds_keep_pictorial_regions_still_in_plate():
    layout = {'regions': [
        {'id': 'l-001', 'label': 'image', 'score': 0.9, 'bbox': [0, 0, 100, 100]},
        {'id': 'l-002', 'label': 'text', 'score': 0.9, 'bbox': [200, 0, 50, 20]},
        {'id': 'l-003', 'label': 'chart', 'score': 0.3, 'bbox': [300, 0, 50, 50]},
        {'id': 'l-004', 'label': 'header_image', 'score': 0.9, 'bbox': [400, 0, 100, 40]},
    ]}
    graph = SimpleNamespace(selected={
        'a-1': {'kind': 'asset', 'bbox': [400, 0, 90, 40]},          # logo already a movable asset
        's-1': {'kind': 'surface', 'bbox': [0, 0, 500, 200]},        # backing surfaces never block
        't-1': {'kind': 'text', 'bbox': [10, 10, 30, 10]},           # small native label over photo
    })
    seeds = layout_evidence.crop_seeds(layout, graph)
    assert [s['id'] for s in seeds] == ['layout:l-001']
    assert seeds[0]['kind'] == 'layout:image'
