import json

import numpy as np
import pytest
from types import SimpleNamespace
from PIL import Image

from product.experiment import mark_page, mark_blocks
from product.experiment_assets import contrast_edge_score
from product.parallel_store import Store, BudgetExceeded
from product.recovery.adaptive_policy import AdaptiveBudget
from product.recovery.marked_fields import striped_backing
from product.runtime.marked_routes import apply_move_groups
from product.recovery.marked_tables import table_from_mark


def test_movement_preserves_overlapping_sibling_order():
    def scene(x):
        return {
            'nodes': [
                {'id': 'box', 'z': 1, 'bbox': [0, 0, 60, 30]},
                {'id': 'neighbor', 'z': 2, 'bbox': [x, 7, 12, 14]},
                {'id': 'label', 'z': 3, 'bbox': [7, 5, 29, 17]},
            ],
            'groups': [],
        }

    ir = scene(80)
    apply_move_groups(ir, [['label', 'box']])
    assert ir['groups'][0]['member_ids'] == ['box', 'label']
    ir = scene(9)
    apply_move_groups(ir, [['label', 'box']])
    assert ir['groups'] == [] and [n['z'] for n in ir['nodes']] == [1, 2, 3]


def test_only_complete_ocr_grid_can_become_table():
    image = np.full((100, 190, 3), 250, dtype='uint8')
    for x in (10, 90, 170):
        image[10:81, x] = 30
    for y in (10, 44, 80):
        image[y, 10:171] = 30
    lines = []
    for i, (x, y) in enumerate(((25, 20), (115, 20), (25, 57), (115, 57))):
        image[y : y + 10, x : x + 14] = 25
        lines.append({'id': str(i), 'text': str(i + 1), 'score': 1, 'bbox': [x, y, 14, 10]})
    args = (image, {'bbox': [9, 9, 163, 73]}, lines, SimpleNamespace(ea='Arial', latin='Arial'))
    assert table_from_mark(*args)[2]['rows'] == [['1', '2'], ['3', '4']]
    lines[0]['bbox'] = [70, 20, 35, 10]
    assert table_from_mark(*args) is None


def test_numbered_blocks_preserve_ocr_identity_and_card_parent(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (300, 200), 'white').save(source)
    layout = {'regions': [{'id': 'card', 'bbox': [0, 0, 285, 190], 'label': 'icon'}]}
    texts = [
        {
            'id': f't{i}',
            'bbox': [5 + 30 * (i % 8), 5 + 20 * (i // 8), 20, 12],
            'text': str(i),
            'ocr_line_ids': [f'line{i}'],
        }
        for i in range(40)
    ]
    rows = mark_page(source, layout, texts, tmp_path)
    blocks = list(mark_blocks(source, rows, tmp_path))
    inventory = [m for _, _, f in blocks for m in json.loads(f.read_text(encoding='utf8'))]
    assert max(b['count'] for b, _, _ in blocks) <= 30
    assert sorted(m['mark'] for m in inventory) == list(range(1, 42))
    assert all(m['parent'] == 'card' for m in inventory if m['label'] == 'text')
    assert 'ocr_ids' not in layout['regions'][0]


def test_stripe_requires_two_unobstructed_consistent_sides():
    image = np.broadcast_to(np.arange(60, dtype='uint8')[:, None, None] + 100, (60, 100, 3)).copy()
    image[16:24, 40:50] = 5
    blocked = np.zeros(image.shape[:2], bool)
    ref, fit = striped_backing(image, [28, 10, 32, 20], blocked)
    assert fit['validation_p95_rgb'] == 0 and np.all(ref[18, 42] == 118)
    blocked[17, 23] = True
    assert striped_backing(image, [28, 10, 32, 20], blocked) is None


def test_new_background_proxy_is_unavailable_for_opaque_image():
    assert contrast_edge_score(Image.new('RGBA', (18, 18), 'red')) is None
    image = np.zeros((18, 18, 4), dtype='uint8')
    image[4:14, 4:14] = [255, 0, 0, 128]
    image[5:13, 5:13] = [255, 0, 0, 255]
    assert contrast_edge_score(Image.fromarray(image))['p90'] == 0
    image[4, :, 1] = 100
    assert contrast_edge_score(Image.fromarray(image))['p90'] > 0


def test_page_reservation_does_not_exhaust_later_page(tmp_path):
    store = Store(tmp_path / 'db')

    def budget(page):
        return AdaptiveBudget(store, {'model_calls': 2}, model_page=(page, 2))

    assert budget(1).reserve('model_calls', 'A')
    assert not budget(1).reserve('model_calls', 'A')
    with pytest.raises(BudgetExceeded, match='page 1'):
        budget(1).reserve('model_calls', 'B')
    assert budget(2).reserve('model_calls', 'B')
    assert budget(1).report()['counts']['model_calls'] == 2
