from types import SimpleNamespace

import numpy as np
import pytest

from product.recovery.native_shapes import try_fit_native_shape_v2
from product.recovery.region_text import plan_regions


def row_header(corners, *, chamfer=False, unknown=False):
    h, w = 92, 115
    yy, xx = np.mgrid[:h, :w] + 0.5
    radius = 14
    coverage = np.ones((h, w))
    for index in corners:
        x = xx if index % 2 == 0 else w - xx
        y = yy if index < 2 else h - yy
        tile = (x < radius) & (y < radius)
        distance = (x + y - radius) / np.sqrt(2) if chamfer else radius - np.hypot(x-radius, y-radius)
        coverage[tile] = np.clip(distance[tile]+0.5, 0, 1)
    parent = np.full((h, w, 3), 249)
    fill = np.stack([36 + 6*xx/w, 87 + 9*xx/w, 178 + 8*xx/w], axis=2)
    source = np.rint(parent + coverage[:, :, None]*(fill-parent)).astype('uint8')
    if unknown:
        source[32:40, 41:49] = [248, 69, 21]
    return source, parent, [0, 0, w, h], np.zeros((h, w), dtype=bool)


@pytest.mark.parametrize("corners", [(0, 2), (1, 3), (0, 1), (2, 3)])
def test_source_measured_label_allows_matching_adjacent_arcs(corners):
    source, parent, box, mask = row_header(corners)
    assert try_fit_native_shape_v2(source, box, parent, [], mask) is None
    fit = try_fit_native_shape_v2(source, box, parent, [], mask, measured_label=True)
    assert fit is not None and fit['geometry'] == 'roundRect'
    assert fit['erase_mask_local'][30, 31]
    corner_offset = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
    for index in corners:
        assert not fit['erase_mask_local'][corner_offset[index]]


@pytest.mark.parametrize("corners,chamfer,unknown", [
    ((0, 2), True, False), ((0, 3), False, False), ((0, 3), True, False),
    ((0, 2), False, True), ((1,), False, False),
])
def test_reject_bevels_diagonal_corners_and_unowned_marks(corners, chamfer, unknown):
    source, parent, box, mask = row_header(corners, chamfer=chamfer, unknown=unknown)
    assert try_fit_native_shape_v2(source, box, parent, [], mask, measured_label=True) is None


def test_text_context_only_uses_confirmed_members_and_respects_photos_and_tables():
    title = SimpleNamespace(id='title', bbox=[10, 10, 18, 6])
    extra = SimpleNamespace(id='extra', bbox=[10, 20, 18, 6])
    graph = SimpleNamespace(source_sha256='test-page', image_size=[160, 80],
                            text=SimpleNamespace(texts=[title, extra]))
    face = {'id':'header', 'measured_label_face':[3, 3, 38, 35], 'move_with':['title']}
    card = {'category':'card_shape', 'id':'row', 'bbox':[1, 2, 150, 45]}
    source = np.zeros((80, 160, 3), np.uint8)
    groups = plan_regions(graph, [title, extra], {'elements':[card]}, source, [face])
    assert [(g['surface'],g['members']) for g in groups] == [('header',['title']),('row',['extra'])]
    for category in ('table','logo','chart'):
        container = {**card,'category':category}
        assert plan_regions(graph, [title], {'elements':[container]}, source, [face]) == []
