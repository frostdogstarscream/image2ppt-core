from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest

from product.parallel_edit_probe import _bounded_single_glyph, _single_glyph_probe


def pictures():
    empty = np.full((30, 32, 3), 255, np.uint8)
    original = empty.copy()
    edited = empty.copy()
    original[8:16, 10:12] = 20
    edited[9:11, 10:22] = 20
    return original, edited, empty


@pytest.mark.parametrize('issue',['lost_dot','new_ink_outside','no_change','missing_ink','edge_of_slide'])
def test_single_character_never_hides_overhanging_marks_or_missing_edit(issue):
    before, after, deleted = pictures()
    frame = (8, 5, 25, 20)
    if issue=='lost_dot':
        before[20, 10] = 0
    if issue=='new_ink_outside':
        after[20, 10] = 0
    if issue=='no_change':
        after = before
    if issue=='missing_ink':
        after = deleted
    if issue=='edge_of_slide':
        frame = (0, 5, 25, 20)
    assert not _bounded_single_glyph(before, after, deleted, frame)


@pytest.mark.parametrize('clipped',[False,True])
def test_expansion_must_render_the_same_complete_character(tmp_path, monkeypatch, clipped):
    monkeypatch.setitem(sys.modules,'pywintypes',SimpleNamespace(com_error=RuntimeError))
    before, after, deleted = pictures()
    blank = tmp_path/'delete.png'
    expanded = tmp_path/'expanded.png'
    Image.fromarray(deleted).save(blank)
    ink = SimpleNamespace(BoundLeft=8, BoundTop=5, BoundWidth=18, BoundHeight=9,
                          Lines=lambda:SimpleNamespace(Count=1))
    shape = SimpleNamespace(Left=8, Top=5, Width=17, Height=15, TextFrame=SimpleNamespace(TextRange=ink))
    wide = after.copy()
    if clipped:
        wide[10,25] = 0
    slide = SimpleNamespace(Export=lambda path, *unused:Image.fromarray(wide).save(path))
    document = SimpleNamespace(Saved=False)
    layout = {'overflow':True}
    _single_glyph_probe(shape, slide, document, before, after, blank, (8,5,25,20), expanded, layout)
    assert layout['overflow'] is clipped
    assert document.Saved
    if not clipped:
        assert layout['reported_overflow'] and layout['delete_sha256'] and layout['expanded_preview_sha256']
