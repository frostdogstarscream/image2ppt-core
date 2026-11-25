"""Sibling recovery uses exposed parent pixels and preserves an independent neighbour."""
import numpy as np

from product.recovery import shared_surface_backing as backing
from product.recovery.layerd_adapter import LowerBackground


def _group(parent):
    children = [(34, 43, 20, 29), (69, 35, 20, 37), (104, 27, 20, 45)]
    source = parent.copy()
    for x, y, w, h in children:
        source[y:y+h, x:x+w] = (12, 26, 110)
    return source, children, [('box', (18, 12, 127, 74)), *(('box', b) for b in children)]


def test_siblings_share_gradient_including_on_scoped_retry():
    yy, xx = np.mgrid[:100, :170]
    paper = np.stack([100+xx//3, 130+yy//2, 200-xx//4], axis=-1).astype('uint8')
    source, children, candidates = _group(paper)
    # The next opaque card begins immediately beyond the right edge of this face.
    source[:, 145:] = (225, 12, 4)
    protected = np.zeros(source.shape[:2], bool)
    protected[20:63, 59:63] = True
    source[protected] = (1, 160, 1)
    patches = backing.shared_face_patches(source, candidates, [children], protected)
    retry = backing.shared_face_patches(source, candidates, [children], protected, active_boxes={children[1]})
    assert set(patches) == set(children) and set(retry) == {children[1]}
    for box, (field, info) in patches.items():
        x, y, w, h = box
        assert info['method'] == 'shared-parent-samples/1'
        assert info['parent_bbox'] == [18, 12, 127, 74]
        assert np.max(np.abs(field - paper[y:y+h, x:x+w])) < 2
    np.testing.assert_array_equal(patches[children[1]][0], retry[children[1]][0])


def test_textured_face_needs_valid_lower_layer_not_an_opaque_patch(monkeypatch):
    yy, xx = np.mgrid[:100, :170]
    texture = np.repeat((130+60*((xx+yy)%2))[...,None], 3, axis=-1).astype('uint8')
    source, children, candidates = _group(texture)
    protected = np.zeros(source.shape[:2], bool)
    assert all(v is None for v in backing.shared_face_patches(source, candidates, [children], protected).values())
    blank = texture.copy()
    for x, y, w, h in children:
        blank[y:y+h, x:x+w] = 158
    meta = {'manifest_sha256':'fixture','below_z':1}
    calls = []
    def layers(image, directory, mask):
        calls.append(mask.copy())
        return [LowerBackground(b.copy(), meta) for b in (source, blank, texture)]
    monkeypatch.setattr(backing,'lower_backgrounds', layers)
    patches = backing.shared_face_patches(source, candidates, [children], protected, source=source, layer_directory='fixture')
    assert len(calls) == 1
    for x, y, w, h in children:
        field, info = patches[(x,y,w,h)]
        assert calls[0][y:y+h, x:x+w].all()
        assert info['method'] == 'shared-parent-layerd/1'
        np.testing.assert_array_equal(field, texture[y:y+h, x:x+w])


def test_bar_edge_recovery_leaves_crossing_peer_and_axis_untouched():
    plate = np.full((110,160,3), 205, dtype='uint8')
    x,y,w,h = box = (44, 27, 32, 65)
    patch = plate[y:y+h,x:x+w].copy()
    plate[y:y+h,x-1] = 198
    plate[y:y+h,x+w] = 210
    plate[y+h] = 30
    protected = np.zeros(plate.shape[:2], bool)
    protected[53:58,x-2:x] = True
    plate[protected] = 202
    before = plate.copy()
    (bx,by,bw,bh), mask, colors = backing.bar_edge_backing(plate, box, patch, protected)
    plate[by:by+bh,bx:bx+bw][mask] = colors[mask]
    assert np.all(plate[40,x-1] == 205) and np.all(plate[42,x+w] == 205)
    np.testing.assert_array_equal(plate[protected],before[protected])
    np.testing.assert_array_equal(plate[y+h],before[y+h])


def test_no_samples_read_from_native_parent_already_lifted_off_plate():
    paper = np.full((100,170,3), 249, dtype='uint8')
    source, children, candidates = _group(paper)
    protected = np.ones(paper.shape[:2], bool)
    for x,y,w,h in children:
        protected[y:y+h,x:x+w] = False
    assert backing.shared_face_patches(source, candidates, [children], protected) == dict.fromkeys(children)
