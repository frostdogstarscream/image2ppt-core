"""Changing a badge's background must preserve its strokes and open holes."""
import cv2
import numpy as np
import pytest

from product.recovery.circular_badge import select_badge_ink
from product.recovery.region_crop import _extract_rgba


def ring_sample(ink, paper, *, pale_ringing=False):
    size = 64
    mask = np.zeros((size,size), np.uint8)
    cv2.circle(mask, (30,30), 16, 255, 4, lineType=cv2.LINE_AA)
    alpha = mask[...,None] / 255.
    backing = np.broadcast_to(np.uint8(paper), (size,size,3)).copy()
    source = np.rint(ink*alpha + backing*(1-alpha)).astype('uint8')
    if pale_ringing:
        # Brief pale overshoots touch the source contour, with open gaps.
        fringe = cv2.dilate((mask > 0).astype('uint8'),np.ones((3,3),np.uint8)).astype(bool) & (mask == 0)
        x,y = np.indices(mask.shape)
        excess = fringe & ((x+2*y) % 9 < 3)
        source[excess] = np.minimum(backing[excess].astype(int)+12,255)
    return source,backing,mask


@pytest.mark.parametrize('ink', [(23,108,65),(151,40,96)])
def test_flat_strokes_decontaminate_on_dark_and_light_and_keep_open_holes(ink):
    source,backing,truth=ring_sample(ink,(226,234,244),pale_ringing=True)
    footprint=(source!=backing).any(2)
    old=_extract_rgba(source,backing,footprint)
    rgba,plan=select_badge_ink(source,backing,old,None,strict_holes=True,compare_edges=True)
    assert plan['selected']=='stroke-unmix'
    old_trial=next(row for row in plan['candidates'] if row['method']=='color-unmix')
    assert old_trial['contrast_edge_p90']['dark']>24
    for bg in (24,240):
        expected=np.asarray(ink)*truth[...,None]/255.+bg*(1-truth[...,None]/255.)
        alpha=rgba[...,3:]/255.
        output=rgba[...,:3]*alpha+bg*(1-alpha)
        assert abs(output-expected).max() < 4
        assert (output[30,30]==bg).all()
    assert np.array_equal(select_badge_ink(source,backing,old,None,strict_holes=True)[0],old)


@pytest.mark.parametrize('attached',[False,True])
def test_measured_second_white_stroke_is_preserved(attached):
    source,backing,_=ring_sample((39,75,167),(211,224,237))
    y = 27 if attached else 54
    cv2.line(source,(42,y),(58,y),(255,255,255),1,cv2.LINE_AA)
    old=_extract_rgba(source,backing,(source!=backing).any(2))
    rgba,decision=select_badge_ink(source,backing,old,None,strict_holes=True,compare_edges=True)
    stroke=next(row for row in decision['candidates'] if row['method']=='stroke-unmix')
    assert stroke['status']=='not-applicable'
    assert np.array_equal(rgba,old) and rgba[y,52,3]>160


def test_white_ink_on_dark_disc_stays_white():
    source,backing,truth=ring_sample((249,248,245),(29,47,77))
    old=_extract_rgba(source,backing,(source!=backing).any(2))
    rgba,_=select_badge_ink(source,backing,old,None,strict_holes=True,compare_edges=True)
    assert (rgba[truth==255,:3] > 240).all() and (rgba[truth==255,3]>245).all()


def test_layer_and_excluded_variants_share_the_same_edge_gate():
    source,backing,mask=ring_sample((49,54,157),(226,234,244),pale_ringing=True)
    old=_extract_rgba(source,backing,(source!=backing).any(2))
    layer=np.zeros((*mask.shape,4),np.uint8); layer[...,:3]=[49,54,157]; layer[...,3]=mask
    actual,decision=select_badge_ink(source,backing,old,layer,strict_holes=True,compare_edges=True,
                                     excluded={'stroke-unmix'})
    assert decision['selected']=='LayerD' and np.array_equal(actual,layer)
    assert next(r for r in decision['candidates'] if r['method']=='stroke-unmix')['status']=='excluded-by-office'
