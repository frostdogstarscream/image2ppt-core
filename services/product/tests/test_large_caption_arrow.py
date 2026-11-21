import cv2
import numpy as np
import pytest
from product.recovery.native_shapes import _detect_preset_geometry


@pytest.mark.parametrize("height",[40,170,340])
def test_large_horizontal_caption_keeps_right_tip(height):
    width=height*4
    background=np.full((height,width,3),255,np.float32)
    pixels=background.copy()
    polygon=np.array([(0,0),(width-height//3,0),(width-1,height//2),
                      (width-height//3,height-1),(0,height-1)],np.int32)
    cv2.fillPoly(pixels,[polygon],(232,239,250))
    kind,_,exterior=_detect_preset_geometry(pixels,background,np.zeros((height,width),bool))
    assert kind=='pentagon'
    assert exterior[0,-1] and not exterior[height//2,-2]

@pytest.mark.parametrize("radius", [0, 26, 85])
def test_flat_or_rounded_panel_is_not_an_arrow(radius):
    height, width = 170, 820
    mask = np.zeros((height, width), np.uint8)
    cv2.rectangle(mask, (radius, 0), (width-radius-1, height-1), 1, -1)
    cv2.rectangle(mask, (0, radius), (width-1, height-radius-1), 1, -1)
    for x in (radius, width-radius-1):
        for y in (radius, height-radius-1):
            cv2.circle(mask, (x,y), radius, 1, -1)
    paper = np.full((height,width,3), 255, np.float32)
    pixels = paper.copy()
    pixels[mask.astype(bool)] = (230,239,249)
    measured = _detect_preset_geometry(pixels, paper, np.zeros_like(mask, bool))
    assert measured is None or measured[0] not in ('pentagon','chevron')
