import numpy as np
import pytest
from PIL import Image, ImageDraw

from product.recovery.embedded_device import detect_device_frames, fit_device_backing


def test_only_source_supported_device_with_intrinsic_text_is_one_replaceable_picture():
    image = Image.new('RGB', (520, 300), '#f8fbfd')
    draw = ImageDraw.Draw(image)
    items = []
    for x in (30, 240):
        draw.rounded_rectangle((x, 42, x+115, 243), radius=12, outline='#161616', width=3, fill='#ddeef5')
        for y in (87, 120, 154):
            draw.line((x+16, y, x+75, y), fill='#494949', width=2)
            items.append({'bbox': [x+13, y-3, 66, 8]})
    draw.rounded_rectangle((66, 42, 111, 54), radius=3, fill='#161616')
    src = np.asarray(image)
    assert detect_device_frames(src, [], None) == []
    boxes = detect_device_frames(src, items, None)
    assert len(boxes) == 1 and boxes[0][0] < 35

    box = boxes[0]
    x, y, w, h = box
    protected = np.zeros(src.shape[:2], bool)
    backing, owner, _ = fit_device_backing(src, box, protected)
    assert owner.any() and not owner.all()
    region = src[y:y+h, x:x+w]
    deleted = np.where(owner[..., None], backing, region)
    assert np.all(deleted == (248, 251, 253))
    assert np.array_equal(np.where(owner[..., None], region, deleted), region)

    protected[y+h//2, x+w//2] = True
    with pytest.raises(ValueError, match='independent object'):
        fit_device_backing(src, box, protected)
    protected[:] = False
    with pytest.raises(ValueError, match='separate text or picture'):
        fit_device_backing(src, box, protected, [[x-10, y+25, w//2, 16]])
    # A rectangle alone cannot re-authorize a different part of this source.
    with pytest.raises(ValueError, match='bezel'):
        fit_device_backing(src, [x+9, y, w, h], protected)
