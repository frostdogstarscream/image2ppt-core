import cv2
import numpy as np

from product.recovery.circular_badge import circular_badge
from product.recovery.marked_fields import footer_fields


def test_badge_hole_and_surrounding_pixels():
    source = np.full((120, 120, 3), 250, np.uint8)
    cv2.circle(source, (60, 60), 27, (230, 236, 242), -1)
    plate = source.copy()
    cv2.circle(source, (60, 60), 9, (40, 72, 170), 3)
    bbox = [43, 43, 34, 34]
    result = circular_badge(source, plate, bbox)
    assert result is not None
    assert result[2][17, 17, 3] == 0
    assert result[2][17, 26, 3] > 240
    plate[40, 60] = 0
    assert circular_badge(source, plate, bbox) is None


def test_footer_split_requires_source_gutters_and_preserves_wording():
    source = np.full((200, 200, 3), 255, np.uint8)
    for x in (12, 22, 32, 80, 90):
        source[181:191, x:x+8] = 50
    fields = footer_fields(source, 'ABCDE', [10, 180, 90, 12])
    assert [p[0] for p in fields] == ['ABC', 'DE']
    assert footer_fields(source, 'ABCDE', [10, 20, 90, 12]) == []
