"""Regression for a real SAM3 PPT trial's chevron compilation failure."""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'services/vision/src'))
from image2ppt_vision.layered.emit_ir import emit


def test_chevron_notch_is_not_emitted_as_rounding(tmp_path):
    source = tmp_path/'perception'
    source.mkdir()
    Image.new('RGB', (400, 200), 'white').save(source/'source.png')
    elements = [dict(id=geometry, kind='native_shape', geometry=geometry,
                     bbox=[10+100*i, 20, 80, 40], fill='0088FF',
                     corner_radius_px=10, source='geometry_fit')
                for i, geometry in enumerate(('chevron', 'roundRect', 'rect'))]
    perception = dict(image_size=[400, 200], ocr=[], text=[], tables=[],
                      elements=elements, background={'solid_color': 'FFFFFF'})
    ir = emit(source, perception, {}, [], tmp_path/'ir')
    nodes = {n['id']: n for n in ir['nodes']}
    assert nodes['chevron']['geometry'] == 'chevron'
    assert 'corner_radius_px' not in nodes['chevron']
    assert nodes['roundRect']['corner_radius_px'] == 10
    assert nodes['rect']['corner_radius_px'] == 10
