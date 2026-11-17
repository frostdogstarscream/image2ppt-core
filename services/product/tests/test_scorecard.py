import io
import zipfile

import numpy as np
from PIL import Image

from product import scorecard

P = 'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" ' \
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" ' \
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
EMU = 9525  # 1 px at 96 dpi


def _xfrm(x, y, w, h, tag='a:xfrm'):
    return f'<{tag}><a:off x="{x*EMU}" y="{y*EMU}"/><a:ext cx="{w*EMU}" cy="{h*EMU}"/></{tag}>'


def _pic(rid, name, x, y, w, h):
    return (f'<p:pic><p:nvPicPr><p:cNvPr id="1" name="{name}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/></p:blipFill><p:spPr>{_xfrm(x, y, w, h)}</p:spPr></p:pic>')


def _sp(name, x, y, w, h, text=''):
    body = f'<p:txBody><a:bodyPr/><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody>' if text else ''
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="2" name="{name}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{_xfrm(x, y, w, h)}</p:spPr>{body}</p:sp>')


def _png(array):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, 'PNG')
    return buf.getvalue()


def build_pptx(path, size, shapes, media):
    w, h = size
    rels = ''.join(f'<Relationship Id="{rid}" Type="image" Target="../media/{rid}.png"/>' for rid in media)
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('ppt/presentation.xml',
                   f'<p:presentation {P}><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst>'
                   f'<p:sldSz cx="{w*EMU}" cy="{h*EMU}"/></p:presentation>')
        z.writestr('ppt/_rels/presentation.xml.rels',
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="slide" Target="/ppt/slides/slide1.xml"/></Relationships>')
        z.writestr('ppt/slides/slide1.xml',
                   f'<p:sld {P}><p:cSld><p:spTree><p:nvGrpSpPr/><p:grpSpPr/>{"".join(shapes)}</p:spTree></p:cSld></p:sld>')
        z.writestr('ppt/slides/_rels/slide1.xml.rels',
                   f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>')
        for rid, data in media.items():
            z.writestr(f'ppt/media/{rid}.png', data)


def _source(tmp_path):
    src = np.full((100, 200, 3), 255, np.uint8)
    src[10:30, 10:90] = 0          # "text" ink block
    src[50:90, 120:180] = [200, 30, 30]  # graphic
    src[50:90, 10:60] = np.random.default_rng(0).integers(0, 255, (40, 50, 3))  # texture
    path = tmp_path / 'src.png'
    Image.fromarray(src).save(path)
    return src, path


def test_ownership_classes_and_content_fractions(tmp_path):
    src, src_path = _source(tmp_path)
    icon = np.zeros((40, 60, 4), np.uint8)
    icon[5:35, 5:55] = [200, 30, 30, 255]  # transparent border -> asset
    crop = np.dstack([src[50:90, 10:60], np.full((40, 50), 255, np.uint8)])  # opaque -> opaque_picture
    plate = np.dstack([src, np.full((100, 200), 255, np.uint8)])
    plate[50:90, 10:60, :3] = 255  # crop pixels moved out of the plate
    pptx = tmp_path / 'r.pptx'
    build_pptx(pptx, (200, 100), [
        _pic('rIdPlate', 'source-plate', 0, 0, 200, 100),
        _pic('rIdCrop', 'crop-1', 10, 50, 50, 40),
        _pic('rIdIcon', 'a-001', 120, 50, 60, 40),
        _sp('s-001', 100, 5, 90, 30),
        _sp('t-001', 10, 10, 80, 20, '\u6807\u9898'),
    ], {'rIdPlate': _png(plate), 'rIdCrop': _png(crop), 'rIdIcon': _png(icon)})

    result = scorecard.score_page(src_path, pptx, render_path=src_path,
                                  ocr={'lines': [{'text': '\u6807 \u9898', 'bbox': [12, 11, 70, 18], 'score': 0.99},
                                                 {'text': '\u7f3a\u5931', 'bbox': [120, 92, 40, 6], 'score': 0.99}]})
    assert result['object_counts'] == {'plate': 1, 'opaque_picture': 1, 'asset': 1, 'shape': 1, 'text': 1}
    area = result['visible_area_fraction']
    assert area['text'] == round(80 * 20 / 20000, 4)
    assert area['opaque_picture'] == round(50 * 40 / 20000, 4)
    # Transparent icon border falls through to the plate.
    assert area['asset'] == round(50 * 30 / 20000, 4)
    assert result['raster_plate_content_fraction'] < 1.0
    assert result['fidelity']['mae_rgb'] == 0.0 and result['fidelity']['ssim'] > 0.999
    assert result['fidelity']['delta_e2000_p95'] < 1e-6
    text = result['text']
    assert text['ocr_lines'] == 2 and text['matched_lines'] == 1
    assert text['line_recall'] == 0.5 and text['unmatched'][0]['ocr_text'] == '\u7f3a\u5931'
    scorecard.save_owner_preview(src_path, pptx, tmp_path / 'owner.png')
    assert (tmp_path / 'owner.png').is_file()


def test_raster_only_page_has_zero_editable_content(tmp_path):
    src, src_path = _source(tmp_path)
    pptx = tmp_path / 'raster.pptx'
    plate = np.dstack([src, np.full((100, 200), 255, np.uint8)])
    build_pptx(pptx, (200, 100), [_pic('rId9', 'source-plate', 0, 0, 200, 100)], {'rId9': _png(plate)})
    result = scorecard.score_page(src_path, pptx)
    assert result['editable_content_fraction'] == 0.0
    assert result['raster_plate_content_fraction'] == 1.0


def test_crop_over_unchanged_plate_is_duplicate_not_editable(tmp_path):
    src, src_path = _source(tmp_path)
    pptx = tmp_path / 'dup.pptx'
    plate = np.dstack([src, np.full((100, 200), 255, np.uint8)])
    crop = np.dstack([src[50:90, 10:60], np.full((40, 50), 255, np.uint8)])
    build_pptx(pptx, (200, 100), [_pic('rIdP', 'source-plate', 0, 0, 200, 100), _pic('rIdC', 'r-001', 10, 50, 50, 40)],
               {'rIdP': _png(plate), 'rIdC': _png(crop)})
    result = scorecard.score_page(src_path, pptx)
    assert result['object_counts'] == {'plate': 1, 'duplicate_picture': 1}
    assert result['editable_content_fraction'] == 0.0


def test_ciede2000_reference_pairs():
    # Sharma et al. (2005) test data, first pairs.
    lab1 = np.array([[50.0, 2.6772, -79.7751], [50.0, 3.1571, -77.2803], [50.0, 2.8361, -74.0200]])
    lab2 = np.array([[50.0, 0.0, -82.7485], [50.0, 0.0, -82.7485], [50.0, 0.0, -82.7485]])
    np.testing.assert_allclose(scorecard.ciede2000(lab1, lab2), [2.0425, 2.8615, 3.4412], atol=1e-4)
