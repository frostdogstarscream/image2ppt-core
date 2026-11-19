"""The visible-ink check must detect remnants even at the end of a long title."""
import numpy as np
import pytest
from PIL import Image

pytest.importorskip('win32com.client')
from probe_region_text import backing_metrics


@pytest.mark.parametrize('light', [True, False])
def test_source_strokes_and_final_letter_remnant_are_rejected(light):
    background = (32, 90, 155) if light else (246, 249, 252)
    foreground = (247, 248, 252) if light else (3, 48, 118)
    clean = np.full((28, 134, 3), background, dtype=np.int16)
    source = clean.copy()
    for x in range(5, 125, 14):
        source[4:23, x:x+4] = foreground
    source[4:23, 126:131] = foreground
    field = [1, 'fixture', [2, 2, 130, 23], 'text', 'new']
    assert backing_metrics(clean, source, field, light)['clean']
    assert not backing_metrics(source, source, field, light)['clean']
    residual = clean.copy()
    residual[4:23, 126:131] = foreground
    assert not backing_metrics(residual, source, field, light)['clean']


def test_blank_but_wrong_band_color_is_rejected():
    source = np.full((18, 90, 3), (40, 100, 165), dtype=np.int16)
    source[4:14, 6:10] = (240, 240, 240)
    erased_background = np.full_like(source, (205, 205, 205))
    result = backing_metrics(erased_background, source, [1,'field',[0,0,90,18],'text','new'], True)
    assert not result['clean']


def test_review_refuses_foreign_or_unsealed_edit_images(tmp_path, monkeypatch):
    import json
    from review_region_text import digest, prepare
    import review_region_text as review
    source = tmp_path/'input.png'
    Image.new('RGB',(1672,941),'white').save(source)
    project = tmp_path/'repo'
    cases = project/'services/experiments/native_edits/workbench.json'
    cases.parent.mkdir(parents=True)
    cases.write_text(json.dumps({'pages':[{'id':'workbench-08','sha256':digest(source)}]}),'utf8')
    monkeypatch.setattr(review,'ROOT',project)
    deck = tmp_path/'fixture.pptx'
    deck.write_bytes(b'sealed dummy deck')
    office = tmp_path/'office'
    office.mkdir()
    before = office/'before.png'
    before.write_bytes(source.read_bytes())
    binding = {'source_sha256':digest(source),'pptx_sha256':digest(deck)}
    (office/'report.json').write_text(json.dumps(binding),'utf8')
    binding['before_render_sha256'] = digest(before)
    binding['office_artifacts_sha256'] = {p.name:digest(p) for p in office.iterdir()}
    (office/'region-report.json').write_text(json.dumps(binding),'utf8')
    evidence, panels = prepare(source,deck,office,tmp_path/'fresh')
    assert len(panels) == 3 and evidence['missing_frames']['main_title']
    foreign = office/'recolored.png'
    foreign.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match='unsealed PowerPoint artifact'):
        prepare(source,deck,office,tmp_path/'unsealed')
    foreign.unlink()
    deck.write_bytes(b'different deck')
    with pytest.raises(ValueError, match='Deck or before image differs'):
        prepare(source,deck,office,tmp_path/'different-deck')
