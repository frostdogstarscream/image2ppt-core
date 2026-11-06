"""Read-only baseline lineage rejects detached, changed and ambiguous evidence."""
import copy
import json
from pathlib import Path

import pytest

from _helpers import SRC
from r7_baseline import load_baseline, _hash
from reconstruction_transaction import scene_fingerprint


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return path


def fixture(tmp_path):
    r6 = tmp_path / 'r6'; r6.mkdir()
    source = r6 / 'source.png'; source.write_bytes(b'source')
    pptx = r6 / 'page.pptx'; pptx.write_bytes(b'pptx')
    preview = r6 / 'preview.png'; preview.write_bytes(b'preview')
    font = tmp_path / 'f.ttf'; font.write_bytes(b'font')
    fonts = {'fonts': [{'file': str(font), 'sha256': _hash(font)}]}
    write(r6 / 'font-manifest.json', fonts)
    ir = write(r6 / 'slideir.json', {'schema_version': '0.1', 'source_image': 'source.png',
        'source_sha256': _hash(source), 'canvas': {'source_px': [20, 15]},
        'background': {'color': 'FFFFFF'}, 'assets': [], 'evidence': [],
        'nodes': [{'id': 'title', 'type': 'text', 'bbox': [1, 1, 8, 8], 'z': 1,
                   'font': {'size_pt': 20}, 'paragraphs': [{'text': '标题'}]}]})
    masks = r6 / 'text-masks'; masks.mkdir()
    (masks / 'l1-glyph.png').write_bytes(b'glyph')
    (masks / 'l1-inspection.png').write_bytes(b'inspection')
    page = {'page': '01_cover', 'directory': str(r6), 'ir': str(ir), 'ir_sha256': _hash(ir),
            'pptx': str(pptx), 'pptx_sha256': _hash(pptx), 'source_sha256': _hash(source),
            'preview': str(preview), 'text_checks': [{'id': 'l1'}]}
    baseline = write(r6 / 'batch.json', {'schema': 'TypographyBatch/1', 'pages': [page]})
    return baseline, page, fonts


def reconstruction(tmp_path, baseline, page, fonts, name='r7'):
    directory = tmp_path / name; directory.mkdir()
    batch = write(directory / 'batch.json', {'schema': 'R7Batch/1', 'input': str(baseline),
                                           'pages': [copy.deepcopy(page)]})
    maskroot = Path(page['directory']) / 'text-masks'
    manifest = {'schema': 'R7RunInputs/1', 'baseline': str(baseline), 'baseline_sha256': _hash(baseline),
                'fonts': fonts, 'pages': [{'page': page['page'], 'scene': scene_fingerprint(page['ir']),
                  'pptx_sha256': page['pptx_sha256'],
                  'source_masks': {str(p.resolve()): _hash(p) for p in maskroot.glob('*.png')}}]}
    write(directory / 'run-manifest.json', manifest)
    return batch


def test_r6_and_r7_normalization_preserves_actual_parent_and_input_bytes(tmp_path):
    baseline, page, fonts = fixture(tmp_path)
    r7 = reconstruction(tmp_path, baseline, page, fonts)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = load_baseline(r7)
    actual = result['pages'][0]
    assert actual['ir'] == page['ir']
    assert actual['evidence_baseline']['text_checks'] == [{'id': 'l1'}]
    assert actual['baseline_refs']['r6']['ir']['path'] == page['ir']
    assert actual['baseline_refs']['r7']['ir']['path'] == page['ir']
    assert ['title', 'replace_short'] in actual['expected_probes']
    context = result['_baseline_context']
    assert len(context['lineage']) == 2
    assert context['font_manifest']['fonts'] == fonts
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize('kind', ['font', 'mask', 'parent', 'ir', 'pptx'])
def test_changed_dependency_is_rejected(tmp_path, kind):
    baseline, page, fonts = fixture(tmp_path)
    r7 = reconstruction(tmp_path, baseline, page, fonts)
    paths = {'font': tmp_path / 'f.ttf', 'mask': Path(page['directory']) / 'text-masks/l1-glyph.png',
             'parent': baseline, 'ir': Path(page['ir']), 'pptx': Path(page['pptx'])}
    paths[kind].write_bytes(paths[kind].read_bytes() + b' ')
    with pytest.raises(ValueError, match='hash|inventory'):
        load_baseline(r7)


def test_r7_missing_manifest_does_not_guess_nearby_fontfile(tmp_path):
    baseline, page, fonts = fixture(tmp_path)
    r7 = reconstruction(tmp_path, baseline, page, fonts)
    (r7.parent / 'run-manifest.json').unlink()
    write(r7.parent / 'font-manifest.json', fonts)
    with pytest.raises(ValueError, match='file missing'):
        load_baseline(r7)


def test_declared_parent_and_batch_input_must_match(tmp_path):
    baseline, page, fonts = fixture(tmp_path)
    r7 = reconstruction(tmp_path, baseline, page, fonts)
    manifest = json.loads((r7.parent / 'run-manifest.json').read_text(encoding='utf-8'))
    manifest['baseline'] = str(tmp_path / 'unrelated.json')
    write(r7.parent / 'run-manifest.json', manifest)
    with pytest.raises(ValueError, match='differs from frozen parent'):
        load_baseline(r7)


def test_missing_current_text_does_not_erase_expected_probe(tmp_path):
    baseline, page, fonts = fixture(tmp_path)
    r7 = reconstruction(tmp_path, baseline, page, fonts)
    candidate = r7.parent / 'candidate'; candidate.mkdir()
    (candidate / 'source.png').write_bytes((baseline.parent / 'source.png').read_bytes())
    ir = json.loads(Path(page['ir']).read_text(encoding='utf-8')); ir['nodes'] = []
    new_ir = write(candidate / 'slideir.json', ir)
    current = json.loads(r7.read_text(encoding='utf-8')); current['pages'][0].update(ir=str(new_ir), ir_sha256=_hash(new_ir))
    write(r7, current)
    assert ['title', 'replace_short'] in load_baseline(r7)['pages'][0]['expected_probes']


def test_masks_missing_from_frozen_manifest_are_not_invented(tmp_path):
    baseline, page, fonts = fixture(tmp_path)
    r7 = reconstruction(tmp_path, baseline, page, fonts)
    manifest = json.loads((r7.parent / 'run-manifest.json').read_text(encoding='utf-8'))
    manifest['pages'][0]['source_masks'] = {}
    write(r7.parent / 'run-manifest.json', manifest)
    with pytest.raises(ValueError, match='mask inventory'):
        load_baseline(r7)


def test_recursive_r7_keeps_original_r7_reference(tmp_path):
    baseline, page, fonts = fixture(tmp_path)
    r7 = reconstruction(tmp_path, baseline, page, fonts)
    r71 = reconstruction(tmp_path, r7, page, fonts, name='r71')
    result = load_baseline(r71)
    assert [x['schema'] for x in result['_baseline_context']['lineage']] == [
        'TypographyBatch/1', 'R7Batch/1', 'R7Batch/1']
    assert result['pages'][0]['baseline_refs']['r6'] == result['pages'][0]['baseline_refs']['r7']


def test_frozen_font_manifest_content_must_match(tmp_path):
    baseline, page, fonts = fixture(tmp_path)
    r7 = reconstruction(tmp_path, baseline, page, fonts)
    changed = copy.deepcopy(fonts); changed['unfrozen_extra'] = True
    write(baseline.parent / 'font-manifest.json', changed)
    with pytest.raises(ValueError, match='font manifest differs'):
        load_baseline(r7)


def test_provenance_cycle_guard(tmp_path):
    # Exercise the recursion guard directly; producing a cryptographic self-
    # referential manifest is neither necessary nor a valid real input.
    from r7_baseline import _load
    file = (tmp_path / 'batch.json').resolve()
    with pytest.raises(ValueError, match='Cyclic'):
        _load(file, {str(file)})


def test_empty_result_probe_list_cannot_override_manifest_expectation(tmp_path):
    baseline, page, fonts = fixture(tmp_path)
    r7 = reconstruction(tmp_path, baseline, page, fonts)
    manifest_path = r7.parent / 'run-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['pages'][0]['expected_probes'] = [['title', 'replace_short']]
    write(manifest_path, manifest)
    current = json.loads(r7.read_text(encoding='utf-8'))
    current['pages'][0]['expected_probes'] = []
    write(r7, current)
    with pytest.raises(ValueError, match='dropped mandatory'):
        load_baseline(r7)


@pytest.mark.parametrize('kind', ['r6', 'r7_without_manifest_probes', 'r7_with_manifest_probes'])
def test_inherited_inventory_cannot_drop_original_r6_text_probe(tmp_path, kind):
    baseline, page, fonts = fixture(tmp_path)
    original_inventory = load_baseline(baseline)['pages'][0]['expected_probes']
    target = baseline if kind == 'r6' else reconstruction(tmp_path, baseline, page, fonts)
    if kind == 'r7_with_manifest_probes':
        manifest_path = target.parent / 'run-manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['pages'][0]['expected_probes'] = original_inventory
        write(manifest_path, manifest)
    batch = json.loads(target.read_text(encoding='utf-8'))
    batch['pages'][0]['expected_probes'] = [x for x in original_inventory if x != ['title', 'replace_short']]
    write(target, batch)
    with pytest.raises(ValueError, match='dropped mandatory'):
        load_baseline(target)


def test_absent_legacy_probe_field_restores_manifest_floor(tmp_path):
    baseline, page, fonts = fixture(tmp_path)
    r7 = reconstruction(tmp_path, baseline, page, fonts)
    manifest_path = r7.parent / 'run-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    # A source-reviewed extra operation preserved by the frozen manifest must
    # survive a legacy result record that never stored expected_probes.
    manifest['pages'][0]['expected_probes'] = [['title', 'reviewed-extra-operation']]
    write(manifest_path, manifest)
    probes = load_baseline(r7)['pages'][0]['expected_probes']
    assert ['title', 'replace_short'] in probes
    assert ['title', 'reviewed-extra-operation'] in probes


def test_current_mandatory_addition_cannot_be_hidden_by_old_inventory(tmp_path):
    baseline, page, fonts = fixture(tmp_path)
    r7 = reconstruction(tmp_path, baseline, page, fonts)
    candidate = r7.parent / 'candidate'; candidate.mkdir()
    (candidate / 'source.png').write_bytes((baseline.parent / 'source.png').read_bytes())
    ir = json.loads(Path(page['ir']).read_text(encoding='utf-8'))
    extra = copy.deepcopy(ir['nodes'][0]); extra['id'] = 'new-title'; extra['font']['size_pt'] = 30
    ir['nodes'].append(extra)
    new_ir = write(candidate / 'slideir.json', ir)
    batch = json.loads(r7.read_text(encoding='utf-8'))
    batch['pages'][0].update(ir=str(new_ir), ir_sha256=_hash(new_ir),
                             expected_probes=load_baseline(baseline)['pages'][0]['expected_probes'])
    write(r7, batch)
    with pytest.raises(ValueError, match='dropped mandatory'):
        load_baseline(r7)


def test_qa_alias_with_validator_code_change_preserves_parent_probe_floor(tmp_path):
    baseline, page, fonts = fixture(tmp_path)
    r7 = reconstruction(tmp_path, baseline, page, fonts)
    probes = load_baseline(r7)['pages'][0]['expected_probes']
    manifest_path = r7.parent / 'run-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['pages'][0]['expected_probes'] = probes
    manifest['code'] = {'ppt-rebuild/src/r7_edit_checks.py': 'changed-validator'}
    write(manifest_path, manifest)
    batch = json.loads(r7.read_text(encoding='utf-8'))
    batch['pages'][0]['expected_probes'] = probes
    write(r7, batch)
    assert load_baseline(r7)['pages'][0]['expected_probes'] == probes
