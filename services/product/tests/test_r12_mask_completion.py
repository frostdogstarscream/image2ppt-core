"""Text boundary, independent background and local staging regressions."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.recovery.text_mask_completion import TextMaskCompletion
from product.recovery.gradient_backing import fit_gradient
from product.recovery.diagnostics import assess_trial, save_trial
from product.recovery.candidate_stage import stage_background
from product.parallel_store import key
from product.parallel_rebuild import build
from product.materialize import digest
from test_parallel_candidates import fixture


def scene():
    source = np.full((80, 120, 3), 235, np.uint8)
    backing = source.copy()
    source[30:48, 45:50] = 30
    source[30:48, 44] = 150
    source[30:48, 50] = 150
    source[30:48, 43] = 250
    source[30:48, 51] = 250
    mask = np.zeros(source.shape[:2], bool)
    mask[30:48, 45:50] = True
    protected = np.zeros_like(mask)
    return source, backing, mask, protected


def test_evidence_partition_and_bright_antialias_completion():
    source, backing, mask, protected = scene()
    original = source.copy()
    evidence = TextMaskCompletion.analyze(source, mask, protected, [35, 25, 30, 30], backing, glyph_height=18)
    assert evidence.evidence['text_antialias'][35, 44]
    assert evidence.evidence['possible_text'][35, 43]
    assert sum(int(m[35, 43]) for m in evidence.evidence.values()) == 1
    assert np.all(np.sum(list(evidence.evidence.values()), axis=0) <= 1)
    assert evidence.masks['original'].sum() < evidence.masks['bounded_expansion'].sum() < evidence.masks['semantic_completion'].sum()
    assert not evidence.masks['semantic_completion'].flags.writeable
    assert np.array_equal(source, original)


def test_protected_object_never_enters_completion():
    source, backing, mask, protected = scene()
    protected[30:48, 43] = True
    evidence = TextMaskCompletion.analyze(source, mask, protected, [35, 25, 30, 30], backing, glyph_height=18)
    assert all(not (m & protected).any() for m in evidence.masks.values())
    assert not evidence.evidence['possible_text'].any()
    assert any(c['protected_conflict'] for c in evidence.report['semantic_components'])


def test_unseeded_contrasting_object_remains_unknown():
    source, backing, mask, protected = scene()
    source[30:35, 60:64] = 30
    evidence = TextMaskCompletion.analyze(source, mask, protected, [35, 25, 35, 30], backing, glyph_height=18)
    assert not evidence.masks['semantic_completion'][32, 62]
    assert evidence.evidence['unknown'][32, 62]
    assert evidence.report['unassigned_contrast_pixels'] == 20


def test_lateral_gradient_excludes_horizontal_border():
    y, x = np.mgrid[:80, :120]
    source = np.repeat(np.rint(220+x*.05+y*.1)[..., None], 3, 2).astype('uint8')
    source[19:21] = [0, 40, 170]
    source[60:62] = [0, 40, 170]
    backing, report = fit_gradient(source, [35, 25, 35, 30], np.zeros((80, 120), bool))
    assert report['validation_p95_rgb'] <= 1
    assert np.abs(backing[30:50, 40:60].astype(int)-source[30:50, 40:60]).max() <= 1


def test_whole_scope_residual_and_raw_brightness_are_separate():
    source, backing, mask, protected = scene()
    box = [35, 25, 30, 30]
    reference = {'validation_p95_rgb': 0}
    incomplete, _, _ = assess_trial(source, mask, backing, protected, box, backing, reference, {})
    assert incomplete['status'] == 'REVIEW_REQUIRED'
    assert incomplete['visible_source_pixels_outside_mask'] > 0
    completed = TextMaskCompletion.analyze(source, mask, protected, box, backing, glyph_height=18).masks['semantic_completion']
    bright = np.full_like(source, 255)
    bad, _, _ = assess_trial(source, completed, bright, protected, box, backing, reference, {})
    assert bad['status'] == 'REVIEW_REQUIRED'
    assert bad['raw_mask_luma_bias'] == 20
    assert bad['writeback_raw_max_difference'] == 0
    good, _, _ = assess_trial(source, completed, backing, protected, box, backing, reference, {})
    assert good['status'] == 'PRECHECK_PASS'
    assert good['remaining_source_signal_pixels'] == 0


def staged_fixture(fixture, monkeypatch):
    graph, snap, source, root = fixture
    monkeypatch.setattr('product.recovery.candidate_stage.font_gate', lambda *args: None)
    monkeypatch.setattr('product.recovery.candidate_stage.native_frame', lambda text, *args: text.model_dump())
    fallback = graph.model_copy(deep=True)
    fallback.ownership['t'] = 'AMBIGUOUS'
    baseline_ir, _ = build(fallback, snap, {'fonts': []}, root/'baseline')
    box = [37, 27, 46, 36]
    protected = np.zeros(source.shape[:2], bool)
    backing, fit_report = fit_gradient(source, box, protected)
    mask = np.zeros_like(protected)
    mask[40:46, 50:60] = True
    save_trial(root/'trial', source, mask, backing, protected, box, backing, fit_report,
        {'status': 'PRECHECK_PASS', 'source_sha256': graph.source_sha256,
         'scene_graph_sha256': key(graph.model_dump()), 'scene_sha256': 'file-binding',
         'object_id': 't', 'mask_variant': 'original', 'method': 'gradient'})
    return graph, root/'trial', baseline_ir.parent, root


def test_stage_rechecks_evidence_and_preserves_baseline(fixture, monkeypatch):
    graph, trial, baseline, root = staged_fixture(fixture, monkeypatch)
    before = {p.name: digest(p) for p in baseline.iterdir() if p.is_file()}
    ir = stage_background(trial, graph, baseline, {'fonts': []}, root/'staged')
    assert any(n['id'] == 't' and n['type'] == 'text' for n in json.loads(ir.read_text('utf8'))['nodes'])
    assert before == {p.name: digest(p) for p in baseline.iterdir() if p.is_file()}
    assert json.loads((ir.parent/'scene.json').read_text('utf8'))['selected']['t']['owner_evidence']['method'].endswith('r12-1')


def test_tampering_and_failed_region_leave_other_regions(fixture, monkeypatch):
    graph, trial, baseline, root = staged_fixture(fixture, monkeypatch)
    stage_background(trial, graph, baseline, {'fonts': []}, root/'good-region')
    before = {p.name: digest(p) for p in (root/'good-region').iterdir() if p.is_file()}
    Image.fromarray(np.zeros((100, 180, 3), np.uint8)).save(trial/'written_back.png')
    with pytest.raises(ValueError, match='binding mismatch'):
        stage_background(trial, graph, baseline, {'fonts': []}, root/'bad-region')
    assert not (root/'bad-region').exists()
    assert before == {p.name: digest(p) for p in (root/'good-region').iterdir() if p.is_file()}


def test_stage_failure_never_exposes_partial_region(fixture, monkeypatch):
    graph, trial, baseline, root = staged_fixture(fixture, monkeypatch)
    def fail_after_write(candidate, graph, baseline, output):
        output.mkdir()
        (output/'partial').write_text('partial')
        raise RuntimeError('injected stage failure')
    monkeypatch.setattr('product.recovery.candidate_stage.stage', fail_after_write)
    with pytest.raises(RuntimeError):
        stage_background(trial, graph, baseline, {'fonts': []}, root/'bad-region')
    assert not (root/'bad-region').exists()
    assert not list(root.glob('r12-stage-*'))


def test_forged_pass_report_does_not_bypass_recomputed_backing(fixture, monkeypatch):
    graph, trial, baseline, root = staged_fixture(fixture, monkeypatch)
    Image.fromarray(np.full((100, 180, 3), 20, np.uint8)).save(trial/'raw_inpaint.png')
    hashes = json.loads((trial/'hashes.json').read_text('utf8'))
    hashes['raw_inpaint.png'] = digest(trial/'raw_inpaint.png')
    (trial/'hashes.json').write_text(json.dumps(hashes), encoding='utf8')
    with pytest.raises(ValueError, match='binding mismatch'):
        stage_background(trial, graph, baseline, {'fonts': []}, root/'forged')
    assert not (root/'forged').exists()


def test_office_scale_probe_scales_absolute_pitch_only():
    from types import SimpleNamespace
    from product.parallel_office import scale_text
    class Ranges:
        def __init__(self, values):
            self.values = values
            self.Count = len(values)
        def __call__(self, index=None):
            return self if index is None else self.values[index-1]
    absolute = SimpleNamespace(LineRuleWithin=0, SpaceWithin=19.5,
                               LineRuleBefore=0, SpaceBefore=2.0, LineRuleAfter=0, SpaceAfter=0)
    relative = SimpleNamespace(LineRuleWithin=-1, SpaceWithin=1.2,
                               LineRuleBefore=-1, SpaceBefore=1.0, LineRuleAfter=0, SpaceAfter=0)
    run = SimpleNamespace(Font=SimpleNamespace(Size=13.5))
    tr = SimpleNamespace(Runs=Ranges([run]), Paragraphs=Ranges([
        SimpleNamespace(ParagraphFormat=absolute), SimpleNamespace(ParagraphFormat=relative)]))
    scale_text(tr, .9)
    assert run.Font.Size == pytest.approx(12.15)
    assert absolute.SpaceWithin == pytest.approx(17.55)
    assert absolute.SpaceBefore == pytest.approx(1.8)
    assert relative.SpaceWithin == 1.2 and relative.SpaceBefore == 1.0


def test_native_frame_adds_bearing_without_moving_text(monkeypatch):
    from types import SimpleNamespace
    from product.contracts import Text, Style
    from product.recovery.candidate_stage import native_frame
    text = Text(id='t', text='测试', bbox=[40., 30., 20., 20.], z=1, style='body')
    style = Style(id='body', role='body', ea='EA', latin='Latin', size_px=18, bold=False, color='111111')
    monkeypatch.setattr('product.recovery.candidate_stage.ImageFont.truetype',
                        lambda *args, **kwargs: SimpleNamespace(getbbox=lambda _: [0, 2, 30, 18],getlength=lambda _:25.5))
    monkeypatch.setattr('product.recovery.candidate_stage.font_line_height', lambda *args: 24.)
    fonts = {'fonts': [{'family': family, 'bold': False, 'file': 'synthetic.ttf', 'index': 0}
                       for family in ('EA', 'Latin')]}
    payload = native_frame(text, {'body': style}, fonts, (120, 80), np.zeros((80, 120), bool), [35, 25, 35, 30])
    assert payload['bbox'] == [40., 30., 27., 25.]
    assert text.bbox == [40., 30., 20., 20.]


def test_unavailable_lama_preserves_independent_background_trials(fixture, monkeypatch):
    from types import SimpleNamespace
    import run_r12_background
    graph, snap, _, root = fixture
    scene_file = root/'scene.json'
    scene_file.write_text(graph.model_dump_json(by_alias=True), encoding='utf8')
    config_file = root/'services.json'
    config_file.write_text(json.dumps({'lama': {'url': 'http://127.0.0.1:1/inpaint',
                          'model_binding': 'synthetic', 'gpu_lease': {}}}), encoding='utf8')
    monkeypatch.setattr(run_r12_background, 'lower_backgrounds', lambda *args: [])
    def forbidden(*args):
        pytest.fail('unconfigured physical GPU lease must block inference')
    monkeypatch.setattr(run_r12_background.LamaClient, '__call__', forbidden)
    output = root/'experiment'
    records = run_r12_background.run(SimpleNamespace(source=snap['path'], scene=scene_file,
        object_id='t', layerd=root, output=output, lama_config=config_file,
        lama_cache=root/'lama-cache', stage_baseline=None))
    assert len(records) == 3 and all(row['method'] == 'gradient' for row in records)
    assert len(list(output.glob('*--big-lama-failure.json'))) == 3
    assert (output/'index.json').exists()
    assert all((output/row['candidate']/'hashes.json').exists() for row in records)
