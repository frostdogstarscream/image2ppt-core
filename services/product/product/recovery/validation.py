"""Fixed compilation and actual Office checks for a single staged region."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..materialize import digest, write
from ..object_extract import rectangle
from ..parallel_contracts import SceneGraph
from ..parallel_edit_probe import inspect_text
from ..parallel_office import inspect
from ..parallel_pipeline import effective_texts
from ..parallel_residual_gate import inspect_residuals
from ..parallel_text_candidates import peer_protection
from ..pipeline import compile_pages
from .candidate_stage import stage_background
from .gradient_backing import fit_gradient


def stage_and_verify(trial, graph, baseline, fonts, output, *, office=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = Path(baseline)
    parent_hashes = {str(p.relative_to(baseline)): digest(p) for p in baseline.rglob('*') if p.is_file()}
    ir = stage_background(trial, graph, baseline, fonts, output/'region')
    staged = SceneGraph.model_validate_json((ir.parent/'scene.json').read_text('utf8'))
    artifacts = compile_pages([ir], output/'compiled', [effective_texts(staged, ir)])
    result = {'schema': 'R12RegionalAcceptance/1', 'artifacts': artifacts, 'status': 'REVIEW_REQUIRED',
              'source_sha256': graph.source_sha256, 'baseline_unchanged': False,
              'published': False, 'human_acceptance': False, 'go': 'NOT_RUN'}
    row = json.loads((Path(trial)/'residual_report.json').read_text('utf8'))
    identifier = row['object_id']
    if office:
        result['office'] = inspect(artifacts['pptx'], [ir], output/'office')
        result['replacement'] = inspect_text(artifacts['pptx'], [ir], output/'replacement', target_ids={identifier})
        result['owner_residual'] = inspect_residuals(ir, output/'office', 1, output/'owner-residual.json')
        source = np.array(Image.open(baseline/'source.png').convert('RGB'))
        text = next(t for t in graph.text.texts if t.id == identifier)
        x, y, w, h = map(round, text.bbox)
        box = [x-3, y-3, w+6, h+6]
        protected = peer_protection(text, graph, source, np.zeros(source.shape[:2], bool))
        reference, fit = fit_gradient(source, box, protected)
        deleted = output/'office'/f'01-{identifier}-delete.png'
        actual = np.array(Image.open(deleted).convert('RGB'))
        if actual.shape != source.shape:
            raise ValueError('Office deletion dimensions differ from source')
        scope = rectangle(box, source.shape[1::-1]) & ~protected
        error = np.abs(actual.astype(float)-reference.astype(float)).max(2)
        original = np.abs(source.astype(float)-reference.astype(float)).max(2)
        residual = scope & (original > 6) & (error > 6)
        target_report = {'status': 'PASS' if not residual.any() and np.percentile(error[scope], 95) <= 5 else 'REVIEW_REQUIRED',
                         'original_signal_residual_pixels': int(residual.sum()),
                         'whole_scope_p95_rgb': float(np.percentile(error[scope], 95)),
                         'whole_scope_mae_rgb': float(error[scope].mean()),
                         'reference_context': fit, 'deleted_preview_sha256': digest(deleted),
                         'reference_scope': 'entire source text search scope, including pixels outside the write mask'}
        write(output/'deleted-background.json', target_report)
        result['deleted_background'] = target_report
        target_owner = next(r for r in result['owner_residual'] if r['id'] == identifier)
        if (result['office']['status'] == 'PASS_APP_AUTOMATED' and result['replacement']['status'] == 'PASS'
                and target_owner['residual_check_status'] == 'PASS' and target_report['status'] == 'PASS'):
            result['regional_automatic_checks'] = 'PASS'
    result['baseline_unchanged'] = parent_hashes == {
        str(p.relative_to(baseline)): digest(p) for p in baseline.rglob('*') if p.is_file()}
    if not result['baseline_unchanged']:
        raise ValueError('regional trial modified baseline')
    write(output/'acceptance.json', result)
    return result
