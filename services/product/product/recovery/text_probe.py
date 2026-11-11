"""Local text recovery tool shared by diagnostics and bounded Agent adapters."""
import json
from pathlib import Path
from types import SimpleNamespace

from ..materialize import digest, write
from ..object_extract import load_page
from ..parallel_candidate_engine import font_bindings
from ..parallel_store import key
from .candidate_stage import stage_background
from .region_candidates import adaptive_candidates


def input_binding(graph, identifier, source_file, layer_directory, baseline_directory, fonts):
    text = next(item for item in graph.text.texts if item.id == identifier)
    if digest(source_file) != graph.source_sha256:
        raise ValueError('text probe source binding mismatch')
    used = {text.style, *[span.style for span in text.spans]}
    styles = [style for style in graph.text.styles if style.id in used]
    font_graph = SimpleNamespace(text=SimpleNamespace(styles=styles))
    font_files = font_bindings(font_graph, fonts)
    package = Path(__file__).resolve().parents[1]
    root = package.parents[2]
    dependencies = [
        Path(__file__),
        *[Path(__file__).with_name(name+'.py') for name in (
            'region_candidates', 'candidate_stage', 'text_mask_completion',
            'gradient_backing', 'layerd_adapter', 'diagnostics')],
        *[package/(name+'.py') for name in (
            'parallel_text_candidates', 'parallel_rebuild', 'parallel_fusion',
            'parallel_candidate_stage', 'parallel_regions',
            'parallel_candidate_contracts', 'parallel_candidate_engine',
            'contracts', 'parallel_contracts', 'object_extract', 'object_contracts',
            'materialize')],
        root/'ppt-rebuild/src/reconstruction.py',
    ]
    layers = {}
    if layer_directory is not None:
        page = load_page(layer_directory)
        paths = [Path(layer_directory)/'manifest.json', page.source_path,
                 *[layer.path for layer in page.layers]]
        layers = {str(path.resolve()): digest(path) for path in paths}
    baseline = Path(baseline_directory).resolve()
    return {
        'version': 'R1204TextProbe/1', 'source_sha256': graph.source_sha256,
        'scene_sha256': key(graph.model_dump()), 'object_id': identifier,
        'font_files': font_files,
        'font_records': [row for row in fonts['fonts'] if str(Path(row['file']).resolve())
                         in font_files],
        'layers': layers,
        'baseline': {str(path.resolve()): digest(path)
                     for path in baseline.rglob('*') if path.is_file()},
        'code': {str(path.relative_to(root)): digest(path) for path in dependencies},
    }


def probe_one(graph, identifier, source_file, layer_directory, baseline_directory, fonts, directory):
    """Generate and stage the first safe local candidate without compile/QA.

    A complete receipt can be reused only while every candidate, baseline,
    source, layer, font and implementation binding still matches. Office and
    independent visual acceptance remain explicit subsequent tool operations.
    """
    binding = input_binding(graph, identifier, source_file, layer_directory, baseline_directory, fonts)
    target = Path(directory).resolve()/('text-'+key(binding)[:24])
    receipt = target/'probe.json'
    if receipt.is_file():
        cached = json.loads(receipt.read_text('utf8'))
        if cached['binding'] != binding or key(cached['result']) != cached['result_sha256']:
            raise ValueError('text probe cache binding mismatch')
        for path, sha in cached['files'].items():
            if digest(path) != sha:
                raise ValueError('text probe cached artifact changed')
        return {**cached['result'], 'cache_hit': True}
    target.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'R1204TextProbeResult/1', 'object_id': identifier,
              'source_sha256': graph.source_sha256, 'scene_sha256': binding['scene_sha256'],
              'state': 'BLOCKED', 'office': 'NOT_RUN', 'visual': 'NOT_RUN',
              'automatic_publication': False, 'attempts': [], 'cache_hit': False}
    stream = adaptive_candidates(graph, identifier, source_file, layer_directory, target/'backgrounds')
    try:
        for row in stream:
            result['attempts'].append({'candidate': row['candidate'], 'status': row['status']})
            if row['status'] != 'PRECHECK_PASS':
                continue
            candidate = target/'backgrounds'/row['candidate']
            ir = stage_background(candidate, graph, baseline_directory, fonts, target/'staged')
            result.update(state='STAGED_LOCAL', candidate_directory=str(candidate), staged_ir=str(ir),
                          residual_report_sha256=digest(candidate/'residual_report.json'),
                          stop_reason='first source-safe editable candidate staged; Office and visual QA required')
            break
    except (ValueError, OSError, KeyError) as exc:
        result.update(failure=getattr(exc, 'code', 'LOCAL_TEXT_RECOVERY_FAILED'),
                      error_type=type(exc).__name__, reason=str(exc))
    finally:
        stream.close()
    files = {str(path.resolve()): digest(path) for path in target.rglob('*') if path.is_file()}
    write(receipt, {'binding': binding, 'result': result, 'result_sha256': key(result), 'files': files})
    return result
