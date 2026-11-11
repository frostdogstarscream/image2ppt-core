"""Evidence-bound text tool actions for a host-registered Agent allowlist."""
import json
from pathlib import Path

from ..materialize import digest
from ..parallel_contracts import SceneGraph
from ..parallel_store import key
from .text_probe import input_binding, probe_one


def read_probe_receipt(file):
    record = json.loads(Path(file).read_text('utf8'))
    if record['result_sha256'] != key(record['result']):
        raise ValueError('previous text receipt changed')
    for path, sha in record['files'].items():
        if digest(path) != sha:
            raise ValueError('previous text evidence changed')
    return record


def read_trial_receipt(graph, identifier, directory):
    candidate = Path(directory).resolve()
    files = json.loads((candidate/'hashes.json').read_text('utf8'))
    for name, sha in files.items():
        if Path(name).name != name or digest(candidate/name) != sha:
            raise ValueError('previous background evidence changed')
    residual = json.loads((candidate/'residual_report.json').read_text('utf8'))
    if (residual['source_sha256'] != graph.source_sha256 or residual['object_id'] != identifier
            or residual['scene_graph_sha256'] != key(graph.model_dump())):
        raise ValueError('previous background is not bound to this source scene')
    return files, residual


def retry_text_mask_completion(graph, identifier, source_file, layer_directory,
                               baseline_directory, fonts, directory, *, previous_receipt):
    """Complete a proved partial glyph mask, or reuse an unchanged failure."""
    previous_file = Path(previous_receipt)
    payload = json.loads(previous_file.read_text('utf8'))
    if payload.get('schema') == 'R12RegionalResidualReport/1':
        _, previous = read_trial_receipt(graph, identifier, previous_file.parent)
        if (payload != previous or previous['status'] != 'REVIEW_REQUIRED'
                or previous['mask_variant'] not in ('original', 'bounded_expansion')
                or previous['checks']['complete_visible_scope']
                or not previous['visible_source_pixels_outside_mask']):
            raise ValueError('mask completion retry requires proved missing original glyph pixels')
        result = probe_one(graph, identifier, source_file, layer_directory,
                           baseline_directory, fonts, directory)
        return {**result, 'action': 'retry_text_mask_completion',
                'action_implementation_sha256': digest(__file__),
                'strategy_changed': previous['mask_variant']+' to TextMaskCompletion',
                'previous_missing_pixels': previous['visible_source_pixels_outside_mask'],
                'previous_receipt_sha256': digest(previous_receipt)}
    previous = read_probe_receipt(previous_receipt)
    old = previous['binding']
    if old['source_sha256'] != graph.source_sha256 or old['object_id'] != identifier:
        raise ValueError('previous mask failure belongs to another object or source')
    if previous['result'].get('failure') != 'MASK_INCOMPLETE':
        raise ValueError('mask retry requires an actual incomplete-mask receipt')
    current = input_binding(graph, identifier, source_file, layer_directory, baseline_directory, fonts)
    fields = ('source_sha256', 'scene_sha256', 'layers', 'code')
    changed = any(current[field] != old[field] for field in fields)
    if changed:
        result = probe_one(graph, identifier, source_file, layer_directory,
                           baseline_directory, fonts, directory)
    else:
        result = {**previous['result'], 'cache_hit': True,
                  'stop_reason': 'mask evidence and strategy unchanged; preserved incomplete-mask result'}
    return {**result, 'action': 'retry_text_mask_completion', 'input_evidence_changed': changed,
            'action_implementation_sha256': digest(__file__),
            'previous_receipt_sha256': digest(previous_receipt)}


def switch_background_candidate(graph, identifier, source_file, layer_directory,
                                baseline_directory, fonts, directory, *,
                                rejected_candidate_directory, rejection_report):
    """Advance past a background rejected by bound source or Office checks."""
    candidate = Path(rejected_candidate_directory).resolve()
    files, residual = read_trial_receipt(graph, identifier, candidate)
    report = json.loads(Path(rejection_report).read_text('utf8'))
    if report.get('schema') == 'R12RegionalResidualReport/1':
        required = ('complete_visible_scope', 'protected_integrity', 'outside_mask_integrity')
        if (report != residual or report['status'] != 'REVIEW_REQUIRED'
                or not all(report['checks'][check] for check in required)):
            raise ValueError('local background switch needs complete legal mask and failed repair evidence')
    elif report.get('schema') == 'R12RegionalAcceptance/1':
        if (report.get('source_sha256') != graph.source_sha256
                or report.get('regional_automatic_checks') == 'PASS'
                or report.get('failure') != 'BACKGROUND_REPAIR_FAILED'):
            raise ValueError('background switch requires a real background-specific rejection')
        scene = SceneGraph.model_validate_json(
            (Path(rejection_report).parent/'region/scene.json').read_text('utf8'))
        proof = scene.selected.get(identifier, {}).get('recovery_evidence', {})
        if (scene.source_sha256 != graph.source_sha256 or proof.get('hashes') != files
                or Path(proof.get('trial_path', '')).resolve() != candidate):
            raise ValueError('regional rejection does not describe this background candidate')
    else:
        raise ValueError('unsupported background rejection receipt')
    result = probe_one(graph, identifier, source_file, layer_directory,
                       baseline_directory, fonts, directory,
                       excluded_candidates=[residual['candidate']])
    return {**result, 'action': 'switch_background_candidate',
            'action_implementation_sha256': digest(__file__),
            'rejected_candidate': residual['candidate'],
            'rejection_report_sha256': digest(rejection_report)}


def classify_failure(result):
    if result['state'] in ('STAGED_LOCAL', 'VERIFIED_REUSED'):
        return None
    code = result.get('failure', '')
    reason = result.get('reason', '').lower()
    if code in ('MASK_INCOMPLETE', 'INVALID_MASK', 'EMPTY_WRITE_MASK', 'MASK_OUTSIDE_SCOPE'):
        return 'MASK_INCOMPLETE'
    if code == 'FONT_LAYOUT_FAILED' or 'native font frame' in reason:
        return 'FONT_LAYOUT_FAILED'
    if code in ('OWNER_CONFLICT', 'PROTECTED_NEIGHBOR') or any(
            item in reason for item in ('owner', 'protected peer', 'indivisible')):
        return 'OWNER_CONFLICT'
    if code in ('QA_INFRASTRUCTURE_UNAVAILABLE', 'INPAINT_RESPONSE_UNKNOWN') or result.get(
            'error_type') in ('TimeoutError', 'ConnectionError', 'PermissionError', 'FileNotFoundError'):
        return 'TOOL_UNAVAILABLE'
    if any(item in reason for item in ('font', 'glyph', 'native frame')):
        return 'FONT_LAYOUT_FAILED'
    return 'BACKGROUND_REPAIR_FAILED'
