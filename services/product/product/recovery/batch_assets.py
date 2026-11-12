"""Bounded, sequential independent asset transactions over frozen evidence."""
import json
from pathlib import Path
import time

from ..materialize import digest, write
from ..parallel_contracts import SceneGraph
from ..parallel_regions import regions
from ..parallel_store import key
from .asset_cache import file_hashes
from .asset_retry import protected_for_asset
from .agent_asset_stage import stage_asset_candidate
from .single_asset import probe, verify_office


def failure_inventory(graph, job):
    historical = {}
    parents = []
    visited = set()
    current = Path(job).resolve()
    while current not in visited:
        visited.add(current)
        parents.append(current)
        request = current/'request.json'
        if not request.is_file():break
        previous = json.loads(request.read_text('utf8')).get('reuse_verified_from')
        if not previous:break
        current = Path(previous).resolve()
    for parent in reversed(parents):
        for path in sorted((parent/'r12').glob('attempt-*/page-*/ax/*/c/result.json')):
            row = json.loads(path.read_text('utf8'))
            if row.get('source_sha256') != graph.source_sha256:continue
            historical[row['object_id']] = {'receipt': str(path.resolve()), 'sha256': digest(path), **row}
    closures = regions(graph)[1]
    rows = []
    for obj in graph.asset.objects:
        if obj.id in graph.selected:
            continue
        previous = historical.get(obj.id, {})
        diagnostic = previous.get('seed_diagnostics', {})
        reasons = []
        if graph.ownership[obj.id] != 'RESOLVED':
            reasons.append('OWNER_UNRESOLVED')
        if closures[obj.id]['member_ids'] != [obj.id]:
            reasons.append('DEPENDENCY_UNRESOLVED')
        if not diagnostic.get('alpha_visible_pixels') or diagnostic.get('alpha_ambiguities'):
            reasons.append('ALPHA_MISSING_OR_UNVERIFIED')
        if diagnostic.get('source_unmix_failure'):
            reasons.append('SOURCE_UNMIX_FAILED')
            if 'background' in diagnostic['source_unmix_failure']:
                reasons.append('BACKGROUND_INSUFFICIENT')
        if diagnostic.get('sam') == 'NOT_SUPPLIED' or not previous:
            reasons.append('SAM_EVIDENCE_INSUFFICIENT')
        if not reasons:
            reasons.append('ALPHA_MISSING_OR_UNVERIFIED')
        rows.append({'object_id': obj.id, 'description': obj.description, 'prompt_en': obj.prompt_en,
                     'kind': obj.kind, 'owner': graph.ownership[obj.id],
                     'members': closures[obj.id]['member_ids'], 'failure_categories': reasons,
                     'prior_failure': previous,
                     'eligible': obj.kind == 'icon' and graph.ownership[obj.id] == 'RESOLVED'
                                 and closures[obj.id]['member_ids'] == [obj.id]})
    return rows


def recover(graph, source, layers, baseline, job, output, fonts, *, limit=5, office=False, sam_budget=5,sam_cache=None):
    if not 1 <= limit <= 5 or sam_budget < 0:
        raise ValueError('batch limit must be 1..5 and SAM budget nonnegative')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = failure_inventory(graph, job)
    write(output/'failure-inventory.json', inventory)
    targets = [row['object_id'] for row in inventory if row['eligible']][:limit]
    initial = graph
    current = Path(baseline).resolve()
    results = []
    used_sam = 0
    started = time.monotonic()
    for identifier in targets:
        object_start = time.monotonic()
        folder = output/'objects'/identifier
        folder.mkdir(parents=True, exist_ok=True)
        graph = SceneGraph.model_validate_json((current/'scene.json').read_text('utf8'))
        row = {'object_id': identifier, 'status': 'LOCAL_FALLBACK', 'attempts': [],
               'parent_scene_sha256': digest(current/'scene.json'), 'model_calls': 0}
        results.append(row)
        try:
            strategies = ['source-halo', 'verified-backing', 'sam']
            old_protection = protected_for_asset(graph, current, identifier)
            new_protection = protected_for_asset(graph, current, identifier, background_context=True)
            if (old_protection == new_protection).all():
                strategies.remove('verified-backing')
            for strategy in strategies:
                if strategy == 'sam' and used_sam >= sam_budget and sam_cache is None:
                    row['stop_reason'] = 'SAM_BUDGET_EXHAUSTED'
                    break
                candidate = probe(graph, identifier, source, layers, current, folder/strategy, strategy=strategy,
                                  sam_cache=sam_cache,allow_model_call=used_sam<sam_budget)
                row['attempts'].append(candidate)
                if candidate['status'] != 'CANDIDATE_READY':
                    row['failure_evidence'] = {'strategy': strategy, 'reason': candidate.get('reason'),
                                               'seed_diagnostics': candidate.get('seed_diagnostics'),
                                               'attempts': candidate.get('attempts'),
                                               'sam_source_failure': candidate.get('sam_source_failure')}
                calls = candidate.get('model_calls', 0)
                row['model_calls'] += calls
                used_sam += calls
                if candidate['status'] == 'CANDIDATE_READY':
                    row.update(status='CANDIDATE_READY', strategy=strategy,
                               candidate_directory=candidate['candidate_directory'])
                    if office:
                        checked = verify_office(candidate, graph, source, layers, current, fonts, folder/strategy)
                        row['office'] = checked
                        if checked['status'] != 'PASS':
                            row.update(status='LOCAL_FALLBACK', stop_reason='OFFICE_OR_RESIDUAL_REJECTED')
                            break
                    stage_root = folder/'stages'/key(candidate['candidate_hashes'])[:16]
                    stage_receipt = stage_root/'receipt.json'
                    if stage_receipt.exists():
                        saved = json.loads(stage_receipt.read_text('utf8'))
                        if saved['candidate'] != candidate['candidate_hashes'] or file_hashes(stage_root/'s') != saved['files']:
                            raise ValueError('batch staged evidence changed')
                        ir = stage_root/'s/slideir.json'
                    else:
                        stage_root.mkdir(parents=True, exist_ok=True)
                        ir = stage_asset_candidate(candidate['candidate_directory'], graph, current,
                                                   stage_root/'s', layer_directory=layers)
                        write(stage_receipt, {'candidate': candidate['candidate_hashes'], 'files': file_hashes(ir.parent)})
                    proposed = SceneGraph.model_validate_json((ir.parent/'scene.json').read_text('utf8'))
                    if ({t.id: t.model_dump() for t in initial.text.texts} !=
                            {t.id: t.model_dump() for t in proposed.text.texts}):
                        raise ValueError('batch changed confirmed text definitions')
                    for old_id, selected in graph.selected.items():
                        if proposed.selected.get(old_id) != selected:
                            raise ValueError('batch changed a previously selected object')
                        if selected['kind'] == 'asset':
                            reference = selected['asset_reference']
                            if digest(ir.parent/reference['path']) != reference['sha256']:
                                raise ValueError('batch changed stable PNG bytes')
                    current = ir.parent
                    row.update(status='VERIFIED_LOCAL' if office else 'STAGED_LOCAL',
                               ir=str(ir), stop_reason='FIRST_ACCEPTABLE_CANDIDATE')
                    break
                if candidate['status'] == 'UNKNOWN':
                    row['stop_reason'] = 'TOOL_UNKNOWN_NO_IDENTICAL_RETRY'
                    break
            if row['status'] == 'LOCAL_FALLBACK' and 'reason' not in row:
                failed = row.get('failure_evidence', {})
                row['reason'] = failed.get('sam_source_failure') or failed.get('reason') or 'candidate hard gates failed'
        except Exception as exc:
            row.update(status='LOCAL_FALLBACK', reason=str(exc)[:500], error_type=type(exc).__name__)
        row['elapsed_seconds'] = round(time.monotonic()-object_start, 3)
        write(folder/('l2.json' if office else 'l1.json'), row)
    report = {'schema': 'BatchAssetRecovery/1', 'source_sha256': initial.source_sha256,
              'targets': targets, 'objects': results, 'combined_ir': str(current/'slideir.json'),
              'verified_candidates': [r['candidate_directory'] for r in results if r['status'] == 'VERIFIED_LOCAL'],
              'model_calls': sum(r['model_calls'] for r in results),
              'elapsed_seconds': round(time.monotonic()-started, 3),
              'final_qa': 'NOT_RUN', 'automatic_publication': False}
    write(output/('batch-l2.json' if office else 'batch-l1.json'), report)
    return report
