"""Record and validate region trial-compilation candidates.

A candidate is a concrete page revision, not a routing label. The state chain is
PROPOSED -> VALIDATED_STRUCTURE -> TRIAL_BUILT -> VERIFIED/REVIEW_REQUIRED ->
ACCEPTED/REJECTED. Acceptance stays a human decision; metrics only rank
candidates and never accept one. This tool never builds or edits a deck.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

SCHEMA_VERSION = 'region-candidates/0.1'
STATES = ('PROPOSED', 'VALIDATED_STRUCTURE', 'TRIAL_BUILT', 'VERIFIED',
          'REVIEW_REQUIRED', 'ACCEPTED', 'REJECTED')
TRANSITIONS = {
    'PROPOSED': {'VALIDATED_STRUCTURE'},
    'VALIDATED_STRUCTURE': {'TRIAL_BUILT'},
    'TRIAL_BUILT': {'VERIFIED', 'REVIEW_REQUIRED'},
    'VERIFIED': {'ACCEPTED', 'REJECTED'},
    'REVIEW_REQUIRED': {'ACCEPTED', 'REJECTED'},
    'ACCEPTED': set(),
    'REJECTED': set(),
}
TERMINAL = {'ACCEPTED', 'REJECTED'}
COVERAGE_STATES = {'covered', 'partial', 'missing', 'retained', 'not_measured'}
METRIC_KEYS = {
    'page_mean_pixel_difference', 'region_mean_pixel_difference',
    'text_correctness', 'edit_target_coverage', 'elapsed_seconds',
}
HASH_RE = re.compile(r'^[a-f0-9]{64}$')
ID_RE = re.compile(r'^[a-z0-9][a-z0-9_-]{1,80}$')


class CandidateError(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, sort_keys=False), encoding='utf8')


def _check_artifact(value, label, errors):
    if not isinstance(value, dict):
        errors.append(f'{label}: artifact must be an object')
        return
    path = value.get('path')
    if (not isinstance(path, str) or not path or Path(path).is_absolute()
            or '..' in Path(path).parts):
        errors.append(f'{label}: artifact path must be repo-relative')
    if not isinstance(value.get('sha256'), str) or not HASH_RE.fullmatch(value['sha256']):
        errors.append(f'{label}: artifact sha256 invalid')
    if not isinstance(value.get('label'), str) or not value['label'].strip():
        errors.append(f'{label}: artifact label missing')


def _check_metrics(value, label, errors):
    if value is None:
        return
    if not isinstance(value, dict):
        errors.append(f'{label}: metrics must be an object')
        return
    for key, metric in value.items():
        if key not in METRIC_KEYS:
            errors.append(f'{label}: unknown metric {key}')
            continue
        if metric is None:
            continue
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            errors.append(f'{label}: metric {key} must be numeric or null')


def _check_coverage(value, label, errors):
    if not isinstance(value, dict):
        errors.append(f'{label}: coverage must be an object')
        return
    for target, item in value.items():
        if not isinstance(target, str) or not target:
            errors.append(f'{label}: coverage target invalid')
            continue
        if not isinstance(item, dict) or item.get('status') not in COVERAGE_STATES:
            errors.append(f'{label}: coverage {target} invalid')
            continue
        if not isinstance(item.get('evidence'), str) or not item['evidence'].strip():
            errors.append(f'{label}: coverage {target} needs evidence')


def _check_budget(value, label, errors):
    if not isinstance(value, dict):
        errors.append(f'{label}: budget must be an object')
        return
    for key in ('nodes', 'text_nodes', 'images', 'shapes'):
        item = value.get(key)
        if item is not None and (type(item) is not int or item < 0):
            errors.append(f'{label}: budget {key} must be a non-negative integer')
    limits = value.get('limits')
    if limits is not None and not isinstance(limits, dict):
        errors.append(f'{label}: budget limits must be an object')


def _check_history(value, label, errors):
    if not isinstance(value, list) or not value:
        errors.append(f'{label}: history must be a non-empty list')
        return None
    states = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or item.get('state') not in STATES:
            errors.append(f'{label}: history item {index} has invalid state')
            return None
        if not isinstance(item.get('basis'), str) or not item['basis'].strip():
            errors.append(f'{label}: history item {index} needs a basis')
        states.append(item['state'])
    if states[0] != 'PROPOSED':
        errors.append(f'{label}: history must start at PROPOSED')
    for previous, current in zip(states, states[1:]):
        if current not in TRANSITIONS[previous]:
            errors.append(f'{label}: illegal transition {previous}->{current}')
    return states[-1]


def validate(data):
    errors = []
    if not isinstance(data, dict) or data.get('schema_version') != SCHEMA_VERSION:
        return [f'schema_version must be {SCHEMA_VERSION}']
    candidates = data.get('candidates')
    if not isinstance(candidates, list) or not candidates:
        return ['candidates must be a non-empty list']
    seen = set()
    for index, item in enumerate(candidates):
        label = f'candidates[{index}]'
        if not isinstance(item, dict):
            errors.append(f'{label}: must be an object')
            continue
        candidate_id = item.get('candidate_id')
        if not isinstance(candidate_id, str) or not ID_RE.fullmatch(candidate_id):
            errors.append(f'{label}: invalid candidate_id')
        elif candidate_id in seen:
            errors.append(f'{label}: duplicate candidate_id {candidate_id}')
        else:
            seen.add(candidate_id)
        for key in ('region_id', 'page', 'decision'):
            if not isinstance(item.get(key), str) or not item[key].strip():
                errors.append(f'{label}: {key} missing')
        last_state = _check_history(item.get('history'), label, errors)
        state = item.get('state')
        if state not in STATES:
            errors.append(f'{label}: invalid state')
        elif last_state is not None and state != last_state:
            errors.append(f'{label}: state does not match last history entry')
        for key in ('baseline', 'candidate'):
            block = item.get(key)
            if not isinstance(block, dict):
                errors.append(f'{label}: {key} missing')
                continue
            if not isinstance(block.get('label'), str) or not block['label'].strip():
                errors.append(f'{label}: {key}.label missing')
            artifacts = block.get('artifacts')
            if not isinstance(artifacts, list) or not artifacts:
                errors.append(f'{label}: {key}.artifacts must be non-empty')
            else:
                for artifact_index, artifact in enumerate(artifacts):
                    _check_artifact(artifact, f'{label}.{key}.artifacts[{artifact_index}]', errors)
            _check_metrics(block.get('metrics'), f'{label}.{key}.metrics', errors)
        candidate = item.get('candidate')
        if isinstance(candidate, dict):
            goals = candidate.get('edit_goals')
            if not isinstance(goals, list) or not goals or not all(isinstance(g, str) and g for g in goals):
                errors.append(f'{label}: candidate.edit_goals must be a non-empty list')
        _check_coverage(item.get('coverage'), label, errors)
        _check_budget(item.get('budget'), label, errors)
        outcome = item.get('outcome')
        if not isinstance(outcome, dict):
            errors.append(f'{label}: outcome missing')
            continue
        if outcome.get('status') != state:
            errors.append(f'{label}: outcome.status must equal state')
        reason = outcome.get('reason')
        if state in TERMINAL:
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f'{label}: terminal outcome needs a reason')
            if state == 'ACCEPTED' and outcome.get('human_acceptance') is not True:
                errors.append(f'{label}: ACCEPTED needs human_acceptance=true')
            if state == 'REJECTED' and outcome.get('human_acceptance') is not False:
                errors.append(f'{label}: REJECTED needs human_acceptance=false')
        elif outcome.get('human_acceptance') is not False:
            errors.append(f'{label}: non-terminal outcome must not claim human acceptance')
    return errors


def verify_files(data, root):
    root = Path(root).resolve()
    rows = []
    for candidate in data.get('candidates', []):
        for section in ('baseline', 'candidate'):
            for artifact in candidate.get(section, {}).get('artifacts', []):
                path = root / artifact['path']
                try:
                    resolved = path.resolve()
                    if not resolved.is_relative_to(root):
                        raise CandidateError('path escapes root')
                    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
                    ok = digest == artifact['sha256']
                    rows.append({'candidate_id': candidate['candidate_id'], 'section': section,
                                 'path': artifact['path'], 'exists': True,
                                 'sha256_match': ok, 'actual_sha256': digest})
                except (OSError, CandidateError) as exc:
                    rows.append({'candidate_id': candidate['candidate_id'], 'section': section,
                                 'path': artifact.get('path'), 'exists': False,
                                 'sha256_match': False, 'error': type(exc).__name__})
    return {'root': '.', 'ok': all(row['sha256_match'] for row in rows), 'rows': rows}


def report(data):
    lines = [
        '# Region candidate trials',
        '',
        'Metrics rank candidates only. A human decision, not a metric, accepts or rejects one.',
        '',
        '| Candidate | Page | State | Baseline MAE | Candidate MAE | Delta | Outcome |',
        '|---|---|---|---:|---:|---:|---|',
    ]
    for item in data['candidates']:
        baseline = item['baseline'].get('metrics') or {}
        candidate = item['candidate'].get('metrics') or {}
        before = baseline.get('page_mean_pixel_difference')
        after = candidate.get('page_mean_pixel_difference')
        delta = round(after - before, 3) if before is not None and after is not None else None
        outcome = item['outcome']
        lines.append(
            f'| {item["candidate_id"]} | {item["page"]} | {item["state"]} | '
            f'{before if before is not None else "-"} | {after if after is not None else "-"} | '
            f'{delta if delta is not None else "-"} | {outcome.get("status")}: {outcome.get("reason", "")} |')
    for item in data['candidates']:
        lines += ['', f'## {item["candidate_id"]}', '',
                  f'- region: {item["region_id"]}',
                  f'- routing decision: {item["decision"]}',
                  f'- baseline: {item["baseline"]["label"]}',
                  f'- candidate: {item["candidate"]["label"]}',
                  f'- edit goals: {", ".join(item["candidate"]["edit_goals"])}',
                  f'- target fidelity: {item["candidate"].get("target_fidelity", "not recorded")}',
                  f'- outcome: {item["outcome"]["status"]} — {item["outcome"]["reason"]}']
        for section in ('baseline', 'candidate'):
            note = item[section].get('note')
            if note:
                lines.append(f'- {section} note: {note}')
        contamination = item['candidate'].get('contamination')
        if contamination:
            lines.append(f'- contamination: {contamination}')
        budget = item.get('budget') or {}
        if budget:
            lines.append(f'- budget: nodes={budget.get("nodes", "-")}, '
                         f'text={budget.get("text_nodes", "-")}, images={budget.get("images", "-")}, '
                         f'shapes={budget.get("shapes", "-")}')
            for warning in budget.get('warnings', []):
                lines.append(f'- budget warning: {warning}')
        coverage = item.get('coverage') or {}
        if coverage:
            lines.append('')
            lines.append('| Edit target | Coverage | Evidence |')
            lines.append('|---|---|---|')
            for target, value in coverage.items():
                lines.append(f'| {target} | {value["status"]} | {value["evidence"]} |')
        lines.append('')
        lines.append('History: ' + ' -> '.join(entry['state'] for entry in item['history']))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    q = sub.add_parser('validate')
    q.add_argument('--records', required=True)
    q.add_argument('--output')
    q = sub.add_parser('verify-files')
    q.add_argument('--records', required=True)
    q.add_argument('--root', required=True)
    q.add_argument('--output')
    q = sub.add_parser('report')
    q.add_argument('--records', required=True)
    q.add_argument('--output', required=True)
    args = parser.parse_args()
    data = read_json(args.records)
    if args.command == 'validate':
        errors = validate(data)
        result = {'schema_version': SCHEMA_VERSION, 'valid': not errors, 'errors': errors}
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        if errors:
            raise SystemExit(1)
    elif args.command == 'verify-files':
        result = verify_files(data, args.root)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        if not result['ok']:
            raise SystemExit(1)
    else:
        text = report(data)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding='utf8')
        print(json.dumps({'report': args.output}, ensure_ascii=False))


if __name__ == '__main__':
    main()
