"""Freeze source/hash-selected holdouts, then run the explicitly authorized loop.

freeze is entirely local. run uses the existing configured development worker
and optional model transport. It refuses source or rule drift after freezing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'services/vision/src')]
from image2ppt_vision.storage import digest
from image2ppt_vision.layered.quality_loop import read, write
from image2ppt_vision.layered.loop_checks import fingerprint


def code_files():
    files = [p for folder in ('scripts', 'src', 'services/vision/src') for p in (ROOT/folder).rglob('*')
             if p.suffix in ('.py', '.mjs') and '__pycache__' not in p.parts]
    files.append(ROOT/'image2ppt.ps1')
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}


def baseline_pages(base):
    base = Path(base).resolve(strict=True); job = read(base/'job.json'); rows = []
    for name, page in job['pages'].items():
        source = base/'input'/(name+'.png')
        row = {'name': name, 'path': str(source), 'sha256': digest(source)}
        if page.get('pptx'):
            pptx = base/'pages'/name/page['pptx']; preview = base/'pages'/name/page['preview']
            if digest(pptx) != page['pptx_sha256']:
                raise ValueError('baseline PPTX changed')
            row.update(pptx=str(pptx), pptx_sha256=digest(pptx), preview=str(preview), preview_sha256=digest(preview),
                       metrics=page.get('metrics', {}), status=page['status'])
        rows.append(row)
    return rows


def freeze(args):
    regression = baseline_pages(args.baseline_job)
    if len(regression) < 10: raise ValueError('need at least ten baseline pages')
    existing = {r['sha256'] for r in regression}; candidates = {}
    for file in Path(args.independent_input_dir).resolve(strict=True).rglob('*.png'):
        sha = digest(file)
        if sha not in existing:
            candidates.setdefault(sha, {'name': file.stem, 'path': str(file), 'sha256': sha})
    holdout = [candidates[k] for k in sorted(candidates)[:args.count]]
    if len(holdout) < 10: raise ValueError('need ten distinct independent source hashes')
    if len({r['name'] for r in holdout}) != len(holdout): raise ValueError('holdout names must be unique')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    rules = code_files()
    manifest = {'schema': 'LayeredValidationSet/1', 'selection': 'ascending SHA256; exclude all regression hashes',
                'rules': rules, 'rules_sha256': fingerprint(rules), 'baseline_job': str(Path(args.baseline_job).resolve()),
                'regression': regression, 'independent': holdout,
                'coverage': {'layout_categories': 'requires visual inventory; hash diversity alone is not layout coverage',
                             'uncovered': ['unverified background/layout types']},
                'human_acceptance': False, 'office_acceptance': False}
    write(output/'frozen.json', manifest)
    print(output/'frozen.json')


def verify_frozen(manifest):
    if code_files() != manifest['rules']: raise ValueError('rules changed after freeze; create a new explicit freeze')
    for row in manifest['regression']+manifest['independent']:
        for field in ('path', 'pptx', 'preview'):
            if field in row and digest(row[field]) != row['sha256' if field == 'path' else field+'_sha256']:
                raise ValueError('frozen artifact changed: '+row[field])


def run(args):
    from image2ppt_vision.layered.quality_loop import run as execute
    from image2ppt_vision.layered.remote import load_config
    manifest = read(args.manifest); verify_frozen(manifest)
    selected = manifest[args.dataset]; remote = model = None
    if args.model_config:
        from image2ppt_vision.auto_transport import SSHRemote, load_config as client
        remote, model = SSHRemote(client(args.model_config)), args.model
    result = execute([r['path'] for r in selected], args.output, worker_config=load_config(args.worker_config),
                     project_root=ROOT, remote=remote, model=model, vision_model=model,
                     baseline_job=manifest['baseline_job'] if args.dataset == 'regression' else None,
                     max_extra_vision_calls=args.max_extra_vision_calls, max_repair_rounds=args.max_repair_rounds)
    verify_frozen(manifest)
    report(Path(args.output)/result['job'], manifest, args.dataset)


def report(job_root, manifest, dataset):
    result = read(job_root/'job.json'); rows = []
    for row in manifest[dataset]:
        page = result['pages'][row['name']]; quality = page.get('quality', {})
        rows.append({'page': row['name'], 'source_sha256': row['sha256'], 'status': page['status'],
                     'baseline_metrics': row.get('metrics'), 'selected_metrics': page.get('metrics'),
                     'counts': page.get('counts'), 'issues': quality.get('issues', []), 'error': page.get('error'),
                     'extra_visual_requests': page.get('extra_visual_requests', 0),
                     'repair_attempts': page.get('repair_attempts', 0), 'selected_revision': page.get('selected_revision')})
    summary = {'schema': 'LayeredValidationReport/1', 'dataset': dataset, 'rules_sha256': manifest['rules_sha256'],
               'pages': rows, 'automated_findings_resolved': all(p['status'] != 'FAILED' and not p['issues'] for p in rows),
               'cross_sample_acceptance': 'REVIEW_REQUIRED', 'human_acceptance': False, 'office_acceptance': False}
    write(job_root/'validation-report.json', summary)
    lines = ['# Quality-loop validation', '', 'Actual artifacts are linked from job.json. This is not human/Office acceptance.', '',
             '| Page | Status | Extra visual calls | Repairs | Selected revision | Issues |', '|---|---|---:|---:|---|---:|']
    for row in rows:
        lines.append(f"| {row['page']} | {row['status']} | {row['extra_visual_requests']} | {row['repair_attempts']} | {row['selected_revision']} | {len(row['issues'])} |")
    (job_root/'VALIDATION.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('freeze'); p.add_argument('--baseline-job', required=True)
    p.add_argument('--independent-input-dir', required=True); p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=10); p.set_defaults(handler=freeze)
    p = sub.add_parser('run'); p.add_argument('--manifest', required=True)
    p.add_argument('--dataset', choices=('regression', 'independent'), required=True)
    p.add_argument('--worker-config', required=True); p.add_argument('--model-config'); p.add_argument('--model', default='deepseek-flash')
    p.add_argument('--output', required=True); p.add_argument('--max-extra-vision-calls', type=int, choices=range(3), default=2)
    p.add_argument('--max-repair-rounds', type=int, choices=range(3), default=2); p.set_defaults(handler=run)
    args = parser.parse_args(); args.handler(args)


if __name__ == '__main__': main()
