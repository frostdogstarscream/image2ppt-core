"""Opt-in quality-loop orchestration. No imports start remote or model work."""
import copy
import json
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from ..storage import digest
from . import emit_ir, qa, review, semantics, typography
from . import loop_checks as checks, loop_regions as regions, loop_styles, loop_scan, loop_probes
from .loop_policy import LoopPolicy
from .loop_review import CropReviewer, clip, intersect
from .loop_text import normalized, regional_text
from .remote import Worker


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def code_inventory(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): digest(p)
            for directory in ('scripts', 'src', 'services/vision/src')
            for p in sorted((root/directory).rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.mjs') and '__pycache__' not in p.parts}


def worker_ocr(worker, job, source, files, directory, subdir):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    worker.upload(job, files, subdir=subdir)
    prefix = '/job/'+subdir
    worker.run(job, ['ocr', prefix+'/ocr.json', *[prefix+'/'+Path(p).name for p in files]], source=source)
    worker.download(job, subdir+'/ocr.json', directory)
    return read(directory/'ocr.json')


def corroborate(report, scan, source_file, directory, worker, job, source, subdir):
    """Changed text requires independent crop OCR; no language-only correction."""
    lines = {t['id']: t for t in scan['ocr']}; records = {r['id']: r for r in scan['text']}
    changed = [d for d in report['decisions'] if d['id'] in lines and d['kind'] == 'text'
               and normalized(d['text']) != normalized(lines[d['id']]['text']) and d['text'].strip()]
    observed = {}; failures = []
    if changed:
        folder = Path(directory); folder.mkdir(parents=True, exist_ok=False); files = []
        with Image.open(source_file) as im:
            for d in changed:
                tid = d['id']; box = records.get(tid, {}).get('ink_bbox') or lines[tid]['bbox']
                x, y, w, h = clip(box, scan['image_size'], 2)
                file = folder/(tid+'.png'); im.crop((x, y, x+w, y+h)).save(file); files.append(file)
        try:
            values = worker_ocr(worker, job, source, files, folder/'result', subdir)
            observed = {Path(k).stem: ''.join(h['text'] for h in sorted(v, key=lambda t: (t['bbox'][1], t['bbox'][0])))
                        for k, v in values.items()}
        except Exception as error:
            failures.append({'code': 'independent_ocr_unavailable', 'detail': str(error)[:500]})
    corrections = {}; held = {}; actions = []
    for d in report['decisions']:
        if d['id'] not in lines:
            continue
        tid = d['id']; original = lines[tid]['text']; action = 'keep_native'
        if d['kind'] == 'text' and normalized(d['text']) == normalized(original):
            pass
        elif d['kind'] == 'text' and normalized(observed.get(tid, '')) == normalized(d['text']) and d['text'].strip():
            corrections[tid] = {'text': d['text'], 'independent_text': d['text'],
                                'raw_independent_text': observed[tid], 'original_text': original}
            action = 'correct_native'
        elif d['kind'] != 'keep':
            held[tid] = 'confirmed_icon' if d['kind'] == 'icon' else 'recognition_uncertain'
            action = 'retain_source'
        actions.append({'id': tid, 'action': action, 'original_text': original, 'visual': d})
    return {'source_sha256': scan['source_sha256'], 'corrections': corrections, 'retained': held,
            'actions': actions, 'issues': failures}


def retained_regions(perception, retained):
    lines = {t['id']: t for t in perception['ocr']}; records = {t['id']: t for t in perception['text']}
    return [{'bbox': records.get(tid, {}).get('badge', {}).get('bbox') or lines[tid]['bbox'], 'code': reason}
            for tid, reason in retained.items() if tid in lines]


def graph_regions(perception, issues):
    """Expand to the smallest complete nearby source object where possible."""
    W, H = perception['image_size']; output = []
    for item in issues:
        if not item.get('bbox'):
            continue
        box = item['bbox']
        choices = [e['bbox'] for e in perception['elements'] if regions.inside(box, clip(e['bbox'], [W, H], 4))
                   and e['bbox'][2]*e['bbox'][3] <= min(.12*W*H, box[2]*box[3]*8)]
        if choices:
            box = min(choices, key=lambda b: b[2]*b[3])
        output.append({'bbox': box, 'code': item['code']})
    return output


def _post_candidates(perception, problems):
    rows = []; known = {t['id']: t for t in perception['ocr']}
    for i, problem in enumerate(problems):
        box = problem.get('bbox')
        if not box:
            continue
        tid = problem.get('target')
        if tid not in known:
            overlapping = [t for t in known.values() if intersect(t['bbox'], box) >= .5*t['bbox'][2]*t['bbox'][3]]
            tid = overlapping[0]['id'] if len(overlapping) == 1 and problem['code'].startswith('text') else f'g{i:03d}'
        if any(r['id'] == tid for r in rows):
            continue
        rows.append({'id': tid, 'text': known.get(tid, {}).get('text', ''), 'bbox': clip(box, perception['image_size']),
                     'context_bbox': clip(box, perception['image_size'], 12), 'reasons': [problem['code']],
                     'priority': 100 if problem['confirmed'] else 80 if tid in known else 60})
    return sorted(rows, key=lambda r: -r['priority'])


def build_page(name, raw_dir, page_dir, *, project_root, worker, job, source, reviewer, policy,
               styles, calibration, learned, remote=None, model=None, vision_model=None, baseline=None):
    from ..candidate_workflow import ArtifactCompiler
    from image2ppt.compiler import ArtifactBackend
    from .pipeline import validate_ir, ocr_coverage_gaps
    raw_dir, page_dir = Path(raw_dir), Path(page_dir)
    original = read(raw_dir/'perception.json'); immutable = checks.inventory(raw_dir)
    if digest(raw_dir/'source.png') != original['source_sha256']:
        raise ValueError('source mismatch')
    snapshot = page_dir/'working-perception'; shutil.copytree(raw_dir, snapshot)
    perception = copy.deepcopy(original)
    perception, object_review = review.review(snapshot, perception, remote=remote, model=vision_model,
                                               record_dir=page_dir, quality_loop=True)
    with Image.open(raw_dir/'source.png') as im:
        rgb = np.asarray(im.convert('RGB')).copy()
    style_changes = loop_styles.apply(rgb, perception, styles, learned)
    held = dict(perception.get('quality_guards', {}).get('retained', {}))
    from .pipeline import doubtful_lines
    held.update({tid: 'doubtful_short_line' for tid in doubtful_lines(perception, styles)})
    table_ids = {i for t in perception['tables'] for i in t['owned_text_ids']}
    held.update({t['id']: 'unfitted_source_text' for t in perception['ocr'] if t['id'] not in styles and t['id'] not in table_ids})
    initial_regions = retained_regions(perception, held)
    grouped_styles = {k: v for k, v in styles.items() if k not in table_ids and k not in held}
    containers = [e['bbox'] for e in perception['elements'] if e['kind'] == 'native_shape']
    try:
        paragraphs, semantic_report = semantics.group(grouped_styles, containers,
                            remote=remote, model=model, record_dir=page_dir)
    except Exception as error:
        paragraphs, semantic_report = semantics.group(grouped_styles, containers)
        semantic_report['issues'].append('semantic_transport_unavailable:'+type(error).__name__)
    write(page_dir/'typography.json', {'styles': styles, 'calibration': calibration, 'changes': style_changes,
                                     'font_inventory': loop_styles.font_inventory(styles)})
    write(page_dir/'semantics.json', {**semantic_report, 'paragraphs': paragraphs})
    with Image.open(raw_dir/'notext.png') as im:
        notext = np.asarray(im.convert('RGB'))
    protected = [box for r in perception['text'] for box in r.get('icon_candidates', [])]
    erasures = checks.graphic_losses(rgb, notext, [t['bbox'] for t in perception['ocr']], stage='inpainting', protected=protected)
    write(page_dir/'stage-losses.json', erasures)
    compiler = ArtifactCompiler(root=project_root); backend = ArtifactBackend(root=project_root, timeout=600)
    builds = []; attempts = []; post_done = False; pending_regions = initial_regions
    reference_boxes = [t['bbox'] for t in original['ocr']] + [e['bbox'] for e in original['elements']]
    current = None; confirmed_regions = []

    def compile_revision(index, repairs):
        revision = page_dir/f'r{index:02d}'; revision.mkdir()
        write(revision/'attempt.json', {'index': index, 'repairs': repairs})
        proposal = revision/'proposal'
        ps = paragraphs if index == 0 else semantics.group({k: v for k, v in styles.items() if k not in table_ids and k not in held},
                       [e['bbox'] for e in perception['elements'] if e['kind'] == 'native_shape'])[0]
        ir = emit_ir.emit(snapshot, perception, styles, ps, proposal,
                          notes=['quality-loop: source-bound local retention; machine checks are not acceptance.'])
        restoration = regions.restore(ir, proposal, repairs)
        module_report = regions.group_modules(ir, perception)
        # Emitter saved the initial IR; this is the sole new-proposal finalization.
        (proposal/'slideir.json').write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding='utf-8')
        ok, message = validate_ir(project_root, proposal/'slideir.json')
        if not ok or ir.get('unresolved_evidence_ids') or ocr_coverage_gaps(perception, ir):
            raise ValueError('invalid or incomplete proposal: '+message)
        write(revision/'lineage.json', checks.lineage(original, perception, ir, restoration))
        write(revision/'modules.json', module_report)
        built = compiler.build(proposal/'slideir.json', revision/'compiled')
        file_check = checks.final_file(ir, built['pptx']); write(revision/'final-file.json', file_check)
        issues = list(file_check['issues']) + checks.text_geometry(ir)
        with Image.open(built['preview']) as im:
            rendered = np.asarray(im.convert('RGB'))
        losses = checks.graphic_losses(rgb, rendered, [t['bbox'] for t in original['ocr']], protected=protected)
        issues.extend(losses)
        for finding in confirmed_regions:
            # Confirmation belongs to a specific region. It can only be cleared
            # by preserving that complete region or verified source agreement.
            kept = any(regions.inside(finding['bbox'], r['bbox']) for r in restoration['regions'])
            if not kept and checks.measured_error(rgb, rendered, [finding['bbox']]) > 2:
                issues.append(finding)
        issues.extend(checks.issue('retention_limit', r['bbox'], reasons=r['reasons']) for r in restoration['rejected'])
        metrics = qa.measure(raw_dir/'source.png', built['preview'], ir, directory=revision)
        # Retained source pixels are disclosed, not scored as native OCR success.
        retained_ids = {eid for r in restoration['regions'] for eid in r['evidence_ids']}
        try:
            results = worker_ocr(worker, job, source, [Path(built['preview'])], revision/'rendered-ocr', f'qa/{name}/r{index:02d}')
            hits = results[Path(built['preview']).name]
            text_report = regional_text(perception, hits, render_size=[rendered.shape[1], rendered.shape[0]], retained_ids=retained_ids)
            metrics['ocr_char_recall'] = qa.text_recall(perception['ocr'], hits)
            for row in text_report['mismatches']:
                issues.append(checks.issue('text_disagreement', row['bbox'], target=row['id'], comparison=row))
            for row in text_report['unexpected_text']:
                issues.append(checks.issue('unexpected_text', row['bbox'], observed=row['text']))
        except Exception as error:
            text_report = {'error': str(error)[:500]}
            issues.append(checks.issue('regional_ocr_unavailable', detail=str(error)[:500]))
        write(revision/'regional-text.json', text_report)
        probes = loop_probes.run(ir, proposal, built['pptx'], built['preview'], revision/'probes', backend)
        issues.extend(probes['issues']); write(revision/'edit-probes.json', probes)
        result = {'index': index, 'revision': revision, 'ir': ir, 'built': built, 'retention': restoration,
                  'issues': issues, 'region_error': checks.measured_error(rgb, rendered, reference_boxes),
                  'native_count': sum(n['type'] == 'shape' or n['type'] == 'text' and
                                      all(styles.get(eid, {}).get('fitted') for eid in n['evidence_ids']) for n in ir['nodes']),
                  'metrics': metrics,
                  'effective_perception': copy.deepcopy(perception)}
        write(revision/'checks.json', {k: result[k] for k in ('index', 'issues', 'region_error', 'native_count', 'metrics')})
        builds.append(result)
        return result

    for index in range(policy.max_repair_rounds+1):
        # Reserve a revision before its compilation, including failed attempts.
        try:
            candidate = compile_revision(index, pending_regions)
            accepted = current is None or checks.better(candidate, current)
            attempts.append({'index': index, 'compiled': True, 'accepted': accepted, 'score': list(checks.score(candidate))})
            if accepted:
                current = candidate
        except Exception as error:
            attempts.append({'index': index, 'compiled': False, 'accepted': False, 'error': str(error)[:2000]})
            write(page_dir/f'r{index:02d}'/'failure.json', attempts[-1])
            if current is None:
                raise
            # A failed repair consumes its slot; preserve the last valid result.
            continue
        if index == policy.max_repair_rounds:
            break
        problems = current['issues']
        fixes = [p for p in problems if p.get('bbox') and p['code'] in
                 ('graphic_loss', 'actual_duplicate', 'text_box_overlap', 'text_disagreement', 'unexpected_text')]
        if not post_done:
            candidates = _post_candidates(perception, fixes)
            post = reviewer.run(candidates, raw_dir/'source.png', phase='post', preview=current['built']['preview'])
            post_done = True
            decision = corroborate(post, perception, raw_dir/'source.png', page_dir/'post-independent', worker, job, source,
                                   f'independent/{name}/post')
            # A changed OCR string is not retyped without a fresh successful fit.
            for tid, correction in decision['corrections'].items():
                changed = copy.deepcopy(perception)
                for line in changed['ocr']:
                    if line['id'] == tid: line['text'] = correction['text']
                for record in changed['text']:
                    if record['id'] == tid: record.pop('icon_text_split', None)
                fitted, _ = typography.measure(rgb, changed)
                if fitted.get(tid, {}).get('fitted'):
                    styles[tid] = fitted[tid]; perception = changed
                else:
                    decision['retained'][tid] = 'corrected_text_fit_uncertain'
            held.update(decision['retained'])
            confirmed_graphics = {d['id'] for d in post['decisions'] if d['kind'] == 'graphic'}
            for c in candidates:
                if c['id'] in confirmed_graphics:
                    finding = checks.issue('source_confirmed_graphic_loss', c['bbox'], target=c['id'], confirmed=True)
                    fixes.append(finding); confirmed_regions.append(finding); current['issues'].append(finding)
            write(page_dir/'post-decisions.json', decision)
        additions = graph_regions(original, fixes) + retained_regions(perception, held)
        if not additions:
            break
        pending_regions = initial_regions + additions
        # Carry forward selected retention, rather than reintroducing fixed defects.
        pending_regions += [{'bbox': r['bbox'], 'code': 'carry_selected_retention', 'pad': 0}
                            for r in current['retention']['regions']]
    checks.verify_inventory(raw_dir, immutable)
    if current is None:
        raise RuntimeError('no usable candidate')
    final_probes = loop_probes.run(current['ir'], current['revision']/'proposal', current['built']['pptx'],
                                  current['built']['preview'], page_dir/'selected-probes', backend, full=True)
    final_issues = current['issues'] + [i for i in final_probes['issues'] if i not in current['issues']]
    for item in original.get('quality_guards', {}).get('issues', []):
        final_issues.append(checks.issue(item['code'], detail=item.get('detail')))
    for item in semantic_report['issues']:
        final_issues.append(checks.issue('semantics_review', detail=item))
    calls = [read(p) for p in sorted((page_dir/'visual-review').glob('*/report.json'))]
    for call in calls:
        final_issues.extend(checks.issue(code, phase=call['phase']) for code in call['issues'])
        if call['overflow_ids']:
            final_issues.append(checks.issue('visual_region_overflow', targets=call['overflow_ids']))
    if any(not a['compiled'] for a in attempts):
        final_issues.append(checks.issue('repair_attempt_failed'))
    if baseline:
        old_file = Path(baseline['preview'])
        with Image.open(old_file) as im:
            old = np.asarray(im.convert('RGB'))
        old_error = checks.measured_error(rgb, old, reference_boxes)
        if current['region_error'] > old_error+.05:
            final_issues.append(checks.issue('baseline_visual_regression', baseline_error=old_error,
                                            current_error=current['region_error']))
    final = current['built']; selected = current['revision'].name
    quality = {'schema': 'LayeredLoopQuality/1', 'issues': final_issues, 'quality_passed': not final_issues,
               'selected_revision': selected, 'attempts': attempts, 'policy': asdict(policy),
               'pptx_sha256': digest(final['pptx']), 'preview_sha256': digest(final['preview']),
               'ir_sha256': digest(current['revision']/'proposal'/'slideir.json'),
               'source_sha256': original['source_sha256'], 'region_error': current['region_error'],
               'original_perception_preserved': True, 'human_acceptance': False, 'office_acceptance': False}
    write(page_dir/'quality.json', quality); write(page_dir/'selected-edit-probes.json', final_probes)
    write(page_dir/'effective-perception.json', current['effective_perception'])
    ir = current['ir']
    return {'page': name, 'status': 'FAILED' if any(i['confirmed'] or i['severity'] == 'blocking' for i in final_issues) else 'REVIEW_REQUIRED',
            'quality_policy': 'quality-loop', 'proposal': selected+'/proposal', 'selected_revision': selected,
            'pptx': str(Path(final['pptx']).relative_to(page_dir)), 'preview': str(Path(final['preview']).relative_to(page_dir)),
            'pptx_sha256': quality['pptx_sha256'], 'preview_sha256': quality['preview_sha256'], 'ir_valid': True,
            'uncovered_ocr_ids': [], 'residual_text_violations': [], 'quality': quality,
            'metrics': current['metrics'], 'review': object_review, 'semantics': semantic_report,
            'extra_visual_requests': sum(r['requests'] for r in calls),
            'extra_visual_tokens': sum(r['usage'].get('total_tokens', 0) for r in calls),
            'repair_attempts': max(0, len(attempts)-1),
            'counts': {'native_text_nodes': sum(n['type'] == 'text' for n in ir['nodes']),
                       'native_shapes': sum(n['type'] == 'shape' for n in ir['nodes']),
                       'cutouts': sum(n['type'] == 'image' for n in ir['nodes']), 'groups': len(ir['groups']),
                       'source_retained_regions': len(current['retention']['regions'])}}


def _run(inputs, output, *, worker_config, project_root, remote=None, model=None, vision_model=None,
        job=None, baseline_job=None, max_extra_vision_calls=2, max_repair_rounds=2):
    policy = LoopPolicy(max_extra_vision_calls, max_repair_rounds)
    paths = [Path(p).resolve(strict=True) for p in inputs]
    names = [p.stem for p in paths]
    if len(names) != len(set(names)) or not paths or any(not re.fullmatch(r'[\w\-\u4e00-\u9fff]+', n) for n in names):
        raise ValueError('inputs need unique safe page stems')
    job = job or 'ql-'+uuid4().hex[:10]
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', job):
        raise ValueError('invalid job identifier')
    root = Path(output).resolve()/job; (root/'input').mkdir(parents=True, exist_ok=False)
    for name, p in zip(names, paths):
        with Image.open(p) as image:
            if image.format != 'PNG': raise ValueError('layered inputs must be PNG screenshots')
        shutil.copyfile(p, root/'input'/(name+'.png'))
    baseline = {}
    if baseline_job:
        base = Path(baseline_job).resolve(strict=True); manifest = read(base/'job.json')
        for name, page in manifest['pages'].items():
            if not page.get('preview'): continue
            source_file = base/'input'/(name+'.png'); preview = base/'pages'/name/page['preview']
            pptx = base/'pages'/name/page['pptx']
            if digest(pptx) != page['pptx_sha256']: raise ValueError('baseline PPTX hash changed')
            baseline[digest(source_file)] = {'preview': str(preview), 'pptx_sha256': page['pptx_sha256']}
        if any(digest(p) not in baseline for p in paths): raise ValueError('baseline source set does not cover inputs')
    worker = Worker(worker_config); source_digest, source = worker.sync_source()
    local_code = code_inventory(project_root)
    code = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=project_root, capture_output=True, text=True)
    write(root/'run-binding.json', {'policy': asdict(policy), 'quality_policy': 'quality-loop',
             'git_head': code.stdout.strip(), 'vision_package_sha256': source_digest,
             'local_code_sha256': checks.fingerprint(local_code), 'local_code_files': local_code,
             'configuration_fingerprint': checks.fingerprint({'worker': worker_config, 'model': model, 'vision_model': vision_model}),
             'baseline_job': str(baseline_job) if baseline_job else None,
             'inputs': {n: digest(p) for n, p in zip(names, paths)}})
    worker.upload(job, sorted((root/'input').glob('*.png')))
    worker.run(job, ['scan', '/job/in', '/job/scan'], source=source)
    worker.download(job, 'scan', root)
    reviewers = {}; guard_files = []
    for name in names:
        page_dir = root/'pages'/name; page_dir.mkdir(parents=True)
        scan = read(root/'scan'/name/'scan.json')
        reviewer = CropReviewer(page_dir/'visual-review', remote=remote, model=vision_model, max_requests=policy.max_extra_vision_calls)
        report = reviewer.run(loop_scan.candidates(scan), root/'scan'/name/'source.png', phase='pre')
        guards = corroborate(report, scan, root/'scan'/name/'source.png', page_dir/'pre-independent', worker, job, source,
                             f'independent/{name}/pre')
        guard_file = root/'guards'/(name+'.json'); write(guard_file, guards); guard_files.append(guard_file); reviewers[name] = reviewer
    worker.upload(job, guard_files, subdir='guards')
    log = worker.run(job, ['pages-quality', '/job/in', '/job/perception', '/job/scan', '/job/guards'], source=source)
    worker.download(job, 'perception', root)
    raw_inventory = checks.inventory(root/'perception'); write(root/'perception-binding.json', raw_inventory)
    typography_cache = {}; training = []
    for name in names:
        perception = read(root/'perception'/name/'perception.json')
        with Image.open(root/'perception'/name/'source.png') as im: rgb = np.asarray(im.convert('RGB')).copy()
        styles, calibration = typography.measure(rgb, perception)
        typography_cache[name] = (styles, calibration); training.append((perception['image_size'], styles))
    learned = loop_styles.learn(training); write(root/'deck-styles.json', learned)
    pages = {}
    for name in names:
        print('quality-loop page '+name, flush=True)
        try:
            styles, calibration = typography_cache[name]
            pages[name] = build_page(name, root/'perception'/name, root/'pages'/name, project_root=project_root,
                        worker=worker, job=job, source=source, reviewer=reviewers[name], policy=policy,
                        styles=styles, calibration=calibration, learned=learned, remote=remote, model=model, vision_model=vision_model,
                        baseline=baseline.get(digest(root/'input'/(name+'.png'))))
        except Exception as error:
            pd = root/'pages'/name
            pages[name] = {'page': name, 'status': 'FAILED', 'error': type(error).__name__+': '+str(error),
                           'extra_visual_requests': len(list((pd/'visual-review').glob('*/request.json'))),
                           'repair_attempts': max(0, len(list(pd.glob('r*/attempt.json')))-1)}
        write(root/'page-results'/(name+'.json'), pages[name])
    checks.verify_inventory(root/'perception', raw_inventory)
    if code_inventory(project_root) != local_code:
        raise ValueError('code changed during quality run; saved candidates remain available')
    from .loop_delivery import compose_delivery
    delivery = compose_delivery(root, names, pages, project_root=project_root)
    tokens = sum(p.get('review', {}).get('usage', {}).get('total_tokens', 0)
                 + p.get('semantics', {}).get('usage', {}).get('total_tokens', 0)+p.get('extra_visual_tokens', 0) for p in pages.values())
    summary = {'schema': 'LayeredJob/1', 'job': job, 'quality_policy': 'quality-loop', 'policy': asdict(policy),
               'model': model, 'vision_model': vision_model, 'worker_source_sha256': source_digest,
               'perception_log': log.strip().splitlines()[-20:], 'total_model_tokens': tokens,
               'pages': pages, 'delivery': delivery, 'human_acceptance': False, 'office_acceptance': False}
    summary['status'] = 'FAILED' if delivery['status'] == 'FAILED' or any(p['status'] == 'FAILED' for p in pages.values()) else 'REVIEW_REQUIRED'
    summary['files'] = checks.inventory(root); write(root/'job.json', summary)
    return summary


def run(inputs, output, **options):
    """Keep an explicit failure record even when the worker fails before pages."""
    job = options.get('job') or 'ql-'+uuid4().hex[:10]
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', job):
        raise ValueError('invalid job identifier')
    root = Path(output).resolve()/job
    if root.exists():
        raise FileExistsError('new quality job directory required')
    try:
        return _run(inputs, str(root.parent), **{**options, 'job': job})
    except Exception as error:
        if not root.exists() or (root/'job.json').exists():
            raise
        pages = {}
        for file in inputs:
            name = Path(file).stem; saved = root/'page-results'/(name+'.json'); pd = root/'pages'/name
            pages[name] = read(saved) if saved.exists() else {
                'page': name, 'status': 'FAILED', 'error': 'run_failed_before_page_completion',
                'extra_visual_requests': len(list((pd/'visual-review').glob('*/request.json'))),
                'repair_attempts': max(0, len(list(pd.glob('r*/attempt.json')))-1)}
        summary = {'schema': 'LayeredJob/1', 'job': job, 'quality_policy': 'quality-loop', 'status': 'FAILED',
                   'error': type(error).__name__+': '+str(error)[:2000], 'pages': pages,
                   'delivery': {'status': 'FAILED'}, 'total_model_tokens': 0,
                   'human_acceptance': False, 'office_acceptance': False, 'files': checks.inventory(root)}
        for report in root.glob('pages/*/visual-review/*/report.json'):
            summary['total_model_tokens'] += read(report).get('usage', {}).get('total_tokens', 0)
        write(root/'job.json', summary)
        return summary
