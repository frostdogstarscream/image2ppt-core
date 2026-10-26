"""Opt-in, resumable server-model/local-compiler automatic validation jobs.

No manual candidate approvals are generated. Images are inputs; model plans,
immutable builds and machine reviews are evidence, never human acceptance.
"""
from copy import deepcopy
from pathlib import Path
import base64
import hashlib
import json
import re
import shutil
import time
from uuid import uuid4

from PIL import Image, ImageOps

from image2ppt.storage import digest, read_json, write_json, run_lock
from .auto_artifacts import build_revision, verify_hashes, file_hashes
from .auto_provider import (save_new, fingerprint, model_body, parse_object, preflight)
from .auto_protocol import (PLANNER, COMPACT_PLANNER, REVIEWER, PATCHER, validate_plan, validate_review, scoped_patch,
                            transform_model_boxes, expand_model_draft)
from .auto_typography import measure_text_defaults, expand_draft, compact_evidence_texts, draft_feedback
from .auto_artifacts import DeadlineBackend
from .auto_transport import SSHRemote
from .element_inventory import build_inventory
from .typography import font_status

POLICY = {'schema': 'AutomaticPolicy/1', 'decision_source': 'user_requested_automatic_trial',
          'max_corrections': 2, 'max_model_requests_per_page': 8, 'page_timeout_seconds': 1800,
          'required_edit_targets': 'all_detected_text', 'preserve_source_layout': True,
          'human_acceptance': False, 'office_acceptance': False}


class BudgetExhausted(RuntimeError):
    pass


def safe_error(error):
    # Validation messages describe structure, not source text or HTTP secrets.
    if isinstance(error, (ValueError, BudgetExhausted)):
        return str(error)[:500]
    return type(error).__name__


def create_job(inputs, output, config, *, gate_pages=None, preference='faithful_editable',
               pipeline_version=1, policy=None, gate_only=False):
    if preference not in ('faithful_editable', 'fidelity_first'):
        raise ValueError('unsupported restoration preference')
    inputs = [Path(p).resolve(strict=True) for p in inputs]
    if not 1 <= len(inputs) <= 200:
        raise ValueError('expected 1 to 200 images')
    gate_pages = list(gate_pages or range(1, min(3, len(inputs))+1))
    if len(gate_pages) != len(set(gate_pages)) or any(type(p) is not int or not 1 <= p <= len(inputs) for p in gate_pages):
        raise ValueError('invalid gate page numbers')
    root = Path(output).resolve()/('auto-'+uuid4().hex[:12])
    root.mkdir(parents=True, exist_ok=False)
    pages = []
    for i, source in enumerate(inputs, 1):
        page = root/f'P{i}'; page.mkdir()
        if source.stat().st_size > 20_000_000:
            raise ValueError('input image exceeds 20 MB')
        with Image.open(source) as im:
            if im.format not in ('PNG', 'JPEG', 'WEBP') or getattr(im, 'n_frames', 1) != 1 or im.width*im.height > 40_000_000:
                raise ValueError('unsupported image input')
            normalized = ImageOps.exif_transpose(im).convert('RGB')
        shutil.copyfile(source, page/'original.image')
        normalized.save(page/'source.png')
        pages.append({'page': i, 'id': f'P{i}', 'source_name': source.name,
                      'original_sha256': digest(page/'original.image'),
                      'source_sha256': digest(page/'source.png'), 'image_size': list(normalized.size)})
    manifest = {'schema': f'AutomaticJob/{pipeline_version}', 'job_id': root.name, 'created_at': time.time(),
                'pages': pages, 'gate_pages': gate_pages, 'preference': preference,
                'policy': deepcopy(policy or POLICY), 'config_sha256': fingerprint(config)}
    if pipeline_version != 1:
        manifest.update(pipeline_version=pipeline_version, gate_only=bool(gate_only))
    save_new(root/'job.json', manifest)
    save_new(root/'connection.json', config)
    write_json(root/'state.json', {'status': 'QUEUED', 'stage': 'preflight', 'pages': {},
                                  'job_sha256': digest(root/'job.json'), 'human_acceptance': False})
    return root


def verified_preflight(directory, config):
    result = read_json(Path(directory)/'result.json')
    if result.get('schema') != 'AutomaticPreflight/1' or result.get('config_sha256') != fingerprint(config):
        raise ValueError('preflight configuration mismatch')
    verify_hashes(directory, result.get('files', {}))
    if not result.get('passed') or result.get('selected_model') not in config['models']:
        raise ValueError('model capability preflight failed')
    return result


class PageSession:
    def __init__(self, directory, remote, model, *, clock=time.time, policy=None):
        self.root = Path(directory)
        self.remote = remote
        self.model = model
        self.clock = clock
        self.policy = deepcopy(policy or POLICY)
        if not (self.root/'budget.json').exists():
            save_new(self.root/'budget.json', {'started_at': clock(), 'timeout_seconds': self.policy['page_timeout_seconds'],
                                             'max_calls': self.policy['max_model_requests_per_page'],
                                             'initial_call_limit': self.policy.get('initial_call_limit')})
        self.budget = read_json(self.root/'budget.json')
        if policy and (self.budget['timeout_seconds'] != policy['page_timeout_seconds'] or
                       self.budget['max_calls'] != policy['max_model_requests_per_page'] or
                       self.budget.get('initial_call_limit') != policy.get('initial_call_limit')):
            raise ValueError('saved page budget differs from immutable job policy')

    @property
    def deadline(self):
        return self.budget['started_at']+self.budget['timeout_seconds']

    def remaining(self):
        value = self.deadline-self.clock()
        if value <= 0:
            raise BudgetExhausted('page_time_budget_exhausted')
        return value

    def call(self, phase, instruction, evidence, images, max_tokens=24000):
        body = model_body(self.model, instruction, evidence, images, max_tokens)
        logical_key = fingerprint({'phase': phase, 'body': body})
        calls = self.root/'calls'; calls.mkdir(exist_ok=True)
        directories = sorted(calls.glob('c[0-9][0-9]'))
        for directory in directories:
            receipt = read_json(directory/'request.json')
            if receipt['logical_key'] != logical_key:
                continue
            request = receipt['request']
            if (directory/'response.json').exists():
                bound = read_json(directory/'receipt.json')
                if bound['request_sha256'] != digest(directory/'request.json') or bound['response_sha256'] != digest(directory/'response.json'):
                    raise ValueError('model call evidence changed')
                return self._response(directory, request)
            # Retry retrieval of this same remote request ID, never create a
            # duplicate inference after a local interruption.
            response = self.remote.request(request, timeout=min(550, self.remaining()))
            self._complete(directory, response)
            return self._response(directory, request)
        remaining = self.remaining()
        if len(directories) >= self.budget['max_calls']:
            raise BudgetExhausted('page_model_request_budget_exhausted')
        initial_limit = self.budget.get('initial_call_limit')
        if phase.startswith('initial/') and initial_limit is not None:
            initial_calls = sum(read_json(p/'request.json')['phase'].startswith('initial/') for p in directories)
            if initial_calls >= initial_limit:
                raise BudgetExhausted('initial_request_budget_exhausted_review_budget_reserved')
        directory = calls/f'c{len(directories):02d}'; directory.mkdir()
        request = {'request_id': uuid4().hex, 'operation': 'model',
                   'timeout': max(1, min(480, int(remaining)-60)), 'body': body}
        save_new(directory/'request.json', {'phase': phase, 'logical_key': logical_key,
                                          'reserved_at': self.clock(), 'request': request})
        write_json(self.root/'progress.json', {'stage': phase, 'model_requests': len(directories)+1,
                                              'elapsed_seconds': round(self.clock()-self.budget['started_at'], 3)})
        response = self.remote.request(request, timeout=min(550, remaining))
        self._complete(directory, response)
        return self._response(directory, request)

    def _response(self, directory, request):
        path = directory/'response.json'
        response = read_json(path)
        if response.get('error') in ('ssh_failed', 'ssh_timeout'):
            # One retrieval of the SAME reserved ID. The worker returns its
            # cached result or incomplete status; it cannot repeat that inference.
            recovered = directory/'recovered-response.json'
            receipt = directory/'recovered-receipt.json'
            if not recovered.exists():
                retry = self.remote.request(request, timeout=min(550, self.remaining()))
                save_new(recovered, retry)
                save_new(receipt, {'original_response_sha256': digest(path),
                                  'request_sha256': digest(directory/'request.json'),
                                  'response_sha256': digest(recovered)})
            bound = read_json(receipt)
            if (bound['original_response_sha256'] != digest(path) or
                bound['request_sha256'] != digest(directory/'request.json') or
                bound['response_sha256'] != digest(recovered)):
                raise ValueError('recovered model call evidence changed')
            path = recovered; response = read_json(path)
        self.last_response_sha256 = digest(path)
        return response

    @staticmethod
    def _complete(directory, response):
        save_new(directory/'response.json', response)
        save_new(directory/'receipt.json', {'request_sha256': digest(directory/'request.json'),
                                          'response_sha256': digest(directory/'response.json')})

    def summary(self):
        calls = []
        usage = {}
        for directory in sorted((self.root/'calls').glob('c[0-9][0-9]')):
            request = read_json(directory/'request.json')
            response_path = directory/('recovered-response.json' if (directory/'recovered-response.json').exists() else 'response.json')
            response = read_json(response_path) if response_path.exists() else {}
            calls.append({'phase': request['phase'], 'request_id': request['request']['request_id'],
                          'ok': response.get('ok', False), 'error': response.get('error'),
                          'elapsed_seconds': response.get('elapsed_seconds', response.get('total_seconds'))})
            for key, value in response.get('usage', {}).items():
                if type(value) in (int, float):
                    usage[key] = usage.get(key, 0)+value
        return {'model_requests': len(calls), 'usage': usage, 'calls': calls,
                'elapsed_seconds': round(self.clock()-self.budget['started_at'], 3)}


TRANSIENT = re.compile(r'model_http_(429|5\d\d)|model_timeout|model_transport_failed|model_worker_timeout')
BACKOFF_SECONDS = (15, 45, 120)


def validator_error(exc):
    """Structural, model-actionable message; never source text or secrets."""
    if isinstance(exc, KeyError):
        return 'missing required field '+repr(exc.args[0])[:80] if exc.args else 'missing required field'
    if isinstance(exc, (TypeError, AttributeError)):
        return 'wrong JSON value type ('+str(exc)[:160]+'); check field types against the schema'
    return safe_error(exc)


def call_with_backoff(session, phase, instruction, evidence, images, max_tokens, sleep=time.sleep):
    """Retry provider throttling/outages with backoff inside the page deadline.

    Each retry is a new, honestly counted request under a distinct phase label;
    it does not use up the caller's validation attempts.
    """
    response = session.call(phase, instruction, evidence, images, max_tokens)
    for retry, delay in enumerate(BACKOFF_SECONDS, 1):
        if response.get('ok') or not TRANSIENT.fullmatch(str(response.get('error', ''))):
            break
        if session.remaining() <= delay+60:
            break
        sleep(delay)
        response = session.call(f'{phase}-retry{retry}', instruction, evidence, images, max_tokens)
    return response


def checked_model_json(session, phase, instruction, evidence, images, validator, max_tokens=24000, feedback_builder=None,
                       max_attempts=3, sleep=time.sleep):
    feedback = None
    for attempt in range(max_attempts):
        current = deepcopy(evidence)
        if feedback:
            current['validation_feedback'] = feedback
        response = call_with_backoff(session, f'{phase}-{attempt}', instruction, current, images, max_tokens, sleep)
        if not response.get('ok'):
            code = response.get('error', 'model_call_failed')
            # Invalid credentials/permission are not transient; avoid repeated cost.
            if code in ('model_http_400', 'model_http_401', 'model_http_403'):
                raise ValueError(code)
            feedback = {**(feedback or {}), 'transport_or_provider_error': code}
            continue
        try:
            value = parse_object(response['content'])
            return validator(value)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            previous_error = (feedback or {}).get('error')
            feedback = {'error': validator_error(exc), 'previous_output': response['content']}
            if previous_error == feedback['error']:
                # The same rejection twice means feedback is not helping; stop spending budget.
                raise ValueError('model_output_validation_exhausted: '+feedback['error'])
            if feedback_builder:
                try:
                    feedback['source_constraint_diagnostics'] = feedback_builder(parse_object(response['content']))
                except (ValueError, KeyError, TypeError, AttributeError):
                    pass
    reason = (feedback or {}).get('error') or (feedback or {}).get('transport_or_provider_error', 'unknown')
    raise ValueError('model_output_validation_exhausted: '+str(reason))


def get_analysis(session, source):
    path = session.root/'analysis.json'
    if path.exists():
        receipt = read_json(session.root/'analysis-binding.json')
        if digest(path) != receipt['sha256']:
            raise ValueError('analysis evidence changed')
        return read_json(path)
    request_path = session.root/'analysis-request.json'
    if request_path.exists():
        request = read_json(request_path)
    else:
        request = {'request_id': uuid4().hex, 'operation': 'analyze', 'source_sha256': digest(source),
                   'image_base64': base64.b64encode(Path(source).read_bytes()).decode('ascii')}
        save_new(request_path, request)
    write_json(session.root/'progress.json', {'stage': 'server_analysis', 'model_requests': 0})
    response = session.remote.request(request, timeout=min(420, session.remaining()))
    if not response.get('ok'):
        raise ValueError(response.get('error', 'analysis_failed'))
    analysis = response['analysis']
    if not any(s.get('name') == 'ocr' and s.get('status') == 'PASS' for s in analysis.get('stages', [])):
        raise ValueError('server OCR did not pass')
    save_new(path, analysis)
    save_new(session.root/'analysis-binding.json', {'sha256': digest(path), 'job_id': response['job_id'],
                                                   'server_result_sha256': response.get('server_result_sha256')})
    return analysis


def get_inventory(session, source, analysis):
    binding_path = session.root/'inventory-binding.json'
    if binding_path.exists():
        binding = read_json(binding_path)
        directory = session.root/binding['directory']
        verify_hashes(directory, binding['files'])
        return read_json(directory/'inventory.json')
    session.remaining()
    directory = session.root/('inventory-'+uuid4().hex[:6])
    result = build_inventory(source, analysis, directory)
    save_new(binding_path, {'directory': directory.name, 'files': file_hashes(directory)})
    return result


def review_evidence(plan, built, policy):
    return transform_model_boxes({'artifact_binding': built['artifact_binding'], 'source_sha256': plan['source_sha256'],
            'coordinate_space': 'normalized_1000_xyxy',
            'image_size': plan['image_size'], 'objects': plan['objects'],
            'text_decisions': plan['text_decisions'], 'probe_targets': built['probe_targets'],
            'structural_issues': built['structural_issues'], 'restoration_policy': policy}, plan['image_size'], to_model=True)


def get_text_defaults(session, source, inventory, project_root):
    path = session.root/'text-defaults.json'
    binding_path = session.root/'text-defaults-binding.json'
    if binding_path.exists():
        if digest(path) != read_json(binding_path)['sha256']:
            raise ValueError('source typography evidence changed')
        return read_json(path)
    environment = DeadlineBackend(project_root, session.deadline).doctor()
    def progress(done, total):
        write_json(session.root/'progress.json', {'stage': 'source_typography', 'completed': done,
                   'total': total, **session.summary()})
    defaults = measure_text_defaults(source, inventory, renderer_version=environment['artifact_version'],
                                     deadline=session.deadline, progress=progress)
    save_new(path, defaults)
    save_new(binding_path, {'sha256': digest(path), 'environment': environment})
    return defaults


def finish_page(directory, session, candidates, errors):
    reviewed = [c for c in candidates if c.get('review')]
    pool = reviewed or candidates
    def combined_issues(candidate):
        return list({i['id']: i for i in candidate.get('review', {}).get('issues', [])+
                     candidate['built']['structural_issues']}.values())
    def rank(c):
        review = c.get('review', {})
        issues = combined_issues(c)
        return (not review.get('quality_passed', False), sum(i['severity'] == 'blocking' for i in issues),
                len(issues), c['built']['metrics']['mean_absolute_pixel_difference'], -c['index'])
    selected = min(pool, key=rank) if pool else None
    passed = bool(selected and selected.get('review', {}).get('quality_passed') and
                  selected['review']['edit_probe_clear'] and not selected['built']['structural_issues'])
    reviewed_selected = bool(selected and selected.get('review'))
    result = {'schema': 'AutomaticPageResult/1',
              'execution_status': 'COMPLETED' if reviewed_selected else 'PARTIAL' if selected else 'FAILED',
              'pipeline_completed': reviewed_selected,
              'review_status': 'VALIDATED' if reviewed_selected else 'INCOMPLETE',
              'machine_quality_status': 'PASSED' if passed else 'REVIEW_REQUIRED' if selected else 'FAILED',
              'selected_revision': selected['revision'] if selected else None,
              'selection_reason': 'machine review and structural issues first; pixel metric only breaks ties',
              'errors': errors, 'model': session.model, **session.summary(),
              'human_acceptance': False, 'office_acceptance': False}
    if selected:
        result.update(build_directory=selected['build_directory'], artifact_binding=selected['built']['artifact_binding'],
                      metrics=selected['built']['metrics'], counts=selected['built']['counts'],
                      issues=combined_issues(selected),
                      revision_count=len(candidates))
    save_new(directory/'result.json', result)
    return result


def run_page(directory, page, manifest, config, model, *, project_root, remote=None, builder=build_revision):
    directory = Path(directory)
    source = directory/'source.png'
    if digest(source) != page['source_sha256'] or digest(directory/'original.image') != page['original_sha256']:
        raise ValueError('input source changed')
    if (directory/'result.json').exists():
        result = read_json(directory/'result.json')
        if result.get('build_directory'):
            built = read_json(directory/result['build_directory']/'result.json')
            verify_hashes(directory/result['build_directory'], built['files'])
            if built['artifact_binding'] != result['artifact_binding']:
                raise ValueError('selected build binding changed')
        return result
    session = PageSession(directory, remote or SSHRemote(config), model)
    errors = []; candidates = []
    try:
        analysis = get_analysis(session, source)
        inventory = get_inventory(session, source, analysis)
        policy = {**manifest['policy'], 'preference': manifest['preference']}
        binding = fingerprint({'policy': policy, 'source': page['source_sha256'], 'config': manifest['config_sha256']})
        previous = None
        for index in range(1+POLICY['max_corrections']):
            revision = directory/f'r{index:03d}'; revision.mkdir(exist_ok=True)
            plan_path = revision/'plan.json'
            if plan_path.exists():
                plan = validate_plan(read_json(plan_path), inventory)
                plan_binding = read_json(revision/'plan-binding.json')
                if plan_binding['plan_sha256'] != digest(plan_path):
                    raise ValueError('saved plan changed')
                if index and previous:
                    # Re-validate correction provenance when resuming.
                    patched = scoped_patch(previous['plan'], read_json(revision/'patch.json'), previous['review'],
                                           inventory, previous['built']['artifact_binding'])
                    if plan != patched:
                        raise ValueError('saved correction plan changed')
            elif index == 0:
                defaults = get_text_defaults(session, source, inventory, project_root)
                evidence = {'source_sha256': page['source_sha256'], 'image_size': inventory['image_size'],
                            'ocr': [{'id': t['id']} for t in inventory['texts']],
                            'available_fonts': font_status()['families'],
                            'text_defaults': compact_evidence_texts(defaults, inventory['image_size']),
                            'restoration_policy': policy}
                plan = checked_model_json(session, 'plan', COMPACT_PLANNER, evidence, [source],
                                          lambda v: expand_draft(v, defaults, inventory), 24000,
                                          feedback_builder=lambda v: draft_feedback(v, inventory))
                save_new(plan_path, plan)
                plan_binding = {'plan_sha256': digest(plan_path), 'response_sha256': session.last_response_sha256}
                save_new(revision/'plan-binding.json', plan_binding)
            else:
                review = previous['review']
                # Structural findings are bound and included in the actual model
                # review. Do not invent human decisions to bypass a failed gate.
                evidence = transform_model_boxes({'plan': previous['plan'], 'review': review,
                            'coordinate_space': 'normalized_1000_xyxy',
                            'plan_schema': COMPACT_PLANNER, 'available_fonts': font_status()['families']},
                            plan['image_size'], to_model=True)
                accepted = {}
                def validate_patch(patch):
                    patch = expand_model_draft(patch, 'ElementPatchDraft/1', 'ElementPatch/1', previous['plan']['image_size'])
                    result = scoped_patch(previous['plan'], patch, review, inventory, previous['built']['artifact_binding'])
                    accepted.update(patch=patch)
                    return result
                images = [source, directory/previous['build_directory']/'evidence/preview.png']
                plan = checked_model_json(session, f'patch-{index}', PATCHER, evidence, images, validate_patch)
                save_new(revision/'patch.json', accepted['patch'])
                save_new(plan_path, plan)
                plan_binding = {'plan_sha256': digest(plan_path), 'response_sha256': session.last_response_sha256}
                save_new(revision/'plan-binding.json', plan_binding)
            build_pointer = revision/'build-pointer.json'
            if build_pointer.exists():
                pointer = read_json(build_pointer)
                build_dir = revision/pointer['directory']
                built = read_json(build_dir/'result.json')
                verify_hashes(build_dir, built['files'])
                if digest(build_dir/'result.json') != pointer['sha256'] or built['plan_sha256'] != fingerprint(plan):
                    raise ValueError('build pointer changed')
            else:
                session.remaining()
                write_json(directory/'progress.json', {'stage': 'compile_and_edit_probe', 'revision': revision.name,
                                                       **session.summary()})
                build_dir = revision/('b-'+uuid4().hex[:6])
                built = builder(source, inventory, plan, build_dir, project_root=project_root,
                                deadline=session.deadline, model_id=model, policy_binding=binding,
                                response_sha256=plan_binding['response_sha256'])
                save_new(build_pointer, {'directory': build_dir.name, 'sha256': digest(build_dir/'result.json')})
            candidate = {'index': index, 'revision': revision.name, 'plan': plan, 'built': built,
                         'build_directory': build_dir.relative_to(directory).as_posix()}
            candidates.append(candidate)
            review_path = revision/'review.json'
            if review_path.exists():
                review = validate_review(read_json(review_path), built['artifact_binding'], plan)
            else:
                evidence = review_evidence(plan, built, policy)
                packet = build_dir/'evidence'
                review = checked_model_json(session, f'review-{index}', REVIEWER, evidence,
                    [packet/'source.png', packet/'preview.png', packet/'delete-probe.png'],
                    lambda v: validate_review(expand_model_draft(v, 'AutomaticReviewDraft/1', 'AutomaticReview/1',
                                              plan['image_size']), built['artifact_binding'], plan), 8000)
                # The machine may not discard deterministic exported-file failures.
                if built['structural_issues']:
                    review['quality_passed'] = False
                    current = {i['id'] for i in review['issues']}
                    review['issues'].extend(i for i in built['structural_issues'] if i['id'] not in current)
                validate_review(review, built['artifact_binding'], plan)
                save_new(review_path, review)
            candidate['review'] = review
            previous = candidate
            write_json(directory/'progress.json', {'stage': 'reviewed', 'revision': revision.name,
                       'machine_quality_status': 'PASSED' if review['quality_passed'] else 'REVIEW_REQUIRED',
                       **session.summary()})
            if review['quality_passed'] and not built['structural_issues']:
                break
    except Exception as exc:
        errors.append({'code': safe_error(exc), 'type': type(exc).__name__})
    return finish_page(directory, session, candidates, errors)


def deliver(root, manifest, results):
    delivery = root/'delivery'
    delivery.mkdir(exist_ok=True)
    rows = []
    for page in manifest['pages']:
        result = results.get(page['id'])
        if not result:
            continue
        rows.append({'page': page['id'], 'source_name': page['source_name'], **result})
        if result.get('build_directory'):
            packet = root/page['id']/result['build_directory']/'evidence'
            for source_name, suffix in [('result.pptx', '.pptx'), ('preview.png', '-preview.png'),
                    ('comparison.png', '-comparison.png'), ('difference.png', '-difference.png'),
                    ('delete-probe.pptx', '-delete-probe.pptx'), ('delete-probe.png', '-delete-probe.png')]:
                target = delivery/(page['id']+suffix)
                if target.exists():
                    if digest(target) != digest(packet/source_name):
                        raise ValueError('delivery already exists with different content')
                else:
                    shutil.copyfile(packet/source_name, target)
    report = {'schema': 'AutomaticDelivery/1', 'pages': rows, 'requested_pages': len(manifest['pages']),
              'executed_pages': len(rows), 'human_acceptance': False, 'office_acceptance': False}
    write_json(delivery/'report.json', report)
    lines = ['# 自动链路效果验证', '', '运行方式：服务器模型与分析 + 本机固定编译与实际文件回渲染。',
             '人工视觉验收与 PowerPoint/WPS 实机验收均未完成。', '',
             '| 页面 | 执行 | 机器质量 | 调用数 | 原生文字 / 形状 / 图片 |', '|---|---|---|---:|---|']
    for row in rows:
        counts = row.get('counts', {})
        lines.append(f"| {row['page']} | {row['execution_status']} | {row['machine_quality_status']} | {row['model_requests']} | "
                     f"{counts.get('native_text',0)} / {counts.get('native_shapes',0)} / {counts.get('raster_assets',0)} |")
    for row in rows:
        lines += ['', '## '+row['page']]
        if row.get('build_directory'):
            pid = row['page']
            lines += ['', f'[{pid}.pptx]({pid}.pptx) · [预览]({pid}-preview.png) · [原图对照]({pid}-comparison.png) · [删除探针]({pid}-delete-probe.png)']
        lines += ['']+[f"- {issue['kind']}: {issue['description']}" for issue in row.get('issues', [])]
        lines += [f"- 执行问题：{error['code']}" for error in row.get('errors', [])]
    (delivery/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def run_job(root, *, project_root, preflight_directory=None, remote=None, builder=build_revision, retry_failed=False):
    root = Path(root).resolve(strict=True)
    with run_lock(root):
        manifest = read_json(root/'job.json'); config = read_json(root/'connection.json')
        state = read_json(root/'state.json')
        if state['job_sha256'] != digest(root/'job.json') or manifest['config_sha256'] != fingerprint(config):
            raise ValueError('job configuration changed')
        remote = remote or SSHRemote(config)
        state.update(status='RUNNING', stage='preflight'); write_json(root/'state.json', state)
        try:
            local_preflight = root/'preflight'
            if not local_preflight.exists():
                if preflight_directory:
                    verified_preflight(preflight_directory, config)
                    shutil.copytree(preflight_directory, local_preflight)
                else:
                    preflight(remote, config, local_preflight)
            selected = verified_preflight(local_preflight, config)['selected_model']
            order = manifest['gate_pages']+[p['page'] for p in manifest['pages'] if p['page'] not in manifest['gate_pages']]
            results = {}
            for index, number in enumerate(order):
                if index == len(manifest['gate_pages']) and any(
                        results[f'P{i}']['machine_quality_status'] != 'PASSED' for i in manifest['gate_pages']):
                    state.update(status='GATE_REVIEW_REQUIRED', stage='finished',
                                 skipped_pages=[f'P{n}' for n in order[index:]],
                                 reason='Representative pages have unresolved blocking findings; batch expansion stopped.')
                    break
                page = manifest['pages'][number-1]
                result_path = root/page['id']/'result.json'
                if retry_failed and result_path.exists():
                    old = read_json(result_path)
                    budget = read_json(root/page['id']/'budget.json')
                    if (old['execution_status'] == 'FAILED' and old['model_requests'] < budget['max_calls'] and
                        time.time() < budget['started_at']+budget['timeout_seconds']):
                        history = root/page['id']/'previous-results'; history.mkdir(exist_ok=True)
                        saved = history/('failed-'+uuid4().hex[:8]+'.json')
                        shutil.copyfile(result_path, saved)
                        if digest(saved) != digest(result_path):
                            raise ValueError('failed-page history copy mismatch')
                        result_path.unlink()
                state.update(stage='page', current_page=page['id']); write_json(root/'state.json', state)
                result = run_page(root/page['id'], page, manifest, config, selected,
                                  project_root=project_root, remote=remote, builder=builder)
                results[page['id']] = result
                state['pages'][page['id']] = result
                write_json(root/'state.json', state)
                deliver(root, manifest, results)
            else:
                status = 'COMPLETED' if all(r['execution_status'] == 'COMPLETED' for r in results.values()) else (
                    'PARTIAL' if any(r.get('build_directory') for r in results.values()) else 'FAILED')
                state.update(status=status, stage='finished')
            report = deliver(root, manifest, results)
            state.update(delivery='delivery/README.md', selected_model=selected,
                         machine_quality_status='PASSED' if len(results) == len(manifest['pages']) and
                         all(r['machine_quality_status'] == 'PASSED' for r in results.values()) else 'REVIEW_REQUIRED')
        except Exception as exc:
            state.update(status='FAILED', stage='finished', error=safe_error(exc))
            write_json(root/'state.json', state)
            raise
        write_json(root/'state.json', state)
        return {'job': str(root), **state}
