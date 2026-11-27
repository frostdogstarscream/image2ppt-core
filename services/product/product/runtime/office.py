"""Actual Office verification, usable by the local or remote Windows worker."""
import json
import math
from pathlib import Path
import shutil
import time

from ..materialize import digest, write
from ..resource_lease import lease


_TRANSIENT_COM_MARKERS = ('com_error', '-2146959355', '-2147417848', '-2147023174', '-2147418111')
_TRANSIENT_COM_HRESULTS = (-2146959355, -2147417848, -2147023174, -2147418111)


def _is_transient_com_error(exc_or_reason) -> bool:
    if isinstance(exc_or_reason, BaseException):
        if type(exc_or_reason).__name__ == 'com_error':
            return True
        if getattr(exc_or_reason, 'hresult', None) in _TRANSIENT_COM_HRESULTS:
            return True
        text = str(exc_or_reason).lower()
    else:
        text = str(exc_or_reason or '').lower()
    return any(marker in text for marker in _TRANSIENT_COM_MARKERS)


def frame_feedback(result,graph,identifier):
    """Expose measured Office overflow separately from the source erase scope."""
    rows=[r for r in result.get('replacements',{}).get('records',[]) if r['id']==identifier and r.get('status') != 'PASS']
    if not rows or not any(r['layout']['overflow'] for r in rows):
        return None
    if any(not r['content_matches'] or not r['siblings_unchanged'] or not r['frame_unchanged']
           or r['layout'].get('readability') or r['layout']['unexpected_wrap'] for r in rows):
        return None
    text=next(t for t in graph.text.texts if t.id==identifier)
    sizes=[]
    for row in rows:
        left,top,width,height=row['layout']['frame']
        bl,bt,bw,bh=row['layout']['text_bounds']
        if width<=0 or height<=0 or bl<left-1 or bt<top-1:
            return None
        sizes.append([math.ceil((bl+bw-left)*text.bbox[2]/width)+1,
                      math.ceil((bt+bh-top)*text.bbox[3]/height)+1])
    return {'failure':'FONT_LAYOUT_FAILED','status':'UNVERIFIED_LAYOUT_PROPOSAL',
            'previous_edit_outside_pixels':[r['outside_pixels'] for r in rows],
            'measured_frame_minimum_px':[max(s[0] for s in sizes),max(s[1] for s in sizes)],
            'meaning':'Measured transparent native-frame extent for create_candidate.frame_minimum. Source text bbox and authorized erase pixels remain independent.'}


def check_local(pptx, irs, directory, *, target_ids=None, original_sites=False, text_layer_audit=False):
    from ..parallel_office import inspect, OfficeValidationError
    from ..parallel_edit_probe import inspect_text
    from ..parallel_residual_gate import inspect_residuals, residual_accepted
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    targets = set(target_ids) if target_ids is not None else None
    probe_options = {'target_ids': targets} if targets is not None else {}
    if text_layer_audit:
        probe_options['text_layer_audit'] = True
    if original_sites:
        probe_options['original_sites'] = True
    result = {'status': 'UNKNOWN', 'pptx_sha256': digest(pptx), 'office_directory': str(directory/'office'),
              'target_ids': sorted(targets) if targets is not None else None}
    try:
        actual = None
        replacement = {'status': 'UNKNOWN', 'reason': 'not executed'}
        with lease('office'):
            has_text = any(n['type'] == 'text' and (targets is None or n['id'] in targets) for ir in irs
                           for n in json.loads(Path(ir).read_text('utf8'))['nodes'])
            for attempt in range(3):
                if attempt > 0:
                    shutil.rmtree(directory/'office', ignore_errors=True)
                    shutil.rmtree(directory/'replacement', ignore_errors=True)
                    time.sleep(2.0 * attempt)
                try:
                    actual = inspect(pptx, irs, directory/'office', **probe_options)
                except OfficeValidationError as exc:
                    report_file = directory/'office/report.json'
                    actual = json.loads(report_file.read_text('utf8')) if report_file.is_file() else {
                        'status': 'FAIL_APP_AUTOMATED', 'failed_object_id': exc.object_id,
                        'reason': str(exc), 'error_type': 'OfficeValidationError',
                    }
                    replacement = {'status': 'NOT_APPLICABLE', 'reason': 'office validation failed'}
                    break
                except Exception as exc:
                    if attempt < 2 and _is_transient_com_error(exc):
                        continue
                    raise
                replacement = inspect_text(pptx, irs, directory/'replacement', include_long=False,
                                           **({'readable_text': True} if original_sites else {}),
                                           **({'target_ids': targets} if targets is not None else {})) if has_text else {
                    'status': 'NOT_APPLICABLE', 'reason': 'no native text objects'}
                if replacement.get('status') == 'UNKNOWN' and attempt < 2 and _is_transient_com_error(replacement.get('reason', '')):
                    continue
                break
        residuals = [inspect_residuals(ir, directory/'office', index, directory/f'residual-{index:02d}.json',
                                       **({'target_ids': targets} if targets is not None else {}))
                     for index, ir in enumerate(irs, 1)]
        if original_sites and actual.get('status') == 'PASS_APP_AUTOMATED':
            from ..office_sites import checked_sites
            for page, ir in enumerate(irs, 1):
                checked_sites(directory/'office', actual, ir, page, targets)
        if replacement.get('status') == 'UNKNOWN':
            result.update(error_type='RuntimeError', failure='QA_INFRASTRUCTURE_UNAVAILABLE',
                          office=actual, replacements=replacement, residuals=residuals)
        else:
            passed = (actual['status'] == 'PASS_APP_AUTOMATED'
                      and replacement['status'] in ('PASS', 'NOT_APPLICABLE')
                      and all(residual_accepted(r) for page in residuals for r in page))
            result.update(status='PASS' if passed else 'REVIEW_REQUIRED', office=actual,
                          replacements=replacement, residuals=residuals)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, reason=str(exc)[:300], failure='QA_INFRASTRUCTURE_UNAVAILABLE')
    write(directory/'checks.json', result)
    result['files'] = {str(p): digest(p) for p in directory.rglob('*') if p.is_file()
                       and not p.name.startswith('~$') and not ('-batch-' in p.name and p.suffix=='.pptx')}
    return result
