"""Actual Office verification, usable by the local or remote Windows worker."""
import json
import math
from pathlib import Path

from ..materialize import digest, write
from ..resource_lease import lease


def frame_feedback(result,graph,identifier):
    """Expose measured Office overflow separately from the source erase scope."""
    rows=[r for r in result.get('replacements',{}).get('records',[]) if r['id']==identifier]
    if not rows or not any(r['layout']['overflow'] for r in rows):
        return None
    if any(not r['content_matches'] or not r['siblings_unchanged'] or not r['frame_unchanged']
           or r['outside_pixels'] or r['layout']['unexpected_wrap'] for r in rows):
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
    return {'failure':'FONT_LAYOUT_FAILED',
            'measured_frame_minimum_px':[max(s[0] for s in sizes),max(s[1] for s in sizes)],
            'meaning':'Measured transparent native-frame extent for create_candidate.frame_minimum. Source text bbox and authorized erase pixels remain independent.'}


def check_local(pptx, irs, directory):
    from ..parallel_office import inspect
    from ..parallel_edit_probe import inspect_text
    from ..parallel_residual_gate import inspect_residuals
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {'status': 'UNKNOWN', 'pptx_sha256': digest(pptx), 'office_directory': str(directory/'office')}
    try:
        with lease('office'):
            actual = inspect(pptx, irs, directory/'office')
            has_text = any(n['type'] == 'text' for ir in irs
                           for n in json.loads(Path(ir).read_text('utf8'))['nodes'])
            replacement = inspect_text(pptx, irs, directory/'replacement', include_long=False) if has_text else {
                'status': 'NOT_APPLICABLE', 'reason': 'no native text objects'}
        residuals = [inspect_residuals(ir, directory/'office', index, directory/f'residual-{index:02d}.json')
                     for index, ir in enumerate(irs, 1)]
        passed = (actual['status'] == 'PASS_APP_AUTOMATED'
                  and replacement['status'] in ('PASS', 'NOT_APPLICABLE')
                  and all(r['residual_check_status'] == 'PASS' for page in residuals for r in page))
        result.update(status='PASS' if passed else 'REVIEW_REQUIRED', office=actual,
                      replacements=replacement, residuals=residuals)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, failure='QA_INFRASTRUCTURE_UNAVAILABLE')
    write(directory/'checks.json', result)
    result['files'] = {str(p): digest(p) for p in directory.rglob('*') if p.is_file()}
    return result
