"""Local, bounded candidate builds. The conversation authors and reviews; tools record.

No network, model calls, acceptance decisions, or edits to submitted bundles.
Each build attempt and each review is immutable. A manifest binds the source,
human scope and baseline; revisions bind their complete evidence packet.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import shutil
import time
from uuid import uuid4

import numpy as np
from PIL import Image, ImageChops

from image2ppt.contracts import AuthoringRecord
from image2ppt.storage import digest, key, local_file, now, read_json, run_lock, write_json
from .region_plan import CAPABILITIES, HUMAN_SCOPE_KEYS, validate_human_decision


class WorkflowError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise WorkflowError(message)


def _box(box, size):
    _require(isinstance(box, list) and len(box) == 4 and all(
        type(v) in (int, float) and math.isfinite(v) for v in box), 'invalid ROI')
    x, y, w, h = box
    _require(x >= 0 and y >= 0 and w > 0 and h > 0 and x+w <= size[0] and y+h <= size[1],
             'ROI outside source')
    return box


def _inside(box, roi):
    x, y, w, h = box
    a, b, c, d = roi
    return x >= a-.01 and y >= b-.01 and x+w <= a+c+.01 and y+h <= b+d+.01


def _bundle_files(file, authoring=False):
    file = Path(file).resolve(strict=True)
    ir = read_json(file)
    files = {file.name: file}
    source = local_file(file.parent, ir['source_image'])
    _require(digest(source) == ir['source_sha256'], 'source hash changed')
    files[ir['source_image']] = source
    with Image.open(source) as image:
        _require(list(image.size) == ir['canvas']['source_px'], 'source dimensions changed')
    for asset in ir['assets']:
        path = local_file(file.parent, asset['path'])
        _require(digest(path) == asset['sha256'], 'asset hash changed: '+asset['id'])
        files[asset['path']] = path
    if authoring:
        path = file.parent / 'authoring.json'
        record = AuthoringRecord.model_validate(read_json(path))
        _require(record.ir_sha256 == digest(file), 'stale authoring record')
        files['authoring.json'] = path
    return ir, files


def _copy_bundle(file, destination, authoring=False):
    ir, files = _bundle_files(file, authoring)
    destination.mkdir(parents=True, exist_ok=False)
    for relative, path in files.items():
        # Retain exact IR bytes and asset paths, including its authoring hash.
        target = destination / relative
        _require(target.resolve().is_relative_to(destination.resolve()), 'bundle path escapes snapshot')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return ir, destination / Path(file).name


def _inventory(directory):
    return {p.relative_to(directory).as_posix(): digest(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def _verify(directory, files):
    for relative, expected in files.items():
        _require(digest(local_file(directory, relative)) == expected, 'evidence hash changed: '+relative)


def prepare(plans_file, human_file, baseline_ir, region_id, output, *, context_nodes=()):
    """Create a new candidate; context nodes explicitly extend the editable ROI.

    Context is for an approved adjacent title/icon, not a permission to rewrite
    the rest of the page. Its IDs and the resulting boundary are recorded.
    """
    plans, human_data = read_json(plans_file), read_json(human_file)
    rows = [r for r in plans['regions'] if r['region_id'] == region_id]
    _require(len(rows) == 1, 'region must exist exactly once')
    row = rows[0]
    humans = human_data.get('decisions', human_data)
    human = validate_human_decision(humans[region_id])
    _require(HUMAN_SCOPE_KEYS <= set(human) and human['decision'] != 'pending',
             'candidate builds require an explicit complete human scope')
    baseline_ir = Path(baseline_ir).resolve(strict=True)
    baseline, files = _bundle_files(baseline_ir)
    roi = _box(list(row['bbox']), baseline['canvas']['source_px'])
    context_nodes = list(context_nodes)
    nodes = {n['id']: n for n in baseline['nodes']}
    _require(len(set(context_nodes)) == len(context_nodes) and all(n in nodes for n in context_nodes),
             'unknown or duplicate context node')
    boxes = [roi] + [nodes[n]['bbox'] for n in context_nodes]
    left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    boundary = [left, top, max(b[0]+b[2] for b in boxes)-left, max(b[1]+b[3] for b in boxes)-top]
    _box(boundary, baseline['canvas']['source_px'])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate = output / ('candidate-'+uuid4().hex[:16])
    candidate.mkdir()
    _, snapshot = _copy_bundle(baseline_ir, candidate/'baseline')
    for source, name in [(plans_file, 'plans.json'), (human_file, 'human-decisions.json')]:
        shutil.copyfile(source, candidate/name)
    originals = {str(path.resolve()): digest(path) for path in [*files.values(), Path(plans_file), Path(human_file)]}
    manifest = {
        'schema': 'CandidateWorkflow/1', 'candidate_id': candidate.name,
        'region_id': region_id, 'page': row['page'], 'created_at': now(),
        'human': human, 'roi': roi, 'editable_boundary': boundary, 'context_node_ids': context_nodes,
        'mutable_node_ids': [n['id'] for n in baseline['nodes'] if _inside(n['bbox'], boundary)],
        'baseline_ir': snapshot.relative_to(candidate).as_posix(),
        'source_sha256': baseline['source_sha256'], 'max_corrections': 2,
        'original_files': originals, 'snapshot_files': _inventory(candidate),
        'network_calls': 0, 'human_acceptance': False,
    }
    write_json(candidate/'candidate.json', manifest, exclusive=True)
    return {'candidate': str(candidate), 'manifest_sha256': digest(candidate/'candidate.json')}


def _manifest(candidate):
    candidate = Path(candidate).resolve(strict=True)
    manifest = read_json(candidate/'candidate.json')
    _require(manifest.get('schema') == 'CandidateWorkflow/1', 'invalid candidate manifest')
    _verify(candidate, manifest['snapshot_files'])
    humans = read_json(candidate/'human-decisions.json')
    expected = validate_human_decision(humans.get('decisions',humans)[manifest['region_id']])
    _require(manifest['human']==expected, 'manifest scope differs from authoritative snapshot')
    for path, expected in manifest['original_files'].items():
        _require(digest(path) == expected, 'prepared input changed: '+path)
    return candidate, manifest


def _without_z(node):
    return {k: v for k, v in node.items() if k != 'z'}


def _scope_check(baseline, proposed, mapping, manifest, *, check_mapping=True):
    _require(proposed['source_sha256'] == manifest['source_sha256'], 'candidate source differs')
    for field in ('canvas', 'background'):
        _require(proposed[field] == baseline[field], 'page '+field+' is frozen')
    before = {n['id']: n for n in baseline['nodes']}
    after = {n['id']: n for n in proposed['nodes']}
    _require(len(after) == len(proposed['nodes']), 'duplicate candidate node')
    mutable = set(manifest['mutable_node_ids'])
    frozen = set(before) - mutable
    for nid in frozen:
        _require(nid in after and _without_z(before[nid]) == _without_z(after[nid]), 'frozen node changed: '+nid)
    ordered = lambda ir: [n['id'] for n in sorted(ir['nodes'], key=lambda n: n['z']) if n['id'] in frozen]
    _require(ordered(baseline) == ordered(proposed), 'frozen node order changed')
    for nid, node in after.items():
        if nid not in frozen:
            _require(_inside(node['bbox'], manifest['editable_boundary']), 'node outside editable boundary: '+nid)
    # Do not mutate an image or evidence owned by an otherwise frozen node.
    old_assets, new_assets = ({a['id']: a for a in ir['assets']} for ir in (baseline, proposed))
    old_evidence, new_evidence = ({e['id']: e for e in ir['evidence']} for ir in (baseline, proposed))
    frozen_evidence = set()
    for nid in frozen:
        node = before[nid]
        frozen_evidence.update(node.get('evidence_ids', []))
        if node['type'] == 'image':
            aid = node['asset_id']
            _require(new_assets.get(aid) == old_assets[aid], 'frozen asset changed: '+aid)
            frozen_evidence.update(old_assets[aid].get('contains_evidence_ids', []))
    for eid in frozen_evidence:
        _require(new_evidence.get(eid) == old_evidence[eid], 'frozen evidence changed: '+eid)
    old_groups = [g for g in baseline['groups'] if set(g['member_ids']) & frozen]
    new_groups = [g for g in proposed['groups'] if set(g['member_ids']) & frozen]
    _require(old_groups == new_groups, 'group containing frozen nodes changed')
    if not check_mapping:
        _require(set(after) <= set(before), 'corrections must retain reviewed object IDs; prepare a new candidate to restructure')
        return
    _require(set(mapping) == {'native', 'retained'}, 'mapping requires native and retained sections')
    native, retained = mapping['native'], mapping['retained']
    human = manifest['human']
    _require(isinstance(native, dict) and isinstance(retained, dict), 'invalid mapping')
    _require(set(retained) <= set(CAPABILITIES['retained_parts']), 'unknown retained part')
    _require(human['decision']!='native' or not retained, 'native human scope cannot retain region images')
    _require(set(native) <= set(human['allowed_native_parts']), 'native targets exceed human scope')
    _require(set(human['required_edit_targets']) <= set(native), 'required edit target missing')
    _require(set(human['must_retain_parts']) <= set(retained), 'required retained part missing')
    native_ids, retained_ids = set(), set()
    for target, ids in native.items():
        _require(isinstance(ids, list) and ids and len(ids) == len(set(ids)), 'empty/duplicate native mapping')
        for nid in ids:
            _require(nid in after and nid not in frozen and after[nid]['type'] in ('text','shape','table','chart'),
                     'native mapping must reference an editable native node: '+str(nid))
        native_ids.update(ids)
    for target, ids in retained.items():
        _require(isinstance(ids, list) and ids and len(ids) == len(set(ids)), 'empty/duplicate retained mapping')
        for aid in ids:
            _require(aid in new_assets and any(n['type']=='image' and n.get('asset_id')==aid and n['id'] not in frozen
                                              for n in after.values()), 'retained mapping requires a region image asset')
        retained_ids.update(ids)
    for nid, node in after.items():
        if nid in frozen:
            continue
        _require((node['type']=='image' and node['asset_id'] in retained_ids) or nid in native_ids,
                 'unmapped region node: '+nid)
    return {'native': native, 'retained': retained,
            'semantic_coverage': 'REVIEW_REQUIRED; IDs/types do not establish visual meaning'}


class ArtifactCompiler:
    def __init__(self, root=None, timeout=300):
        from image2ppt.compiler import ArtifactBackend
        self.backend = ArtifactBackend(root=root, timeout=timeout)

    def build(self, ir_file, output):
        from image2ppt.runner import Runner
        # Keep the proprietary exporter below Windows' legacy path limit.
        # Retain this run as compiler evidence and copy its completed revision.
        staging = self.backend.root/'.build'/('cw-'+uuid4().hex[:8])
        result = Runner(self.backend).replay(staging, ir=ir_file)
        source = Path(result['build']).parent
        destination = Path(output)/'build'
        shutil.copytree(source, destination)
        return {**result, **{field:str(destination/name) for field,name in
                 [('build','build.json'),('pptx','result.pptx'),('preview','preview.png'),
                  ('review','review.json'),('objects','objects.json')]}}


def _comparison(source_file, preview_file, roi, output, prefix):
    with Image.open(source_file) as image:
        source = image.convert('RGB')
    with Image.open(preview_file) as image:
        preview = image.convert('RGB').resize(source.size, Image.Resampling.BILINEAR)
    x, y, w, h = roi
    bounds = (math.floor(x), math.floor(y), math.ceil(x+w), math.ceil(y+h))
    original, rendered = source.crop(bounds), preview.crop(bounds)
    side = Image.new('RGB', (original.width*2, original.height), 'white')
    side.paste(original, (0,0)); side.paste(rendered, (original.width,0))
    for suffix, image in [('source',original),('rendered',rendered),('side-by-side',side),
                          ('difference',ImageChops.difference(original,rendered))]:
        image.save(output/(prefix+'-'+suffix+'.png'))
    mae = lambda a,b: float(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float)).mean())
    return {'page_mean_pixel_difference': mae(source,preview), 'region_mean_pixel_difference':mae(original,rendered)}


def build(candidate, candidate_ir, mapping_file, *, compiler=None):
    candidate = Path(candidate).resolve(strict=True)
    with run_lock(candidate):
        revisions = candidate/'revisions'; revisions.mkdir(exist_ok=True)
        previous = sorted(revisions.glob('r[0-9][0-9][0-9]'))
        successful = [p for p in previous if (p/'result.json').exists() and read_json(p/'result.json')['state']=='TRIAL_BUILT']
        _require(len(successful) < 3, 'two correction rounds exhausted; review required')
        revision = revisions/f'r{len(previous):03d}'; revision.mkdir()
        started = time.monotonic()
        state = {'schema':'CandidateBuild/1','revision':revision.name,'correction_index':len(successful),'state':'PROPOSED',
                 'history':[{'state':'PROPOSED','basis':'new immutable attempt'}], 'human_acceptance':False}
        try:
            _, manifest = _manifest(candidate)
            state['manifest_sha256'] = digest(candidate/'candidate.json')
            if previous:
                prior = read_json(previous[-1]/'result.json')
                _require(prior.get('manifest_sha256', state['manifest_sha256']) == state['manifest_sha256'], 'candidate manifest changed')
                _verify(previous[-1], prior.get('files', {}))
            if successful:
                _require((successful[-1]/'review.json').is_file(), 'review previous build before correcting')
            proposed, copied = _copy_bundle(Path(candidate_ir).resolve(strict=True), revision/'input', authoring=True)
            mapping = read_json(mapping_file)
            write_json(revision/'mapping.json', mapping, exclusive=True)
            baseline_file = local_file(candidate, manifest['baseline_ir'])
            coverage = _scope_check(read_json(baseline_file), proposed, mapping, manifest)
            if successful:
                last = successful[-1]
                last_review = read_json(last/'review.json')
                previous_ir = next(p for p in (last/'input').glob('*.json')
                                   if read_json(p).get('schema_version')=='0.1')
                reviewed_ids = sorted({nid for issue in last_review['issues'] for nid in issue['object_ids']})
                _scope_check(read_json(previous_ir), proposed, mapping,
                             {**manifest,'mutable_node_ids':reviewed_ids},check_mapping=False)
            state['history'].append({'state':'VALIDATED_STRUCTURE','basis':'bound source, authoring, scope and object mapping'})
            compiler = compiler or ArtifactCompiler()
            # Both sides use this exact compiler/runtime; historical metrics remain separate.
            base_result = compiler.build(baseline_file, revision/'baseline-build')
            built = compiler.build(copied, revision/'candidate-build')
            packet = revision/'evidence'; packet.mkdir()
            source_file = local_file(copied.parent, proposed['source_image'])
            shutil.copyfile(source_file,packet/'source.png')
            for name, result in [('baseline',base_result), ('candidate',built)]:
                for field, suffix in [('pptx','.pptx'),('preview','-preview.png'),('objects','-objects.json'),('build','-build.json')]:
                    shutil.copyfile(result[field],packet/(name+suffix))
            metrics = {name:_comparison(source_file, result['preview'], manifest['roi'], packet, name)
                       for name,result in [('baseline',base_result),('candidate',built)]}
            write_json(packet/'coverage.json',coverage,exclusive=True)
            write_json(packet/'metrics.json',metrics,exclusive=True)
            state['history'].append({'state':'TRIAL_BUILT','basis':'final PPTX reimport and fixed-source ROI comparison'})
            state.update(state='TRIAL_BUILT',metrics=metrics,coverage=coverage,
                         elapsed_seconds=round(time.monotonic()-started,3))
            # Bind the complete evidence packet and all revision inputs, not only the image.
            state['files'] = _inventory(revision)
            state['artifact_binding'] = key({'manifest':state['manifest_sha256'],'revision':revision.name,'files':state['files']})
        except Exception as exc:
            state.update(state='FAILED',error={'type':type(exc).__name__,'message':str(exc)[:2000]},
                         elapsed_seconds=round(time.monotonic()-started,3))
            state['files'] = _inventory(revision)
            state['artifact_binding'] = key({'manifest':state.get('manifest_sha256'),'revision':revision.name,'files':state['files']})
        write_json(revision/'result.json',state,exclusive=True)
        _report(candidate)
        return {**state, 'revision':str(revision)}


def review(candidate, revision_name, review_file):
    candidate = Path(candidate).resolve(strict=True)
    with run_lock(candidate):
        _, manifest = _manifest(candidate)
        _require(revision_name in [p.name for p in (candidate/'revisions').glob('r[0-9][0-9][0-9]')], 'unknown revision')
        revision = candidate/'revisions'/revision_name
        result = read_json(revision/'result.json')
        _require(result['state']=='TRIAL_BUILT', 'only a successful build can be reviewed')
        _require(digest(candidate/'candidate.json')==result['manifest_sha256'], 'candidate manifest changed')
        _verify(revision,result['files'])
        _require(result['artifact_binding']==key({'manifest':result['manifest_sha256'],'revision':revision_name,'files':result['files']}),
                 'invalid artifact binding')
        record = read_json(review_file)
        _require(set(record)=={'schema','artifact_binding','reviewer','issues','unresolved','summary','human_acceptance'},
                 'invalid review fields')
        _require(record['schema']=='CandidateReview/1' and record['artifact_binding']==result['artifact_binding'],
                 'stale review binding')
        _require(record['human_acceptance'] is False, 'conversation review cannot grant human acceptance')
        _require(isinstance(record['reviewer'],str) and record['reviewer'].strip() and
                 isinstance(record['summary'],str) and record['summary'].strip(), 'review author and summary required')
        _require(isinstance(record['issues'],list) and isinstance(record['unresolved'],list) and
                 all(isinstance(v,str) and v.strip() for v in record['unresolved']), 'invalid review issues')
        # Use the snapshot IR name recorded by its content, not directory iteration order.
        ir_file = next(p for p in (revision/'input').glob('*.json') if read_json(p).get('schema_version')=='0.1')
        nodes = {n['id'] for n in read_json(ir_file)['nodes']}
        for issue in record['issues']:
            _require(set(issue)=={'object_ids','kind','source_evidence','rendered_evidence','suggestion'}, 'invalid issue fields')
            _require(isinstance(issue['object_ids'],list) and issue['object_ids'] and
                     set(issue['object_ids']) <= nodes, 'unknown review object')
            _require(issue['kind'] in ('text','style','position','crop','scope','fidelity'), 'invalid issue kind')
            _require(isinstance(issue['suggestion'],str) and issue['suggestion'].strip(), 'specific correction required')
            for field in ('source_evidence','rendered_evidence'):
                _require(issue[field] in result['files'] and issue[field].startswith('evidence/') and
                         issue[field].endswith('.png'), 'issue evidence must belong to this packet')
        status = 'REVIEW_REQUIRED' if record['issues'] or record['unresolved'] else 'VERIFIED'
        correction_index = sum(1 for p in (candidate/'revisions').glob('r[0-9][0-9][0-9]')
                               if p.name < revision_name and (p/'result.json').exists()
                               and read_json(p/'result.json')['state']=='TRIAL_BUILT')
        record = {**record,'state':status,'reviewed_at':now(),
                  'correction_budget_exhausted':correction_index>=manifest['max_corrections']}
        write_json(revision/'review.json',record,exclusive=True)
        _report(candidate)
        return record


def _report(candidate):
    manifest = read_json(candidate/'candidate.json')
    lines = ['# Candidate workflow', '', 'Human acceptance: false. Visual review is not Office/WPS acceptance.', '',
             '| Revision | State | Baseline region MAE | Candidate region MAE |', '|---|---|---:|---:|']
    records=[]
    for path in sorted((candidate/'revisions').glob('r[0-9][0-9][0-9]')):
        if not (path/'result.json').exists():continue
        result=read_json(path/'result.json'); reviewed=read_json(path/'review.json') if (path/'review.json').exists() else None
        state=reviewed['state'] if reviewed else result['state']; metrics=result.get('metrics',{})
        lines.append(f"| {path.name} | {state} | {metrics.get('baseline',{}).get('region_mean_pixel_difference','-')} | {metrics.get('candidate',{}).get('region_mean_pixel_difference','-')} |")
        if result['state']=='FAILED':
            lines += ['', path.name+': '+result['error']['message'], '']
            continue
        ir_file=next(p for p in (path/'input').glob('*.json') if read_json(p).get('schema_version')=='0.1')
        ir=read_json(ir_file)
        artifact=lambda name:{'label':name,'path':(path/'evidence'/name).relative_to(candidate).as_posix(),'sha256':digest(path/'evidence'/name)}
        history=deepcopy(result['history'])
        if reviewed:history.append({'state':state,'basis':reviewed['summary']})
        records.append({'candidate_id':candidate.name+'-'+path.name,'region_id':manifest['region_id'],
            'page':manifest['page'],'decision':manifest['human']['decision'],'state':state,'history':history,
            'baseline':{'label':'same-runtime baseline','artifacts':[artifact('baseline.pptx')],'metrics':metrics['baseline']},
            'candidate':{'label':'conversation-authored candidate','artifacts':[artifact('candidate.pptx')],
                         'metrics':metrics['candidate'],'edit_goals':manifest['human']['required_edit_targets'] or ['retain_module'],
                         'target_fidelity':manifest['human']['fidelity_target']},
            'coverage':{target:{'status':'partial','evidence':'Mapped IDs; semantic coverage requires visual review.'}
                        for target in result['coverage']['native']},
            'budget':{'nodes':len(ir['nodes']),'text_nodes':sum(n['type']=='text' for n in ir['nodes']),
                      'images':sum(n['type']=='image' for n in ir['nodes']),'shapes':sum(n['type']=='shape' for n in ir['nodes'])},
            'outcome':{'status':state,'human_acceptance':False,'reason':reviewed['summary'] if reviewed else 'Awaiting conversation review'}})
    write_json(candidate/'records.json',{'schema_version':'region-candidates/0.1','candidates':records})
    (candidate/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
