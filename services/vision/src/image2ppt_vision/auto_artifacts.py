"""Actual PPTX builds and edit probes reused by the automatic coordinator."""
from pathlib import Path
import shutil
import time
from uuid import uuid4

import numpy as np
from PIL import Image, ImageChops

from image2ppt.compiler import ArtifactBackend
from image2ppt.contracts import ProbeCase
from image2ppt.quality import edit_copy, inspect_pptx
from image2ppt.storage import digest, read_json
from .auto_provider import save_new, fingerprint
from .candidate_workflow import ArtifactCompiler
from .supervised_elements import materialize_elements


class DeadlineBackend(ArtifactBackend):
    def __init__(self, root, deadline):
        super().__init__(root=root)
        self.deadline = deadline

    def invoke(self, script, *args):
        remaining = self.deadline-time.time()
        if remaining <= 0:
            raise TimeoutError('page deadline exhausted')
        self.timeout = min(300, remaining)
        return super().invoke(script, *args)


def file_hashes(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def verify_hashes(root, files):
    root = Path(root).resolve()
    if not files:
        raise ValueError('empty artifact binding')
    for name, expected in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ValueError('automatic artifact integrity mismatch')


def choose_probe_targets(plan):
    texts = [t for t in plan['text_decisions'] if t['action'] == 'native']
    if not texts:
        return []
    top = [t for t in texts if t['bbox'][1] < plan['image_size'][1]*.28 and len(t['text']) >= 3] or texts
    title = max(top, key=lambda t: (t['font_size_px'], len(t['text'])))
    numbers = [t for t in texts if t['id'] != title['id'] and any(c.isdigit() for c in t['text'])]
    body = [t for t in texts if t['id'] != title['id'] and t['bbox'][1] >= plan['image_size'][1]*.28]
    selected = [title['id']]
    for pool in (numbers, body):
        available = [t for t in pool if t['id'] not in selected]
        if available:
            selected.append(max(available, key=lambda t: len(t['text']))['id'])
    return selected


def comparison(source, preview, directory):
    with Image.open(source) as f:
        a = f.convert('RGB')
    with Image.open(preview) as f:
        b = f.convert('RGB').resize(a.size, Image.Resampling.LANCZOS)
    side = Image.new('RGB', (a.width*2, a.height), 'white')
    side.paste(a); side.paste(b, (a.width, 0))
    side.save(directory/'comparison.png')
    ImageChops.difference(a, b).save(directory/'difference.png')
    return {'mean_absolute_pixel_difference': float(np.abs(np.asarray(a, dtype=float)-np.asarray(b, dtype=float)).mean()),
            'purpose': 'comparison_only_not_acceptance'}


def build_revision(source, inventory, plan, directory, *, project_root, deadline, model_id, policy_binding, response_sha256,
                   raster_scopes=None, allowed_retained_ids=()):
    """Each invocation creates a fresh directory, including failed build evidence."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    proposal = directory/'proposal'
    materialize_elements(source, inventory, plan, proposal, model_id=model_id,
                         response_sha256=response_sha256, raster_scopes=raster_scopes)
    compiler = ArtifactCompiler(root=project_root)
    compiler.backend = DeadlineBackend(project_root, deadline)
    built = compiler.build(proposal/'slideir.json', directory/'compiled')
    pptx = Path(built['pptx']); preview = Path(built['preview'])
    actual = inspect_pptx(pptx)
    records = {x['name']: x for x in actual['slides'][0]['objects']}
    expected = {t['id']: t for t in plan['text_decisions']}
    issues = []
    for tid, text in expected.items():
        emitted = records.get(tid, {})
        if text['action'] == 'retained_in_asset' and tid in allowed_retained_ids:
            continue
        if text['action'] != 'native' or 'text' not in emitted:
            issues.append({'id': 'native-'+tid, 'kind': 'editable', 'object_ids': [tid],
                           'severity': 'blocking', 'bbox': text['bbox'],
                           'description': 'OCR text is not a native editable text object in the final PPTX.'})
        elif emitted['text'].replace('\n', '') != text['text'].replace('\n', ''):
            issues.append({'id': 'text-'+tid, 'kind': 'text', 'object_ids': [tid],
                           'severity': 'blocking', 'bbox': text['bbox'],
                           'description': 'Exported text differs from the validated element plan.'})
    probes = choose_probe_targets(plan)
    packet = directory/'evidence'; packet.mkdir()
    shutil.copyfile(source, packet/'source.png')
    shutil.copyfile(pptx, packet/'result.pptx')
    shutil.copyfile(preview, packet/'preview.png')
    save_new(packet/'objects.json', actual)
    before = digest(pptx)
    if probes:
        edited = packet/'delete-probe.pptx'
        edit_copy(pptx, edited, ProbeCase(id='automatic-delete', action='delete_text', targets=probes))
        # Avoid native exporter's legacy Windows path limit for diagnostic renders.
        staging = Path(project_root)/'.build'/('ap-'+uuid4().hex[:8])
        staged = staging/'edited.pptx'; staging.mkdir(parents=True)
        shutil.copyfile(edited, staged)
        rendered = compiler.backend.render(staged, staging/'render')
        shutil.copyfile(rendered['pages'][0]['path'], packet/'delete-probe.png')
        if digest(pptx) != before:
            raise ValueError('original PPTX changed during edit probe')
    else:
        shutil.copyfile(preview, packet/'delete-probe.png')
        issues.append({'id': 'no-native-text', 'kind': 'editable', 'object_ids': [],
                       'severity': 'blocking', 'bbox': [0, 0, *plan['image_size']],
                       'description': 'No native text was available for an actual-file deletion probe.'})
    metrics = comparison(source, preview, packet)
    save_new(packet/'metrics.json', metrics)
    save_new(packet/'probe.json', {'action': 'delete_text', 'targets': probes,
                                 'original_pptx_unchanged': digest(pptx) == before,
                                 'actual_file_edited': bool(probes), 'human_acceptance': False})
    files = file_hashes(directory)
    result = {'schema': 'AutomaticBuild/1', 'files': files, 'policy_binding': policy_binding,
              'plan_sha256': fingerprint(plan), 'source_sha256': digest(source),
              'probe_targets': probes, 'structural_issues': issues, 'metrics': metrics,
              'counts': {'native_text': sum(t['action'] == 'native' for t in plan['text_decisions']),
                         'native_shapes': sum(o['kind'] == 'native_shape' for o in plan['objects']),
                         'raster_assets': sum(o['kind'] == 'raster_asset' for o in plan['objects']),
                         'retained_text': sum(t['action'] != 'native' for t in plan['text_decisions'])},
              'human_acceptance': False, 'office_acceptance': False,
              'retention_scope_sha256': fingerprint({'scopes': raster_scopes or {},
                                                    'text_ids': sorted(allowed_retained_ids)})}
    result['artifact_binding'] = fingerprint({'directory': directory.name, **result})
    save_new(directory/'result.json', result)
    return result
