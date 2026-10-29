"""Layered v3 orchestration (local side).

1. Preserve inputs; run GPU perception on the dev worker (OCR, ink masks,
   LaMa, SAM2 cutouts, tables, residual OCR).
2. Locally: one optional vision call that names hard rectangles. Measurement
   keeps a native shape only when it reproduces the box; every other accepted
   box is a hard rectangular crop that replaces a covered fuzzy mask. Leftover
   unnamed ink is lifted the same way. The slide fill is a solid colour; cutouts
   still come from the text-free page. Then page-calibrated typography,
   paragraph semantics (one optional text-only model call), SlideIR emission,
   fixed Artifact Tool compile.
3. Re-OCR the rendered previews on the worker and record QA metrics.

Every stage writes new files; nothing is overwritten. Results are
REVIEW_REQUIRED at best: no human or Office acceptance is implied.
"""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from ..storage import digest
from . import emit_ir, qa, semantics, typography
from .remote import Worker

SCHEMA = 'LayeredJob/1'


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_ir(project_root, ir_file):
    script = ("import('./scripts/ir.mjs').then(async m=>{const fs=await import('node:fs');"
              "m.validateIR(JSON.parse(fs.readFileSync(process.argv[1],'utf8')));console.log('ok')})"
              ".catch(e=>{console.error(e.message);process.exit(1)})")
    process = subprocess.run(['node', '--input-type=module', '-e', script, str(ir_file)], cwd=project_root,
                             capture_output=True, text=True, encoding='utf-8')
    return process.returncode == 0, (process.stderr or process.stdout).strip()


def build_page(name, perception_dir, page_dir, *, project_root, remote=None, model=None, vision_model=None):
    perception = _read(perception_dir/'perception.json')
    with Image.open(perception_dir/'source.png') as image:
        rgb = np.asarray(image.convert('RGB')).copy()
    if digest(perception_dir/'source.png') != perception['source_sha256']:
        raise ValueError('perception source hash mismatch')
    page_dir.mkdir(parents=True)
    from .review import review
    perception, reviewed = review(perception_dir, perception, remote=remote, model=vision_model, record_dir=page_dir)
    styles, calibration = typography.measure(rgb, perception)
    table_ids = {i for t in perception['tables'] for i in t['owned_text_ids']}
    crops = doubtful_lines(perception, styles)
    styles = {k: v for k, v in styles.items() if k not in table_ids and k not in crops}
    containers = [e['bbox'] for e in perception['elements'] if e['kind'] == 'native_shape']
    paragraphs, report = semantics.group(styles, containers, remote=remote, model=model, record_dir=page_dir)
    _write(page_dir/'typography.json', {'calibration': calibration, 'styles': styles})
    _write(page_dir/'semantics.json', {**report, 'paragraphs': paragraphs})
    notes = [f'Paragraph grouping: {"model-validated" if report["model_used"] else "deterministic"}.']
    if crops:
        notes.append(f'{len(crops)} doubtful short OCR line(s) kept as source pixels, not native text: '+', '.join(sorted(crops)))
    ir = emit_ir.emit(perception_dir, perception, styles, paragraphs, page_dir/'proposal', notes=notes,
                      source_crops=sorted(crops))
    ok, message = validate_ir(project_root, page_dir/'proposal'/'slideir.json')
    retained = {i for a in ir['assets'] for i in a['contains_evidence_ids']} | set(ir.get('unresolved_evidence_ids', []))
    violations = qa.residual_violations(perception, retained)
    uncovered = ocr_coverage_gaps(perception, ir)
    result = {'page': name, 'ir_valid': ok, 'ir_message': message if not ok else '',
              'uncovered_ocr_ids': uncovered,
              'semantics': {'model_used': report['model_used'], 'issues': report['issues'], 'usage': report['usage']},
              'review': {'model_used': reviewed['model_used'], 'issues': reviewed['issues'], 'usage': reviewed['usage'],
                         'applied': reviewed['applied']},
              'typography': calibration, 'residual_text_violations': violations,
              'counts': {'native_text_nodes': sum(n['type'] == 'text' for n in ir['nodes']),
                         'native_text_lines': sum(len(n['evidence_ids']) for n in ir['nodes'] if n['type'] == 'text'),
                         'native_shapes': sum(n['type'] == 'shape' for n in ir['nodes']),
                         'tables': sum(n['type'] == 'table' for n in ir['nodes']),
                         'cutouts': sum(n['type'] == 'image' for n in ir['nodes']),
                         'retained_text': len(retained & {t['id'] for t in perception['ocr']}),
                         'ocr_lines': len(perception['ocr'])}}
    if not ok:
        result['status'] = 'FAILED'
        return result
    from ..candidate_workflow import ArtifactCompiler
    built = ArtifactCompiler(root=project_root).build(page_dir/'proposal'/'slideir.json', page_dir/'compiled')
    result.update(pptx=str(Path(built['pptx']).relative_to(page_dir)), preview=str(Path(built['preview']).relative_to(page_dir)),
                  pptx_sha256=digest(built['pptx']))
    result['metrics'] = qa.measure(perception_dir/'source.png', built['preview'], ir, directory=page_dir)
    duplicates = qa.duplicate_under_objects(perception_dir, perception)
    result['duplicates_under_objects'] = duplicates
    qa.lift_probe(perception_dir, perception, page_dir/'lift-probe.png')
    result['lift_probe'] = 'lift-probe.png'
    result['status'] = 'FAILED' if violations or uncovered or duplicates else 'REVIEW_REQUIRED'
    return result


def doubtful_lines(perception, styles):
    """Short low-confidence lines no installed font explains are usually icons
    read as glyphs ('B', '?', '品'). Their ink is already erased, so they return
    as source-pixel crops rather than being re-typed or lost."""
    score = {t['id']: t['score'] for t in perception['ocr']}
    return {i for i, s in styles.items() if not s['fitted'] and len(s['text'].strip()) <= 3
            and (score[i] < .8 or len(s['text'].strip()) == 1 and score[i] < .95)}


def ocr_coverage_gaps(perception, ir):
    """OCR lines that no node, table cell, cutout or unresolved list accounts for."""
    owned = set(ir.get('unresolved_evidence_ids', []))
    for node in ir['nodes']:
        owned.update(node.get('evidence_ids', []))
        owned.update(node.get('contains_evidence_ids', []))
    for asset in ir['assets']:
        owned.update(asset.get('contains_evidence_ids', []))
    return sorted(t['id'] for t in perception['ocr'] if t['id'] not in owned)


def run(inputs, output, *, worker_config, project_root, remote=None, model=None, vision_model=None, job=None):
    job = job or 'lv3-'+uuid4().hex[:10]
    root = Path(output)/job
    (root/'input').mkdir(parents=True, exist_ok=False)
    names = []
    for path in inputs:
        name = Path(path).stem
        shutil.copyfile(path, root/'input'/f'{name}.png')
        names.append(name)
    worker = Worker(worker_config)
    started = time.time()
    source_digest, source_dir = worker.sync_source()
    worker.upload(job, sorted((root/'input').glob('*.png')))
    log = worker.run(job, ['pages', '/job/in', '/job/out'], source=source_dir)
    worker.download(job, 'out', root)
    (root/'out').rename(root/'perception')
    perception_seconds = time.time()-started
    pages = {}
    for name in names:
        try:
            pages[name] = build_page(name, root/'perception'/name, root/'pages'/name, project_root=project_root,
                                     remote=remote, model=model, vision_model=vision_model)
        except Exception as error:  # recorded, the remaining pages still run
            pages[name] = {'page': name, 'status': 'FAILED', 'error': f'{type(error).__name__}: {error}'}
    previews = [root/'pages'/n/p['preview'] for n, p in pages.items() if p.get('preview')]
    if previews:
        staged = root/'qa-previews'; staged.mkdir()
        for name, page in pages.items():
            if page.get('preview'):
                shutil.copyfile(root/'pages'/name/page['preview'], staged/f'{name}.png')
        worker.upload(job, sorted(staged.glob('*.png')), subdir='qa')
        worker.run(job, ['ocr', '/job/qa/ocr.json', *[f'/job/qa/{p.name}' for p in sorted(staged.glob('*.png'))]],
                   source=source_dir)
        worker.download(job, 'qa/ocr.json', root)
        rendered = _read(root/'ocr.json')
        for name, page in pages.items():
            if page.get('preview'):
                perception = _read(root/'perception'/name/'perception.json')
                page['metrics']['ocr_char_recall'] = qa.text_recall(perception['ocr'], rendered.get(f'{name}.png', []))
    tokens = sum(p.get('semantics', {}).get('usage', {}).get('total_tokens', 0)
                 + p.get('review', {}).get('usage', {}).get('total_tokens', 0) for p in pages.values())
    summary = {'schema': SCHEMA, 'job': job, 'model': model, 'worker_source_sha256': source_digest,
               'perception_seconds': round(perception_seconds, 1), 'perception_log': log.strip().splitlines()[-20:],
               'total_model_tokens': tokens, 'pages': pages, 'human_acceptance': False, 'office_acceptance': False}
    summary['files'] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(root.rglob('*')) if p.is_file() and 'compiled' not in p.parts}
    _write(root/'job.json', summary)
    return summary
