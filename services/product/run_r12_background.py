"""Frozen-scene regional R12 experiment. Never publishes a generation."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from product.materialize import digest, write
from product.object_extract import rectangle
from product.parallel_contracts import SceneGraph
from product.parallel_store import key
from product.parallel_fusion import glyph_evidence
from product.parallel_rebuild import background_model
from product.parallel_text_candidates import peer_protection
from product.recovery.diagnostics import save_trial
from product.recovery.gradient_backing import fit_gradient
from product.recovery.layerd_adapter import lower_backgrounds
from product.recovery.lama_adapter import LamaClient
from product.recovery.remote_gpu_lease import remote_gpu_lease
from product.recovery.text_mask_completion import TextMaskCompletion
from product.recovery.background_adapter import RecoveryBlocked


def run(args):
    source_file, scene_file = Path(args.source), Path(args.scene)
    graph = SceneGraph.model_validate_json(scene_file.read_text('utf8'))
    if digest(source_file) != graph.source_sha256:
        raise ValueError('source binding mismatch')
    original_scene = digest(scene_file)
    source = np.array(Image.open(source_file).convert('RGB'))
    text = next(t for t in graph.text.texts if t.id == args.object_id)
    if graph.ownership[text.id] != 'RESOLVED':
        raise ValueError('text content/ownership must be resolved before trials')
    style = next(s for s in graph.text.styles if s.id == text.style)
    x, y, w, h = map(round, text.bbox)
    box = [x-3, y-3, w+6, h+6]
    if min(box[:2]) < 0 or box[0]+box[2] > source.shape[1] or box[1]+box[3] > source.shape[0]:
        raise ValueError('text scope leaves source')
    protected = peer_protection(text, graph, source, np.zeros(source.shape[:2], bool))
    initial_backing, _ = background_model(source, box, protected)
    ink, legacy_unknown = glyph_evidence(source, box, style.color, initial_backing)
    mask = np.zeros(source.shape[:2], bool)
    mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = ink
    reference, fit_report = fit_gradient(source, box, protected)
    completion = TextMaskCompletion.analyze(source, mask, protected, box, reference, glyph_height=style.size_px)
    legacy_uncertain = np.zeros_like(mask)
    legacy_uncertain[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = legacy_unknown
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'gradient-reference.json', fit_report)
    write(output/'mask-completion.json', {**completion.report,
          'legacy_unknown_pixels': int(legacy_unknown.sum()), 'source_file_sha256': digest(source_file),
          'legacy_unknown_reclassified': {name: int((legacy_uncertain & pixels).sum())
                                          for name, pixels in completion.evidence.items()},
          'scene_sha256': original_scene, 'object_id': text.id, 'bbox': box})
    np.savez_compressed(output/'mask-evidence.npz', **completion.evidence, **completion.masks)
    palette = {'confirmed_text': [220, 35, 40], 'text_antialias': [255, 170, 0],
               'possible_text': [160, 30, 220], 'protected_peer': [0, 160, 80], 'unknown': [80, 150, 220]}
    visual = np.zeros_like(source)
    bx, by, bw, bh = box
    for name, evidence in completion.evidence.items():
        visual[evidence] = palette[name]
        Image.fromarray(evidence[by:by+bh, bx:bx+bw].astype('uint8')*255).save(output/(name+'.png'))
    Image.fromarray(visual[by:by+bh, bx:bx+bw]).save(output/'evidence-classes.png')
    lower = lower_backgrounds(source, args.layerd, mask)
    config = json.loads(Path(args.lama_config).read_text('utf8'))['lama'] if args.lama_config else None
    client = LamaClient(config['url'], config['model_binding'], args.lama_cache) if config else None
    records = []
    for mask_name, selected in completion.masks.items():
        if (selected & protected).any():
            raise ValueError('completed mask crossed protected peer')
        variants = [('gradient', reference, {'fit': fit_report})]
        variants.extend(('layerd-'+str(i), bg.image, bg.provenance) for i, bg in enumerate(lower))
        def record_trial(method, raw, provenance):
            candidate_id = mask_name+'--'+method
            row = save_trial(output/candidate_id, source, selected, raw, protected, box,
                             reference, fit_report, {'candidate': candidate_id,
                             'mask_variant': mask_name, 'method': method, 'provenance': provenance,
                             'source_sha256': graph.source_sha256, 'scene_sha256': original_scene,
                             'scene_graph_sha256': key(graph.model_dump()),
                             'object_id': text.id})
            records.append(row)
            print(candidate_id, row['status'], 'missing', row['visible_source_pixels_outside_mask'],
                  'residual', row['remaining_source_signal_pixels'], flush=True)
        for method, raw, provenance in variants:
            record_trial(method, raw, provenance)
        if client is not None:
            try:
                with remote_gpu_lease(config['gpu_lease']) as gpu_receipt:
                    if gpu_receipt['status'] != 'ACQUIRED':
                        raise RecoveryBlocked('GPU_LEASE_UNCONFIGURED', 'physical GPU lease is required')
                    raw = client(source, selected)
                record_trial('big-lama', raw, {'request': client.receipt, 'gpu_lease': gpu_receipt})
            except (RecoveryBlocked, OSError, TimeoutError, ValueError) as error:
                write(output/(mask_name+'--big-lama-failure.json'), {
                    'status': 'REVIEW_REQUIRED', 'error_type': type(error).__name__,
                    'reason_code': getattr(error, 'code', 'LAMA_UNAVAILABLE'),
                    'other_candidates_preserved': True, 'automatic_retry': False})
    if digest(source_file) != graph.source_sha256 or digest(scene_file) != original_scene:
        raise ValueError('regional experiment mutated input')
    write(output/'index.json', {'schema': 'R12RegionalTrials/1', 'source_sha256': graph.source_sha256,
          'scene_sha256': original_scene, 'object_id': text.id, 'bbox': box,
          'published': False, 'candidates': records})
    write(output/'input-bindings.json', {'source': str(source_file.resolve()), 'scene': str(scene_file.resolve()),
          'layerd': str(Path(args.layerd).resolve()), 'source_sha256': digest(source_file),
          'scene_sha256': original_scene, 'code': {p.name: digest(p) for p in
          [Path(__file__), *Path(__file__).with_name('product').joinpath('recovery').glob('*.py')]}})
    if args.stage_baseline:
        from product.recovery.validation import stage_and_verify
        if not args.fonts:
            raise ValueError('staging requires the verified installed font manifest')
        passed = [row for row in records if row['status'] == 'PRECHECK_PASS']
        if passed:
            best = min(passed, key=lambda row: row['scope_error_mae_rgb'])
            try:
                stage_and_verify(output/best['candidate'], graph, args.stage_baseline,
                                 json.loads(Path(args.fonts).read_text('utf8')), output/'acceptance', office=args.office)
            except Exception as error:
                write(output/'stage-failure.json', {'status': 'REVIEW_REQUIRED',
                      'error_type': type(error).__name__, 'selected_trial': best['candidate'],
                      'published': False, 'baseline_retained': True})
                raise
        else:
            write(output/'stage-result.json', {'status': 'REVIEW_REQUIRED', 'reason': 'No background passed regional checks'})
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--layerd', required=True)
    parser.add_argument('--object-id', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lama-config')
    parser.add_argument('--lama-cache', default='.runtime/r12-lama-cache')
    parser.add_argument('--stage-baseline')
    parser.add_argument('--fonts')
    parser.add_argument('--office', action='store_true')
    run(parser.parse_args())
