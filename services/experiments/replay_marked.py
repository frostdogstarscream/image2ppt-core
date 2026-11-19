"""Replay recorded marked plans and successful text trials on committed scenes."""

import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ('services/product', 'services/vision/src', 'ppt-rebuild/src')]

from product.materialize import digest, write
from product.parallel_contracts import SceneGraph
from product.parallel_pipeline import effective_texts
from product.resource_lease import lease
from product.runtime.batch_backbone import _chain_stage_from_spec, _collect_office_defects
from product.runtime.compiler import compile_local
from product.runtime.marked_routes import bind_plan
from product.parallel_fusion import fuse_incremental_scene
from product.recovery.marked_fields import FieldWidthMismatch, separated_fields
from product.recovery.region_candidates import adaptive_candidates


def replay_split_trials(base, graph, row, recorded, out):
    """Regenerate trials bound to the changed scene; never rewrite a saved trial digest."""
    source = row['snapshot']['path']
    image = np.array(Image.open(source).convert('RGB'))
    graph = graph.model_copy(deep=True)
    styles = {s.id: s for s in graph.text.styles}
    wanted = {t['object_id'] for t in recorded}
    for text in list(graph.text.texts):
        if len(graph.text.members.get(text.id, [])) != 1 or text.id in graph.selected:
            continue
        try:
            fields = separated_fields(image, text.text, text.bbox)
        except FieldWidthMismatch:
            wanted.discard(text.id)
            continue
        if not fields:
            continue
        graph.text.texts.remove(text)
        for i, (value, box, color) in enumerate(fields):
            tid = text.id + ('-detail' if i else '')
            style = styles[text.style].model_copy(update={'id': 'style-' + tid, 'color': color})
            graph.text.styles.append(style)
            graph.text.texts.append(
                text.model_copy(
                    update={'id': tid, 'text': value, 'bbox': box, 'style': style.id, 'spans': []}
                )
            )
            graph.text.members[tid] = list(graph.text.members[text.id])
            wanted.add(tid)
    graph = fuse_incremental_scene(base, base_graph=graph)
    out.mkdir(parents=True)
    copy = out / 'baseline'
    shutil.copytree(base, copy)
    write(copy / 'scene.json', graph.model_dump())
    trials = []
    failures = []
    for tid in sorted(wanted):
        folder = out / tid
        stream = adaptive_candidates(
            graph, tid, source, row.get('layers'), folder, baseline_directory=copy, marked_fields=True
        )
        try:
            for attempt in stream:
                if attempt['status'] == 'PRECHECK_PASS':
                    trials.append({'object_id': tid, 'trial_dir': str(folder / attempt['candidate'])})
                    break
            else:
                failures.append({'id': tid, 'reason': 'no passing local trial'})
        except ValueError as exc:
            failures.append({'id': tid, 'reason': str(exc)})
        finally:
            stream.close()
    write(out / 'trial-failures.json', failures)
    return copy, graph, trials


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def page_feedback(previous, page, source_sha):
    if previous is None:
        return {}, set()
    checks = read(Path((previous / 'office-check-path.txt').read_text(encoding='utf8').strip()))
    (deck,) = list((previous / 'composition').rglob('deck.pptx'))
    if checks['pptx_sha256'] != digest(deck) or not checks.get('residuals'):
        raise ValueError('expected Office checks bound to the previous replay deck')
    ir = Path(read(previous / 'replay.json')['pages'][page - 1]['ir'])
    graph = SceneGraph.model_validate(read(ir.parent / 'scene.json'))
    if graph.source_sha256 != source_sha:
        raise ValueError('Office feedback source changed')
    replacements = [r for r in checks['replacements']['records'] if r['page'] == page]
    scoped = {'residuals': [checks['residuals'][page - 1]], 'replacements': {'records': replacements}}
    repairs, bad = _collect_office_defects(scoped, graph, list(graph.selected))
    return {k: r['measured_frame_minimum_px'] for k, r in repairs.items()}, bad


def run(job, out, previous=None):
    state = read(job / 'runtime/state.json')
    out.mkdir(parents=True, exist_ok=False)
    irs, texts, pages = [], [], []
    for page, row in enumerate(state['pages'], 1):
        base = Path(row['ir']).parent
        graph = SceneGraph.model_validate(read(base / 'scene.json'))
        overrides, skip = page_feedback(previous, page, graph.source_sha256)
        candidate = next(
            c
            for c in state['candidates'].values()
            if c.get('status') == 'COMMITTED'
            and c.get('page') == page
            and c.get('batch_spec')
            and Path(c['ir']).resolve() == Path(row['ir']).resolve()
        )
        spec = candidate['batch_spec']
        original = SceneGraph.model_validate(read(Path(spec['initial_baseline']) / 'scene.json'))
        routes = bind_plan(
            state['marked_plans'][str(page)],
            state['experiment_marks'][str(page)],
            graph,
            row['snapshot']['path'],
        )
        # A table must acquire its cell pixels before its children are staged.
        # Rebuild text-only pages with table routes from their recorded pre-batch
        # scene so the replay exercises the production ordering.
        trials = [t for t in spec['trials'] if t['object_id'] not in graph.selected]
        if any(r['route'] == 'table' for r in routes) and all(
            v['kind'] == 'text' for v in graph.selected.values()
        ):
            base = Path(spec['initial_baseline'])
            retained = [t for t in spec['trials'] if t['object_id'] in graph.selected]
            base, _, _ = _chain_stage_from_spec(
                initial_graph=original,
                initial_baseline=base,
                fonts=read(state['fonts']),
                source_path=row['snapshot']['path'],
                directory=out,
                tag=f'page-{page}-retained',
                page_cache_dir=out,
                layers_dir=None,
                deferred_path=None,
                trials_spec=retained,
                frame_overrides=overrides,
                skip_ids=skip,
                include_assets=False,
                include_containers=False,
                include_region_crops=False,
                wanted_filter=None,
                sam_enabled=False,
                experiment_profile='combined',
                marked_routes=routes,
            )
            staged = SceneGraph.model_validate(read(base / 'scene.json'))
            base, original, trials = replay_split_trials(base, staged, row, [], out / f'fields-{page}')
        stage, ids, failed = _chain_stage_from_spec(
            initial_graph=original,
            initial_baseline=base,
            fonts=read(state['fonts']),
            source_path=row['snapshot']['path'],
            directory=out,
            tag=f'page-{page}',
            page_cache_dir=out,
            layers_dir=None,
            deferred_path=None,
            trials_spec=trials,
            frame_overrides=overrides,
            skip_ids=skip,
            include_assets=True,
            include_containers=True,
            include_region_crops=False,
            wanted_filter=None,
            sam_enabled=False,
            experiment_profile='combined',
            marked_routes=routes,
        )
        ir = stage / 'slideir.json'
        irs.append(ir)
        texts.append(effective_texts(SceneGraph.model_validate(read(stage / 'scene.json')), ir))
        pages.append(
            {
                'ir': str(ir),
                'staged': ids,
                'failures': failed,
                'routes': routes,
                'office_frame_repairs': overrides,
                'office_exclusions': sorted(skip),
            }
        )
        print('page', page, 'staged', ids, flush=True)
    write(out / 'replay.json', {'origin': 'recorded plan and text trial replay', 'pages': pages})
    with lease('compiler'):
        result = compile_local(irs, out / 'composition', texts)
    print(result['pptx'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument(
        '--repair-from', type=Path, help='Preceding replay directory with real Office checks'
    )
    args = parser.parse_args()
    run(args.job.resolve(), args.out.resolve(), args.repair_from)
