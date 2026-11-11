"""Batch independent asset recovery using a verified job's frozen evidence."""
import argparse
import json
from pathlib import Path

from product.parallel_contracts import SceneGraph
from product.parallel_store import Store
from product.recovery.batch_assets import recover


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'layers', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--sam-budget', type=int, default=5)
    parser.add_argument('--office', action='store_true')
    args = parser.parse_args()
    job = Path(args.job).resolve()
    committed = Store(job/'state').committed()
    if committed is None:
        raise ValueError('verified immutable parent job required')
    result = committed['result']
    if not result['office_pass'] or not all(r.get('passed') for r in result['independent_review']):
        raise ValueError('parent job requires Office and independent review')
    base = Path(result['slide_irs'][0]).parent
    graph = SceneGraph.model_validate_json((base/'scene.json').read_text('utf8'))
    report = recover(graph, base/'source.png', args.layers, base, job, args.output,
                     json.loads((job/'fonts.json').read_text('utf8')),
                     limit=args.limit, office=args.office, sam_budget=args.sam_budget)
    print(json.dumps({'targets': report['targets'], 'objects': [
        {k: r.get(k) for k in ('object_id', 'status', 'strategy', 'reason', 'elapsed_seconds')}
        for r in report['objects']], 'elapsed_seconds': report['elapsed_seconds'],
        'model_calls': report['model_calls']}), flush=True)


if __name__ == '__main__':
    main()
