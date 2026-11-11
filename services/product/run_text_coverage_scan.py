"""L1 scan of every unverified text in a committed real-evidence job."""
import argparse
import json
from collections import Counter
from pathlib import Path

from product.materialize import digest, write
from product.parallel_contracts import SceneGraph
from product.parallel_store import Store, key
from product.recovery import text_actions, text_probe


def scan(job, layer_directory, directory, page=1):
    job, directory = Path(job).resolve(), Path(directory).resolve()
    commit = Store(job/'state').committed()
    if commit is None:
        raise ValueError('a hash-verified committed real-evidence job is required')
    result = commit['result']
    original = job/f'page-{page:02d}/barrier-1.json'
    graph = SceneGraph.model_validate_json(original.read_text('utf8'))
    request = json.loads((job/'request.json').read_text('utf8'))
    source = job/'inputs'/request['sources'][page-1]
    if digest(source) != graph.source_sha256:
        raise ValueError('scan source and scene differ')
    fonts = json.loads((job/'fonts.json').read_text('utf8'))
    baseline = Path(result['slide_irs'][page-1]).parent
    verified = {row['object_id'] for row in result['regions'][page-1]
                if row['state'] == 'VERIFIED_LOCAL'}
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for text in graph.text.texts:
        if text.id in verified:
            row = {'object_id': text.id, 'state': 'VERIFIED_REUSED', 'office': 'PREVIOUSLY_VERIFIED'}
        else:
            try:
                row = text_probe.probe_one(graph, text.id, source, layer_directory,
                                           baseline, fonts, directory/'probes')
            except (ValueError, OSError, KeyError) as exc:
                row = {'object_id': text.id, 'state': 'BLOCKED',
                       'error_type': type(exc).__name__, 'reason': str(exc)}
        row.update(text=text.text, role=next(style.role for style in graph.text.styles if style.id == text.style),
                   source_bbox=text.bbox, failure_category=text_actions.classify_failure(row))
        rows.append(row)
        print(json.dumps({field: row.get(field) for field in ('object_id', 'state', 'failure_category')}), flush=True)
    binding = {'source_sha256': graph.source_sha256, 'source_scene_sha256': digest(original),
               'parent_ir_sha256': digest(baseline/'slideir.json'),
               'code': {str(path): digest(path) for path in
                        [Path(__file__), Path(text_probe.__file__), Path(text_actions.__file__)]}}
    summary = {'schema': 'R1205TextCoverageScan/1', 'binding': binding,
               'source_job': str(job), 'page': page, 'total_source_text_objects': len(graph.text.texts),
               'state_counts': dict(Counter(row['state'] for row in rows)),
               'failure_counts': dict(Counter(row['failure_category'] for row in rows if row['failure_category'])),
               'objects': rows, 'external_model_calls': 0, 'compiler_calls': 0, 'office_calls': 0,
               'automatic_publication': False, 'next_gate': 'Actual Office and independent visual verification'}
    report = directory/('scan-'+key(binding)[:24]+'.json')
    write(report, summary)
    print(json.dumps({'report': str(report), 'state_counts': summary['state_counts']}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--layers', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--page', type=int, default=1)
    args = parser.parse_args()
    scan(args.job, Path(args.layers).resolve(), args.output, args.page)


if __name__ == '__main__':
    main()
