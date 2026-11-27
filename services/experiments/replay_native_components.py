"""Recompile an archived MoMA response on its original measured scene for local regression."""
import argparse
from pathlib import Path
import time

from product.materialize import digest, write
from product.parallel_contracts import SceneGraph
from product.parallel_pipeline import effective_texts
from product.runtime.compiler import compile_local
from product.runtime.native_synthesis import synthesize_page


def replay(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    pages, expected, timings = [], [], []
    for index, archive in enumerate(inputs, 1):
        started = time.monotonic()
        measured = archive / 'measured'
        page = synthesize_page({'ir': str(measured / 'slideir.json')}, output / f'page-{index:02d}',
                               replay_directory=archive)
        graph = SceneGraph.model_validate_json(Path(page['scene']).read_text('utf8'))
        pages.append(page)
        expected.append(effective_texts(graph, Path(page['ir'])))
        timings.append(round(time.monotonic() - started, 2))
    started = time.monotonic()
    artifacts = compile_local([Path(p['ir']) for p in pages], output / 'compiled', expected)
    write(output / 'composition.json', {
        'mode': 'verified_response_replay', 'model_calls': 0,
        'page_seconds': timings, 'compile_seconds': round(time.monotonic() - started, 2),
        'pptx_sha256': digest(artifacts['pptx']),
        'composition': {'artifacts': artifacts, 'pages': pages},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('measured_page', type=Path, nargs='+')
    args = parser.parse_args()
    replay([p.resolve() for p in args.measured_page], args.output.resolve())
