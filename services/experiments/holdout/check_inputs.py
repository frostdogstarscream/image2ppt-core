"""Check frozen source identities and case structure without invoking reconstruction or Office."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check(project_root, workbench_inputs):
    manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
    cases_file = HERE / manifest['cases']
    cases_text = cases_file.read_text(encoding='utf-8')
    if sha256(cases_text.encode()) != manifest['cases_sha256_lf']:
        raise ValueError('Frozen case file changed')
    probe = HERE.parents[2] / manifest['probe']
    if sha256(probe.read_text(encoding='utf-8').encode()) != manifest['probe_sha256_lf']:
        raise ValueError('Probe differs from the version used to define these cases')

    pages = manifest['pages']
    if not 8 <= len(pages) <= 10 or [p['page'] for p in pages] != list(range(1, len(pages) + 1)):
        raise ValueError('Expected 8 to 10 pages in frozen upload order')
    roots = {'project': project_root, 'workbench_inputs': workbench_inputs}
    excluded = manifest['excluded_development_pages']
    seen_files = {p['sha256'] for p in excluded}
    seen_pixels = {p['rgb_sha256'] for p in excluded}
    for page in pages:
        source = roots[page['source_root']] / page['path']
        file_hash = sha256(source.read_bytes())
        with Image.open(source) as image:
            size = list(image.size)
            pixel_hash = sha256(image.convert('RGB').tobytes())
        if size != page['size'] or file_hash != page['sha256'] or pixel_hash != page['rgb_sha256']:
            raise ValueError(f'Changed input: {source}')
        if file_hash in seen_files or pixel_hash in seen_pixels:
            raise ValueError(f'Duplicate or excluded development page: {source}')
        seen_files.add(file_hash)
        seen_pixels.add(pixel_hash)

    counts = Counter()
    actions = Counter()
    width, height = manifest['probe_size']
    for i, (page, label, bbox, action, value) in enumerate(json.loads(cases_text), 1):
        if type(page) is not int or not 1 <= page <= len(pages) or not isinstance(label, str) or not label.strip():
            raise ValueError(f'Invalid page or label in case {i}')
        if len(bbox) != 4 or not all(type(v) is int for v in bbox):
            raise ValueError(f'Expected integer xywh in case {i}')
        x, y, w, h = bbox
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
            raise ValueError(f'Out-of-bounds box in case {i}')
        if action not in ('text', 'move', 'fill', 'delete'):
            raise ValueError(f'Unknown action in case {i}')
        if action == 'text':
            if not isinstance(value, str) or not value.strip() or ''.join(value.split()) == ''.join(label.split()):
                raise ValueError(f'Missing or unchanged replacement in case {i}')
        elif value != '':
            raise ValueError(f'Unexpected replacement in case {i}')
        counts[page] += 1
        actions[action] += 1

    for page in pages:
        count = counts[page['page']]
        if not 3 <= count <= 6 or count != page['case_count']:
            raise ValueError(f"Case count mismatch for page {page['page']}")
    return {'pages': len(pages), 'cases': sum(counts.values()), 'actions': dict(actions)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--workbench-inputs', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.project_root, args.workbench_inputs)))
