"""Compare a layered v3 job with a v2 automatic job on the same pages.

Read-only. Both previews are scored against the same source pixels with the
same measures; neither score is visual or human acceptance.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'services/vision/src')]


def score(source, preview):
    import numpy as np
    from image2ppt_vision.layered.qa import _load_pair, ssim_map
    a, b = _load_pair(source, preview)
    return {'ssim': round(float(ssim_map(a, b).mean()), 4),
            'mean_absolute_pixel_difference': round(float(np.abs(a.astype(float)-b.astype(float)).mean()), 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layered', required=True, help='layered job directory (LayeredJob/1)')
    parser.add_argument('--v2', required=True, help='v2 automatic job directory')
    parser.add_argument('--pair', nargs='+', required=True, help='v3page=v2page, e.g. 06=P5')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    layered, v2 = Path(args.layered), Path(args.v2)
    job = json.loads((layered/'job.json').read_text(encoding='utf-8'))
    rows = []
    for pair in args.pair:
        mine, theirs = pair.split('=')
        page = job['pages'][mine]
        source = layered/'perception'/mine/'source.png'
        result = json.loads((v2/theirs/'result.json').read_text(encoding='utf-8'))
        rows.append({'v3_page': mine, 'v2_page': theirs,
                     'v3': {**score(source, layered/'pages'/mine/page['preview']), 'status': page['status'],
                            'ocr_char_recall': page['metrics'].get('ocr_char_recall'),
                            'model_tokens': page['semantics']['usage'].get('total_tokens', 0), 'model_requests': int(page['semantics']['model_used'])},
                     'v2': {**score(source, v2/'delivery'/f'{theirs}-preview.png'), 'status': result['machine_quality_status'],
                            'model_tokens': result['usage'].get('total_tokens', 0), 'model_requests': result['model_requests']}})
    report = {'schema': 'LayeredV2Comparison/1', 'layered_job': job['job'], 'v2_job': v2.name, 'pages': rows,
              'caution': 'Same-source pixel measures only; not visual, Office or human acceptance.'}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
