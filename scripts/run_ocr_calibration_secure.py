"""Run already-prepared OCR API jobs with a credential held only in memory."""
import argparse
import getpass
import json
import os
from pathlib import Path
import sys
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'services/vision/src'))
from image2ppt_vision.ocr_calibration import run_calibrations, _validated_input
from image2ppt_vision.model_provider import _NoRedirect, _endpoint
from image2ppt_vision.storage import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    args = parser.parse_args()
    runs = [Path(p).resolve() for p in args.runs]
    for run in runs:
        _validated_input(run)
    settings = [read_json(run / 'settings.json') for run in runs]
    if len({(s['base_url'], s['model']) for s in settings}) != 1:
        raise ValueError('one explicit provider and model per invocation required')
    key = getpass.getpass('API credential (hidden): ')
    if not key or any(ord(c) < 32 or ord(c) > 126 for c in key):
        raise SystemExit('Invalid credential')
    env = os.environ.copy()
    env['VISION_LLM_API_KEY'] = key
    try:
        url = _endpoint(settings[0]['base_url']).removesuffix('/chat/completions') + '/models'
        req = request.Request(url, headers={'Authorization': 'Bearer ' + key, 'Accept': 'application/json'})
        with request.build_opener(_NoRedirect).open(req, timeout=30) as response:
            data = json.loads(response.read(2_000_000))
        ids = [row['id'] for row in data.get('data', []) if isinstance(row, dict)
               and isinstance(row.get('id'), str) and key not in row['id'] and len(row['id']) <= 256]
        if settings[0]['model'] not in ids:
            raise SystemExit('Configured model not listed; no images transmitted')
        for run in runs:
            if not (run / 'model-discovery.json').exists():
                write_json(run / 'model-discovery.json', {'status': 'LISTED', 'model': settings[0]['model'],
                                                        'requested_model_listed': True})
        print(json.dumps({'model': settings[0]['model'], 'status': 'LISTED'}), flush=True)
        result = run_calibrations(runs, env=env, progress=lambda s: print(s, flush=True))
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        env.pop('VISION_LLM_API_KEY', None)
        key = None


if __name__ == '__main__':
    main()
