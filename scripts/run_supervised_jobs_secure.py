"""Explicit experimental jobs. Secret is read without echo or from stdin, held only in memory."""
import argparse
import getpass
import json
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'services/vision/src'))
from image2ppt_vision.job_binding import JobBindingError, verify_job
from image2ppt_vision.storage import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', help='jobs root for interactive mode')
    parser.add_argument('--jobs', nargs='+', help='job directory names to run once')
    parser.add_argument('--key-stdin', action='store_true', help='read credential from stdin (for non-interactive runs)')
    args = parser.parse_args()
    if args.jobs and not args.root:
        raise SystemExit('jobs root is required with --jobs')
    if not args.root:
        raise SystemExit('jobs root is required')
    root = Path(args.root).resolve()
    if args.key_stdin:
        key = sys.stdin.readline().strip()
    else:
        key = getpass.getpass('API credential (hidden): ')
    if not key or any(ord(c) < 32 or ord(c) > 126 for c in key):
        raise SystemExit('Invalid credential')
    env = os.environ.copy()
    env['VISION_LLM_API_KEY'] = key
    env['PYTHONPATH'] = str(ROOT / 'services/vision/src')

    def run(job):
        path = (root / job).resolve()
        if not path.is_relative_to(root) or not (path / 'request.json').is_file():
            return {'job': job, 'status': 'INVALID_JOB'}
        if (path / 'transport.json').exists() or (path / 'response.body').exists():
            return {'job': job, 'status': 'EXISTS'}
        settings = path / 'settings.json' if (path / 'settings.json').exists() else root / 'settings.json'
        try:
            verified = verify_job(path, settings)
        except JobBindingError as exc:
            write_json(path / 'transport.json', {
                'status': 'FAILED', 'error_code': 'job_binding_mismatch', 'binding_error': exc.code})
            return {'job': job, **json.loads((path / 'transport.json').read_text(encoding='utf8'))}
        timeout = verified['settings']['timeout_seconds']
        try:
            process = subprocess.run(
                [sys.executable, '-X', 'utf8', '-m', 'image2ppt_vision.api_eval_worker',
                 str(path / 'request.json'), str(settings), str(path)],
                env=env, cwd=ROOT, timeout=timeout + 30, capture_output=True)
            if not (path / 'transport.json').exists():
                write_json(path / 'transport.json', {
                    'status': 'FAILED', 'error_code': 'worker_failed', 'exit_code': process.returncode})
        except subprocess.TimeoutExpired:
            write_json(path / 'transport.json', {'status': 'FAILED', 'error_code': 'wall_timeout'})
        return {'job': job, **json.loads((path / 'transport.json').read_text(encoding='utf8'))}

    try:
        if args.jobs:
            with ThreadPoolExecutor(max_workers=3) as pool:
                for task in as_completed([pool.submit(run, job) for job in args.jobs]):
                    print(json.dumps(task.result()), flush=True)
            return
        while True:
            line = input('Jobs (space separated) or quit > ').strip()
            if line == 'quit':
                break
            jobs = line.split()
            if not 1 <= len(jobs) <= 6:
                continue
            with ThreadPoolExecutor(max_workers=3) as pool:
                for task in as_completed([pool.submit(run, job) for job in jobs]):
                    print(json.dumps(task.result()), flush=True)
    finally:
        env.pop('VISION_LLM_API_KEY', None)
        key = None


if __name__ == '__main__':
    main()
