"""Layered v3 pipeline: GPU perception on the dev worker, local compile and QA.

Explicit entry point; importing the project never starts remote work or model calls.
Configs hold host names and paths only. Tokens and API keys stay on the server.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'services/vision/src')]


def main():
    from image2ppt_vision.layered.pipeline import run
    from image2ppt_vision.layered.remote import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', default=str(ROOT/'outputs'/'layered'))
    parser.add_argument('--worker-config', default=str(ROOT/'.build'/'layered-worker.json'),
                        help='{"ssh_host", "remote_root"} of the GPU dev worker')
    parser.add_argument('--model-config', help='auto_transport client config; omit for deterministic grouping')
    parser.add_argument('--model', default='deepseek-flash')
    args = parser.parse_args()
    remote = model = None
    if args.model_config:
        from image2ppt_vision.auto_transport import SSHRemote, load_config as load_client
        remote, model = SSHRemote(load_client(args.model_config)), args.model
    inputs = [str(Path(p).resolve(strict=True)) for p in args.inputs]
    summary = run(inputs, args.output, worker_config=load_config(args.worker_config), project_root=ROOT,
                  remote=remote, model=model)
    brief = {'job': summary['job'], 'output': str(Path(args.output)/summary['job']),
             'total_model_tokens': summary['total_model_tokens'],
             'pages': {k: {'status': v['status'], 'ssim': v.get('metrics', {}).get('ssim'),
                           'ocr_char_recall': v.get('metrics', {}).get('ocr_char_recall')} for k, v in summary['pages'].items()}}
    sys.stdout.reconfigure(errors='backslashreplace')
    print(json.dumps(brief, ensure_ascii=False, indent=2), flush=True)
    if any(p['status'] == 'FAILED' for p in summary['pages'].values()):
        sys.exit(3)


if __name__ == '__main__':
    main()
