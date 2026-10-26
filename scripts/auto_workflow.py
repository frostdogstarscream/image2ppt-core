"""Explicit entry point; importing the project never starts model calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'services/vision/src')]


def main():
    from image2ppt_vision.auto_transport import load_config, SSHRemote
    from image2ppt_vision.auto_provider import preflight
    from image2ppt_vision.auto_workflow import create_job, run_job
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('preflight'); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    p = commands.add_parser('run'); p.add_argument('--config', required=True); p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--output', required=True); p.add_argument('--gate-pages', nargs='+', type=int)
    p.add_argument('--preflight'); p.add_argument('--preference', choices=['faithful_editable', 'fidelity_first'], default='faithful_editable')
    p = commands.add_parser('resume'); p.add_argument('--job', required=True)
    p.add_argument('--retry-failed', action='store_true', help='Retry failed pages only within their original time/call budgets; preserve previous results.')
    p = commands.add_parser('serve'); p.add_argument('--config', required=True); p.add_argument('--data', required=True)
    p.add_argument('--port', type=int, default=18765); p.add_argument('--preflight')
    args = parser.parse_args()
    if args.command == 'preflight':
        config = load_config(args.config)
        result = preflight(SSHRemote(config), config, args.output)
    elif args.command == 'run':
        config = load_config(args.config)
        job = create_job(args.inputs, args.output, config, gate_pages=args.gate_pages, preference=args.preference)
        print(json.dumps({'job': str(job), 'status': 'QUEUED'}), flush=True)
        result = run_job(job, project_root=ROOT, preflight_directory=args.preflight)
    elif args.command == 'resume':
        result = run_job(args.job, project_root=ROOT, retry_failed=args.retry_failed)
    else:
        if not 1024 <= args.port <= 65535:
            parser.error('port must be 1024..65535')
        import uvicorn
        from image2ppt_vision.auto_api import create_app
        app = create_app(args.data, args.config, ROOT, cached_preflight=args.preflight)
        uvicorn.run(app, host='127.0.0.1', port=args.port)
        return
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
