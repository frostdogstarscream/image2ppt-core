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
    parser.add_argument('--quality-policy', choices=('legacy', 'quality-loop'), default='legacy')
    parser.add_argument('--baseline-job')
    parser.add_argument('--segmentation-plans', help='LayeredPlanInputs/1 manifest of saved source-bound plan jobs; quality-loop only')
    parser.add_argument('--segmentation-backend', choices=('sam2-auto', 'sam3-prompted'), default='sam2-auto')
    parser.add_argument('--sam3-worker-config', help='independent SAM3 development worker host/root')
    parser.add_argument('--sam3-box-policy', choices=('prompt', 'crop-hard'), default='prompt',
                        help='crop-hard runs each original plan box as an isolated crop and forbids mask spill')
    parser.add_argument('--max-extra-vision-calls', type=int, choices=range(3), default=2)
    parser.add_argument('--max-repair-rounds', type=int, choices=range(3), default=2)
    parser.add_argument('--vision-model', default=None,
                        help='model that accepts images for the component review; default IMAGE2PPT_VISION_MODEL or --model')
    args = parser.parse_args()
    remote = model = vision = None
    if args.model_config:
        import os
        from image2ppt_vision.auto_transport import SSHRemote, load_config as load_client
        remote, model = SSHRemote(load_client(args.model_config)), args.model
        vision = args.vision_model or os.environ.get('IMAGE2PPT_VISION_MODEL') or model
    inputs = [str(Path(p).resolve(strict=True)) for p in args.inputs]
    options = {}
    if args.quality_policy == 'quality-loop':
        from image2ppt_vision.layered.quality_loop import run
        options = dict(baseline_job=args.baseline_job, max_extra_vision_calls=args.max_extra_vision_calls,
                       max_repair_rounds=args.max_repair_rounds, segmentation_plans=args.segmentation_plans,
                       segmentation_backend=args.segmentation_backend,
                       sam3_box_policy=args.sam3_box_policy,
                       sam3_worker_config=load_config(args.sam3_worker_config) if args.sam3_worker_config else None)
    elif args.baseline_job or args.segmentation_plans or args.sam3_worker_config or args.segmentation_backend != 'sam2-auto' or args.sam3_box_policy != 'prompt':
        parser.error('--baseline-job and --segmentation-plans require --quality-policy quality-loop')
    summary = run(inputs, args.output, worker_config=load_config(args.worker_config), project_root=ROOT,
                  remote=remote, model=model, vision_model=vision, **options)
    brief = {'job': summary['job'], 'output': str(Path(args.output)/summary['job']),
             'total_model_tokens': summary['total_model_tokens'],
             'pages': {k: {'status': v['status'], 'ssim': v.get('metrics', {}).get('ssim'),
                           'ocr_char_recall': v.get('metrics', {}).get('ocr_char_recall')} for k, v in summary['pages'].items()}}
    sys.stdout.reconfigure(errors='backslashreplace')
    print(json.dumps(brief, ensure_ascii=False, indent=2), flush=True)
    if summary.get('status') == 'FAILED' or summary.get('delivery', {}).get('status') == 'FAILED' or any(p['status'] == 'FAILED' for p in summary['pages'].values()):
        sys.exit(3)


if __name__ == '__main__':
    main()
