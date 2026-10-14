import argparse
import json
import sys
from pathlib import Path
from .compiler import ArtifactBackend
from .quality import inspect_pptx, probe
from .runner import Runner, regress
from .storage import read_json


def main(argv=None):
    parser=argparse.ArgumentParser(prog='image2ppt-core',description='Offline SlideIR 0.1 replay and actual-PPTX checks')
    parser.add_argument('--project-root',help='Complete source/toolchain root; defaults to IMAGE2PPT_PROJECT_ROOT or checkout')
    parser.add_argument('--timeout',type=float,default=300)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('doctor')
    replay=commands.add_parser('replay'); inputs=replay.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--ir'); inputs.add_argument('--build'); replay.add_argument('--output',required=True)
    inspect=commands.add_parser('inspect'); inspect.add_argument('--pptx',required=True)
    probes=commands.add_parser('probe'); probes.add_argument('--build',required=True); probes.add_argument('--cases',required=True)
    regression=commands.add_parser('regress'); regression.add_argument('--suite',required=True); regression.add_argument('--output',required=True)
    resume=commands.add_parser('resume'); resume.add_argument('--run',required=True)
    args=parser.parse_args(argv)
    try:
        import math
        if not math.isfinite(args.timeout) or args.timeout<=0:
            raise ValueError('timeout must be finite and positive')
        if args.command=='inspect':
            result=inspect_pptx(args.pptx)
        else:
            root=args.project_root
            if args.command=='resume' and root is None:
                root=read_json(Path(args.run)/'run.json')['config']['project_root']
            backend=ArtifactBackend(root,args.timeout)
            if args.command=='doctor': result=backend.doctor()
            elif args.command=='replay': result=Runner(backend).replay(args.output,ir=args.ir,build=args.build)
            elif args.command=='resume': result=Runner(backend).resume(args.run)
            elif args.command=='probe': result=probe(args.build,args.cases,backend)
            else: result=regress(args.suite,args.output,backend)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 3 if result.get('execution_status')=='FAILED' else 0
    except Exception as error:
        print(json.dumps({'execution_status':'FAILED','type':type(error).__name__,'error':str(error)},ensure_ascii=False),file=sys.stderr)
        return 2
