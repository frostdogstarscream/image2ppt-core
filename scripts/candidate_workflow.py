"""Offline candidate preparation, actual-file builds and conversation review."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'services/vision/src')]
from image2ppt_vision.candidate_workflow import ArtifactCompiler, prepare, build, review


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for name in ('plans','human-decisions','baseline-ir','region','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--context-node',action='append',default=[])
    p=sub.add_parser('build')
    for name in ('candidate','ir','mapping'):p.add_argument('--'+name,required=True)
    p.add_argument('--timeout',type=float,default=300)
    p=sub.add_parser('review')
    for name in ('candidate','revision','record'):p.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        result=prepare(args.plans,args.human_decisions,args.baseline_ir,args.region,args.output,context_nodes=args.context_node)
    elif args.command=='build':
        result=build(args.candidate,args.ir,args.mapping,compiler=ArtifactCompiler(ROOT,args.timeout))
        result={k:v for k,v in result.items() if k!='files'}
    else:result=review(args.candidate,args.revision,args.record)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result.get('state')=='FAILED':raise SystemExit(1)


if __name__=='__main__':main()
