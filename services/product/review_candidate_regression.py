"""Optional r11 shadow Go calls. Uses only existing explicitly configured providers."""
import argparse,os
from pathlib import Path
from product.parallel_quality_arbitrator import bundle_from_trials,review_shadow
from product.parallel_store import Store
from product.materialize import write


def run(manifest,output,actor=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    cache=Path(os.environ.get('IMAGE2PPT_PARALLEL_CACHE',str(root/'.runtime/parallel-cache')))
    # Same durable call cache as the upload pipeline, shared across report paths.
    store=Store(cache/'model-calls');results=[]
    for index,bundle in enumerate(bundle_from_trials(manifest)):
        results.append(review_shadow(bundle,store,actor,output/f'region-{index:02d}'))
    write(output/'shadow-summary.json',{'mode':'shadow','decisions':results,'publish_eligible':False,'human_acceptance':False})
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('manifest');parser.add_argument('output')
    parser.add_argument('--call-go',action='store_true');parser.add_argument('--opencode-profile')
    parser.add_argument('--prompt-key',action='store_true',help='Read Go key with a hidden prompt, memory only')
    args=parser.parse_args();actor=None
    if args.call_go:
        if args.prompt_key:
            import getpass,os
            os.environ['OPENCODE_GO_API_KEY']=getpass.getpass('Go key (hidden): ')
        if args.opencode_profile:
            from product.go_provider import load_profile
            load_profile(args.opencode_profile)
        from product.parallel_provider import Reviewer
        actor=Reviewer(args.output)
    run(args.manifest,args.output,actor)
