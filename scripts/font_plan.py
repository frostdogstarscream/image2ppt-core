"""Prepare a font-only API job, or apply its retained response without OCR edits."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'services/vision/src'))
from image2ppt_vision.font_plan import prepare_font_job,apply_font_job,verify_font_export

p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
prep=sub.add_parser('prepare');prep.add_argument('--ir',required=True);prep.add_argument('--output',required=True)
prep.add_argument('--model',required=True);prep.add_argument('--base-url',required=True)
apply=sub.add_parser('apply');apply.add_argument('--job',required=True);apply.add_argument('--output',required=True)
verify=sub.add_parser('verify');verify.add_argument('--ir',required=True);verify.add_argument('--pptx',required=True);verify.add_argument('--slide',type=int,default=1)
a=p.parse_args()
if a.command=='prepare':result=prepare_font_job(a.ir,a.output,model=a.model,base_url=a.base_url)
elif a.command=='apply':result=apply_font_job(a.job,a.output)
else:result=verify_font_export(a.ir,a.pptx,slide=a.slide)
print(json.dumps(result,ensure_ascii=False))
