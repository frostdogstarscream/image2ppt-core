"""Review frozen A/B candidate package; no automatic release or registry update."""
import argparse,getpass,os,json
from pathlib import Path
from product.materialize import write,digest
from product.parallel_provider import Reviewer
from product.parallel_store import Store
from product.parallel_review import PROMPT,panels
from product.parallel_residual_gate import inspect_residuals
from product.resource_lease import lease
from product.contracts import Review

def main():
 p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--prompt-key',action='store_true');a=p.parse_args();root=Path(a.output).resolve()
 if a.prompt_key:os.environ['OPENCODE_GO_API_KEY']=getpass.getpass('Go key (hidden): ')
 comparison=json.loads((root/'comparison.json').read_text(encoding='utf-8'));store=Store(root/'review-state');actor=Reviewer(root)
 office=json.loads((root/'after-office/report.json').read_text(encoding='utf-8'));results=[]
 for index,row in enumerate(comparison['pages'],1):
  ir=Path(row['after']['ir']);source=ir.parent/'source.png';preview=root/'after-office'/f'page-{index:02d}-before.png'
  residuals=inspect_residuals(ir,root/'after-office',index,root/f"page-{row['page']:02d}-residuals.json")
  details,refs=panels(source,preview,ir,root/'after-office',index,root/f"page-{row['page']:02d}-panels")
  ctx={'source_sha256':digest(source),'plan_sha256':digest(ir),'preview_sha256':digest(preview),
       'scene':json.loads(ir.read_text(encoding='utf-8')),'transactions':json.loads((ir.parent/'transactions.json').read_text(encoding='utf-8')),
       'raster_remainder_is_explicit_fallback':True,'office_pass':office['status']=='PASS_APP_AUTOMATED','actual_edit_evidence':refs,'source_owner_residuals':residuals}
  try:
   r=store.call('region-fix-review',actor,PROMPT,ctx,[str(source),str(preview),*details],Review,lease('go'))
   if any(getattr(r,k)!=ctx[k] for k in ('source_sha256','plan_sha256','preview_sha256')):raise ValueError('review binding mismatch')
   review=r.model_dump()
  except Exception as exc:review={'quality':'UNKNOWN','reason':type(exc).__name__+':'+str(exc)[:200]}
  result={'source_page':row['page'],'review':review,'residuals':residuals,'human_acceptance':False};results.append(result)
  write(root/f"page-{row['page']:02d}-review.json",result)
  print(json.dumps({'page':row['page'],'review':review},ensure_ascii=False),flush=True)
 write(root/'independent-review.json',{'pages':results,'status':'REVIEW_REQUIRED','parallel_auto':False})
if __name__=='__main__':main()
