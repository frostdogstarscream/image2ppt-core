"""Independent Go negative control using an actual Office deletion preview."""
import argparse
import getpass
import json
import os
from pathlib import Path

from product.materialize import digest, write
from product.parallel_provider import Reviewer
from product.parallel_store import Store
from product.recovery.product_pipeline import review


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--prompt-go-key',action='store_true')
    args=parser.parse_args()
    job=Path(args.job).resolve();output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    commit=Store(job/'state').committed()
    if commit is None:raise ValueError('committed Office-verified candidate required')
    result=commit['result']
    if not result['office_pass']:raise ValueError('actual Office evidence required')
    ir=Path(result['slide_irs'][0])
    graph=json.loads(ir.read_text('utf8'))
    identifier=next(node['id'] for node in graph['nodes'] if node['type']=='text')
    office=next((job/'r12').glob('attempt-*/office-combined'))
    preview=office/f'01-{identifier}-delete.png'
    source=next((job/'snapshots').glob('*.png'))
    if args.prompt_go_key:
        os.environ['OPENCODE_GO_API_KEY']=getpass.getpass('OpenCode Go credential (hidden): ')
    snap={'path':str(source),'source_sha256':digest(source)}
    response=review(Store(output/'state'),Reviewer(output),snap,ir,preview,office,1,output/'panels',adaptive=True)
    write(output/'review.json',response)
    detected=not response['passed'] and any(identifier in f['node_ids'] for f in response['findings'])
    write(output/'control.json',{'schema':'R1203VisualNegativeControl/1','expected_missing_id':identifier,
        'source_job':str(job),'source_sha256':digest(source),'ir_sha256':digest(ir),
        'actual_office_delete_preview':str(preview),'preview_sha256':digest(preview),
        'severe_defect_detected':detected,'published':False,
        'purpose':'deliberately use actual deleted text as final preview; diagnostic only'})
    print(json.dumps({'severe_defect_detected':detected,'review':str(output/'review.json')}),flush=True)
    if not detected:raise SystemExit(1)


if __name__=='__main__':main()
