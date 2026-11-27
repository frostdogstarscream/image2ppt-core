"""CLI bridge for source-bound text reconciliation and actual-file checks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'services/vision/src'))
from image2ppt_vision.text_ledger import reconcile_text, verify_ledger

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest='action', required=True)
prepare = sub.add_parser('reconcile')
for name in ('ir', 'calibration', 'current-pptx', 'output'):
    prepare.add_argument('--' + name, required=True)
prepare.add_argument('--slide', type=int, default=1)
verify = sub.add_parser('verify')
verify.add_argument('--folder', required=True)
verify.add_argument('--pptx')
verify.add_argument('--slide', type=int, default=1)
a = p.parse_args()
result = (reconcile_text(a.ir, a.calibration, a.current_pptx, a.output, slide=a.slide)
          if a.action == 'reconcile' else verify_ledger(a.folder, a.pptx, slide=a.slide))
print(json.dumps(result, ensure_ascii=False))
