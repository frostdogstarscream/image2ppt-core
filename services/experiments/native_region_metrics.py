"""Score frozen edit regions against an actual PowerPoint render of matching dimensions."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'product'))
from product.materialize import digest, write
from product.scorecard import fidelity


def measure(source, preview, cases):
    with Image.open(source) as im:
        src = im.convert('RGB')
    with Image.open(preview) as im:
        render = im.convert('RGB').resize(src.size, Image.Resampling.LANCZOS)
    rows = []
    for _, label, (x,y,w,h), action, _ in cases:
        bounds = (x,y,x+w,y+h)
        if min(x,y)<0 or x+w>src.width or y+h>src.height or min(w,h)<=0:
            raise ValueError('Frozen case exceeds source extent')
        value = fidelity(np.asarray(src.crop(bounds)), np.asarray(render.crop(bounds)))
        rows.append(dict(target=label, action=action, bbox=[x,y,w,h], ssim=value['ssim'],
                         delta_e_p95=value['delta_e2000_p95'], mae_rgb=value['mae_rgb']))
    return dict(source_sha256=digest(source), render_sha256=digest(preview), regions=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path)
    p.add_argument('id')
    p.add_argument('source',type=Path)
    p.add_argument('preview',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    row=next(r for r in json.loads(a.manifest.read_text(encoding='utf8'))['pages'] if r['id']==a.id)
    if digest(a.source)!=row['sha256']:
        raise ValueError('Source differs from the frozen manifest')
    write(a.output,measure(a.source,a.preview,row['cases']))
