from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'experiments'))
from native_results import results
from native_region_metrics import measure


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


def test_case_and_score_require_matching_deck(tmp_path):
    edition = {'sha256':'host','actions':[{'action':'text','page':1,'status':'NO_TARGET_OBJECT'}]}
    save(tmp_path/'comparison.json',dict(job_id='job',state='REVIEW_REQUIRED',sources=['source'],versions={'host':edition}))
    save(tmp_path/'delivery/baseline-scorecard.json',dict(pptx_sha256='wrong',office_status='PASS',pages=[]))
    row, = results(tmp_path)
    assert row['office']=='UNKNOWN' and row['observed_edits']==0 and row['frozen_text_hits']==0
    raw=tmp_path/'agent-before-fallback'
    raw.mkdir()
    (raw/'result.pptx').write_bytes(b'fixture')
    save(raw/'office/report.json',[])
    with pytest.raises(ValueError,match='differs'):
        list(results(tmp_path))
    save(tmp_path/'delivery/agent-scorecard.json',
         dict(pptx_sha256=sha256(b'fixture').hexdigest(),office_status='PASS',pages=[]))
    _, agent=results(tmp_path)
    assert agent['version']=='agent_candidate' and agent['observed_edits']=='UNKNOWN'


def test_region_measurement_retains_source_and_render_bindings(tmp_path):
    src, render = tmp_path/'original.png', tmp_path/'office.png'
    Image.new('RGB',(80,48),'white').save(src)
    image=Image.open(src).convert('RGB')
    image.paste('black',(22,9,49,30))
    image.save(render)
    value = measure(src,render,[[1,'title',(22,9,27,21),'text','edited'],[1,'photo',(0,0,10,8),'delete',None]])
    assert value['source_sha256'] != value['render_sha256']
    assert value['regions'][0]['ssim'] < .1 and value['regions'][1]['ssim']==1
