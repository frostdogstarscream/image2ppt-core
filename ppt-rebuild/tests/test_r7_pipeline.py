"""R7 integration gates: metadata freeze and final failure aggregation."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from _helpers import SRC
import r7_pipeline as pipeline
import r7_edit_checks


def baseline():
    return {'schema_version':'0.1','source_image':'source.png','source_sha256':'a'*64,
            'canvas':{'source_px':[20,15],'target_px':[20,15]},'background':{'color':'FFFFFF'},
            'nodes':[{'id':'text','type':'text','bbox':[2,3,5,4],'z':1,'evidence_ids':['e1'],
                      'text':'原文','font':{'ea':'黑体','latin':'Arial','size_pt':10}}],
            'assets':[],'evidence':[{'id':'e1','kind':'text','text':'原文','bbox':[2,3,5,4]}],'groups':[]}


@pytest.mark.parametrize('mutation', [
    lambda ir: ir['background'].update(color='112233'),
    lambda ir: ir.update(source_image='other.png'),
    lambda ir: ir.update(source_sha256='b'*64),
    lambda ir: ir['canvas'].update(target_px=[40,30]),
    lambda ir: ir['evidence'][0].update(text='改写'),
])
def test_frozen_scene_includes_source_background_and_original_evidence(mutation):
    before=baseline();after=copy.deepcopy(before);mutation(after)
    with pytest.raises(ValueError):pipeline.frozen_scene(before,after)


def test_frozen_scene_allows_declared_new_object_without_rewriting_old_evidence():
    before=baseline();after=copy.deepcopy(before)
    after['nodes'].append({'id':'arrow','type':'shape','bbox':[12,3,5,4],'z':2,'evidence_ids':['e2']})
    after['evidence'].append({'id':'e2','kind':'shape','bbox':[12,3,5,4]})
    assert pipeline.frozen_scene(before,after,added=['arrow'])['status']=='PASS'


def test_final_deck_qa_processes_all_pages_and_reports_required_edit_failure(tmp_path,monkeypatch):
    """No renderer/model calls: test orchestration, not real-image acceptance."""
    ir=baseline()
    pages=[{'page':f'{i:02d}_page','ir':'fake-ir','source':'fake-source',
            'preview':'fake-normal','deleted_text_preview':'fake-deleted',
            'execution_directory':str(tmp_path/f'{i:02d}'),'action':'BASELINE_RETAINED',
            'visual':{}} for i in range(1,13)]
    batch={'input':'fake-input','pages':pages,'deck':{'pptx':'fake-deck'}}
    def read(path):
        if str(path)=='fake-input':return {'pages':pages}
        if str(path).endswith('relations.json'):return {'elements':[]}
        return ir
    class Backend:
        def render(self,*args):return {'pages':[{'path':'fake-render'} for _ in pages]}
    monkeypatch.setattr(pipeline,'read',read)
    monkeypatch.setattr(pipeline,'delete_deck_text',lambda *a:None)
    monkeypatch.setattr(pipeline,'verify_styles',lambda *a:None)
    monkeypatch.setattr(pipeline,'verify_native_shape_properties',lambda *a:None)
    monkeypatch.setattr(pipeline,'pixels',lambda *a:np.zeros((15,20,3),np.uint8))
    monkeypatch.setattr(pipeline,'clean_check',lambda *a:{'status':'PASS'})
    monkeypatch.setattr(pipeline,'check_relations',lambda *a:{'status':'NOT_APPLICABLE'})
    monkeypatch.setattr(r7_edit_checks,'audit_page',lambda p,*a:{
        'status':'REVIEW_REQUIRED' if p['page'].startswith('03') else 'NOT_APPLICABLE',
        'probes':[{'target':'icon','status':'REVIEW_REQUIRED','reason':'old_site_unverified'}]
        if p['page'].startswith('03') else []})
    result=pipeline.audit(batch,tmp_path,Backend())
    assert len(result['pages'])==12
    assert '03_page' in result['hard_failure_pages']


def test_edit_gate_distinguishes_unknown_clean_plate_from_failed_edit_behavior():
    probe={'target':'icon','action':'move','status':'REVIEW_REQUIRED',
           'export_consistency':'PASS','protected_pixels':'PASS','outside_changed_pixels':0,
           'actual_changed_pixels':100,'protected_xml':'PASS','package_parts':'PASS',
           'clean_plate_correctness':{'status':'REVIEW_REQUIRED'}}
    assert pipeline.edit_hard_failures({'probes':[probe]},[('icon','move')])==[]
    probe['export_consistency']='FAIL'
    assert pipeline.edit_hard_failures({'probes':[probe]},[('icon','move')])
    assert pipeline.edit_hard_failures({'probes':[{'target':'icon','action':'move'}]},[('icon','move')])


def test_expected_long_text_overflow_is_recorded_without_hiding_other_edit_failures():
    probe={'target':'body','action':'replace_long','overflow_detected':True,'error_type':'ValueError',
           'reason':'Diagnostic text extends original layout','outside_changed_pixels':90}
    assert pipeline.edit_hard_failures({'probes':[probe]},[('body','replace_long')])==[]
    probe['export_consistency']='FAIL'
    assert pipeline.edit_hard_failures({'probes':[probe]},[('body','replace_long')])


def qa_environment():
    return {'schema':'R7RunInputs/1','baseline':'frozen-input.json','baseline_sha256':'input-hash',
            'pages':[{'scene':'scene-hash','source_masks':{'mask':'mask-hash'}}],
            'fonts':{'font-hash':'abc'},'runtime':{'node':'runtime-hash'},'machine':'4090',
            'code':{'r7_edit_checks.py':'old-validator','r7_candidates.py':'unchanged'}}


def test_versioned_qa_allows_only_explicit_code_change():
    old=qa_environment();current=copy.deepcopy(old)
    current['code']['r7_edit_checks.py']='fixed-validator'
    with pytest.raises(ValueError,match='code changed'):
        pipeline.validate_qa_environment(old,current)
    record=pipeline.validate_qa_environment(old,current,allow_code_change=True)
    assert record['changed_code_files']==['r7_edit_checks.py']
    assert pipeline.validate_qa_environment(old,copy.deepcopy(old))['changed_code_files']==[]


@pytest.mark.parametrize('key',['baseline_sha256','pages','fonts','runtime','machine'])
def test_versioned_qa_never_relaxes_noncode_frozen_inputs(key):
    old=qa_environment();current=copy.deepcopy(old)
    current['code']['r7_edit_checks.py']='fixed-validator';current[key]='changed'
    with pytest.raises(ValueError,match='frozen'):
        pipeline.validate_qa_environment(old,current,allow_code_change=True)


def test_new_qa_saves_provenance_and_never_reconstructs_or_overwrites_parent(tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.mkdir()
    destination=tmp_path/'qa-v2';source=tmp_path/'input.json'
    source.write_text(json.dumps({'pages':[{'page':f'{i:02d}'} for i in range(1,13)]}),encoding='utf-8')
    deck=parent/'deck.pptx';deck.write_bytes(b'unchanged-parent-deck')
    batch={'input':str(source),'directory':str(parent),'pages':[{'page':f'{i:02d}'} for i in range(1,13)],
           'deck':{'pptx':str(deck)}}
    old=qa_environment();current=copy.deepcopy(old);current['code']['r7_edit_checks.py']='fixed-validator'
    (parent/'batch.json').write_text(json.dumps(batch),encoding='utf-8')
    (parent/'run-manifest.json').write_text(json.dumps(old),encoding='utf-8')
    before={name:(parent/name).read_bytes() for name in ('batch.json','run-manifest.json','deck.pptx')}
    monkeypatch.setattr(pipeline,'ArtifactBackend',lambda *a:object())
    monkeypatch.setattr(pipeline,'environment',lambda *a:current)
    monkeypatch.setattr(pipeline,'validate_completed_batch',lambda b:{'deck_sha256':pipeline.sha256(b['deck']['pptx'])})
    def audit(b,run,backend):
        assert Path(run)==destination and Path(b['directory'])==destination
        assert b['deck']==batch['deck'] and len(b['pages'])==12
        return {'status':'REVIEW_REQUIRED'}
    def package(b,qa,run):
        link=pipeline.read(Path(run)/'parent-link.json')
        assert link['original_reconstruction_code']==old['code']
        assert link['current_validation_code']==current['code']
        assert link['hashes']['deck_sha256']==pipeline.sha256(deck)
        assert link['reconstruction_rerun'] is False
        return {'pages':12,'deck_sha256':pipeline.sha256(deck)}
    monkeypatch.setattr(pipeline,'audit',audit);monkeypatch.setattr(pipeline,'package',package)
    monkeypatch.setattr(pipeline,'process_page',lambda *a:pytest.fail('QA must not reconstruct candidates'))
    result=pipeline.qa_run(parent,destination)
    assert result['pages']==12
    assert {name:(parent/name).read_bytes() for name in before}==before
    assert (destination/'parent-run-manifest.json').read_bytes()==before['run-manifest.json']
    assert (destination/'parent-batch.json').read_bytes()==before['batch.json']
    with pytest.raises(ValueError,match='new independent'):
        pipeline.qa_run(parent,destination)
    with pytest.raises(ValueError,match='new independent'):
        pipeline.qa_run(parent,parent/'nested-qa')


def test_completed_qa_rejects_changed_dependency_or_composed_page_order(tmp_path,monkeypatch):
    ir=tmp_path/'slideir.json';ir.write_text('{}',encoding='utf-8')
    pptx=tmp_path/'page.pptx';pptx.write_bytes(b'page')
    deck=tmp_path/'deck.pptx';deck.write_bytes(b'deck')
    source=tmp_path/'source-batch.json';quality=tmp_path/'quality.json'
    pages=[{'page':f'{i:02d}','ir':str(ir),'ir_sha256':pipeline.sha256(ir),
            'scene_sha256':'scene','pptx':str(pptx),'pptx_sha256':pipeline.sha256(pptx),
            'cache_dependency_hashes':{'frozen-mask':'original'}} for i in range(1,13)]
    source.write_text(json.dumps({'pages':pages}),encoding='utf-8')
    receipt={'slide_count':12,'pptx_sha256':pipeline.sha256(deck),
             'pages':[{'ir_sha256':p['ir_sha256']} for p in pages]}
    quality.write_text(json.dumps(receipt),encoding='utf-8')
    batch={'input':str(source),'pages':pages,'deck':{'pptx':str(deck),'quality':str(quality)}}
    monkeypatch.setattr(pipeline,'scene_fingerprint',lambda *a:{'sha256':'scene'})
    monkeypatch.setattr(pipeline,'checkpoint_files',lambda p:{'frozen-mask':'original'})
    assert pipeline.validate_completed_batch(batch)['deck_sha256']==pipeline.sha256(deck)
    monkeypatch.setattr(pipeline,'checkpoint_files',lambda p:{'frozen-mask':'tampered'})
    with pytest.raises(ValueError,match='render/spec/mask/receipt'):
        pipeline.validate_completed_batch(batch)
    monkeypatch.setattr(pipeline,'checkpoint_files',lambda p:{'frozen-mask':'original'})
    receipt['pages'][0]['ir_sha256']='wrong-order-or-scene'
    quality.write_text(json.dumps(receipt),encoding='utf-8')
    with pytest.raises(ValueError,match='page order/IR'):
        pipeline.validate_completed_batch(batch)
