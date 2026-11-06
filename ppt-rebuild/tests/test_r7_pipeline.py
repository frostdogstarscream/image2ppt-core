"""R7 integration gates: metadata freeze and final failure aggregation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from PIL import Image

import numpy as np
import pytest
from _helpers import SRC
import r7_pipeline as pipeline
import r7_edit_checks
from object_resolver import RepairAuthorizationError
from reconstruction_transaction import scene_fingerprint


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
            'expected_probes':[['body','replace_short']], 'visual':{}} for i in range(1,13)]
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
    monkeypatch.setattr(pipeline,'check_relations',lambda *a:{'status':'PASS'})
    calls=[]
    def edit_page(page,*args,**kwargs):
        calls.append((page['page'],kwargs))
        if page['page'].startswith('03'):
            return {'status':'REVIEW_REQUIRED','probes':[]}  # required object/probe vanished
        probe=valid_probe()
        if page['page'].startswith('04'):
            probe.update(status='REVIEW_REQUIRED',clean_plate_correctness={
                'status':'REVIEW_REQUIRED','reason':'independent_plate_missing'})
        if page['page'].startswith('05'):
            probe.update(status='FAIL', protected_pixels='FAIL', outside_changed_pixels=1,
                         clean_plate_correctness={'status':'REVIEW_REQUIRED','reason':'independent_plate_missing'})
        return {'status':probe['status'],'probes':[probe]}
    monkeypatch.setattr(r7_edit_checks,'audit_page',edit_page)
    monkeypatch.setattr(r7_edit_checks,'probe_specs',lambda *a:pytest.fail('Frozen inventory must not be re-derived'))
    result=pipeline.audit(batch,tmp_path,Backend())
    assert len(result['pages'])==12
    assert result['hard_failure_pages']==['03_page','05_page']
    assert result['status']=='FAIL'
    assert result['pages'][3]['status']=='REVIEW_REQUIRED'
    assert result['pages'][0]['status']=='PASS'
    assert result['pages'][4]['status']=='FAIL'
    pending=[item for item in result['pages'][4]['review_required'] if item.get('reason')=='independent_plate_missing']
    assert len(pending)==1
    assert pending[0]['path'][-1]=='clean_plate_correctness'
    assert '05_page' in result['review_required_pages']
    assert [kwargs['slide_number'] for _,kwargs in calls]==list(range(1,13))
    assert all(kwargs['deck_pptx']=='fake-deck' and kwargs['expected_probes']==[['body','replace_short']]
               and len(kwargs['baseline_deck_render']['pages'])==12 for _,kwargs in calls)


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
    assert pipeline.edit_hard_failures({'probes':[probe]},[('body','replace_long')])
    probe=valid_probe('replace_long')
    probe.update(overflow_detected=True,outside_changed_pixels=90,protected_pixels='FAIL',
                 capacity_overflow={'verified':True,'reason':'UNSUPPORTED_REFLOW',
                                    'outside_capacity_pixels':0,'protected_overlap_pixels':0})
    assert pipeline.edit_hard_failures({'probes':[probe]},[('body','replace_long')])==[]
    for change in ({'export_consistency':'FAIL'}, {'error_type':'PermissionError'},
                   {'protected_xml':'FAIL'}, {'protected_overlap_pixels':1}):
        broken={**probe,**change}
        assert pipeline.edit_hard_failures({'probes':[broken]},[('body','replace_long')])


def valid_probe(action='replace_short'):
    return {'target':'body','action':action,'status':'PASS','export_consistency':'PASS',
            'protected_pixels':'PASS','outside_changed_pixels':0,'actual_changed_pixels':100,
            'protected_xml':'PASS','package_parts':'PASS','overflow_detected':False,
            'run_styles_preserved':'PASS','neighbor_evidence':{'status':'PASS'},'clean_plate_correctness':{'status':'PASS'}}


def test_short_overflow_inside_old_footprint_still_fails():
    probe=valid_probe();probe['overflow_detected']=True
    assert pipeline.edit_hard_failures({'probes':[probe]},[('body','replace_short')])


@pytest.mark.parametrize('state',['FAIL','nonsense',None])
def test_explicit_failure_or_malformed_probe_cannot_disappear(state):
    probe=valid_probe();probe['status']=state
    assert pipeline.edit_hard_failures({'probes':[probe]},[('body','replace_short')])


@pytest.mark.parametrize('state',['UNKNOWN','REVIEW_REQUIRED'])
def test_unknown_probe_remains_pending_without_becoming_hard_failure(state):
    probe=valid_probe();probe['status']=state
    result=pipeline.summarize_edit_report({'probes':[probe]},[('body','replace_short')])
    assert result['status']=='REVIEW_REQUIRED' and not result['failures']


def test_missing_frozen_probe_is_hard_failure_even_if_current_ir_has_no_target():
    result=pipeline.summarize_edit_report({'probes':[]},[('body','replace_short')])
    assert result['status']=='FAIL'


def test_container_authorization_only_permits_exact_vertical_geometry_change():
    before=baseline();after=copy.deepcopy(before)
    after['nodes'][0]['bbox']=[2,2,5,6]
    permission={'text':{'bbox':{'before':[2,3,5,4],'after':[2,2,5,6]}}}
    assert pipeline.frozen_scene(before,after,allowed_fields=permission)['status']=='PASS'
    with pytest.raises(ValueError):pipeline.frozen_scene(before,after)
    after['nodes'][0]['font']['size_pt']=9
    with pytest.raises(ValueError):pipeline.frozen_scene(before,after,allowed_fields=permission)
    after=copy.deepcopy(before);after['nodes'][0]['bbox']=[2,2,6,6]
    permission['text']['bbox']['after']=[2,2,6,6]
    with pytest.raises(ValueError,match='vertical'):pipeline.frozen_scene(before,after,allowed_fields=permission)


def qa_environment():
    return {'schema':'R7RunInputs/1','baseline':'frozen-input.json','baseline_sha256':'input-hash',
            'pages':[{'scene':'scene-hash','source_masks':{'mask':'mask-hash'}}],
            'fonts':{'font-hash':'abc'},'runtime':{'node':'runtime-hash'},'machine':'4090',
            'code':{'ppt-rebuild/src/r7_edit_checks.py':'old-validator','ppt-rebuild/src/r7_candidates.py':'unchanged'}}


def test_versioned_qa_allows_only_explicit_code_change():
    old=qa_environment();current=copy.deepcopy(old)
    current['code']['ppt-rebuild/src/r7_edit_checks.py']='fixed-validator'
    with pytest.raises(ValueError,match='code changed'):
        pipeline.validate_qa_environment(old,current)
    record=pipeline.validate_qa_environment(old,current,allow_code_change=True)
    assert record['changed_code_files']==['ppt-rebuild/src/r7_edit_checks.py']
    assert pipeline.validate_qa_environment(old,copy.deepcopy(old))['changed_code_files']==[]


@pytest.mark.parametrize('path',['scripts/compile.mjs','scripts/compose.mjs',
                                'ppt-rebuild/src/slideir_adapter.py','ppt-rebuild/src/r7_candidates.py',
                                'ppt-rebuild/src/r7_pipeline.py'])
def test_versioned_qa_rejects_generation_code_changes(path):
    old=qa_environment();current=copy.deepcopy(old)
    current['code'][path]='changed'
    with pytest.raises(ValueError,match='generation dependency'):
        pipeline.validate_qa_environment(old,current,allow_code_change=True)


@pytest.mark.parametrize('key',['baseline_sha256','pages','fonts','runtime','machine'])
def test_versioned_qa_never_relaxes_noncode_frozen_inputs(key):
    old=qa_environment();current=copy.deepcopy(old)
    current['code']['ppt-rebuild/src/r7_edit_checks.py']='fixed-validator';current[key]='changed'
    with pytest.raises(ValueError,match='frozen'):
        pipeline.validate_qa_environment(old,current,allow_code_change=True)


def test_new_qa_saves_provenance_and_never_reconstructs_or_overwrites_parent(tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.mkdir()
    destination=tmp_path/'qa-v2';source=tmp_path/'input.json'
    source.write_text(json.dumps({'pages':[{'page':f'{i:02d}'} for i in range(1,13)]}),encoding='utf-8')
    deck=parent/'deck.pptx';deck.write_bytes(b'unchanged-parent-deck')
    batch={'input':str(source),'directory':str(parent),'pages':[{'page':f'{i:02d}'} for i in range(1,13)],
           'deck':{'pptx':str(deck)}}
    old=qa_environment();current=copy.deepcopy(old);current['code']['ppt-rebuild/src/r7_edit_checks.py']='fixed-validator'
    (parent/'batch.json').write_text(json.dumps(batch),encoding='utf-8')
    (parent/'run-manifest.json').write_text(json.dumps(old),encoding='utf-8')
    before={name:(parent/name).read_bytes() for name in ('batch.json','run-manifest.json','deck.pptx')}
    monkeypatch.setattr(pipeline,'ArtifactBackend',lambda *a:object())
    monkeypatch.setattr(pipeline,'load_baseline',lambda path:pipeline.read(path))
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


def page_fixture(tmp_path,monkeypatch,states,*,prefix='02'):
    """Real immutable scene files; substitute only renderer and relation sensors."""
    parent=tmp_path/'parent';parent.mkdir()
    Image.new('RGB',(20,15),'white').save(parent/'source.png')
    ir=baseline();ir['source_sha256']=pipeline.sha256(parent/'source.png')
    irfile=parent/'slideir.json';irfile.write_text(json.dumps(ir),encoding='utf-8')
    pptx=parent/'page.pptx';pptx.write_bytes(b'frozen-pptx')
    preview=parent/'preview.png';Image.new('RGB',(20,15),'white').save(preview)
    refs={'ir':{'path':str(irfile),'sha256':pipeline.sha256(irfile)},
          'pptx':{'path':str(pptx),'sha256':pipeline.sha256(pptx)},
          'preview':{'path':str(preview),'sha256':pipeline.sha256(preview)}}
    page={'page':prefix+'_page','ir':str(irfile),'pptx':str(pptx),'preview':str(preview),
          'directory':str(parent),'source_sha256':ir['source_sha256'],
          'ir_sha256':pipeline.sha256(irfile),'pptx_sha256':pipeline.sha256(pptx),
          'baseline_refs':{'source':{'path':str(parent/'source.png'),'sha256':ir['source_sha256']},
                           'r6':refs,'r7':refs,'current':refs},
          'expected_probes':[['text','replace_short']], 'evidence_baseline':{'directory':str(parent)}}
    def prepare(prefix,source,out,**kwargs):
        out.mkdir(parents=True)
        spec={'source_sha256':pipeline.sha256(source),'page':prefix,'canvas_px':[20,15],
              'elements':[{'id':identifier,'repairable':True,'masks':{}} for identifier in states]}
        pipeline.write_json(out/'relations.json',spec)
        return spec
    def compile_(backend,path):
        out=Path(path).parent/'compiled';out.mkdir(exist_ok=True)
        (out/'page.pptx').write_bytes(b'recompiled-frozen-pptx')
        Image.new('RGB',(20,15),'white').save(out/'preview.png')
        return {'pptx':str(out/'page.pptx'),'preview':str(out/'preview.png')}
    def delete_(delivery,out,backend):
        out.mkdir(parents=True)
        (out/'deleted.pptx').write_bytes(b'deleted')
        Image.new('RGB',(20,15),'white').save(out/'deleted.png')
        return str(out/'deleted.pptx'),str(out/'deleted.png')
    def relation(*args,**kwargs):
        rows=[{'id':identifier,'status':state} for identifier,state in states.items()]
        return {'status':'PASS' if all(s=='PASS' for s in states.values()) else 'REVIEW_REQUIRED',
                'objects':rows}
    monkeypatch.setattr(pipeline,'prepare_spec',prepare)
    monkeypatch.setattr(pipeline,'compile_ir',compile_)
    monkeypatch.setattr(pipeline,'check_relations',relation)
    monkeypatch.setattr(pipeline,'deletion',delete_)
    monkeypatch.setattr(pipeline,'clean_check',lambda *a,**kw:{'status':'REVIEW_REQUIRED','reason':'fixture_evidence_missing'})
    monkeypatch.setattr(pipeline,'metrics',lambda *a,**kw:{'mae':0})
    return page


@pytest.mark.parametrize('prefix',['02','08','03'])
def test_approved_page_repeated_in_new_runs_is_noop_without_transaction(tmp_path,monkeypatch,prefix):
    page=page_fixture(tmp_path,monkeypatch,{'approved':'PASS'},prefix=prefix)
    before=scene_fingerprint(page['ir'])
    def forbidden(*a,**kw):pytest.fail('Satisfied input must not construct candidates or transactions')
    monkeypatch.setattr(pipeline,'build_candidate',forbidden)
    monkeypatch.setattr(pipeline,'ReconstructionTransaction',forbidden)
    first=pipeline.process_page(page,tmp_path/'first-run',object())
    second=pipeline.process_page(first,tmp_path/'second-run',object())
    assert first['action']==second['action']=='NO_OP'
    assert first['attempts']==second['attempts']==[]
    assert scene_fingerprint(first['ir'])==scene_fingerprint(second['ir'])==before
    assert first['expected_probes']==second['expected_probes']==[['text','replace_short']]
    assert not list((tmp_path/'first-run').rglob('prepare.json'))
    assert not list((tmp_path/'second-run').rglob('prepare.json'))


def test_mixed_page_only_attempts_failed_object_and_records_missing_authorization(tmp_path,monkeypatch):
    states={'approved':'PASS','broken':'FAIL','unknown':'UNKNOWN'}
    page=page_fixture(tmp_path,monkeypatch,states)
    calls=[]
    def builder(ir,spec,out,**kwargs):
        calls.append(kwargs)
        raise RepairAuthorizationError('independent_neighbor_protection_missing','REVIEW_REQUIRED')
    monkeypatch.setattr(pipeline,'build_candidate',builder)
    monkeypatch.setattr(pipeline,'ReconstructionTransaction',lambda *a,**kw:pytest.fail('Unauthorized candidate must not start transaction'))
    result=pipeline.process_page(page,tmp_path/'run',object())
    assert len(calls)==1 and calls[0]['object_ids']==['broken']
    assert calls[0]['tx_id'] and calls[0]['parent_scene_sha256']==scene_fingerprint(page['ir'])['sha256']
    assert result['action']=='BASELINE_RETAINED'
    assert result['attempts'][0]['status']=='REVIEW_REQUIRED'
    assert {row['id']:row['action'] for row in result['object_decisions']}=={
        'approved':'NO_OP','broken':'REPAIR','unknown':'REVIEW_REQUIRED'}
    assert scene_fingerprint(result['ir'])==scene_fingerprint(page['ir'])


def test_missing_qa_measurement_keeps_expected_inventory_for_final_gate(tmp_path,monkeypatch):
    page=page_fixture(tmp_path,monkeypatch,{'unknown':'UNKNOWN'})
    monkeypatch.setattr(pipeline,'build_candidate',lambda *a,**kw:pytest.fail('Unknown is not repair permission'))
    result=pipeline.process_page(page,tmp_path/'run',object())
    assert result['expected_probes']==page['expected_probes']
    assert result['action']=='BASELINE_RETAINED' and result['attempts']==[]


def clean_fixture(tmp_path, monkeypatch, *, expected_ids=('p',)):
    directory=tmp_path/'source-evidence';directory.mkdir()
    masks=directory/'text-masks';masks.mkdir()
    Image.new('L',(20,15),255).save(masks/'p-glyph.png')
    Image.new('L',(20,15),255).save(masks/'p-inspection.png')
    source=np.full((15,20,3),255,np.uint8);source[4:10,4:6]=0
    Image.fromarray(source).save(directory/'source.png')
    Image.fromarray(source).save(directory/'rendered.png')
    ir=baseline();(directory/'slideir.json').write_text(json.dumps(ir),encoding='utf-8')
    page={'ir':str(directory/'slideir.json'),'directory':str(directory),
          'text_checks':[{'id':identifier} for identifier in expected_ids]}
    monkeypatch.setattr(pipeline,'detect',lambda *a,**k:[])
    return page,directory/'source.png',directory/'rendered.png'


def test_clean_check_real_unchanged_source_strokes_fail_numeric_residue_gate(tmp_path,monkeypatch):
    page,source,rendered=clean_fixture(tmp_path,monkeypatch)
    result=pipeline.clean_check(page,source,rendered,tmp_path/'qa')
    assert result['status']=='FAIL' and not result['verified_target_ids']
    measured=result['paragraphs'][0]
    assert measured['checked_pixels']>0 and measured['residual_fraction']>.01
    assert measured['reason']=='SOURCE_STROKE_RESIDUE_LIMIT_EXCEEDED'
    assert measured['prior_status']=='REVIEW_REQUIRED' and measured['observed']==measured['residual_fraction']


@pytest.mark.parametrize('measurement,expected',[
    ({'status':'REVIEW_REQUIRED','reason':'no_original_glyph_evidence'},'REVIEW_REQUIRED'),
    ({'status':'REVIEW_REQUIRED','reason':'insufficient_independent_stroke_evidence','checked_pixels':0},'REVIEW_REQUIRED'),
    ({'status':'PASS','checked_pixels':12,'residual_fraction':.01},'PASS'),
    ({'status':'FAIL','reason':'explicit_measurement_failure'},'FAIL'),
])
def test_clean_check_preserves_evidence_absence_boundary_and_explicit_failure(tmp_path,monkeypatch,measurement,expected):
    page,source,rendered=clean_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(pipeline,'inspect_residue',lambda *a:copy.deepcopy(measurement))
    result=pipeline.clean_check(page,source,rendered,tmp_path/'qa')
    assert result['status']==expected
    assert bool(result['verified_target_ids'])==(expected=='PASS')
    assert 'prior_status' not in result['paragraphs'][0]


def test_clean_check_ocr_only_hit_is_pending_without_hard_failure(tmp_path,monkeypatch):
    page,source,rendered=clean_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(pipeline,'inspect_residue',lambda *a:{'status':'PASS','checked_pixels':12,'residual_fraction':0})
    monkeypatch.setattr(pipeline,'detect',lambda *a,**k:[SimpleNamespace(text='uncorroborated OCR',bbox=[2,3,5,4])])
    result=pipeline.clean_check(page,source,rendered,tmp_path/'qa')
    findings=pipeline.aggregate_checks({'clean':result})
    assert result['status']=='REVIEW_REQUIRED' and result['ocr_hits']
    assert not findings['failures'] and len(findings['review_required'])==1


def test_clean_check_reports_missing_coverage_even_when_measured_residue_fails(tmp_path,monkeypatch):
    page,source,rendered=clean_fixture(tmp_path,monkeypatch,expected_ids=('p','missing'))
    result=pipeline.clean_check(page,source,rendered,tmp_path/'qa')
    findings=pipeline.aggregate_checks({'clean':result})
    assert result['status']=='FAIL' and result['mask_coverage']['missing']==['missing']
    assert len(findings['failures'])==1 and len(findings['review_required'])==1


@pytest.mark.parametrize('artifact',['authorization.json','transactions/cover-container-1/acceptance.json','short-probe/report.json'])
def test_cover_evidence_is_checkpointed_and_tampering_rejects_resume(tmp_path,monkeypatch,artifact):
    page=page_fixture(tmp_path,monkeypatch,{'approved':'PASS'},prefix='02')
    run=tmp_path/'run'
    result=pipeline.process_page(page,run,object())
    covered=Path(result['execution_directory'])/'cover-container'/artifact
    covered.parent.mkdir(parents=True,exist_ok=True);covered.write_text('{"verified":"original"}',encoding='utf-8')
    result['cache_dependency_hashes']=pipeline.checkpoint_files(result)
    assert str(covered.resolve()) in result['cache_dependency_hashes']
    pipeline._atomic_write(run/'pages'/'02'/'result.json',result)
    assert pipeline.process_page(page,run,object())['cache_dependency_hashes']==result['cache_dependency_hashes']
    covered.write_text('{"verified":"changed"}',encoding='utf-8')
    with pytest.raises(ValueError,match='render/spec/mask/receipt changed'):
        pipeline.process_page(page,run,object())
