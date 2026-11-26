from copy import deepcopy

import pytest

from product.materialize import digest, write
from product.runtime import host_baseline as host


def measured_score(deck='deck', source='source'):
    return {
        'verified': True, 'office_status': 'PASS', 'pptx_sha256': deck,
        'pages': [{
            'page': 1, 'source_sha256': source, 'editable_count': 1, 'native_field_count': 1,
            'selected_editables': ['t-1'], 'passed_edits': ['t-1/move-scale', 't-1/delete'],
            'ocr_native_hits': ['ocr-1'], 'max_local_mae': 0, 'unscoped_crops': [],
            'plate_audit': {'status': 'MEASURED', 'source_sha256': source, 'pptx_sha256': deck,
                            'rows': [{'status': 'plate_only', 'text': 'O', 'bbox': [2, 3, 10, 10]}]},
        }],
    }


@pytest.mark.parametrize('bad', ['no_office', 'wrong_deck', 'wrong_plate', 'wrong_source',
                                 'missing_page', 'missing_edit', 'raster_only', 'whole_page_crop'])
def test_partial_copy_requires_bound_edits_on_every_page(bad):
    score = measured_score()
    last = deepcopy(score['pages'][0])
    last['page'] = 2
    score['pages'].append(last)
    if bad == 'no_office':
        score['verified'] = False
    elif bad == 'wrong_deck':
        score['pptx_sha256'] = 'other'
    elif bad == 'wrong_plate':
        last['plate_audit']['pptx_sha256'] = 'other'
    elif bad == 'wrong_source':
        last['source_sha256'] = 'other'
    elif bad == 'missing_page':
        score['pages'].pop()
    elif bad == 'missing_edit':
        last['passed_edits'] = ['t-1/move-scale']
    elif bad == 'raster_only':
        last['selected_editables'] = []
    else:
        last['unscoped_crops'] = ['r-whole-slide']
    assert not host._verified_partial_copy(score, 'deck', ['source', 'source'], host.delivery_defects(score))


@pytest.fixture
def frozen_job(tmp_path, monkeypatch):
    monkeypatch.setenv('IMAGE2PPT_RESOURCE_ROOT', str(tmp_path/'leases'))
    job = tmp_path/'job'
    frozen = job/'host-baseline'
    frozen.mkdir(parents=True)
    source = job/'inputs/1.png'
    source.parent.mkdir()
    source.write_bytes(b'fixture source')
    (frozen/'baseline.pptx').write_bytes(b'fixture deck with mocked Office receipts')
    deck_sha = digest(frozen/'baseline.pptx')
    score = measured_score(deck_sha, digest(source))
    write(job/'request.json', {'sources': ['1.png'], 'object_scoped_delivery': True})
    write(frozen/'scorecard.json', score)
    write(frozen/'state.json', {'version': 0, 'composition': {'office': {'office_directory': str(frozen)}}})
    write(frozen/'manifest.json', {'sources': [digest(source)], 'files': {
        name: digest(frozen/name) for name in ('state.json', 'scorecard.json', 'baseline.pptx')}})
    return job


def test_verified_incomplete_package_remains_reviewable_after_restart(frozen_job):
    job = frozen_job
    result = host.publish(job, {'reason': 'test timeout'}, ['TIME_BUDGET_EXHAUSTED'])
    assert result['status'] == 'REVIEW_REQUIRED' and result['completion'] == 'PARTIAL_EDITABLE'
    assert not result['automatic_publication'] and result['office_pass']
    assert result['failed_pages'][0]['retained_fields'] == measured_score()['pages'][0]['plate_audit']['rows']
    assert result['artifacts']['result.pptx'] == digest(job/'host-baseline/baseline.pptx') == digest(job/'delivery/result.pptx')
    assert host.selected_result(job) == result


@pytest.mark.parametrize('bound', [True, False])
def test_incomplete_agent_package_requires_its_own_office_bound_bytes(frozen_job, bound):
    job = frozen_job
    staged = job/'runtime/agent-delivery/result.pptx'
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b'agent package fixture')
    sha = digest(staged)
    candidate = measured_score(sha if bound else 'unrelated', digest(job/'inputs/1.png'))
    result = host.publish(job, {'status': 'SUCCEEDED', 'artifacts': {'result.pptx': sha}},
                          candidate=candidate)
    assert result['status'] == 'REVIEW_REQUIRED' and result['completion'] == 'PARTIAL_EDITABLE'
    expected = staged if bound else job/'host-baseline/baseline.pptx'
    assert result['artifacts']['result.pptx'] == digest(expected)
    assert result['agent']['delivered_changes'] is bound
    if not bound:
        assert 'AGENT_PACKAGE_HASH_MISMATCH' in result['agent']['fallback_reasons']


def test_new_plate_word_keeps_the_verified_baseline_download(frozen_job):
    job = frozen_job
    deck = job/'runtime/agent-delivery/result.pptx'
    deck.parent.mkdir(parents=True)
    deck.write_bytes(b'candidate with additional visible wording')
    candidate = measured_score(digest(deck), digest(job/'inputs/1.png'))
    rows = candidate['pages'][0]['plate_audit']['rows']
    rows[0]['bbox'][0] += 1  # A small measured shift still identifies the same character.
    rows.append({**rows[0], 'bbox':[80, 3, 10, 10]})
    result = host.publish(job, {'status':'SUCCEEDED', 'artifacts':{'result.pptx':digest(deck)}},
                          candidate=candidate)
    assert result['artifacts']['result.pptx'] == digest(job/'host-baseline/baseline.pptx')
    assert result['completion'] == 'PARTIAL_EDITABLE'


def test_timeout_publishes_checked_deck_during_baseline_refinement(frozen_job, monkeypatch):
    from product.runtime import runner
    job = frozen_job
    (job/'host-baseline').rename(job/'host-checkpoint')
    request = host.read(job/'request.json')
    write(job/'request.json', {**request, 'host_stage0':True})
    expected = host.checkpoint_file(job)[1]
    def expired(job):
        raise TimeoutError('later frame work timed out')
    monkeypatch.setattr(host,'ensure_baseline',expired)
    result = runner.run(job)
    assert result['status']=='REVIEW_REQUIRED' and result['checkpoint_delivery']
    assert result['artifacts']['result.pptx']==expected
    assert runner.run(job)==result
    (job/'host-checkpoint/baseline.pptx').write_bytes(b'changed')
    assert host.checkpoint_file(job) is None


def test_unlocated_hidden_view_reading_stays_reviewable_without_blame_on_plate():
    score = measured_score()
    row = score['pages'][0]['plate_audit']['rows'][0]
    row['residual_location'] = 'text_hidden_view_only'
    defects = host.delivery_defects(score)
    assert defects[0]['reasons']==['READING_IN_TEXT_HIDDEN_VIEW']
    assert defects[0]['retained_fields']==[row]
    assert host._verified_partial_copy(score,'deck',['source'],defects)


def test_later_ineligible_baseline_keeps_earlier_verified_download(frozen_job):
    import shutil
    root = frozen_job/'host-baseline'
    shutil.copytree(root,frozen_job/'host-checkpoint')
    score = host.read(root/'scorecard.json')
    score['verified'] = False
    write(root/'scorecard.json',score)
    manifest = host.read(root/'manifest.json')
    manifest['files']['scorecard.json'] = digest(root/'scorecard.json')
    write(root/'manifest.json',manifest)
    result = host.publish(frozen_job, {'reason':'refinement unavailable'}, ['BASELINE_ONLY'])
    assert result['checkpoint_delivery'] and result['office_pass']
    assert result['artifacts']['result.pptx']==host.checkpoint_file(frozen_job)[1]
