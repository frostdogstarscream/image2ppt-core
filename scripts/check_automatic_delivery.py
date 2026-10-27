"""Verify saved automatic output without new inference or changing page records."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'services/vision/src')]
from image2ppt.storage import read_json, digest
from image2ppt.quality import inspect_pptx
from image2ppt_vision.auto_artifacts import verify_hashes
from image2ppt_vision.auto_provider import save_new, fingerprint
from image2ppt_vision.auto_protocol import validate_review


def verify_page_calls(directory,page,result,policy):
    if digest(directory/'source.png')!=page['source_sha256'] or digest(directory/'original.image')!=page['original_sha256']:
        raise ValueError('page source binding changed')
    calls=sorted((directory/'calls').glob('c[0-9][0-9]')); pending=[]; initial=0
    for call in calls:
        request=read_json(call/'request.json'); initial+=request['phase'].startswith('initial/')
        if not (call/'response.json').exists():
            pending.append(request['request']['request_id']);continue
        receipt=read_json(call/'receipt.json')
        if receipt['request_sha256']!=digest(call/'request.json') or receipt['response_sha256']!=digest(call/'response.json'):
            raise ValueError('model request/response receipt changed')
        if (call/'recovered-response.json').exists():
            recovered=read_json(call/'recovered-receipt.json')
            if (recovered['original_response_sha256']!=digest(call/'response.json') or
                recovered['request_sha256']!=digest(call/'request.json') or
                recovered['response_sha256']!=digest(call/'recovered-response.json')):
                raise ValueError('recovered response receipt changed')
    if len(calls)!=result['model_requests'] or len(calls)>policy['max_model_requests_per_page']:
        raise ValueError('page model-call budget/accounting mismatch')
    if policy.get('initial_call_limit') is not None and initial>policy['initial_call_limit']:
        raise ValueError('initial calls borrowed reserved review budget')
    return {'request_reservations':len(calls),'initial_requests':initial,'unreturned_request_ids':pending,
            'usage_disclosure':'Only received provider usage is summed; unavailable usage is not estimated.',
            'source_and_receipt_hashes_valid':True,'call_budget_valid':True}


def verify_v2_review(directory, revision, built_dir, built):
    from image2ppt_vision.auto_region_protocol import assemble
    from image2ppt_vision.auto_region_evidence import validate_visual, validate_probe
    from image2ppt_vision.auto_protocol import transform_model_boxes
    layout=read_json(directory/'layout.json')
    inventory_binding=read_json(directory/'inventory-binding.json')
    inventory=read_json(directory/inventory_binding['directory']/'inventory.json')
    regions=read_json(revision/'regions.json')
    plan,_,_=assemble(layout,inventory,regions['decisions'],regions['elements'])
    if fingerprint(plan)!=fingerprint(read_json(revision/'plan.json')):
        raise ValueError('v2 region execution differs from compiled plan')
    for binding_file in directory.rglob('*.binding.json'):
        binding=read_json(binding_file)
        target=binding_file.with_name(binding_file.name.replace('.binding.json','.json'))
        if binding.get('sha256') and digest(target)!=binding['sha256']:
            raise ValueError('v2 saved evidence hash mismatch: '+str(target))
    aggregate=built_dir/'reviews/aggregate.json'
    if not aggregate.exists(): return False
    for region in layout['regions']+[{'id':'page','bbox':[0,0,*inventory['image_size']]}]:
        rid=region['id']; is_page=rid=='page'
        path=built_dir/'reviews'/('page-visual.json' if is_page else rid+'-visual.json')
        review=read_json(path)
        review.pop('inherited_from',None); review.pop('program_repair',None)
        program=[i for i in review['issues'] if 'evidence_origin' in i]
        if program:
            # Program checks are appended after model validation and may only have failed a model pass.
            review['issues']=[i for i in review['issues'] if 'evidence_origin' not in i]
            if not review['issues'] and not review['redecision_required']: review['quality_passed']=True
        raw=transform_model_boxes(review,inventory['image_size'],to_model=True)
        raw.update(schema='RegionVisualReviewDraft/1',coordinate_space='normalized_1000_xyxy')
        decision_binding=(fingerprint(regions['decisions']) if is_page else regions['decisions'][rid]['binding'])
        validate_visual(raw,region=region,elements=plan if is_page else regions['elements'][rid],
                        binding=built['artifact_binding'],decision_binding=decision_binding,size=inventory['image_size'])
        if not is_page:
            targets=[tid for tid in built['probe_targets'] if tid in region['text_ids']]
            if targets:
                probe=read_json(built_dir/'reviews'/(rid+'-probe.json'));probe.pop('inherited_from',None); probe.pop('program_repair',None)
                validate_probe(probe,region_id=rid,binding=built['artifact_binding'],targets=targets)
    saved=read_json(aggregate)
    if saved['schema']!='AutomaticReview/2' or saved['artifact_binding']!=built['artifact_binding']:
        raise ValueError('v2 aggregate review binding mismatch')
    if saved['quality_passed'] and (not saved['edit_probe_clear'] or any(i['severity']=='blocking' for i in saved['issues'])):
        raise ValueError('v2 aggregate pass contradicts findings')
    return True


def verify_layered_page(root, name, page):
    """Re-derive a layered v3 page's claims from its saved files only."""
    import hashlib
    from image2ppt_vision.layered.pipeline import ocr_coverage_gaps
    from image2ppt_vision.layered.qa import residual_violations
    perception_dir, page_dir = root/'perception'/name, root/'pages'/name
    perception = read_json(perception_dir/'perception.json')
    if digest(perception_dir/'source.png') != perception['source_sha256'] or digest(root/'input'/(name+'.png')) != perception['input_sha256']:
        raise ValueError(name+': perception is not bound to the preserved input')
    ir = read_json(page_dir/'proposal'/'slideir.json')
    retained = {i for a in ir['assets'] for i in a['contains_evidence_ids']} | set(ir.get('unresolved_evidence_ids', []))
    uncovered, violations = ocr_coverage_gaps(perception, ir), residual_violations(perception, retained)
    semantics = read_json(page_dir/'semantics.json')
    if semantics.get('model_used'):
        response = read_json(page_dir/'semantics-response.json')
        if hashlib.sha256(response['content'].encode('utf-8')).hexdigest() != semantics['response_sha256']:
            raise ValueError(name+': semantics response hash mismatch')
        if digest_json(read_json(page_dir/'semantics-request.json')) != semantics['request_sha256']:
            raise ValueError(name+': semantics request hash mismatch')
    row = {'page': name, 'recorded_status': page['status'], 'valid_pptx': False, 'uncovered_ocr_ids': uncovered,
           'residual_text_violations': len(violations), 'semantics_model_used': bool(semantics.get('model_used')),
           'usage': semantics.get('usage', {}), 'human_acceptance': False, 'office_acceptance': False}
    if page.get('pptx'):
        pptx = page_dir/page['pptx']
        if digest(pptx) != page['pptx_sha256']:
            raise ValueError(name+': PPTX differs from the recorded build')
        objects = inspect_pptx(pptx)['slides'][0]['objects']
        native = {o['name'] for o in objects if 'text' in o}
        expected = {n['id'] for n in ir['nodes'] if n['type'] == 'text'}
        row.update(valid_pptx=True, pptx_sha256=page['pptx_sha256'], native_text_in_file=len(native & expected),
                   native_text_in_ir=len(expected), missing_native_text=sorted(expected-native),
                   metrics=page.get('metrics', {}))
    row['pipeline_status'] = ('FAILED' if not row['valid_pptx'] or uncovered or violations or row.get('missing_native_text')
                              else 'REVIEW_REQUIRED')
    return row


def digest_json(value):
    import hashlib, json
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def verify_layered(root, manifest):
    for relative, expected in manifest['files'].items():
        if relative != 'job.json' and digest(root/relative) != expected:
            raise ValueError('layered job file changed: '+relative)
    rows = [verify_layered_page(root, name, page) for name, page in manifest['pages'].items()]
    report = {'schema': 'LayeredEvidenceVerification/1', 'job_id': root.name, 'pages': rows,
              'total_model_tokens': manifest['total_model_tokens'], 'new_model_requests': 0,
              'modified_page_records': False, 'human_acceptance': False, 'office_acceptance': False,
              'caution': 'SSIM and OCR recall are comparison measurements, not visual acceptance.'}
    destination = root/'verification'
    save_new(destination/'verified-summary.json', report)
    lines = ['# 分层 v3 实际文件核验', '',
             '只读检查已保存的输入、感知结果、SlideIR、模型记录与实际 PPTX；没有重新推理。',
             '人工视觉验收及 PowerPoint/WPS 实机验收未完成。', '',
             '| 页面 | 状态 | 原生文字（文件/IR） | 未覆盖 OCR | 背景残留文字 | SSIM | OCR 字符召回 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        m = row.get('metrics', {})
        lines.append(f"| {row['page']} | {row['pipeline_status']} | {row.get('native_text_in_file', 0)}/{row.get('native_text_in_ir', 0)} | "
                     f"{len(row['uncovered_ocr_ids'])} | {row['residual_text_violations']} | {m.get('ssim', '')} | {m.get('ocr_char_recall', '')} |")
    (destination/'VERIFIED-RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(destination/'VERIFIED-RESULTS.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    args = parser.parse_args(); root = Path(args.job).resolve(strict=True)
    manifest = read_json(root/'job.json'); rows = []
    if manifest.get('schema') == 'LayeredJob/1':
        return verify_layered(root, manifest)
    for page in manifest['pages']:
        directory = root/page['id']; result_file = directory/'result.json'
        if not result_file.exists():
            continue
        result = read_json(result_file)
        call_proof=verify_page_calls(directory,page,result,manifest['policy'])
        row = {'page': page['id'], 'source_name': page['source_name'],
               'recorded_execution_status': result['execution_status'],
               'machine_quality_status': result['machine_quality_status'],
               'model_requests': result['model_requests'], 'usage': result['usage'],
               'valid_pptx': False, 'validated_model_review': False,
               'human_acceptance': False, 'office_acceptance': False, 'errors': result['errors'],
               'call_verification':call_proof}
        if result.get('build_directory'):
            built_dir = directory/result['build_directory']; built = read_json(built_dir/'result.json')
            verify_hashes(built_dir, built['files'])
            packet = built_dir/'evidence'
            if digest(root/'delivery'/(page['id']+'.pptx')) != digest(packet/'result.pptx'):
                raise ValueError('delivery PPTX differs from selected build')
            original = inspect_pptx(packet/'result.pptx'); edited = inspect_pptx(packet/'delete-probe.pptx')
            before = {o['name']:o for o in original['slides'][0]['objects']}
            after = {o['name']:o for o in edited['slides'][0]['objects']}
            probes = built['probe_targets']
            checks = {tid: {'native_before': 'text' in before.get(tid, {}),
                            'absent_after_delete': tid not in after,
                            'text': before.get(tid, {}).get('text')} for tid in probes}
            without_index = lambda record: {k:v for k,v in record.items() if k != 'z'}
            unchanged = ({k:without_index(v) for k,v in before.items() if k not in probes} ==
                         {k:without_index(v) for k,v in after.items()} and
                         [k for k in before if k not in probes] == list(after))
            if not unchanged or not all(v['native_before'] and v['absent_after_delete'] for v in checks.values()):
                raise ValueError('actual-file edit probe did not preserve non-target objects')
            revision = directory/result['selected_revision']
            review_file = revision/'review.json'
            if manifest.get('pipeline_version')==2:
                row['validated_model_review']=verify_v2_review(directory,revision,built_dir,built)
                row['regions']=result.get('regions',[])
            elif review_file.exists():
                validate_review(read_json(review_file), built['artifact_binding'], read_json(revision/'plan.json'))
                row['validated_model_review'] = True
            row.update(valid_pptx=True, selected_revision=result['selected_revision'],
                       counts=built['counts'], probe_checks=checks, other_objects_unchanged=unchanged,
                       structural_issues=built['structural_issues'], artifact_binding=built['artifact_binding'],
                       visual_fidelity='REVIEW_REQUIRED', pptx_sha256=digest(packet/'result.pptx'))
        row['pipeline_status'] = ('COMPLETED' if row['validated_model_review'] else
                                  'PARTIAL' if row['valid_pptx'] else 'FAILED')
        rows.append(row)
    report = {'schema':'AutomaticEvidenceVerification/1', 'job_id':root.name, 'pages':rows,
              'new_model_requests':0, 'modified_page_records':False,
              'human_acceptance':False, 'office_acceptance':False,
              'caution':'A valid model-review schema does not establish that its visual claims are correct.'}
    destination = root/'delivery'
    save_new(destination/'verified-summary.json', report)
    lines = ['# 实际文件核验结果', '',
             '本报告只读检查已保存的文件与模型记录，没有重新推理，也没有修改原方案或复检答复。',
             '质量与执行状态分别列示；人工视觉验收及 PowerPoint/WPS 实机验收未完成。', '',
             '| 页面 | 真实文件 / 链路 | 原生文字 / 形状 / 图片 | 模型请求 | 质量 |',
             '|---|---|---|---:|---|']
    for row in rows:
        counts = row.get('counts', {})
        lines.append(f"| {row['page']} | {row['pipeline_status']} | {counts.get('native_text',0)} / {counts.get('native_shapes',0)} / {counts.get('raster_assets',0)} | {row['model_requests']} | {row['machine_quality_status']} |")
    lines += ['', '`PARTIAL` 表示已生成实际 PPTX，但没有通过格式与版本绑定校验的模型复检。',
              '上表依据实际复检文件及绑定校验确定链路状态，原记录保留。', '',
              '模型复检意见与 OOXML 检查分开看待。特别是“缺失／不可编辑／重影”的模型判断不能代替实际文件证据。']
    for row in rows:
        lines += ['', '## '+row['page']]
        if row['valid_pptx']:
            pid = row['page']
            lines += [f'[{pid}.pptx]({pid}.pptx) · [实际预览]({pid}-preview.png) · [原图对照]({pid}-comparison.png) · [删除探针]({pid}-delete-probe.png)', '',
                      f"原生文字 {row['counts']['native_text']}，保留在图片内的 OCR 文字 {row['counts']['retained_text']}。",
                      '以下目标在实际 PPTX 中为原生文字，删除副本中已不存在；其余对象保持不变：']
            for tid, check in row['probe_checks'].items():
                lines.append('- '+tid+'：'+str(check['text']).replace('\n',' / '))
        else:
            lines += ['没有有效 PPTX。']
        lines += ['- 执行记录：'+e['code'] for e in row['errors']]
    (destination/'VERIFIED-RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(destination/'VERIFIED-RESULTS.md')


if __name__ == '__main__': main()
