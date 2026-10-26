"""Region rebuild planning in shadow mode.

The planner only proposes; it never edits an IR or a deck. Inventory measures a
retained raster region, suggest records an evidence-backed proposal, and score
compares that proposal with the currently accepted decision. The schema and
validator live next to this script; no model API is called here.
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = Path(__file__).with_name('region_plan.schema.json')
sys.path.insert(0, str(ROOT / 'services/vision/src'))
from image2ppt_vision.region_plan import (
    HUMAN_SCOPE_KEYS, apply_jobs, prepare_region_job, validate_decision, validate_human_decision,
)
from image2ppt_vision.region_features import features as source_region_features


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_human_decisions(path):
    data = read_json(path)
    decisions = data.get('decisions', data) if isinstance(data, dict) else data
    if not isinstance(decisions, dict):
        raise ValueError('invalid human decisions')
    result = {}
    for region_id, value in decisions.items():
        try:
            result[region_id] = validate_human_decision(value)
        except ValueError as exc:
            raise ValueError(f'{region_id}: {exc}') from exc
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def features(crop):
    return source_region_features(crop)


def classify(description):
    text = description or ''
    if any(word in text for word in ('仪表盘', '驾驶舱', '看板')):
        return 'dashboard'
    if any(word in text for word in ('关系图', '关系网络', '线索关联', '关联研判')):
        return 'network'
    if any(word in text for word in ('条形图', '柱状图', '饼图', '趋势图')):
        return 'chart'
    if any(word in text for word in ('表格', '清单', '台账')):
        return 'table'
    if '地图' in text:
        return 'photo_map'
    if any(word in text for word in ('照片', '图示', '示意')):
        return 'photo_diagram'
    if any(word in text for word in ('软件界面', '界面', '平台')):
        return 'ui_mockup'
    if any(word in text for word in ('流程', '步骤', '箭头')):
        return 'flow'
    return 'mixed'


def inventory(regions, output, human_decisions=None):
    output = resolve(output)
    decisions = load_human_decisions(human_decisions) if human_decisions else {}
    crops = output / 'crops'
    crops.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in read_json(regions):
        ir_path = resolve(item['ir'])
        ir = read_json(ir_path)
        node = next((value for value in ir['nodes'] if value['id'] == item['node_id']), None)
        if node is None or node.get('type') != 'image':
            raise ValueError('image node not found: ' + item['node_id'])
        asset = next(value for value in ir['assets'] if value['id'] == node['asset_id'])
        source = ir_path.parent / ir['source_image']
        x, y, width, height = [int(round(value)) for value in node['bbox']]
        with Image.open(source) as image:
            crop = image.convert('RGB').crop((x, y, x + width, y + height))
        crop_path = crops / (item['region_id'] + '.png')
        crop.save(crop_path)
        measured = features(crop)
        human = decisions.get(item['region_id'], {})
        rows.append({
            'region_id': item['region_id'],
            'page': item['page'],
            'ir': Path(item['ir']).as_posix(),
            'node_id': item['node_id'],
            'bbox': node['bbox'],
            'classification': item.get('classification') or classify(asset.get('description', '')),
            'description': asset.get('description', ''),
            'crop': crop_path.relative_to(output).as_posix(),
            'source_image': Path(ir['source_image']).as_posix(),
            'existing_decision': human.get('decision', item.get('existing_decision', 'retain_module')),
            'human_note': human.get('note', item.get('human_note', '当前已验收 deck 的选择；正式人工复核可覆盖。')),
            'features': measured,
        })
    result = {'schema_version': 'region-inventory/0.1', 'regions': rows}
    write_json(output / 'inventory.json', result)
    return result


def suggestion(row):
    kind = row['classification']
    f = row['features']
    microtext = f['median_text_height_px'] and f['median_text_height_px'] < 10
    many_parts = f['text_components'] > 90
    common = {
        'confidence': 0.9,
        'can_rebuild': {'feasible': True, 'native_parts': [], 'blockers': []},
        'should_rebuild': {'editability_value': 'low', 'fidelity_risk': 'high', 'cost': 'medium'},
        'native_parts': [],
        'retained_parts': ['module_body'],
        'retain_reasons': [],
        'evidence': [],
    }
    if kind == 'network':
        return {**common, 'decision': 'retain_module', 'confidence': 0.97,
                'can_rebuild': {'feasible': False, 'native_parts': [], 'blockers': ['关系连线与曲线需要原生连接线支持']},
                'should_rebuild': {'editability_value': 'low', 'fidelity_risk': 'high', 'cost': 'high'},
                'retain_reasons': ['network', 'non_native_geometry'],
                'evidence': ['区域为多节点关系网络，含交叉连线与资金往来方向', '当前 IR 不支持原生连接线，拆分会退化为大量线段与文字']}
    if kind == 'photo_map':
        return {**common, 'decision': 'retain_module', 'confidence': 0.98,
                'can_rebuild': {'feasible': False, 'native_parts': [], 'blockers': ['地图底图与轨迹属于不可重建的照片层']},
                'retain_reasons': ['map', 'photo'],
                'evidence': ['区域以地图底图和照片为主体', '地图标注与轨迹不能还原为可编辑矢量，且编辑价值低']}
    if kind == 'photo_diagram':
        return {**common, 'decision': 'retain_module', 'confidence': 0.95,
                'can_rebuild': {'feasible': False, 'native_parts': [], 'blockers': ['照片与投影关系无法矢量化']},
                'retain_reasons': ['photo', 'non_native_geometry'],
                'evidence': ['区域由照片、投影和叠加标注组成', '照片部分无法用形状重建，强行拆分会产生明显接缝']}
    if kind == 'ui_mockup':
        return {**common, 'decision': 'retain_module', 'confidence': 0.92,
                'can_rebuild': {'feasible': True, 'native_parts': ['标题'], 'blockers': ['界面内文字普遍小于 10px', '状态标签与控件样式不是 IR 原生形状']},
                'should_rebuild': {'editability_value': 'low', 'fidelity_risk': 'high', 'cost': 'high'},
                'retain_reasons': ['ui_mockup', 'microtext'],
                'evidence': ['区域是软件界面截图，不是流程图或规则表格', '界面文字与状态控件密集，原生重建的视觉风险高于编辑收益']}
    if kind == 'dashboard':
        return {**common, 'decision': 'retain_module', 'confidence': 0.92,
                'can_rebuild': {'feasible': True, 'native_parts': ['标题', 'KPI'], 'blockers': ['图表没有可追溯数据', '界面文字普遍小于 10px', '包含多类异构控件']},
                'should_rebuild': {'editability_value': 'low', 'fidelity_risk': 'high', 'cost': 'high'},
                'retain_reasons': ['ui_mockup', 'microtext', 'no_data', 'low_edit_value'],
                'evidence': ['区域是仪表盘/看板示例，价值在整体示意而非单元格编辑', '图表数据只存在于像素中，无法追溯为原生图表数据']}
    if kind == 'chart':
        if f['median_text_height_px'] >= 9 and f['text_components'] <= 80 and not many_parts:
            return {**common, 'decision': 'native', 'confidence': 0.78,
                    'can_rebuild': {'feasible': True, 'native_parts': ['坐标轴/条形', '分类标签', '数值标签'], 'blockers': ['当前图表节点只覆盖竖向柱状，横向条需要用形状+文字或扩展图表节点']},
                    'should_rebuild': {'editability_value': 'medium', 'fidelity_risk': 'low', 'cost': 'low'},
                    'native_parts': ['bars', 'labels', 'percentages'], 'retained_parts': [],
                    'retain_reasons': [],
                    'evidence': ['区域只有少量分类、数值和条形，数据结构可读', '文字高度足够，条形可由原生矩形和文本重建']}
        return {**common, 'decision': 'retain_module', 'confidence': 0.8,
                'can_rebuild': {'feasible': True, 'native_parts': ['图表'], 'blockers': ['图表数据不可追溯', '文字过小或控件过多']},
                'retain_reasons': ['no_data', 'microtext'],
                'evidence': ['图表缺少可追溯的数据系列', '文字高度不足以稳定重建']}
    if kind == 'table':
        if f['microtext_fraction'] < 0.5 and f['text_components'] <= 100:
            return {**common, 'decision': 'hybrid', 'confidence': 0.74,
                    'can_rebuild': {'feasible': True, 'native_parts': ['表格'], 'blockers': ['状态标签和摘录框不是规则表格', '单元格文字接近 10px']},
                    'should_rebuild': {'editability_value': 'medium', 'fidelity_risk': 'medium', 'cost': 'medium'},
                    'native_parts': ['table'], 'retained_parts': ['letter_excerpt', 'status_chips'],
                    'retain_reasons': ['microtext', 'non_native_geometry'],
                    'evidence': ['表格行列清晰，可转为原生表格', '来信摘录和状态胶囊仍需保留为局部资产']}
        return {**common, 'decision': 'retain_module', 'confidence': 0.82,
                'can_rebuild': {'feasible': True, 'native_parts': ['表格'], 'blockers': ['微小文字和状态控件比例高']},
                'retain_reasons': ['microtext', 'non_native_geometry'],
                'evidence': ['表格文字密度高，原生表格会明显改变行高与换行', '状态标签不是规则表格结构']}
    return {**common, 'decision': 'retain_module', 'confidence': 0.7,
            'can_rebuild': {'feasible': False, 'native_parts': [], 'blockers': ['区域类型不是稳定流程图或规则表格']},
            'retain_reasons': ['low_edit_value'],
            'evidence': ['区域含多种异构内容，未达到可稳定重建的结构化程度']}


def suggest(inventory, output, overrides=None, human_decisions=None):
    data = read_json(inventory)
    overrides_file = overrides
    overrides = read_json(overrides_file) if overrides_file else {}
    decisions = load_human_decisions(human_decisions) if human_decisions else {}
    rows = []
    for row in data['regions']:
        agent = suggestion(row)
        if row['region_id'] in overrides:
            agent.update(overrides[row['region_id']])
        human = decisions.get(row['region_id'], {})
        rows.append({
            'region_id': row['region_id'],
            'page': row['page'],
            'ir': row['ir'],
            'node_id': row['node_id'],
            'bbox': row['bbox'],
            'classification': row['classification'],
            'features': row['features'],
            'agent': agent,
            'human': {
                'decision': human.get('decision', row.get('existing_decision', 'retain_module')),
                'source': human.get('source', 'accepted_deck_2026-10-22 (proxy)'),
                'note': human.get('note', row.get('human_note', '当前已验收 deck 的选择；正式人工复核可覆盖。')),
            },
        })
    result = {'schema_version': 'region-plan/0.1', 'regions': rows}
    write_json(output, result)
    return result


def semantic_errors(plans):
    errors = []
    ids = set()
    for row in plans['regions']:
        rid = row['region_id']
        if rid in ids:
            errors.append(f'{rid}: duplicate region_id')
        ids.add(rid)
        ir_path = resolve(row['ir'])
        if not ir_path.exists():
            errors.append(f'{rid}: IR not found: {row["ir"]}')
            continue
        ir = read_json(ir_path)
        node = next((value for value in ir['nodes'] if value['id'] == row['node_id']), None)
        if node is None or node.get('type') != 'image':
            errors.append(f'{rid}: image node not found')
            continue
        if [round(value, 3) for value in node['bbox']] != [round(value, 3) for value in row['bbox']]:
            errors.append(f'{rid}: bbox does not match IR node')
        width, height = row['bbox'][2], row['bbox'][3]
        if abs(width * height - row['features']['area_px']) > max(1, width * height * .01):
            errors.append(f'{rid}: area_px does not match bbox')
        agent = row['agent']
        decision = agent['decision']
        retain_reasons = agent.get('retain_reasons', [])
        if decision == 'native':
            if not agent['can_rebuild']['feasible']:
                errors.append(f'{rid}: native decision with infeasible rebuild')
            if not agent.get('native_parts'):
                errors.append(f'{rid}: native decision without native_parts')
            if agent['should_rebuild']['editability_value'] == 'low':
                errors.append(f'{rid}: native decision with low editability value')
            if agent['should_rebuild']['fidelity_risk'] == 'high':
                errors.append(f'{rid}: native decision with high fidelity risk')
        if decision == 'hybrid':
            if not agent.get('native_parts') or not agent.get('retained_parts'):
                errors.append(f'{rid}: hybrid needs native_parts and retained_parts')
            if not retain_reasons:
                errors.append(f'{rid}: hybrid needs retain_reasons')
        if decision == 'retain_module':
            if not agent.get('retained_parts'):
                errors.append(f'{rid}: retain decision without retained_parts')
            if not retain_reasons:
                errors.append(f'{rid}: retain decision without retain_reasons')
        if not agent.get('evidence'):
            errors.append(f'{rid}: decision without evidence')
    return errors


def validate(plans, output, human_decisions=None):
    plans_path = resolve(plans)
    plans = read_json(plans_path)
    schema = read_json(SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    errors = [f'{list(error.path)}: {error.message}' for error in sorted(validator.iter_errors(plans), key=lambda e: list(e.path))]
    errors.extend(semantic_errors(plans))
    if plans.get('planner', {}).get('kind') == 'llm':
        for row in plans['regions']:
            try:
                validate_decision(row['agent'])
            except (ValueError, TypeError) as exc:
                errors.append(f'{row["region_id"]}: strict LLM decision invalid: {exc}')
    checked = {'plans_sha256': file_sha(plans_path)}
    if human_decisions:
        decisions_path = resolve(human_decisions)
        checked['human_decisions_sha256'] = file_sha(decisions_path)
        decisions = load_human_decisions(decisions_path)
        for row in plans['regions']:
            expected = decisions.get(row['region_id'])
            if expected is None:
                errors.append(f'{row["region_id"]}: missing authoritative human decision')
            else:
                try:
                    actual = validate_human_decision(row.get('human'))
                    if HUMAN_SCOPE_KEYS <= set(expected) and not HUMAN_SCOPE_KEYS <= set(actual):
                        errors.append(f'{row["region_id"]}: human scope missing; reapply saved responses with current human decisions')
                    elif actual != expected:
                        errors.append(f'{row["region_id"]}: human decision differs from authoritative file')
                except (ValueError, TypeError) as exc:
                    errors.append(f'{row["region_id"]}: invalid human decision: {exc}')
    result = {'schema': 'RegionValidation/1', 'valid': not errors, 'errors': errors, 'checked': checked}
    write_json(output, result)
    return result


def scope_score(row, human):
    """Compare concrete native/retained scope, not only the routing label."""
    if (not isinstance(human, dict) or human.get('decision') in (None, 'pending')
            or not HUMAN_SCOPE_KEYS <= set(human)):
        return {'annotated': False, 'agreement': None}
    agent = row['agent']
    native = set(agent['native_parts'])
    retained = set(agent['retained_parts'])
    allowed = set(human['allowed_native_parts'])
    required = set(human['required_edit_targets'])
    must_retain = set(human['must_retain_parts'])
    over = sorted(native - allowed)
    under = sorted(required - native)
    retain_missing = sorted(must_retain - retained)
    overlap = sorted(native & must_retain)
    required_retained = sorted(required & retained)
    routing = agent['decision'] == human['decision']
    violations = bool(over or under or retain_missing or overlap or required_retained)
    return {
        'annotated': True,
        'agreement': routing and not violations,
        'routing_agreement': routing,
        'annotation_level': human['annotation_level'],
        'retain_scope': human['retain_scope'],
        'over_decomposition': over,
        'under_decomposition': under,
        'retain_violations': retain_missing,
        'native_retained_overlap': overlap,
        'required_retained_conflict': required_retained,
    }


def score(plans, output, human_decisions=None):
    plans = read_json(plans)
    decisions = load_human_decisions(human_decisions) if human_decisions else None
    rows = []
    for row in plans['regions']:
        agent = row['agent']['decision']
        human = (decisions.get(row['region_id'], {}).get('decision')
                 if decisions is not None else row['human']['decision'])
        agreement = None if human in (None, 'pending') else agent == human
        annotation = decisions.get(row['region_id'], {}) if decisions is not None else row.get('human', {})
        rows.append({'region_id': row['region_id'], 'page': row['page'],
                     'agent': agent, 'human': human, 'agreement': agreement,
                     'scope': scope_score(row, annotation)})
    decided = [row for row in rows if row['agreement'] is not None]
    agreed = [row for row in decided if row['agreement']]
    routing = {
        'decided': len(decided),
        'agreed': len(agreed),
        'agreement_rate': round(len(agreed) / len(decided), 4) if decided else None,
        'always_retain_baseline': (
            round(sum(1 for row in decided if row['human'] == 'retain_module') / len(decided), 4)
            if decided else None),
    }
    scope_rows = [row for row in rows if row['scope']['annotated'] and row['scope']['agreement'] is not None]
    scope_agreed = [row for row in scope_rows if row['scope']['agreement']]
    scope = {
        'decided': len(scope_rows),
        'agreed': len(scope_agreed),
        'agreement_rate': round(len(scope_agreed) / len(scope_rows), 4) if scope_rows else None,
        'over_decomposition': sum(1 for row in scope_rows if row['scope']['over_decomposition']),
        'under_decomposition': sum(1 for row in scope_rows if row['scope']['under_decomposition']),
        'retain_violations': sum(1 for row in scope_rows if row['scope']['retain_violations']),
        'disagreements': [row for row in scope_rows if not row['scope']['agreement']],
    }
    result = {
        'total': len(rows),
        'decided': len(decided),
        'agreed': len(agreed),
        'agreement_rate': routing['agreement_rate'],
        'disagreements': [row for row in decided if not row['agreement']],
        'routing': routing,
        'scope': scope,
        'rows': rows,
    }
    write_json(output, result)
    return result


def human_decisions(plans, output):
    data = read_json(plans)
    decisions = {}
    for row in data['regions']:
        human = row.get('human')
        if not isinstance(human, dict) or human.get('decision') not in ('native', 'hybrid', 'retain_module', 'pending'):
            raise ValueError('invalid human decision for ' + row['region_id'])
        decision = {
            'decision': human['decision'],
            'source': human.get('source', 'accepted_deck (proxy)'),
            'note': human.get('note', ''),
        }
        for key in HUMAN_SCOPE_KEYS:
            if key in human:
                decision[key] = human[key]
        decisions[row['region_id']] = decision
    result = {'schema_version': 'region-human-decisions/0.2', 'decisions': decisions}
    write_json(output, result)
    return result


def prepare_llm(inventory, jobs, model, base_url, region=None, dry_run=False):
    inventory = resolve(inventory)
    data = read_json(inventory)
    inventory_sha256 = file_sha(inventory)
    selected = set(region or [])
    known = {row['region_id'] for row in data['regions']}
    unknown = sorted(selected - known)
    if unknown:
        raise ValueError('unknown region: ' + ', '.join(unknown))
    rows = [row for row in data['regions'] if not selected or row['region_id'] in selected]
    jobs = resolve(jobs)
    results = []
    for row in rows:
        results.append(prepare_region_job(
            row, inventory.parent / row['crop'], jobs / row['region_id'],
            model=model, base_url=base_url, inventory_sha256=inventory_sha256,
            dry_run=dry_run))
    return {
        'jobs_root': str(jobs),
        'dry_run': dry_run,
        'count': len(results),
        'statuses': {item['region_id']: item['status'] for item in results},
        'jobs': results,
        'api_called': False,
    }


def apply_llm(inventory, jobs, output, human_decisions, allow_partial=False, dry_run=False,
              apply_report=None, provenance=None):
    inventory = resolve(inventory)
    data = read_json(inventory)
    decisions = load_human_decisions(human_decisions)
    result = apply_jobs(data['regions'], resolve(jobs), decisions,
                        inventory_sha256=file_sha(inventory), strict=not allow_partial)
    output = resolve(output)
    report_path = resolve(apply_report) if apply_report else output.parent / 'apply-report.json'
    provenance_path = resolve(provenance) if provenance else output.parent / 'llm-provenance.json'
    if not dry_run:
        write_json(report_path, result['report'])
        write_json(provenance_path, result['provenance'])
        if result['plans'] is not None:
            write_json(output, result['plans'])
    return {
        'status': result['status'],
        'plans': str(output) if result['plans'] is not None and not dry_run else None,
        'apply_report': str(report_path) if not dry_run else None,
        'provenance': str(provenance_path) if not dry_run else None,
        'report': result['report'],
        'provenance_data': result['provenance'],
    }


def report(plans, score, output, provenance=None, validation=None, apply_report=None):
    plans = read_json(plans)
    scored = read_json(score)
    routing = scored.get('routing', {
        'decided': scored['decided'], 'agreed': scored['agreed'],
        'agreement_rate': scored['agreement_rate'], 'always_retain_baseline': None,
    })
    scope = scored.get('scope')
    lines = [
        '# Region rebuild shadow mode',
        '',
        f'- regions: {scored["total"]}',
        f'- routing decided: {routing["decided"]}',
        f'- routing agreement: {routing["agreed"]}/{routing["decided"]} = {routing["agreement_rate"]}',
        f'- always-retain baseline: {routing["always_retain_baseline"]}',
    ]
    if scope:
        lines += [
            f'- scope annotated: {scope["decided"]}',
            f'- scope agreement: {scope["agreed"]}/{scope["decided"]} = {scope["agreement_rate"]}',
            f'- over / under / retain violations: {scope["over_decomposition"]} / '
            f'{scope["under_decomposition"]} / {scope["retain_violations"]}',
        ]
    lines += [
        '',
        '| Region | Page | Type | Agent | Human | Route | Scope | Confidence | Evidence |',
        '|---|---|---|---|---|---|---|---:|---|',
    ]
    for row in plans['regions']:
        match = next(item for item in scored['rows'] if item['region_id'] == row['region_id'])
        agree = 'pending' if match['agreement'] is None else ('yes' if match['agreement'] else 'no')
        item_scope = match.get('scope', {})
        scope_text = '-' if not item_scope.get('annotated') else (
            'yes' if item_scope.get('agreement') else 'no')
        evidence = '; '.join(row['agent']['evidence']).replace('|', '/')[:180]
        lines.append(f'| {row["region_id"]} | {row["page"]} | {row["classification"]} | '
                     f'{row["agent"]["decision"]} | {row["human"]["decision"]} | {agree} | {scope_text} | '
                     f'{row["agent"]["confidence"]:.2f} | {evidence} |')
    lines += ['', '## Routing disagreements', '']
    if scored['disagreements']:
        for row in scored['disagreements']:
            lines.append(f'- {row["region_id"]}: agent={row["agent"]}, human={row["human"]}')
    else:
        lines.append('- none')
    if scope:
        lines += ['', '## Scope disagreements', '']
        if scope['disagreements']:
            for row in scope['disagreements']:
                item = row['scope']
                lines.append(
                    f'- {row["region_id"]}: over={item["over_decomposition"]}, '
                    f'under={item["under_decomposition"]}, '
                    f'retain_missing={item["retain_violations"]}, '
                    f'native/retained overlap={item["native_retained_overlap"]}, '
                    f'required/retained conflict={item["required_retained_conflict"]}')
        else:
            lines.append('- none')
    lines += ['', '## Measured features', '',
              '| Region | Area px | Aspect | Colors | Edge density | Median text px | Microtext |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in plans['regions']:
        f = row['features']
        lines.append(f'| {row["region_id"]} | {f["area_px"]} | {f["aspect"]} | {f["unique_colors"]} | '
                     f'{f["edge_density"]} | {f["median_text_height_px"]} | {f["microtext_fraction"]} |')
    if provenance:
        prov = read_json(provenance)
        lines += ['', '## Provenance', '',
                  f'- prompt_version: {prov.get("prompt_version")}',
                  f'- capabilities_sha256: {prov.get("capabilities_sha256")}',
                  '', '| Region | Status | Transport | Response sha256 |',
                  '|---|---|---|---|']
        for row in prov.get('rows', []):
            digest = row.get('response_sha256') or ''
            lines.append(f'| {row.get("region_id")} | {row.get("status")} | '
                         f'{row.get("transport_status", "")} | {digest[:16]} |')
    if validation:
        checked = read_json(validation)
        lines += ['', '## Validation', '', f'- valid: {checked.get("valid")}']
        for error in checked.get('errors', [])[:20]:
            lines.append(f'- {error}')
    if apply_report:
        applied = read_json(apply_report)
        lines += ['', '## Apply report', '',
                  f'- status: {applied.get("status")}',
                  f'- validated: {applied.get("validated")}/{applied.get("total")}',
                  '- human_acceptance: false', '- shadow_only: true']
    Path(output).write_text('\n'.join(lines) + '\n', encoding='utf8')
    return {'report': str(output)}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    q = sub.add_parser('inventory')
    q.add_argument('--regions', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--human-decisions')
    q = sub.add_parser('suggest')
    q.add_argument('--inventory', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--overrides')
    q.add_argument('--human-decisions')
    q = sub.add_parser('validate')
    q.add_argument('--plans', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--human-decisions')
    q = sub.add_parser('score')
    q.add_argument('--plans', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--human-decisions')
    q = sub.add_parser('report')
    q.add_argument('--plans', required=True)
    q.add_argument('--score', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--provenance')
    q.add_argument('--validation')
    q.add_argument('--apply-report')
    q = sub.add_parser('human-decisions')
    q.add_argument('--plans', required=True)
    q.add_argument('--output', required=True)
    q = sub.add_parser('prepare-llm')
    q.add_argument('--inventory', required=True)
    q.add_argument('--jobs', required=True)
    q.add_argument('--model', required=True)
    q.add_argument('--base-url', required=True)
    q.add_argument('--region', action='append')
    q.add_argument('--dry-run', action='store_true')
    q = sub.add_parser('apply-llm')
    q.add_argument('--inventory', required=True)
    q.add_argument('--jobs', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--human-decisions', required=True)
    q.add_argument('--allow-partial', action='store_true')
    q.add_argument('--dry-run', action='store_true')
    q.add_argument('--apply-report')
    q.add_argument('--provenance')
    args = vars(parser.parse_args())
    command = args.pop('command')
    result = {'inventory': inventory, 'suggest': suggest, 'validate': validate,
              'score': score, 'report': report, 'human-decisions': human_decisions,
              'prepare-llm': prepare_llm, 'apply-llm': apply_llm}[command](**args)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if command == 'validate' and not result['valid']:
        sys.exit(1)
    if command == 'apply-llm' and result['status'] == 'INCOMPLETE':
        sys.exit(1)


if __name__ == '__main__':
    main()
