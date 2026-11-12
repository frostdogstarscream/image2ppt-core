"""Source-bound font calibration and deterministic single-line containment."""
from __future__ import annotations

import copy
from pathlib import Path

from PIL import ImageFont

from model_text_layout import calibrate
from font_matcher import CANDIDATE_FILES, FONT_DIR


def _font_paths(font, bold, include_latin=True):
    aliases = {'Microsoft YaHei': '微软雅黑', 'SimHei': '黑体',
               'SimSun': '宋体', 'DengXian': '等线'}
    paths = set()
    names = (font['latin'], font['ea']) if include_latin else (font['ea'],)
    for name in names:
        files = CANDIDATE_FILES.get(aliases.get(name, name), [])
        if files:
            path = FONT_DIR / files[1 if bold and len(files) > 1 else 0]
            if path.exists():
                paths.add(path)
    return sorted(paths)


def _measure(node, pixels_per_point):
    width = height = 0
    runs = node.get('runs') or [{'text': node['text']}]
    for run in runs:
        paths = _font_paths(node['font'], run.get('bold', node.get('bold', False)), run['text'].isascii())
        if not paths:
            return None
        dimensions = []
        for path in paths:
            face = ImageFont.truetype(str(path), max(1, round(node['font']['size_pt'] * pixels_per_point)))
            box = face.getbbox(run['text'])
            dimensions.append((max(float(face.getlength(run['text'])), box[2]-box[0]), box[3]-box[1]))
        width += max(item[0] for item in dimensions)
        height = max(height, max(item[1] for item in dimensions))
    return width, height


def fit_model_text(ir: dict, source: Path, assets_base: Path):
    """Reuse source-ink fitting, then shrink measured overflow locally."""
    eligible = [node for node in ir['nodes'] if node['type'] == 'text'
                and '\n' not in node['text'] and not node.get('paragraphs') and not node.get('wrap')
                and _font_paths(node['font'], node.get('bold', False), node['text'].isascii())]
    eligible_ids = {node['id'] for node in eligible}
    skipped = [node for node in ir['nodes'] if node['type'] == 'text' and node['id'] not in eligible_ids]
    if eligible:
        measured_input = copy.deepcopy(ir)
        skipped_ids = {node['id'] for node in skipped}
        measured_input['nodes'] = [node for node in measured_input['nodes'] if node['id'] not in skipped_ids]
        fitted, source_report = calibrate(measured_input, source, assets_base)
        fitted['nodes'].extend(copy.deepcopy(skipped))
    else:
        fitted = copy.deepcopy(ir)
        source_report = {'status': 'SKIPPED', 'rows': [], 'splits': []}
    target_to_source = ir['canvas']['source_px'][0] / ir['canvas']['target_px'][0]
    scale = 96 / 72 * target_to_source
    containment = []
    for node in fitted['nodes']:
        if node['type'] != 'text' or '\n' in node['text'] or node.get('paragraphs') or node.get('wrap'):
            continue
        dimensions = _measure(node, scale)
        if dimensions is None:
            containment.append({'id': node['id'], 'status': 'FONT_MEASUREMENT_UNAVAILABLE'})
            continue
        left, right, top, bottom = node.get('insets', [0, 0, 0, 0])
        available = [max(.1, node['bbox'][2]-(left+right)*target_to_source),
                     max(.1, node['bbox'][3]-(top+bottom)*target_to_source)]
        ratio = min(1, available[0]/max(.1, dimensions[0]), available[1]/max(.1, dimensions[1]))
        if ratio < 1:
            original = node['font']['size_pt']
            node['font']['size_pt'] = max(.1, round(original * ratio * .98, 2))
            containment.append({'id': node['id'], 'status': 'SHRUNK_TO_FIT',
                'old_size_pt': original, 'size_pt': node['font']['size_pt'],
                'measured_source_px': list(dimensions), 'available_source_px': available})
    report = {'source_calibration': source_report, 'containment': containment,
              'skipped_text_nodes': [node['id'] for node in skipped]}
    summary = {'source_fitted': sum(row['status'] == 'FITTED' for row in source_report['rows']),
               'source_uncertain': [row['node_id'] for row in source_report['rows'] if row['status'] != 'FITTED'],
               'containment': containment, 'splits': source_report.get('splits', []),
               'skipped_text_nodes': report['skipped_text_nodes']}
    return fitted, report, summary
