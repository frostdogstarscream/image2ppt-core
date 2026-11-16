"""Deterministic perception preparation and batch regional reconstruction backbone.

Preserves all source-pixel ownership, peer protection, Office editing, and
hash-bound visual review invariants while reducing 100+ object slides from
hundreds of Agent tool turns to a single batch backbone pass plus targeted repair.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import threading

import cv2
import numpy as np
from PIL import Image

from ..contracts import intersection
from ..materialize import digest, write
from ..object_contracts import ClosedObject
from ..object_extract import rectangle
from ..parallel_contracts import Relation, SceneGraph
from ..parallel_fusion import is_decorative_subspan
from ..parallel_store import Store, key
from ..parallel_typography import calibrate, snap_ooxml_size_px, measure_text_appearance
from ..resource_lease import lease


def split_decorative_ocr_line(source: np.ndarray, line: dict) -> tuple[str, list[float], str]:
    """Strip purely non-alphanumeric leading/trailing affixes when a clean pixel gap exists."""
    raw_text = line.get('text', '').strip()
    bbox = [float(v) for v in line['bbox']]
    compact = raw_text.replace(' ', '').replace('\n', '')
    if len(compact) < 3 or not any(ch.isalnum() for ch in compact):
        return raw_text, bbox, ''
    start = 0
    while start < len(compact) and not compact[start].isalnum():
        start += 1
    end = len(compact)
    while end > start and not compact[end - 1].isalnum():
        end -= 1
    core = compact[start:end]
    prefix, suffix = compact[:start], compact[end:]
    # Keep paired brackets or trailing punctuation that belong to wording (e.g. percentages/colons).
    keep_marks = set('%％:：()（）[]【】《》“”"\'.,，。!?！？+=/')
    if (not prefix and not suffix) or len(core) < 2:
        return raw_text, bbox, ''
    if any(ch in keep_marks for ch in (prefix + suffix)):
        return raw_text, bbox, ''
    # Recover spaces inside the core text while stripping outer decorative marks.
    stripped = raw_text
    while stripped and not stripped[0].isalnum() and stripped[0] not in keep_marks:
        stripped = stripped[1:].lstrip()
    while stripped and not stripped[-1].isalnum() and stripped[-1] not in keep_marks:
        stripped = stripped[:-1].rstrip()
    if len(stripped.replace(' ', '')) < 2 or stripped == raw_text:
        return raw_text, bbox, ''
    lx, ly, lw, lh = [round(v) for v in bbox]
    h_img, w_img = source.shape[:2]
    if min(lw, lh) <= 4 or lx < 0 or ly < 0 or lx + lw > w_img or ly + lh > h_img:
        return raw_text, bbox, ''
    crop = source[ly:ly + lh, lx:lx + lw].astype(float)
    border = np.concatenate([crop[0], crop[-1]], axis=0)
    bg = np.median(border, axis=0)
    col_ink = (np.abs(crop - bg).max(2) > 30.0).sum(axis=0) > 0
    runs: list[tuple[int, int]] = []
    idx = 0
    while idx < lw:
        if not col_ink[idx]:
            idx += 1
            continue
        j = idx
        while j < lw and col_ink[j]:
            j += 1
        if runs and idx - runs[-1][1] <= 2:
            runs[-1] = (runs[-1][0], j)
        else:
            runs.append((idx, j))
        idx = j
    if len(runs) < 2:
        return raw_text, bbox, ''
    sub_left, sub_right = 0, lw
    if prefix and not suffix:
        # Drop leading decorative run(s) up to proportional prefix fraction.
        max_cut = lw * min(0.45, (len(prefix) / len(compact)) + 0.18)
        kept = [r for r in runs if r[0] >= max(2, round(runs[0][1])) and r[0] <= max_cut]
        sub_left = kept[0][0] if kept else runs[1][0]
    elif suffix and not prefix:
        min_cut = lw * max(0.50, 1.0 - (len(suffix) / len(compact)) - 0.22)
        candidates = [i for i in range(1, len(runs)) if runs[i][0] >= min_cut and runs[i][0] - runs[i - 1][1] >= 2]
        if not candidates:
            return raw_text, bbox, ''
        cut_idx = candidates[0]
        sub_right = runs[cut_idx - 1][1]
    else:
        return raw_text, bbox, ''
    sub_bbox = [float(lx + sub_left), float(ly), float(max(4, sub_right - sub_left)), float(lh)]
    if is_decorative_subspan({'text': stripped, 'bbox': sub_bbox}, line):
        return stripped, sub_bbox, (prefix or suffix)
    return raw_text, bbox, ''


def discover_layerd_assets(source: np.ndarray, layers_dir: Path,
                           text_boxes: list[list[float]]) -> tuple[list[dict], list[dict]]:
    """Deterministically propose whole graphic objects (icons, composite badges, illustrations)
    from foreground LayerD layers and detexted source closure without VLM round-trips.

    Returns `(immediate_assets, deferred_assets)`, where `deferred_assets` are objects
    within 10px of a text bbox (staged in Phase 2 after texts are cleanly erased onto
    `source-plate.png`, avoiding conservative bbox veto during text background fitting).
    """
    manifest_file = Path(layers_dir) / 'manifest.json'
    if not manifest_file.is_file():
        return [], []
    manifest = json.loads(manifest_file.read_text('utf8'))
    layer_entries = manifest.get('layers', [])
    if len(layer_entries) < 2:
        return [], []
    h_img, w_img = source.shape[:2]
    base_entry = next((r for r in layer_entries if r.get('z', 0) == 0), layer_entries[0])
    base_path = Path(layers_dir) / base_entry['file']
    if not base_path.is_file():
        return [], []
    layer0_rgb = np.array(Image.open(base_path).convert('RGB'))
    if layer0_rgb.shape[:2] != (h_img, w_img):
        return [], []

    text_rect_mask = np.zeros((h_img, w_img), bool)
    for tb in text_boxes:
        text_rect_mask |= rectangle(tb, (w_img, h_img), pad=1)
    detext_src = source.copy()
    detext_src[text_rect_mask] = layer0_rgb[text_rect_mask]

    def _expand_on_detext(bx: int, by: int, bw: int, bh: int,
                          seed_mask_full: np.ndarray | None, pad: int) -> tuple[int, int, int, int] | None:
        wx0, wy0 = max(0, bx - pad), max(0, by - pad)
        wx1, wy1 = min(w_img, bx + bw + pad), min(h_img, by + bh + pad)
        win = detext_src[wy0:wy1, wx0:wx1].astype(float)
        wh, ww = win.shape[:2]
        if wh < 6 or ww < 6:
            return None
        bdr = np.concatenate([win[:2].reshape(-1, 3), win[-2:].reshape(-1, 3),
                              win[:, :2].reshape(-1, 3), win[:, -2:].reshape(-1, 3)], axis=0)
        bg_med = np.median(bdr, axis=0)
        if float(np.percentile(np.abs(bdr - bg_med).max(1), 75)) > 5.0:
            return None
        fg_win = (np.abs(win - bg_med).max(2) > 6.0).astype('uint8')
        _, w_lbl, _, _ = cv2.connectedComponentsWithStats(fg_win, 8)
        if seed_mask_full is not None:
            seed_local = seed_mask_full[wy0:wy1, wx0:wx1]
        else:
            seed_local = np.zeros((wh, ww), bool)
            seed_local[by - wy0:by - wy0 + bh, bx - wx0:bx - wx0 + bw] = fg_win[by - wy0:by - wy0 + bh, bx - wx0:bx - wx0 + bw] > 0
        hit_ids = [int(v) for v in np.unique(w_lbl[seed_local]) if v > 0]
        if not hit_ids:
            return None
        merged_win = np.isin(w_lbl, hit_ids)
        touches_edge = ((wx0 > 0 and merged_win[:, 0].any())
                        or (wx1 < w_img and merged_win[:, -1].any())
                        or (wy0 > 0 and merged_win[0, :].any())
                        or (wy1 < h_img and merged_win[-1, :].any()))
        if touches_edge:
            return None
        yy, xx = np.nonzero(merged_win)
        ex, ey = int(wx0 + xx.min()), int(wy0 + yy.min())
        ew, eh = int(wx0 + xx.max() - ex + 1), int(wy0 + yy.max() - ey + 1)
        if ew * eh > 0.38 * (w_img * h_img):
            return None
        if any(intersection([ex, ey, ew, eh], tb) >= 0.55 * max(1.0, tb[2] * tb[3]) for tb in text_boxes):
            return None
        return ex, ey, ew, eh

    raw_parts: list[dict] = []
    for entry in layer_entries:
        if entry.get('z', 0) <= 0:
            continue
        fname = entry['file']
        fpath = Path(layers_dir) / fname
        if not fpath.is_file():
            continue
        arr = np.array(Image.open(fpath).convert('RGBA'))
        if arr.shape[:2] != (h_img, w_img):
            continue
        fg_mask = (arr[..., 3] > 16) & (~text_rect_mask)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask.astype('uint8'), 8)
        for cid in range(1, num_labels):
            bx, by, bw, bh, area = [int(v) for v in stats[cid]]
            if area < 60 or min(bw, bh) < 6:
                continue
            if bw * bh > 0.38 * (w_img * h_img):
                continue
            if (bw / max(1, bh) > 8.0 and bh <= 14) or (bh / max(1, bw) > 8.0 and bw <= 14):
                continue
            if any(intersection([bx, by, bw, bh], tb) >= 0.25 * max(1.0, tb[2] * tb[3]) for tb in text_boxes):
                continue
            expanded = _expand_on_detext(bx, by, bw, bh, labels == cid, pad=36)
            if expanded is not None:
                bx, by, bw, bh = expanded
            raw_parts.append({'bbox': [bx, by, bw, bh], 'layers': {fname}})

    if not raw_parts:
        return [], []

    clusters = list(raw_parts)
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            if changed:
                break
            ax, ay, aw, ah = clusters[i]['bbox']
            for j in range(i + 1, len(clusters)):
                bx, by, bw, bh = clusters[j]['bbox']
                gap_x = max(0, max(ax, bx) - min(ax + aw, bx + bw))
                gap_y = max(0, max(ay, by) - min(ay + ah, by + bh))
                if gap_x > 28 or gap_y > 28:
                    continue
                mx0, my0 = min(ax, bx), min(ay, by)
                mx1, my1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                mbox = [mx0, my0, mx1 - mx0, my1 - my0]
                if mbox[2] * mbox[3] > 0.38 * (w_img * h_img):
                    continue
                bad_text_swallow = False
                for tb in text_boxes:
                    ov = intersection(mbox, tb)
                    if ov >= 0.55 * max(1.0, tb[2] * tb[3]):
                        bad_text_swallow = True
                        break
                    if ov >= 0.25 * max(1.0, tb[2] * tb[3]) and intersection(clusters[i]['bbox'], tb) == 0 and intersection(clusters[j]['bbox'], tb) == 0:
                        bad_text_swallow = True
                        break
                if bad_text_swallow:
                    continue
                clusters[i] = {'bbox': mbox, 'layers': clusters[i]['layers'] | clusters[j]['layers']}
                clusters.pop(j)
                changed = True
                break

    for cl in clusters:
        cx, cy, cw, ch = cl['bbox']
        expanded = _expand_on_detext(cx, cy, cw, ch, None, pad=40)
        if expanded is not None:
            ex, ey, ew, eh = expanded
            newly_hit = any(intersection([ex, ey, ew, eh], tb) >= 0.25 * max(1.0, tb[2] * tb[3])
                            and intersection(cl['bbox'], tb) == 0 for tb in text_boxes)
            if not newly_hit:
                cl['bbox'] = [ex, ey, ew, eh]

    clusters.sort(key=lambda c: (c['bbox'][1] // 40, c['bbox'][0]))
    immediate_assets: list[dict] = []
    deferred_assets: list[dict] = []
    for idx, cl in enumerate(clusters, 1):
        bx, by, bw, bh = cl['bbox']
        area_box = bw * bh
        aspect = max(bw / max(1, bh), bh / max(1, bw))
        if (max(bw, bh) >= 180 and min(bw, bh) >= 60) or area_box >= 25000:
            kind = 'illustration'
        elif aspect >= 3.0 or area_box < 1200:
            kind = 'decoration'
        else:
            kind = 'icon'
        sorted_layers = sorted(cl['layers'])
        ibox = [int(bx), int(by), int(bw), int(bh)]
        obj_dict = {
            'id': f'a-{idx:03d}',
            'kind': kind,
            'bbox': ibox,
            'layers': sorted_layers,
            'components': ['whole'],
            'parts': [{'name': 'whole', 'bbox': ibox, 'layers': sorted_layers}],
            'exclusions': [],
            'replaces': [],
            'required_owners': [],
            'text_policy': 'exclude_independent_text',
            'intrinsic_text': '',
            'description': f'whole independent {kind}',
            'prompt_en': f'whole independent {kind}',
        }
        ClosedObject.model_validate(obj_dict)
        padded_box = [bx - 10.0, by - 10.0, bw + 20.0, bh + 20.0]
        if any(intersection(padded_box, tb) > 0 for tb in text_boxes):
            deferred_assets.append(obj_dict)
        else:
            immediate_assets.append(obj_dict)
    return immediate_assets, deferred_assets


def prepare_page_evidence(service, state, current, directory: Path, *, page: int,
                          auto_create_scene: bool = True, run_layers: bool = True) -> dict:
    """Run local OCR, optional LayerD, pixel ink color/size measurement, and style tiering."""
    source_path = service.source(state, page)
    if str(page) not in current.get('ocr', {}):
        ocr_dir = directory / 'ocr'
        ocr_dir.mkdir(parents=True, exist_ok=True)
        service.run_ocr(state, current, ocr_dir, page=page)
    ocr = json.loads(Path(current['ocr'][str(page)]).read_text('utf8'))
    if 'fonts' not in current:
        font_dir = directory / 'fonts'
        font_dir.mkdir(parents=True, exist_ok=True)
        service.list_fonts(state, current, font_dir)
    fonts = json.loads(Path(current['fonts']).read_text('utf8'))
    layerd_status = 'SKIPPED'
    layer_images = []
    if str(page) in current.get('layers', {}):
        layerd_status = 'CACHED'
        layers_dir = Path(current['layers'][str(page)])
        meta_file = layers_dir / 'manifest.json'
        if meta_file.is_file():
            meta = json.loads(meta_file.read_text('utf8'))
            layer_images = [state.register(layers_dir / r['file'], 'layer') for r in meta.get('layers', []) if (layers_dir / r['file']).is_file()]
    elif run_layers:
        try:
            layer_dir = directory / 'layers'
            layer_dir.mkdir(parents=True, exist_ok=True)
            dec = service.decompose_layers(state, current, layer_dir, page=page)
            layer_images = dec.get('images', [])
            layerd_status = 'READY'
        except Exception as exc:
            layerd_status = f'UNAVAILABLE:{type(exc).__name__}'

    source = np.array(Image.open(source_path).convert('RGB'))
    h_img, w_img = source.shape[:2]
    cjk_family = next((f['family'] for f in fonts['fonts'] if f.get('cjk') and '雅黑' in f['family']), None)
    if not cjk_family:
        cjk_family = next((f['family'] for f in fonts['fonts'] if f.get('cjk')), 'Arial')
    latin_family = next((f['family'] for f in fonts['fonts'] if f['family'] in ('Arial', 'Segoe UI', 'Calibri')), cjk_family)

    valid_lines = [row for row in ocr.get('lines', []) if row.get('score', 0) >= 0.85 and row.get('text', '').strip()]
    line_rects = [rectangle(row['bbox'], (w_img, h_img), pad=1) for row in valid_lines]

    measured_items = []
    for index, row in enumerate(valid_lines, 1):
        peers_mask = np.zeros((h_img, w_img), bool)
        for j, rect_mask in enumerate(line_rects):
            if j != index - 1:
                peers_mask |= rect_mask
        text_val, item_bbox, stripped_affix = split_decorative_ocr_line(source, row)
        appearance = measure_text_appearance(source, item_bbox, peers_mask)
        color_hex = appearance['color_hex'] if appearance else '222222'
        backing_class = appearance['backing_class'] if appearance else 'COMPLEX'
        raw_h = item_bbox[3]
        if appearance is not None and appearance['ink'].any():
            yy = np.nonzero(appearance['ink'])[0]
            ink_span = float(yy.max() - yy.min() + 1)
            est_px = snap_ooxml_size_px(max(9.0, min(96.0, max(ink_span * 0.96, raw_h * 0.72))))
        else:
            est_px = snap_ooxml_size_px(max(9.0, min(96.0, raw_h * 0.76)))
        role = 'title' if est_px >= 26.0 else ('subtitle' if est_px >= 18.0 else ('caption' if est_px <= 12.0 else 'body'))
        bold = bool(est_px >= 20.0)
        measured_items.append({
            'id': f't-{index:03d}',
            'ocr_line_id': row['id'],
            'text': text_val,
            'bbox': [round(float(v), 2) for v in item_bbox],
            'color_hex': color_hex,
            'est_size_px': float(est_px),
            'role': role,
            'bold': bold,
            'backing_class': backing_class,
            'stripped_affix': stripped_affix,
        })

    # Cluster styles by (role, bold, quantized RGB, 1.5px size bucket)
    style_groups: list[dict] = []
    for item in measured_items:
        rgb = np.array([int(item['color_hex'][i:i + 2], 16) for i in (0, 2, 4)], dtype=float)
        matched = None
        for grp in style_groups:
            if (grp['role'] == item['role'] and grp['bold'] == item['bold']
                    and abs(grp['size_sum'] / grp['count'] - item['est_size_px']) <= 1.5
                    and np.abs(grp['rgb_sum'] / grp['count'] - rgb).max() <= 16.0):
                matched = grp
                break
        if matched is None:
            matched = {'id': f's-{len(style_groups) + 1:02d}', 'role': item['role'], 'bold': item['bold'],
                       'size_sum': 0.0, 'rgb_sum': np.zeros(3, dtype=float), 'count': 0}
            style_groups.append(matched)
        matched['size_sum'] += item['est_size_px']
        matched['rgb_sum'] += rgb
        matched['count'] += 1
        item['style'] = matched['id']

    styles = []
    style_lookup = {}
    for grp in style_groups:
        avg_size = snap_ooxml_size_px(grp['size_sum'] / grp['count'])
        avg_rgb = np.clip(np.rint(grp['rgb_sum'] / grp['count']), 0, 255).astype(int)
        hex_c = ''.join(f'{int(v):02X}' for v in avg_rgb)
        st = {'id': grp['id'], 'role': grp['role'], 'ea': cjk_family, 'latin': latin_family,
              'size_px': avg_size, 'bold': bool(grp['bold']), 'color': hex_c}
        styles.append(st)
        style_lookup[grp['id']] = st

    texts = []
    members = {}
    for z_idx, item in enumerate(measured_items, 1):
        st = style_lookup[item['style']]
        item['color_hex'] = st['color']
        item['size_px'] = st['size_px']
        texts.append({'id': item['id'], 'bbox': item['bbox'], 'z': int(z_idx),
                      'style': item['style'], 'text': item['text'], 'align': 'left'})
        members[item['id']] = [item['ocr_line_id']]

    immediate_assets: list[dict] = []
    deferred_assets: list[dict] = []
    layers_path_str = current.get('layers', {}).get(str(page))
    if layers_path_str and Path(layers_path_str).is_dir():
        try:
            immediate_assets, deferred_assets = discover_layerd_assets(
                source, Path(layers_path_str), [it['bbox'] for it in measured_items])
        except Exception:
            immediate_assets, deferred_assets = [], []

    existing = current['pages'][page - 1] if len(current['pages']) >= page else None
    scene_initialized = False
    if auto_create_scene and not existing:
        scene_dir = directory / 'scene-init'
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_evidence: dict = {'text': {'styles': styles, 'texts': texts, 'members': members}}
        if immediate_assets:
            scene_evidence['asset'] = {'objects': immediate_assets}
        service.create_scene(state, current, scene_dir, page=page, evidence=scene_evidence)
        if deferred_assets:
            deferred_file = directory / 'deferred-assets.json'
            write(deferred_file, deferred_assets)
            current.setdefault('deferred_assets', {})[str(page)] = str(deferred_file)
        row = service.page(current, page)
        graph = service.graph(row)
        store = Store(state.job / 'state')
        with lease('cpu'):
            calibrated = calibrate(graph, source, fonts, store, ocr=ocr, adaptive=True)
        if calibrated.model_dump() != graph.model_dump():
            from ..parallel_rebuild import build
            cal_dir = directory / 'calibrated-backing'
            ir, _ = build(calibrated, row['snapshot'], fonts, cal_dir,
                          strategy='off', text_candidates=False, surface_candidates=False)
            shutil.copyfile(Path(row['ir']).parent / 'runtime-ocr-protection.json',
                            ir.parent / 'runtime-ocr-protection.json')
            row['ir'] = str(ir)
            row['scene'] = str(ir.parent / 'scene.json')
        scene_initialized = True

    row = current['pages'][page - 1] if len(current['pages']) >= page else None
    active_graph = service.graph(row) if row else None
    final_styles = [s.model_dump() for s in active_graph.text.styles] if active_graph else styles
    active_style_map = {s['id']: s for s in final_styles}
    for item in measured_items:
        if active_graph and item['id'] in active_graph.ownership:
            item['ownership'] = active_graph.ownership[item['id']]
            t_obj = next((t for t in active_graph.text.texts if t.id == item['id']), None)
            if t_obj and t_obj.style in active_style_map:
                item['style'] = t_obj.style
                item['color_hex'] = active_style_map[t_obj.style]['color']
                item['size_px'] = active_style_map[t_obj.style]['size_px']
        else:
            item['ownership'] = 'DRAFT_PROPOSAL'

    counts = Counter(item['backing_class'] for item in measured_items)
    resolved_count = sum(1 for item in measured_items if item['ownership'] == 'RESOLVED')
    asset_summaries = (
        [{'id': a['id'], 'kind': a['kind'], 'bbox': a['bbox'], 'layers': a['layers'], 'mode': 'immediate'}
         for a in immediate_assets]
        + [{'id': a['id'], 'kind': a['kind'], 'bbox': a['bbox'], 'layers': a['layers'], 'mode': 'deferred_after_text'}
           for a in deferred_assets]
    )
    payload = {
        'status': 'PREPARED',
        'page': page,
        'image_size': [w_img, h_img],
        'scene_initialized': scene_initialized,
        'scene_already_existed': bool(existing),
        'layerd_status': layerd_status,
        'layer_images': layer_images,
        'sam_available': bool(os.environ.get('IMAGE2PPT_SAM_URL') or os.environ.get('SAM_URL')),
        'text_count': len(measured_items),
        'resolved_text_count': resolved_count,
        'unresolved_text_count': len(measured_items) - resolved_count,
        'asset_count': len(asset_summaries),
        'immediate_asset_count': len(immediate_assets),
        'deferred_asset_count': len(deferred_assets),
        'assets': asset_summaries,
        'backing_summary': {'FLAT': counts.get('FLAT', 0), 'GRADIENT': counts.get('GRADIENT', 0),
                            'COMPLEX': counts.get('COMPLEX', 0)},
        'styles': final_styles,
        'items': measured_items,
    }
    write(directory / 'page-evidence.json', payload)
    evidence_ref = state.register(directory / 'page-evidence.json', 'page_evidence_summary')
    return {**payload, 'evidence': evidence_ref}


def _threadsafe_inpaint(raw_inpaint):
    if raw_inpaint is None:
        return None
    lock = threading.Lock()

    def wrapped(source, mask):
        with lock:
            return raw_inpaint(source, mask)

    wrapped.config = getattr(raw_inpaint, 'config', {})
    return wrapped


def _collect_office_defects(office_result: dict, graph: SceneGraph, staged_ids: list[str]) -> tuple[dict, set[str]]:
    from .office import frame_feedback
    frame_repairs = {}
    bad_ids: set[str] = set()
    staged_set = set(staged_ids)
    for page_rows in office_result.get('residuals', []):
        for rec in page_rows:
            if rec.get('id') not in staged_set or rec.get('residual_check_status') == 'PASS':
                continue
            if rec.get('reason') == 'missing source-owner or actual-delete evidence':
                continue
            bad_ids.add(rec['id'])
    for rec in office_result.get('replacements', {}).get('records', []):
        oid = rec.get('id')
        if oid not in staged_set:
            continue
        ok = (rec.get('content_matches') and rec.get('siblings_unchanged')
              and rec.get('frame_unchanged') and rec.get('outside_pixels', 0) == 0
              and not rec.get('layout', {}).get('overflow')
              and not rec.get('layout', {}).get('unexpected_wrap'))
        if not ok:
            fb = frame_feedback(office_result, graph, oid)
            if fb and oid not in bad_ids:
                frame_repairs[oid] = fb
            else:
                bad_ids.add(oid)
    failed_oid = (office_result.get('office') or {}).get('failed_object_id')
    if failed_oid in staged_set and failed_oid not in frame_repairs:
        bad_ids.add(failed_oid)
    return frame_repairs, bad_ids


def batch_reconstruct_page(service, state, current, directory: Path, *, page: int,
                           object_ids: list[str] | None = None, include_assets: bool = True,
                           include_containers: bool = True, use_inpaint: bool = True,
                           verify_office: bool = True) -> dict:
    """Parallel local text trials + sequential text staging + whole-asset extraction/staging + single batch Office check."""
    from ..object_stage import infer_sam
    from ..parallel_config import ROOT
    from ..parallel_pipeline import effective_texts
    from ..recovery.adaptive_policy import AdaptiveBudget
    from ..recovery.agent_asset_stage import stage_asset_candidate
    from ..recovery.asset_retry import retry_asset_extraction
    from ..recovery.candidate_stage import stage_background
    from ..recovery.product_pipeline import failure as classify_exc
    from ..recovery.region_candidates import adaptive_candidates
    from .background import inpaint_service
    from .compiler import compile_local

    row = service.page(current, page)
    initial_graph = service.graph(row)
    if 'fonts' not in current:
        service.list_fonts(state, current, directory / 'fonts')
    fonts = json.loads(Path(current['fonts']).read_text('utf8'))
    source_path = row['snapshot']['path']
    initial_baseline = Path(row['ir']).parent
    wanted_filter = set(object_ids) if object_ids is not None else None
    sam_enabled = bool(os.environ.get('IMAGE2PPT_SAM_URL') or os.environ.get('SAM_URL'))

    failed_regions: list[dict] = []

    # Phase 1: Parallel local text candidate trials against initial_baseline.
    text_targets = [t for t in initial_graph.text.texts
                    if t.id not in initial_graph.selected
                    and initial_graph.ownership.get(t.id) == 'RESOLVED'
                    and (wanted_filter is None or t.id in wanted_filter)]

    request = json.loads((state.job / 'request.json').read_text('utf8'))
    budget = AdaptiveBudget(Store(state.job / 'state'), request.get('adaptive_budget'))
    config_file = ROOT / '.runtime/parallel-services.json'
    lama_cfg = json.loads(config_file.read_text('utf8')).get('lama') if (use_inpaint and config_file.exists()) else None
    raw_inpaint = inpaint_service(lama_cfg, state.job / 'cache/lama', budget) if lama_cfg else None
    safe_inpaint = _threadsafe_inpaint(raw_inpaint)

    def _probe_text(text_obj):
        t_dir = directory / 'trials' / f'{text_obj.id}-{key(text_obj.id)[:8]}'
        stream = adaptive_candidates(initial_graph, text_obj.id, source_path, row.get('layers'),
                                     t_dir, baseline_directory=initial_baseline, inpaint=safe_inpaint)
        attempts = []
        try:
            for cand_row in stream:
                attempts.append({'candidate': cand_row['candidate'], 'status': cand_row['status']})
                if cand_row['status'] == 'PRECHECK_PASS':
                    return {'object': text_obj, 'status': 'PRECHECK_PASS',
                            'trial_dir': t_dir / cand_row['candidate'], 'method': cand_row['method'], 'attempts': attempts}
            return {'object': text_obj, 'status': 'BLOCKED', 'failure': 'BACKGROUND_REPAIR_FAILED',
                    'reason': 'no local or inpaint candidate passed precheck', 'trial_root': t_dir, 'attempts': attempts}
        except Exception as exc:
            return {'object': text_obj, 'status': 'BLOCKED',
                    'failure': getattr(exc, 'code', classify_exc(exc, 'generation')),
                    'reason': str(exc), 'trial_root': t_dir, 'attempts': attempts}
        finally:
            stream.close()

    passed_trials = []
    if text_targets:
        workers = min(6, max(1, len(text_targets)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_probe_text, text_targets))
        for item in outcomes:
            t_obj = item['object']
            if item['status'] == 'PRECHECK_PASS':
                passed_trials.append(item)
            else:
                ev = []
                root_dir = item.get('trial_root')
                if root_dir and root_dir.exists():
                    for png in sorted(root_dir.rglob('*.png')):
                        if png.name in ('source_crop.png', 'written_crop.png'):
                            ev.append(state.register(png, 'candidate_observation')['id'])
                failed_regions.append({'object_id': t_obj.id, 'kind': 'text', 'text': t_obj.text, 'bbox': t_obj.bbox,
                                       'failure': item['failure'], 'reason': item['reason'], 'evidence_ids': ev[:4]})

    sam_cache: dict[str, tuple[Path, Path] | None] = {}

    # Phase 2 & 3: Sequential text staging -> deferred badge merge -> whole-asset extraction & staging.
    def _chain_stage(trials_list, tag: str, frame_overrides: dict, skip_ids: set[str]):
        cur_base = initial_baseline
        staged_ids: list[str] = []
        stage_errors: list[dict] = []
        for idx, tr in enumerate(trials_list):
            t_obj = tr['object']
            if t_obj.id in skip_ids:
                continue
            step_out = directory / tag / f'step-{idx:03d}-{t_obj.id}'
            frame_opt = {'size': frame_overrides[t_obj.id], 'method': 'office_batch_auto_frame_repair/1'} if t_obj.id in frame_overrides else None
            try:
                ir = stage_background(tr['trial_dir'], initial_graph, cur_base, fonts, step_out, frame_minimum=frame_opt)
                cur_base = ir.parent
                staged_ids.append(t_obj.id)
            except Exception as exc:
                msg = str(exc)
                code = 'FONT_LAYOUT_FAILED' if 'frame allowance' in msg else ('OWNER_CONFLICT' if 'stable owner' in msg else 'STAGE_REJECTED')
                stage_errors.append({'object_id': t_obj.id, 'kind': 'text', 'text': t_obj.text,
                                     'bbox': t_obj.bbox, 'failure': code, 'reason': msg})

        if include_assets and row.get('layers'):
            cur_graph = SceneGraph.model_validate_json((cur_base / 'scene.json').read_text('utf8'))
            deferred_path = current.get('deferred_assets', {}).get(str(page))
            if deferred_path and Path(deferred_path).is_file():
                deferred_list = json.loads(Path(deferred_path).read_text('utf8'))
                existing_ids = {a.id for a in cur_graph.asset.objects}
                to_add: list[ClosedObject] = []
                for raw_obj in deferred_list:
                    oid = raw_obj.get('id')
                    if not oid or oid in existing_ids or oid in skip_ids:
                        continue
                    if wanted_filter is not None and oid not in wanted_filter:
                        continue
                    d_box = raw_obj['bbox']
                    d_area = max(1.0, d_box[2] * d_box[3])
                    if any(intersection(a.bbox, d_box) >= 0.30 * min(max(1.0, a.bbox[2] * a.bbox[3]), d_area)
                           for a in cur_graph.asset.objects):
                        continue
                    to_add.append(ClosedObject.model_validate(raw_obj))
                if to_add:
                    prep_base = directory / tag / 'deferred-scene-prep'
                    shutil.copytree(cur_base, prep_base)
                    for obj in to_add:
                        cur_graph.asset.objects.append(obj)
                        cur_graph.ownership[obj.id] = 'RESOLVED'
                        for t in cur_graph.text.texts:
                            if intersection(t.bbox, obj.bbox) > 0:
                                cur_graph.relations.append(Relation(
                                    source=t.id, target=obj.id, kind='label_for',
                                    status='AMBIGUOUS', evidence=['source', 'ocr', 'asset-intent']))
                    write(prep_base / 'scene.json', cur_graph.model_dump())
                    cur_base = prep_base

            cur_graph = SceneGraph.model_validate_json((cur_base / 'scene.json').read_text('utf8'))
            asset_targets = [a for a in cur_graph.asset.objects
                             if a.id not in cur_graph.selected
                             and cur_graph.ownership.get(a.id) == 'RESOLVED'
                             and a.id not in skip_ids
                             and (wanted_filter is None or a.id in wanted_filter)]
            for idx, asset in enumerate(asset_targets):
                step_dir = directory / tag / f'asset-{idx:03d}-{key(asset.id)[:12]}'
                step_dir.mkdir(parents=True, exist_ok=True)
                active_graph = SceneGraph.model_validate_json((cur_base / 'scene.json').read_text('utf8'))
                try:
                    res = retry_asset_extraction(active_graph, asset.id, source_path, row['layers'],
                                                 cur_base, step_dir / 'extract', strategy='verified-backing')
                    if res.get('status') != 'CANDIDATE_READY' and sam_enabled:
                        if asset.id not in sam_cache:
                            sam_root = directory / 'sam-cache' / key(asset.id)[:12]
                            sam_root.mkdir(parents=True, exist_ok=True)
                            plan_file = sam_root / 'sam-plan.json'
                            write(plan_file, {
                                'schema': 'LogicalObjects/2',
                                'source_sha256': active_graph.source_sha256,
                                'image_size': active_graph.image_size,
                                'objects': [a.model_dump() for a in active_graph.asset.objects],
                                'limitations': [],
                            })
                            try:
                                sam_dir = infer_sam(source_path, plan_file, sam_root / 'sam', 'sam3', [asset.id])
                            except Exception:
                                sam_dir = None
                            sam_cache[asset.id] = (plan_file, Path(sam_dir)) if sam_dir else None
                        cached_sam = sam_cache.get(asset.id)
                        if cached_sam is not None:
                            res_sam = retry_asset_extraction(
                                active_graph, asset.id, source_path, row['layers'],
                                cur_base, step_dir / 'extract-sam', strategy='sam',
                                sam_plan=cached_sam[0], sam_directory=cached_sam[1])
                            if res_sam.get('status') == 'CANDIDATE_READY':
                                res = res_sam
                    if res.get('status') == 'CANDIDATE_READY':
                        ir = stage_asset_candidate(res['candidate_directory'], active_graph, cur_base,
                                                   step_dir / 'staged', layer_directory=row['layers'])
                        cur_base = ir.parent
                        staged_ids.append(asset.id)
                    else:
                        stage_errors.append({'object_id': asset.id, 'kind': 'asset', 'bbox': asset.bbox,
                                             'failure': 'ASSET_EXTRACTION_UNCERTAIN', 'reason': res.get('reason', 'unverified')})
                except Exception as exc:
                    stage_errors.append({'object_id': asset.id, 'kind': 'asset', 'bbox': asset.bbox,
                                         'failure': classify_exc(exc, 'generation'), 'reason': str(exc)})
        if include_assets and include_containers and staged_ids and wanted_filter is None:
            from ..recovery.container_stage import stage_containers_and_surfaces
            cont_dir = directory / tag / 'containers-staged'
            try:
                cont_ir, cont_ids = stage_containers_and_surfaces(cur_base, cont_dir, skip_ids=skip_ids)
                if cont_ids:
                    cur_base = cont_ir.parent
                    staged_ids.extend(cont_ids)
            except Exception as exc:
                stage_errors.append({'object_id': 'containers', 'kind': 'surface', 'bbox': [],
                                     'failure': classify_exc(exc, 'generation'), 'reason': str(exc)})
        return cur_base, staged_ids, stage_errors

    final_base, staged_ids, stage_errors = _chain_stage(passed_trials, 'stage-pass1', {}, set())
    failed_regions.extend(stage_errors)
    pass1_failed_ids = {err['object_id'] for err in stage_errors}

    if not staged_ids:
        return {'status': 'NO_CANDIDATES_STAGED', 'page': page, 'committed_count': 0,
                'committed_object_ids': [], 'failed_count': len(failed_regions), 'failed_regions': failed_regions}

    final_ir = final_base / 'slideir.json'
    batch_cid = 'c-' + key(str(directory.relative_to(state.job)) + ':batch')[:24]

    if not verify_office:
        cand = {
            'id': batch_cid, 'page': page, 'object_id': staged_ids[0], 'object_ids': staged_ids,
            'kind': 'batch', 'status': 'STAGED_LOCAL', 'ir': str(final_ir),
            'parent_ir_sha256': digest(row['ir']), 'parent_scene_sha256': digest(row['scene']),
            'strategy': 'batch_backbone', 'code': service.code_binding(),
            'fingerprint': key({'batch': staged_ids, 'ir': digest(final_ir)}),
            'files': {str(p): digest(p) for p in final_base.rglob('*') if p.is_file()},
        }
        current['candidates'][batch_cid] = cand
        return {'status': 'STAGED_LOCAL', 'candidate_id': batch_cid, 'page': page,
                'staged_count': len(staged_ids), 'staged_object_ids': staged_ids,
                'committed_count': 0, 'failed_count': len(failed_regions), 'failed_regions': failed_regions}

    # Phase 4: Single batch compilation + Office verification (+ 1-pass auto frame/culprit repair).
    def _compile_and_check(ir_path: Path, tag: str):
        staged_g = SceneGraph.model_validate_json((ir_path.parent / 'scene.json').read_text('utf8'))
        with lease('compiler'):
            art = compile_local([ir_path], directory / f'compiled-{tag}', [effective_texts(staged_g, ir_path)])
        off = service.office_check(state, art['pptx'], [ir_path], directory / f'office-{tag}')
        return staged_g, art, off

    staged_graph, artifacts, office_res = _compile_and_check(final_ir, 'pass1')
    auto_repaired_frames = []

    if office_res.get('status') == 'REVIEW_REQUIRED':
        frame_repairs, bad_ids = _collect_office_defects(office_res, staged_graph, staged_ids)
        if frame_repairs or (bad_ids and len(bad_ids) < len(staged_ids)):
            overrides = {oid: fb['measured_frame_minimum_px'] for oid, fb in frame_repairs.items()}
            repaired_base, repaired_ids, retry_errors = _chain_stage(
                passed_trials, 'stage-pass2', overrides, bad_ids | pass1_failed_ids)
            for oid in bad_ids:
                t_match = next((t for t in staged_graph.text.texts if t.id == oid), None)
                a_match = next((a for a in staged_graph.asset.objects if a.id == oid), None)
                s_match = next((s for s in staged_graph.surface.shapes if s.id == oid), None)
                failed_regions.append({
                    'object_id': oid,
                    'kind': 'text' if t_match else ('surface' if s_match else 'asset'),
                    'text': t_match.text if t_match else '',
                    'bbox': t_match.bbox if t_match else (s_match.bbox if s_match else (a_match.bbox if a_match else [])),
                    'failure': 'OFFICE_RESIDUAL_OR_EDIT_FAILED',
                    'reason': 'isolated during batch Office verification; inspect locally',
                })
            for err in retry_errors:
                if err['object_id'] in frame_repairs:
                    err['measured_frame_minimum_px'] = frame_repairs[err['object_id']]['measured_frame_minimum_px']
                failed_regions.append(err)
            if repaired_ids:
                final_base = repaired_base
                final_ir = final_base / 'slideir.json'
                staged_ids = repaired_ids
                auto_repaired_frames = [oid for oid in overrides if oid in staged_ids]
                staged_graph, artifacts, office_res = _compile_and_check(final_ir, 'pass2')

    if office_res.get('status') == 'PASS':
        row.update(ir=str(final_ir), scene=str(final_ir.parent / 'scene.json'))
        cand = {
            'id': batch_cid, 'page': page, 'object_id': staged_ids[0], 'object_ids': staged_ids,
            'kind': 'batch', 'status': 'COMMITTED', 'ir': str(final_ir),
            'parent_ir_sha256': digest(initial_baseline / 'slideir.json'),
            'parent_scene_sha256': digest(initial_baseline / 'scene.json'),
            'strategy': 'batch_backbone', 'code': service.code_binding(),
            'fingerprint': key({'batch': staged_ids, 'ir': digest(final_ir)}),
            'validation': office_res,
            'files': {str(p): digest(p) for p in final_base.rglob('*') if p.is_file()},
        }
        current['candidates'][batch_cid] = cand
        current['last_verified_bundle'] = {
            'irs': [digest(final_ir)],
            'pptx': str(artifacts['pptx']),
            'pptx_sha256': digest(artifacts['pptx']),
        }
        current.pop('composition', None)
        current.pop('visual_review', None)
        return {
            'status': 'COMMITTED', 'office_status': 'PASS', 'already_committed': True,
            'candidate_id': batch_cid, 'page': page,
            'committed_count': len(staged_ids), 'committed_object_ids': staged_ids,
            'auto_repaired_frame_ids': auto_repaired_frames,
            'failed_count': len(failed_regions), 'failed_regions': failed_regions,
            'next_step': 'Objects in committed_object_ids are ALREADY committed to the page scene (do not call commit_candidate). Proceed to compose_delivery -> validate_office -> get_evidence -> record_visual_review -> finalize_job, or repair specific high-value failed_regions first.',
        }

    cand_status = 'STAGED_LOCAL' if office_res.get('status') == 'UNKNOWN' else 'REVIEW_REQUIRED'
    cand = {
        'id': batch_cid, 'page': page, 'object_id': staged_ids[0], 'object_ids': staged_ids,
        'kind': 'batch', 'status': cand_status, 'ir': str(final_ir),
        'parent_ir_sha256': digest(row['ir']), 'parent_scene_sha256': digest(row['scene']),
        'strategy': 'batch_backbone', 'code': service.code_binding(),
        'fingerprint': key({'batch': staged_ids, 'ir': digest(final_ir)}),
        'validation': office_res,
        'files': {str(p): digest(p) for p in final_base.rglob('*') if p.is_file()},
    }
    current['candidates'][batch_cid] = cand
    return {
        'status': 'OFFICE_UNAVAILABLE' if office_res.get('status') == 'UNKNOWN' else 'REVIEW_REQUIRED',
        'candidate_id': batch_cid, 'page': page,
        'staged_count': len(staged_ids), 'staged_object_ids': staged_ids,
        'committed_count': 0, 'office_status': office_res.get('status'),
        'failed_count': len(failed_regions), 'failed_regions': failed_regions,
        'next_step': f'Batch candidate {batch_cid} is staged but not yet committed; call commit_candidate(candidate_id="{batch_cid}") to validate and commit.',
    }
