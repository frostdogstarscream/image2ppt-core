"""Deterministic perception preparation and batch regional reconstruction backbone.

Preserves all source-pixel ownership, peer protection, Office editing, and
hash-bound visual review invariants while reducing 100+ object slides from
hundreds of Agent tool turns to a single batch backbone pass plus targeted repair.
"""
from __future__ import annotations

from collections import Counter
import subprocess
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
from ..parallel_fusion import fuse_incremental_scene, is_decorative_subspan
from ..parallel_store import Store, key
from ..parallel_typography import calibrate, snap_ooxml_size_px, measure_text_appearance
from ..resource_lease import lease


def _merge_structural_ocr_items(
    source: np.ndarray,
    raw_items: list[dict],
    asset_records: list[dict],
    extra_logo_boxes: list[list[float]] | None = None,
) -> tuple[list[dict], dict]:
    """Perform pre-scene structural grouping and relation analysis (Astra Step 2).

    Identifies top-right brand logo lockups, merges split same-row titles, merges
    consecutive same-card vertical lines into multi-line paragraphs, and flags
    dual-color colon lines before `create_scene` locks individual single-line bboxes.
    """
    h_img, w_img = source.shape[:2]
    items = []
    for it in raw_items:
        rgb = [int(it['color_hex'][i:i + 2], 16) for i in (0, 2, 4)]
        items.append({
            **it,
            'ocr_line_ids': [it['ocr_line_id']],
            'rgb': rgb,
            'align': 'left',
            'line_spacing_px': None,
            'lines': [{'id': it['ocr_line_id'], 'text': it['text'], 'bbox': list(it['bbox']), 'rgb': rgb, 'est_size_px': it['est_size_px'], 'bold': it['bold'], 'stripped_affix': it.get('stripped_affix', '')}],
            'is_header_merged': False,
        })

    # 1. Detect top-right brand logo lockup lettering & asset-overlapping logo texts.
    logo_ids: set[str] = set()
    logo_assets = [
        a for a in asset_records
        if a.get('kind') == 'logo'
    ]
    for it in items:
        x, y, w, h = it['bbox']
        in_top_right = x >= 0.80 * w_img and y <= 0.14 * h_img and w <= 0.18 * w_img
        hits_logo = any(intersection(it['bbox'], la['bbox']) > 0 for la in logo_assets) or any(
            intersection(it['bbox'], lb) > 0 for lb in (extra_logo_boxes or ())
        )
        paired_wordmark = in_top_right and any(
            peer['id'] != it['id'] and not any(c.isdigit() for c in it['text']+peer['text'])
            and abs(peer['bbox'][0]-x) <= 3 and max(abs(a-b) for a,b in zip(peer['rgb'],it['rgb'])) <= 30
            and intersection([x-2,y-6,w+4,h+12],peer['bbox']) > 0 for peer in items)
        if paired_wordmark or hits_logo:
            logo_ids.add(it['id'])
    # Propagate logo lockup flag to immediate vertical partner lines in top-right quadrant.
    changed = True
    while changed:
        changed = False
        for it in items:
            if it['id'] in logo_ids or any(c.isdigit() for c in it['text']):
                continue
            x, y, w, h = it['bbox']
            if x < 0.78 * w_img or y > 0.18 * h_img or w > 0.20 * w_img:
                continue
            for lid in list(logo_ids):
                other = next(o for o in items if o['id'] == lid)
                if intersection([x - 4, y - 6, w + 8, h + 12], other['bbox']) > 0:
                    logo_ids.add(it['id'])
                    changed = True
                    break

    # 2. Detect dual-color colon lines (e.g. navy label prefix + gray suffix) before merging.
    dual_color_ids: set[str] = set()
    dual_color_colon_lines: list[dict] = []
    for it in items:
        if it['id'] in logo_ids:
            continue
        txt = it['text']
        colon_pos = max(txt.find('：'), txt.find(':'))
        if 0 < colon_pos < len(txt) - 1:
            x, y, w, h = [round(v) for v in it['bbox']]
            if w >= 40 and h >= 10:
                split_x = max(6, min(w - 6, round(w * min(0.45, (colon_pos + 1.0) / max(1, len(txt))))))
                left_app = measure_text_appearance(source, [x, y, split_x, h])
                right_app = measure_text_appearance(source, [x + split_x, y, w - split_x, h])
                if left_app and right_app:
                    c_diff = int(np.abs(left_app['color_rgb'] - right_app['color_rgb']).max())
                    if c_diff > 35:
                        dual_color_ids.add(it['id'])
                        dual_color_colon_lines.append({
                            'id': it['id'],
                            'text': txt,
                            'prefix_color': left_app['color_hex'],
                            'suffix_color': right_app['color_hex'],
                            'suggestion': 'keep_in_source_plate_or_split_spans',
                        })

    def _union_bbox(b1: list[float], b2: list[float]) -> list[float]:
        nx0 = min(b1[0], b2[0])
        ny0 = min(b1[1], b2[1])
        nx1 = max(b1[0] + b1[2], b2[0] + b2[2])
        ny1 = max(b1[1] + b1[3], b2[1] + b2[3])
        return [round(nx0, 2), round(ny0, 2), round(nx1 - nx0, 2), round(ny1 - ny0, 2)]

    # 3. Horizontal split-header merge (only adjacent headers without huge visual whitespace gap).
    merged_headers: list[dict] = []
    items.sort(key=lambda it: (it['bbox'][1], it['bbox'][0]))
    after_header: list[dict] = []
    i = 0
    while i < len(items):
        cur = items[i]
        while i + 1 < len(items):
            nxt = items[i + 1]
            if (cur['id'] in logo_ids or nxt['id'] in logo_ids
                    or cur.get('stripped_affix') or nxt.get('stripped_affix')):
                break
            if min(cur['est_size_px'], nxt['est_size_px']) < 24.0:
                break
            cy1 = cur['bbox'][1] + cur['bbox'][3] / 2.0
            cy2 = nxt['bbox'][1] + nxt['bbox'][3] / 2.0
            gap_x = nxt['bbox'][0] - (cur['bbox'][0] + cur['bbox'][2])
            rgb_diff = max(abs(a - b) for a, b in zip(cur['rgb'], nxt['rgb']))
            # Measure actual ink gap if OCR boxes slightly overlap so numbered titles with a wide bar gap don't shift left.
            ink_gap_ok = True
            if gap_x < -4.0 or gap_x > 6.0:
                x_band0 = max(0, round(min(cur['bbox'][0] + cur['bbox'][2], nxt['bbox'][0]) - 24))
                x_band1 = min(w_img, round(max(cur['bbox'][0] + cur['bbox'][2], nxt['bbox'][0]) + 24))
                # Quick check: skip merging tiny <=3 char section number + long title when visual gap exceeds 30px
                if len(cur['text'].strip()) <= 3 and cur['text'].strip().isdigit():
                    ax0, ay0, aw, ah = [max(0, round(v)) for v in cur['bbox']]
                    bx0, by0, bw, bh = [max(0, round(v)) for v in nxt['bbox']]
                    c_crop = source[ay0:min(h_img, ay0 + ah), ax0:min(w_img, ax0 + aw)].astype(float)
                    b_crop = source[by0:min(h_img, by0 + bh), bx0:min(w_img, bx0 + bw)].astype(float)
                    if c_crop.size and b_crop.size:
                        c_bg = np.median(c_crop[0], axis=0)
                        b_bg = np.median(b_crop[0], axis=0)
                        c_cols = np.nonzero((np.abs(c_crop - c_bg).max(2) > 25.0).any(0))[0]
                        b_cols = np.nonzero((np.abs(b_crop - b_bg).max(2) > 25.0).any(0))[0]
                        if len(c_cols) and len(b_cols):
                            real_gap = (bx0 + int(b_cols[0])) - (ax0 + int(c_cols[-1]) + 1)
                            if real_gap > 32:
                                ink_gap_ok = False
            gap_band = [
                min(cur['bbox'][0] + cur['bbox'][2], nxt['bbox'][0]) - 2,
                min(cur['bbox'][1], nxt['bbox'][1]),
                max(4.0, abs(gap_x) + 4),
                max(cur['bbox'][3], nxt['bbox'][3]),
            ]
            asset_between = any(intersection(gap_band, ar['bbox']) > 0 for ar in asset_records)
            if (abs(cy1 - cy2) <= 8.0 and -24.0 <= gap_x <= 36.0 and rgb_diff <= 30
                    and ink_gap_ok and not asset_between):
                join_sep = ''
                tail_head = (cur['text'][-1:] + nxt['text'][:1])
                if gap_x > 6.0 and '|' not in tail_head and '丨' not in tail_head:
                    join_sep = ' '
                cur = {
                    **cur,
                    'text': cur['text'] + join_sep + nxt['text'],
                    'bbox': _union_bbox(cur['bbox'], nxt['bbox']),
                    'ocr_line_ids': cur['ocr_line_ids'] + nxt['ocr_line_ids'],
                    'lines': cur['lines'] + nxt['lines'],
                    'est_size_px': snap_ooxml_size_px(max(cur['est_size_px'], nxt['est_size_px'])),
                    'is_header_merged': True,
                }
                i += 1
                continue
            break
        if cur['is_header_merged']:
            merged_headers.append({
                'source_ocr_ids': list(cur['ocr_line_ids']),
                'merged_text': cur['text'],
                'bbox': cur['bbox'],
            })
        after_header.append(cur)
        i += 1

    # 4. Same-card vertical paragraph merge.
    merged_paragraphs: list[dict] = []
    consumed = [False] * len(after_header)
    final_items: list[dict] = []

    def _can_merge_vertical(cur_item: dict, cand: dict) -> tuple[bool, str]:
        if (cur_item['id'] in logo_ids or cand['id'] in logo_ids
                or cur_item['id'] in dual_color_ids or cand['id'] in dual_color_ids
                or cur_item.get('stripped_affix') or cand.get('stripped_affix')
                or cur_item.get('is_header_merged') or cand.get('is_header_merged')):
            return False, 'left'
        if cur_item['est_size_px'] > 23.0 or cand['est_size_px'] > 23.0:
            return False, 'left'
        last_line = cur_item['lines'][-1]
        if abs(last_line['est_size_px'] - cand['est_size_px']) > 2.4:
            return False, 'left'
        h_a, h_b = max(1.0, last_line['bbox'][3]), max(1.0, cand['bbox'][3])
        if max(h_a, h_b) / min(h_a, h_b) > 1.25:
            return False, 'left'
        if max(abs(a - b) for a, b in zip(last_line['rgb'], cand['rgb'])) > 26:
            return False, 'left'
        ax, ay, aw, ah = last_line['bbox']
        bx, by, bw, bh = cand['bbox']
        gap_y = by - (ay + ah)
        avg_h = 0.5 * (ah + bh)
        if not (-4.0 <= gap_y <= max(9.0, 0.55 * avg_h)):
            return False, 'left'
        x_ov = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
        min_w = max(1.0, min(aw, bw))
        if x_ov < 0.35 * min_w:
            return False, 'left'
        avg_size = 0.5 * (last_line['est_size_px'] + cand['est_size_px'])
        left_aligned = abs(ax - bx) <= max(12.0, 0.75 * avg_size)
        cx_a, cx_b = ax + aw / 2.0, bx + bw / 2.0
        center_aligned = abs(cx_a - cx_b) <= 14.0 and max(aw, bw) <= 0.32 * w_img
        if not (left_aligned or center_aligned):
            return False, 'left'
        # Do not merge across an asset or horizontal divider line.
        x_min = max(0, int(round(min(ax, bx))))
        x_max = min(w_img, int(round(max(ax + aw, bx + bw))))
        strip_box = [float(x_min), ay + ah - 1.0, float(max(1, x_max - x_min)), max(2.0, gap_y + 2.0)]
        if any(intersection(strip_box, ar['bbox']) > 0 for ar in asset_records):
            return False, 'left'
        if gap_y >= 2.0 and x_max > x_min + 4:
            sy0 = max(0, int(round(ay + ah)))
            sy1 = min(h_img, int(round(by)))
            if sy1 > sy0:
                strip = source[sy0:sy1, x_min:x_max].astype(float)
                med_col = np.median(strip.reshape(-1, 3), axis=0)
                dev = np.abs(strip.reshape(-1, 3) - med_col).max(axis=1)
                if float(np.percentile(dev, 90)) > 16.0:
                    return False, 'left'
        return True, ('center' if (center_aligned and not left_aligned) else 'left')

    for idx_a, base in enumerate(after_header):
        if consumed[idx_a]:
            continue
        consumed[idx_a] = True
        cur = dict(base)
        pitches: list[float] = []
        aligns: list[str] = []
        while True:
            best_j = None
            best_align = 'left'
            cur_last = cur['lines'][-1]['bbox']
            cur_center_y = cur_last[1] + cur_last[3]
            for idx_b in range(idx_a + 1, len(after_header)):
                if consumed[idx_b]:
                    continue
                cand = after_header[idx_b]
                if cand['bbox'][1] - cur_center_y > max(24.0, cur_last[3] * 1.2):
                    break
                ok, aln = _can_merge_vertical(cur, cand)
                if ok:
                    best_j = idx_b
                    best_align = aln
                    break
            if best_j is None:
                break
            nxt = after_header[best_j]
            consumed[best_j] = True
            pitches.append(max(cur['est_size_px'] * 1.05, nxt['bbox'][1] - cur['lines'][-1]['bbox'][1]))
            aligns.append(best_align)
            cur['text'] = cur['text'] + '\n' + nxt['text']
            cur['bbox'] = _union_bbox(cur['bbox'], nxt['bbox'])
            cur['ocr_line_ids'] = cur['ocr_line_ids'] + nxt['ocr_line_ids']
            cur['lines'] = cur['lines'] + nxt['lines']
        if len(cur['lines']) > 1:
            cur['est_size_px'] = snap_ooxml_size_px(float(np.mean([ln['est_size_px'] for ln in cur['lines']])))
            cur['bold'] = bool(cur['est_size_px'] >= 20.0)
            avg_pitch = float(np.median(pitches)) if pitches else cur['est_size_px'] * 1.2
            cur['line_spacing_px'] = round(max(cur['est_size_px'] * 1.05, avg_pitch), 2)
            cur['align'] = 'center' if all(a == 'center' for a in aligns) else 'left'
            merged_paragraphs.append({
                'source_ocr_ids': list(cur['ocr_line_ids']),
                'line_count': len(cur['lines']),
                'line_spacing_px': cur['line_spacing_px'],
                'align': cur['align'],
                'text_preview': cur['text'].replace('\n', ' / '),
                'bbox': cur['bbox'],
            })
        final_items.append(cur)

    # Assign contiguous deterministic t-001..t-NNN IDs after merging and update relation refs.
    old_to_new: dict[str, str] = {}
    cleaned: list[dict] = []
    for new_idx, it in enumerate(final_items, 1):
        new_id = f't-{new_idx:03d}'
        old_to_new[it['id']] = new_id
        for hdr in merged_headers:
            if hdr['source_ocr_ids'] == it['ocr_line_ids']:
                hdr['id'] = new_id
        for para in merged_paragraphs:
            if para['source_ocr_ids'] == it['ocr_line_ids']:
                para['id'] = new_id
        cleaned.append({
            **{k: v for k, v in it.items() if k not in ('lines', 'is_header_merged')},
            'id': new_id,
        })

    logo_lockup_candidates = [
        {
            'id': old_to_new[it['id']],
            'text': it['text'],
            'bbox': it['bbox'],
            'suggestion': 'remove_from_text_draft_if_brand_logo',
        }
        for it in items
        if it['id'] in logo_ids and it['id'] in old_to_new
    ]
    for dcol in dual_color_colon_lines:
        dcol['id'] = old_to_new.get(dcol['id'], dcol['id'])

    scene_relations = {
        'raw_ocr_lines': len(raw_items),
        'merged_text_objects': len(cleaned),
        'merged_paragraphs': merged_paragraphs,
        'merged_headers': merged_headers,
        'logo_lockup_candidates': logo_lockup_candidates,
        'dual_color_colon_lines': dual_color_colon_lines,
    }
    return cleaned, scene_relations


def _classify_failed_region(
    entry: dict,
    graph: SceneGraph,
    logo_candidate_ids: set[str] | None = None,
    staged_crop_boxes: list[tuple[str, list[float]]] | None = None,
) -> dict:
    """Attach prescriptive failure taxonomy and repair cap (Astra Step 3)."""
    oid = entry.get('object_id', '')
    kind = entry.get('kind', '')
    failure = entry.get('failure', '')
    bbox = entry.get('bbox') or []
    logo_ids = logo_candidate_ids or set()

    peers: list[str] = []
    same_style_text_peer = False
    asset_peer = False
    text_by_id = {t.id: t for t in graph.text.texts}
    self_text = text_by_id.get(oid)

    if len(bbox) == 4:
        pbox = [bbox[0] - 4.0, bbox[1] - 4.0, bbox[2] + 8.0, bbox[3] + 8.0]
        for t in graph.text.texts:
            if t.id == oid:
                continue
            if intersection(pbox, t.bbox) > 0:
                peers.append(t.id)
                if self_text and t.style == self_text.style:
                    same_style_text_peer = True
        for a in graph.asset.objects:
            if a.id == oid:
                continue
            if intersection(pbox, a.bbox) > 0:
                peers.append(a.id)
                asset_peer = True

    in_top_right_logo = (
        oid in logo_ids
        or (kind == 'text' and len(bbox) == 4 and any(
            a.kind == 'logo' and intersection(bbox, a.bbox) >= 0.8 * max(1.0, bbox[2] * bbox[3])
            for a in graph.asset.objects))
    )

    if kind == 'asset' or kind == 'surface':
        category, action, cap = 'UNVERIFIED_ASSET_CLOSURE', 'retain_in_source_plate', 0
    elif in_top_right_logo:
        category, action, cap = 'LOGO_OR_INTRINSIC_TEXT', 'retain_in_source_plate', 0
    elif failure == 'FONT_LAYOUT_FAILED':
        category, action, cap = 'FONT_FRAME_NEIGHBOR_COLLISION', 'fit_text_layout_or_reduce_size_1px', 1
    elif failure == 'INSUFFICIENT_CONTEXT':
        category, action, cap = 'INSUFFICIENT_LATERAL_CONTEXT', 'fit_background_context_36', 1
    elif failure == 'PROTECTED_NEIGHBOR':
        if asset_peer and not same_style_text_peer:
            category, action, cap = 'ADJACENT_ASSET_OVERLAP', 'shrink_text_bbox_away_from_asset', 1
        elif same_style_text_peer or peers:
            category, action, cap = 'PARAGRAPH_OR_PEER_OVERLAP', 'merge_with_peer_or_shrink_bbox_2px', 1
        else:
            category, action, cap = 'PROTECTED_NEIGHBOR_CONFLICT', 'shrink_text_bbox_2px', 1
    elif failure in ('MASK_INCOMPLETE', 'BACKGROUND_REPAIR_FAILED', 'OWNER_CONFLICT'):
        category, action, cap = 'COMPLEX_OR_DUAL_COLOR_BACKING', 'inspect_local_backing_and_retry', 1
    else:
        category, action, cap = 'OTHER_REPAIR_FAILURE', 'retain_in_source_plate', 0

    covered_by = None
    if len(bbox) == 4 and staged_crop_boxes:
        area = max(1.0, float(bbox[2]) * float(bbox[3]))
        for rid, cbox in staged_crop_boxes:
            if intersection(bbox, cbox) >= 0.8 * area:
                covered_by = rid
                if oid in graph.selected.get(rid, {}).get('seed_ids', []) and (kind != 'text' or in_top_right_logo):
                    action = 'already_cropped_as_region_image'
                    cap = 0
                break

    out = {
        **entry,
        'diagnostic_category': category,
        'text_edit_intent': ('intrinsic' if in_top_right_logo else 'native_field' if cap else 'unresolved')
                            if kind == 'text' else None,
        'conflicting_peer_ids': peers,
        'recommended_action': action,
        'max_repair_attempts': cap,
    }
    if covered_by is not None:
        out['covered_by_region_crop'] = covered_by
    return out


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
                          auto_create_scene: bool = True, run_layers: bool = True,
                          run_layout: bool | None = None,
                          run_structure: bool | None = None) -> dict:
    """Run local OCR, optional LayerD/PP-DocLayout, VLM OCR proofreading/inventory, pixel ink measurement, and style tiering."""
    from .. import layout_evidence
    from .page_structure import parse_page_structure
    source_path = service.source(state, page)
    if str(page) not in current.get('ocr', {}):
        ocr_dir = directory / 'ocr'
        ocr_dir.mkdir(parents=True, exist_ok=True)
        try:
            service.run_ocr(state, current, ocr_dir, page=page)
        except (OSError, RuntimeError, KeyError, ValueError, subprocess.SubprocessError) as exc:
            if not json.loads((state.job/'request.json').read_text('utf8')).get('host_stage0'):
                raise
            file = ocr_dir/'ocr-unavailable.json'
            write(file,{'source_sha256':digest(source_path),'status':'UNAVAILABLE',
                        'error_type':type(exc).__name__,'lines':[]})
            current.setdefault('ocr',{})[str(page)]=str(file)
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
    cjk_family = next((f['family'] for f in fonts['fonts'] if f.get('cjk') and f['family'] == '微软雅黑'), None)
    if not cjk_family:
        cjk_family = next((f['family'] for f in fonts['fonts'] if f.get('cjk') and '雅黑' in f['family']), None)
    if not cjk_family:
        cjk_family = next((f['family'] for f in fonts['fonts'] if f.get('cjk')), 'Arial')
    latin_family = next((f['family'] for f in fonts['fonts'] if f['family'] in ('Arial', 'Segoe UI', 'Calibri')), cjk_family)

    layout_data: dict | None = None
    layout_status = 'SKIPPED'
    do_layout = bool(run_layers) if run_layout is None else bool(run_layout)
    request = json.loads((state.job/'request.json').read_text(encoding='utf8'))
    profile = request.get('experiment_profile')
    preflight = request.get('host_stage0')
    if preflight and do_layout and str(page) not in current.get('layout', {}):
        from ..experiment import detect
        config = os.environ.get('IMAGE2PPT_EXPERIMENT_CONFIG')
        configured = json.loads(Path(config).read_text(encoding='utf-8-sig')) if config else {}
        detectors = [name for name in ('v3','pp-l','doclayout','omni') if name in configured]
        layout_data = detect(source_path, profile, directory/'model-runs', detectors=detectors)
        if 'v3' not in detectors and layout_evidence.available():
            try:
                native = layout_evidence.detect({'path':str(source_path),'source_sha256':digest(source_path)}, state.job/'cache')
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                layout_data['models'].append({'model':'v3','status':'unavailable','error_type':type(exc).__name__})
            else:
                layout_data['regions'].extend(native['regions'])
                layout_data['models'].append({'model':'v3','status':'candidate' if native['regions'] else 'no_candidates'})
        for name in ('v3','omni'):
            if not any(m['model']==name for m in layout_data['models']):
                layout_data['models'].append({'model':name,'status':'unavailable'})
        layout_file = directory/'layout-regions.json'
        write(layout_file, layout_data)
        current.setdefault('layout', {})[str(page)] = str(layout_file)
        layout_status = 'READY' if any(m['status'] in ('candidate','no_candidates') for m in layout_data['models']) else 'UNAVAILABLE'
    elif profile and do_layout and str(page) not in current.get('layout', {}):
        from ..experiment import detect
        layout_data = detect(source_path, profile, directory/'model-runs')
        layout_file = directory/'layout-regions.json'
        write(layout_file, layout_data)
        current.setdefault('layout', {})[str(page)] = str(layout_file)
        layout_status = 'READY' if any(r['status'] in ('candidate','no_candidates') for r in layout_data['models']) else 'UNAVAILABLE'
    elif str(page) in current.get('layout', {}) and Path(current['layout'][str(page)]).is_file():
        layout_data = json.loads(Path(current['layout'][str(page)]).read_text('utf8'))
        layout_status = 'CACHED'
    elif do_layout and layout_evidence.available():
        try:
            snap_info = {'path': str(source_path), 'source_sha256': digest(source_path)}
            layout_data = layout_evidence.detect(snap_info, state.job / 'cache')
            layout_file = directory / 'layout-regions.json'
            write(layout_file, layout_data)
            current.setdefault('layout', {})[str(page)] = str(layout_file)
            layout_status = 'READY'
        except Exception as exc:
            layout_status = f'UNAVAILABLE:{type(exc).__name__}'

    # Run deterministic + optional VLM page structure inventory & pixel-verified OCR proofreading
    has_vlm_creds = bool(
        os.environ.get('IMAGE2PPT_AGENT_API_KEY')
        or os.environ.get('OPENCODE_GO_API_KEY')
        or os.environ.get('MOMA_API_KEY')
        or os.environ.get('PLANNER_API_KEY')
    )
    do_vlm_structure = not preflight and ((bool(run_layers) and has_vlm_creds) if run_structure is None else bool(run_structure))
    structure_info = parse_page_structure(
        state, source_path, ocr, layout=layout_data, assets=[], fonts=fonts, use_vlm=do_vlm_structure
    )
    if structure_info.get('applied_ocr_corrections'):
        proof_ocr_file = directory / 'ocr-proofread.json'
        write(proof_ocr_file, ocr)
        current.setdefault('ocr', {})[str(page)] = str(proof_ocr_file)
    structure_file = directory / 'page-structure.json'
    write(structure_file, structure_info)
    current.setdefault('page_structure', {})[str(page)] = str(structure_file)

    valid_lines = [row for row in ocr.get('lines', []) if row.get('score', 0) >= 0.85 and row.get('text', '').strip()]
    line_rects = [rectangle(row['bbox'], (w_img, h_img), pad=1) for row in valid_lines]

    raw_items = []
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
        raw_items.append({
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

    immediate_assets: list[dict] = []
    deferred_assets: list[dict] = []
    layers_path_str = current.get('layers', {}).get(str(page))
    if layers_path_str and Path(layers_path_str).is_dir():
        try:
            immediate_assets, deferred_assets = discover_layerd_assets(
                source, Path(layers_path_str), [it['bbox'] for it in raw_items])
        except Exception:
            immediate_assets, deferred_assets = [], []

    extra_logo_boxes = [
        list(r['bbox']) for r in (layout_data or {}).get('regions', [])
        if r.get('label') in ('header_image', 'seal') and float(r.get('score', 0)) >= 0.5
    ] + [
        list(el['bbox']) for el in structure_info.get('elements', [])
        if el.get('category') == 'logo' and len(el.get('bbox', [])) == 4
    ]

    measured_items, scene_relations = _merge_structural_ocr_items(
        source, raw_items, [*immediate_assets, *deferred_assets], extra_logo_boxes=extra_logo_boxes)
    if preflight or (profile and profile != 'baseline'):
        from ..recovery.marked_fields import split_items
        retained_fields = scene_relations.setdefault('source_retained_fields', [])
        measured_items = split_items(source, measured_items, retained_fields)
    for item in measured_items:
        est_px = item['est_size_px']
        item['role'] = 'title' if est_px >= 26.0 else ('subtitle' if est_px >= 18.0 else ('caption' if est_px <= 12.0 else 'body'))
    merged_text_boxes = [it['bbox'] for it in measured_items]
    all_assets = [*immediate_assets, *deferred_assets]
    immediate_assets, deferred_assets = [], []
    for a_obj in all_assets:
        bx, by, bw, bh = a_obj['bbox']
        padded_box = [bx - 10.0, by - 10.0, bw + 20.0, bh + 20.0]
        if any(intersection(padded_box, tb) > 0 for tb in merged_text_boxes):
            deferred_assets.append(a_obj)
        else:
            immediate_assets.append(a_obj)

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
        text_entry = {'id': item['id'], 'bbox': item['bbox'], 'z': int(z_idx),
                      'style': item['style'], 'text': item['text'], 'align': item.get('align', 'left')}
        if item.get('line_spacing_px') is not None:
            text_entry['line_spacing_px'] = item['line_spacing_px']
        texts.append(text_entry)
        members[item['id']] = list(item['ocr_line_ids'])

    current.setdefault('scene_relations', {})[str(page)] = scene_relations

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
        ui_excluded = {'微软雅黑 UI', 'Microsoft JhengHei UI', 'FangSong_GB2312'}
        shared_font_list = [f for f in fonts['fonts'] if f['family'] not in ui_excluded]
        shared_fonts = {'schema': fonts.get('schema', 'FontManifest/1'), 'fonts': shared_font_list} if shared_font_list else fonts
        with lease('cpu'):
            calibrated = calibrate(graph, source, shared_fonts, store, ocr=ocr, adaptive=True)
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
        'layout_available': bool(layout_data) if profile else layout_evidence.available(),
        'layout_status': layout_status,
        'layout_region_count': len((layout_data or {}).get('regions', [])),
        'layout_pictorial_regions': [
            r for r in (layout_data or {}).get('regions', [])
            if r.get('label') in layout_evidence.CROP_LABELS and float(r.get('score', 0)) >= 0.5
        ],
        'sam_available': bool(os.environ.get('IMAGE2PPT_SAM_URL') or os.environ.get('SAM_URL')),
        'text_count': len(measured_items),
        'resolved_text_count': resolved_count,
        'unresolved_text_count': len(measured_items) - resolved_count,
        'asset_count': len(asset_summaries),
        'immediate_asset_count': len(immediate_assets),
        'deferred_asset_count': len(deferred_assets),
        'assets': asset_summaries,
        'scene_relations': scene_relations,
        'backing_summary': {'FLAT': counts.get('FLAT', 0), 'GRADIENT': counts.get('GRADIENT', 0),
                            'COMPLEX': counts.get('COMPLEX', 0)},
        'page_structure_status': structure_info.get('vlm_status', 'SKIPPED'),
        'ocr_corrections_applied': structure_info.get('applied_ocr_corrections', []),
        'inventory_count': len(structure_info.get('elements', [])),
        'styles': final_styles,
        'items': measured_items,
    }
    if preflight or (profile and profile!='baseline'):
        from ..experiment import mark_page,mark_blocks
        layout = layout_data or {'regions':[]}
        low_confidence = [{'id':'ocr-review-'+str(line['id']), 'kind':'uncertain_ocr',
                           'text':line['text'], 'bbox':line['bbox'], 'ocr_line_ids':[line['id']]}
                          for line in ocr.get('lines', []) if preflight and line.get('text') and line.get('score',0)<0.85]
        marks = mark_page(source_path, layout, [*measured_items,*asset_summaries,*low_confidence], directory)
        current.setdefault('experiment_marks',{})[str(page)] = str(directory/'marks.json')
        payload['marked_image']=state.register(directory/'marks.png','numbered_layout')
        payload['mark_inventory']=state.register(directory/'marks.json','numbered_inventory')
        payload['mark_count']=len(marks)
        payload['blocks']=[{**block,'image':state.register(pic,'numbered_block_image'),
                            'inventory':state.register(info,'numbered_block_inventory')}
                           for block,pic,info in mark_blocks(source_path,marks,directory)]
    write(directory / 'page-evidence.json', payload)
    evidence_ref = state.register(directory / 'page-evidence.json', 'page_evidence_summary')
    compact = {k:v for k,v in payload.items() if k not in
               ('items','assets','styles','scene_relations','layout_pictorial_regions')} if payload.get('blocks') else payload
    return {**compact, 'evidence': evidence_ref}


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
    from ..parallel_residual_gate import residual_accepted
    from .office import frame_feedback
    frame_repairs = {}
    bad_ids: set[str] = set()
    staged_set = set(staged_ids)
    for page_rows in office_result.get('residuals', []):
        for rec in page_rows:
            if rec.get('id') not in staged_set or residual_accepted(rec):
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


def _chain_stage_from_spec(
    *,
    initial_graph: SceneGraph,
    initial_baseline: Path,
    fonts: dict,
    source_path: str,
    layers_dir: str | None,
    deferred_path: str | None,
    page_cache_dir: Path,
    directory: Path,
    tag: str,
    trials_spec: list[dict],
    frame_overrides: dict,
    skip_ids: set[str],
    include_assets: bool,
    include_containers: bool,
    include_region_crops: bool = False,
    allow_move_unsafe: bool = False,
    layout_path: str | None = None,
    unrecoverable_text_ids: set[str] | None = None,
    logo_ids: set[str] | None = None,
    wanted_filter: set[str] | None,
    sam_enabled: bool,
    experiment_profile: str | None = None,
    marked_routes: tuple | list = (),
    prepared_tables=None,
) -> tuple[Path, list[str], list[dict]]:
    from .. import layout_evidence, matting
    from ..object_stage import infer_sam
    from ..recovery.agent_asset_stage import stage_asset_candidate
    from ..recovery.asset_retry import retry_asset_extraction
    from ..recovery.candidate_stage import stage_background
    from ..recovery.product_pipeline import failure as classify_exc

    cur_base = initial_baseline
    staged_ids: list[str] = []
    stage_errors: list[dict] = []
    table_fields = [r for r in marked_routes if r['route']=='table'
                    and (wanted_filter is None or r['id'] in wanted_filter)]
    covered_texts = {tid for record in initial_graph.selected.values() for tid in record.get('covered_text_ids', [])}
    if table_fields:
        from ..recovery.marked_tables import stage_tables
        cur_base, tables, covered = prepared_tables or stage_tables(cur_base, directory/tag/'marked-tables', table_fields, skip_ids)
        covered_texts.update(covered)
        staged_ids.extend(tables)
    if include_containers and wanted_filter is None and layout_path:
        from ..recovery.chart_stage import stage_native_charts
        layout = json.loads(Path(layout_path).read_text(encoding='utf8'))
        chart_ir, chart_ids = stage_native_charts(cur_base, directory/tag/'charts-staged', layout=layout,
                                                  fonts=fonts, skip_ids=skip_ids)
        if chart_ids:
            cur_base = chart_ir.parent
            staged_ids.extend(chart_ids)
            chart_scene = json.loads((cur_base/'scene.json').read_text(encoding='utf8'))
            for cid in chart_ids:
                covered_texts.update(chart_scene['selected'][cid]['covered_text_ids'])
    for idx, tr in enumerate(trials_spec):
        oid = tr['object_id']
        if oid in skip_ids or oid in covered_texts:
            continue
        step_out = directory / tag / f'step-{idx:03d}-{oid}'
        frame_opt = (
            {'size': frame_overrides[oid], 'method': 'office_batch_auto_frame_repair/1'}
            if oid in frame_overrides else None
        )
        try:
            ir = stage_background(
                Path(tr['trial_dir']),
                initial_graph,
                cur_base,
                fonts,
                step_out,
                frame_minimum=frame_opt,
                fit_origin=experiment_profile not in (None, 'baseline'),
            )
            cur_base = ir.parent
            staged_ids.append(oid)
        except Exception as exc:
            msg = str(exc)
            code = 'FONT_LAYOUT_FAILED' if 'frame allowance' in msg else ('OWNER_CONFLICT' if 'stable owner' in msg else 'STAGE_REJECTED')
            stage_errors.append({
                'object_id': oid,
                'kind': 'text',
                'text': tr.get('text', ''),
                'bbox': list(tr.get('bbox', [])),
                'failure': code,
                'reason': msg,
            })

    if include_assets:
        from ..recovery.region_crop import stage_region_crops
        for field in marked_routes:
            frame = field.get('picture_frame')
            if not frame or field['id'] in skip_ids or (wanted_filter is not None and field['id'] not in wanted_filter):
                continue
            seed = {'id': field['id'], 'bbox': frame, 'kind': 'asset',
                    'picture_frame': frame if field.get('intrinsic_ui') else None}
            pic_ir, pic_ids, evidence = stage_region_crops(
                cur_base, directory/tag/('picture-'+key(field['id'])[:12]), seeds=[seed], skip_ids=skip_ids)
            if pic_ids:
                cur_base = pic_ir.parent
                staged_ids.extend(pic_ids)
            elif not any(field['id'] in rec.get('seed_ids', []) for rec in initial_graph.selected.values()):
                stage_errors.append({'object_id': field['id'], 'kind': 'asset', 'bbox': frame,
                                     'failure': 'PICTURE_NOT_ISOLATED', 'reason': str(evidence)[:500]})

    if include_assets and layers_dir:
        cur_graph = SceneGraph.model_validate_json((cur_base / 'scene.json').read_text('utf8'))
        from .marked_routes import bind_asset_layers
        deferred_list = bind_asset_layers([r['new_asset'] for r in marked_routes if r.get('new_asset')],
                                          layers_dir,cur_graph.source_sha256)
        if deferred_path and Path(deferred_path).is_file():
            deferred_list.extend(json.loads(Path(deferred_path).read_text('utf8')))
        if deferred_list:
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
                existing_ids.add(oid)
            if to_add:
                prep_base = directory / tag / 'deferred-scene-prep'
                shutil.copytree(cur_base, prep_base)
                # Host fusion alone derives deferred asset ownership and semantic relations.
                cur_graph = fuse_incremental_scene(cur_base, base_graph=cur_graph, added_assets=to_add)
                write(prep_base / 'scene.json', cur_graph.model_dump())
                cur_base = prep_base
                for obj in to_add:
                    if cur_graph.ownership.get(obj.id) != 'RESOLVED':
                        stage_errors.append({
                            'object_id': obj.id,
                            'kind': 'asset',
                            'bbox': list(obj.bbox),
                            'failure': 'OWNERSHIP_UNRESOLVED',
                            'reason': f"host fusion derived ownership={cur_graph.ownership.get(obj.id)}",
                        })

        cur_graph = SceneGraph.model_validate_json((cur_base / 'scene.json').read_text('utf8'))
        asset_targets = [
            a for a in cur_graph.asset.objects
            if a.id not in cur_graph.selected
            and cur_graph.ownership.get(a.id) == 'RESOLVED'
            and a.id not in skip_ids
            and (wanted_filter is None or a.id in wanted_filter)
        ]
        for idx, asset in enumerate(asset_targets):
            step_dir = directory / tag / f'asset-{idx:03d}-{key(asset.id)[:12]}'
            step_dir.mkdir(parents=True, exist_ok=True)
            active_graph = SceneGraph.model_validate_json((cur_base / 'scene.json').read_text('utf8'))
            try:
                res = retry_asset_extraction(
                    active_graph, asset.id, source_path, layers_dir,
                    cur_base, step_dir / 'extract', strategy='verified-backing',
                )
                choices = [('verified-backing', res)]
                compare = asset.kind in ('icon', 'logo', 'decoration')
                if (compare or res.get('status') != 'CANDIDATE_READY') and sam_enabled:
                    sam_root = page_cache_dir / 'sam-cache' / key(asset.id)[:12]
                    plan_file = sam_root / 'sam-plan.json'
                    sam_out = sam_root / 'sam'
                    if not (sam_out / 'manifest.json').is_file() and not any(sam_out.glob('*.png')) if sam_out.exists() else True:
                        sam_root.mkdir(parents=True, exist_ok=True)
                        write(plan_file, {
                            'schema': 'LogicalObjects/2',
                            'source_sha256': active_graph.source_sha256,
                            'image_size': active_graph.image_size,
                            'objects': [a.model_dump() for a in active_graph.asset.objects],
                            'limitations': [],
                        })
                        try:
                            infer_sam(source_path, plan_file, sam_out, 'sam3', [asset.id])
                        except Exception:
                            pass
                    if plan_file.is_file() and sam_out.is_dir():
                        try:
                            res_sam = retry_asset_extraction(
                                active_graph, asset.id, source_path, layers_dir,
                                cur_base, step_dir / 'extract-sam', strategy='sam',
                                sam_plan=plan_file, sam_directory=sam_out,
                            )
                            choices.append(('sam',res_sam))
                            if res_sam.get('status') == 'CANDIDATE_READY':
                                res = res_sam
                        except (ValueError,OSError,RuntimeError) as exc:
                            choices.append(('sam',{'status':'UNAVAILABLE','reason':type(exc).__name__}))
                if compare and matting.available():
                    try:
                        res_matte = retry_asset_extraction(
                            active_graph, asset.id, source_path, layers_dir,
                            cur_base, step_dir / 'extract-rmbg2', strategy='rmbg2',
                        )
                        choices.append(('rmbg2',res_matte))
                    except Exception as exc:
                        choices.append(('rmbg2',{'status':'UNAVAILABLE','reason':type(exc).__name__}))
                for op in ('vitmatte','local-layerd'):
                    if experiment_profile not in (op,'combined'):
                        continue
                    if op=='vitmatte' and not compare:
                        continue
                    if op=='local-layerd' and any(c.get('status')=='CANDIDATE_READY' for _,c in choices):
                        continue
                    try:
                        candidate=retry_asset_extraction(active_graph,asset.id,source_path,layers_dir,
                            cur_base,step_dir/op,strategy=op)
                        choices.append((op,candidate))
                    except (ValueError, OSError, RuntimeError) as exc:
                        choices.append((op,{'status':'UNAVAILABLE','reason':str(exc)[:300]}))
                scored = []
                for strategy, candidate in choices:
                    if candidate.get('status') != 'CANDIDATE_READY':
                        continue
                    attempt = next(a for a in candidate['attempts'] if a['status']=='CANDIDATE_READY')
                    folder = Path(candidate['candidate_directory'])
                    with Image.open(folder/'asset.png') as im:
                        graphic = im.convert('RGBA')
                        contrast = Image.new('RGBA',graphic.size,'#70368C')
                        contrast.alpha_composite(graphic)
                        contrast.convert('RGB').save(step_dir/(strategy+'-contrast.png'))
                    from ..experiment_assets import contrast_edge_score
                    edge = contrast_edge_score(graphic) if compare and experiment_profile not in (None,'baseline') else None
                    scored.append({'strategy':strategy,'recomposition_p95':attempt['source_recomposition_p95'],
                                   'background_error':attempt['background_error'],'contrast_edge_proxy':edge,
                                   'candidate':candidate})
                edges = [r['contrast_edge_proxy'] for r in scored]
                comparable = bool(edges and all(edges)) and all(
                    min(e[k] for e in edges) >= .80*max(e[k] for e in edges) for k in ('samples','visible'))
                for r in scored:
                    r['rank_score'] = r['recomposition_p95'] + (r['contrast_edge_proxy']['p90'] if comparable else 0)
                scored.sort(key=lambda r:(r['rank_score'],r['background_error'],r['strategy']))
                write(step_dir/'strategy-comparison.json',{
                    'attempts':[{'strategy':name,'status':c.get('status'),'reason':c.get('reason')} for name,c in choices],
                    'comparable_edge_support':comparable,
                    'ranked':[{k:v for k,v in r.items() if k!='candidate'} for r in scored]})
                if scored:
                    res = scored[0]['candidate']
                if res.get('status') == 'CANDIDATE_READY':
                    ir = stage_asset_candidate(
                        res['candidate_directory'], active_graph, cur_base,
                        step_dir / 'staged', layer_directory=layers_dir,
                    )
                    cur_base = ir.parent
                    staged_ids.append(asset.id)
                    if compare and experiment_profile in ('vtracer','combined'):
                        from ..experiment_assets import trace_asset
                        try:
                            trace_asset(Path(res['candidate_directory'])/'asset.png',cur_base,asset.id,step_dir/'trace')
                        except (ValueError, RuntimeError, OSError, ImportError) as exc:
                            write(step_dir/'trace-failure.json',{'reason':str(exc)[:300]})
                else:
                    stage_errors.append({
                        'object_id': asset.id, 'kind': 'asset', 'bbox': list(asset.bbox),
                        'failure': 'ASSET_EXTRACTION_UNCERTAIN', 'reason': res.get('reason', 'unverified'),
                    })
            except Exception as exc:
                stage_errors.append({
                    'object_id': asset.id, 'kind': 'asset', 'bbox': list(asset.bbox),
                    'failure': classify_exc(exc, 'generation'), 'reason': str(exc),
                })

    if include_assets:
        for route in marked_routes:
            if not route.get('new_asset') or route['id'] in skip_ids or (wanted_filter is not None and route['id'] not in wanted_filter):
                continue
            from ..recovery.region_crop import stage_region_crops
            graph=SceneGraph.model_validate_json((cur_base/'scene.json').read_text('utf8'))
            if route['id'] not in graph.selected:
                ir,ids,_ = stage_region_crops(cur_base,directory/tag/('marked-asset-'+key(route['id'])[:12]),
                    seeds=[{'id':route['id'],'bbox':route['bbox'],'kind':'asset'}],skip_ids=skip_ids)
                if ids:
                    cur_base=ir.parent
                    staged_ids.extend(ids)

    if include_containers and (wanted_filter is None or any(i.startswith(('s-', 'a-c')) for i in wanted_filter)):
        from ..recovery.container_stage import stage_containers_and_surfaces
        cont_dir = directory / tag / 'containers-staged'
        try:
            cont_ir, cont_ids = stage_containers_and_surfaces(
                cur_base, cont_dir, skip_ids=skip_ids, logo_ids=logo_ids, wanted_ids=wanted_filter
            )
            if cont_ids:
                cur_base = cont_ir.parent
                staged_ids.extend(cont_ids)
        except Exception as exc:
            stage_errors.append({
                'object_id': 'containers', 'kind': 'surface', 'bbox': [],
                'failure': classify_exc(exc, 'generation'), 'reason': str(exc),
            })

    surface_routes = [r for r in marked_routes if r['route']=='surface'
                      and (wanted_filter is None or r['id'] in wanted_filter)]
    if include_containers and surface_routes:
        from ..recovery.marked_surfaces import stage_marked_surfaces
        card_ir, card_ids = stage_marked_surfaces(cur_base, directory/tag/'marked-cards',
                                                  surface_routes, skip_ids)
        if card_ids:
            cur_base = card_ir.parent
            staged_ids.extend(card_ids)

    if include_region_crops and wanted_filter is None:
        from ..recovery.region_crop import collect_crop_seeds, stage_region_crops
        cur_graph = SceneGraph.model_validate_json((cur_base / 'scene.json').read_text('utf8'))
        layout_data = None
        if layout_path and Path(layout_path).is_file():
            layout_data = json.loads(Path(layout_path).read_text('utf8'))
        elif layout_evidence.available():
            try:
                layout_data = layout_evidence.detect(
                    {'path': source_path, 'source_sha256': cur_graph.source_sha256},
                    page_cache_dir / 'layout-cache',
                )
            except Exception as exc:
                stage_errors.append({
                    'object_id': 'layout_evidence', 'kind': 'asset', 'bbox': [],
                    'failure': classify_exc(exc, 'generation'), 'reason': str(exc),
                })
                layout_data = None
        crop_seeds, protected_text_ids = collect_crop_seeds(
            cur_graph,
            unrecoverable_text_ids=set(unrecoverable_text_ids or ()),
            layout=layout_data,
            logo_ids=logo_ids,
            skip_seed_ids=skip_ids,
        )
        if crop_seeds:
            crop_dir = directory / tag / 'region-crops-staged'
            try:
                crop_ir, crop_ids, _ = stage_region_crops(
                    cur_base, crop_dir,
                    seeds=crop_seeds,
                    allow_move_unsafe=bool(allow_move_unsafe),
                    protected_text_ids=protected_text_ids,
                    skip_seed_ids=skip_ids,
                )
                if crop_ids:
                    cur_base = crop_ir.parent
                    staged_ids.extend(crop_ids)
                    if include_containers:
                        from ..recovery.container_stage import stage_containers_and_surfaces
                        post_cont_dir = directory / tag / 'post-crop-containers-staged'
                        pc_ir, pc_ids = stage_containers_and_surfaces(
                            cur_base, post_cont_dir, skip_ids=skip_ids, logo_ids=logo_ids
                        )
                        if pc_ids:
                            cur_base = pc_ir.parent
                            staged_ids.extend(pc_ids)
            except Exception as exc:
                stage_errors.append({
                    'object_id': 'region_crops', 'kind': 'asset', 'bbox': [],
                    'failure': classify_exc(exc, 'generation'), 'reason': str(exc),
                })

    if marked_routes:
        from .marked_routes import route_trace,plan_move_groups
        graph=SceneGraph.model_validate_json((cur_base/'scene.json').read_text('utf8'))
        movement_routes = [r if wanted_filter is None or r['id'] in wanted_filter
                           else {**r,'move_with':[]} for r in marked_routes]
        groups = [g for g in plan_move_groups(movement_routes,graph)
                  if not graph.selected[g[0]].get('move_members') and not set(g).intersection(skip_ids)]
        if groups:
            from ..parallel_rebuild import adapt
            for group in groups:
                graph.selected[group[0]]['move_members']=group
            ir = adapt(graph,cur_base)
            written = {frozenset(g['member_ids']) for g in ir.get('groups',[])}
            accepted = [g for g in groups if frozenset(g) in written]
            if accepted:
                for group in groups:
                    if group not in accepted:
                        graph.selected[group[0]].pop('move_members',None)
                if cur_base == initial_baseline:
                    grouped = directory/tag/'movement'
                    shutil.copytree(cur_base,grouped)
                    cur_base = grouped
                write(cur_base/'scene.json',graph.model_dump())
                write(cur_base/'slideir.json',ir)
                # A group-only change must go through the same compile and Office commit.
                staged_ids.extend(i for g in accepted for i in g if i not in staged_ids)
        write(directory/tag/'route-trace.json',route_trace(marked_routes,graph))
    return cur_base, staged_ids, stage_errors


def restage_batch_candidate_subset(
    service,
    state,
    current,
    directory: Path,
    candidate: dict,
    office_res: dict,
) -> bool:
    """Isolate failing batch objects and apply measured frame_minimum repairs without touching parent scene."""
    spec = candidate.get('batch_spec')
    if not spec:
        return False
    staged_graph = SceneGraph.model_validate_json((Path(candidate['ir']).parent / 'scene.json').read_text('utf8'))
    staged_ids = candidate.get('object_ids') or [candidate['object_id']]
    frame_repairs, bad_ids = _collect_office_defects(office_res, staged_graph, staged_ids)
    if not (frame_repairs or (bad_ids and len(bad_ids) < len(staged_ids))):
        return False

    row = service.page(current, candidate['page'])
    initial_baseline = Path(spec['initial_baseline'])
    initial_graph = SceneGraph.model_validate_json((initial_baseline / 'scene.json').read_text('utf8'))
    fonts = json.loads(Path(current['fonts']).read_text('utf8'))
    overrides = {oid: fb['measured_frame_minimum_px'] for oid, fb in frame_repairs.items()}
    bad_crop_ids = {oid for oid in bad_ids if staged_graph.selected.get(oid, {}).get('region_crop')}
    bad_crop_seeds = {sid for oid in bad_crop_ids for sid in staged_graph.selected[oid].get('seed_ids', [])}
    skip = (bad_ids - bad_crop_ids) | bad_crop_seeds | set(spec.get('pass1_failed_ids', []))
    skip.update(i for r in spec.get('marked_routes', []) if r['route']=='preserve'
                for i in [r['id'],*r['target_ids']])
    unrec_pass2 = set(spec.get('unrecoverable_text_ids', [])) - bad_crop_seeds
    wanted_list = spec.get('wanted_filter')
    wanted_filter = set(wanted_list) if wanted_list is not None else None

    directory.mkdir(parents=True, exist_ok=True)
    repaired_base, repaired_ids, retry_errors = _chain_stage_from_spec(
        initial_graph=initial_graph,
        initial_baseline=initial_baseline,
        fonts=fonts,
        source_path=row['snapshot']['path'],
        layers_dir=row.get('layers'),
        deferred_path=current.get('deferred_assets', {}).get(str(candidate['page'])),
        page_cache_dir=Path(spec['cache_dir']),
        directory=directory,
        tag='stage-pass2',
        experiment_profile=spec.get('experiment_profile'),
        marked_routes=spec.get('marked_routes', ()),
        trials_spec=spec['trials'],
        frame_overrides=overrides,
        skip_ids=skip,
        include_assets=bool(spec.get('include_assets', True)),
        include_containers=bool(spec.get('include_containers', True)),
        include_region_crops=bool(spec.get('include_region_crops', False)),
        allow_move_unsafe=bool(spec.get('allow_move_unsafe', False)),
        layout_path=spec.get('layout_path'),
        unrecoverable_text_ids=unrec_pass2,
        logo_ids=set(spec.get('logo_ids', [])),
        wanted_filter=wanted_filter,
        sam_enabled=bool(spec.get('sam_enabled', False)),
    )

    isolation_failures: list[dict] = []
    for oid in sorted(bad_ids):
        t_match = next((t for t in staged_graph.text.texts if t.id == oid), None)
        a_match = next((a for a in staged_graph.asset.objects if a.id == oid), None)
        s_match = next((s for s in staged_graph.surface.shapes if s.id == oid), None)
        isolation_failures.append({
            'object_id': oid,
            'kind': 'text' if t_match else ('surface' if s_match else 'asset'),
            'text': t_match.text if t_match else '',
            'bbox': list(t_match.bbox) if t_match else (list(s_match.bbox) if s_match else (list(a_match.bbox) if a_match else [])),
            'failure': 'OFFICE_RESIDUAL_OR_EDIT_FAILED',
            'reason': 'isolated during batch Office verification; inspect locally',
        })
    for err in retry_errors:
        if err['object_id'] in frame_repairs:
            err['measured_frame_minimum_px'] = frame_repairs[err['object_id']]['measured_frame_minimum_px']
        isolation_failures.append(err)

    candidate['isolation_attempted'] = True
    candidate['isolation_failures'] = isolation_failures
    candidate['isolated_bad_ids'] = sorted(bad_ids)
    if not repaired_ids:
        return False

    final_ir = repaired_base / 'slideir.json'
    candidate['ir'] = str(final_ir)
    candidate['object_id'] = repaired_ids[0]
    candidate['object_ids'] = repaired_ids
    candidate['auto_repaired_frame_ids'] = [oid for oid in overrides if oid in repaired_ids]
    candidate['fingerprint'] = key({'batch': repaired_ids, 'ir': digest(final_ir)})
    candidate['files'] = {str(p): digest(p) for p in repaired_base.rglob('*') if p.is_file()}
    return True


def batch_reconstruct_page(service, state, current, directory: Path, *, page: int,
                           object_ids: list[str] | None = None, include_assets: bool = True,
                           include_containers: bool = True, include_region_crops: bool = False,
                           allow_move_unsafe: bool = False, use_inpaint: bool = True,
                           verify_office: bool = True) -> dict:
    """Parallel local text trials + sequential staging + batch validation candidate (requires commit_candidate)."""
    from ..recovery.adaptive_policy import AdaptiveBudget
    from ..recovery.product_pipeline import failure as classify_exc
    from ..recovery.region_candidates import adaptive_candidates
    from .background import inpaint_service

    request=json.loads((state.job/'request.json').read_text(encoding='utf8'))
    if not request.get('host_stage0') and request.get('experiment_profile') not in (None,'baseline'):
        missing=[i for i in range(1,len(request['sources'])+1) if not current.get('marked_plans',{}).get(str(i))]
        if missing:
            return {'status':'PAGE_PLANS_REQUIRED','pages':missing}
    row = service.page(current, page)
    initial_graph = service.graph(row)
    if 'fonts' not in current:
        service.list_fonts(state, current, directory / 'fonts')
    fonts = json.loads(Path(current['fonts']).read_text('utf8'))
    source_path = row['snapshot']['path']
    initial_baseline = Path(row['ir']).parent
    wanted_filter = set(object_ids) if object_ids is not None else None
    fields = current.get('marked_plans', {}).get(str(page), [])
    routes = []
    if fields and (request.get('host_stage0') or request.get('experiment_profile') not in (None, 'baseline')):
        from .marked_routes import bind_plan
        routes = bind_plan(fields, current['experiment_marks'][str(page)], initial_graph, source_path)
        if wanted_filter is not None:
            for field in routes:
                if field['id'] in wanted_filter:
                    wanted_filter.update(field['target_ids'])
        write(directory/'plan-bindings.json', routes)
    sam_enabled = bool(os.environ.get('IMAGE2PPT_SAM_URL') or os.environ.get('SAM_URL'))

    failed_regions: list[dict] = []

    # Phase 1: Parallel local text candidate trials against initial_baseline.
    text_targets = [
        t for t in initial_graph.text.texts
        if t.id not in initial_graph.selected
        and initial_graph.ownership.get(t.id) == 'RESOLVED'
        and (wanted_filter is None or t.id in wanted_filter)
    ]
    priority_ids = {tid for r in routes if r['route'] in ('text', 'table') for tid in r['target_ids']}
    preserved_ids = {i for r in routes if r['route']=='preserve' for i in [r['id'],*r['target_ids']]}
    prepared_tables=None
    table_fields=[r for r in routes if r['route']=='table' and (wanted_filter is None or r['id'] in wanted_filter)]
    covered=set()
    if table_fields:
        from ..recovery.marked_tables import stage_tables
        prepared_tables=stage_tables(initial_baseline,directory/'stage-pass1'/'marked-tables',table_fields,preserved_ids)
        covered=prepared_tables[2]
    inventory_file = current.get('page_structure', {}).get(str(page))
    inventory = json.loads(Path(inventory_file).read_text('utf8')) if inventory_file else {}
    title_boxes = [(el['bbox'], rank) for el in inventory.get('elements', [])
                   for rank, role in enumerate(('title', 'section_header')) if el['category'] == role]
    def priority(t):
        rank = min((rank for box,rank in title_boxes if intersection(t.bbox, box) >= .7*t.bbox[2]*t.bbox[3]), default=2)
        return (rank, t.id not in priority_ids, t.bbox[1], t.bbox[0])
    text_targets = sorted((t for t in text_targets if t.id not in preserved_ids and t.id not in covered), key=priority)

    request = json.loads((state.job / 'request.json').read_text('utf8'))
    model_page = (page,len(request['sources'])) if routes else None
    budget = AdaptiveBudget(Store(state.job / 'state'), request.get('adaptive_budget'), model_page=model_page)
    from ..parallel_config import ROOT
    config_file = ROOT / '.runtime/parallel-services.json'
    lama_cfg = json.loads(config_file.read_text('utf8')).get('lama') if (use_inpaint and config_file.exists()) else None
    raw_inpaint = inpaint_service(lama_cfg, state.job / 'cache/lama', budget) if lama_cfg else None
    safe_inpaint = _threadsafe_inpaint(raw_inpaint)

    def _probe_text(text_obj, *, folder="trials", inpaint=None):
        t_dir = directory / folder / f'{text_obj.id}-{key(text_obj.id)[:8]}'
        stream = adaptive_candidates(
            initial_graph, text_obj.id, source_path, row.get('layers'),
            t_dir, baseline_directory=initial_baseline, inpaint=inpaint,
            marked_fields=bool(routes),
        )
        attempts = []
        try:
            for cand_row in stream:
                attempts.append({'candidate': cand_row['candidate'], 'status': cand_row['status']})
                if cand_row['status'] == 'PRECHECK_PASS':
                    return {
                        'object': text_obj, 'status': 'PRECHECK_PASS',
                        'trial_dir': t_dir / cand_row['candidate'], 'method': cand_row['method'], 'attempts': attempts,
                    }
            return {
                'object': text_obj, 'status': 'BLOCKED', 'failure': 'BACKGROUND_REPAIR_FAILED',
                'reason': 'no local or inpaint candidate passed precheck', 'trial_root': t_dir, 'attempts': attempts,
            }
        except Exception as exc:
            return {
                'object': text_obj, 'status': 'BLOCKED',
                'failure': getattr(exc, 'code', classify_exc(exc, 'generation')),
                'reason': str(exc), 'trial_root': t_dir, 'attempts': attempts,
            }
        finally:
            stream.close()

    passed_trials = []
    if text_targets:
        workers = min(6, max(1, len(text_targets)))
        # Capture only masks that survived ownership and text-mask completion.
        # Sharing a model call does not share pixel ownership or commit decisions.
        from ..parallel_rebuild import array_sha
        masks = {}
        def local_probe(obj):
            def capture(image, mask):
                masks[obj.id] = mask.copy()
                raise ValueError('deferred to local group repair')
            return _probe_text(obj, inpaint=capture if safe_inpaint else None)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(local_probe, text_targets))
        containers = [el for el in inventory.get('elements', []) if el['category'] in ('card_shape','photo','ribbon_shape','table')]
        groups = {}
        for idx, outcome in enumerate(outcomes):
            obj = outcome['object']
            if outcome['status'] != 'PRECHECK_PASS' and obj.id in masks:
                parents = [e for e in containers if intersection(e['bbox'],obj.bbox) >= .95*obj.bbox[2]*obj.bbox[3]]
                parent = min(parents, key=lambda e:e['bbox'][2]*e['bbox'][3])['id'] if parents else obj.id
                groups.setdefault(parent, []).append((idx,obj))
        if groups:
            with Image.open(source_path) as im:
                source_pixels = np.array(im.convert('RGB'))
        for members in groups.values():
            union = np.zeros(source_pixels.shape[:2], dtype=bool)
            safe_members = []
            for idx,obj in members:
                if not (union & masks[obj.id]).any():
                    union |= masks[obj.id]
                    safe_members.append((idx,obj))
            try:
                from ..recovery.region_candidates import lower_backgrounds
                candidates = lower_backgrounds(source_pixels, row['layers'], union) if row.get('layers') else ()
                background = next(iter(candidates), None)
            except (OSError,KeyError,ValueError,ImportError):
                background = None
            try:
                repaired = background.image if background is not None else safe_inpaint(source_pixels, union)
            except (OSError,TimeoutError,ValueError,RuntimeError):
                continue
            for idx, obj in safe_members:
                def authorized_patch(image, mask):
                    if not np.array_equal(mask, masks[obj.id]):
                        raise ValueError('local ownership changed before repair')
                    return repaired
                authorized_patch.receipt = {'group': [member.id for _,member in safe_members],
                                            'union_sha256': array_sha(union.astype('uint8')),
                                            'provider': 'layerd' if background is not None else 'lama'}
                outcomes[idx] = _probe_text(obj, folder='joint-trials', inpaint=authorized_patch)
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
                failed_regions.append({
                    'object_id': t_obj.id, 'kind': 'text', 'text': t_obj.text, 'bbox': t_obj.bbox,
                    'failure': item['failure'], 'reason': item['reason'], 'evidence_ids': ev[:4],
                })

    trials_spec = [
        {
            'object_id': tr['object'].id,
            'text': tr['object'].text,
            'bbox': list(tr['object'].bbox),
            'trial_dir': str(tr['trial_dir']),
        }
        for tr in passed_trials
    ]

    logo_ids = {
        x['id'] for x in current.get('scene_relations', {}).get(str(page), {}).get('logo_lockup_candidates', [])
        if x.get('id')
    }
    crop_flag = bool(include_region_crops)
    layout_path = current.get('layout', {}).get(str(page))
    unrecoverable_text_ids = (
        {fr['object_id'] for fr in failed_regions
         if fr.get('kind') == 'text'
         and _classify_failed_region(fr, initial_graph, logo_ids, [])['max_repair_attempts'] == 0}
        | {t.id for t in initial_graph.text.texts
           if t.id not in initial_graph.selected and (t.id in logo_ids or initial_graph.ownership.get(t.id) != 'RESOLVED')}
    )
    current.setdefault('batch_unrecoverable_texts', {})[str(page)] = sorted(unrecoverable_text_ids)

    cand_ir_ref: list[Path] = []

    def _enriched_failures() -> list[dict]:
        staged_crop_boxes: list[tuple[str, list[float]]] = []
        if cand_ir_ref and cand_ir_ref[0].is_file():
            cg = SceneGraph.model_validate_json((cand_ir_ref[0].parent / 'scene.json').read_text('utf8'))
            staged_crop_boxes = [
                (rid, list(rec['bbox'])) for rid, rec in cg.selected.items()
                if rec.get('region_crop') and len(rec.get('bbox', [])) == 4
            ]
        return [_classify_failed_region(err, cg if staged_crop_boxes else initial_graph, logo_ids, staged_crop_boxes) for err in failed_regions]

    final_base, staged_ids, stage_errors = _chain_stage_from_spec(
        initial_graph=initial_graph,
        initial_baseline=initial_baseline,
        fonts=fonts,
        source_path=source_path,
        layers_dir=row.get('layers'),
        deferred_path=current.get('deferred_assets', {}).get(str(page)),
        page_cache_dir=directory,
        directory=directory,
        tag='stage-pass1',
        prepared_tables=prepared_tables,
        experiment_profile=request.get('experiment_profile'),
        marked_routes=routes,
        trials_spec=trials_spec,
        frame_overrides={},
        skip_ids=preserved_ids,
        include_assets=include_assets,
        include_containers=include_containers,
        include_region_crops=crop_flag,
        allow_move_unsafe=bool(allow_move_unsafe),
        layout_path=layout_path,
        unrecoverable_text_ids=unrecoverable_text_ids,
        logo_ids=logo_ids,
        wanted_filter=wanted_filter,
        sam_enabled=sam_enabled,
    )
    failed_regions.extend(stage_errors)
    pass1_failed_ids = sorted(err['object_id'] for err in stage_errors if err.get('object_id'))

    if not staged_ids:
        enriched = _enriched_failures()
        return {
            'status': 'NO_CANDIDATES_STAGED', 'page': page, 'committed_count': 0,
            'committed_object_ids': [], 'failed_count': len(enriched), 'failed_regions': enriched,
        }

    final_ir = final_base / 'slideir.json'
    cand_ir_ref.append(final_ir)
    batch_cid = 'c-' + key(str(directory.relative_to(state.job)) + ':batch')[:24]
    cand = {
        'id': batch_cid,
        'page': page,
        'object_id': staged_ids[0],
        'object_ids': staged_ids,
        'kind': 'batch',
        'status': 'STAGED_LOCAL',
        'ir': str(final_ir),
        'parent_ir_sha256': digest(row['ir']),
        'parent_scene_sha256': digest(row['scene']),
        'strategy': 'batch_backbone',
        'code': service.code_binding(),
        'fingerprint': key({'batch': staged_ids, 'ir': digest(final_ir)}),
        'files': {str(p): digest(p) for p in final_base.rglob('*') if p.is_file()},
        'batch_spec': {
            'initial_baseline': str(initial_baseline),
            'experiment_profile':request.get('experiment_profile'),
            'marked_routes':routes,
            'cache_dir': str(directory),
            'trials': trials_spec,
            'pass1_failed_ids': pass1_failed_ids,
            'include_assets': bool(include_assets),
            'include_containers': bool(include_containers),
            'include_region_crops': bool(crop_flag),
            'allow_move_unsafe': bool(allow_move_unsafe),
            'layout_path': layout_path,
            'unrecoverable_text_ids': sorted(unrecoverable_text_ids),
            'logo_ids': sorted(logo_ids),
            'wanted_filter': sorted(wanted_filter) if wanted_filter is not None else None,
            'sam_enabled': bool(sam_enabled),
        },
        'auto_repaired_frame_ids': [],
        'isolated_bad_ids': [],
        'isolation_failures': [],
    }
    current['candidates'][batch_cid] = cand
    next_version = current['version'] + 1

    if not verify_office:
        enriched = _enriched_failures()
        return {
            'status': 'STAGED_LOCAL',
            'already_committed': False,
            'candidate_id': batch_cid,
            'page': page,
            'staged_count': len(staged_ids),
            'staged_object_ids': staged_ids,
            'committed_count': 0,
            'committed_object_ids': [],
            'failed_count': len(enriched),
            'failed_regions': enriched,
            'next_step': f'Batch candidate {batch_cid} is staged locally; call commit_candidate(candidate_id="{batch_cid}", expected_version={next_version}) to validate and commit.',
        }

    # Unified Office check & per-object failure isolation through RuntimeService.validate_candidate.
    val = service.validate_candidate(state, current, directory / 'validate', candidate_id=batch_cid)
    cand_ir_ref[0] = Path(cand['ir'])
    failed_regions.extend(cand.get('isolation_failures', []))
    staged_ids = cand.get('object_ids', staged_ids)
    auto_repaired_frames = cand.get('auto_repaired_frame_ids', [])
    isolated_bad_ids = cand.get('isolated_bad_ids', [])
    enriched = _enriched_failures()

    if cand.get('status') == 'VALIDATED' and val.get('status') == 'PASS':
        return {
            'status': 'VALIDATED',
            'office_status': 'PASS',
            'already_committed': False,
            'candidate_id': batch_cid,
            'page': page,
            'staged_count': len(staged_ids),
            'staged_object_ids': staged_ids,
            'committed_count': 0,
            'committed_object_ids': [],
            'auto_repaired_frame_ids': auto_repaired_frames,
            'isolated_dropped_ids': isolated_bad_ids,
            'failed_count': len(enriched),
            'failed_regions': enriched,
            'next_step': f'Batch candidate {batch_cid} is VALIDATED ({len(staged_ids)} staged, 0 committed). Call commit_candidate(candidate_id="{batch_cid}", expected_version={next_version}) to formally commit, then inspect failed_regions or compose_delivery.',
        }

    return {
        'status': 'OFFICE_UNAVAILABLE' if val.get('status') == 'UNKNOWN' else 'REVIEW_REQUIRED',
        'already_committed': False,
        'candidate_id': batch_cid,
        'page': page,
        'staged_count': len(staged_ids),
        'staged_object_ids': staged_ids,
        'committed_count': 0,
        'committed_object_ids': [],
        'isolated_dropped_ids': isolated_bad_ids,
        'office_status': val.get('status'),
        'failed_count': len(enriched),
        'failed_regions': enriched,
        'next_step': f'Batch candidate {batch_cid} is not validated (status={val.get("status")}); commit_candidate will remain blocked until hard/Office checks pass.',
    }
