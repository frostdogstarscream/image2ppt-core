"""Deterministic perception preparation and batch regional reconstruction backbone.

Preserves all source-pixel ownership, peer protection, Office editing, and
hash-bound visual review invariants while reducing 100+ object slides from
hundreds of Agent tool turns to a single batch backbone pass plus targeted repair.
"""
from __future__ import annotations

from collections import Counter
import gc
import subprocess
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time

import cv2
import numpy as np
from PIL import Image

from ..contracts import intersection
from ..materialize import digest, write
from ..object_contracts import ClosedObject
from ..object_extract import rectangle
from ..parallel_contracts import FLOWING_BODY, Relation, SceneGraph
from ..parallel_fusion import fuse_incremental_scene, is_decorative_subspan
from ..parallel_store import Store, key
from ..parallel_typography import calibrate, snap_ooxml_size_px, measure_text_appearance
from ..resource_lease import lease
from .semantic_text import apply_heading_weights, is_body_continuation, plain_line_gap, reuse_template_fonts


def _merge_structural_ocr_items(
    source: np.ndarray,
    raw_items: list[dict],
    asset_records: list[dict],
    extra_logo_boxes: list[list[float]] | None = None,
    layout_regions: list[dict] | None = None,
    editable_paragraphs: bool = False,
    ocr_boxes: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Perform pre-scene structural grouping and relation analysis (Astra Step 2).

    Identifies top-right brand logo lockups, merges split same-row titles, merges
    consecutive same-card vertical lines into multi-line paragraphs, and flags
    dual-color colon lines before `create_scene` locks individual single-line bboxes.
    """
    h_img, w_img = source.shape[:2]
    adjacent_ocr_lines = {
        pair for box in (ocr_boxes or []) if box.get('score', 0) >= .85
        for pair in zip(box.get('lines', []), box.get('lines', [])[1:])
    } if editable_paragraphs else set()
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

    bullet_boxes = [
        ar['bbox'] for ar in asset_records
        if ar.get('kind') in ('icon', 'decoration', 'illustration')
        and max(ar['bbox'][2], ar['bbox'][3]) <= 96.0
    ] + [
        r['bbox'] for r in (layout_regions or ())
        if r.get('label') == 'image' and float(r.get('score', 0)) >= 0.5
        and 10.0 <= r['bbox'][2] <= 96.0 and 10.0 <= r['bbox'][3] <= 96.0
    ]
    layout_text_boxes = [
        r['bbox'] for r in (layout_regions or ())
        if r.get('label') == 'text' and float(r.get('score', 0)) >= 0.72
        and r['bbox'][3] >= 36.0
    ]

    def _left_bullet_index(lbox: list[float]) -> int | None:
        ax0, ay0, _, ah0 = lbox
        cy_l = ay0 + 0.5 * ah0
        best_idx = None
        best_gap = 1e9
        for b_idx, bb in enumerate(bullet_boxes):
            ix, iy, iw, ih = bb
            gap = ax0 - (ix + iw)
            if -6.0 <= gap <= 96.0 and abs(cy_l - (iy + 0.5 * ih)) <= max(10.0, 0.65 * max(ah0, ih)):
                if abs(gap) < best_gap:
                    best_gap = abs(gap)
                    best_idx = b_idx
        return best_idx

    def _has_independent_left_markers(box_a: list[float], box_b: list[float], source_continuation: bool) -> bool:
        b1, b2 = _left_bullet_index(box_a), _left_bullet_index(box_b)
        if b1 is not None and b2 is not None and b1 != b2:
            return True
        x_right = int(round(min(box_a[0], box_b[0]))) - 3
        x_left = max(0, x_right - 56)
        if x_right - x_left < 14:
            return False
        ya0, ya1 = max(0, int(round(box_a[1]))), min(h_img, int(round(box_a[1] + box_a[3])))
        yb0, yb1 = max(0, int(round(box_b[1]))), min(h_img, int(round(box_b[1] + box_b[3])))
        if ya1 <= ya0 + 4 or yb1 <= yb0 + 4 or yb0 <= ya1 + 1:
            return False
        band_y0 = max(0, ya0 - 2)
        band_y1 = min(h_img, yb1 + 2)
        panel = source[band_y0:band_y1, max(0, x_left - 2):min(w_img, x_right + 2)].astype(np.float32)
        if panel.size == 0:
            return False
        bg = np.median(np.concatenate([panel[0], panel[-1], panel[:, 0]], axis=0), axis=0)
        fg_a = np.abs(source[ya0:ya1, x_left:x_right].astype(np.float32) - bg).max(2) > 18.0
        fg_b = np.abs(source[yb0:yb1, x_left:x_right].astype(np.float32) - bg).max(2) > 18.0
        if int(fg_a.sum()) < 8 or int(fg_b.sum()) < 8:
            return False
        fg_gap = np.abs(source[ya1:yb0, x_left:x_right].astype(np.float32) - bg).max(2) > 16.0
        if source_continuation:
            # A continuous card separator touches both lines but is not two bullets.
            rail = fg_a.all(0) & fg_gap.all(0) & fg_b.all(0)
            if rail.any():
                other = ~cv2.dilate(rail.astype(np.uint8)[None, :], np.ones((1, 5), np.uint8))[0].astype(bool)
                if not (fg_a[:, other].sum() >= 8 and fg_b[:, other].sum() >= 8):
                    return False
        if float(fg_gap.mean()) > 0.08:
            return False
        # Require left marker blobs near a similar x-column
        cols_a = np.nonzero(fg_a.any(0))[0]
        cols_b = np.nonzero(fg_b.any(0))[0]
        return len(cols_a) >= 3 and len(cols_b) >= 3 and abs(int(cols_a.min()) - int(cols_b.min())) <= 8

    def _same_layout_paragraph(box_a: list[float], box_b: list[float]) -> bool:
        a_area = max(1.0, box_a[2] * box_a[3])
        b_area = max(1.0, box_b[2] * box_b[3])
        for lb in layout_text_boxes:
            if intersection(box_a, lb) >= 0.72 * a_area and intersection(box_b, lb) >= 0.72 * b_area:
                return True
        return False

    def _can_merge_vertical(cur_item: dict, cand: dict) -> tuple[bool, str]:
        if (cur_item['id'] in logo_ids or cand['id'] in logo_ids
                or cur_item['id'] in dual_color_ids or cand['id'] in dual_color_ids
                or cur_item.get('stripped_affix') or cand.get('stripped_affix')
                or cur_item.get('is_header_merged') or cand.get('is_header_merged')):
            return False, 'left'
        last_line = cur_item['lines'][-1]
        ax, ay, aw, ah = last_line['bbox']
        bx, by, bw, bh = cand['bbox']
        gap_y = by - (ay + ah)
        avg_h = 0.5 * (ah + bh)
        source_continuation = (
            (last_line['id'], cand['ocr_line_id']) in adjacent_ocr_lines
            and is_body_continuation({'align': 'left'}, [last_line, cand])
            and gap_y <= .80 * avg_h
        )
        rgb_diff = max(abs(a - b) for a, b in zip(last_line['rgb'], cand['rgb']))
        max_sz = max(cur_item['est_size_px'], cand['est_size_px'])
        stacked_short_header = (
            23.0 < max_sz <= 28.5
            and len(cur_item['lines']) == 1
            and len(last_line['text'].strip()) <= 6
            and len(cand['text'].strip()) <= 6
            and max(aw, bw) <= 0.16 * w_img
            and gap_y <= max(8.0, 0.48 * avg_h)
            and rgb_diff <= 22
        )
        wrapped_body_prose = (
            23.0 < max_sz <= 26.0
            and aw >= 250.0
            and len(last_line['text'].replace(' ', '')) >= 10
            and abs(ax - bx) <= 10.0
            and bw <= 1.05 * aw
            and gap_y <= max(8.0, 0.45 * avg_h)
            and rgb_diff <= 20
        )
        if max_sz > 23.0 and not (stacked_short_header or wrapped_body_prose or source_continuation):
            return False, 'left'
        in_layout_para = _same_layout_paragraph(last_line['bbox'], cand['bbox'])
        size_diff = abs(last_line['est_size_px'] - cand['est_size_px'])
        if size_diff > (4.8 if in_layout_para else 3.5):
            return False, 'left'
        h_a, h_b = max(1.0, ah), max(1.0, bh)
        if max(h_a, h_b) / min(h_a, h_b) > 1.45:
            return False, 'left'
        if rgb_diff > 26:
            return False, 'left'
        gap_limit = max(14.0, .78 * avg_h) if in_layout_para else max(10.0, .62 * avg_h)
        if gap_y < -5.0 or (gap_y > gap_limit and not source_continuation):
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
        if not in_layout_para and size_diff > 2.8 and not (gap_y <= 4.0 and left_aligned and rgb_diff <= 20):
            return False, 'left'
        if _has_independent_left_markers(last_line['bbox'], cand['bbox'], source_continuation):
            return False, 'left'
        # Do not merge across an asset or horizontal divider line.
        x_min = max(0, int(round(min(ax, bx))))
        x_max = min(w_img, int(round(max(ax + aw, bx + bw))))
        strip_box = [float(x_min), ay + ah - 1.0, float(max(1, x_max - x_min)), max(2.0, gap_y + 2.0)]
        if any(intersection(strip_box, ar['bbox']) > 0 for ar in asset_records):
            gap_start = math.ceil(ay + ah) + 1
            gap = [x_min, gap_start, x_max-x_min, math.floor(by)-1-gap_start]
            if not (editable_paragraphs and left_aligned and is_body_continuation(
                    {'align':'left'}, [last_line,cand]) and plain_line_gap(source,gap)):
                return False, 'left'
        if gap_y >= 2.0 and x_max > x_min + 4:
            sy0 = max(0, int(round(ay + ah)))
            sy1 = min(h_img, int(round(by)))
            if sy1 > sy0:
                strip = source[sy0:sy1, x_min:x_max].astype(float)
                med_col = np.median(strip.reshape(-1, 3), axis=0)
                row_cov = (np.abs(strip - med_col).max(axis=2) > 16.0).mean(axis=1)
                if float(row_cov.max()) > (0.62 if in_layout_para else 0.48):
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
            if editable_paragraphs:
                cur['natural_wrap'] = is_body_continuation(cur,cur['lines'])
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


def _deoverlap_same_row_ocr_lines(source: np.ndarray, ocr: dict) -> bool:
    """Trim padded OCR boxes that partially overlap a neighbor on the same text row at a clean gap."""
    lines = ocr.get('lines') or []
    if len(lines) < 2:
        return False
    h_img, w_img = source.shape[:2]
    changed = False
    order = sorted(
        range(len(lines)),
        key=lambda i: (float(lines[i]['bbox'][1]), float(lines[i]['bbox'][0])),
    )
    for idx_a in range(len(order)):
        ia = order[idx_a]
        for idx_b in range(idx_a + 1, len(order)):
            ib = order[idx_b]
            la, lb = lines[ia], lines[ib]
            ba, bb = la.get('bbox') or [], lb.get('bbox') or []
            if len(ba) != 4 or len(bb) != 4:
                continue
            x1, y1, w1, h1 = [float(v) for v in ba]
            x2, y2, w2, h2 = [float(v) for v in bb]
            if x2 < x1:
                la, lb = lb, la
                x1, y1, w1, h1, x2, y2, w2, h2 = x2, y2, w2, h2, x1, y1, w1, h1
            if min(w1, h1, w2, h2) <= 0:
                continue
            ov_y = min(y1 + h1, y2 + h2) - max(y1, y2)
            if ov_y < 0.55 * min(h1, h2):
                continue
            # Partial horizontal overlap: left box bleeds into start of right box
            if not (x1 < x2 < x1 + w1 + 2.0 and x2 + w2 > x1 + w1 - 4.0):
                continue
            rx0 = max(0, int(math.floor(x1)))
            rx1 = min(w_img, int(math.ceil(x2 + w2)))
            ry0 = max(0, int(math.floor(min(y1, y2))))
            ry1 = min(h_img, int(math.ceil(max(y1 + h1, y2 + h2))))
            oy0 = max(ry0, int(math.floor(max(y1, y2))))
            oy1 = min(ry1, int(math.ceil(min(y1 + h1, y2 + h2))))
            if rx1 - rx0 < 16 or oy1 - oy0 < 6:
                continue
            strip = source[ry0:ry1, rx0:rx1].astype(np.float32)
            top_bot = np.concatenate([strip[0:2].reshape(-1, 3), strip[-2:].reshape(-1, 3)], axis=0)
            bg = np.median(top_bot, axis=0)
            band = source[oy0:oy1, rx0:rx1].astype(np.float32)
            col_ink = (np.abs(band - bg).max(2) > 22.0).sum(axis=0) >= 2
            s_lo = max(2, int(round(max(x1 + 6.0, x2 - 48.0))) - rx0)
            s_hi = min(rx1 - rx0 - 2, int(round(min(x1 + w1 + 24.0, x2 + w2 - 6.0))) - rx0)
            if s_hi - s_lo < 6:
                continue
            best_gap = None
            cur_start = None
            for col in range(s_lo, s_hi):
                if not col_ink[col]:
                    if cur_start is None:
                        cur_start = col
                elif cur_start is not None:
                    if col - cur_start >= 4 and col_ink[:cur_start].any() and col_ink[col:].any():
                        if best_gap is None or (col - cur_start) > (best_gap[1] - best_gap[0]):
                            best_gap = (cur_start, col)
                    cur_start = None
            if cur_start is not None and (s_hi - cur_start) >= 4 and col_ink[:cur_start].any() and col_ink[s_hi:].any():
                if best_gap is None or (s_hi - cur_start) > (best_gap[1] - best_gap[0]):
                    best_gap = (cur_start, s_hi)
            if best_gap is None:
                continue
            g0, g1 = best_gap
            new_r1 = float(min(rx0 + g0 + 1, int(round(x2)) - 1))
            new_x2 = float(max(x2, rx0 + g1 - 1))
            if new_r1 - x1 >= 8.0 and new_r1 < x1 + w1 - 0.5:
                la['bbox'] = [x1, y1, round(new_r1 - x1, 2), h1]
                if la.get('polygon'):
                    la['polygon'] = [[min(float(pt[0]), new_r1), float(pt[1])] for pt in la['polygon']]
                changed = True
            if new_x2 > x2 + 0.5 and (x2 + w2) - new_x2 >= 8.0 and new_x2 >= new_r1:
                lb['bbox'] = [round(new_x2, 2), y2, round((x2 + w2) - new_x2, 2), h2]
                if lb.get('polygon'):
                    lb['polygon'] = [[max(float(pt[0]), new_x2), float(pt[1])] for pt in lb['polygon']]
                changed = True
    return changed


def _split_pill_divider_ocr_lines(source: np.ndarray, ocr: dict) -> bool:
    """Split single OCR lines that span multiple UI pill items separated by vertical `|` / `丨` divider strokes."""
    lines = ocr.get('lines') or []
    if not lines:
        return False
    h_img, w_img = source.shape[:2]
    changed = False
    rebuilt: list[dict] = []
    for row in lines:
        raw_text = str(row.get('text') or '').strip()
        bbox = row.get('bbox') or []
        if (
            row.get('score', 0) < 0.85
            or '\n' in raw_text
            or len(bbox) != 4
            or not any(ch in raw_text for ch in ('|', '丨', '｜'))
        ):
            rebuilt.append(row)
            continue
        segments = re.split(r'[|丨｜]', raw_text)
        if len(segments) < 2 or any(len(seg.replace(' ', '')) < 2 for seg in segments):
            rebuilt.append(row)
            continue
        bx, by, bw, bh = [int(round(float(v))) for v in bbox]
        x0, y0 = max(0, bx), max(0, by)
        x1, y1 = min(w_img, bx + bw), min(h_img, by + bh)
        cw, ch = x1 - x0, y1 - y0
        if cw < 24 or ch < 10:
            rebuilt.append(row)
            continue
        crop = source[y0:y1, x0:x1].astype(np.float32)
        border = np.concatenate([crop[0], crop[-1]], axis=0)
        bg = np.median(border, axis=0)
        if float(np.percentile(np.abs(border - bg).max(1), 80)) > 12.0:
            rebuilt.append(row)
            continue
        delta = np.abs(crop - bg).max(2)
        col_ink = (delta > 16.0).any(axis=0)
        runs: list[tuple[int, int]] = []
        col_i = 0
        while col_i < cw:
            if not col_ink[col_i]:
                col_i += 1
                continue
            col_j = col_i
            while col_j < cw and col_ink[col_j]:
                col_j += 1
            runs.append((col_i, col_j))
            col_i = col_j
        min_gap = max(4, int(round(0.20 * ch)))
        max_div_w = max(4, int(round(0.14 * ch)))
        valid_seps: list[tuple[int, int]] = []
        for idx_r in range(1, len(runs) - 1):
            c_left, c_right = runs[idx_r]
            if c_right - c_left > max_div_w:
                continue
            l_gap = c_left - runs[idx_r - 1][1]
            r_gap = runs[idx_r + 1][0] - c_right
            vert_cov = float((delta[:, c_left:c_right] > 14.0).any(axis=1).mean())
            if l_gap >= min_gap and r_gap >= min_gap and vert_cov >= 0.42:
                valid_seps.append((c_left, c_right))
        if len(valid_seps) != len(segments) - 1:
            rebuilt.append(row)
            continue
        sep_peaks = [float(delta[:, s0:s1].max()) for s0, s1 in valid_seps]
        sep_mask = np.zeros(cw, dtype=bool)
        for s0, s1 in valid_seps:
            sep_mask[max(0, s0 - 1):min(cw, s1 + 1)] = True
        glyph_deltas = delta[:, col_ink & (~sep_mask)]
        if glyph_deltas.size < 12:
            rebuilt.append(row)
            continue
        glyph_p90 = float(np.percentile(glyph_deltas, 90))
        if len(segments) == 2 and '|' in raw_text and '丨' not in raw_text and '｜' not in raw_text and max(sep_peaks) > max(120.0, 0.90 * glyph_p90):
            rebuilt.append(row)
            continue
        bounds = [(0, valid_seps[0][0])]
        for idx_s in range(len(valid_seps) - 1):
            bounds.append((valid_seps[idx_s][1], valid_seps[idx_s + 1][0]))
        bounds.append((valid_seps[-1][1], cw))
        split_rows: list[dict] = []
        for idx_s, (seg_txt, (rx0, rx1)) in enumerate(zip(segments, bounds), 1):
            seg_clean = seg_txt.strip()
            nz = np.nonzero((delta[:, rx0:rx1] > 18.0).any(axis=0))[0]
            if not seg_clean or len(nz) < 4:
                split_rows = []
                break
            sx0 = x0 + max(0, rx0 + int(nz[0]) - 2)
            sx1 = min(w_img, x0 + rx0 + int(nz[-1]) + 3)
            seg_box = [round(float(sx0), 2), round(float(y0), 2), round(float(max(4, sx1 - sx0)), 2), round(float(ch), 2)]
            split_rows.append({
                **row,
                'id': f"{row['id']}__s{idx_s}",
                'text': seg_clean,
                'bbox': seg_box,
                'polygon': None,
            })
        if split_rows:
            rebuilt.extend(split_rows)
            changed = True
        else:
            rebuilt.append(row)
    if changed:
        ocr['lines'] = rebuilt
    return changed


def _include_screen_edges(source, box, texts):
    """Trace a dark screen's contiguous edge into a smooth surrounding ground."""
    x, y, w, h = box
    exterior_texts = [item['bbox'] for item in texts if intersection(item['bbox'], box) == 0]
    bounds = [x, y, x+w, y+h]
    for axis, side in ((0, -1), (0, 1), (1, -1), (1, 1)):
        view = source if axis else source.transpose(1, 0, 2)
        start, length = (x, w) if axis else (y, h)
        lo, hi = start + length//10, start + length - length//10
        edge = (y if side < 0 else y+h-1) if axis else (x if side < 0 else x+w-1)
        if not (0 <= edge+side*22 < len(view)):
            continue
        outer = np.stack([view[edge+side*d, lo:hi] for d in range(17, 23)]).astype(float)
        ground = np.median(outer, axis=0)
        if np.percentile(ground, 10) < 185:
            continue
        ink = view[edge-side*2, lo:hi].astype(float)
        if np.median((ground-ink).max(1)) < 65:
            continue
        last = 0
        for d in range(1, 17):
            delta = ground-view[edge+side*d, lo:hi]
            # Only an edge-wide darkening contiguous with the screen is eligible.
            strong_rim = d == 1 and np.mean(delta.max(1) > 15) >= .85
            if not strong_rim and (np.percentile(np.abs(outer-ground), 90) > 6
                                    or np.mean((delta.max(1) > 5) & (delta.min(1) >= -5)) < .80):
                break
            last = d
        if last and last < 16:
            strip = ([x, edge-last if side < 0 else edge+1, w, last] if axis else
                     [edge-last if side < 0 else edge+1, y, last, h])
            if not any(intersection(strip, other) > 0 for other in exterior_texts):
                bounds[axis + (2 if side > 0 else 0)] += side*last
    return [bounds[0], bounds[1], bounds[2]-bounds[0], bounds[3]-bounds[1]]


def _detect_embedded_ui_frames(
    source: np.ndarray,
    measured_items: list[dict],
    layout_data: dict | None = None,
    *, trace_edges: bool = False,
) -> list[list[int]]:
    """Detect embedded software UI screenshots / dark BI dashboards so they extract intact instead of fragmenting."""
    h_img, w_img = source.shape[:2]
    page_area = float(w_img * h_img)
    if h_img < 120 or w_img < 200 or not measured_items:
        return []
    m_bdr = min(16, max(4, min(h_img, w_img) // 24))
    border_px = np.concatenate([
        source[:m_bdr, :].reshape(-1, 3),
        source[-m_bdr:, :].reshape(-1, 3),
        source[m_bdr:-m_bdr, :m_bdr].reshape(-1, 3),
        source[m_bdr:-m_bdr, -m_bdr:].reshape(-1, 3),
    ], axis=0)
    border_luma = float(np.median(border_px))
    frames: list[list[int]] = []

    def _inside_count(box: list[float], items: list[dict], thresh: float = 0.75) -> int:
        cnt = 0
        for it in items:
            ib = it['bbox']
            if intersection(ib, box) >= thresh * max(1.0, ib[2] * ib[3]):
                cnt += 1
        return cnt

    # Rule 1: Inset dark BI / supervisory dashboard on a bright presentation canvas
    if border_luma >= 175.0:
        dark_mask = (source.mean(axis=2) < 95.0).astype(np.uint8)
        closed_d = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        n_d, _, stats_d, _ = cv2.connectedComponentsWithStats(closed_d, 8)
        for cid in range(1, n_d):
            bx, by, bw, bh, area = [int(v) for v in stats_d[cid]]
            box_area = float(bw * bh)
            if not (0.15 * page_area <= box_area <= 0.62 * page_area):
                continue
            if bw < int(0.34 * w_img) or bh < int(0.34 * h_img) or by < int(0.10 * h_img):
                continue
            if area / max(1.0, box_area) < 0.76:
                continue
            cand = [bx, by, bw, bh]
            if _inside_count(cand, measured_items) >= 5:
                frames.append(_include_screen_edges(source, cand, measured_items) if trace_edges else cand)

    # Rule 2: Inset light multi-pane application window mockup (micro UI chrome + tiny UI icons)
    micro_texts = [
        it for it in measured_items
        if float(it.get('est_size_px', 99.0)) <= 16.8 and it['bbox'][1] >= 0.12 * h_img
    ]
    regions = (layout_data or {}).get('regions', [])
    micro_layout = [
        {'bbox': list(r['bbox']), 'kind': str(r.get('label'))}
        for r in regions
        if float(r.get('score', 0)) >= 0.45
        and r['bbox'][1] >= 0.12 * h_img
        and (
            (r.get('label') == 'vision_footnote' and r['bbox'][2] <= 260)
            or (r.get('label') == 'image' and 12.0 <= r['bbox'][2] <= 42.0 and 12.0 <= r['bbox'][3] <= 42.0)
        )
    ]
    if len(micro_texts) >= 8 and len(micro_layout) >= 5:
        pts = [*({'bbox': list(it['bbox']), 'is_text': True, 'is_icon': False} for it in micro_texts),
               *({'bbox': m['bbox'], 'is_text': False, 'is_icon': m['kind'] == 'image'} for m in micro_layout)]
        clusters: list[dict] = []
        max_join_x = min(200.0, 0.12 * w_img)
        max_join_y = min(120.0, 0.13 * h_img)
        for pt in sorted(pts, key=lambda p: (p['bbox'][1], p['bbox'][0])):
            pb = pt['bbox']
            x0, y0, x1, y1 = pb[0], pb[1], pb[0] + pb[2], pb[1] + pb[3]
            placed = None
            for cl in clusters:
                gap_x = max(0.0, max(x0, cl['x0']) - min(x1, cl['x1']))
                gap_y = max(0.0, max(y0, cl['y0']) - min(y1, cl['y1']))
                cand_w = max(x1, cl['x1']) - min(x0, cl['x0'])
                if gap_x <= max_join_x and gap_y <= max_join_y and cand_w <= 0.60 * w_img:
                    placed = cl
                    break
            if placed is None:
                clusters.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'pts': [pt]})
            else:
                placed['x0'] = min(placed['x0'], x0)
                placed['y0'] = min(placed['y0'], y0)
                placed['x1'] = max(placed['x1'], x1)
                placed['y1'] = max(placed['y1'], y1)
                placed['pts'].append(pt)
        # Merge touching/overlapping clusters
        merged_c = True
        while merged_c:
            merged_c = False
            for i in range(len(clusters)):
                if merged_c:
                    break
                for j in range(i + 1, len(clusters)):
                    ca, cb = clusters[i], clusters[j]
                    gx = max(0.0, max(ca['x0'], cb['x0']) - min(ca['x1'], cb['x1']))
                    gy = max(0.0, max(ca['y0'], cb['y0']) - min(ca['y1'], cb['y1']))
                    if gx <= max_join_x and gy <= max_join_y:
                        cand_w = max(ca['x1'], cb['x1']) - min(ca['x0'], cb['x0'])
                        cand_h = max(ca['y1'], cb['y1']) - min(ca['y0'], cb['y0'])
                        if cand_w <= 0.60 * w_img and cand_h <= 0.65 * h_img:
                            ca['x0'] = min(ca['x0'], cb['x0'])
                            ca['y0'] = min(ca['y0'], cb['y0'])
                            ca['x1'] = max(ca['x1'], cb['x1'])
                            ca['y1'] = max(ca['y1'], cb['y1'])
                            ca['pts'].extend(cb['pts'])
                            clusters.pop(j)
                            merged_c = True
                            break

        huge_texts = [it for it in measured_items if float(it.get('est_size_px', 0.0)) >= 28.0]
        ext_headers = [it for it in measured_items if float(it.get('est_size_px', 0.0)) >= 18.0]
        for cl in clusters:
            cw0, ch0 = cl['x1'] - cl['x0'], cl['y1'] - cl['y0']
            n_t = sum(1 for p in cl['pts'] if p['is_text'])
            n_ic = sum(1 for p in cl['pts'] if p['is_icon'])
            if cw0 < 0.28 * w_img or ch0 < 0.24 * h_img or cw0 > 0.62 * w_img or ch0 > 0.65 * h_img:
                continue
            if n_t < 8 or n_ic < 5:
                continue
            c_box = [cl['x0'], cl['y0'], cw0, ch0]
            if any(intersection(bt['bbox'], c_box) > 0.3 * bt['bbox'][2] * bt['bbox'][3] for bt in huge_texts):
                continue
            in_items = [
                it for it in measured_items
                if intersection(it['bbox'], c_box) >= 0.75 * max(1.0, it['bbox'][2] * it['bbox'][3])
            ]
            if not in_items or sum(1 for it in in_items if float(it.get('est_size_px', 99.0)) <= 17.0) / len(in_items) < 0.65:
                continue
            out_items = [
                it for it in measured_items
                if intersection(it['bbox'], [cl['x0'] - 60, cl['y0'] - 60, cw0 + 120, ch0 + 120]) == 0
            ]
            if out_items and float(np.median([float(it['est_size_px']) for it in out_items])) < 16.5:
                continue
            # Expand cluster bounds to clean enclosing window border gutter without crossing external >=18px headers
            min_top = max(0, int(round(cl['y0'])) - 90)
            for bt in ext_headers:
                bx_t, by_t, bw_t, bh_t = bt['bbox']
                if min(bx_t + bw_t, cl['x1'] + 80) > max(bx_t, cl['x0'] - 80):
                    if by_t + bh_t <= cl['y0'] + 4:
                        min_top = max(min_top, int(math.ceil(by_t + bh_t)) + 6)
            max_bot = min(h_img, int(round(cl['y1'])) + 90)
            min_left = max(0, int(round(cl['x0'])) - 160)
            max_right = min(w_img, int(round(cl['x1'])) + 160)
            for it in ext_headers:
                ib = it['bbox']
                if intersection(ib, c_box) > 0:
                    continue
                if ib[1] + ib[3] > min_top and ib[1] < max_bot:
                    if ib[0] + ib[2] <= cl['x0']:
                        min_left = max(min_left, int(math.ceil(ib[0] + ib[2])) + 6)
                    elif ib[0] >= cl['x1']:
                        max_right = min(max_right, int(math.floor(ib[0])) - 6)
                if ib[0] + ib[2] > min_left and ib[0] < max_right:
                    if ib[1] >= cl['y1']:
                        max_bot = min(max_bot, int(math.floor(ib[1])) - 6)

            def _outer_clean_strip(axis: str, start: int, limit: int, step: int, span_a: int, span_b: int) -> int | None:
                if span_b <= span_a + 16:
                    return None
                cur = start
                first_hit = None
                while (cur <= limit if step > 0 else cur >= limit):
                    if axis == 'top':
                        band = source[max(0, cur - 6):cur, span_a:span_b].astype(np.float32)
                    elif axis == 'bot':
                        band = source[cur:min(h_img, cur + 6), span_a:span_b].astype(np.float32)
                    elif axis == 'left':
                        band = source[span_a:span_b, max(0, cur - 6):cur].astype(np.float32)
                    else:
                        band = source[span_a:span_b, cur:min(w_img, cur + 6)].astype(np.float32)
                    if band.size >= 48:
                        med = np.median(band, axis=(0, 1))
                        if float(np.percentile(np.abs(band - med).max(2), 85)) <= 4.5:
                            if first_hit is None:
                                first_hit = (cur, med)
                            elif abs(cur - first_hit[0]) >= 32:
                                break
                            # Keep advancing up to 28px ONLY on bottom axis if an input bar still lies immediately ahead
                            # (require >= 15% strip deviation so a 1px outer card border is never crossed).
                            if axis == 'bot' and abs(cur - first_hit[0]) >= 8:
                                ahead = source[min(h_img, cur + 4):min(h_img, cur + 28), span_a:span_b].astype(np.float32)
                                if ahead.size >= 48 and float(np.percentile(np.abs(ahead - med).max(2), 85)) > 7.5:
                                    first_hit = None
                                else:
                                    return cur
                    cur += step
                return first_hit[0] if first_hit is not None else None

            ix0, ix1 = max(min_left, int(round(cl['x0']))), min(max_right, int(round(cl['x1'])))
            iy0, iy1 = max(min_top, int(round(cl['y0']))), min(max_bot, int(round(cl['y1'])))
            fy0 = _outer_clean_strip('top', int(round(cl['y0'])) - 4, min_top, -1, ix0, ix1)
            fy1 = _outer_clean_strip('bot', int(round(cl['y1'])) + 4, max_bot, 1, ix0, ix1)
            if None in (fy0, fy1):
                continue
            fx0 = _outer_clean_strip('left', int(round(cl['x0'])) - 4, min_left, -1, fy0 + 6, fy1 - 6)
            fx1 = _outer_clean_strip('right', int(round(cl['x1'])) + 4, max_right, 1, fy0 + 6, fy1 - 6)
            if None in (fx0, fx1):
                continue
            fy0_r = _outer_clean_strip('top', int(round(cl['y0'])) - 4, min_top, -1, fx0 + 6, fx1 - 6)
            fy1_r = _outer_clean_strip('bot', int(round(cl['y1'])) + 4, max_bot, 1, fx0 + 6, fx1 - 6)
            if fy0_r is not None:
                fy0 = fy0_r
            if fy1_r is not None:
                fy1 = fy1_r
            fw, fh = fx1 - fx0, fy1 - fy0
            if 0.14 * page_area <= fw * fh <= 0.58 * page_area and fw >= int(0.32 * w_img) and fh >= int(0.30 * h_img):
                win_box = [int(fx0), int(fy0), int(fw), int(fh)]
                if not any(intersection(win_box, f) > 0.3 * min(win_box[2] * win_box[3], f[2] * f[3]) for f in frames):
                    frames.append(win_box)
    return frames


def split_decorative_ocr_line(source: np.ndarray, line: dict) -> tuple[str, list[float], str]:
    """Strip purely non-alphanumeric leading/trailing affixes or thin edge `|`/`丨` bars when a clean pixel gap exists."""
    raw_text = line.get('text', '').strip()
    bbox = [float(v) for v in line['bbox']]
    compact = raw_text.replace(' ', '').replace('\n', '')
    if len(compact) < 3 or not any(ch.isalnum() for ch in compact):
        return raw_text, bbox, ''
    _edge_pipes = ('|', '丨', '｜')
    start = 0
    while start < len(compact) and (not compact[start].isalnum() or compact[start] in _edge_pipes):
        start += 1
    end = len(compact)
    while end > start and (not compact[end - 1].isalnum() or compact[end - 1] in _edge_pipes):
        end -= 1
    core = compact[start:end]
    prefix, suffix = compact[:start], compact[end:]
    # Keep paired brackets or trailing punctuation that belong to wording (e.g. percentages/colons/Chinese commas).
    keep_marks = set('%％:：()（）[]【】《》“”"\'.,，。、；;!?！？+=/—–·…~')
    if (not prefix and not suffix) or len(core) < 2:
        return raw_text, bbox, ''
    if any(ch in keep_marks for ch in (prefix + suffix)):
        return raw_text, bbox, ''
    has_edge_cjk_pipe = any(ch in _edge_pipes for ch in (prefix + suffix))
    # Recover spaces inside the core text while stripping outer decorative marks.
    stripped = raw_text
    while stripped and (not stripped[0].isalnum() or stripped[0] in _edge_pipes) and stripped[0] not in keep_marks:
        stripped = stripped[1:].lstrip()
    while stripped and (not stripped[-1].isalnum() or stripped[-1] in _edge_pipes) and stripped[-1] not in keep_marks:
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
    col_ink = (np.abs(crop - bg).max(2) > 22.0 if has_edge_cjk_pipe else np.abs(crop - bg).max(2) > 30.0).sum(axis=0) > 0
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
    if has_edge_cjk_pipe:
        if prefix and (runs[0][1] - runs[0][0] > max(4, round(0.14 * lh)) or runs[1][0] - runs[0][1] < 3):
            return raw_text, bbox, ''
        if suffix and (runs[-1][1] - runs[-1][0] > max(4, round(0.14 * lh)) or runs[-1][0] - runs[-2][1] < 3):
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
    from .page_structure import parse_page_structure, isolated_label_neighbors, resolve_isolated_roles
    source_path = service.source(state, page)
    if str(page) not in current.get('ocr', {}):
        ocr_dir = directory / 'ocr'
        ocr_dir.mkdir(parents=True, exist_ok=True)
        for _ocr_try in range(2):
            try:
                service.run_ocr(state, current, ocr_dir, page=page)
                break
            except (MemoryError, OSError, RuntimeError, KeyError, ValueError, subprocess.SubprocessError) as exc:
                msg_exc = str(exc).lower()
                if _ocr_try == 0 and (isinstance(exc, MemoryError) or 'bad allocation' in msg_exc or 'unable to allocate' in msg_exc):
                    gc.collect()
                    time.sleep(2.0)
                    continue
                if not json.loads((state.job/'request.json').read_text('utf8')).get('host_stage0'):
                    raise
                file = ocr_dir/'ocr-unavailable.json'
                write(file,{'source_sha256':digest(source_path),'status':'UNAVAILABLE',
                            'error_type':type(exc).__name__,'lines':[]})
                current.setdefault('ocr',{})[str(page)]=str(file)
                break
    ocr = json.loads(Path(current['ocr'][str(page)]).read_text('utf8'))
    original_ocr_path = Path(current['ocr'][str(page)])
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
        or os.environ.get('DEEPSEEK_API_KEY')
        or os.environ.get('OPENCODE_GO_API_KEY')
        or os.environ.get('MOMA_API_KEY')
        or os.environ.get('MOMA_API_KEYS')
        or os.environ.get('PLANNER_API_KEY')
    )
    do_vlm_structure = (bool(run_layers) and has_vlm_creds) if run_structure is None else bool(run_structure)
    isolated = isolated_label_neighbors(ocr.get('lines', [])) if request.get('orchestrator') == 'editable_reconstruction' else {}
    if run_structure is None and request.get('orchestrator') == 'editable_reconstruction':
        do_vlm_structure = bool(isolated) and has_vlm_creds
    structure_info = parse_page_structure(
        state, source_path, ocr, layout=layout_data, assets=[], fonts=fonts, use_vlm=do_vlm_structure
    )
    deoverlapped = _deoverlap_same_row_ocr_lines(source, ocr)
    pill_split = _split_pill_divider_ocr_lines(source, ocr)
    if structure_info.get('applied_ocr_corrections') or deoverlapped or pill_split:
        proof_ocr_file = directory / 'ocr-proofread.json'
        write(proof_ocr_file, ocr)
        current.setdefault('ocr', {})[str(page)] = str(proof_ocr_file)
    structure_file = directory / 'page-structure.json'
    graphic_fields = resolve_isolated_roles(source, ocr, layout_data, structure_info, fonts, isolated)
    structure_info['graphic_or_uncertain_ocr'] = graphic_fields
    for element in structure_info.get('elements', []):
        candidate = graphic_fields.get(element.get('ocr_id'))
        if candidate:
            element['category'] = 'icon' if candidate['disposition'] == 'graphic' else 'unclassified'
            element['label'] = ''

    write(structure_file, structure_info)
    current.setdefault('page_structure', {})[str(page)] = str(structure_file)

    valid_lines = [row for row in ocr.get('lines', []) if row.get('score', 0) >= 0.85 and row.get('text', '').strip()]
    line_rects = [rectangle(row['bbox'], (w_img, h_img), pad=1) for row in valid_lines]

    raw_items = []
    for index, row in enumerate(valid_lines, 1):
        if row['id'] in graphic_fields:
            continue
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
            if ink_span >= 10.0 and raw_h > ink_span * 1.32:
                est_px = snap_ooxml_size_px(max(9.0, min(96.0, ink_span * 0.98)))
            else:
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
        source, raw_items, [*immediate_assets, *deferred_assets],
        extra_logo_boxes=extra_logo_boxes,
        layout_regions=(layout_data or {}).get('regions', []),
        editable_paragraphs=request.get('orchestrator') == 'editable_reconstruction',
        ocr_boxes=ocr.get('boxes'),
    )
    if preflight or (profile and profile != 'baseline'):
        from ..recovery.marked_fields import split_items
        retained_fields = scene_relations.setdefault('source_retained_fields', [])
        measured_items = split_items(source, measured_items, retained_fields)
    for item in measured_items:
        est_px = item['est_size_px']
        item['role'] = 'title' if est_px >= 26.0 else ('subtitle' if est_px >= 18.0 else ('caption' if est_px <= 12.0 else 'body'))
        if item.get('natural_wrap'):
            item['role'] = FLOWING_BODY
    ui_frames = _detect_embedded_ui_frames(source, measured_items, layout_data,
                                           trace_edges=request.get('orchestrator') == 'editable_reconstruction')
    device_frames = []
    if request.get('orchestrator') == 'editable_reconstruction':
        from ..recovery.embedded_device import detect_device_frames
        device_frames = detect_device_frames(source, raw_items, layout_data)
        # Keep a phone inside an existing, larger application screenshot in its parent.
        device_frames = [frame for frame in device_frames if not any(
            intersection(frame, other) > 0.5 * frame[2] * frame[3] for other in ui_frames)]
        ui_frames.extend(device_frames)
        scene_relations['embedded_device_frames'] = device_frames
    if ui_frames:
        scene_relations['embedded_ui_frames'] = ui_frames
    merged_text_boxes = [it['bbox'] for it in measured_items]
    all_assets = [
        a_obj for a_obj in [*immediate_assets, *deferred_assets]
        if not any(
            intersection(a_obj['bbox'], uf) >= 0.70 * max(1.0, a_obj['bbox'][2] * a_obj['bbox'][3])
            for uf in ui_frames
        )
    ]
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
                    and np.abs(grp['rgb_sum'] / grp['count'] - rgb).max() <= 9.0):
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
        if request.get('orchestrator') == 'editable_reconstruction':
            scene_relations['heading_weight_evidence'] = apply_heading_weights(
                calibrated, source, shared_fonts, structure_info)
            if page > 1:
                previous_pages = ((service.graph(p), p['snapshot']['path'])
                                  for p in current['pages'][:page-1] if p)
                local_members = {member for group in scene_relations['heading_weight_evidence']
                                 for member in group['members']}
                scene_relations['repeated_template_fonts'] = reuse_template_fonts(
                    calibrated, source, previous_pages, shared_fonts, local_members)
        if calibrated.model_dump() != graph.model_dump():
            from ..parallel_rebuild import build
            cal_dir = directory / 'calibrated-backing'
            ir, _ = build(calibrated, row['snapshot'], fonts, cal_dir,
                          strategy='off', text_candidates=False, surface_candidates=False)
            shutil.copyfile(Path(row['ir']).parent / 'runtime-ocr-protection.json',
                            ir.parent / 'runtime-ocr-protection.json')
            row['ir'] = str(ir)
            row['scene'] = str(ir.parent / 'scene.json')
        shutil.copyfile(structure_file, Path(row['ir']).parent / 'page-structure.json')
        write(Path(row['ir']).parent / 'embedded-ui-frames.json', ui_frames)
        if device_frames:
            write(Path(row['ir']).parent / 'embedded-device-frames.json', device_frames)
        if deferred_assets and (directory / 'deferred-assets.json').is_file():
            shutil.copyfile(directory / 'deferred-assets.json', Path(row['ir']).parent / 'deferred-assets.json')
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

    if request.get('orchestrator') == 'editable_reconstruction':
        from .editable_draft import _page_objects
        objects, relations = _page_objects(active_graph, structure_info) if active_graph else ([], [])
        ledger = directory / 'source-text-inventory.json'
        write(ledger, {
            'source_sha256': digest(source_path), 'source': str(source_path),
            'original_ocr_sha256': digest(original_ocr_path), 'original_ocr': str(original_ocr_path),
            'lines': ocr.get('lines', []), 'ocr_paragraphs': ocr.get('boxes', []),
            'corrections': structure_info.get('applied_ocr_corrections', []),
            'visual_status': structure_info.get('vlm_status', 'SKIPPED'),
            'objects': objects, 'relations': relations, 'styles': final_styles,
            'embedded_frames': ui_frames,
            'retained_low_confidence': [line['id'] for line in ocr.get('lines', [])
                                        if line.get('score', 0) < .85],
        })
        current.setdefault('source_text_inventory', {})[str(page)] = str(ledger)
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
    from ..parallel_edit_probe import replacement_accepted
    from .office import frame_feedback
    frame_repairs = {}
    bad_ids: set[str] = set()
    sibling_bad_ids: set[str] = set()
    staged_set = set(staged_ids)
    for page_rows in office_result.get('residuals', []):
        for rec in page_rows:
            oid = rec.get('id')
            if not oid or residual_accepted(rec):
                continue
            if rec.get('reason') == 'missing source-owner or actual-delete evidence':
                continue
            if oid in staged_set:
                bad_ids.add(oid)
            else:
                sibling_bad_ids.add(oid)
    for rec in office_result.get('replacements', {}).get('records', []):
        oid = rec.get('id')
        if not oid:
            continue
        ok = replacement_accepted(rec)
        if not ok:
            if oid in staged_set:
                fb = frame_feedback(office_result, graph, oid)
                if fb and oid not in bad_ids:
                    frame_repairs[oid] = fb
                else:
                    bad_ids.add(oid)
            else:
                sibling_bad_ids.add(oid)
    failed_oid = (office_result.get('office') or {}).get('failed_object_id')
    if failed_oid:
        if failed_oid in staged_set and failed_oid not in frame_repairs:
            bad_ids.add(failed_oid)
        elif failed_oid not in staged_set:
            sibling_bad_ids.add(failed_oid)

    if sibling_bad_ids:
        def _sel_box(oid):
            rec = graph.selected.get(oid) or {}
            box = rec.get('restore_bbox') or rec.get('bbox') or (rec.get('payload') or {}).get('bbox')
            return box if isinstance(box, (list, tuple)) and len(box) == 4 else None
        def _boxes_near(b1, b2, pad=0.0):
            return (
                b1[0] - pad < b2[0] + b2[2] + pad
                and b2[0] - pad < b1[0] + b1[2] + pad
                and b1[1] - pad < b2[1] + b2[3] + pad
                and b2[1] - pad < b1[1] + b1[3] + pad
            )
        for other_id in sibling_bad_ids:
            ob = _sel_box(other_id)
            if ob is None:
                continue
            other_area = max(1.0, float(ob[2]) * float(ob[3]))
            other_is_surface = (graph.selected.get(other_id) or {}).get('kind') == 'surface'
            direct_overlaps = [
                (sid, sb, intersection(ob, sb))
                for sid in staged_ids
                if (sb := _sel_box(sid)) is not None and intersection(ob, sb) > 0.0
            ]
            if direct_overlaps:
                surfaces = [
                    (sid, sb, ov) for sid, sb, ov in direct_overlaps
                    if (graph.selected.get(sid) or {}).get('kind') == 'surface'
                    and (not other_is_surface or float(sb[2]) * float(sb[3]) > other_area)
                ]
                if surfaces:
                    # Blame ONLY the innermost (smallest-area / highest-z) underlying surface(s)
                    # that directly back `other_id`, never an outer enclosing card when an inner
                    # surface sits between them.
                    covering = [
                        (sid, sb, ov) for sid, sb, ov in surfaces
                        if ov >= 0.45 * other_area
                    ]
                    if covering:
                        innermost = min(covering, key=lambda item: float(item[1][2]) * float(item[1][3]))
                        bad_ids.add(innermost[0])
                    else:
                        best = max(surfaces, key=lambda item: (item[2], -(float(item[1][2]) * float(item[1][3]))))
                        bad_ids.add(best[0])
                else:
                    bad_ids.update(sid for sid, _, _ in direct_overlaps)
            else:
                near_overlaps = [
                    sid for sid in staged_ids
                    if (sb := _sel_box(sid)) is not None and _boxes_near(ob, sb, pad=2.0)
                ]
                if near_overlaps:
                    surfaces = [sid for sid in near_overlaps if (graph.selected.get(sid) or {}).get('kind') == 'surface']
                    if surfaces:
                        innermost = min(surfaces, key=lambda sid: float(_sel_box(sid)[2]) * float(_sel_box(sid)[3]))
                        bad_ids.add(innermost)
                    else:
                        bad_ids.update(near_overlaps)

    return frame_repairs, bad_ids


def _has_line_art_hole(source, bbox):
    """Choose a matte proposal for sparse strokes with an enclosed area on a plain ground."""
    x, y, w, h = map(round, bbox)
    if min(w, h) < 5 or x < 0 or y < 0 or x + w > source.shape[1] or y + h > source.shape[0]:
        return False
    crop = source[y:y+h, x:x+w].astype(np.int16)
    rim = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
    ground = np.median(rim, axis=0)
    if np.percentile(np.abs(rim - ground).max(axis=1), 65) > 20:
        return False
    ink = (np.abs(crop - ground).max(axis=2) > 42).astype(np.uint8)
    if not .04 <= ink.mean() <= .55:
        return False
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return hierarchy is not None and any(
        relation[3] >= 0 and cv2.contourArea(contour) >= 4
        for contour, relation in zip(contours, hierarchy[0])
    )


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
    recover_light_tile_marks: bool = False,
    prefer_rmbg_icons: bool = False,
    marked_routes: tuple | list = (),
    prepared_tables=None,
    post_text_checkpoint: dict | None = None,
    embedded_ui_frames: list[list[int]] | None = None,
    group_resolved_surfaces: bool = False,
    embedded_device_frames: list[list[int]] | None = None,
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
    devices_file = initial_baseline / 'embedded-device-frames.json'
    devices = embedded_device_frames
    if devices is None:
        devices = json.loads(devices_file.read_text('utf8')) if devices_file.is_file() else []
    retained_devices = list(devices)

    def touches_device(box):
        return box and any(intersection(box, frame) > .10 * min(box[2]*box[3], frame[2]*frame[3])
                           for frame in devices)

    marked_routes = [r for r in marked_routes if
                     not (r.get('route') in ('asset', 'table', 'picture') and touches_device(r.get('bbox')))
                     and not (r.get('route') == 'surface' and any(
                         intersection(r['bbox'], frame) >= .75*r['bbox'][2]*r['bbox'][3] for frame in devices))]
    reused_post_text = False
    if (
        post_text_checkpoint
        and post_text_checkpoint.get('path')
        and Path(post_text_checkpoint['path']).joinpath('scene.json').is_file()
        and not frame_overrides
        and not (skip_ids & set(post_text_checkpoint.get('staged_ids') or []))
    ):
        cur_base = Path(post_text_checkpoint['path'])
        staged_ids = list(post_text_checkpoint.get('staged_ids') or [])
        reused_post_text = True

    if not reused_post_text:
        table_fields = [r for r in marked_routes if r['route']=='table'
                        and (wanted_filter is None or r['id'] in wanted_filter)]
        covered_texts = {tid for record in initial_graph.selected.values() for tid in record.get('covered_text_ids', [])}
        if table_fields:
            from ..recovery.marked_tables import stage_tables
            cur_base, tables, covered = prepared_tables or stage_tables(cur_base, directory/tag/'marked-tables', table_fields, skip_ids)
            covered_texts.update(covered)
            staged_ids.extend(tables)
        elif include_containers and wanted_filter is None and layout_path and Path(layout_path).is_file():
            _layout_tbl = json.loads(Path(layout_path).read_text(encoding='utf8'))
            auto_tables = [
                {'id': f'auto-tbl-{t_idx}', 'route': 'table', 'bbox': list(r['bbox'])}
                for t_idx, r in enumerate(_layout_tbl.get('regions', []), 1)
                if r.get('label') == 'table' and float(r.get('score', 0)) >= 0.65
                and not touches_device(r.get('bbox'))
            ]
            if auto_tables:
                from ..recovery.marked_tables import stage_tables
                try:
                    cur_base, tables, covered = stage_tables(cur_base, directory / tag / 'auto-tables', auto_tables, skip_ids)
                    covered_texts.update(covered)
                    staged_ids.extend(tables)
                except Exception:
                    pass
        if include_containers and wanted_filter is None and layout_path:
            from ..recovery.chart_stage import stage_native_charts
            layout = json.loads(Path(layout_path).read_text(encoding='utf8'))
            if devices:
                layout['regions'] = [r for r in layout.get('regions', []) if not touches_device(r.get('bbox'))]
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
            if oid in skip_ids or oid in covered_texts or touches_device(tr.get('bbox')):
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
                    fit_origin=(experiment_profile != 'baseline'),
                    layer_directory=layers_dir,
                    edit_headroom=(frame_opt is None),
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
        if post_text_checkpoint is not None and not post_text_checkpoint.get('path'):
            post_text_checkpoint['path'] = str(cur_base)
            post_text_checkpoint['staged_ids'] = list(staged_ids)

    if include_assets:
        from ..recovery.region_crop import stage_region_crops
        for ui_idx, uframe in enumerate(embedded_ui_frames or (), 1):
            ui_id = f'ui-frame-{ui_idx}'
            if ui_id in skip_ids or (wanted_filter is not None and ui_id not in wanted_filter):
                continue
            device = uframe in devices
            if device and any(rec.get('bbox') == uframe and rec.get('payload', {}).get(
                    'background_evidence', {}).get('method') == 'source-device-bezel-and-paired-side-strips/1'
                    for rec in initial_graph.selected.values()):
                retained_devices.remove(uframe)
                continue
            if not device and any(intersection(f.get('picture_frame', [0, 0, 0, 0]), uframe) >= 0.70 * max(1.0, uframe[2] * uframe[3])
                   for f in marked_routes if f.get('picture_frame')):
                continue
            ui_seed = {'id': ui_id, 'bbox': uframe, 'kind': 'asset', 'picture_frame': uframe,
                       'device_frame': device, 'screen_backing': recover_light_tile_marks and not device}
            if device:
                ui_seed['text'] = ' / '.join(t.text for t in initial_graph.text.texts
                    if intersection(t.bbox, uframe) >= .9 * t.bbox[2] * t.bbox[3])
            ui_ir, ui_ids, evidence = stage_region_crops(
                cur_base,
                directory / tag / f'ui-frame-{ui_idx}',
                seeds=[ui_seed],
                allow_transparent_upgrade=False,
                max_area_fraction=0.65,
                protected_text_ids=set(),  # Device path checks source-measured peer gutters before every write.
                skip_seed_ids=skip_ids,
            )
            if ui_ids:
                cur_base = ui_ir.parent
                staged_ids.extend(ui_ids)
                if device:
                    retained_devices.remove(uframe)
            elif device:
                stage_errors.append({'object_id': ui_id, 'kind': 'asset', 'bbox': uframe,
                                     'failure': 'PICTURE_NOT_ISOLATED', 'reason': str(evidence)[:500]})
        for field in marked_routes:
            frame = field.get('picture_frame')
            if not frame or field['id'] in skip_ids or (wanted_filter is not None and field['id'] not in wanted_filter):
                continue
            if any(intersection(frame, device_box) > 0 for device_box in devices):
                continue
            seed = {'id': field['id'], 'bbox': frame, 'kind': 'asset',
                    'picture_frame': frame if field.get('intrinsic_ui') else None}
            pic_ir, pic_ids, evidence = stage_region_crops(
                cur_base, directory/tag/('picture-'+key(field['id'])[:12]), seeds=[seed], skip_seed_ids=skip_ids)
            if pic_ids:
                cur_base = pic_ir.parent
                staged_ids.extend(pic_ids)
            elif not any(field['id'] in rec.get('seed_ids', []) for rec in initial_graph.selected.values()):
                stage_errors.append({'object_id': field['id'], 'kind': 'asset', 'bbox': frame,
                                     'failure': 'PICTURE_NOT_ISOLATED', 'reason': str(evidence)[:500]})

    if include_assets and layers_dir:
        cur_graph = SceneGraph.model_validate_json((cur_base / 'scene.json').read_text('utf8'))
        from .marked_routes import bind_asset_layers
        deferred_list = bind_asset_layers(
            [r['new_asset'] for r in marked_routes if r.get('new_asset') and r.get('id') not in skip_ids],
            layers_dir, cur_graph.source_sha256,
        )
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
                if touches_device(d_box):
                    continue
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
            and not touches_device(a.bbox)
            and (wanted_filter is None or a.id in wanted_filter)
        ]
        asset_cache_root = page_cache_dir / 'asset-extract-cache'
        routing_pixels = None
        if prefer_rmbg_icons and asset_targets and matting.available():
            with Image.open(source_path) as image:
                routing_pixels = np.asarray(image.convert('RGB'))
        for idx, asset in enumerate(asset_targets):
            step_dir = directory / tag / f'asset-{idx:03d}-{key(asset.id)[:12]}'
            step_dir.mkdir(parents=True, exist_ok=True)
            active_graph = SceneGraph.model_validate_json((cur_base / 'scene.json').read_text('utf8'))
            # Content-addressed cache key on local plate crop + nearby selected owners so Pass 2
            # isolation reuses exact extraction results without re-running SAM / RMBG-2.0.
            ax0, ay0, aw0, ah0 = [int(round(v)) for v in asset.bbox]
            plate_path_cur = cur_base / (active_graph.raster_backing or {}).get('path', 'source-plate.png')
            plate_patch_sha = ''
            if plate_path_cur.is_file():
                with Image.open(plate_path_cur) as pim:
                    pw_c, ph_c = pim.size
                    crop_box_c = (max(0, ax0 - 16), max(0, ay0 - 16), min(pw_c, ax0 + aw0 + 16), min(ph_c, ay0 + ah0 + 16))
                    plate_patch_sha = key(pim.crop(crop_box_c).convert('RGB').tobytes().hex()[:4096] + str(crop_box_c))
            nearby_sel = sorted(
                (oid, rec.get('owner_evidence', {}).get('sha256', ''))
                for oid, rec in active_graph.selected.items()
                if (rb := rec.get('restore_bbox') or rec.get('bbox')) is not None
                and intersection([ax0 - 16, ay0 - 16, aw0 + 32, ah0 + 32], rb) > 0
            )
            base_cache_sig = {
                'asset_id': asset.id,
                'bbox': list(asset.bbox),
                'source': active_graph.source_sha256,
                'layers': str(layers_dir),
                'plate_patch': plate_patch_sha,
                'nearby_sel': nearby_sel,
                'recover_light_tile_marks': recover_light_tile_marks,
            }

            def _cached_retry(strategy_name: str, fallback_dir: Path, **extra_kw):
                ckey = key({**base_cache_sig, 'strategy': strategy_name})[:16]
                cdir = asset_cache_root / f'{asset.id}-{strategy_name}-{ckey}'
                meta_f = cdir / 'cached-result.json'
                if meta_f.is_file():
                    try:
                        cached_res = json.loads(meta_f.read_text('utf8'))
                        cand_d = cached_res.get('candidate_directory')
                        if cached_res.get('status') != 'CANDIDATE_READY' or (cand_d and Path(cand_d).is_dir()):
                            return cached_res
                    except Exception:
                        pass
                run_dir = cdir / 'run'
                if run_dir.exists():
                    shutil.rmtree(run_dir, ignore_errors=True)
                cdir.mkdir(parents=True, exist_ok=True)
                out_res = retry_asset_extraction(
                    active_graph, asset.id, source_path, layers_dir,
                    cur_base, run_dir, strategy=strategy_name, light_tile_marks=recover_light_tile_marks, **extra_kw,
                )
                write(meta_f, out_res)
                return out_res

            try:
                choices = []
                matte_ir = None
                prefer_matte = (asset.kind == 'icon' and routing_pixels is not None
                                and _has_line_art_hole(routing_pixels, asset.bbox))
                if prefer_matte:
                    try:
                        matte = _cached_retry('rmbg2', step_dir / 'extract-rmbg2')
                        if matte.get('status') == 'CANDIDATE_READY':
                            matte_ir = stage_asset_candidate(
                                matte['candidate_directory'], active_graph, cur_base,
                                step_dir / 'staged-matte', layer_directory=layers_dir)
                        choices.append(('rmbg2', matte))
                    except Exception as exc:
                        choices.append(('rmbg2', {'status':'REJECTED', 'reason':str(exc)[:300]}))
                if matte_ir is None:
                    res = _cached_retry('verified-backing', step_dir / 'extract')
                    choices.append(('verified-backing', res))
                else:
                    res = matte
                compare = asset.kind in ('icon', 'logo', 'decoration')
                if matte_ir is None and (compare or res.get('status') != 'CANDIDATE_READY') and sam_enabled:
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
                            res_sam = _cached_retry(
                                'sam', step_dir / 'extract-sam',
                                sam_plan=plan_file, sam_directory=sam_out,
                            )
                            choices.append(('sam',res_sam))
                            if res_sam.get('status') == 'CANDIDATE_READY':
                                res = res_sam
                        except (ValueError,OSError,RuntimeError) as exc:
                            choices.append(('sam',{'status':'UNAVAILABLE','reason':type(exc).__name__}))
                need_rmbg2 = (
                    not prefer_matte and compare and matting.available()
                    and (experiment_profile not in (None, 'baseline')
                         or not any(c.get('status') == 'CANDIDATE_READY' for _, c in choices))
                )
                if need_rmbg2:
                    try:
                        res_matte = _cached_retry('rmbg2', step_dir / 'extract-rmbg2')
                        choices.append(('rmbg2',res_matte))
                    except Exception as exc:
                        choices.append(('rmbg2',{'status':'UNAVAILABLE','reason':type(exc).__name__}))
                for op in ('vitmatte','local-layerd'):
                    if matte_ir is not None:
                        break
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
                    'preferred_route':'rmbg2' if prefer_matte else 'verified-backing',
                    'attempts':[{'strategy':name,'status':c.get('status'),'reason':c.get('reason')} for name,c in choices],
                    'comparable_edge_support':comparable,
                    'ranked':[{k:v for k,v in r.items() if k!='candidate'} for r in scored]})
                if scored:
                    res = scored[0]['candidate']
                if res.get('status') == 'CANDIDATE_READY':
                    ir = matte_ir or stage_asset_candidate(
                        res['candidate_directory'], active_graph, cur_base,
                        step_dir / 'staged', layer_directory=layers_dir,
                    )
                    cur_base = ir.parent
                    staged_id = asset.id
                    if asset.kind == 'icon':
                        from ..recovery.vector_icons import upgrade_staged_icon_to_vector_shape
                        try:
                            vec = upgrade_staged_icon_to_vector_shape(cur_base, asset.id, step_dir / 'vector')
                            if vec is not None:
                                cur_base = vec[0].parent
                                staged_id = vec[1]
                        except Exception:
                            pass
                    staged_ids.append(staged_id)
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
            if touches_device(route['bbox']):
                continue
            from ..recovery.region_crop import stage_region_crops
            graph=SceneGraph.model_validate_json((cur_base/'scene.json').read_text('utf8'))
            if route['id'] not in graph.selected:
                ir,ids,_ = stage_region_crops(cur_base,directory/tag/('marked-asset-'+key(route['id'])[:12]),
                    seeds=[{'id':route['id'],'bbox':route['bbox'],'kind':'asset'}],skip_seed_ids=skip_ids)
                if ids:
                    cur_base=ir.parent
                    staged_ids.extend(ids)

    if include_containers and (wanted_filter is None or any(i.startswith(('s-', 'a-c')) for i in wanted_filter)):
        from ..recovery.container_stage import stage_containers_and_surfaces
        cont_dir = directory / tag / 'containers-staged'
        try:
            cont_ir, cont_ids = stage_containers_and_surfaces(
                cur_base, cont_dir, skip_ids=skip_ids, logo_ids=logo_ids, wanted_ids=wanted_filter,
                retained_frames=retained_devices,
                recover_light_tile_marks=recover_light_tile_marks,
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
        try:
            from ..recovery.marked_surfaces import stage_marked_surfaces
            card_ir, card_ids = stage_marked_surfaces(cur_base, directory/tag/'marked-cards',
                                                      surface_routes, skip_ids)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            for proposal in surface_routes:
                stage_errors.append({'object_id':proposal['id'], 'kind':'surface', 'bbox':proposal['bbox'],
                                     'failure':classify_exc(exc, 'generation'), 'reason':str(exc)[:300]})
        else:
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
        crop_seeds = [seed for seed in crop_seeds if not touches_device(seed['bbox'])]
        if crop_seeds:
            crop_dir = directory / tag / 'region-crops-staged'
            try:
                crop_ir, crop_ids, _ = stage_region_crops(
                    cur_base, crop_dir,
                    seeds=crop_seeds,
                    allow_move_unsafe=bool(allow_move_unsafe),
                    protected_text_ids=protected_text_ids,
                    skip_seed_ids=skip_ids,
                    preserve_boxes=retained_devices,
                )
                if crop_ids:
                    from ..recovery.vector_icons import upgrade_crops_to_vector_shapes
                    vec_ir, crop_ids = upgrade_crops_to_vector_shapes(
                        crop_ir.parent, crop_ids, directory / tag / 'region-crops-vector'
                    )
                    cur_base = vec_ir.parent
                    staged_ids.extend(crop_ids)
                    if include_containers:
                        from ..recovery.container_stage import stage_containers_and_surfaces
                        post_cont_dir = directory / tag / 'post-crop-containers-staged'
                        pc_ir, pc_ids = stage_containers_and_surfaces(
                            cur_base, post_cont_dir, skip_ids=skip_ids, logo_ids=logo_ids,
                            retained_frames=retained_devices,
                            recover_light_tile_marks=recover_light_tile_marks,
                        )
                        if pc_ids:
                            cur_base = pc_ir.parent
                            staged_ids.extend(pc_ids)
            except Exception as exc:
                stage_errors.append({
                    'object_id': 'region_crops', 'kind': 'asset', 'bbox': [],
                    'failure': classify_exc(exc, 'generation'), 'reason': str(exc),
                })

    discover_groups = group_resolved_surfaces and include_containers and wanted_filter is None
    if marked_routes or discover_groups:
        from .marked_routes import route_trace, plan_move_groups, resolved_surface_groups
        graph=SceneGraph.model_validate_json((cur_base/'scene.json').read_text('utf8'))
        movement_routes = [r if wanted_filter is None or r['id'] in wanted_filter
                           else {**r,'move_with':[]} for r in marked_routes]
        proposals = plan_move_groups(movement_routes, graph)
        if discover_groups:
            group_issues = []
            proposals.extend(resolved_surface_groups(graph, cur_base, diagnostics=group_issues))
            blocked_groups = {frozenset(err['members']) for err in group_issues}
            proposals = [group for group in proposals if frozenset(group) not in blocked_groups]
            stage_errors.extend(group_issues)
        occupied = set(skip_ids)
        for entry in graph.selected.values():
            occupied.update(entry.get('move_members', []))
        groups = []
        for group in proposals:
            group_key = 'g-'+key(sorted(group))[:16]
            if group_key not in skip_ids and not occupied.intersection(group):
                groups.append(group)
                occupied.update(group)
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
                if cur_base == initial_baseline or (reused_post_text and cur_base == Path(post_text_checkpoint['path'])):
                    grouped = directory/tag/'movement'
                    shutil.copytree(cur_base,grouped)
                    cur_base = grouped
                write(cur_base/'scene.json',graph.model_dump())
                write(cur_base/'slideir.json',ir)
                # A group-only change must go through the same compile and Office commit.
                staged_ids.extend(i for g in accepted for i in g if i not in staged_ids)
        if marked_routes:
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
    crop_spec = candidate.get('crop_spec')
    if not spec and not crop_spec:
        return False
    staged_graph = SceneGraph.model_validate_json((Path(candidate['ir']).parent / 'scene.json').read_text('utf8'))
    staged_ids = candidate.get('object_ids') or [candidate['object_id']]
    failed_group = (office_res.get('office') or {}).get('failed_object_id')
    if spec and spec.get('group_resolved_surfaces') and isinstance(failed_group,str) and failed_group.startswith('g-'):
        from ..parallel_rebuild import adapt
        current_ir = Path(candidate['ir'])
        groups = {g['id']:g['member_ids'] for g in json.loads(current_ir.read_text('utf8'))['groups']}
        parent_ir = json.loads((Path(spec['initial_baseline'])/'slideir.json').read_text('utf8'))
        if failed_group in groups and failed_group not in {g['id'] for g in parent_ir['groups']}:
            members = set(groups[failed_group])
            holder = next((rec for rec in staged_graph.selected.values()
                           if set(rec.get('move_members', [])) == members), None)
            if holder is not None:
                holder.pop('move_members')
                holder.pop('card_move_group',None)
                ir = adapt(staged_graph, current_ir.parent)
                if {g['id'] for g in ir['groups']} == set(groups)-{failed_group}:
                    base = directory/'ungrouped'
                    shutil.copytree(current_ir.parent,base)
                    write(base/'scene.json',staged_graph.model_dump())
                    write(base/'slideir.json',ir)
                    candidate['ir'] = str(base/'slideir.json')
                    candidate['files'] = {str(p):digest(p) for p in base.rglob('*') if p.is_file()}
                    candidate['fingerprint'] = key({'removed_group':failed_group,'ir':digest(base/'slideir.json')})
                    candidate['isolation_attempts'] = int(candidate.get('isolation_attempts',0))+1
                    candidate['isolation_attempted'] = True
                    candidate['isolated_skip_ids'] = sorted(set(candidate.get('isolated_skip_ids',[])) | {failed_group})
                    candidate.setdefault('isolation_failures',[]).append({'object_id':failed_group,'kind':'group',
                        'members':sorted(members),'failure':'GROUP_EDIT_FAILED','reason':'Kept all native member objects'})
                    return True
    frame_repairs, bad_ids = _collect_office_defects(office_res, staged_graph, staged_ids)
    if spec:
        if not (frame_repairs or (bad_ids and len(bad_ids) < len(staged_ids))):
            return False
    else:
        if not (bad_ids and len(bad_ids) < len(staged_ids)):
            return False

    attempt_idx = int(candidate.get('isolation_attempts', 1 if candidate.get('isolation_attempted') else 0)) + 1
    prior_skip = set(candidate.get('isolated_skip_ids') or [])
    prior_overrides = dict(candidate.get('auto_frame_overrides') or {})
    bad_seeds = {
        sid
        for oid in bad_ids
        for sid in (staged_graph.selected.get(oid, {}).get('seed_ids') or [])
        if sid
    }

    retry_errors: list[dict] = []
    overrides: dict = {}
    if crop_spec and not spec:
        from ..recovery.region_crop import stage_region_crops
        from ..recovery.container_stage import stage_containers_and_surfaces

        skip = prior_skip | set(bad_ids) | bad_seeds
        initial_baseline = Path(crop_spec['initial_baseline'])
        protected_ids = (
            set(crop_spec['protected_text_ids'])
            if crop_spec.get('protected_text_ids') is not None
            else None
        )
        directory.mkdir(parents=True, exist_ok=True)
        crop_ir, crop_ids, _ = stage_region_crops(
            initial_baseline,
            directory / f'crop-pass{attempt_idx + 1}',
            seeds=crop_spec['seeds'],
            allow_movable=bool(crop_spec.get('allow_movable', True)),
            allow_move_unsafe=bool(crop_spec.get('allow_move_unsafe', False)),
            protected_text_ids=protected_ids,
            skip_seed_ids=skip,
        )
        if crop_ids:
            from ..recovery.vector_icons import upgrade_crops_to_vector_shapes
            crop_ir, crop_ids = upgrade_crops_to_vector_shapes(
                crop_ir.parent, crop_ids, directory / f'crop-vector-pass{attempt_idx + 1}'
            )
        repaired_base = crop_ir.parent if crop_ids else initial_baseline
        repaired_ids = list(crop_ids)
        if crop_ids and crop_spec.get('upgrade_containers'):
            try:
                cont_ir, cont_ids = stage_containers_and_surfaces(
                    repaired_base,
                    directory / f'post-crop-containers-pass{attempt_idx + 1}',
                    skip_ids=skip,
                    logo_ids=set(crop_spec.get('logo_ids') or []),
                )
                if cont_ids:
                    repaired_base = cont_ir.parent
                    repaired_ids.extend(cont_ids)
            except Exception as exc:
                retry_errors.append({
                    'object_id': 'post_crop_containers',
                    'kind': 'surface',
                    'bbox': [],
                    'failure': 'CONTAINER_STAGE_SKIPPED',
                    'reason': str(exc),
                })
        merged_overrides = prior_overrides
        fp_key = {'crop': repaired_ids}
    else:
        row = service.page(current, candidate['page'])
        initial_baseline = Path(spec['initial_baseline'])
        initial_graph = SceneGraph.model_validate_json((initial_baseline / 'scene.json').read_text('utf8'))
        fonts = json.loads(Path(current['fonts']).read_text('utf8'))
        overrides = {oid: fb['measured_frame_minimum_px'] for oid, fb in frame_repairs.items()}
        merged_overrides = {**prior_overrides, **overrides}
        skip = prior_skip | set(bad_ids) | bad_seeds | set(spec.get('pass1_failed_ids', []))
        skip.update(i for r in spec.get('marked_routes', []) if r['route']=='preserve'
                    for i in [r['id'],*r['target_ids']])
        unrec_pass2 = set(spec.get('unrecoverable_text_ids', [])) - bad_seeds
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
            tag=f'stage-pass{attempt_idx + 1}',
            experiment_profile=spec.get('experiment_profile'),
            recover_light_tile_marks=spec.get('recover_light_tile_marks', False),
            prefer_rmbg_icons=spec.get('prefer_rmbg_icons', False),
            marked_routes=spec.get('marked_routes', ()),
            trials_spec=spec['trials'],
            frame_overrides=merged_overrides,
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
            post_text_checkpoint=spec.get('post_text_checkpoint'),
            embedded_ui_frames=spec.get('embedded_ui_frames'),
            group_resolved_surfaces=bool(spec.get('group_resolved_surfaces')),
            embedded_device_frames=spec.get('embedded_device_frames'),
        )
        fp_key = {'batch': repaired_ids}

    isolation_failures: list[dict] = list(candidate.get('isolation_failures') or [])
    existing_keys = {(f.get('object_id'), f.get('failure')) for f in isolation_failures}
    for oid in sorted(bad_ids):
        if (oid, 'OFFICE_RESIDUAL_OR_EDIT_FAILED') in existing_keys:
            continue
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
        existing_keys.add((oid, 'OFFICE_RESIDUAL_OR_EDIT_FAILED'))
    for err in retry_errors:
        if err['object_id'] in frame_repairs:
            err['measured_frame_minimum_px'] = frame_repairs[err['object_id']]['measured_frame_minimum_px']
        if (err.get('object_id'), err.get('failure')) not in existing_keys:
            isolation_failures.append(err)
            existing_keys.add((err.get('object_id'), err.get('failure')))

    candidate['isolation_attempts'] = attempt_idx
    candidate['isolation_attempted'] = True
    candidate['isolation_failures'] = isolation_failures
    candidate['isolated_skip_ids'] = sorted(skip)
    candidate['isolated_bad_ids'] = sorted(set(candidate.get('isolated_bad_ids') or []) | set(bad_ids))
    if spec:
        candidate['auto_frame_overrides'] = merged_overrides
    if not repaired_ids:
        return False

    final_ir = repaired_base / 'slideir.json'
    candidate['ir'] = str(final_ir)
    candidate['object_id'] = repaired_ids[0]
    candidate['object_ids'] = repaired_ids
    candidate['auto_repaired_frame_ids'] = [oid for oid in merged_overrides if oid in repaired_ids]
    candidate['fingerprint'] = key({**fp_key, 'ir': digest(final_ir)})
    candidate['files'] = {str(p): digest(p) for p in repaired_base.rglob('*') if p.is_file()}
    return True


def editable_batch_arguments(arguments):
    defaults = {'include_assets': True, 'include_containers': True, 'include_region_crops': False,
                'allow_move_unsafe': False, 'use_inpaint': True, 'verify_office': True}
    unsupported = set(arguments) - set(defaults) - {'page', 'object_ids'}
    if unsupported:
        raise ValueError('unsupported batch arguments: ' + ', '.join(sorted(unsupported)))
    page = arguments.get('page')
    if type(page) is not int or page < 1:
        raise ValueError('batch page must be a positive integer')
    ids = arguments.get('object_ids')
    if ids is not None and (not isinstance(ids, list)
            or any(not isinstance(oid, str) or not oid or oid.strip() != oid for oid in ids)):
        raise ValueError('object_ids must be a list of nonempty object IDs or null')
    flags = {name: arguments.get(name, default) for name, default in defaults.items()}
    if any(type(value) is not bool for value in flags.values()):
        raise ValueError('batch flags require JSON true or false')
    return {'page': page, 'object_ids': sorted(set(ids)) if ids is not None else None, **flags}


def _batch_request_binding(state, current, row, page, routes, options):
    """Identify the reconstruction inputs independently of the tool turn or candidate ID."""
    from ..parallel_config import ROOT
    from ..recovery.asset_cache import (code_fingerprint, compiler_runtime_fingerprint,
                                       file_hashes, python_runtime)
    from ..object_registry import SAM_BINDING
    from ..tolerance_config import load_tolerance_config

    page_key = str(page)
    evidence = {
        name: current.get(name, {}).get(page_key)
        for name in ('ocr', 'page_structure', 'layout', 'deferred_assets')
    }
    environment_files = [os.environ.get(name) for name in (
        'IMAGE2PPT_EXPERIMENT_CONFIG', 'IMAGE2PPT_SAM_CONFIG')]
    configuration_files = [
        ROOT / '.runtime/parallel-services.json', ROOT / '.runtime/sam-client.json',
        *environment_files,
    ]
    font_config = json.loads(Path(current['fonts']).read_text('utf8'))
    request = json.loads((state.job/'request.json').read_text('utf8'))
    matte = None
    if options['include_assets'] or options['include_region_crops']:
        from .. import matting
        if request.get('experiment_profile') in ('vitmatte', 'local-layerd', 'vtracer', 'combined'):
            # These optional workers run outside this process and expose no complete
            # versioned toolchain for reuse. Their existing Office receipts remain valid.
            return None
        matte = matting.runtime_binding()
    return key({
        'page': page, 'options': options, 'request': digest(state.job / 'request.json'),
        'source': digest(row['snapshot']['path']), 'scene': digest(row['scene']),
        'baseline': file_hashes(Path(row['ir']).parent),
        'layers': file_hashes(row['layers']) if row.get('layers') else None,
        'evidence': {name: digest(path) if path else None for name, path in evidence.items()},
        'routes': routes, 'relations': current.get('scene_relations', {}).get(page_key),
        'font_config': font_config,
        'font_files': {path: digest(path) for path in {f['file'] for f in font_config['fonts']}},
        'configuration': {str(path): digest(path) for path in configuration_files
                          if path and Path(path).is_file()},
        'tools': {name: os.environ.get(name) for name in (
            'IMAGE2PPT_SAM_URL', 'SAM_URL', 'LAYERD_MODEL_BINDING', 'IMAGE2PPT_OFFICE_MODE',
            'IMAGE2PPT_NODE')},
        'sam_model': SAM_BINDING, 'matting': matte,
        'compiler': compiler_runtime_fingerprint(local_office=os.environ.get('IMAGE2PPT_OFFICE_MODE') != 'remote'),
        'tolerance': load_tolerance_config(),
        'implementation': code_fingerprint(ROOT, compiler=True), 'python': python_runtime(),
    })


def _reuse_validated_batch(service, state, current, directory, binding):
    from .state import RuntimeConflict

    for candidate in current['candidates'].values():
        if (candidate.get('batch_request_binding') != binding or candidate['status'] != 'VALIDATED'
                or candidate.get('validation', {}).get('status') != 'PASS'):
            continue
        receipt = candidate.get('batch_result')
        if not receipt or not candidate['validation'].get('files'):
            continue
        if os.environ.get('IMAGE2PPT_OFFICE_MODE') == 'remote':
            # The worker that actually ran Office can register after generation started.
            worker = state.job.parent / 'office-worker-binding.json'
            actual_worker = candidate['validation'].get('worker_binding')
            if (not actual_worker or not worker.is_file()
                    or json.loads(worker.read_text('utf8')).get('binding') != actual_worker):
                continue
        path = Path(receipt['path'])
        if not path.is_file() or digest(path) != receipt['sha256']:
            raise RuntimeConflict('validated batch result changed')
        # The VALIDATED path verifies parent, code, candidate files and Office evidence
        # without rerunning the compiler, GPU tools or Office.
        service.validate_candidate(state, current, directory, candidate_id=candidate['id'])
        result = json.loads(path.read_text('utf8'))
        result.pop('timings', None)  # Timings in the immutable receipt belong to the first run.
        return {**result, 'cache_hit': True, 'next_step': _batch_commit_hint(
            candidate['id'], len(candidate['object_ids']), current['version'] + 1)}
    return None


def _batch_commit_hint(candidate_id, count, version):
    return (f'Batch candidate {candidate_id} is VALIDATED ({count} staged, 0 committed). '
            f'Call commit_candidate(candidate_id="{candidate_id}", expected_version={version}) '
            'to formally commit, then inspect failed_regions or compose_delivery.')


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
    variant = request.get('native_variant','combined')
    include_containers = include_containers and variant != 'joint-only'
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
    binding = None
    if request.get('orchestrator') == 'editable_reconstruction' and verify_office:
        options = {
            'object_ids': sorted(wanted_filter) if wanted_filter is not None else None,
            'include_assets': include_assets, 'include_containers': include_containers,
            'include_region_crops': include_region_crops, 'allow_move_unsafe': allow_move_unsafe,
            'use_inpaint': use_inpaint,
        }
        try:
            binding = _batch_request_binding(state, current, row, page, routes, options)
        except (ImportError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            # Incomplete optional inputs (for example an uninstalled font) cannot
            # justify a cache hit. Let the normal builder measure the actual result.
            state.event({'tool': 'batch_reconstruct_page', 'page': page,
                         'cache_unavailable': type(exc).__name__})
        if binding is not None:
            existing = _reuse_validated_batch(service, state, current, directory, binding)
            if existing is not None:
                return existing
    t_batch_start = time.perf_counter()
    timings: dict[str, float] = {}

    failed_regions: list[dict] = []

    # Phase 1: Parallel local text candidate trials against initial_baseline.
    ui_frames = current.get('scene_relations', {}).get(str(page), {}).get('embedded_ui_frames')
    if ui_frames is None:
        ui_f_path = initial_baseline / 'embedded-ui-frames.json'
        ui_frames = json.loads(ui_f_path.read_text('utf8')) if ui_f_path.is_file() else []
    device_frames = current.get('scene_relations', {}).get(str(page), {}).get('embedded_device_frames')
    if device_frames is None:
        devices_file = initial_baseline / 'embedded-device-frames.json'
        device_frames = json.loads(devices_file.read_text('utf8')) if devices_file.is_file() else []
    ui_text_ids = {
        t.id for t in initial_graph.text.texts
        if any(intersection(t.bbox, uf) >= 0.75 * max(1.0, t.bbox[2] * t.bbox[3]) for uf in ui_frames)
    }
    text_targets = [
        t for t in initial_graph.text.texts
        if t.id not in initial_graph.selected
        and t.id not in ui_text_ids
        and initial_graph.ownership.get(t.id) == 'RESOLVED'
        and (wanted_filter is None or t.id in wanted_filter)
    ]
    priority_ids = {tid for r in routes if r['route'] in ('text', 'table') for tid in r['target_ids']}
    preserved_ids = {i for r in routes if r['route']=='preserve' for i in [r['id'],*r['target_ids']]}
    prepared_tables=None
    table_fields=[r for r in routes if r['route']=='table' and (wanted_filter is None or r['id'] in wanted_filter)]
    table_fields = [r for r in table_fields if not any(intersection(r['bbox'], frame) >
                    .1 * min(r['bbox'][2]*r['bbox'][3], frame[2]*frame[3]) for frame in device_frames)]
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
    page_count = max(1, len(request.get('sources') or [1]))
    model_page = (page, len(request['sources'])) if routes else None
    budget = AdaptiveBudget(Store(state.job / 'state'), request.get('adaptive_budget'), model_page=model_page, page_count=page_count)
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
        workers = min(3, max(1, len(text_targets)))
        # Capture only masks that survived ownership and text-mask completion.
        # Sharing a model call does not share pixel ownership or commit decisions.
        from ..parallel_rebuild import array_sha
        masks = {}
        def local_probe(obj):
            def capture(image, mask):
                masks[obj.id] = mask.copy()
                raise ValueError('deferred to local group repair')
            return _probe_text(obj, inpaint=(safe_inpaint if variant == 'shape-only' else capture) if safe_inpaint else None)
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
            still_failed = [(idx, obj) for idx, obj in safe_members if outcomes[idx]['status'] != 'PRECHECK_PASS']
            if background is not None and still_failed and safe_inpaint is not None:
                try:
                    lama_repaired = safe_inpaint(source_pixels, union)
                except (OSError, TimeoutError, ValueError, RuntimeError):
                    lama_repaired = None
                if lama_repaired is not None:
                    for idx, obj in still_failed:
                        def lama_patch(image, mask):
                            if not np.array_equal(mask, masks[obj.id]):
                                raise ValueError('local ownership changed before repair')
                            return lama_repaired
                        lama_patch.receipt = {'group': [member.id for _, member in safe_members],
                                              'union_sha256': array_sha(union.astype('uint8')),
                                              'provider': 'lama'}
                        outcomes[idx] = _probe_text(obj, folder='joint-trials-lama', inpaint=lama_patch)
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
        | ui_text_ids
        | {t.id for t in initial_graph.text.texts
           if t.id not in initial_graph.selected and (t.id in logo_ids or initial_graph.ownership.get(t.id) != 'RESOLVED')}
    )
    current.setdefault('batch_unrecoverable_texts', {})[str(page)] = sorted(unrecoverable_text_ids)

    cand_ir_ref: list[Path] = []
    prefer_rmbg_icons = (request.get('orchestrator') == 'editable_reconstruction'
                         and request.get('experiment_profile') in (None, 'baseline'))

    def _enriched_failures() -> list[dict]:
        staged_crop_boxes: list[tuple[str, list[float]]] = []
        if cand_ir_ref and cand_ir_ref[0].is_file():
            cg = SceneGraph.model_validate_json((cand_ir_ref[0].parent / 'scene.json').read_text('utf8'))
            staged_crop_boxes = [
                (rid, list(rec['bbox'])) for rid, rec in cg.selected.items()
                if rec.get('region_crop') and len(rec.get('bbox', [])) == 4
            ]
        return [_classify_failed_region(err, cg if staged_crop_boxes else initial_graph, logo_ids, staged_crop_boxes) for err in failed_regions]

    t_stage_start = time.perf_counter()
    timings['text_probe_s'] = round(t_stage_start - t_batch_start, 2)
    gc.collect()

    post_text_ckpt: dict = {}
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
        recover_light_tile_marks=request.get('orchestrator') == 'editable_reconstruction',
        prefer_rmbg_icons=prefer_rmbg_icons,
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
        post_text_checkpoint=post_text_ckpt,
        embedded_ui_frames=ui_frames,
        group_resolved_surfaces=request.get('orchestrator') == 'editable_reconstruction',
        embedded_device_frames=device_frames,
    )
    timings['stage_pass1_s'] = round(time.perf_counter() - t_stage_start, 2)
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
            'recover_light_tile_marks':request.get('orchestrator') == 'editable_reconstruction',
            'prefer_rmbg_icons':prefer_rmbg_icons,
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
            'post_text_checkpoint': post_text_ckpt,
            'embedded_ui_frames': ui_frames,
            'group_resolved_surfaces': request.get('orchestrator') == 'editable_reconstruction',
            'embedded_device_frames': device_frames,
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
    t_val_start = time.perf_counter()
    val = service.validate_candidate(state, current, directory / 'validate', candidate_id=batch_cid)
    timings['validate_office_s'] = round(time.perf_counter() - t_val_start, 2)
    timings['total_s'] = round(time.perf_counter() - t_batch_start, 2)
    if isinstance(val.get('timings'), dict):
        timings.update({f'val_{k}': v for k, v in val['timings'].items()})
    cand['timings'] = timings
    cand_ir_ref[0] = Path(cand['ir'])
    failed_regions.extend(cand.get('isolation_failures', []))
    staged_ids = cand.get('object_ids', staged_ids)
    auto_repaired_frames = cand.get('auto_repaired_frame_ids', [])
    isolated_bad_ids = cand.get('isolated_bad_ids', [])
    enriched = _enriched_failures()

    if cand.get('status') == 'VALIDATED' and val.get('status') == 'PASS':
        result = {
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
            'timings': timings,
            'failed_count': len(enriched),
            'failed_regions': enriched,
            'next_step': _batch_commit_hint(batch_cid, len(staged_ids), next_version),
        }
        if binding is not None:
            receipt = directory / 'batch-result.json'
            write(receipt, result)
            cand['batch_request_binding'] = binding
            cand['batch_result'] = {'path': str(receipt), 'sha256': digest(receipt)}
        return result

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
        'timings': timings,
        'office_status': val.get('status'),
        'failed_count': len(enriched),
        'failed_regions': enriched,
        'next_step': f'Batch candidate {batch_cid} is not validated (status={val.get("status")}); commit_candidate will remain blocked until hard/Office checks pass.',
    }
