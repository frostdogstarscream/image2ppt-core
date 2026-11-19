"""Native PowerPoint `chart` stage for data-bearing bar/column charts with axis/category labels.

Unlabeled decorative bar-staircase graphics (which have no axis ticks or category
wording) are deliberately NOT turned into Excel-backed charts here so we never
invent fake axis numbers; instead `native_shapes` reconstructs each bar as a
native gradient `Shape` (`rect`/`roundRect`). When a `chart` region DOES carry
aligned category labels (and optional numeric data/axis labels), this module
stages a true native SlideIR `chart` node (`COMPILER.native_charts = true`).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..contracts import intersection
from ..materialize import digest, write
from ..object_extract import rectangle
from ..parallel_contracts import SceneGraph
from ..parallel_rebuild import adapt, array_sha

_NUM_RE = re.compile(r'^[-+]?\d+(?:\.\d+)?%?$')


def _parse_num(text: str) -> float | None:
    s = (text or '').strip().replace(',', '')
    if not _NUM_RE.match(s):
        return None
    if s.endswith('%'):
        return None  # The current chart payload cannot preserve a percent display format.
    return float(s)


def stage_native_charts(
    baseline_directory: Path | str,
    output_directory: Path | str,
    *,
    layout: dict | None = None,
    fonts: dict | None = None,
    skip_ids: set[str] | None = None,
) -> tuple[Path, list[str]]:
    """Detect and stage data-bearing bar charts with real category labels as native `chart` nodes."""
    skip = skip_ids or set()
    baseline = Path(baseline_directory).resolve()
    output = Path(output_directory).resolve()
    if not layout or not layout.get('regions'):
        return baseline / 'slideir.json', []

    chart_regions = [
        r for r in layout.get('regions', [])
        if r.get('label') == 'chart' and float(r.get('score', 0.0)) >= 0.55 and len(r.get('bbox', [])) == 4
    ]
    if not chart_regions:
        return baseline / 'slideir.json', []

    graph = SceneGraph.model_validate_json((baseline / 'scene.json').read_text('utf8'))
    rb = graph.raster_backing or {}
    plate_in = (baseline / rb.get('path', 'source-plate.png')).resolve()
    if not plate_in.is_file() or digest(plate_in) != rb.get('sha256'):
        return baseline / 'slideir.json', []

    work_plate = np.array(Image.open(plate_in).convert('RGB'))
    h_img, w_img = work_plate.shape[:2]

    # Choose an installed approved CJK/Latin font for chart axes/labels
    ea_font = '微软雅黑'
    latin_font = 'Arial'
    if graph.text.styles:
        ea_font = graph.text.styles[0].ea
        latin_font = graph.text.styles[0].latin
    elif fonts and fonts.get('fonts'):
        cjk = [f['family'] for f in fonts['fonts'] if f.get('cjk')]
        if cjk:
            ea_font = cjk[0]

    # Collect existing owner masks so we never double-erase or collide illegally
    already_claimed = np.zeros((h_img, w_img), bool)
    for rec in graph.selected.values():
        ref = rec.get('owner_evidence')
        if ref and (baseline / ref['file']).is_file():
            with np.load(baseline / ref['file'], allow_pickle=False) as d:
                already_claimed |= d['mask'].astype(bool)

    staged_charts: list[dict] = []
    ch_seq = 1
    used_ids = set(graph.selected) | {x.id for x in [*graph.text.texts, *graph.asset.objects, *graph.surface.shapes]}

    for reg in chart_regions:
        rx, ry, rw, rh = [int(round(v)) for v in reg['bbox']]
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(w_img - rx, rw), min(h_img - ry, rh)
        if rw < 140 or rh < 90:
            continue

        # Find OCR / scene text items inside or immediately below/left of the chart bbox
        expanded = [max(0, rx - 24), max(0, ry - 16), min(w_img, rw + 48), min(h_img, rh + 36)]
        inside_texts = [
            t for t in graph.text.texts
            if intersection(t.bbox, expanded) >= 0.5 * max(1.0, t.bbox[2] * t.bbox[3])
        ]
        # Separate bottom X-axis category labels vs Y-axis/bar numeric labels
        cat_texts = [
            t for t in inside_texts
            if t.bbox[1] >= ry + 0.68 * rh and _parse_num(t.text) is None and len(t.text.strip()) <= 16
        ]
        num_texts = [
            t for t in inside_texts
            if _parse_num(t.text) is not None
        ]
        # Require genuine axis/data numbers OR >=3 aligned X-axis category labels WITH numeric ticks/values!
        # (Prevents turning icon+label rows below an unlabeled decorative graphic into a chart.)
        if len(cat_texts) < 2 or len(num_texts) < 2:
            continue

        cat_texts.sort(key=lambda t: t.bbox[0])
        # Sample background from border of chart box
        crop = work_plate[ry:ry + rh, rx:rx + rw].astype(np.float32)
        border_px = np.concatenate([crop[:4, :].reshape(-1, 3), crop[-4:, :].reshape(-1, 3)], axis=0)
        bg_med = np.median(border_px, axis=0)
        if float(np.percentile(np.abs(border_px - bg_med).max(1), 80)) > 8.0:
            continue

        # Detect vertical bars aligned with category centers
        fg = np.abs(crop - bg_med).max(2) > 12.0
        n_c, component_labels, stats_c, _ = cv2.connectedComponentsWithStats(fg.astype('uint8'), 8)
        bars = []
        for cid in range(1, n_c):
            bx, by, bw, bh, area = [int(v) for v in stats_c[cid]]
            if bw >= 10 and bh >= 14 and area >= 100 and (area / float(bw * bh)) >= 0.78:
                bars.append([rx + bx, ry + by, bw, bh])
        if len(bars) < len(cat_texts):
            continue

        categories: list[str] = []
        values: list[float] = []
        matched_bars: list[list[int]] = []
        label_ids = set()
        chart_mask = np.zeros(fg.shape, bool)

        for ct in cat_texts:
            cx_mid = ct.bbox[0] + 0.5 * ct.bbox[2]
            best_b = min(bars, key=lambda b: abs((b[0] + 0.5 * b[2]) - cx_mid))
            if abs((best_b[0] + 0.5 * best_b[2]) - cx_mid) > max(32.0, ct.bbox[2]):
                continue
            # Check if a numeric data label sits directly above this bar
            top_labels = [nt for nt in num_texts
                          if abs((nt.bbox[0] + nt.bbox[2]/2) - (best_b[0] + best_b[2]/2)) <= best_b[2]/2
                          and -nt.bbox[3] <= best_b[1] - (nt.bbox[1]+nt.bbox[3]) <= nt.bbox[3]]
            if len(top_labels) != 1 or best_b in matched_bars or top_labels[0].id in label_ids:
                break
            value_label = top_labels[0]
            label_ids.update((ct.id, value_label.id))
            val = _parse_num(value_label.text)
            bx, by, bw, bh = best_b
            component = component_labels[by-ry+bh//2, bx-rx+bw//2]
            if component == 0:
                break
            chart_mask |= component_labels == component
            categories.append(ct.text.replace('\n', ' ').strip())
            values.append(val)
            matched_bars.append(best_b)

        if len(categories) != len(cat_texts) or len(matched_bars) != len(bars):
            continue

        while f'ch-{ch_seq:02d}' in used_ids:
            ch_seq += 1
        cid_str = f'ch-{ch_seq:02d}'
        ch_seq += 1
        if cid_str in skip:
            continue

        full_mask = np.zeros((h_img, w_img), bool)
        from ..parallel_fusion import glyph_evidence
        styles = {style.id:style for style in graph.text.styles}
        if {t.id for t in inside_texts} != label_ids:
            continue
        pending_labels = set(label_ids)
        for t in inside_texts:
            if t.id not in label_ids or graph.ownership.get(t.id) != 'RESOLVED':
                break
            x, y, w, h = map(round, t.bbox)
            if x < rx or y < ry or x+w > rx+rw or y+h > ry+rh:
                break
            backing = np.broadcast_to(bg_med, (h,w,3))
            ink, uncertain = glyph_evidence(work_plate, [x,y,w,h], styles[t.style].color, backing)
            if uncertain.any():
                break
            chart_mask[y-ry:y-ry+h, x-rx:x-rx+w] |= ink
            pending_labels.remove(t.id)
        if pending_labels:
            continue
        # Only connected bar pixels and source-authorized label strokes are replaced. Preserve legends, arrows and other unowned ink.
        if already_claimed[ry:ry+rh, rx:rx+rw].any() or (fg & ~chart_mask).any():
            continue
        if any(t.id in graph.selected for t in inside_texts):
            continue
        full_mask[ry:ry + rh, rx:rx + rw] = chart_mask
        if int(full_mask.sum()) < 60:
            continue

        staged_charts.append({
            'id': cid_str,
            'bbox': [float(rx), float(ry), float(rw), float(rh)],
            'restore_bbox': [rx, ry, rw, rh],
            'bg_uint8': np.clip(np.rint(bg_med), 0, 255).astype(np.uint8),
            'mask': full_mask,
            'covered_text_ids': [t.id for t in inside_texts],
            'payload': {
                'chart_type': 'bar',
                'font': {'ea': ea_font, 'latin': latin_font, 'size_pt': 9.0},
                'categories': categories,
                'series': [{'name': '数据', 'values': values}],
                'has_legend': False,
            },
        })
        already_claimed |= full_mask
        used_ids.add(cid_str)

    if not staged_charts:
        return baseline / 'slideir.json', []

    shutil.copytree(baseline, output)
    new_ids: list[str] = []
    for ch in staged_charts:
        rx, ry, rw, rh = ch['restore_bbox']
        bg_u8 = ch['bg_uint8']
        local_m = ch['mask'][ry:ry + rh, rx:rx + rw]
        work_plate[ry:ry + rh, rx:rx + rw][local_m] = bg_u8
        proof_file = output / f"{ch['id']}-owner.npz"
        backing_patch = np.broadcast_to(bg_u8, (rh, rw, 3)).copy()
        np.savez_compressed(proof_file, mask=ch['mask'], restore_bbox=[rx, ry, rw, rh], backing_patch=backing_patch)
        record = {
            'id': ch['id'],
            'kind': 'chart',
            'bbox': ch['bbox'],
            'restore_bbox': [rx, ry, rw, rh],
            'payload': ch['payload'],
            'method': 'layout-bar-chart-v2', 'covered_text_ids': ch['covered_text_ids'],
            'state': 'STAGED',
            'quality': 'REVIEW_REQUIRED',
            'human_acceptance': False,
            'source_sha256': graph.source_sha256,
            'original_owner': 'source-plate',
            'complete_owner_scope': True,
            'mask_sha256': array_sha(ch['mask'].astype('uint8')),
            'owner_evidence': {
                'file': proof_file.name,
                'sha256': digest(proof_file),
                'method': 'chart-border-sampled-backing/1',
            },
        }
        graph.transactions.append(record)
        graph.selected[ch['id']] = record
        new_ids.append(ch['id'])

    plate_rgba = np.dstack([work_plate, np.full((h_img, w_img), 255, dtype=np.uint8)])
    Image.fromarray(plate_rgba).save(output / 'source-plate.png')
    graph.raster_backing = {
        'path': 'source-plate.png',
        'sha256': digest(output / 'source-plate.png'),
        'source_sha256': graph.source_sha256,
        'bbox': [0, 0, *graph.image_size],
    }
    ir = adapt(graph, output)
    write(output / 'scene.json', graph.model_dump())
    write(output / 'slideir.json', ir)
    return output / 'slideir.json', new_ids
