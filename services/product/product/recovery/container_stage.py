"""Hierarchical recovery of cards, ribbons, badges, gradient bars, and structural dividers.

Runs after foreground texts and independent icons/illustrations have been staged
onto source-plate.png. Geometric surfaces (solid or linear-gradient `rect`,
`roundRect`, `ellipse`, `pentagon`, `chevron`) become native PowerPoint shapes
(`SceneGraph.surface.shapes` -> SlideIR `shape` nodes); chamfered frames, curves,
and non-parametric dividers become clean movable transparent PNG layers
(`SceneGraph.asset.objects` -> SlideIR `image` nodes) ordered behind foreground
icons and native text.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..contracts import Shape, intersection
from ..materialize import digest, write
from ..object_contracts import ClosedObject
from ..object_extract import rectangle
from ..parallel_contracts import SceneGraph
from ..parallel_fusion import fuse_incremental_scene
from ..parallel_rebuild import adapt, array_sha
from .native_shapes import (
    _segment_max_err,
    _split_bar_series_on_baseline,
    _split_two_tone_ribbon,
    discover_bordered_cards,
    estimate_canvas_bg_flat,
    estimate_canvas_field,
    estimate_card_shadow,
    hex_rgb as _hex,
    soft_shadow_halo_mask,
    try_fit_native_shape_v2,
    try_fit_two_tone_ribbon_pair,
)


def _estimate_canvas_bg(plate: np.ndarray) -> tuple[np.ndarray, float]:
    return estimate_canvas_bg_flat(plate)


def _split_protruding_ticks(fg_support: np.ndarray, box: list[int], size: tuple[int, int]) -> list[tuple[str, list[int]]]:
    """Separate a thin 2-4px timeline connector tick touching the side of a tall pill bar."""
    w_img, h_img = size
    bx, by, bw, bh = box
    if bw < 80 or bh < 24:
        kind = 'line' if (bw >= 36 and bh <= 10) or (bh >= 36 and bw <= 10) else 'box'
        return [(kind, box)]
    sub = fg_support[by:by + bh, bx:bx + bw]
    col_counts = sub.sum(axis=0)
    max_probe = min(28, bw // 4)
    thin_limit = max(6, int(0.32 * bh)) if (bw >= 130 and bh >= 52) else 6
    body_limit = max(18, int(0.58 * bh))

    results: list[tuple[str, list[int]]] = []
    left_trim = 0
    if col_counts[0] <= thin_limit:
        for cx in range(4, max_probe):
            if col_counts[cx] >= body_limit and np.all(col_counts[:cx - 1] <= thin_limit):
                left_trim = cx
                break
    right_trim = 0
    if col_counts[-1] <= thin_limit:
        for off in range(4, max_probe):
            cx = bw - 1 - off
            if col_counts[cx] >= body_limit and np.all(col_counts[cx + 2:] <= thin_limit):
                right_trim = off
                break

    if left_trim >= 4:
        tick_slice = sub[:, :max(1, left_trim - 1)]
        yy, xx = np.nonzero(tick_slice)
        if len(xx) > 0:
            ty0 = max(0, int(by + yy.min()) - 2)
            ty1 = min(h_img, max(ty0 + 6, int(by + yy.max()) + 3))
            tx0 = max(0, int(bx + xx.min()))
            tx1 = min(w_img, int(bx + left_trim - 1))
            if tx1 - tx0 >= 6:
                results.append(('tick', [tx0, ty0, tx1 - tx0, ty1 - ty0]))
    if right_trim >= 4:
        tick_slice = sub[:, bw - right_trim + 1:]
        yy, xx = np.nonzero(tick_slice)
        if len(xx) > 0:
            ty0 = max(0, int(by + yy.min()) - 2)
            ty1 = min(h_img, max(ty0 + 6, int(by + yy.max()) + 3))
            tx0 = max(0, int(bx + bw - right_trim + 1))
            tx1 = min(w_img, int(bx + bw - right_trim + 1 + xx.max()) + 1)
            if tx1 - tx0 >= 6:
                results.append(('tick', [tx0, ty0, tx1 - tx0, ty1 - ty0]))

    mx0 = bx + left_trim
    mw = bw - left_trim - right_trim
    if mw >= 48:
        main_sub = fg_support[by:by + bh, mx0:mx0 + mw]
        yy, xx = np.nonzero(main_sub)
        if len(xx) > 0:
            my0 = int(by + yy.min())
            mh = int(yy.max() - yy.min() + 1)
            mx_real = int(mx0 + xx.min())
            mw_real = int(xx.max() - xx.min() + 1)
            results.append(('box', [mx_real, my0, mw_real, mh]))
    return results or [('box', box)]


def _discover_candidates(
    work_plate: np.ndarray,
    canvas_field: np.ndarray,
    scenery_guard: np.ndarray,
    *,
    is_flat_canvas: bool = True,
    bar_series: list | None = None,
) -> list[tuple[str, list[int]]]:
    h_img, w_img = work_plate.shape[:2]
    diff = np.abs(work_plate.astype(np.float32) - canvas_field.astype(np.float32)).max(2)
    diff[scenery_guard] = 0.0
    core = diff > (9.5 if is_flat_canvas else 13.5)
    support = diff > (3.8 if is_flat_canvas else 7.5)

    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(support.astype('uint8'), 8)
    found: list[tuple[str, list[int]]] = []
    seen_boxes: list[list[int]] = []

    def _add(kind: str, b: list[int], *, allow_tighter_shadow_box: bool = False):
        x, y, w, h = b
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return
        if w * h > 0.68 * (w_img * h_img):
            return
        area_b = float(w * h)
        for sx, sy, sw, sh in seen_boxes:
            if abs(x - sx) <= 5 and abs(y - sy) <= 5 and abs(w - sw) <= 10 and abs(h - sh) <= 10:
                if not (allow_tighter_shadow_box and (sw - w >= 7 or sh - h >= 7)):
                    return
            ov = intersection(b, [sx, sy, sw, sh])
            if ov > 0 and ov / max(1.0, area_b + sw * sh - ov) >= 0.86:
                if not (allow_tighter_shadow_box and (sw - w >= 7 or sh - h >= 7)):
                    return
        seen_boxes.append(b)
        found.append((kind, b))

    def add_series(parts):
        for kind, bbox in parts:
            _add(kind, bbox)
        if bar_series is not None:
            bar_series.append([tuple(bbox) for kind, bbox in parts if kind == 'box'])

    def _scan_nested_children(parent_box: list[int]):
        sx, sy, sw, sh = parent_box
        if sw < 140 or sh < 24:
            return
        pad_in = 6 if sh >= 80 else 2
        if sh - 2 * pad_in >= 20 and sw - 2 * pad_in >= 40:
            inner_crop = work_plate[sy + pad_in:sy + sh - pad_in, sx + pad_in:sx + sw - pad_in]
            inner_med = np.median(inner_crop.reshape(-1, 3), axis=0)
            inner_field = np.broadcast_to(inner_med.astype(np.float32), inner_crop.shape)
            idiff = np.abs(inner_crop.astype(np.float32) - inner_field).max(2)
            large_parent = (sw >= 360 and sh >= 220)
            isup = idiff > (2.6 if large_parent else 4.2)
            icore = idiff > (5.0 if large_parent else 9.5)
            if large_parent:
                for _, sub_box in discover_bordered_cards(inner_crop, np.zeros(inner_crop.shape[:2], bool)):
                    cx, cy, cw, ch = sub_box
                    _add('box', [sx + pad_in + cx, sy + pad_in + cy, cw, ch])
            in_cnt, ilbl, istats, _ = cv2.connectedComponentsWithStats(isup.astype('uint8'), 8)
            for ic in range(1, in_cnt):
                ix, iy, iw, ih, iarea = [int(v) for v in istats[ic]]
                is_short_accent = (3 <= min(iw, ih) <= 14 and 14 <= max(iw, ih) <= 48 and max(iw, ih) >= 1.9 * min(iw, ih) and iarea >= 36)
                if not is_short_accent and (iarea < 80 or max(iw, ih) < 32):
                    continue
                if int(((ilbl == ic) & icore).sum()) < 16:
                    continue
                cbox = [sx + pad_in + ix, sy + pad_in + iy, iw, ih]
                # Check if this child is a baseline-connected bar series (e.g. column 01 bar chart)
                full_inner_field = np.broadcast_to(inner_med.astype(np.float32), work_plate.shape)
                bar_split = _split_bar_series_on_baseline(work_plate, full_inner_field, cbox)
                if bar_split is not None:
                    add_series(bar_split)
                    continue
                if is_short_accent or (min(iw, ih) <= 8 and max(iw, ih) >= 36):
                    _add('line', cbox)
                elif min(iw, ih) >= 18:
                    for r_kind, r_box in _split_two_tone_ribbon(work_plate, cbox):
                        _add(r_kind, r_box)
        else:
            inner_med = np.median(work_plate[sy:sy + sh, sx:sx + sw].reshape(-1, 3), axis=0)

        # High-contrast flush-corner children (e.g. `01 应用成效` pentagon ribbon at card top-left `(sx+1, sy+1)`)
        flush_crop = work_plate[sy + 2:sy + sh - 2, sx + 1:sx + sw - 1].astype(np.float32) if sh > 8 else work_plate[sy:sy + sh, sx:sx + sw].astype(np.float32)
        f_y0 = sy + 2 if sh > 8 else sy
        f_x0 = sx + 1 if sh > 8 else sx
        fdiff = (inner_med.astype(np.float32).mean() - flush_crop.mean(axis=2))
        fsup = (fdiff > 18.0) & (np.abs(flush_crop - inner_med.astype(np.float32)).max(2) > 22.0)
        fn_cnt, _, fstats, _ = cv2.connectedComponentsWithStats(fsup.astype('uint8'), 8)
        for fc in range(1, fn_cnt):
            fx, fy, fw, fh, farea = [int(v) for v in fstats[fc]]
            if farea >= 350 and fw >= 44 and fh >= 20 and not (fw >= sw - 6 and fh >= sh - 6):
                for r_kind, r_box in _split_two_tone_ribbon(work_plate, [f_x0 + fx, f_y0 + fy, fw, fh]):
                    _add(r_kind, r_box)

        # Also check header band strip at top of card (`y = sy .. sy + min(90, sh//3)`)
        max_hdr = min(92, sh // 2)
        if max_hdr >= 28:
            row_meds = np.median(work_plate[sy:sy + max_hdr, sx + 8:sx + sw - 8].astype(np.float32), axis=1)
            row_step = np.abs(row_meds[2:] - row_meds[:-2]).max(axis=1)
            if len(row_step) > 18:
                s_slice = row_step[16:]
                if len(s_slice) > 0 and float(s_slice.max()) >= 5.0:
                    hdr_h = int(s_slice.argmax()) + 17
                    hdr_med = np.median(row_meds[4:max(6, hdr_h - 3)], axis=0)
                    while (
                        hdr_h < max_hdr
                        and float(np.abs(row_meds[hdr_h] - hdr_med).max())
                        < float(np.abs(row_meds[hdr_h] - inner_med).max())
                    ):
                        hdr_h += 1
                    # Accept header band when right edge matches hdr_med even if a dark-blue ribbon covers left edge,
                    # OR when vertical row-to-row profiles inside a horizontal gradient header are smooth.
                    l_strip = np.median(work_plate[sy + 4:sy + max(6, hdr_h - 3), sx + 2:min(sx + sw, sx + 10)].astype(np.float32), axis=(0, 1))
                    r_strip = np.median(work_plate[sy + 4:sy + max(6, hdr_h - 3), max(sx, sx + sw - 10):sx + sw - 2].astype(np.float32), axis=(0, 1))
                    edges_match = (
                        float(np.abs(r_strip - hdr_med).max()) <= 8.0
                        and float(np.abs(r_strip - inner_med).max()) >= 4.0
                    ) or (
                        float(np.abs(l_strip - hdr_med).max()) <= 8.0
                        and not (
                            float(np.abs(l_strip - inner_med).max()) <= 4.0
                            and float(np.abs(r_strip - inner_med).max()) <= 4.0
                        )
                    )
                    hdr_band_rows = row_meds[4:max(6, hdr_h - 3)]
                    smooth_hdr_vert = (len(hdr_band_rows) < 2) or (float(np.median(np.abs(np.diff(hdr_band_rows, axis=0)))) <= 3.5)
                    horiz_grad_match = (
                        smooth_hdr_vert
                        and float(np.abs(l_strip - inner_med).max()) >= 10.0
                        and float(np.abs(r_strip - inner_med).max()) >= 10.0
                    )
                    if (edges_match or horiz_grad_match) and float(np.abs(hdr_med - inner_med).max()) >= 5.0 and 24 <= hdr_h <= 88:
                        _add('box', [sx, sy, sw, hdr_h])
                        # Check flush ribbon or compact accent divider inside header band (exclude 2px outer border)
                        hy_off = 2 if hdr_h > 10 else 0
                        hdr_crop = work_plate[sy + hy_off:sy + hdr_h - hy_off, sx + 1:sx + sw - 1].astype(np.float32)
                        if hdr_crop.shape[0] >= 12 and hdr_crop.shape[1] >= 24:
                            tb_rows = np.concatenate([hdr_crop[1:4], hdr_crop[-4:-1]], axis=0)
                            raw_x_bg = np.median(tb_rows, axis=0, keepdims=True).astype(np.float32)
                            hdr_x_bg = cv2.blur(raw_x_bg, (21, 1))
                        else:
                            hdr_x_bg = hdr_med.reshape(1, 1, 3).astype(np.float32)
                        r_sup = (
                            (((hdr_med.mean() - hdr_crop.mean(axis=2)) > 18.0) & (np.abs(hdr_crop - hdr_med).max(2) > 22.0))
                            | (np.abs(hdr_crop - hdr_x_bg).max(2) > 20.0)
                        )
                        rn, _, rstats, _ = cv2.connectedComponentsWithStats(r_sup.astype('uint8'), 8)
                        for rcid in range(1, rn):
                            rx_l, ry_l, rw_l, rh_l, rarea = [int(v) for v in rstats[rcid]]
                            is_hdr_accent = (
                                3 <= min(rw_l, rh_l) <= 14
                                and 14 <= max(rw_l, rh_l) <= 96
                                and max(rw_l, rh_l) >= 1.9 * min(rw_l, rh_l)
                                and rarea >= 36
                            )
                            if is_hdr_accent:
                                _add('line', [sx + 1 + rx_l, sy + hy_off + ry_l, rw_l, rh_l])
                            elif rarea >= 400 and rw_l >= 60 and rh_l >= 20:
                                for r_kind, r_box in _split_two_tone_ribbon(
                                    work_plate, [sx + 1 + rx_l, sy + hy_off + ry_l, rw_l, rh_l]
                                ):
                                    _add(r_kind, r_box)

    for cid in range(1, n_lbl):
        bx, by, bw, bh, area = [int(v) for v in stats[cid]]
        is_short_accent = (3 <= min(bw, bh) <= 14 and 14 <= max(bw, bh) <= 48 and max(bw, bh) >= 1.9 * min(bw, bh) and area >= 36)
        if not is_short_accent and (area < 72 or max(bw, bh) < 32):
            continue
        comp_mask = labels == cid
        if int((comp_mask & core).sum()) < 16:
            continue
        if is_short_accent:
            _add('line', [bx, by, bw, bh])
            continue
        bar_split = _split_bar_series_on_baseline(work_plate, canvas_field, [bx, by, bw, bh])
        if bar_split is not None:
            add_series(bar_split)
            continue
        for kind, split_box in _split_protruding_ticks(comp_mask, [bx, by, bw, bh], (w_img, h_img)):
            if kind == 'box':
                for r_kind, r_box in _split_two_tone_ribbon(work_plate, split_box):
                    _add(r_kind, r_box)
            else:
                _add(kind, split_box)
            _scan_nested_children(split_box)

    # Discover border-locked cards after diff>3.8 CCs, allowing tighter frames to replace >=7px shadow halos.
    for kind, card_box in discover_bordered_cards(work_plate, scenery_guard):
        _add(kind, card_box, allow_tighter_shadow_box=True)
        _scan_nested_children(card_box)

    cleaned: list[tuple[str, list[int]]] = []
    for kind, b in found:
        x, y, w, h = b
        if kind == 'box' and (w < 24 or h < 18):
            continue
        if kind in ('line', 'tick') and max(w, h) < 8:
            continue
        cleaned.append((kind, b))

    # Innermost / smallest area first so children peel off cleanly onto their parent
    # fill BEFORE the enclosing parent card is evaluated!
    cleaned.sort(key=lambda item: (item[1][2] * item[1][3], item[1][1], item[1][0]))
    return cleaned


def _ring_parent_bg(
    work_plate: np.ndarray,
    box: list[int],
    pad: int = 5,
) -> tuple[np.ndarray, float] | None:
    """Estimate flat parent backing median RGB and 80th-percentile residual from exterior ring."""
    h_img, w_img = work_plate.shape[:2]
    x, y, w, h = [int(v) for v in box]
    rx0, ry0 = max(0, x - pad), max(0, y - pad)
    rx1, ry1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
    ring_mask = np.ones((ry1 - ry0, rx1 - rx0), bool)
    ring_mask[y - ry0:y - ry0 + h, x - rx0:x - rx0 + w] = False
    vals = work_plate[ry0:ry1, rx0:rx1][ring_mask]
    if len(vals) < 16:
        return None
    med0 = np.median(vals, axis=0).astype(np.float32)
    inliers = vals[np.abs(vals.astype(np.float32) - med0).max(1) <= 12.0]
    if len(inliers) < max(12, int(0.45 * len(vals))):
        return None
    parent_bg = np.median(inliers, axis=0).astype(np.float32)
    err = float(np.percentile(np.abs(inliers.astype(np.float32) - parent_bg).max(1), 80))
    return parent_bg, err


def _ring_parent_patch(
    work_plate: np.ndarray,
    canvas_field: np.ndarray,
    box: list[int],
    pad: int = 5,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Estimate smooth parent backing patch `(h, w, 3)` and median RGB from exterior ring."""
    h_img, w_img = work_plate.shape[:2]
    x, y, w, h = box
    rx0, ry0 = max(0, x - pad), max(0, y - pad)
    rx1, ry1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
    ring_mask = np.ones((ry1 - ry0, rx1 - rx0), bool)
    ring_mask[y - ry0:y - ry0 + h, x - rx0:x - rx0 + w] = False
    vals = work_plate[ry0:ry1, rx0:rx1][ring_mask]
    if len(vals) < 16:
        return None

    # Flush-left ribbon inside a card header band: right edge sits inside the clean header fill (distinct from canvas sky).
    # Offset +2px past the right tip to clear 1px anti-aliasing halo on single-tone chevron/homePlate tips.
    if 20 <= h <= 96 and 110 <= w <= 360 and x + w + 2 + pad <= w_img:
        r_in = work_plate[y + 3:max(y + 5, y + h - 3), x + w + 2:x + w + 2 + pad].reshape(-1, 3)
        if len(r_in) >= 12:
            rm_in = np.median(r_in.astype(np.float32), axis=0)
            r_err = float(np.percentile(np.abs(r_in.astype(np.float32) - rm_in).max(1), 80))
            cf_med = np.median(canvas_field[y:y + h, x:x + w].reshape(-1, 3), axis=0)
            box_med = np.median(work_plate[y:y + h, x:x + w].reshape(-1, 3).astype(np.float32), axis=0)
            if (r_err <= 4.5
                    and float(np.abs(rm_in - cf_med).max()) >= 10.0
                    and float(np.abs(box_med - rm_in).max()) >= 22.0):
                return np.broadcast_to(rm_in, (h, w, 3)).copy(), rm_in, r_err

    def _try_strip_or_field() -> tuple[np.ndarray, np.ndarray, float] | None:
        # Flush-left ribbon or cap sitting inside a header band (right edge borders the clean header band)
        if 20 <= h <= 96 and w >= 32 and x + w + 2 + pad <= w_img:
            ry_in0, ry_in1 = y + 3, max(y + 5, y + h - 3)
            r_in = work_plate[ry_in0:ry_in1, x + w + 2:x + w + 2 + pad].reshape(-1, 3)
            if len(r_in) >= 12:
                rm_in = np.median(r_in.astype(np.float32), axis=0)
                r_err = float(np.percentile(np.abs(r_in.astype(np.float32) - rm_in).max(1), 80))
                box_med = np.median(work_plate[y:y + h, x:x + w].reshape(-1, 3).astype(np.float32), axis=0)
                if r_err <= 4.5 and float(np.abs(box_med - rm_in).max()) >= 12.0:
                    return np.broadcast_to(rm_in, (h, w, 3)).copy(), rm_in, r_err
        # Full-width top header band sitting directly above clean card body
        if 24 <= h <= 92 and w >= 160 and y + h + pad <= h_img:
            bx_in0, bx_in1 = x + 8, max(x + 12, x + w - 8)
            b_in = work_plate[y + h:y + h + pad, bx_in0:bx_in1].reshape(-1, 3)
            if len(b_in) >= 32:
                bm_in = np.median(b_in.astype(np.float32), axis=0)
                b_err = float(np.percentile(np.abs(b_in.astype(np.float32) - bm_in).max(1), 80))
                box_med = np.median(work_plate[y:y + h, x:x + w].reshape(-1, 3).astype(np.float32), axis=0)
                if b_err <= 4.5 and float(np.abs(box_med - bm_in).max()) >= 4.5:
                    return np.broadcast_to(bm_in, (h, w, 3)).copy(), bm_in, b_err
        # Check flush-corner child inside a card: right + bottom strips agree on parent card fill
        r_strip = work_plate[y:y + h, min(w_img, x + w):min(w_img, x + w + pad)].reshape(-1, 3)
        b_strip = work_plate[min(h_img, y + h):min(h_img, y + h + pad), x:x + w].reshape(-1, 3)
        if len(r_strip) >= 8 and len(b_strip) >= 8:
            rm = np.median(r_strip.astype(np.float32), axis=0)
            bm = np.median(b_strip.astype(np.float32), axis=0)
            if float(np.abs(rm - bm).max()) <= 6.0:
                comb = np.concatenate([r_strip, b_strip], axis=0).astype(np.float32)
                c_med = np.median(comb, axis=0)
                c_err = float(np.percentile(np.abs(comb - c_med).max(1), 80))
                if c_err <= 6.5:
                    return np.broadcast_to(c_med, (h, w, 3)).copy(), c_med, c_err
        # Check if exterior ring matches canvas_field (e.g. along sky gradient or partial shadow)
        f_ring = canvas_field[ry0:ry1, rx0:rx1][ring_mask]
        f_diff = np.abs(vals.astype(np.float32) - f_ring).max(1)
        f_inliers = f_diff <= 14.0
        if int(f_inliers.sum()) >= max(12, int(0.45 * len(vals))):
            patch = canvas_field[y:y + h, x:x + w].copy()
            return patch, np.median(patch.reshape(-1, 3), axis=0), float(np.percentile(f_diff[f_inliers], 80))
        return None

    med0 = np.median(vals, axis=0).astype(np.float32)
    inliers = vals[np.abs(vals.astype(np.float32) - med0).max(1) <= 12.0]
    if len(inliers) < max(12, int(0.45 * len(vals))):
        return _try_strip_or_field()
    parent_bg = np.median(inliers, axis=0).astype(np.float32)
    err = float(np.percentile(np.abs(inliers.astype(np.float32) - parent_bg).max(1), 80))
    if err > 6.5:
        fb = _try_strip_or_field()
        if fb is not None:
            return fb
    # If parent_bg is very close to canvas_field, use smooth canvas_field patch so gradients stay seamless
    field_patch = canvas_field[y:y + h, x:x + w]
    if float(np.abs(np.median(field_patch.reshape(-1, 3), axis=0) - parent_bg).max()) <= 5.5:
        return field_patch.copy(), parent_bg, min(err, 4.5)
    return np.broadcast_to(parent_bg, (h, w, 3)).copy(), parent_bg, err


def _clamped_rect_mask(bbox, size: tuple[int, int], pad: int = 0) -> np.ndarray:
    w_img, h_img = size
    bx, by, bw, bh = [float(v) for v in bbox]
    x0 = max(0, int(math.floor(bx)) - pad)
    y0 = max(0, int(math.floor(by)) - pad)
    x1 = min(w_img, int(math.ceil(bx + bw)) + pad)
    y1 = min(h_img, int(math.ceil(by + bh)) + pad)
    mask = np.zeros((h_img, w_img), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def _opaque_tile_interior(crop, parent, protected):
    """Recover light lettering enclosed by a source-visible, solid coloured tile."""
    h, w = crop.shape[:2]
    if min(h, w) < 16 or max(h, w) > 2 * min(h, w):
        return None
    rim = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
    low, median, high = np.percentile(rim, [1, 50, 99], axis=0)
    if (median < 225).any() or (high - low).max() > 8 or abs(median - parent).max() > 3:
        return None
    strong = np.abs(crop.astype(float) - median).max(2) > 35
    contours, _ = cv2.findContours(strong.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return None
    contour = contours[0]
    x, y, cw, ch = cv2.boundingRect(contour)
    if min(x, y, w-x-cw, h-y-ch) < 1 or min(cw, ch) < 0.8 * max(cw, ch):
        return None
    tile = np.zeros((h, w), np.uint8)
    cv2.drawContours(tile, [contour], -1, 1, -1)
    if tile.sum() < 0.9 * cw * ch or strong.sum() < 0.6 * tile.sum():
        return None
    if protected[tile.astype(bool)].any():
        return None
    interior = cv2.erode(tile, np.ones((3, 3), np.uint8)).astype(bool)
    # Each enclosed pale mark must measurably differ from the exterior.
    # Ambiguous cutouts and transparent counters retain the normal matte.
    marks = interior & ~strong
    count, labels, stats, _ = cv2.connectedComponentsWithStats(marks.astype('uint8'), 8)
    if count <= 1:
        return None
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 6:
            return None
        samples = crop[labels == label].astype(float)
        distinct = ((samples >= high + 1) & (samples >= median + 3)).any(1)
        if distinct.mean() < .15:
            return None
    return interior


def attach_bar_groups(selected, series):
    """Only group complete, detected series whose individual native bars were staged."""
    by_box = {tuple(obj['bbox']): oid for oid, obj in selected.items() if obj.get('kind') == 'surface'
              and obj.get('payload', {}).get('geometry') == 'rect'}
    for boxes in series:
        if len(boxes) >= 3 and all(b in by_box for b in boxes):
            members = [by_box[b] for b in boxes]
            selected[members[0]]['move_members'] = members


def attach_card_groups(
    selected: dict,
    image_size: tuple[int, int],
    unresolved_guard: np.ndarray | None = None,
) -> None:
    """Safely group eligible leaf cards with strictly contained committed children into `<p:grpSp>`."""
    w_img, h_img = image_size
    page_area = float(max(1, w_img * h_img))

    def _obj_bbox(rec: dict) -> list[float] | None:
        box = rec.get('bbox') or rec.get('restore_bbox') or (rec.get('payload') or {}).get('bbox')
        if isinstance(box, (list, tuple)) and len(box) == 4 and box[2] > 0 and box[3] > 0:
            return [float(v) for v in box]
        return None

    # Refresh non-bar card groups on post-crop re-runs while preserving bar-series / manual groups.
    for oid, obj in list(selected.items()):
        if obj.get('card_move_group'):
            obj.pop('move_members', None)
            obj.pop('card_move_group', None)

    already_grouped: set[str] = {
        mid for obj in selected.values() for mid in (obj.get('move_members') or [])
    }

    cand_cards: list[tuple[float, str, list[float]]] = []
    surface_cards: list[tuple[str, list[float]]] = []
    for oid, obj in selected.items():
        if obj.get('kind') != 'surface':
            continue
        bbox = _obj_bbox(obj)
        if bbox is None:
            continue
        geom = (obj.get('payload') or {}).get('geometry')
        if geom not in ('roundRect', 'rect', 'chevron', 'pentagon', 'custom'):
            continue
        _, _, bw, bh = bbox
        if bw >= 80.0 and bh >= 28.0:
            surface_cards.append((oid, bbox))
        if oid in already_grouped:
            continue
        area = bw * bh
        if bw >= 84.0 and bh >= 32.0 and min(bw, bh) >= 28.0 and 0.005 * page_area <= area <= 0.24 * page_area:
            cand_cards.append((area, oid, bbox))

    def _is_multi_card_wrapper(oid: str, sb: list[float]) -> bool:
        sw, sh = sb[2], sb[3]
        inner = []
        for sub_id, sub_b in surface_cards:
            if sub_id == oid:
                continue
            sub_area = sub_b[2] * sub_b[3]
            if intersection(sub_b, sb) >= 0.85 * sub_area:
                inner.append(sub_b)
        if not inner:
            return False
        if len(inner) == 1:
            ib = inner[0]
            return not (ib[2] >= 0.80 * sw and ib[3] <= 0.52 * sh)
        if len(inner) == 2:
            b1, b2 = inner[0], inner[1]
            same_band = abs(b1[1] - b2[1]) <= 6.0 and abs(b1[3] - b2[3]) <= 6.0
            combined_w = b1[2] + b2[2] <= 1.04 * sw
            max_h = max(b1[3], b2[3]) <= 0.52 * sh
            return not (same_band and combined_w and max_h)
        return True

    # Process larger eligible leaf cards before any inner header band so 2-zone cards stay single groups.
    cand_cards.sort(key=lambda item: (-item[0], item[2][1], item[2][0]))
    for s_area, oid, sb in cand_cards:
        if oid in already_grouped:
            continue
        if _is_multi_card_wrapper(oid, sb):
            continue
        if unresolved_guard is not None and (_clamped_rect_mask(sb, image_size, pad=0) & unresolved_guard).any():
            continue

        contained: list[str] = []
        has_crosser = False
        for other_id, other_obj in selected.items():
            if other_id == oid:
                continue
            ob = _obj_bbox(other_obj)
            if ob is None:
                continue
            o_area = ob[2] * ob[3]
            ov = intersection(ob, sb)
            if ov <= 0.0:
                continue
            if ov >= 0.80 * o_area:
                if other_id in already_grouped:
                    has_crosser = True
                    break
                contained.append(other_id)
            elif o_area < 1.15 * s_area and ov > 0.04 * min(o_area, s_area):
                has_crosser = True
                break

        if has_crosser or not contained:
            continue
        if not any(selected[cid].get('kind') in ('text', 'asset') for cid in contained):
            continue
        members = [oid, *contained]
        selected[oid]['move_members'] = members
        selected[oid]['card_move_group'] = True
        already_grouped.update(members)


def stage_containers_and_surfaces(
    baseline_directory: Path | str,
    output_directory: Path | str,
    *,
    skip_ids: set[str] | None = None,
    logo_ids: set[str] | None = None,
    wanted_ids: set[str] | None = None,
    retained_frames=(),
    recover_light_tile_marks: bool = False,
) -> tuple[Path, list[str]]:
    """Discover and stage native shapes (`s-*`) and structural frame/curve/line assets (`a-c*`)."""
    from reconstruction import extract_rgba

    skip = skip_ids or set()
    logos = logo_ids or set()
    baseline = Path(baseline_directory).resolve()
    output = Path(output_directory).resolve()
    graph = SceneGraph.model_validate_json((baseline / 'scene.json').read_text('utf8'))
    if digest(baseline / 'source.png') != graph.source_sha256:
        raise ValueError('container stage source binding mismatch')
    rb = graph.raster_backing or {}
    plate_in = (baseline / rb.get('path', 'source-plate.png')).resolve()
    if not plate_in.is_file() or digest(plate_in) != rb.get('sha256'):
        raise ValueError('container stage source-plate binding mismatch')

    work_plate = np.array(Image.open(plate_in).convert('RGB'))
    h_img, w_img = work_plate.shape[:2]
    size = (w_img, h_img)

    # Load existing owner masks and build guard for unresolved/unstaged text & brand logos
    existing_proofs: dict[str, dict] = {}
    already_claimed = np.zeros((h_img, w_img), bool)
    higher_asset_vis = np.zeros((h_img, w_img), bool)
    higher_crop_vis = np.zeros((h_img, w_img), bool)
    for oid, rec in graph.selected.items():
        ref = rec.get('owner_evidence')
        if ref:
            pfile = (baseline / ref['file']).resolve()
            if digest(pfile) != ref['sha256']:
                raise ValueError('existing owner proof hash mismatch')
            with np.load(pfile, allow_pickle=False) as d:
                m = d['mask'].astype(bool)
                existing_proofs[oid] = {
                    'mask': m,
                    'restore_bbox': [int(v) for v in d['restore_bbox']],
                    'backing_patch': d['backing_patch'].astype(float),
                }
                already_claimed |= m
        if rec.get('kind') == 'asset' and rec.get('asset_reference'):
            aref = rec['asset_reference']
            ax, ay, aw, ah = [round(v) for v in aref['bbox']]
            apng = np.array(Image.open(baseline / aref['path']).convert('RGBA'))
            if rec.get('region_crop'):
                higher_crop_vis[ay:ay + ah, ax:ax + aw] |= apng[..., 3] > 8
            else:
                higher_asset_vis[ay:ay + ah, ax:ax + aw] |= apng[..., 3] > 8
        elif rec.get('foreground_shape') and oid in existing_proofs:
            higher_asset_vis |= existing_proofs[oid]['mask']

    # Text guard: NEVER erase or cover uncommitted text!
    # Exempt uncommitted/OCR items strictly enclosed inside an already-committed MOVABLE `region_crop`
    # (e.g., an embedded dashboard/UI mockup already extracted cleanly as an editable `image` object).
    movable_crop_boxes = [
        [float(v) for v in (rec.get('restore_bbox') or rec.get('bbox'))]
        for rec in graph.selected.values()
        if rec.get('region_crop') and rec.get('complete_owner_scope') and (rec.get('restore_bbox') or rec.get('bbox'))
    ]

    def _in_movable_crop(box) -> bool:
        area_b = max(1.0, float(box[2]) * float(box[3]))
        return any(intersection(box, cb) >= 0.85 * area_b for cb in movable_crop_boxes)

    # (Draft LayerD asset proposals are NOT blindly rect-blocked here so ribbons/badges/bars
    # can convert to native shapes; pixel-level anomaly checks in `try_fit_native_shape_v2`
    # protect any unmodeled icon/photo pixels.)
    unresolved_guard = np.zeros((h_img, w_img), bool)
    for frame in retained_frames:
        unresolved_guard |= _clamped_rect_mask(frame, size, pad=0)
    for t in graph.text.texts:
        if t.id not in graph.selected and not _in_movable_crop(t.bbox):
            unresolved_guard |= _clamped_rect_mask(t.bbox, size, pad=4)
    for a in graph.asset.objects:
        if a.id not in graph.selected and (a.id in logos or a.kind == 'logo' or a.text_policy == 'intrinsic_label'):
            unresolved_guard |= _clamped_rect_mask(a.bbox, size, pad=4)

    ocr_guard_file = baseline / 'runtime-ocr-protection.json'
    if ocr_guard_file.is_file():
        ocr_guard = json.loads(ocr_guard_file.read_text('utf8'))
        if ocr_guard.get('source_sha256') and ocr_guard['source_sha256'] != graph.source_sha256:
            raise ValueError('runtime OCR protection source binding mismatch')
        owners_by_ocr: dict[str, list[str]] = {}
        for tid, rec in graph.selected.items():
            if rec.get('kind') == 'text':
                for lid in graph.text.members.get(tid, []):
                    owners_by_ocr.setdefault(lid, []).append(tid)
        for line in ocr_guard.get('lines', []):
            lid = line.get('id')
            lbox = line.get('bbox')
            if not lbox or len(lbox) != 4:
                continue
            owners = owners_by_ocr.get(lid, [])
            if not owners:
                if not _in_movable_crop(lbox):
                    unresolved_guard |= _clamped_rect_mask(lbox, size, pad=3)
            else:
                owner_boxes = [
                    existing_proofs.get(tid, {}).get('restore_bbox')
                    or graph.selected[tid].get('restore_bbox')
                    or graph.selected[tid].get('bbox')
                    for tid in owners
                ]
                l_area = max(1.0, float(lbox[2]) * float(lbox[3]))
                covered = sum(intersection(lbox, rb_box) for rb_box in owner_boxes if rb_box)
                if covered < 0.85 * l_area:
                    line_mask = _clamped_rect_mask(lbox, size, pad=3)
                    owner_union = np.zeros((h_img, w_img), bool)
                    for rb_box in owner_boxes:
                        if rb_box:
                            owner_union |= _clamped_rect_mask(rb_box, size, pad=1)
                    unresolved_guard |= (line_mask & ~owner_union)

    canvas_field, canvas_hex, scenery_guard, is_flat_canvas = estimate_canvas_field(
        work_plate, excluded_mask=(already_claimed | unresolved_guard)
    )
    unresolved_guard |= scenery_guard

    bar_series = []
    candidates = _discover_candidates(
        work_plate, canvas_field, scenery_guard, is_flat_canvas=is_flat_canvas, bar_series=bar_series
    )
    for ps_cand in (baseline / 'page-structure.json', baseline.parent / 'page-structure.json'):
        if ps_cand.is_file():
            try:
                ps_data = json.loads(ps_cand.read_text('utf8'))
                existing_boxes = [list(b) for _, b in candidates]
                added_ps = False
                for el in ps_data.get('elements', []):
                    if el.get('category') not in ('card_shape', 'ribbon_shape', 'badge_shape'):
                        continue
                    rb = el.get('bbox')
                    if not rb or len(rb) != 4:
                        continue
                    ex = max(0, int(round(rb[0])))
                    ey = max(0, int(round(rb[1])))
                    ew = min(w_img - ex, int(round(rb[2])))
                    eh = min(h_img - ey, int(round(rb[3])))
                    if ew < 24 or eh < 18 or ew * eh > 0.68 * w_img * h_img:
                        continue
                    dup = any(
                        (abs(ex - sx) <= 5 and abs(ey - sy) <= 5 and abs(ew - sw) <= 10 and abs(eh - sh) <= 10)
                        or (intersection([ex, ey, ew, eh], [sx, sy, sw, sh]) / max(1.0, float(ew * eh + sw * sh) - intersection([ex, ey, ew, eh], [sx, sy, sw, sh])) >= 0.86)
                        for sx, sy, sw, sh in existing_boxes
                    )
                    if not dup:
                        cbox = [ex, ey, ew, eh]
                        candidates.append(('box', cbox))
                        existing_boxes.append(cbox)
                        added_ps = True
                if added_ps:
                    candidates.sort(key=lambda item: (item[1][2] * item[1][3], item[1][1], item[1][0]))
            except Exception:
                pass
            break
    if not candidates:
        return baseline / 'slideir.json', []

    shutil.copytree(baseline, output)
    graph.surface.background = canvas_hex

    backed_by_surface: dict[str, np.ndarray] = {}
    for oid, pinfo in existing_proofs.items():
        rx, ry, rw, rh = pinfo['restore_bbox']
        if graph.selected.get(oid, {}).get('kind') == 'text':
            backed_by_surface[oid] = (
                higher_asset_vis[ry:ry + rh, rx:rx + rw] | higher_crop_vis[ry:ry + rh, rx:rx + rw]
            ).copy()
        else:
            backed_by_surface[oid] = np.zeros((rh, rw), dtype=bool)

    def _refresh_overlying_backing_patches(
        surface_box: list[int],
        rendered_mask_local: np.ndarray,
        pred_fill_u8: np.ndarray,
        new_sids: set[str],
    ) -> None:
        """Keep overlying objects' owner_evidence backing_patch in sync with newly staged native surfaces."""
        sx, sy, sw, sh = surface_box
        s_area = sw * sh
        for oid, pinfo in existing_proofs.items():
            if oid in new_sids or oid not in graph.selected:
                continue
            rec = graph.selected[oid]
            ref = rec.get('owner_evidence')
            if not ref or not ref.get('file'):
                continue
            rx, ry, rw, rh = pinfo['restore_bbox']
            if rec.get('kind') == 'surface' and (rw * rh) >= s_area:
                continue
            ox0, oy0 = max(sx, rx), max(sy, ry)
            ox1, oy1 = min(sx + sw, rx + rw), min(sy + sh, ry + rh)
            if ox1 <= ox0 or oy1 <= oy0:
                continue
            sl_s = (slice(oy0 - sy, oy1 - sy), slice(ox0 - sx, ox1 - sx))
            sl_r = (slice(oy0 - ry, oy1 - ry), slice(ox0 - rx, ox1 - rx))
            sl_g = (slice(oy0, oy1), slice(ox0, ox1))
            already_backed = backed_by_surface.setdefault(oid, np.zeros((rh, rw), dtype=bool))
            cand_mask = pinfo['mask'][sl_g] & rendered_mask_local[sl_s] & (~already_backed[sl_r])
            if not cand_mask.any():
                continue
            old_bp = pinfo['backing_patch']
            color_diff = np.abs(pred_fill_u8[sl_s].astype(float) - old_bp[sl_r]).max(axis=2)
            upd = cand_mask & (color_diff <= 28.0)
            if not upd.any():
                continue
            new_bp_u8 = np.clip(np.rint(old_bp), 0, 255).astype(np.uint8)
            sub_bp = new_bp_u8[sl_r]
            sub_bp[upd] = pred_fill_u8[sl_s][upd]
            new_bp_u8[sl_r] = sub_bp
            already_backed[sl_r] |= upd
            out_proof = output / ref['file']
            np.savez_compressed(
                out_proof,
                mask=pinfo['mask'],
                restore_bbox=pinfo['restore_bbox'],
                backing_patch=new_bp_u8,
            )
            new_sha = digest(out_proof)
            ref['sha256'] = new_sha
            for tx in graph.transactions:
                if tx.get('id') == oid and isinstance(tx.get('owner_evidence'), dict) and tx['owner_evidence'].get('file') == ref['file']:
                    tx['owner_evidence']['sha256'] = new_sha
            pinfo['backing_patch'] = new_bp_u8.astype(float)

    used_ids = {x.id for x in [*graph.text.texts, *graph.asset.objects, *graph.surface.shapes]} | set(graph.selected)
    claimed_child_ids: set[str] = set()
    staged_new_ids: list[str] = []
    surface_claimed_mask = np.zeros((h_img, w_img), bool)
    container_asset_mask = np.zeros((h_img, w_img), bool)
    s_seq = sum(1 for k in used_ids if k.startswith('s-')) + 1
    c_seq = sum(1 for k in used_ids if k.startswith('a-c')) + 1

    for cand_kind, raw_box in candidates:
        x, y, w, h = [int(v) for v in raw_box]
        x = max(0, min(w_img - 2, x))
        y = max(0, min(h_img - 2, y))
        w = max(2, min(w_img - x, w))
        h = max(2, min(h_img - y, h))
        rect_area = rectangle([x, y, w, h], size)
        if (rect_area & unresolved_guard).any():
            continue

        ring_res = _ring_parent_patch(work_plate, canvas_field, [x, y, w, h], pad=5)
        if ring_res is None:
            continue
        parent_patch, parent_bg, parent_err = ring_res
        if parent_err > 7.5:
            continue
        crop_cur = work_plate[y:y + h, x:x + w]
        if float(np.abs(crop_cur.astype(float) - parent_patch).max()) < 3.5:
            continue

        # Collect direct staged children sitting inside this container
        direct_patches: list[np.ndarray] = []
        direct_oids: list[str] = []
        for oid, pinfo in existing_proofs.items():
            if oid in claimed_child_ids:
                continue
            rx, ry, rw, rh = pinfo['restore_bbox']
            if intersection([x, y, w, h], [rx, ry, rw, rh]) >= 0.5 * max(1, rw * rh):
                cmask = pinfo['mask'][ry:ry + rh, rx:rx + rw].copy()
                if graph.selected.get(oid, {}).get('kind') == 'text':
                    cmask &= ~higher_asset_vis[ry:ry + rh, rx:rx + rw]
                    cmask &= ~higher_crop_vis[ry:ry + rh, rx:rx + rw]
                if cmask.any():
                    direct_patches.append(pinfo['backing_patch'][cmask])
                direct_oids.append(oid)

        direct_children_mask_local = already_claimed[y:y + h, x:x + w].copy()
        ribbon_pair = (
            try_fit_two_tone_ribbon_pair(
                work_plate, [x, y, w, h], parent_patch, direct_children_mask_local
            )
            if cand_kind == 'box' else None
        )
        if ribbon_pair is not None:
            pair_shapes = []
            pair_sids = []
            for p_idx, p_fit in enumerate(ribbon_pair):
                while f's-{s_seq:03d}' in used_ids:
                    s_seq += 1
                p_sid = f's-{s_seq:03d}'
                s_seq += 1
                if p_sid in skip or (wanted_ids is not None and p_sid not in wanted_ids):
                    pair_shapes = []
                    break
                used_ids.add(p_sid)
                p_boff = p_fit.get('bbox_offset_local') or [0.0, 0.0, float(w), float(h)]
                pair_shapes.append(Shape(
                    id=p_sid,
                    bbox=[float(x + p_boff[0]), float(y + p_boff[1]), float(p_boff[2]), float(p_boff[3])],
                    z=1,
                    geometry=p_fit['geometry'],
                    fill=p_fit['fill'],
                    fill_gradient=p_fit.get('fill_gradient'),
                    line_color=p_fit['line_color'],
                    line_width_pt=float(p_fit['line_width_pt']),
                    corner_radius_px=p_fit['corner_radius_px'],
                    custom_paths=p_fit.get('custom_paths'),
                ))
                pair_sids.append(p_sid)
            if len(pair_shapes) == 2:
                full_erase = np.zeros((h_img, w_img), bool)
                full_erase_loc = ribbon_pair[0]['full_erase_mask_local']
                full_erase[y:y + h, x:x + w] = full_erase_loc
                s_mask = full_erase & ~already_claimed
                fused_try = fuse_incremental_scene(
                    baseline,
                    base_graph=graph,
                    added_shapes=pair_shapes,
                    background_hex=canvas_hex,
                )
                split_w = int(round(pair_shapes[0].bbox[2]))
                mask_left = s_mask.copy()
                mask_left[:, x + split_w:] = False
                mask_right = s_mask.copy()
                mask_right[:, :x + split_w] = False
                if (
                    0 < split_w < w
                    and int(mask_left.sum()) >= 16
                    and int(mask_right.sum()) >= 16
                    and all(fused_try.ownership.get(ps) == 'RESOLVED' for ps in pair_sids)
                ):
                    parent_u8 = np.clip(np.rint(parent_patch), 0, 255).astype(np.uint8)
                    work_plate[y:y + h, x:x + w][full_erase_loc] = parent_u8[full_erase_loc]
                    already_claimed |= s_mask
                    surface_claimed_mask |= s_mask
                    claimed_child_ids.update(direct_oids)
                    graph = fused_try
                    comb_pred_u8 = np.clip(
                        np.rint(np.concatenate([ribbon_pair[0]['pred_fill'], ribbon_pair[1]['pred_fill']], axis=1)),
                        0,
                        255,
                    ).astype(np.uint8)
                    _refresh_overlying_backing_patches([x, y, w, h], full_erase_loc, comb_pred_u8, set(pair_sids))
                    pair_proof_specs = [
                        (mask_left, [x, y, split_w, h], parent_u8[:, :split_w]),
                        (mask_right, [x + split_w, y, w - split_w, h], parent_u8[:, split_w:]),
                    ]
                    for p_sid, p_shp, (p_mask, p_rbox, p_back) in zip(pair_sids, pair_shapes, pair_proof_specs):
                        p_proof = output / f'{p_sid}-owner.npz'
                        np.savez_compressed(p_proof, mask=p_mask, restore_bbox=p_rbox, backing_patch=p_back)
                        existing_proofs[p_sid] = {
                            'mask': p_mask,
                            'restore_bbox': p_rbox,
                            'backing_patch': p_back.astype(float),
                        }
                        backed_by_surface[p_sid] = np.zeros((p_rbox[3], p_rbox[2]), dtype=bool)
                        record = {
                            'id': p_sid, 'kind': 'surface', 'bbox': list(p_shp.bbox), 'restore_bbox': p_rbox,
                            'payload': p_shp.model_dump(), 'method': 'container-surface-v2-two-tone-ribbon/1',
                            'state': 'STAGED', 'quality': 'REVIEW_REQUIRED', 'human_acceptance': False,
                            'source_sha256': graph.source_sha256, 'original_owner': 'source-plate',
                            'complete_owner_scope': True, 'mask_sha256': array_sha(p_mask.astype('uint8')),
                            'owner_evidence': {'file': p_proof.name, 'sha256': digest(p_proof), 'method': 'source-sampled-parent-backing/1'},
                        }
                        graph.transactions.append(record)
                        graph.selected[p_sid] = record
                        staged_new_ids.append(p_sid)
                    continue
        shape_fit = (
            try_fit_native_shape_v2(
                work_plate, [x, y, w, h], parent_patch, direct_patches,
                direct_children_mask_local, cand_kind=cand_kind,
            )
            if cand_kind in ('box', 'line') else None
        )

        if shape_fit is not None:
            while f's-{s_seq:03d}' in used_ids:
                s_seq += 1
            sid = f's-{s_seq:03d}'
            s_seq += 1
            if sid in skip or (wanted_ids is not None and sid not in wanted_ids):
                continue
            full_erase = np.zeros((h_img, w_img), bool)
            full_erase[y:y + h, x:x + w] = shape_fit['erase_mask_local']

            # Optional soft drop-shadow cleanup outside top-level cards on sky/canvas
            halo_mask = None
            halo_clean = None
            card_shadow = None
            cf_med_local = np.median(canvas_field[y:y + h, x:x + w].reshape(-1, 3), axis=0)
            on_canvas_directly = float(np.abs(parent_bg - cf_med_local).max()) <= 6.5
            if not is_flat_canvas and on_canvas_directly and w >= 120 and h >= 60:
                halo_mask = soft_shadow_halo_mask(
                    work_plate, canvas_field, [x, y, w, h], already_claimed, unresolved_guard
                )
                if halo_mask is not None and halo_mask.any():
                    halo_clean = halo_mask & ~already_claimed
                    if halo_clean.any():
                        card_shadow = estimate_card_shadow(work_plate, canvas_field, halo_clean)

            s_mask = full_erase & ~already_claimed
            if int(s_mask.sum()) < 16:
                continue

            boff = shape_fit.get('bbox_offset_local')
            sbbox = ([float(x + boff[0]), float(y + boff[1]), float(boff[2]), float(boff[3])]
                     if boff else [float(x), float(y), float(w), float(h)])
            shape_obj = Shape(
                id=sid,
                bbox=sbbox,
                z=1,
                geometry=shape_fit['geometry'],
                fill=shape_fit['fill'],
                fill_opacity=shape_fit.get('fill_opacity'),
                line_color=shape_fit['line_color'],
                line_width_pt=float(shape_fit['line_width_pt']),
                line_opacity=shape_fit.get('line_opacity'),
                line_dash=shape_fit.get('line_dash'),
                head_end=shape_fit.get('head_end'),
                tail_end=shape_fit.get('tail_end'),
                corner_radius_px=shape_fit['corner_radius_px'],
                fill_gradient=shape_fit.get('fill_gradient'),
                shadow=card_shadow,
                custom_paths=shape_fit.get('custom_paths'),
            )
            fused_try = fuse_incremental_scene(
                baseline,
                base_graph=graph,
                added_shapes=[shape_obj],
                background_hex=canvas_hex,
            )
            if fused_try.ownership.get(sid) != 'RESOLVED':
                continue

            parent_u8 = np.clip(np.rint(parent_patch), 0, 255).astype(np.uint8)
            erase_loc = shape_fit['erase_mask_local']
            work_plate[y:y + h, x:x + w][erase_loc] = parent_u8[erase_loc]
            restore_box = [x, y, w, h]
            backing_save = parent_u8

            if halo_clean is not None and halo_clean.any():
                field_u8 = np.clip(np.rint(canvas_field), 0, 255).astype(np.uint8)
                work_plate[halo_clean] = field_u8[halo_clean]
                s_mask = s_mask | halo_clean
                yy_m, xx_m = np.nonzero(s_mask)
                mx0, my0 = int(xx_m.min()), int(yy_m.min())
                mw0, mh0 = int(xx_m.max() - mx0 + 1), int(yy_m.max() - my0 + 1)
                restore_box = [mx0, my0, mw0, mh0]
                backing_save = np.clip(np.rint(canvas_field[my0:my0 + mh0, mx0:mx0 + mw0]), 0, 255).astype(np.uint8)

            already_claimed |= s_mask
            surface_claimed_mask |= s_mask
            claimed_child_ids.update(direct_oids)
            used_ids.add(sid)

            graph = fused_try
            pred_u8 = np.clip(np.rint(shape_fit['pred_fill']), 0, 255).astype(np.uint8)
            _refresh_overlying_backing_patches([x, y, w, h], erase_loc, pred_u8, {sid})
            proof_file = output / f'{sid}-owner.npz'
            np.savez_compressed(proof_file, mask=s_mask, restore_bbox=restore_box, backing_patch=backing_save)
            existing_proofs[sid] = {
                'mask': s_mask,
                'restore_bbox': restore_box,
                'backing_patch': backing_save.astype(float),
            }
            backed_by_surface[sid] = np.zeros((restore_box[3], restore_box[2]), dtype=bool)
            record = {
                'id': sid,
                'kind': 'surface',
                'bbox': [float(x), float(y), float(w), float(h)],
                'restore_bbox': restore_box,
                'payload': shape_obj.model_dump(),
                'method': 'container-surface-v2-gradient-preset/1',
                'state': 'STAGED',
                'quality': 'REVIEW_REQUIRED',
                'human_acceptance': False,
                'source_sha256': graph.source_sha256,
                'original_owner': 'source-plate',
                'complete_owner_scope': True,
                'mask_sha256': array_sha(s_mask.astype('uint8')),
                'owner_evidence': {
                    'file': proof_file.name,
                    'sha256': digest(proof_file),
                    'method': 'source-sampled-parent-backing/1',
                },
            }
            graph.transactions.append(record)
            graph.selected[sid] = record
            staged_new_ids.append(sid)
        else:
            # Clean Container Frame, Detached Curve/Arrow, or Structural Divider Asset PNG
            while f'a-c{c_seq:02d}' in used_ids:
                c_seq += 1
            aid = f'a-c{c_seq:02d}'
            c_seq += 1
            if aid in skip or (wanted_ids is not None and aid not in wanted_ids):
                continue

            hpad = 4 if cand_kind == 'box' else (0 if cand_kind == 'tick' else 2)
            px0 = max(0, x - hpad)
            py0 = max(0, y - (3 if h < 6 else 1))
            px1 = min(w_img, x + w + hpad)
            py1 = min(h_img, max(py0 + 6, y + h + (3 if h < 6 else 1)))
            ax, ay, aw, ah = px0, py0, px1 - px0, py1 - py0
            if (rectangle([ax, ay, aw, ah], size) & unresolved_guard).any():
                continue
            acrop = work_plate[ay:ay + ah, ax:ax + aw]
            parent_uint8 = np.clip(np.rint(parent_bg), 0, 255).astype(np.uint8)
            parent_patch_a = np.broadcast_to(parent_uint8.astype(float), (ah, aw, 3))

            fg_local = np.abs(acrop.astype(float) - parent_patch_a).max(2) > 3.0
            if not fg_local.any():
                continue

            fg_local &= ~already_claimed[ay:ay + ah, ax:ax + aw]
            rgba = extract_rgba(acrop, parent_patch_a, fg_local)
            is_fg_decor = False
            if cand_kind == 'asset_curve':
                # Detached curve/arrow above bar chart: exclude any already-claimed pixels
                child_local_c = already_claimed[ay:ay + ah, ax:ax + aw]
                rgba[child_local_c, 3] = 0
                fg_local &= ~child_local_c
            elif cand_kind == 'box':
                if surface_claimed_mask[ay:ay + ah, ax:ax + aw].any():
                    continue
                enclosing_parents = [
                    eb for ek, eb in candidates
                    if ek == 'box' and eb != raw_box and eb[2] * eb[3] >= 3 * (w * h)
                    and intersection([x, y, w, h], eb) >= 0.85 * float(w * h)
                ]
                straddles_larger_box = any(
                    0.05 * float(aw * ah) < intersection([ax, ay, aw, ah], eb) < 0.90 * float(aw * ah)
                    for ek, eb in candidates
                    if ek == 'box' and eb != raw_box and eb[2] * eb[3] >= 2 * (aw * ah)
                )
                hollow_ok = False
                if aw >= 60 and ah >= 32 and not straddles_larger_box:
                    closed_border = cv2.dilate(fg_local.astype('uint8'), np.ones((3, 3), np.uint8))
                    ext_ff = closed_border.copy()
                    ff_mask = np.zeros((ah + 2, aw + 2), np.uint8)
                    for sx_pt in range(aw):
                        if ext_ff[0, sx_pt] == 0:
                            cv2.floodFill(ext_ff, ff_mask, (sx_pt, 0), 2)
                        if ext_ff[ah - 1, sx_pt] == 0:
                            cv2.floodFill(ext_ff, ff_mask, (sx_pt, ah - 1), 2)
                    for sy_pt in range(ah):
                        if ext_ff[sy_pt, 0] == 0:
                            cv2.floodFill(ext_ff, ff_mask, (0, sy_pt), 2)
                        if ext_ff[sy_pt, aw - 1] == 0:
                            cv2.floodFill(ext_ff, ff_mask, (aw - 1, sy_pt), 2)
                    enclosed = ext_ff != 2
                    interior_solid = cv2.erode(enclosed.astype('uint8'), np.ones((3, 3), np.uint8)) > 0
                    inner_core = cv2.erode(enclosed.astype('uint8'), np.ones((7, 7), np.uint8)) > 0
                    child_local_a = already_claimed[ay:ay + ah, ax:ax + aw]
                    unclaimed_inner = inner_core & ~child_local_a
                    if int(interior_solid.sum()) >= 16 and int(inner_core.sum()) >= 16 and unclaimed_inner.any():
                        hollow_ok = True
                        inner_fill_med = np.median(acrop[unclaimed_inner].astype(float), axis=0)
                        border_ring = enclosed & ~inner_core
                        border_px = acrop[border_ring & ~child_local_a].astype(float)
                        parent_f = parent_bg.astype(float)
                        border_cands = border_px[
                            (np.abs(border_px - inner_fill_med).max(1) > 12.0)
                            & (np.abs(border_px - parent_f).max(1) > 8.0)
                        ] if len(border_px) > 0 else np.empty((0, 3), float)
                        if len(border_cands) < 24:
                            hollow_ok = False
                        else:
                            b_dists = np.abs(border_cands - inner_fill_med).max(1)
                            peak_cands = border_cands[b_dists >= np.percentile(b_dists, 70)]
                            border_stroke_rgb = np.median(peak_cands, axis=0)
                            all_unclaimed = enclosed & ~child_local_a
                            if all_unclaimed.any():
                                all_px = acrop[all_unclaimed].astype(float)
                                min_err = np.minimum.reduce([
                                    _segment_max_err(all_px, parent_f, border_stroke_rgb),
                                    _segment_max_err(all_px, inner_fill_med, border_stroke_rgb),
                                    _segment_max_err(all_px, parent_f, inner_fill_med),
                                ])
                                if float(min_err.max()) > 25.0:
                                    hollow_ok = False
                            if hollow_ok:
                                inner_anom = unclaimed_inner & (np.abs(acrop.astype(float) - inner_fill_med).max(2) > 11.0)
                                if inner_anom.any():
                                    if float(_segment_max_err(acrop[inner_anom].astype(float), inner_fill_med, border_stroke_rgb).max()) > 25.0:
                                        hollow_ok = False
                                    elif int((cv2.erode(inner_anom.astype('uint8'), np.ones((4, 4), np.uint8)) > 0).sum()) > 0:
                                        hollow_ok = False
                                    else:
                                        n_ac, _, astats, _ = cv2.connectedComponentsWithStats(inner_anom.astype('uint8'), 8)
                                        if any(min(int(astats[cid_i][2]), int(astats[cid_i][3])) > 4 for cid_i in range(1, n_ac)):
                                            hollow_ok = False
                        if hollow_ok:
                            rgba[interior_solid, :3] = acrop[interior_solid]
                            rgba[interior_solid, 3] = 255
                            fg_local |= enclosed
                if not hollow_ok:
                    # Compact foreground corner decoration inside a clean larger container (e.g. subtle banner wave)
                    if w * h <= 22000 and parent_err <= 3.5 and enclosing_parents:
                        ep = max(enclosing_parents, key=lambda b: b[2] * b[3])
                        ax = max(ep[0] + 2, x - 2)
                        ay = max(ep[1] + 2, y - 2)
                        px1_c = min(ep[0] + ep[2] - 2, x + w + 2)
                        py1_c = min(ep[1] + ep[3] - 2, y + h + 2)
                        aw, ah = px1_c - ax, py1_c - ay
                        if aw < 12 or ah < 8 or (rectangle([ax, ay, aw, ah], size) & unresolved_guard).any():
                            continue
                        acrop = work_plate[ay:ay + ah, ax:ax + aw]
                        parent_patch_a = np.broadcast_to(parent_uint8.astype(float), (ah, aw, 3))
                        fg_local = (np.abs(acrop.astype(float) - parent_patch_a).max(2) > 4.0) & (~already_claimed[ay:ay + ah, ax:ax + aw])
                        if int(fg_local.sum()) < 24:
                            continue
                        rgba = extract_rgba(acrop, parent_patch_a, fg_local)
                        is_fg_decor = True
                    else:
                        continue

            recovered_tile = None
            if recover_light_tile_marks and is_fg_decor:
                local_guard = already_claimed[ay:ay+ah, ax:ax+aw] | unresolved_guard[ay:ay+ah, ax:ax+aw]
                recovered_tile = _opaque_tile_interior(acrop, parent_uint8, local_guard)
                if recovered_tile is not None:
                    rgba[recovered_tile, :3] = acrop[recovered_tile]
                    rgba[recovered_tile, 3] = 255
            vis_local = rgba[..., 3] > 0
            if int(vis_local.sum()) < 12:
                continue
            full_vis = np.zeros((h_img, w_img), bool)
            full_vis[ay:ay + ah, ax:ax + aw] = vis_local
            a_mask = full_vis & ~already_claimed
            if int(a_mask.sum()) < 12:
                continue

            kind_str = 'decoration' if (cand_kind in ('line', 'tick') or is_fg_decor) else 'illustration'
            c_obj = ClosedObject(
                id=aid,
                kind=kind_str,
                bbox=[ax, ay, aw, ah],
                layers=['layer-00.png'],
                components=['whole'],
                parts=[{'name': 'whole', 'bbox': [ax, ay, aw, ah], 'layers': ['layer-00.png']}],
                exclusions=[],
                replaces=[],
                required_owners=[],
                text_policy='exclude_independent_text',
                intrinsic_text='',
                description=f'whole independent container {cand_kind}',
                prompt_en=f'whole independent container {cand_kind}',
            )
            fused_try = fuse_incremental_scene(
                baseline,
                base_graph=graph,
                added_assets=[c_obj],
                background_hex=canvas_hex,
            )
            if fused_try.ownership.get(aid) != 'RESOLVED':
                continue

            work_plate[ay:ay + ah, ax:ax + aw][vis_local] = parent_uint8
            already_claimed |= a_mask
            if not is_fg_decor:
                container_asset_mask |= a_mask
            claimed_child_ids.update(direct_oids)
            used_ids.add(aid)

            png_file = output / f'{aid}.png'
            Image.fromarray(rgba).save(png_file)
            proof_file = output / f'{aid}-owner.npz'
            bpatch_a = np.broadcast_to(parent_uint8, (ah, aw, 3)).copy()
            np.savez_compressed(
                proof_file,
                mask=a_mask,
                restore_bbox=[ax, ay, aw, ah],
                backing_patch=bpatch_a,
            )
            existing_proofs[aid] = {
                'mask': a_mask,
                'restore_bbox': [ax, ay, aw, ah],
                'backing_patch': bpatch_a.astype(float),
            }
            graph = fused_try
            record = {
                'id': aid,
                'kind': 'asset',
                'container_layer': (cand_kind != 'asset_curve') and (not is_fg_decor),
                'bbox': [ax, ay, aw, ah],
                'restore_bbox': [ax, ay, aw, ah],
                'payload': {'method': ('source-enclosed-opaque-tile/1' if recovered_tile is not None
                                       else 'cleaned-plate-container-unmix/1')},
                'state': 'STAGED',
                'quality': 'REVIEW_REQUIRED',
                'human_acceptance': False,
                'source_sha256': graph.source_sha256,
                'original_owner': 'source-plate',
                'complete_owner_scope': True,
                'mask_sha256': array_sha(a_mask.astype('uint8')),
                'asset_reference': {
                    'path': png_file.name,
                    'sha256': digest(png_file),
                    'source_sha256': graph.source_sha256,
                    'bbox': [ax, ay, aw, ah],
                    'crop_offset': [ax, ay],
                },
                'owner_evidence': {
                    'file': proof_file.name,
                    'sha256': digest(proof_file),
                    'method': 'source-derived-backing-and-ink/1',
                },
            }
            graph.transactions.append(record)
            graph.selected[aid] = record
            staged_new_ids.append(aid)

    if not staged_new_ids:
        shutil.rmtree(output, ignore_errors=True)
        return baseline / 'slideir.json', []

    attach_bar_groups(graph.selected, bar_series)
    attach_card_groups(graph.selected, size, unresolved_guard)

    # Assign explicit outer-to-inner z values on surface shapes
    sorted_shapes = sorted(graph.surface.shapes, key=lambda s: -(s.bbox[2] * s.bbox[3]))
    z_map = {s.id: idx for idx, s in enumerate(sorted_shapes, 1)}
    for s in graph.surface.shapes:
        s.z = z_map[s.id]
        if s.id in graph.selected and graph.selected[s.id].get('kind') == 'surface':
            graph.selected[s.id]['payload'] = s.model_dump()

    plate_rgba = np.dstack([work_plate, np.full((h_img, w_img), 255, dtype=np.uint8)])
    Image.fromarray(plate_rgba).save(output / 'source-plate.png')
    graph.raster_backing = {
        'path': 'source-plate.png',
        'sha256': digest(output / 'source-plate.png'),
        'source_sha256': graph.source_sha256,
        'bbox': [0, 0, *graph.image_size],
    }

    if (baseline / 'ownership.npz').is_file() and (baseline / 'ownership.json').is_file():
        with np.load(baseline / 'ownership.npz', allow_pickle=False) as ow_data:
            ownership_map = ow_data['ownership'].copy()
            uncertain_text = ow_data['uncertain_text'].copy()
        ownership_map[surface_claimed_mask] = 2
        ownership_map[container_asset_mask] = 0
        np.savez_compressed(output / 'ownership.npz', ownership=ownership_map, uncertain_text=uncertain_text)
        ow_meta = json.loads((baseline / 'ownership.json').read_text('utf8'))
        ow_meta['map_sha256'] = digest(output / 'ownership.npz')
        write(output / 'ownership.json', ow_meta)

    ir = adapt(graph, output)
    write(output / 'scene.json', graph.model_dump())
    write(output / 'slideir.json', ir)
    return output / 'slideir.json', staged_new_ids
