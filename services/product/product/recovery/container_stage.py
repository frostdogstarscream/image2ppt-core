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
    hex_rgb as _hex,
    soft_shadow_halo_mask,
    try_fit_native_shape_v2,
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
    thin_limit = 6
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

    def _add(kind: str, b: list[int]):
        x, y, w, h = b
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return
        if w * h > 0.68 * (w_img * h_img):
            return
        area_b = float(w * h)
        for sx, sy, sw, sh in seen_boxes:
            if abs(x - sx) <= 5 and abs(y - sy) <= 5 and abs(w - sw) <= 10 and abs(h - sh) <= 10:
                return
            ov = intersection(b, [sx, sy, sw, sh])
            if ov > 0 and ov / max(1.0, area_b + sw * sh - ov) >= 0.86:
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
        if sw < 140 or sh < 80:
            return
        pad_in = 6
        if sh - 2 * pad_in < 20 or sw - 2 * pad_in < 40:
            return
        inner_crop = work_plate[sy + pad_in:sy + sh - pad_in, sx + pad_in:sx + sw - pad_in]
        inner_med = np.median(inner_crop.reshape(-1, 3), axis=0)
        inner_field = np.broadcast_to(inner_med.astype(np.float32), inner_crop.shape)
        idiff = np.abs(inner_crop.astype(np.float32) - inner_field).max(2)
        isup = idiff > 4.2
        icore = idiff > 9.5
        in_cnt, ilbl, istats, _ = cv2.connectedComponentsWithStats(isup.astype('uint8'), 8)
        for ic in range(1, in_cnt):
            ix, iy, iw, ih, iarea = [int(v) for v in istats[ic]]
            if iarea < 80 or max(iw, ih) < 32:
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
            if min(iw, ih) <= 8 and max(iw, ih) >= 36:
                _add('line', cbox)
            elif min(iw, ih) >= 18:
                for r_kind, r_box in _split_two_tone_ribbon(work_plate, cbox):
                    _add(r_kind, r_box)

        # High-contrast flush-corner children (e.g. `01 应用成效` pentagon ribbon at card top-left `(sx+1, sy+1)`)
        flush_crop = work_plate[sy + 1:sy + sh - 1, sx + 1:sx + sw - 1].astype(np.float32)
        fdiff = np.abs(flush_crop - inner_med.astype(np.float32)).max(2)
        fsup = fdiff > 16.0
        fn_cnt, _, fstats, _ = cv2.connectedComponentsWithStats(fsup.astype('uint8'), 8)
        for fc in range(1, fn_cnt):
            fx, fy, fw, fh, farea = [int(v) for v in fstats[fc]]
            if farea >= 350 and fw >= 44 and fh >= 20 and not (fw >= sw - 6 and fh >= sh - 6):
                for r_kind, r_box in _split_two_tone_ribbon(work_plate, [sx + 1 + fx, sy + 1 + fy, fw, fh]):
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
                    # A genuine full-width header band reaches the card's left & right edges (not an indented inner bar)
                    l_strip = np.median(work_plate[sy + 4:sy + max(6, hdr_h - 3), sx + 2:min(sx + sw, sx + 10)].astype(np.float32), axis=(0, 1))
                    r_strip = np.median(work_plate[sy + 4:sy + max(6, hdr_h - 3), max(sx, sx + sw - 10):sx + sw - 2].astype(np.float32), axis=(0, 1))
                    edges_match = (
                        float(np.abs(l_strip - hdr_med).max()) <= 8.0 or float(np.abs(r_strip - hdr_med).max()) <= 8.0
                    ) and not (
                        float(np.abs(l_strip - inner_med).max()) <= 4.0 and float(np.abs(r_strip - inner_med).max()) <= 4.0
                    )
                    if edges_match and float(np.abs(hdr_med - inner_med).max()) >= 5.0 and 24 <= hdr_h <= 88:
                        _add('box', [sx, sy, sw, hdr_h])
                        # Check ribbon sitting inside top-left of header band
                        hdr_crop = work_plate[sy:sy + hdr_h, sx:sx + sw].astype(np.float32)
                        r_sup = np.abs(hdr_crop - hdr_med).max(2) > 14.0
                        rn, _, rstats, _ = cv2.connectedComponentsWithStats(r_sup.astype('uint8'), 8)
                        for rcid in range(1, rn):
                            rx_l, ry_l, rw_l, rh_l, rarea = [int(v) for v in rstats[rcid]]
                            if rarea >= 400 and rw_l >= 60 and rh_l >= 20:
                                for r_kind, r_box in _split_two_tone_ribbon(
                                    work_plate, [sx + rx_l, sy + ry_l, rw_l, rh_l]
                                ):
                                    _add(r_kind, r_box)

    for cid in range(1, n_lbl):
        bx, by, bw, bh, area = [int(v) for v in stats[cid]]
        if area < 72 or max(bw, bh) < 32:
            continue
        comp_mask = labels == cid
        if int((comp_mask & core).sum()) < 16:
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

    # Discover white/subtle bordered cards (whose fill is within 2-6 RGB of sky background)
    for kind, card_box in discover_bordered_cards(work_plate, scenery_guard):
        _add(kind, card_box)
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
    med0 = np.median(vals, axis=0).astype(np.float32)
    inliers = vals[np.abs(vals.astype(np.float32) - med0).max(1) <= 12.0]
    if len(inliers) < max(12, int(0.45 * len(vals))):
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
    parent_bg = np.median(inliers, axis=0).astype(np.float32)
    err = float(np.percentile(np.abs(inliers.astype(np.float32) - parent_bg).max(1), 80))
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


def attach_bar_groups(selected, series):
    """Only group complete, detected series whose individual native bars were staged."""
    by_box = {tuple(obj['bbox']): oid for oid, obj in selected.items() if obj.get('kind') == 'surface'
              and obj.get('payload', {}).get('geometry') == 'rect'}
    for boxes in series:
        if len(boxes) >= 3 and all(b in by_box for b in boxes):
            members = [by_box[b] for b in boxes]
            selected[members[0]]['move_members'] = members


def stage_containers_and_surfaces(
    baseline_directory: Path | str,
    output_directory: Path | str,
    *,
    skip_ids: set[str] | None = None,
    logo_ids: set[str] | None = None,
    wanted_ids: set[str] | None = None,
    marked_boxes: dict | None = None,
    measured_faces: dict | None = None,
    stable_ids: bool = False,
    scope_boxes: list | None = None,
    native_only: bool = False,
    shared_parent_backing: bool = False,
    layer_directory: Path | str | None = None,
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

    # Text guard: NEVER erase or cover uncommitted text!
    # (Draft LayerD asset proposals are NOT blindly rect-blocked here so ribbons/badges/bars
    # can convert to native shapes; pixel-level anomaly checks in `try_fit_native_shape_v2`
    # protect any unmodeled icon/photo pixels.)
    unresolved_guard = np.zeros((h_img, w_img), bool)
    for t in graph.text.texts:
        if t.id not in graph.selected:
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
    if measured_faces:
        # A detector may see just the label within a connected row. Fit the measured
        # colour component independently of the proposals made by the full-page scan.
        candidates = [('box', list(bounds)) for bounds in measured_faces] + [
            item for item in candidates if tuple(item[1]) not in measured_faces]
    if not candidates:
        return baseline / 'slideir.json', []

    shared_patches = {}
    series_members = {tuple(b) for series in bar_series for b in series}
    if shared_parent_backing:
        from .shared_surface_backing import bar_edge_backing, shared_face_patches

        source = np.asarray(Image.open(baseline / 'source.png').convert('RGB')) if layer_directory else None
        active_boxes = {tuple(bounds) for _, bounds in candidates
                        if (marked_boxes is None or tuple(bounds) in marked_boxes or tuple(bounds) in (measured_faces or {}))
                        and (scope_boxes is None or any(intersection(bounds, scope) >= bounds[2]*bounds[3]
                                                        for scope in scope_boxes))}
        shared_patches = shared_face_patches(
            work_plate, candidates, bar_series, unresolved_guard | already_claimed | higher_crop_vis | higher_asset_vis,
            source=source, layer_directory=layer_directory, active_boxes=active_boxes,
        )

    shutil.copytree(baseline, output)
    graph.surface.background = canvas_hex

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
        if scope_boxes is not None and not any(
            intersection([x, y, w, h], scope) >= w * h for scope in scope_boxes
        ):
            continue
        rect_area = rectangle([x, y, w, h], size)
        mark = (marked_boxes or {}).get((x,y,w,h))
        face_mark = (measured_faces or {}).get((x,y,w,h))
        if marked_boxes is not None and mark is None and face_mark is None:
            continue
        if face_mark and any(intersection(raw_box, rec['bbox']) > 0 and
                             intersection(raw_box, rec['bbox']) < rec['bbox'][2]*rec['bbox'][3]
                             for rec in graph.selected.values()):
            continue
        if (rect_area & unresolved_guard).any():
            continue

        ring_res = _ring_parent_patch(work_plate, canvas_field, [x, y, w, h], pad=5)
        shared = shared_patches.get((x, y, w, h))
        unresolved_parent = shared is None and (x, y, w, h) in shared_patches
        if shared is not None:
            parent_patch, backing_info = shared
            parent_bg = np.median(parent_patch.reshape(-1, 3), axis=0)
            if backing_info['method'] == 'shared-parent-samples/1' and (ring_res is None or ring_res[2] > 7.5):
                continue
        else:
            if ring_res is None:
                continue
            parent_patch, parent_bg, parent_err = ring_res
            if parent_err > 7.5:
                continue
        crop_cur = work_plate[y:y + h, x:x + w]
        # Recognition remains independent of the proposed replacement. A nicer
        # extrapolated backing cannot by itself turn a prior OCR repair into a shape.
        recognition_patch = ring_res[0] if ring_res is not None and ring_res[2] <= 7.5 else parent_patch
        if float(np.abs(crop_cur.astype(float) - recognition_patch).max()) < 5.5:
            continue

        # Collect direct staged children sitting inside this container
        direct_patches: list[np.ndarray] = []
        direct_oids: list[str] = []
        backing_child_ids = []
        child_backing = np.zeros((h,w), bool) if shared_parent_backing else None
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
                if child_backing is not None and x <= rx and y <= ry and rx+rw <= x+w and ry+rh <= y+h:
                    child_backing |= pinfo['mask'][y:y+h, x:x+w]
                    backing_child_ids.append(oid)

        direct_children_mask_local = already_claimed[y:y + h, x:x + w].copy()
        shape_fit = (
            try_fit_native_shape_v2(
                work_plate, [x, y, w, h], recognition_patch, direct_patches,
                direct_children_mask_local, cand_kind=cand_kind, measured_label=face_mark is not None,
            )
            if cand_kind in ('box', 'line') else None
        )

        if shape_fit is not None:
            if unresolved_parent:
                continue
            while f's-{s_seq:03d}' in used_ids:
                s_seq += 1
            sid = f's-{s_seq:03d}'
            s_seq += 1
            if mark and not face_mark and shape_fit.get('geometry') not in ('pentagon','chevron'):
                continue
            if mark or face_mark or stable_ids:
                from ..parallel_store import key
                sid = 's-' + key([graph.source_sha256, 'marked-surface', [x,y,w,h]])[:16]
            if sid in used_ids or sid in skip or (wanted_ids is not None and sid not in wanted_ids):
                continue
            full_erase = np.zeros((h_img, w_img), bool)
            full_erase[y:y + h, x:x + w] = shape_fit['erase_mask_local']

            # Optional soft drop-shadow cleanup outside top-level cards on sky/canvas
            halo_mask = None
            if shared is None and not face_mark and not is_flat_canvas and w >= 120 and h >= 60:
                halo_mask = soft_shadow_halo_mask(
                    work_plate, canvas_field, [x, y, w, h], already_claimed, unresolved_guard
                )

            s_mask = full_erase & ~already_claimed
            if int(s_mask.sum()) < 16:
                continue

            shape_obj = Shape(
                id=sid,
                bbox=[float(x), float(y), float(w), float(h)],
                z=1,
                geometry=shape_fit['geometry'],
                fill=shape_fit['fill'],
                line_color=shape_fit['line_color'],
                line_width_pt=float(shape_fit['line_width_pt']),
                corner_radius_px=shape_fit['corner_radius_px'],
                fill_gradient=shape_fit.get('fill_gradient'),
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
            if child_backing is not None:
                erase_loc = erase_loc & (s_mask[y:y+h, x:x+w] | child_backing)
            work_plate[y:y + h, x:x + w][erase_loc] = parent_u8[erase_loc]
            restore_box = [x, y, w, h]
            backing_save = parent_u8

            if shared is not None and (x,y,w,h) in series_members:
                extra = bar_edge_backing(work_plate, (x,y,w,h), parent_u8,
                                         unresolved_guard | already_claimed | higher_asset_vis | higher_crop_vis)
                if extra is not None:
                    restore_box, border, backing_save = extra
                    bx, by, bw, bh = restore_box
                    work_plate[by:by+bh, bx:bx+bw][border] = backing_save[border]
                    s_mask[by:by+bh, bx:bx+bw] |= border

            if halo_mask is not None and halo_mask.any():
                halo_clean = halo_mask & ~already_claimed
                if halo_clean.any():
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
            proof_file = output / f'{sid}-owner.npz'
            np.savez_compressed(proof_file, mask=s_mask, restore_bbox=restore_box, backing_patch=backing_save)
            existing_proofs[sid] = {
                'mask': s_mask,
                'restore_bbox': restore_box,
                'backing_patch': backing_save.astype(float),
            }
            record = {
                'id': sid,
                'kind': 'surface',
                'bbox': [float(x), float(y), float(w), float(h)],
                'restore_bbox': restore_box,
                'payload': shape_obj.model_dump(),
                **({'geometry_approximation':shape_fit['geometry_approximation']}
                   if 'geometry_approximation' in shape_fit else {}),
                **({'arrow_adjustment':shape_fit['arrow_adjustment']} if 'arrow_adjustment' in shape_fit else {}),
                'method': 'container-surface-v2-gradient-preset/1',
                **({'parent_backing':backing_info} if shared is not None else {}),
                **({'backing_child_ids':sorted(backing_child_ids)} if backing_child_ids else {}),
                **({'mark_id':face_mark} if face_mark else {}),
                **({'contained_children':sorted(oid for oid in direct_oids
                     if intersection([x,y,w,h], graph.selected[oid]['bbox']) >=
                     graph.selected[oid]['bbox'][2]*graph.selected[oid]['bbox'][3])} if stable_ids or face_mark else {}),
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
                    'method': backing_info['method'] if shared is not None else 'source-sampled-parent-backing/1',
                },
            }
            graph.transactions.append(record)
            graph.selected[sid] = record
            staged_new_ids.append(sid)
        elif not native_only:
            # Clean Container Frame, Detached Curve/Arrow, or Structural Divider Asset PNG
            while f'a-c{c_seq:02d}' in used_ids:
                c_seq += 1
            aid = f'a-c{c_seq:02d}'
            c_seq += 1
            if stable_ids:
                from ..parallel_store import key
                aid = 'a-c' + key([graph.source_sha256, cand_kind, [x,y,w,h]])[:16]
            if marked_boxes is not None or aid in used_ids or aid in skip or (wanted_ids is not None and aid not in wanted_ids):
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
            if shared is not None:
                parent_patch_a = np.pad(parent_patch,
                    ((y-ay, ay+ah-y-h), (x-ax, ax+aw-x-w), (0,0)), mode='edge')

            fg_local = np.abs(acrop.astype(float) - parent_patch_a).max(2) > 3.0
            if not fg_local.any():
                continue

            rgba = extract_rgba(acrop, parent_patch_a, fg_local)
            if cand_kind == 'asset_curve':
                # Detached curve/arrow above bar chart: exclude any already-claimed pixels
                child_local_c = already_claimed[ay:ay + ah, ax:ax + aw]
                rgba[child_local_c, 3] = 0
                fg_local &= ~child_local_c
            elif cand_kind == 'box' and aw >= 60 and ah >= 32:
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
                if int(interior_solid.sum()) < 16 or int(inner_core.sum()) < 16:
                    continue
                child_local_a = already_claimed[ay:ay + ah, ax:ax + aw]
                unclaimed_inner = inner_core & ~child_local_a
                if not unclaimed_inner.any():
                    continue
                inner_fill_med = np.median(acrop[unclaimed_inner].astype(float), axis=0)
                border_ring = enclosed & ~inner_core
                border_px = acrop[border_ring & ~child_local_a].astype(float)
                parent_f = parent_bg.astype(float)
                border_cands = border_px[
                    (np.abs(border_px - inner_fill_med).max(1) > 12.0)
                    & (np.abs(border_px - parent_f).max(1) > 8.0)
                ] if len(border_px) > 0 else np.empty((0, 3), float)
                if len(border_cands) < 24:
                    continue
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
                        continue
                inner_anom = unclaimed_inner & (np.abs(acrop.astype(float) - inner_fill_med).max(2) > 11.0)
                if inner_anom.any():
                    if float(_segment_max_err(acrop[inner_anom].astype(float), inner_fill_med, border_stroke_rgb).max()) > 25.0:
                        continue
                    if int((cv2.erode(inner_anom.astype('uint8'), np.ones((4, 4), np.uint8)) > 0).sum()) > 0:
                        continue
                    n_ac, _, astats, _ = cv2.connectedComponentsWithStats(inner_anom.astype('uint8'), 8)
                    if any(min(int(astats[cid_i][2]), int(astats[cid_i][3])) > 4 for cid_i in range(1, n_ac)):
                        continue
                rgba[interior_solid, :3] = acrop[interior_solid]
                rgba[interior_solid, 3] = 255
                fg_local |= enclosed
            elif cand_kind == 'box':
                continue

            vis_local = rgba[..., 3] > 0
            if int(vis_local.sum()) < 12:
                continue
            full_vis = np.zeros((h_img, w_img), bool)
            full_vis[ay:ay + ah, ax:ax + aw] = vis_local
            a_mask = full_vis & ~already_claimed
            if int(a_mask.sum()) < 12:
                continue

            kind_str = 'decoration' if cand_kind in ('line', 'tick') else 'illustration'
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

            bpatch_a = np.clip(np.rint(parent_patch_a), 0, 255).astype('uint8')
            writes = vis_local
            if child_backing is not None:
                permitted = a_mask[ay:ay+ah, ax:ax+aw].copy()
                permitted[y-ay:y-ay+h, x-ax:x-ax+w] |= child_backing
                writes = vis_local & permitted
            work_plate[ay:ay + ah, ax:ax + aw][writes] = bpatch_a[writes]
            already_claimed |= a_mask
            container_asset_mask |= a_mask
            claimed_child_ids.update(direct_oids)
            used_ids.add(aid)

            png_file = output / f'{aid}.png'
            Image.fromarray(rgba).save(png_file)
            proof_file = output / f'{aid}-owner.npz'
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
                'container_layer': cand_kind != 'asset_curve',
                'bbox': [ax, ay, aw, ah],
                'restore_bbox': [ax, ay, aw, ah],
                'payload': {'method': 'cleaned-plate-container-unmix/1'},
                **({'parent_backing':backing_info} if shared is not None else {}),
                **({'backing_child_ids':sorted(backing_child_ids)} if backing_child_ids else {}),
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
