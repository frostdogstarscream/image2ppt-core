"""Hierarchical recovery of text background boxes, cards, and structural dividers.

Runs after foreground texts and independent icons/illustrations have been staged
onto source-plate.png. Clean geometric boxes with flat interiors become native
PowerPoint shapes (SceneGraph.surface.shapes -> SlideIR shape nodes); chamfered
or internally-divided banner frames and gradient divider rules become clean
movable transparent PNG layers (SceneGraph.asset.objects -> SlideIR image nodes)
ordered strictly behind foreground icons and native text.
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
from ..parallel_rebuild import adapt, array_sha


def _hex(rgb: np.ndarray) -> str:
    vals = np.clip(np.rint(rgb), 0, 255).astype(int)
    return ''.join(f'{int(v):02X}' for v in vals[:3])


def _estimate_canvas_bg(plate: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = plate.shape[:2]
    m = min(24, max(4, min(h, w) // 20))
    border = np.concatenate([
        plate[:m, :].reshape(-1, 3),
        plate[-m:, :].reshape(-1, 3),
        plate[m:-m, :m].reshape(-1, 3),
        plate[m:-m, -m:].reshape(-1, 3),
    ], axis=0)
    quant = (border // 2) * 2
    vals, counts = np.unique(quant, axis=0, return_counts=True)
    mode_q = vals[counts.argmax()].astype(float)
    near = border[np.abs(border.astype(float) - mode_q).max(1) <= 6.0]
    bg = np.median(near, axis=0) if len(near) >= 32 else np.median(border, axis=0)
    err = float(np.percentile(np.abs(border.astype(float) - bg).max(1), 75))
    return bg, err


def _split_protruding_ticks(fg_support: np.ndarray, box: list[int], size: tuple[int, int]) -> list[tuple[str, list[int]]]:
    """Separate a thin 2-4px timeline connector tick touching the side of a tall pill bar."""
    w_img, h_img = size
    bx, by, bw, bh = box
    if bw < 80 or bh < 24:
        kind = 'line' if (bw >= 48 and bh <= 12) or (bh >= 48 and bw <= 12) else 'box'
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


def _discover_candidates(work_plate: np.ndarray, canvas_bg: np.ndarray) -> list[tuple[str, list[int]]]:
    h_img, w_img = work_plate.shape[:2]
    diff = np.abs(work_plate.astype(float) - canvas_bg.astype(float)).max(2)
    core = diff > 11.0
    support = diff > 4.0

    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(support.astype('uint8'), 8)
    found: list[tuple[str, list[int]]] = []
    seen_boxes: list[list[int]] = []

    def _add(kind: str, b: list[int]):
        x, y, w, h = b
        if w * h > 0.68 * (w_img * h_img):
            return
        for sx, sy, sw, sh in seen_boxes:
            if abs(x - sx) <= 3 and abs(y - sy) <= 3 and abs(w - sw) <= 6 and abs(h - sh) <= 6:
                return
        seen_boxes.append(b)
        found.append((kind, b))

    for cid in range(1, n_lbl):
        bx, by, bw, bh, area = [int(v) for v in stats[cid]]
        if area < 90 or max(bw, bh) < 48:
            continue
        comp_mask = labels == cid
        if int((comp_mask & core).sum()) < 20:
            continue
        for kind, split_box in _split_protruding_ticks(comp_mask, [bx, by, bw, bh], (w_img, h_img)):
            _add(kind, split_box)
            # If this is a large tinted container, scan inside it for nested child cards/bars.
            sx, sy, sw, sh = split_box
            if kind == 'box' and sw >= 160 and sh >= 100:
                inner_crop = work_plate[sy + 8:sy + sh - 8, sx + 8:sx + sw - 8]
                inner_med = np.median(inner_crop.reshape(-1, 3), axis=0)
                if np.abs(inner_med - canvas_bg).max() > 6.0:
                    idiff = np.abs(inner_crop.astype(float) - inner_med).max(2)
                    isup = idiff > 5.0
                    icore = idiff > 12.0
                    in_cnt, ilbl, istats, _ = cv2.connectedComponentsWithStats(isup.astype('uint8'), 8)
                    for ic in range(1, in_cnt):
                        ix, iy, iw, ih, iarea = [int(v) for v in istats[ic]]
                        if iarea < 120 or min(iw, ih) <= 14 or max(iw, ih) < 48:
                            continue
                        if int(((ilbl == ic) & icore).sum()) < 24:
                            continue
                        _add('box', [sx + 8 + ix, sy + 8 + iy, iw, ih])

    # Discard tiny debris boxes wholly contained inside a non-hollow container if they aren't real bars/cards
    cleaned: list[tuple[str, list[int]]] = []
    for kind, b in found:
        x, y, w, h = b
        if kind == 'box' and (w < 44 or h < 22):
            continue
        if kind in ('line', 'tick') and max(w, h) < 8:
            continue
        cleaned.append((kind, b))

    # Sort strictly innermost / smallest area first so children are extracted & replaced
    # by their parent fill BEFORE the enclosing parent container is evaluated!
    cleaned.sort(key=lambda item: (item[1][2] * item[1][3], item[1][1], item[1][0]))
    return cleaned


def _ring_parent_bg(work_plate: np.ndarray, box: list[int], pad: int = 4) -> tuple[np.ndarray, float] | None:
    h_img, w_img = work_plate.shape[:2]
    x, y, w, h = box
    rx0, ry0 = max(0, x - pad), max(0, y - pad)
    rx1, ry1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
    ring_mask = np.zeros((ry1 - ry0, rx1 - rx0), bool)
    ring_mask[:, :] = True
    ring_mask[y - ry0:y - ry0 + h, x - rx0:x - rx0 + w] = False
    vals = work_plate[ry0:ry1, rx0:rx1][ring_mask]
    if len(vals) < 16:
        return None
    med0 = np.median(vals, axis=0)
    inliers = vals[np.abs(vals.astype(float) - med0).max(1) <= 10.0]
    if len(inliers) < max(12, int(0.55 * len(vals))):
        return None
    parent_bg = np.median(inliers, axis=0)
    err = float(np.percentile(np.abs(inliers.astype(float) - parent_bg).max(1), 80))
    return parent_bg, err


def _try_fit_native_shape(
    work_plate: np.ndarray,
    box: list[int],
    parent_bg: np.ndarray,
    direct_child_patches: list[np.ndarray],
) -> dict | None:
    x, y, w, h = box
    if w < 44 or h < 22:
        return None
    crop = work_plate[y:y + h, x:x + w].astype(float)

    # 1. Check corners for large 45-degree chamfer cuts (e.g. tech banner cards).
    cs = min(18, max(4, w // 8), max(4, h // 4))
    corners = [
        crop[:cs, :cs],
        crop[:cs, -cs:],
        crop[-cs:, :cs],
        crop[-cs:, -cs:],
    ]
    corner_bg_counts = [int((np.abs(c - parent_bg).max(2) <= 4.5).sum()) for c in corners]
    # Note: If interior fill == parent_bg (white card with border on white canvas),
    # measure outside-of-border corner pixels using flood fill from each corner pixel!
    ix = min(12, max(4, w // 8))
    iy = min(10, max(4, h // 5))
    if h - 2 * iy < 6 or w - 2 * ix < 12:
        return None
    inner = crop[iy:h - iy, ix:w - ix]
    fill_med = np.median(inner.reshape(-1, 3), axis=0)

    inner_dev = np.abs(inner - fill_med).max(2)
    if float(np.percentile(inner_dev, 95)) > 9.0:
        return None
    strong_inner = int((inner_dev > 16.0).sum())
    if strong_inner > max(20, int(0.002 * inner.shape[0] * inner.shape[1])):
        return None

    # Measure true exterior corner cut pixels (outside border / fill)
    if np.abs(fill_med - parent_bg).max() <= 6.0:
        # Bordered card whose fill matches parent_bg: trace exterior corner triangles before hitting border stroke
        border_barrier = (np.abs(crop - parent_bg).max(2) > 6.5).astype('uint8')
        border_barrier = cv2.dilate(border_barrier, np.ones((2, 2), np.uint8))
        ext_counts = []
        for cy0, cx0, seed_pt in [(0, 0, (0, 0)), (0, w - cs, (cs - 1, 0)),
                                  (h - cs, 0, (0, cs - 1)), (h - cs, w - cs, (cs - 1, cs - 1))]:
            c_bar = border_barrier[cy0:cy0 + cs, cx0:cx0 + cs]
            if c_bar[seed_pt[1], seed_pt[0]] > 0:
                ext_counts.append(0)
            else:
                ff = c_bar.copy()
                mask_ff = np.zeros((cs + 2, cs + 2), np.uint8)
                cv2.floodFill(ff, mask_ff, seed_pt, 2)
                ext_counts.append(int((ff == 2).sum()))
        corner_bg_counts = ext_counts

    if max(corner_bg_counts) > 58:
        return None

    # 2. Optimize fill_rgb within fill_med +- 4 so every direct child's backing_patch stays <= 7.8 RGB!
    base_rgb = np.rint(fill_med).astype(int)
    best_rgb = base_rgb.copy()
    if direct_child_patches:
        all_child_px = np.concatenate(direct_child_patches, axis=0)
        c_min = all_child_px.min(axis=0)
        c_max = all_child_px.max(axis=0)
        lo_bound = np.ceil(c_max - 8.0).astype(int)
        hi_bound = np.floor(c_min + 8.0).astype(int)
        if np.any(lo_bound > hi_bound):
            return None
        cand_rgb = np.clip(base_rgb, lo_bound, hi_bound)
        if np.abs(cand_rgb - base_rgb).max() > 6:
            return None
        best_rgb = np.clip(cand_rgb, 0, 255)

    # 3. Detect 4-sided border stroke vs borderless solid pill bar
    edge_strip = min(4, max(2, min(w, h) // 10))
    trim_x = min(20, w // 5)
    trim_y = min(20, h // 5)
    edges = [
        crop[:edge_strip, trim_x:w - trim_x],          # top (shape: strip, len)
        crop[h - edge_strip:, trim_x:w - trim_x],      # bottom
        crop[trim_y:h - trim_y, :edge_strip].transpose(1, 0, 2),      # left -> (strip, len, 3)
        crop[trim_y:h - trim_y, w - edge_strip:].transpose(1, 0, 2),  # right -> (strip, len, 3)
    ]
    ref_mid = 0.5 * (best_rgb.astype(float) + parent_bg.astype(float))
    edge_Contrasts = []
    edge_colors = []
    for e_arr in edges:
        if e_arr.shape[1] < 4:
            continue
        d_e = np.abs(e_arr - ref_mid).max(2)
        idx_peak = d_e.argmax(axis=0)
        cols = np.arange(e_arr.shape[1])
        peak_px = e_arr[idx_peak, cols]
        peak_d = np.minimum(
            np.abs(peak_px - best_rgb.astype(float)).max(1),
            np.abs(peak_px - parent_bg.astype(float)).max(1),
        )
        edge_Contrasts.append(float(np.median(peak_d)))
        edge_colors.append(np.median(peak_px, axis=0))

    has_border = sum(c >= 6.5 for c in edge_Contrasts) >= 3
    if has_border:
        raw_line = np.median(np.stack(edge_colors, axis=0), axis=0)
        # De-blend ~25% background anti-aliasing on 1px (0.75pt) strokes so border stays crisp
        deblended = np.clip(raw_line + 0.25 * (raw_line - ref_mid), 0, 255)
        line_hex = _hex(deblended)
        line_width_pt = 0.75
    else:
        if float(np.abs(best_rgb.astype(float) - parent_bg).max()) < 7.0:
            return None
        line_hex = _hex(best_rgb)
        line_width_pt = 0.0

    avg_corner_px = float(np.median(corner_bg_counts))
    est_r = round(math.sqrt(max(0.0, avg_corner_px) / 0.2146))
    max_r = max(0, min(14, int(min(w, h) // 2) - 1))
    if est_r < 3 or max_r < 3:
        geometry = 'rect'
        corner_r = None
    else:
        geometry = 'roundRect'
        corner_r = float(min(max_r, max(4, est_r)))

    return {
        'geometry': geometry,
        'fill': _hex(best_rgb),
        'fill_rgb': best_rgb,
        'line_color': line_hex,
        'line_width_pt': line_width_pt,
        'corner_radius_px': corner_r,
    }


def stage_containers_and_surfaces(
    baseline_directory: Path | str,
    output_directory: Path | str,
    *,
    skip_ids: set[str] | None = None,
) -> tuple[Path, list[str]]:
    """Discover and stage draggable container surfaces and structural frame/line assets."""
    from reconstruction import extract_rgba

    skip = skip_ids or set()
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

    canvas_bg, canvas_err = _estimate_canvas_bg(work_plate)
    if not np.isfinite(canvas_err) or canvas_err > 6.0:
        return baseline / 'slideir.json', []

    # Load existing owner masks and build guard for any unresolved/unstaged foreground objects
    existing_proofs: dict[str, dict] = {}
    already_claimed = np.zeros((h_img, w_img), bool)
    higher_asset_vis = np.zeros((h_img, w_img), bool)
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
            higher_asset_vis[ay:ay + ah, ax:ax + aw] |= apng[..., 3] > 8

    unresolved_guard = np.zeros((h_img, w_img), bool)
    for t in graph.text.texts:
        if t.id not in graph.selected:
            unresolved_guard |= rectangle(t.bbox, size, pad=4)
    for a in graph.asset.objects:
        if a.id not in graph.selected:
            unresolved_guard |= rectangle(a.bbox, size, pad=4)

    candidates = _discover_candidates(work_plate, canvas_bg)
    if not candidates:
        return baseline / 'slideir.json', []

    shutil.copytree(baseline, output)
    graph.surface.background = _hex(canvas_bg)

    used_ids = {x.id for x in [*graph.text.texts, *graph.asset.objects, *graph.surface.shapes]}
    claimed_child_ids: set[str] = set()
    staged_new_ids: list[str] = []
    surface_claimed_mask = np.zeros((h_img, w_img), bool)
    container_asset_mask = np.zeros((h_img, w_img), bool)
    s_seq = 1
    c_seq = 1

    for cand_kind, raw_box in candidates:
        x, y, w, h = [int(v) for v in raw_box]
        x = max(0, min(w_img - 2, x))
        y = max(0, min(h_img - 2, y))
        w = max(2, min(w_img - x, w))
        h = max(2, min(h_img - y, h))
        rect_area = rectangle([x, y, w, h], size)
        if (rect_area & unresolved_guard).any():
            continue

        # Check if component still has visible signal on current work_plate
        ring_res = _ring_parent_bg(work_plate, [x, y, w, h], pad=4)
        if ring_res is None:
            continue
        parent_bg, parent_err = ring_res
        if parent_err > 5.0:
            continue
        crop_cur = work_plate[y:y + h, x:x + w]
        if float(np.abs(crop_cur.astype(float) - parent_bg).max()) < 7.0:
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
                # Exclude pixels occluded by a foreground badge asset (e.g. caption over circle badge)
                if graph.selected[oid].get('kind') == 'text':
                    cmask &= ~higher_asset_vis[ry:ry + rh, rx:rx + rw]
                if cmask.any():
                    direct_patches.append(pinfo['backing_patch'][cmask])
                direct_oids.append(oid)

        shape_fit = _try_fit_native_shape(work_plate, [x, y, w, h], parent_bg, direct_patches) if cand_kind == 'box' else None

        if shape_fit is not None:
            while f's-{s_seq:03d}' in used_ids:
                s_seq += 1
            sid = f's-{s_seq:03d}'
            s_seq += 1
            if sid in skip:
                continue
            s_mask = rect_area & ~already_claimed
            if int(s_mask.sum()) < 24:
                continue

            parent_uint8 = np.clip(np.rint(parent_bg), 0, 255).astype(np.uint8)
            work_plate[y:y + h, x:x + w] = parent_uint8
            already_claimed |= s_mask
            surface_claimed_mask |= s_mask
            claimed_child_ids.update(direct_oids)
            used_ids.add(sid)

            shape_obj = Shape(
                id=sid,
                bbox=[float(x), float(y), float(w), float(h)],
                z=1,
                geometry=shape_fit['geometry'],
                fill=shape_fit['fill'],
                line_color=shape_fit['line_color'],
                line_width_pt=float(shape_fit['line_width_pt']),
                corner_radius_px=shape_fit['corner_radius_px'],
            )
            graph.surface.shapes.append(shape_obj)
            graph.ownership[sid] = 'RESOLVED'

            proof_file = output / f'{sid}-owner.npz'
            backing_patch = np.broadcast_to(parent_uint8, (h, w, 3)).copy()
            np.savez_compressed(proof_file, mask=s_mask, restore_bbox=[x, y, w, h], backing_patch=backing_patch)
            record = {
                'id': sid,
                'kind': 'surface',
                'bbox': [float(x), float(y), float(w), float(h)],
                'restore_bbox': [x, y, w, h],
                'payload': shape_obj.model_dump(),
                'method': 'container-surface-confirmed/1',
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
            # Clean Container Frame or Structural Divider Asset PNG
            while f'a-c{c_seq:02d}' in used_ids:
                c_seq += 1
            aid = f'a-c{c_seq:02d}'
            c_seq += 1
            if aid in skip:
                continue

            # Ensure height >= 6 and width >= 10 so PowerPoint 0.9x move-scale probe has zero rounding drift;
            # for side-split ticks, do NOT expand horizontally into the adjacent container!
            hpad = 4 if cand_kind == 'box' else (0 if cand_kind == 'tick' else 1)
            px0 = max(0, x - hpad)
            py0 = max(0, y - (3 if h < 6 else 1))
            px1 = min(w_img, x + w + hpad)
            py1 = min(h_img, max(py0 + 6, y + h + (3 if h < 6 else 1)))
            ax, ay, aw, ah = px0, py0, px1 - px0, py1 - py0
            acrop = work_plate[ay:ay + ah, ax:ax + aw]
            parent_uint8 = np.clip(np.rint(parent_bg), 0, 255).astype(np.uint8)
            parent_patch = np.broadcast_to(parent_uint8.astype(float), (ah, aw, 3))

            fg_local = np.abs(acrop.astype(float) - parent_patch).max(2) > 2.0
            if not fg_local.any():
                continue

            rgba = extract_rgba(acrop, parent_patch, fg_local)
            if cand_kind == 'box' and aw >= 60 and ah >= 32:
                # Fill enclosed interior of stylized container card so card fill drags opaque as one unit
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
                rgba[interior_solid, :3] = acrop[interior_solid]
                rgba[interior_solid, 3] = 255
                fg_local |= enclosed

            vis_local = rgba[..., 3] > 0
            if int(vis_local.sum()) < 12:
                continue
            full_vis = np.zeros((h_img, w_img), bool)
            full_vis[ay:ay + ah, ax:ax + aw] = vis_local
            a_mask = full_vis & ~already_claimed
            if int(a_mask.sum()) < 12:
                continue

            work_plate[ay:ay + ah, ax:ax + aw][vis_local] = parent_uint8
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
                backing_patch=np.broadcast_to(parent_uint8, (ah, aw, 3)).copy(),
            )
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
            graph.asset.objects.append(c_obj)
            graph.ownership[aid] = 'RESOLVED'
            record = {
                'id': aid,
                'kind': 'asset',
                'container_layer': True,
                'bbox': [ax, ay, aw, ah],
                'restore_bbox': [ax, ay, aw, ah],
                'payload': {'method': 'cleaned-plate-container-unmix/1'},
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
