"""Expanded native shape discovery & fitting (gradients, preset geometries, white bordered cards, bar series).

Operates strictly within the existing SlideIR `shape` contract (`rect`, `roundRect`,
`ellipse`, `line`, `pentagon`, `chevron`, `fill_gradient`) without modifying the
fixed compiler. Enforces hard safety invariants: never erases unmodeled icons or
unresolved text, while allowing bounded visual tolerances from `tolerance_config`.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from ..contracts import Gradient, GradientStop
from ..tolerance_config import load_tolerance_config


def hex_rgb(rgb: np.ndarray) -> str:
    vals = np.clip(np.rint(rgb), 0, 255).astype(int)
    return ''.join(f'{int(v):02X}' for v in vals[:3])


def estimate_canvas_bg_flat(plate: np.ndarray) -> tuple[np.ndarray, float]:
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


def estimate_canvas_field(
    plate: np.ndarray,
    excluded_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, str, np.ndarray, bool]:
    """Return (canvas_field_float_HxWx3, background_hex, scenery_guard_HxW, is_flat).

    When the 4-edge border is uniform (`err <= 6.0`), returns a flat field identical
    to legacy `_estimate_canvas_bg`. When the slide has a smooth sky gradient and/or
    decorative bottom banner (e.g. cityscape silhouette), builds a smooth 2D canvas
    background field and marks non-card footer scenery in `scenery_guard`.
    """
    h, w = plate.shape[:2]
    flat_bg, flat_err = estimate_canvas_bg_flat(plate)
    if np.isfinite(flat_err) and flat_err <= 6.0:
        field = np.broadcast_to(flat_bg.astype(np.float32), (h, w, 3)).copy()
        return field, hex_rgb(flat_bg), np.zeros((h, w), dtype=bool), True

    excl = excluded_mask if excluded_mask is not None else np.zeros((h, w), dtype=bool)
    gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean5 = cv2.blur(gray, (5, 5))
    std5 = np.sqrt(np.maximum(cv2.blur(gray * gray, (5, 5)) - mean5 * mean5, 0.0))
    smooth_px = (std5 <= 4.5) & (~excl)

    # Anchor reference color on top + upper-side margins (y < 0.78 * h).
    m = min(28, max(6, min(h, w) // 24))
    upper_h = max(m + 1, int(0.78 * h))
    margin_mask = np.zeros((h, w), dtype=bool)
    margin_mask[:m, :] = True
    margin_mask[:upper_h, :m] = True
    margin_mask[:upper_h, -m:] = True
    anchor_px = plate[margin_mask & smooth_px]
    if len(anchor_px) < 64:
        anchor_px = plate[:m, :].reshape(-1, 3)
    quant = (anchor_px // 4) * 4
    vals, counts = np.unique(quant, axis=0, return_counts=True)
    mode_rgb = vals[counts.argmax()].astype(np.float32)
    inliers = anchor_px[np.abs(anchor_px.astype(np.float32) - mode_rgb).max(1) <= 18.0]
    ref_rgb = np.median(inliers, axis=0).astype(np.float32) if len(inliers) >= 32 else mode_rgb

    # Coarse grid interpolation of smooth background cells
    cell = 48
    gh = max(2, int(math.ceil(h / cell)))
    gw = max(2, int(math.ceil(w / cell)))
    grid_rgb = np.zeros((gh, gw, 3), dtype=np.float32)
    grid_valid = np.zeros((gh, gw), dtype=bool)

    for gy in range(gh):
        y0, y1 = gy * cell, min(h, (gy + 1) * cell)
        for gx in range(gw):
            x0, x1 = gx * cell, min(w, (gx + 1) * cell)
            sm = smooth_px[y0:y1, x0:x1]
            if int(sm.sum()) < max(24, int(0.20 * (y1 - y0) * (x1 - x0))):
                continue
            px = plate[y0:y1, x0:x1][sm].astype(np.float32)
            med = np.median(px, axis=0)
            if float(np.abs(med - ref_rgb).max()) <= 24.0:
                grid_rgb[gy, gx] = med
                grid_valid[gy, gx] = True

    if not grid_valid.any():
        field = np.broadcast_to(ref_rgb, (h, w, 3)).copy()
    else:
        yy, xx = np.nonzero(grid_valid)
        known_pts = np.stack([yy, xx], axis=1).astype(np.float32)
        known_vals = grid_rgb[yy, xx]
        for gy in range(gh):
            for gx in range(gw):
                if not grid_valid[gy, gx]:
                    d2 = (known_pts[:, 0] - gy) ** 2 + (known_pts[:, 1] - gx) ** 2
                    wgt = 1.0 / np.maximum(d2, 0.25)
                    grid_rgb[gy, gx] = (wgt[:, None] * known_vals).sum(axis=0) / wgt.sum()
        grid_rgb = cv2.GaussianBlur(grid_rgb, (3, 3), 0.8)
        field = cv2.resize(grid_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    # Scenery guard: protect bottom/edge decorative artwork (e.g. cityscape footer)
    scenery_guard = np.zeros((h, w), dtype=bool)
    footer_y = int(0.83 * h)
    footer_diff = np.abs(plate[footer_y:, :].astype(np.float32) - field[footer_y:, :]).max(2)
    footer_tex = (footer_diff > 13.0) | (std5[footer_y:, :] > 5.5)
    if float(footer_tex[-6:, :].mean()) >= 0.25:
        n_c, lbl_c, stats_c, _ = cv2.connectedComponentsWithStats(
            cv2.dilate(footer_tex.astype('uint8'), np.ones((5, 5), np.uint8)), 8
        )
        f_h = h - footer_y
        for cid in range(1, n_c):
            cy = int(stats_c[cid][1])
            ch = int(stats_c[cid][3])
            cw = int(stats_c[cid][2])
            if cy + ch >= f_h - 4 and cw >= int(0.25 * w):
                scenery_guard[footer_y:, :][lbl_c == cid] = True
        if scenery_guard.any():
            scenery_guard = cv2.dilate(scenery_guard.astype('uint8'), np.ones((7, 7), np.uint8)) > 0

    return field, hex_rgb(ref_rgb), scenery_guard, False


def _split_two_tone_ribbon(plate: np.ndarray, box: list[int]) -> list[tuple[str, list[int]]]:
    """Split a two-tone horizontal header ribbon (e.g. `[01 | 应用成效]` pentagon) at its vertical color step."""
    x, y, w, h = box
    if w < 110 or h < 22 or h > 96 or w < 1.8 * h:
        return [('box', box)]
    crop = plate[y:y + h, x:x + w].astype(np.float32)
    iy0, iy1 = max(2, h // 6), min(h - 2, h - h // 6)
    if iy1 - iy0 < 8:
        return [('box', box)]
    band = crop[iy0:iy1, :, :]
    # Measure column-to-column color step consistent across vertical rows
    col_med = np.median(band, axis=0)  # (w, 3)
    step = np.abs(col_med[2:] - col_med[:-2]).max(axis=1)  # len w-2, index i -> x_local = i+1
    margin_x = max(22, int(0.16 * w))
    if w - 2 * margin_x < 16:
        return [('box', box)]
    search = step[margin_x - 1:w - margin_x - 1]
    if len(search) == 0 or float(search.max()) < 22.0:
        return [('box', box)]
    split_local = int(search.argmax()) + margin_x
    left_med = np.median(col_med[max(4, split_local - 24):max(5, split_local - 3)], axis=0)
    right_med = np.median(col_med[min(w - 5, split_local + 3):min(w - 4, split_local + 24)], axis=0)
    if float(np.abs(left_med - right_med).max()) < 22.0:
        return [('box', box)]
    # Verify vertical straightness of the color step across rows
    d_left = np.abs(band[:, max(0, split_local - 4):split_local, :] - left_med).max(axis=2)
    d_right = np.abs(band[:, split_local + 1:min(w, split_local + 5), :] - right_med).max(axis=2)
    if float(np.percentile(d_left, 80)) > 16.0 or float(np.percentile(d_right, 80)) > 16.0:
        return [('box', box)]
    return [
        ('box', [x, y, split_local, h]),
        ('box', [x + split_local, y, w - split_local, h]),
    ]


def _split_bar_series_on_baseline(
    plate: np.ndarray,
    parent_field: np.ndarray,
    box: list[int],
) -> list[tuple[str, list[int]]] | None:
    """Detect >=3 aligned vertical bars resting on a thin horizontal baseline and split them cleanly."""
    x, y, w, h = box
    if w < 140 or h < 60:
        return None
    crop = plate[y:y + h, x:x + w].astype(np.float32)
    bg = parent_field[y:y + h, x:x + w].astype(np.float32)
    fg = np.abs(crop - bg).max(2) > 11.0
    row_cov = fg.mean(axis=1)
    # Look for a thin horizontal baseline near the bottom (last 18 pixels)
    bot_band = row_cov[max(0, h - 18):]
    high_rows = np.where(bot_band >= 0.65)[0]
    if len(high_rows) == 0 or len(high_rows) > 8:
        return None
    base_y0_local = max(0, h - 18) + int(high_rows.min())
    base_y1_local = max(0, h - 18) + int(high_rows.max()) + 1
    if base_y0_local < 32:
        return None

    above = fg[:max(1, base_y0_local - 1), :]
    n_c, _, stats_c, _ = cv2.connectedComponentsWithStats(above.astype('uint8'), 8)
    bars: list[list[int]] = []
    other_parts: list[tuple[str, list[int]]] = []
    for cid in range(1, n_c):
        bx, by, bw, bh, area = [int(v) for v in stats_c[cid]]
        if area < 80:
            continue
        fill_ratio = area / float(max(1, bw * bh))
        touches_baseline = (by + bh) >= base_y0_local - 4
        if touches_baseline and bw >= 14 and bh >= 16 and fill_ratio >= 0.82:
            bars.append([x + bx, y + by, bw, bh])
        elif area >= 150:
            other_parts.append(('asset_curve', [x + bx, y + by, bw, bh]))

    if len(bars) < 3:
        return None
    # Verify bars have comparable widths (e.g. uniform column chart bars)
    widths = [b[2] for b in bars]
    med_w = float(np.median(widths))
    if sum(abs(bw - med_w) <= max(6.0, 0.25 * med_w) for bw in widths) < 3:
        return None

    base_xs = np.where(fg[base_y0_local:base_y1_local, :].any(axis=0))[0]
    lx0 = x + int(base_xs.min()) if len(base_xs) else x
    lw = int(base_xs.max() - base_xs.min() + 1) if len(base_xs) else w
    baseline_h = max(3, base_y1_local - base_y0_local)
    out: list[tuple[str, list[int]]] = [('line', [lx0, y + base_y0_local, lw, baseline_h])]
    for b in sorted(bars, key=lambda item: item[0]):
        out.append(('box', b))
    out.extend(other_parts)
    return out


def discover_bordered_cards(
    plate: np.ndarray,
    scenery_guard: np.ndarray,
) -> list[tuple[str, list[int]]]:
    """Find axis-aligned white/subtle-fill cards bounded by 1px strokes or shadow edges."""
    h, w = plate.shape[:2]
    gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)) / 4.0
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)) / 4.0

    h_edge = ((gy > 2.2) & (gx < 2.2) & (~scenery_guard)).astype('uint8')
    v_edge = ((gx > 2.2) & (gy < 2.2) & (~scenery_guard)).astype('uint8')

    h_long = cv2.morphologyEx(h_edge, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (64, 1)))
    v_long = cv2.morphologyEx(v_edge, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 48)))

    # Bridge rounded corners (up to 18px radius) so closed card perimeters form sealed loops!
    wall = cv2.dilate(h_long | v_long, np.ones((3, 3), np.uint8))
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))

    # Flood-fill from outer border; enclosed interior chambers are candidate card interiors
    inv = (wall == 0).astype('uint8')
    n_c, _, stats_c, _ = cv2.connectedComponentsWithStats(inv, 4)
    results: list[tuple[str, list[int]]] = []
    page_area = float(w * h)
    for cid in range(1, n_c):
        bx, by, bw, bh, area = [int(v) for v in stats_c[cid]]
        if bx <= 2 or by <= 2 or bx + bw >= w - 2 or by + bh >= h - 2:
            continue
        if bw < 110 or bh < 44:
            continue
        rect_area = float(bw * bh)
        if rect_area > 0.68 * page_area or area < 0.72 * rect_area:
            continue
        # Expand slightly to include the border stroke around the enclosed chamber
        rx = max(0, bx - 3)
        ry = max(0, by - 3)
        rw = min(w - rx, bw + 6)
        rh = min(h - ry, bh + 6)
        results.append(('box', [rx, ry, rw, rh]))
    return results


def _segment_max_err(px: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    v = c1 - c0
    vv = (v * v).sum(axis=-1)
    t = np.where(vv >= 1.0, ((px - c0) * v).sum(axis=-1) / np.maximum(vv, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = c0 + t[..., None] * v
    return np.abs(px - proj).max(axis=-1)


def _corner_ext_mask(
    crop: np.ndarray,
    parent_patch: np.ndarray,
    cs: int,
    *,
    dilate_barrier: bool,
    thresh: float = 6.5,
) -> tuple[np.ndarray, list[int]]:
    h, w = crop.shape[:2]
    barrier = (np.abs(crop - parent_patch).max(2) > thresh).astype('uint8')
    if dilate_barrier:
        barrier = cv2.dilate(barrier, np.ones((2, 2), np.uint8))
    ext_mask = np.zeros((h, w), dtype=bool)
    ext_counts: list[int] = []
    for cy0, cx0, seed_pt in [
        (0, 0, (0, 0)),
        (0, w - cs, (cs - 1, 0)),
        (h - cs, 0, (0, cs - 1)),
        (h - cs, w - cs, (cs - 1, cs - 1)),
    ]:
        sub_bar = barrier[cy0:cy0 + cs, cx0:cx0 + cs]
        if sub_bar[seed_pt[1], seed_pt[0]] > 0:
            ext_counts.append(0)
            continue
        ff = sub_bar.copy()
        ff_mask = np.zeros((cs + 2, cs + 2), np.uint8)
        cv2.floodFill(ff, ff_mask, seed_pt, 2)
        filled = ff == 2
        ext_mask[cy0:cy0 + cs, cx0:cx0 + cs] |= filled
        ext_counts.append(int(filled.sum()))
    return ext_mask, ext_counts


def _detect_preset_geometry(
    crop: np.ndarray,
    parent_patch: np.ndarray,
    direct_children_mask_local: np.ndarray,
) -> tuple[str, float | None, np.ndarray] | None:
    """Detect `ellipse`, `pentagon`, `chevron`, `roundRect`, or `rect` with analytical exterior mask."""
    h, w = crop.shape[:2]
    fg = (np.abs(crop - parent_patch).max(2) > 6.5) | direct_children_mask_local

    # 1. Check `ellipse` for near-circular badges / icon backings
    if min(w, h) >= 24 and 0.72 <= (w / float(h)) <= 1.38:
        yy, xx = np.mgrid[:h, :w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        ry, rx = max(1.0, h / 2.0 - 0.5), max(1.0, w / 2.0 - 0.5)
        ell_in = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        ell_closed = cv2.morphologyEx(fg.astype('uint8'), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        inter = int((ell_in & ell_closed).sum())
        union = int((ell_in | ell_closed).sum())
        iou = inter / float(max(1, union))
        # Must clearly cut all 4 corners compared to a rectangle
        cs_e = max(4, int(min(w, h) * 0.22))
        corners_bg = [
            int((~ell_closed[:cs_e, :cs_e]).sum()),
            int((~ell_closed[:cs_e, w - cs_e:]).sum()),
            int((~ell_closed[h - cs_e:, :cs_e]).sum()),
            int((~ell_closed[h - cs_e:, w - cs_e:]).sum()),
        ]
        if iou >= 0.87 and min(corners_bg) >= max(6, int(0.18 * cs_e * cs_e)):
            return 'ellipse', None, ~ell_in

    # 2. Check right-pointing `pentagon` / `chevron` ribbons (e.g. `01 应用成效`, `平台架构`)
    if h >= 12 and w >= 1.2 * h:
        tip_w = min(h // 2 + 6, w // 2)
        if tip_w >= 10:
            fg_closed = cv2.morphologyEx(fg.astype('uint8'), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
            tr_bg = int((~fg_closed[:h // 2, w - tip_w:]).sum())
            br_bg = int((~fg_closed[h - h // 2:, w - tip_w:]).sum())
            mid_fg = bool(fg_closed[h // 2, max(0, w - 5):w].any())
            tri_area = 0.5 * (h / 2.0) * tip_w
            if (mid_fg
                    and tr_bg >= max(14, int(0.22 * tri_area))
                    and br_bg >= max(14, int(0.22 * tri_area))
                    and abs(tr_bg - br_bg) <= max(18, int(0.45 * max(tr_bg, br_bg)))):
                # Estimate tip start x where top/bottom rows drop into background
                top_fg = np.where(fg_closed[min(2, h - 1), :])[0]
                bot_fg = np.where(fg_closed[max(0, h - 3), :])[0]
                if len(top_fg) > 0 and len(bot_fg) > 0:
                    shoulder_x = int(round(0.5 * (top_fg.max() + bot_fg.max())))
                    if w - shoulder_x >= 7:
                        yy, xx = np.mgrid[:h, :w]
                        half_h = max(1.0, (h - 1) / 2.0)
                        dy = np.abs(yy - half_h) / half_h  # 0 at center, 1 at top/bottom
                        max_x = shoulder_x + (1.0 - dy) * (w - 1 - shoulder_x)
                        pent_in = xx <= (max_x + 0.5)
                        # Check left notch for chevron vs flat left edge for pentagon
                        left_notch_bg = int((~fg_closed[h // 4:h - h // 4, :min(tip_w, w // 4)]).sum())
                        left_corners_fg = bool(fg_closed[2, :4].any() and fg_closed[h - 3, :4].any())
                        if left_corners_fg and left_notch_bg >= max(16, int(0.25 * tri_area)):
                            notch_w = max(8, w - 1 - shoulder_x)
                            min_x = (1.0 - dy) * notch_w
                            chev_in = pent_in & (xx >= min_x - 0.5)
                            iou_c = int((chev_in & fg_closed).sum()) / float(max(1, int((chev_in | fg_closed).sum())))
                            if iou_c >= 0.84:
                                return 'chevron', None, ~chev_in
                        iou_p = int((pent_in & fg_closed).sum()) / float(max(1, int((pent_in | fg_closed).sum())))
                        if left_corners_fg and iou_p >= 0.85:
                            return 'pentagon', None, ~pent_in

    return None


def _fit_linear_gradient(
    crop: np.ndarray,
    fit_mask: np.ndarray,
    tol_p95: float,
) -> tuple[np.ndarray, float, Gradient | None] | None:
    """Fit 2D linear RGB gradient over `fit_mask` and return (pred_HxWx3, p95_err, Gradient|None)."""
    h, w = crop.shape[:2]
    if int(fit_mask.sum()) < 36:
        return None
    yy, xx = np.mgrid[:h, :w]
    nx = xx.astype(np.float64) / max(1.0, float(w - 1))
    ny = yy.astype(np.float64) / max(1.0, float(h - 1))
    design = np.stack([np.ones_like(nx), nx, ny], axis=2)
    rgb = crop.astype(np.float64)

    keep = fit_mask.copy()
    coeff = None
    for _ in range(5):
        if int(keep.sum()) < 30:
            return None
        coeff = np.linalg.lstsq(design[keep], rgb[keep], rcond=None)[0]
        err = np.abs(design @ coeff - rgb).max(2)
        med = float(np.median(err[keep]))
        mad = float(np.median(np.abs(err[keep] - med)))
        nxt = fit_mask & (err <= max(3.5, med + 3.2 * mad))
        if int(nxt.sum()) < 30 or np.array_equal(nxt, keep):
            break
        keep = nxt
    if coeff is None:
        return None
    pred = np.clip(design @ coeff, 0.0, 255.0)
    full_err = np.abs(pred - rgb).max(2)
    p95 = float(np.percentile(full_err[fit_mask], 95))
    if p95 > tol_p95:
        return None

    gx = coeff[1]  # RGB delta across full width (nx: 0 -> 1)
    gy = coeff[2]  # RGB delta across full height (ny: 0 -> 1)
    span_x = float(np.linalg.norm(gx))
    span_y = float(np.linalg.norm(gy))
    total_span = math.hypot(span_x, span_y)
    if total_span < 7.5:
        # Essentially flat color
        med_rgb = np.median(pred[fit_mask], axis=0)
        flat_pred = np.broadcast_to(med_rgb, (h, w, 3)).copy()
        flat_p95 = float(np.percentile(np.abs(flat_pred - rgb).max(2)[fit_mask], 95))
        return flat_pred, flat_p95, None

    # Dominant gradient direction in normalized (nx, ny) coordinates -> OOXML angle_deg
    # (0 deg = left->right, 90 deg = top->bottom)
    ux = span_x if abs(gx).max() >= abs(gy).max() else span_x
    # Signed direction from luminance/dominant channel derivative
    dom_ch = int(np.argmax(np.abs(gx) + np.abs(gy)))
    dx, dy = float(gx[dom_ch]), float(gy[dom_ch])
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        dx, dy = span_x, span_y
    angle_rad = math.atan2(dy, dx)
    angle_deg = round(math.degrees(angle_rad) % 360.0, 1)

    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    proj = nx * cos_a + ny * sin_a
    p_min, p_max = float(proj.min()), float(proj.max())
    p_mid = 0.5 * (p_min + p_max)

    def _color_at(t_val: float) -> str:
        # Find pixel closest to projection value t_val along gradient axis
        idx = int(np.argmin(np.abs(proj - t_val)))
        ry, rx = divmod(idx, w)
        return hex_rgb(pred[ry, rx])

    c0 = _color_at(p_min)
    cm = _color_at(p_mid)
    c1 = _color_at(p_max)
    stops = [
        GradientStop(offset=0, color=c0),
        GradientStop(offset=50000, color=cm),
        GradientStop(offset=100000, color=c1),
    ] if cm not in (c0, c1) else [
        GradientStop(offset=0, color=c0),
        GradientStop(offset=100000, color=c1),
    ]
    return pred, p95, Gradient(angle_deg=angle_deg, stops=stops)


def try_fit_native_shape_v2(
    work_plate: np.ndarray,
    box: list[int],
    parent_patch: np.ndarray,
    direct_child_patches: list[np.ndarray],
    direct_children_mask_local: np.ndarray,
    *,
    cand_kind: str = 'box',
) -> dict | None:
    """Fit solid or gradient native `Shape` (`rect`, `roundRect`, `ellipse`, `pentagon`, `chevron`)."""
    tol = load_tolerance_config().get('shape_fitting', {})
    solid_tol = float(tol.get('solid_interior_p95_rgb', 13.5))
    grad_tol = float(tol.get('gradient_interior_p95_rgb', 16.5))
    border_tol = float(tol.get('border_segment_max_rgb', 15.0))
    max_unexp_frac = float(tol.get('max_unexplained_pixel_fraction', 0.035))
    max_comp_side = int(tol.get('max_unexplained_component_min_side_px', 4))
    max_comp_area = int(tol.get('max_unexplained_component_area_px', 20))

    x, y, w, h = box
    crop = work_plate[y:y + h, x:x + w].astype(np.float64)
    parent_f = parent_patch.astype(np.float64) if parent_patch.ndim == 3 else np.broadcast_to(
        np.asarray(parent_patch, dtype=np.float64), (h, w, 3)
    )
    parent_med = np.median(parent_f.reshape(-1, 3), axis=0)

    # Thin divider rules (`line` / thin `rect` bar, e.g. header accent rule or vertical column divider)
    if cand_kind == 'line' or (min(w, h) <= 6 and max(w, h) >= 36):
        if min(w, h) < 2 or max(w, h) < 32:
            return None
        unclaimed = ~direct_children_mask_local
        if int(unclaimed.sum()) < 16:
            return None
        fg = np.abs(crop - parent_f).max(2) > 6.0
        if float(fg.mean()) < 0.45:
            return None
        grad_res = _fit_linear_gradient(crop, fg & unclaimed, grad_tol)
        if grad_res is None:
            return None
        pred_fill, _, grad_obj = grad_res
        med_rgb = np.rint(np.median(pred_fill[fg], axis=0)).astype(int)
        return {
            'geometry': 'rect',
            'fill': hex_rgb(med_rgb),
            'fill_rgb': med_rgb,
            'fill_gradient': grad_obj,
            'pred_fill': pred_fill,
            'line_color': hex_rgb(med_rgb),
            'line_width_pt': 0.0,
            'corner_radius_px': None,
            'erase_mask_local': fg | direct_children_mask_local,
        }

    if w < 24 or h < 18:
        return None

    # 1. Check preset geometry (`ellipse`, `pentagon`, `chevron`) vs `rect`/`roundRect`
    preset = _detect_preset_geometry(crop, parent_f, direct_children_mask_local)
    if preset is not None:
        geometry, corner_r, ext_corner_mask = preset
        corner_bg_counts = [0, 0, 0, 0]
    else:
        if w < 36 or h < 20:
            return None
        ix = min(12, max(3, w // 8))
        iy = min(10, max(3, h // 5))
        if h - 2 * iy < 6 or w - 2 * ix < 10:
            return None
        inner = crop[iy:h - iy, ix:w - ix]
        inner_child = direct_children_mask_local[iy:h - iy, ix:w - ix]
        inner_unclaimed = ~inner_child
        if int(inner_unclaimed.sum()) < 24:
            return None
        fill_med0 = np.median(inner[inner_unclaimed], axis=0)
        same_fill = float(np.abs(fill_med0 - parent_med).max()) <= 6.5
        cs = min(22, max(4, w // 7), max(4, h // 4))
        ext_corner_mask, corner_bg_counts = _corner_ext_mask(
            crop, parent_f, cs, dilate_barrier=same_fill
        )
        if max(corner_bg_counts) > int(0.85 * cs * cs) and max(corner_bg_counts) > 140:
            return None
        # Reject asymmetric chamfered polygons (e.g. two opposite 45-deg cut corners and two sharp corners)
        if max(corner_bg_counts) >= 20 and min(corner_bg_counts) <= 2:
            return None
        avg_corner_px = float(np.median(corner_bg_counts))
        est_r = round(math.sqrt(max(0.0, avg_corner_px) / 0.2146))
        max_r = max(0, min(22, int(min(w, h) // 2) - 1))
        if est_r < 3 or max_r < 3:
            geometry = 'rect'
            corner_r = None
        else:
            geometry = 'roundRect'
            corner_r = float(min(max_r, max(4, est_r)))

    # 2. Define perimeter vs interior zones
    border_t = 3
    perimeter_zone = np.zeros((h, w), dtype=bool)
    perimeter_zone[:border_t, :] = True
    perimeter_zone[-border_t:, :] = True
    perimeter_zone[:, :border_t] = True
    perimeter_zone[:, -border_t:] = True
    if ext_corner_mask.any():
        perimeter_zone |= cv2.dilate(ext_corner_mask.astype('uint8'), np.ones((5, 5), np.uint8)) > 0

    interior_unclaimed = (~ext_corner_mask) & (~perimeter_zone) & (~direct_children_mask_local)
    if int(interior_unclaimed.sum()) < 20:
        return None

    # 3. Try flat solid fill first, then 2D linear gradient fill
    fill_med = np.median(crop[interior_unclaimed], axis=0)
    flat_dev = np.abs(crop - fill_med).max(2)
    flat_p95 = float(np.percentile(flat_dev[interior_unclaimed], 95))
    grad_obj: Gradient | None = None

    if flat_p95 <= solid_tol:
        base_rgb = np.rint(fill_med).astype(int)
        best_rgb = base_rgb.copy()
        if direct_child_patches:
            all_child_px = np.concatenate(direct_child_patches, axis=0)
            c_min = all_child_px.min(axis=0)
            c_max = all_child_px.max(axis=0)
            lo_bound = np.ceil(c_max - 10.0).astype(int)
            hi_bound = np.floor(c_min + 10.0).astype(int)
            if np.any(lo_bound <= hi_bound):
                cand_rgb = np.clip(base_rgb, lo_bound, hi_bound)
                if np.abs(cand_rgb - base_rgb).max() <= 7:
                    best_rgb = np.clip(cand_rgb, 0, 255)
        pred_fill = np.broadcast_to(best_rgb.astype(np.float64), (h, w, 3)).copy()
    else:
        grad_res = _fit_linear_gradient(crop, interior_unclaimed, grad_tol)
        if grad_res is None:
            return None
        pred_fill, _, grad_obj = grad_res
        best_rgb = np.clip(np.rint(np.median(pred_fill[interior_unclaimed], axis=0)).astype(int), 0, 255)

    # Verify child backing patches agree reasonably with predicted shape fill
    if direct_child_patches and grad_obj is None:
        all_child_px = np.concatenate(direct_child_patches, axis=0)
        if float(np.percentile(np.abs(all_child_px - best_rgb.astype(float)).max(1), 90)) > 16.0:
            return None

    # 4. Detect optional 1px border stroke on rectangular/rounded shapes
    has_border = False
    if geometry in ('rect', 'roundRect'):
        edge_strip = min(4, max(2, min(w, h) // 10))
        trim_x = min(20, w // 5)
        trim_y = min(20, h // 5)
        edge_slices = [
            (crop[:edge_strip, trim_x:w - trim_x], pred_fill[:edge_strip, trim_x:w - trim_x], parent_f[:edge_strip, trim_x:w - trim_x]),
            (crop[h - edge_strip:, trim_x:w - trim_x], pred_fill[h - edge_strip:, trim_x:w - trim_x], parent_f[h - edge_strip:, trim_x:w - trim_x]),
            (crop[trim_y:h - trim_y, :edge_strip].transpose(1, 0, 2), pred_fill[trim_y:h - trim_y, :edge_strip].transpose(1, 0, 2), parent_f[trim_y:h - trim_y, :edge_strip].transpose(1, 0, 2)),
            (crop[trim_y:h - trim_y, w - edge_strip:].transpose(1, 0, 2), pred_fill[trim_y:h - trim_y, w - edge_strip:].transpose(1, 0, 2), parent_f[trim_y:h - trim_y, w - edge_strip:].transpose(1, 0, 2)),
        ]
        edge_contrasts = []
        edge_colors = []
        for e_crop, e_fill, e_par in edge_slices:
            if e_crop.shape[1] < 4:
                continue
            ref_mid = 0.5 * (e_fill + e_par)
            d_e = np.abs(e_crop - ref_mid).max(2)
            idx_peak = d_e.argmax(axis=0)
            cols = np.arange(e_crop.shape[1])
            peak_px = e_crop[idx_peak, cols]
            fill_col = e_fill[idx_peak, cols]
            par_col = e_par[idx_peak, cols]
            peak_d = np.minimum(
                np.abs(peak_px - fill_col).max(1),
                np.abs(peak_px - par_col).max(1),
            )
            edge_contrasts.append(float(np.median(peak_d)))
            edge_colors.append(np.median(peak_px, axis=0))
        has_border = sum(c >= 4.5 for c in edge_contrasts) >= 3
    else:
        edge_colors = []

    if has_border and edge_colors:
        raw_line = np.median(np.stack(edge_colors, axis=0), axis=0)
        ref_mid_med = 0.5 * (best_rgb.astype(float) + parent_med)
        deblended = np.clip(raw_line + 0.25 * (raw_line - ref_mid_med), 0, 255)
        line_hex = hex_rgb(deblended)
        line_width_pt = 0.75
        line_rgb = np.array([int(line_hex[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)
    else:
        # Borderless shape must visibly contrast with parent background!
        fill_vs_parent = float(np.percentile(np.abs(pred_fill[interior_unclaimed] - parent_f[interior_unclaimed]).max(1), 75))
        if fill_vs_parent < 6.0:
            return None
        line_hex = hex_rgb(best_rgb)
        line_width_pt = 0.0
        raw_line = best_rgb.astype(np.float64)
        line_rgb = best_rgb.astype(np.float64)

    # 5. Strict unmodeled-object safety check + soft tolerance check across ENTIRE box!
    fill_diff = np.abs(crop - pred_fill).max(2)
    if float(np.percentile(fill_diff[interior_unclaimed], 95)) > (grad_tol if grad_obj else solid_tol):
        return None

    proven_fill = (~ext_corner_mask) & (fill_diff <= max(13.0, (grad_tol if grad_obj else solid_tol) + 1.5))
    if has_border:
        line_b = np.broadcast_to(line_rgb, (h, w, 3))
        raw_b = np.broadcast_to(raw_line, (h, w, 3))
        seg_err = np.minimum.reduce([
            _segment_max_err(crop, parent_f, line_b),
            _segment_max_err(crop, pred_fill, line_b),
            _segment_max_err(crop, parent_f, raw_b),
            _segment_max_err(crop, pred_fill, raw_b),
            _segment_max_err(crop, parent_f, pred_fill),
        ])
    else:
        seg_err = _segment_max_err(crop, parent_f, pred_fill)

    proven_border = perimeter_zone & (~ext_corner_mask) & (seg_err <= border_tol)

    outer_rim = np.zeros((h, w), dtype=bool)
    outer_rim[:2, :] = True
    outer_rim[-2:, :] = True
    outer_rim[:, :2] = True
    outer_rim[:, -2:] = True
    if ext_corner_mask.any():
        outer_rim |= cv2.dilate(ext_corner_mask.astype('uint8'), np.ones((3, 3), np.uint8)) > 0
    ext_aa_bg = outer_rim & (np.abs(crop - parent_f).max(2) <= 9.0) & (~proven_fill) & (~proven_border)

    explained = direct_children_mask_local | ext_corner_mask | ext_aa_bg | proven_fill | proven_border
    unexplained = ~explained
    unexp_count = int(unexplained.sum())

    if unexp_count > 0:
        if unexp_count / float(w * h) > max_unexp_frac:
            return None
        # HARD SAFETY INVARIANT: Never swallow any compact unmodeled icon/glyph component
        # anywhere inside the interior OR outer margin band!
        n_u, _, u_stats, _ = cv2.connectedComponentsWithStats(unexplained.astype('uint8'), 8)
        for uid in range(1, n_u):
            uw, uh, uarea = int(u_stats[uid][2]), int(u_stats[uid][3]), int(u_stats[uid][4])
            if (uw >= max_comp_side and uh >= max_comp_side) or uarea >= max_comp_area:
                return None

    # Double-check interior anomalies (e.g. internal dashed dividers or small icons)
    strong_anom = (~ext_corner_mask) & (~direct_children_mask_local) & (~perimeter_zone) & (fill_diff > 14.5)
    if strong_anom.any():
        if int(strong_anom.sum()) >= 18:
            return None
        n_a, _, a_stats, _ = cv2.connectedComponentsWithStats(strong_anom.astype('uint8'), 8)
        for aid in range(1, n_a):
            aw_c, ah_c, a_area = int(a_stats[aid][2]), int(a_stats[aid][3]), int(a_stats[aid][4])
            if (aw_c >= 3 and ah_c >= 3 and a_area >= 10) or a_area >= 14:
                return None

    erase_mask_local = (direct_children_mask_local | proven_fill | proven_border) & ~ext_corner_mask
    return {
        'geometry': geometry,
        **({'arrow_adjustment':round(100000*int(ext_corner_mask[0].sum())/min(w,h))}
           if geometry=='pentagon' else {}),
        'fill': hex_rgb(best_rgb),
        'fill_rgb': best_rgb,
        'fill_gradient': grad_obj,
        'pred_fill': pred_fill,
        'line_color': line_hex,
        'line_width_pt': line_width_pt,
        'corner_radius_px': corner_r,
        'erase_mask_local': erase_mask_local,
    }


def soft_shadow_halo_mask(
    work_plate: np.ndarray,
    canvas_field: np.ndarray,
    box: list[int],
    already_claimed: np.ndarray,
    unresolved_guard: np.ndarray,
) -> np.ndarray | None:
    """Detect a smooth exterior drop-shadow halo immediately outside a card so clean backing removes it."""
    tol = load_tolerance_config().get('shape_fitting', {})
    pad = int(tol.get('shadow_halo_pad_px', 7))
    max_diff = float(tol.get('shadow_max_rgb_diff', 28.0))
    if pad <= 0:
        return None
    h_img, w_img = work_plate.shape[:2]
    x, y, w, h = box
    if w < 100 or h < 44:
        return None
    rx0, ry0 = max(0, x - pad), max(0, y - pad)
    rx1, ry1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
    ring = np.ones((ry1 - ry0, rx1 - rx0), dtype=bool)
    ring[y - ry0:y - ry0 + h, x - rx0:x - rx0 + w] = False
    guard_sub = already_claimed[ry0:ry1, rx0:rx1] | unresolved_guard[ry0:ry1, rx0:rx1]
    ring &= ~guard_sub
    if not ring.any():
        return None

    plate_sub = work_plate[ry0:ry1, rx0:rx1].astype(np.float32)
    field_sub = canvas_field[ry0:ry1, rx0:rx1].astype(np.float32)
    diff = np.abs(plate_sub - field_sub).max(2)
    darker_or_tinted = (field_sub.mean(2) - plate_sub.mean(2)) >= -2.0
    # Reject high-frequency strokes inside the shadow ring
    blur_sub = cv2.GaussianBlur(plate_sub, (5, 5), 1.2)
    smooth = np.abs(plate_sub - blur_sub).max(2) <= 5.5
    halo_local = ring & (diff > 1.5) & (diff <= max_diff) & darker_or_tinted & smooth
    if int(halo_local.sum()) < 16:
        return None
    full = np.zeros((h_img, w_img), dtype=bool)
    full[ry0:ry1, rx0:rx1] = halo_local
    return full
