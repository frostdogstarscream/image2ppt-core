"""Small-first region crops and transparent fallback assets for unowned content.

Region crops run after Stage 4 text repairs and never steal repairable
(``max_repair_attempts >= 1``) text lines:

* Repairable texts are excluded from crop seeds and never swallowed; intersecting
  crop boxes are shrunk around them or skipped.
* Blocking checks and backing fits evaluate against the lower-layer composite
  (source plate + committed surfaces + committed container layers) rather than
  treating background cards as foreground blockers.
* Eligible ``MOVABLE`` crops with transparent perimeters (and not 4-edge framed
  photos/tables/charts) upgrade to transparent RGBA assets via color-unmix or
  ``RMBG-2.0`` matting.
* ``MOVE_UNSAFE`` crops are disabled by default (``allow_move_unsafe=False``)
  and never modify underlying plate pixels.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..materialize import digest, write
from ..object_contracts import ClosedObject
from ..object_extract import rectangle
from ..parallel_config import ROOT
from ..parallel_contracts import SceneGraph

METHOD = 'source-plate-region-crop/1'
MAX_GROW_PX = 48
MAX_AREA_FRACTION = 0.25
MIN_CONTENT_FRACTION = 0.01
MOVABLE_RING_ERROR = 4.0
TRANSPARENT_MIN_TIGHT_RATIO = 0.12
_SEQ_CROP_ID = re.compile(r'^r-\d+$')
_FRAMED_KINDS = frozenset({'layout:image', 'layout:header_image', 'layout:footer_image', 'layout:chart', 'layout:table'})
_UNRECOVERABLE_REASONS = (
    'mask_incomplete',
    'background_repair_failed',
    'owner_conflict',
    'glyph ownership uncertain',
    'owner conflict',
)


def _extract_rgba(crop: np.ndarray, backing: np.ndarray, mask: np.ndarray) -> np.ndarray:
    src_dir = str(ROOT / 'ppt-rebuild/src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from reconstruction import extract_rgba
    return extract_rgba(crop, backing, mask)


def _clean_skip_seed_ids(skip_seed_ids: set[str] | None) -> set[str]:
    if not skip_seed_ids:
        return set()
    return {str(s) for s in skip_seed_ids if s and not _SEQ_CROP_ID.match(str(s))}


def _content(plate: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (5, 5))
    sq = np.square(gray)
    del gray
    var = cv2.blur(sq, (5, 5))
    del sq
    np.subtract(var, np.square(mean), out=var)
    np.maximum(var, 0, out=var)
    np.sqrt(var, out=var)
    return var > 6.0


def _clip(box, size):
    w_img, h_img = size
    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x1 = min(w_img, int(np.ceil(box[0] + box[2])))
    y1 = min(h_img, int(np.ceil(box[1] + box[3])))
    return [x0, y0, max(0, x1 - x0), max(0, y1 - y0)]


def _union(a, b):
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return [x0, y0, x1 - x0, y1 - y0]


def _intersects(a, b, gap=0):
    return not (a[0] + a[2] + gap <= b[0] or b[0] + b[2] + gap <= a[0]
                or a[1] + a[3] + gap <= b[1] or b[1] + b[3] + gap <= a[1])


def _intersection_area(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy


def _contains(outer, inner):
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[0] + outer[2] >= inner[0] + inner[2] and outer[1] + outer[3] >= inner[1] + inner[3])


def unrecoverable_text_ids_from_report(graph: SceneGraph, report: list[dict] | None = None, *,
                                       logo_ids: set[str] | None = None) -> set[str]:
    """Derive unrecoverable text IDs from a build report + graph ownership (unified rule)."""
    logos = set(logo_ids or ())
    by_id = {str(r.get('id')): r for r in (report or []) if r.get('id')}
    out: set[str] = set()
    for t in graph.text.texts:
        if t.id in graph.selected:
            continue
        if t.id in logos or graph.ownership.get(t.id) != 'RESOLVED':
            out.add(t.id)
            continue
        rec = by_id.get(t.id)
        if not rec or rec.get('state') != 'FALLBACK':
            continue
        reason = f"{rec.get('reason_code', '')} {rec.get('reason', '')}".lower()
        if any(tok in reason for tok in _UNRECOVERABLE_REASONS):
            out.add(t.id)
    return out


def collect_crop_seeds(
    graph: SceneGraph,
    *,
    unrecoverable_text_ids: set[str] | None = None,
    layout: dict | None = None,
    logo_ids: set[str] | None = None,
    skip_seed_ids: set[str] | None = None,
    max_committed_overlap: float = 0.3,
) -> tuple[list[dict], set[str]]:
    """Return (crop_seeds, protected_repairable_text_ids) using unified r10/runtime rules."""
    from .. import layout_evidence

    logos = set(logo_ids or ())
    skip = _clean_skip_seed_ids(skip_seed_ids)
    unrec = None if unrecoverable_text_ids is None else (set(unrecoverable_text_ids) | logos)
    seeds: list[dict] = []
    protected_text_ids: set[str] = set()

    for t in graph.text.texts:
        if t.id in graph.selected:
            continue
        if unrec is None:
            is_unrecoverable = (t.id in logos) or (graph.ownership.get(t.id) != 'RESOLVED')
        else:
            is_unrecoverable = t.id in unrec
        if is_unrecoverable:
            if t.id not in skip:
                seeds.append({'id': t.id, 'kind': 'text', 'bbox': list(t.bbox), 'text': t.text})
        else:
            protected_text_ids.add(t.id)

    for a in graph.asset.objects:
        if a.id not in graph.selected and a.id not in skip:
            seeds.append({'id': a.id, 'kind': 'asset', 'bbox': list(a.bbox), 'text': a.intrinsic_text})

    if layout:
        for s in layout_evidence.crop_seeds(layout, graph, max_committed_overlap=max_committed_overlap):
            if s['id'] not in skip:
                seeds.append(s)

    return seeds, protected_text_ids


def default_seeds(graph: SceneGraph, **kwargs) -> list[dict]:
    """Unified remaining non-repairable scene objects eligible as crop seeds."""
    return collect_crop_seeds(graph, **kwargs)[0]


def grow_seed(content: np.ndarray, seed, size, *, max_grow=MAX_GROW_PX):
    """Grow a seed box to its connected visible content; return (box, cuts_content)."""
    seed = _clip([seed[0] - 2, seed[1] - 2, seed[2] + 4, seed[3] + 4], size)
    if seed[2] <= 0 or seed[3] <= 0:
        return [seed[0], seed[1], 0, 0], False

    def _attempt(grow_px: int):
        window = _clip([seed[0] - grow_px, seed[1] - grow_px, seed[2] + 2 * grow_px, seed[3] + 2 * grow_px], size)
        wx, wy, ww, wh = window
        local = cv2.dilate(content[wy:wy + wh, wx:wx + ww].astype(np.uint8), np.ones((3, 3), np.uint8))
        _, labels = cv2.connectedComponents(local, connectivity=8)
        sx, sy = seed[0] - wx, seed[1] - wy
        hit = [int(v) for v in np.unique(labels[sy:sy + seed[3], sx:sx + seed[2]]) if v > 0]
        if not hit:
            return seed, False
        mask = np.isin(labels, hit)
        touches = ((wx > 0 and mask[:, 0].any()) or (wx + ww < size[0] and mask[:, -1].any())
                   or (wy > 0 and mask[0].any()) or (wy + wh < size[1] and mask[-1].any()))
        if touches:
            return seed, True
        ys, xs = np.nonzero(mask)
        grown = [wx + int(xs.min()), wy + int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        return _clip(_union(seed, [grown[0] - 1, grown[1] - 1, grown[2] + 2, grown[3] + 2]), size), False

    box, cuts = _attempt(max_grow)
    if cuts and max_grow < 240 and max(seed[2], seed[3]) >= 56 and (seed[2] * seed[3]) >= 3600:
        # Try one bounded composite-illustration expansion (e.g. rising arrow + badges inside a card)
        # only for medium/large illustration seeds, rejecting tiny icons or oversized card-wide merges.
        box2, cuts2 = _attempt(240)
        if not cuts2 and box2[2] <= int(0.28 * size[0]) and box2[3] <= int(0.32 * size[1]):
            return box2, False
    return box, cuts


def _detect_flush_card_photo(
    appearance: np.ndarray,
    lower_comp: np.ndarray,
    seed_box: list[int],
    fit_protected: np.ndarray,
) -> tuple[list[int], np.ndarray, float] | None:
    """Detect a rectangular card-inset photo flush with thin 1-2px borders and fit its card-body backing."""
    h_img, w_img = appearance.shape[:2]
    bx, by, bw, bh = _clip(seed_box, (w_img, h_img))
    if bw < 90 or bh < 70:
        return None
    sub0 = appearance[by:by + bh, bx:bx + bw].astype(np.float32)

    def _is_border_line(vals: np.ndarray) -> bool:
        med = np.median(vals, axis=0)
        return float(np.percentile(np.abs(vals - med).max(1), 85)) <= 6.5

    tx0, ty0, tx1, ty1 = bx, by, bx + bw, by + bh
    for _ in range(3):
        if tx1 - tx0 > 76 and _is_border_line(appearance[ty0:ty1, tx0]):
            tx0 += 1
        else:
            break
    for _ in range(3):
        if tx1 - tx0 > 76 and _is_border_line(appearance[ty0:ty1, tx1 - 1]):
            tx1 -= 1
        else:
            break
    for _ in range(3):
        if ty1 - ty0 > 56 and _is_border_line(appearance[ty0, tx0:tx1]):
            ty0 += 1
        else:
            break
    for _ in range(3):
        if ty1 - ty0 > 56 and _is_border_line(appearance[ty1 - 1, tx0:tx1]):
            ty1 -= 1
        else:
            break

    tw, th = tx1 - tx0, ty1 - ty0
    if tw < 76 or th < 56:
        return None
    sub = appearance[ty0:ty1, tx0:tx1]
    gray_std = float(cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY).std())
    if gray_std < 16.0 or float(_content(sub).mean()) < 0.42:
        return None

    margin_x = max(2, min(10, tw // 16))
    sx0, sx1 = tx0 + margin_x, tx1 - margin_x
    if sx1 - sx0 < 24:
        return None

    strip_candidates: list[tuple[int, int]] = []
    if ty1 + 14 <= h_img:
        strip_candidates.append((ty1 + 2, min(h_img, ty1 + 14)))
    if ty0 - 14 >= 0:
        strip_candidates.append((max(0, ty0 - 14), ty0 - 2))

    for sy0, sy1 in strip_candidates:
        if sy1 - sy0 < 4:
            continue
        allow = ~fit_protected[sy0:sy1, sx0:sx1]
        if int(allow.sum()) < 40:
            continue
        vals = lower_comp[sy0:sy1, sx0:sx1][allow].astype(np.float32)
        med = np.median(vals, axis=0)
        err = float(np.percentile(np.abs(vals - med).max(1), 80))
        if err > 4.0:
            continue
        photo_contrast = float(np.abs(sub.astype(np.float32) - med).max(2).mean())
        if photo_contrast < 14.0:
            continue
        # Corroborate flat card interior or surrounding 1-2px card frame at left/right edges
        has_card_side_walls = False
        if tx0 >= 5 and tx1 + 5 <= w_img:
            l_outer = lower_comp[ty0 + 8:ty1 - 8, tx0 - 5:tx0 + 1].astype(np.float32)
            r_outer = lower_comp[ty0 + 8:ty1 - 8, tx1 - 1:tx1 + 5].astype(np.float32)
            l_edge_diff = float(np.abs(l_outer - med).max())
            r_edge_diff = float(np.abs(r_outer - med).max())
            has_card_side_walls = (l_edge_diff >= 6.0 and r_edge_diff >= 6.0)
        if not has_card_side_walls and float(med.mean()) < 220.0:
            continue
        backing = np.broadcast_to(np.rint(med).astype(np.uint8), (th, tw, 3)).copy()
        return [tx0, ty0, tw, th], backing, err
    return None


def _shrink_box_away_from_protected(
    box: list[int],
    seed_boxes: list[list[float]],
    protected_boxes: list[list[int]],
    size: tuple[int, int],
) -> tuple[list[int] | None, bool]:
    """Shrink crop box so it never intersects repairable text; return (box_or_None, was_shrunk)."""
    cur = _clip(box, size)
    shrunk = False
    for pbox in protected_boxes:
        if not _intersects(cur, pbox):
            continue
        bx, by, bw, bh = cur
        px, py, pw, ph = pbox
        candidates = []
        if px > bx:
            candidates.append([bx, by, px - bx, bh])
        if px + pw < bx + bw:
            candidates.append([px + pw, by, (bx + bw) - (px + pw), bh])
        if py > by:
            candidates.append([bx, by, bw, py - by])
        if py + ph < by + bh:
            candidates.append([bx, py + ph, bw, (by + bh) - (py + ph)])
        valid = [_clip(c, size) for c in candidates if c[2] >= 2 and c[3] >= 2]
        if not valid:
            return None, True
        best = max(
            valid,
            key=lambda c: (sum(_intersection_area(c, sb) for sb in seed_boxes), c[2] * c[3]),
        )
        if best[2] < 2 or best[3] < 2:
            return None, True
        if seed_boxes and sum(_intersection_area(best, sb) for sb in seed_boxes) <= 0:
            return None, True
        cur = best
        shrunk = True
    if any(_intersects(cur, pb) for pb in protected_boxes):
        return None, True
    return cur, shrunk


def _split_oversized_crop(
    crop: dict,
    content: np.ndarray,
    size: tuple[int, int],
    max_area: float,
    protected_boxes: list[list[int]],
) -> list[dict]:
    """Split an oversized crop by connected content components or empty gutters."""
    x, y, w, h = crop['box']
    if w * h <= max_area:
        return [crop]
    local = content[y:y + h, x:x + w].astype(np.uint8)
    for ksize in (3, 1):
        work = cv2.dilate(local, np.ones((ksize, ksize), np.uint8)) if ksize > 1 else local
        count, _, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
        boxes: list[list[int]] = []
        for idx in range(1, count):
            area_px = int(stats[idx, cv2.CC_STAT_AREA])
            if area_px < 4:
                continue
            bx = x + int(stats[idx, cv2.CC_STAT_LEFT]) - 1
            by = y + int(stats[idx, cv2.CC_STAT_TOP]) - 1
            bw = int(stats[idx, cv2.CC_STAT_WIDTH]) + 2
            bh = int(stats[idx, cv2.CC_STAT_HEIGHT]) + 2
            b = _clip([bx, by, bw, bh], size)
            if b[2] >= 2 and b[3] >= 2:
                boxes.append(b)
        if len(boxes) >= 2:
            grouped: list[list[int]] = []
            for b in sorted(boxes, key=lambda r: (r[1], r[0])):
                merged = False
                for i, g in enumerate(grouped):
                    cand = _union(g, b)
                    if cand[2] * cand[3] <= max_area and not any(_intersects(cand, pb) for pb in protected_boxes):
                        if _intersects(g, b, gap=4):
                            grouped[i] = cand
                            merged = True
                            break
                if not merged:
                    grouped.append(b)
            pieces = []
            for b in grouped:
                shrunk_box, _ = _shrink_box_away_from_protected(b, [b], protected_boxes, size)
                if shrunk_box is None:
                    continue
                px, py, pw, ph = shrunk_box
                if pw >= 2 and ph >= 2 and float(content[py:py + ph, px:px + pw].mean()) >= MIN_CONTENT_FRACTION:
                    pieces.append({
                        'box': shrunk_box,
                        'seed_ids': list(crop['seed_ids']),
                        'texts': list(crop['texts']),
                        'kinds': list(crop.get('kinds', [])),
                        'seed_boxes': list(crop.get('seed_boxes', [crop['box']])),
                        'cuts': True,
                    })
            if pieces:
                return pieces
    # Try clean empty gutter split near midpoint along longer axis.
    def _gutter_split(box: list[int], depth: int) -> list[list[int]]:
        bx, by, bw, bh = box
        if bw * bh <= max_area or depth >= 4:
            return [box]
        sub = content[by:by + bh, bx:bx + bw] > 0
        best_split = None
        if bw >= bh:
            cols = ~sub.any(axis=0)
            mid = bw // 2
            zeros = [int(i) for i in np.nonzero(cols)[0] if 2 <= i <= bw - 3]
            if zeros:
                cx = min(zeros, key=lambda i: abs(i - mid))
                best_split = ([bx, by, cx, bh], [bx + cx + 1, by, bw - cx - 1, bh])
        else:
            rows = ~sub.any(axis=1)
            mid = bh // 2
            zeros = [int(i) for i in np.nonzero(rows)[0] if 2 <= i <= bh - 3]
            if zeros:
                cy = min(zeros, key=lambda i: abs(i - mid))
                best_split = ([bx, by, bw, cy], [bx, by + cy + 1, bw, bh - cy - 1])
        if best_split is None:
            return [box]
        out: list[list[int]] = []
        for sb in best_split:
            cb = _clip(sb, size)
            if cb[2] >= 2 and cb[3] >= 2:
                out.extend(_gutter_split(cb, depth + 1))
        return out or [box]

    gutter_boxes = _gutter_split([x, y, w, h], 0)
    if len(gutter_boxes) >= 2:
        pieces = []
        for b in gutter_boxes:
            shrunk_box, _ = _shrink_box_away_from_protected(b, [b], protected_boxes, size)
            if shrunk_box is None:
                continue
            px, py, pw, ph = shrunk_box
            if pw >= 2 and ph >= 2 and float(content[py:py + ph, px:px + pw].mean()) >= MIN_CONTENT_FRACTION:
                pieces.append({
                    'box': shrunk_box,
                    'seed_ids': list(crop['seed_ids']),
                    'texts': list(crop['texts']),
                    'kinds': list(crop.get('kinds', [])),
                    'seed_boxes': list(crop.get('seed_boxes', [crop['box']])),
                    'cuts': True,
                })
        if pieces:
            return pieces
    return [crop]


def plan_region_crops(
    graph: SceneGraph,
    plate: np.ndarray,
    seeds: list[dict] | None = None,
    *,
    max_area_fraction: float = MAX_AREA_FRACTION,
    protected_text_ids: set[str] | None = None,
    skip_seed_ids: set[str] | None = None,
    preserve_boxes=(),
) -> tuple[list[dict], list[dict]]:
    size = (plate.shape[1], plate.shape[0])
    content = _content(plate)
    page = float(size[0] * size[1])
    max_area = float(max_area_fraction) * page
    skip = _clean_skip_seed_ids(skip_seed_ids)

    default_s, default_prot = collect_crop_seeds(graph, skip_seed_ids=skip)
    if seeds is None:
        active_seeds = default_s
    else:
        active_seeds = [s for s in seeds if s.get('id') not in skip]

    seed_id_set = {s['id'] for s in active_seeds if s.get('id')}
    if protected_text_ids is not None:
        prot_ids = set(protected_text_ids) - seed_id_set
    else:
        prot_ids = set(default_prot) - seed_id_set

    absorbable_texts = [s for s in active_seeds if s.get('kind') == 'text']
    protected_boxes = [
        _clip(t.bbox, size)
        for t in graph.text.texts
        if t.id in prot_ids and len(t.bbox) == 4
    ]
    protected_boxes = [b for b in protected_boxes if b[2] > 0 and b[3] > 0]
    protected_boxes.extend(_clip(box, size) for box in preserve_boxes)

    crops: list[dict] = []
    report: list[dict] = []
    for seed in active_seeds:
        sb = _clip(seed['bbox'], size)
        if seed.get('picture_frame'):
            box = _clip(seed['picture_frame'], size)
            cuts = False
            fixed_frame = True
        else:
            box, cuts = grow_seed(content, seed['bbox'], size)
            fixed_frame = False
        shrunk_box, _ = _shrink_box_away_from_protected(box, [sb], protected_boxes, size)
        if shrunk_box is None:
            report.append({'seed_ids': [seed['id']], 'bbox': box, 'state': 'SKIPPED',
                           'reason': 'CONFLICTS_WITH_PROTECTED_TEXT'})
            continue
        crops.append({
            'box': shrunk_box,
            'seed_ids': [seed['id']],
            'texts': [seed.get('text') or ''],
            'kinds': [str(seed.get('kind') or 'asset')],
            'seed_boxes': [sb],
            'cuts': cuts,
            'fixed_frame': fixed_frame,
            'device_frame': bool(seed.get('device_frame')),
            'screen_backing': bool(seed.get('screen_backing')),
        })

    changed = True
    while changed:
        changed = False
        # Absorb only unrecoverable text seeds when partially overlapped; never swallow protected texts.
        for crop in crops:
            if crop.get('fixed_frame'):
                continue
            for t in absorbable_texts:
                tb = _clip(t['bbox'], size)
                if tb[2] <= 0 or tb[3] <= 0:
                    continue
                if _intersects(crop['box'], tb) and not _contains(crop['box'], tb):
                    cand = _clip(_union(crop['box'], [tb[0] - 2, tb[1] - 2, tb[2] + 4, tb[3] + 4]), size)
                    if cand[2] * cand[3] > max_area or any(_intersects(cand, pb) for pb in protected_boxes):
                        continue
                    crop['box'] = cand
                    if t['id'] not in crop['seed_ids']:
                        crop['seed_ids'].append(t['id'])
                        crop['texts'].append(t.get('text') or '')
                        crop['kinds'].append(str(t.get('kind') or 'text'))
                        crop['seed_boxes'].append(tb)
                    changed = True

        merged: list[dict] = []
        for crop in crops:
            target = None
            if not crop.get('fixed_frame'):
                for m in merged:
                    if m.get('fixed_frame') or not _intersects(m['box'], crop['box'], gap=2):
                        continue
                    cand = _union(m['box'], crop['box'])
                    if cand[2] * cand[3] <= max_area and not any(_intersects(cand, pb) for pb in protected_boxes):
                        target = m
                        break
            if target is None:
                merged.append(crop)
                continue
            target['box'] = _union(target['box'], crop['box'])
            target['seed_ids'] += [i for i in crop['seed_ids'] if i not in target['seed_ids']]
            target['texts'] += crop['texts']
            target['kinds'] += crop.get('kinds', [])
            target['seed_boxes'] += crop.get('seed_boxes', [])
            target['cuts'] = target['cuts'] or crop['cuts']
            changed = True
        crops = merged

    expanded: list[dict] = []
    for crop in crops:
        shrunk_box, _ = _shrink_box_away_from_protected(
            crop['box'], crop.get('seed_boxes', [crop['box']]), protected_boxes, size,
        )
        if shrunk_box is None:
            report.append({'seed_ids': crop['seed_ids'], 'bbox': crop['box'], 'state': 'SKIPPED',
                           'reason': 'CONFLICTS_WITH_PROTECTED_TEXT'})
            continue
        crop['box'] = shrunk_box
        if shrunk_box[2] * shrunk_box[3] > max_area:
            expanded.extend(_split_oversized_crop(crop, content, size, max_area, protected_boxes))
        else:
            expanded.append(crop)

    accepted = []
    for crop in expanded:
        x, y, w, h = crop['box']
        row = {'seed_ids': crop['seed_ids'], 'bbox': crop['box']}
        if w < 2 or h < 2:
            report.append({**row, 'state': 'SKIPPED', 'reason': 'EMPTY'})
            continue
        if w * h > max_area:
            report.append({**row, 'state': 'SKIPPED', 'reason': 'TOO_LARGE_FOR_SMALL_FIRST_CROP'})
            continue
        if float(content[y:y + h, x:x + w].mean()) < MIN_CONTENT_FRACTION:
            report.append({**row, 'state': 'SKIPPED', 'reason': 'NO_VISIBLE_CONTENT'})
            continue
        accepted.append(crop)
    return accepted, report


def _upper_foreground_records(graph: SceneGraph) -> list[dict]:
    """Committed foreground text and non-container asset records sitting above region crops."""
    rows = []
    for record in graph.selected.values():
        if record.get('kind') == 'surface' or record.get('container_layer'):
            continue
        box = record.get('restore_bbox') or record.get('bbox')
        if box and len(box) == 4:
            rows.append(record)
    return rows


def _committed_boxes(graph: SceneGraph) -> list[list[float]]:
    boxes = []
    for record in _upper_foreground_records(graph):
        box = record.get('restore_bbox') or record.get('bbox')
        if box and len(box) == 4:
            boxes.append([float(v) for v in box])
    return boxes


def _gradient_tile(w: int, h: int, grad) -> np.ndarray:
    """Render linear gradient over (w, h) RGBA tile with alpha=255."""
    stops = sorted(grad.stops, key=lambda st: st.offset)
    rad = math.radians(float(grad.angle_deg))
    dx, dy = math.cos(rad), math.sin(rad)
    xs = np.linspace(-0.5, 0.5, max(1, w), dtype=np.float32)
    ys = np.linspace(-0.5, 0.5, max(1, h), dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    proj = gx * dx + gy * dy
    pmin, pmax = float(proj.min()), float(proj.max())
    t = np.zeros_like(proj) if pmax <= pmin else (proj - pmin) / (pmax - pmin)
    offs = np.array([st.offset / 100000.0 for st in stops], dtype=np.float32)
    cols = np.array([[int(st.color[i:i + 2], 16) for i in (0, 2, 4)] for st in stops], dtype=np.float32)
    rgb = np.zeros((max(1, h), max(1, w), 3), dtype=np.uint8)
    for ch in range(3):
        rgb[..., ch] = np.clip(np.rint(np.interp(t, offs, cols[:, ch])), 0, 255).astype(np.uint8)
    return np.dstack([rgb, np.full((max(1, h), max(1, w)), 255, np.uint8)])


def _hex_rgba(hex6: str | None) -> tuple[int, int, int, int] | None:
    if not hex6 or hex6 == 'none' or len(hex6) != 6:
        return None
    return (int(hex6[0:2], 16), int(hex6[2:4], 16), int(hex6[4:6], 16), 255)


def _render_lower_composite(plate: np.ndarray, graph: SceneGraph, baseline: Path) -> tuple[np.ndarray, np.ndarray]:
    """Render lower layers: source-plate + committed surfaces + committed container_layer assets."""
    size = (plate.shape[1], plate.shape[0])
    canvas = Image.fromarray(plate).convert('RGBA')
    lower_covered = np.zeros(plate.shape[:2], bool)
    chosen = list(graph.selected.values())
    shape_by_id = {s.id: s for s in graph.surface.shapes}
    surfaces = sorted(
        [c for c in chosen if c.get('kind') == 'surface' and c.get('id') in shape_by_id],
        key=lambda c: (shape_by_id[c['id']].z, -(shape_by_id[c['id']].bbox[2] * shape_by_id[c['id']].bbox[3])),
    )
    for c in surfaces:
        s = shape_by_id[c['id']]
        fill_rgba = _hex_rgba(s.fill)
        line_rgba = _hex_rgba(s.line_color) if s.line_width_pt > 0 else None
        if fill_rgba is None and s.fill_gradient is None and line_rgba is None:
            continue
        x0, y0, bw, bh = [float(v) for v in s.bbox]
        ix0, iy0 = int(round(x0)), int(round(y0))
        iw, ih = max(1, int(round(bw))), max(1, int(round(bh)))
        mask = Image.new('L', size, 0)
        mdraw = ImageDraw.Draw(mask)
        box = (ix0, iy0, ix0 + iw - 1, iy0 + ih - 1)
        if s.geometry == 'ellipse':
            mdraw.ellipse(box, fill=255)
        elif s.geometry == 'roundRect':
            rad = min(float(s.corner_radius_px), min(bw, bh) / 2.0) if s.corner_radius_px is not None else min(bw, bh) * 0.16
            mdraw.rounded_rectangle(box, radius=rad, fill=255)
        else:
            mdraw.rectangle(box, fill=255)
        overlay = Image.new('RGBA', size, (0, 0, 0, 0))
        if s.fill_gradient is not None:
            tile = Image.fromarray(_gradient_tile(iw, ih, s.fill_gradient))
            grad_full = Image.new('RGBA', size, (0, 0, 0, 0))
            grad_full.paste(tile, (ix0, iy0))
            grad_full.putalpha(mask)
            overlay = grad_full
        elif fill_rgba is not None:
            solid = Image.new('RGBA', size, fill_rgba)
            solid.putalpha(mask)
            overlay = solid
        if line_rgba is not None:
            lw = max(1, int(round(float(s.line_width_pt) / 0.75)))
            odraw = ImageDraw.Draw(overlay)
            if s.geometry == 'ellipse':
                odraw.ellipse(box, outline=line_rgba, width=lw)
            elif s.geometry == 'roundRect':
                rad = min(float(s.corner_radius_px), min(bw, bh) / 2.0) if s.corner_radius_px is not None else min(bw, bh) * 0.16
                odraw.rounded_rectangle(box, radius=rad, outline=line_rgba, width=lw)
            else:
                odraw.rectangle(box, outline=line_rgba, width=lw)
        over_arr = np.asarray(overlay)
        lower_covered |= over_arr[..., 3] > 0
        canvas.alpha_composite(overlay)

    container_assets = sorted(
        [c for c in chosen if c.get('kind') == 'asset' and c.get('container_layer')],
        key=lambda c: -(c['bbox'][2] * c['bbox'][3]),
    )
    base_resolved = baseline.resolve()
    for c in container_assets:
        ref = c.get('asset_reference') or {}
        if not ref.get('path'):
            continue
        asset_path = (baseline / ref['path']).resolve()
        if not asset_path.is_relative_to(base_resolved) or not asset_path.is_file():
            continue
        if ref.get('sha256') and digest(asset_path) != ref['sha256']:
            raise ValueError('region crop lower container binding mismatch')
        ox, oy = int(round(c['bbox'][0])), int(round(c['bbox'][1]))
        png = Image.open(asset_path).convert('RGBA')
        arr = np.asarray(png)
        layer = Image.new('RGBA', size, (0, 0, 0, 0))
        layer.paste(png, (ox, oy))
        y0, x0 = max(0, oy), max(0, ox)
        y1, x1 = min(size[1], oy + arr.shape[0]), min(size[0], ox + arr.shape[1])
        if y1 > y0 and x1 > x0:
            lower_covered[y0:y1, x0:x1] |= arr[y0 - oy:y1 - oy, x0 - ox:x1 - ox, 3] > 0
        canvas.alpha_composite(layer)

    return np.array(canvas.convert('RGB')), lower_covered


def _is_four_edge_framed(core: np.ndarray, tx0: int, ty0: int, tx1: int, ty1: int, threshold: float = 0.80) -> bool:
    edges = (
        float(core[ty0, tx0:tx1 + 1].mean()),
        float(core[ty1, tx0:tx1 + 1].mean()),
        float(core[ty0:ty1 + 1, tx0].mean()),
        float(core[ty0:ty1 + 1, tx1].mean()),
    )
    return min(edges) >= threshold


def _detect_and_prepare_rounded_photo(
    pixels: np.ndarray,
    backing: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Detect symmetric rounded outer corners on a MOVABLE opaque photo crop and clean corner bleed."""
    h, w = pixels.shape[:2]
    if w < 80 or h < 60 or backing.shape[:2] != (h, w):
        return None
    px_f = pixels.astype(np.float32)
    bg_f = backing.astype(np.float32)
    # Require all 4 extreme 3x3 corners to match backing while mid-edges differ cleanly
    for ys, xs in ((slice(0, 3), slice(0, 3)), (slice(0, 3), slice(w - 3, w)),
                   (slice(h - 3, h), slice(0, 3)), (slice(h - 3, h), slice(w - 3, w))):
        if float(np.abs(px_f[ys, xs] - bg_f[ys, xs]).max()) > 12.0:
            return None
    mid_diffs = [
        float(np.median(np.abs(px_f[:2, w // 3:2 * w // 3] - bg_f[:2, w // 3:2 * w // 3]).max(2))),
        float(np.median(np.abs(px_f[h - 2:, w // 3:2 * w // 3] - bg_f[h - 2:, w // 3:2 * w // 3]).max(2))),
        float(np.median(np.abs(px_f[h // 3:2 * h // 3, :2] - bg_f[h // 3:2 * h // 3, :2]).max(2))),
        float(np.median(np.abs(px_f[h // 3:2 * h // 3, w - 2:] - bg_f[h // 3:2 * h // 3, w - 2:]).max(2))),
    ]
    if min(mid_diffs) <= 22.0:
        return None

    cs = min(28, w // 4, h // 4)
    if cs < 5:
        return None
    bg_match = (np.abs(px_f - bg_f).max(2) <= 12.0).astype(np.uint8)
    corner_mask = np.zeros((h, w), dtype=np.uint8)
    counts: list[int] = []
    for cy0, cx0, seed_pt in (
        (0, 0, (0, 0)),
        (0, w - cs, (cs - 1, 0)),
        (h - cs, 0, (0, cs - 1)),
        (h - cs, w - cs, (cs - 1, cs - 1)),
    ):
        sub = (1 - bg_match[cy0:cy0 + cs, cx0:cx0 + cs]).copy()
        if sub[seed_pt[1], seed_pt[0]] != 0:
            counts.append(0)
            continue
        ff_mask = np.zeros((cs + 2, cs + 2), np.uint8)
        cv2.floodFill(sub, ff_mask, seed_pt, 2)
        filled = (sub == 2)
        cnt = int(filled.sum())
        counts.append(cnt)
        corner_mask[cy0:cy0 + cs, cx0:cx0 + cs][filled] = 255
    if min(counts) < 3 or max(counts) > 2.5 * max(4, min(counts)):
        return None
    est_r = float(min(cs, max(4, round(math.sqrt(float(np.median(counts)) / 0.2146)))))
    cleaned = cv2.inpaint(pixels, corner_mask, 3, cv2.INPAINT_TELEA)
    return cleaned, est_r


def _detect_icon_in_tinted_circle(
    pixels: np.ndarray,
    backing: np.ndarray,
    fg_local: np.ndarray,
    kinds: list[str],
) -> tuple[list[int], np.ndarray] | None:
    """Detect a high-contrast icon inside a low-contrast uniform tinted circular/pad badge.

    Handles both centered circular badge crops (e.g. from LayerD) and off-center or clipped
    tinted-pad crops (e.g. from PP-DocLayoutV3 `layout:image` boxes). Extracting ONLY the
    inner icon (backed by the pad's fill color) prevents `source-unmix` from turning a subtle
    10-16 RGB tinted circle/pad into mottled semi-transparent holes, and leaves the clean empty
    circle/pad on `source-plate.png` for `post-crop-containers` to upgrade into a native
    editable PowerPoint `ellipse`/`roundRect` shape.
    """
    if any(k in ('layout:chart', 'layout:table') for k in kinds):
        return None
    h, w = pixels.shape[:2]
    if not (48 <= min(w, h) <= 185 and max(w, h) <= 210 and 0.55 <= (w / float(h)) <= 1.80):
        return None
    px_f = pixels.astype(np.float32)
    bg_f = backing.astype(np.float32)
    bg_med = np.median(bg_f.reshape(-1, 3), axis=0)

    # Path 1: Centered circular badge (exact annulus check)
    if 0.78 <= (w / float(h)) <= 1.28:
        yy, xx = np.mgrid[:h, :w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        r = min(w, h) * 0.46
        dist = np.hypot(xx - cx, yy - cy)
        ann = (dist >= 0.65 * r) & (dist <= 0.90 * r) & (~fg_local)
        if int(ann.sum()) >= 48:
            med_ann = np.median(px_f[ann], axis=0)
            ann_diff = float(np.abs(med_ann - bg_med).max())
            ann_p90 = float(np.percentile(np.abs(px_f[ann] - med_ann).max(1), 90))
            if 5.0 <= ann_diff <= 38.0 and ann_p90 <= 10.0:
                diff_ann = np.abs(px_f - med_ann).max(2)
                inner_icon = (dist <= 0.72 * r) & (diff_ann > 42.0) & (~fg_local)
                rim_icon = (dist >= 0.84 * r) & (diff_ann > 42.0) & (~fg_local)
                if int(inner_icon.sum()) >= 40 and int(rim_icon.sum()) <= 4:
                    ys, xs = np.nonzero(inner_icon)
                    ix0 = max(0, int(xs.min()) - 2)
                    iy0 = max(0, int(ys.min()) - 2)
                    ix1 = min(w, int(xs.max()) + 3)
                    iy1 = min(h, int(ys.max()) + 3)
                    iw, ih = ix1 - ix0, iy1 - iy0
                    if 12 <= iw < int(0.85 * w) and 12 <= ih < int(0.85 * h):
                        ring_m = np.zeros((h, w), dtype=bool)
                        ring_m[iy0:iy1, ix0:ix1] = True
                        ring_m[min(iy1, iy0 + 1):max(iy0, iy1 - 1), min(ix1, ix0 + 1):max(ix0, ix1 - 1)] = False
                        ring_m &= ~fg_local
                        if ring_m.any() and float(np.percentile(diff_ann[ring_m], 85)) <= 12.0:
                            inner_backing = np.broadcast_to(med_ann.astype(np.float32), (ih, iw, 3)).copy()
                            return [ix0, iy0, iw, ih], inner_backing

    # Path 2: Off-center or clipped tinted pad (icon surrounded by a uniform subtle tinted pad)
    diff_bg = np.abs(px_f - bg_f).max(2)
    hi = (diff_bg > 42.0) & (~fg_local)
    if int(hi.sum()) < 40:
        return None
    ys, xs = np.nonzero(hi)
    hx0, hy0, hx1, hy1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    hw, hh = hx1 - hx0, hy1 - hy0
    if not (12 <= hw <= 115 and 12 <= hh <= 115) or (hw * hh) > 0.72 * (w * h):
        return None
    # Sample 2..9px surround ring around the high-contrast icon bbox
    pad_out = np.zeros((h, w), dtype=bool)
    pad_out[max(0, hy0 - 9):min(h, hy1 + 9), max(0, hx0 - 9):min(w, hx1 + 9)] = True
    pad_in = np.zeros((h, w), dtype=bool)
    pad_in[max(0, hy0 - 1):min(h, hy1 + 1), max(0, hx0 - 1):min(w, hx1 + 1)] = True
    surround = pad_out & (~pad_in) & (~hi) & (~fg_local)
    if int(surround.sum()) < 48:
        return None
    med_pad = np.median(px_f[surround], axis=0)
    pad_diff = float(np.abs(med_pad - bg_med).max())
    pad_p85 = float(np.percentile(np.abs(px_f[surround] - med_pad).max(1), 85))
    if not (5.0 <= pad_diff <= 38.0 and pad_p85 <= 10.0):
        return None
    diff_pad = np.abs(px_f - med_pad).max(2)
    ix0 = max(0, hx0 - 2)
    iy0 = max(0, hy0 - 2)
    ix1 = min(w, hx1 + 2)
    iy1 = min(h, hy1 + 2)
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw < 12 or ih < 12 or (iw * ih) > 0.78 * (w * h):
        return None
    ring_m = np.zeros((h, w), dtype=bool)
    ring_m[iy0:iy1, ix0:ix1] = True
    ring_m[min(iy1, iy0 + 1):max(iy0, iy1 - 1), min(ix1, ix0 + 1):max(ix0, ix1 - 1)] = False
    ring_m &= ~fg_local
    if not ring_m.any() or float(np.percentile(diff_pad[ring_m], 85)) > 12.0:
        return None
    inner_backing = np.broadcast_to(med_pad.astype(np.float32), (ih, iw, 3)).copy()
    return [ix0, iy0, iw, ih], inner_backing


def _try_transparent_upgrade(
    pixels: np.ndarray,
    backing: np.ndarray,
    error: float,
    fg_local: np.ndarray,
    kinds: list[str],
) -> tuple[np.ndarray, list[int], str, float] | None:
    """Attempt transparent RGBA upgrade for eligible MOVABLE crops; return (tight_rgba, tight_offset_wh, method, ratio) or None."""
    from .. import matting

    delta = np.abs(pixels.astype(np.float32) - backing.astype(np.float32)).max(2)
    delta[fg_local] = 0.0
    core = delta > max(12.0, float(error) * 3.0)
    if int(core.sum()) < 6:
        return None
    ys, xs = np.nonzero(core)
    tx0, ty0, tx1, ty1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    tw, th = tx1 - tx0 + 1, ty1 - ty0 + 1
    if tw < 2 or th < 2:
        return None
    is_framed_kind = any(k in _FRAMED_KINDS for k in kinds)
    if _is_four_edge_framed(core, tx0, ty0, tx1, ty1, threshold=0.70 if is_framed_kind else 0.80):
        return None

    # 1) Try source-background color unmix (tighter edge floor suppresses faint sky/JPEG haze).
    edge_thresh = max(4.5, float(error) * 1.8)
    edge_mask = (cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (delta > edge_thresh)
    edge_mask[fg_local] = False
    if edge_mask.any():
        try:
            rgba = _extract_rgba(pixels, backing, edge_mask)
            rgba[fg_local] = 0
            # Suppress negligible outer haze (alpha < 14 outside core) without hurting AA transitions
            low_haze = (~core) & (rgba[..., 3] < 14)
            rgba[low_haze] = 0
            tight = rgba[ty0:ty1 + 1, tx0:tx1 + 1]
            alpha = tight[..., 3:4].astype(np.float32) / 255.0
            recomp = tight[..., :3].astype(np.float32) * alpha + backing[ty0:ty1 + 1, tx0:tx1 + 1].astype(np.float32) * (1.0 - alpha)
            sub_mask = edge_mask[ty0:ty1 + 1, tx0:tx1 + 1]
            sub_fg = fg_local[ty0:ty1 + 1, tx0:tx1 + 1]
            if sub_mask.any():
                err = np.abs(recomp - pixels[ty0:ty1 + 1, tx0:tx1 + 1].astype(np.float32)).max(2)[sub_mask]
                valid_area = tight[..., 3][~sub_fg]
                ratio = float((valid_area == 0).mean()) if valid_area.size else 0.0
                if float(err.mean()) <= 2.5 and float(np.percentile(err, 95)) <= 6.5 and ratio >= TRANSPARENT_MIN_TIGHT_RATIO and (tight[..., 3] > 0).any():
                    return tight, [tx0, ty0, tw, th], 'source-unmix-crop-upgrade/1', ratio
        except Exception:
            pass

    # 2) Try RMBG-2.0 matting when available and not framed photo/chart/table.
    if not is_framed_kind and matting.available():
        try:
            alpha = matting.predict_alpha(pixels)
            alpha[fg_local] = 0.0
            rgba = matting.unmix(pixels, backing, alpha)
            rgba[fg_local] = 0
            vis = rgba[..., 3] > 0
            if not vis.any():
                return None
            vys, vxs = np.nonzero(vis)
            vx0, vy0, vx1, vy1 = int(vxs.min()), int(vys.min()), int(vxs.max()), int(vys.max())
            vw, vh = vx1 - vx0 + 1, vy1 - vy0 + 1
            if vw < 2 or vh < 2 or _is_four_edge_framed(vis, vx0, vy0, vx1, vy1, threshold=0.80):
                return None
            strong_missing = (delta > 35.0) & (~vis)
            if float(strong_missing.sum()) > 0.03 * max(1, int(core.sum())):
                return None
            tight = rgba[vy0:vy1 + 1, vx0:vx1 + 1]
            sub_fg = fg_local[vy0:vy1 + 1, vx0:vx1 + 1]
            valid_area = tight[..., 3][~sub_fg]
            ratio = float((valid_area == 0).mean()) if valid_area.size else 0.0
            if ratio >= TRANSPARENT_MIN_TIGHT_RATIO:
                t_alpha = tight[..., 3:4].astype(np.float32) / 255.0
                recomp = tight[..., :3].astype(np.float32) * t_alpha + backing[vy0:vy1 + 1, vx0:vx1 + 1].astype(np.float32) * (1.0 - t_alpha)
                m_vis = tight[..., 3] > 0
                err = np.abs(recomp - pixels[vy0:vy1 + 1, vx0:vx1 + 1].astype(np.float32)).max(2)[m_vis]
                if err.size and float(np.percentile(err, 95)) <= 16.0:
                    return tight, [vx0, vy0, vw, vh], 'rmbg2-crop-upgrade/1', ratio
        except Exception:
            pass
    return None


def _submask_any(entry: tuple[int, int, int, int, np.ndarray], box: list[int]) -> bool:
    """Return True if any True pixel inside a tight sub-mask intersects `box`."""
    mx0, my0, mw0, mh0, sub_m = entry
    bx0, by0, bw0, bh0 = box
    ix0, iy0 = max(bx0, mx0), max(by0, my0)
    ix1, iy1 = min(bx0 + bw0, mx0 + mw0), min(by0 + bh0, my0 + mh0)
    if ix0 >= ix1 or iy0 >= iy1:
        return False
    return bool(sub_m[iy0 - my0:iy1 - my0, ix0 - mx0:ix1 - mx0].any())


def _screen_backing_from_edges(source, box, protected, preliminary):
    """Continue a pale carrier into a removed screen using four held-out edge strips."""
    x,y,w,h = box
    if min(x,y,source.shape[1]-x-w,source.shape[0]-y-h) < 6 or min(w,h) < 80:
        return None
    result = preliminary.astype(np.float32).copy()
    yy,xx = np.ogrid[:h,:w]
    corrections = np.zeros_like(result)
    weights = np.zeros((h,w,1), np.float32)
    old_errors, validation = [], []
    for axis, side in ((0,-1),(0,1),(1,-1),(1,1)):
        view = source if axis else source.transpose(1,0,2)
        guard = protected if axis else protected.T
        start, length = (x,w) if axis else (y,h)
        edge = (y if side < 0 else y+h-1) if axis else (x if side < 0 else x+w-1)
        offsets = edge+side*np.arange(1,7)
        strips = view[offsets,start:start+length].astype(np.float32)
        if guard[offsets,start:start+length].any() or np.percentile(strips,2) < 190:
            return None
        profile = np.median(strips[2:],axis=0).astype(np.float32)
        profile = cv2.GaussianBlur(profile[None],(11,1),2)[0]
        held = np.median(strips[:2],axis=0)
        if np.percentile(np.abs(held-profile).max(1),95) > 6:
            return None
        index = 0 if side < 0 else -1
        ref = result[index] if axis else result[:,index]
        correction = profile-ref
        if np.max(np.abs(correction)) > 18:
            return None
        distance = (yy if side < 0 else h-1-yy) if axis else (xx if side < 0 else w-1-xx)
        weight = np.exp(-distance / max(8,min(h,w)*.12))[...,None]
        corrections += weight*(correction[None] if axis else correction[:,None])
        weights += weight
        old_errors.append(np.abs(held-ref).max(1))
        validation.append((axis,index,held))
    result = np.uint8(np.clip(np.rint(result+corrections/np.maximum(1,weights)),0,255))
    before = float(np.percentile(np.concatenate(old_errors),95))
    after = float(np.percentile(np.concatenate([np.abs(held-(result[i] if axis else result[:,i])).max(1)
                                                for axis,i,held in validation]),95))
    if after >= before or after > 6:
        return None
    return result, {'method':'four-side-held-out-carrier/1','edge_p95_before':before, 'edge_p95_after':after}


def stage_region_crops(
    baseline: str | Path,
    output: str | Path,
    *,
    seeds: list[dict] | None = None,
    allow_movable: bool = True,
    allow_move_unsafe: bool = False,
    allow_transparent_upgrade: bool = True,
    max_area_fraction: float = MAX_AREA_FRACTION,
    protected_text_ids: set[str] | None = None,
    skip_seed_ids: set[str] | None = None,
    preserve_boxes=(),
) -> tuple[Path, list[str], list[dict]]:
    """Stage crops on a built baseline directory; return (slideir path, staged ids, report)."""
    from ..object_unmix import backing_from_evidence
    from ..parallel_fusion import fuse_incremental_scene
    from ..parallel_rebuild import adapt, array_sha

    baseline, output = Path(baseline), Path(output)
    skip = _clean_skip_seed_ids(skip_seed_ids)
    graph = SceneGraph.model_validate_json((baseline / 'scene.json').read_text('utf8'))
    rb = graph.raster_backing or {}
    plate_file = (baseline / rb.get('path', 'source-plate.png')).resolve()
    if not plate_file.is_relative_to(baseline.resolve()) or digest(plate_file) != rb.get('sha256'):
        raise ValueError('region crop source-plate binding mismatch')
    plate_rgba = np.array(Image.open(plate_file).convert('RGBA'))
    plate = plate_rgba[..., :3].copy()
    plate_alpha = plate_rgba[..., 3].copy()
    del plate_rgba
    source = np.array(Image.open(baseline / 'source.png').convert('RGB'))
    if digest(baseline / 'source.png') != graph.source_sha256 or source.shape != plate.shape:
        raise ValueError('region crop source snapshot mismatch')
    size = (plate.shape[1], plate.shape[0])

    foreground = np.zeros(plate.shape[:2], bool)
    fg_tight_masks: dict[str, tuple[int, int, int, int, np.ndarray]] = {}
    for record in graph.selected.values():
        ref = record.get('owner_evidence')
        if not ref or record.get('region_crop') or record.get('container_layer'):
            continue
        if record.get('kind') not in ('text', 'asset'):
            continue
        proof = (baseline / ref['file']).resolve()
        if not proof.is_relative_to(baseline.resolve()) or digest(proof) != ref['sha256']:
            raise ValueError('region crop owner evidence binding mismatch')
        with np.load(proof, allow_pickle=False) as data:
            m = data['mask'].astype(bool)
            foreground |= m
            ys_m, xs_m = np.nonzero(m)
            if ys_m.size:
                mx0, my0 = int(xs_m.min()), int(ys_m.min())
                mw0, mh0 = int(xs_m.max()) - mx0 + 1, int(ys_m.max()) - my0 + 1
                fg_tight_masks[record['id']] = (mx0, my0, mw0, mh0, m[my0:my0 + mh0, mx0:mx0 + mw0].copy())
            del m

    lower_comp, lower_covered = _render_lower_composite(plate, graph, baseline)
    appearance = np.where(foreground[..., None], lower_comp, source)
    crops, report = plan_region_crops(
        graph,
        appearance,
        seeds,
        max_area_fraction=max_area_fraction,
        protected_text_ids=protected_text_ids,
        skip_seed_ids=skip,
        preserve_boxes=preserve_boxes,
    )
    if not crops:
        return baseline / 'slideir.json', [], report
    if output.exists():
        raise ValueError('region crop output already exists')
    shutil.copytree(baseline, output)

    upper_records = _upper_foreground_records(graph)
    planned_rects = [rectangle(c['box'], size) for c in crops]
    used = {x.id for x in [*graph.text.texts, *graph.asset.objects, *graph.surface.shapes]} | set(graph.selected)
    staged: list[str] = []
    seq = 1
    movable_mask = np.zeros(plate.shape[:2], bool)
    upgraded_mask = np.zeros(plate.shape[:2], bool)

    for idx_c, crop in enumerate(crops):
        if any(sid in skip for sid in crop['seed_ids']):
            continue
        while f'r-{seq:03d}' in used:
            seq += 1
        rid = f'r-{seq:03d}'
        seq += 1
        used.add(rid)
        x, y, w, h = crop['box']
        intrinsic = ' / '.join(t.strip() for t in crop['texts'] if t and t.strip())[:200]
        crop_obj = ClosedObject(
            id=rid, kind='illustration', bbox=[x, y, w, h], layers=['source-plate.png'],
            components=['whole'], parts=[{'name': 'whole', 'bbox': [x, y, w, h], 'layers': ['source-plate.png']}],
            exclusions=[], replaces=[], required_owners=[],
            text_policy='intrinsic_label' if intrinsic else 'exclude_independent_text', intrinsic_text=intrinsic,
            description='opaque source region crop (no semantic claim)', prompt_en='opaque source region crop',
        )
        fused_try = fuse_incremental_scene(baseline, base_graph=graph, added_assets=[crop_obj])
        if fused_try.ownership.get(rid) != 'RESOLVED':
            report.append({'id': rid, 'state': 'SKIPPED', 'reason': f"host fusion ownership={fused_try.ownership.get(rid)}",
                           'bbox': [x, y, w, h], 'seed_ids': crop['seed_ids']})
            continue

        # Lower layers (surface shapes & container assets) never block crops above them.
        # Wholly-contained upper foreground labels/icons sit above the crop; for committed text
        # whose padded bbox grazes an adjacent small icon (e.g. icon next to label), check actual
        # pixel mask intersection and tighten away from glyph pixels rather than falsely blocking.
        blocked_by_upper = False
        contained_fg = np.zeros(plate.shape[:2], bool)
        crop_box_cur = list(crop['box'])
        for rec in upper_records:
            rb_box = rec.get('restore_bbox') or rec.get('bbox')
            if not rb_box or not _intersects(crop_box_cur, [float(v) for v in rb_box]):
                continue
            rid_up = rec.get('id')
            if rec.get('region_crop'):
                blocked_by_upper = True
                break
            if _contains(crop_box_cur, [float(v) for v in rb_box]):
                if rid_up and rid_up in fg_tight_masks:
                    mx0, my0, mw0, mh0, sub_m = fg_tight_masks[rid_up]
                    contained_fg[my0:my0 + mh0, mx0:mx0 + mw0] |= sub_m
                continue
            if rec.get('kind') == 'text' and rid_up and rid_up in fg_tight_masks:
                up_entry = fg_tight_masks[rid_up]
                if not _submask_any(up_entry, crop_box_cur):
                    # Only padded whitespace margin touched; actual glyph pixels are disjoint!
                    continue
                mx0, my0, mw0, mh0, _ = up_entry
                tight_u = [mx0 - 1, my0 - 1, mw0 + 2, mh0 + 2]
                shrunk_box, _ = _shrink_box_away_from_protected(
                    crop_box_cur, crop.get('seed_boxes', [crop_box_cur]), [_clip(tight_u, size)], size
                )
                if shrunk_box is not None and shrunk_box[2] >= 6 and shrunk_box[3] >= 6:
                    if not _submask_any(up_entry, shrunk_box):
                        crop_box_cur = shrunk_box
                        continue
            blocked_by_upper = True
            break
        if crop_box_cur != crop['box']:
            x, y, w, h = crop_box_cur
            crop_obj = ClosedObject(
                id=rid, kind='illustration', bbox=[x, y, w, h], layers=['source-plate.png'],
                components=['whole'], parts=[{'name': 'whole', 'bbox': [x, y, w, h], 'layers': ['source-plate.png']}],
                exclusions=[], replaces=[], required_owners=[],
                text_policy='intrinsic_label' if intrinsic else 'exclude_independent_text', intrinsic_text=intrinsic,
                description='opaque source region crop (no semantic claim)', prompt_en='opaque source region crop',
            )
            fused_try = fuse_incremental_scene(baseline, base_graph=graph, added_assets=[crop_obj])
            if fused_try.ownership.get(rid) != 'RESOLVED':
                report.append({'id': rid, 'state': 'SKIPPED', 'reason': f"host fusion ownership={fused_try.ownership.get(rid)}",
                               'bbox': [x, y, w, h], 'seed_ids': crop['seed_ids']})
                continue
        if blocked_by_upper:
            isolated = False
        else:
            # Also block if another already-staged crop in this pass intersects crop_box_cur.
            other_crops = [
                r.get('restore_bbox') or r.get('bbox')
                for r in graph.selected.values()
                if r.get('region_crop') or r.get('id') in staged
            ]
            isolated = not any(_intersects(crop_box_cur, [float(v) for v in ob]) for ob in other_crops if ob)

        pixels = appearance[y:y + h, x:x + w].copy()
        safety, backing_meta, backing = 'MOVE_UNSAFE', {'reason': 'not attempted'}, None
        device_owner = None
        flush_photo_used = False
        if not allow_movable:
            backing_meta = {'reason': 'movable crops disabled'}
        elif not isolated:
            backing_meta = {'reason': 'overlaps committed editable object'}
        else:
            other_planned = np.zeros(plate.shape[:2], bool)
            for j_c, cand_c in enumerate(crops):
                if j_c == idx_c or any(sid in skip for sid in cand_c['seed_ids']):
                    continue
                other_planned |= planned_rects[j_c]
            fit_protected = movable_mask | upgraded_mask | foreground | other_planned
            if crop['cuts']:
                backing_meta = {'reason': 'crop boundary cuts continuing content'}
            elif crop.get('device_frame'):
                from .embedded_device import fit_device_backing
                peers = [obj.bbox for obj in (*graph.text.texts, *graph.asset.objects)
                         if _intersection_area(obj.bbox, crop_box_cur) < .9 * obj.bbox[2] * obj.bbox[3]]
                try:
                    backing, device_owner, proof = fit_device_backing(source, [x, y, w, h], fit_protected, peers)
                except ValueError as exc:
                    backing_meta = {'reason': str(exc)}
                else:
                    if lower_covered[y:y+h, x:x+w][device_owner].any():
                        backing_meta = {'reason': 'device backing already claimed by a lower transaction'}
                    else:
                        backing_meta = {'background_evidence': proof}
                        safety = 'MOVABLE'
            else:
                try:
                    backing, error, evidence = backing_from_evidence(lower_comp, [x, y, w, h], fit_protected, graph.surface)
                    if np.isfinite(error) and error <= MOVABLE_RING_ERROR:
                        safety = 'MOVABLE'
                        backing_meta = {'ring_error': float(error), 'background_evidence': evidence}
                        if crop.get('screen_backing') and crop.get('fixed_frame'):
                            refined = _screen_backing_from_edges(lower_comp, [x,y,w,h], fit_protected, backing)
                            if refined is not None:
                                backing, proof = refined
                                backing_meta['background_evidence'] = {**proof, 'base': evidence}
                    else:
                        backing_meta = {'reason': 'surrounding backing not independently fitted', 'ring_error': float(error)}
                except ValueError as exc:
                    backing_meta = {'reason': f'backing unavailable: {exc}'}

            if safety != 'MOVABLE' and not crop.get('fixed_frame') and w >= 90 and h >= 70 and (set(crop.get('kinds', ())) & {'layout:image', 'asset', 'layout'}):
                flush = _detect_flush_card_photo(appearance, lower_comp, [x, y, w, h], fit_protected)
                if flush is not None:
                    f_box, f_backing, f_err = flush
                    if f_box != [x, y, w, h]:
                        fx, fy, fw, fh = f_box
                        f_obj = ClosedObject(
                            id=rid, kind='illustration', bbox=[fx, fy, fw, fh], layers=['source-plate.png'],
                            components=['whole'], parts=[{'name': 'whole', 'bbox': [fx, fy, fw, fh], 'layers': ['source-plate.png']}],
                            exclusions=[], replaces=[], required_owners=[],
                            text_policy='intrinsic_label' if intrinsic else 'exclude_independent_text', intrinsic_text=intrinsic,
                            description='opaque flush card photo crop (no semantic claim)', prompt_en='opaque flush card photo crop',
                        )
                        f_try = fuse_incremental_scene(baseline, base_graph=graph, added_assets=[f_obj])
                        if f_try.ownership.get(rid) == 'RESOLVED':
                            fused_try = f_try
                            x, y, w, h = fx, fy, fw, fh
                            crop_box_cur = [fx, fy, fw, fh]
                            pixels = appearance[y:y + h, x:x + w].copy()
                        else:
                            flush = None
                    if flush is not None:
                        safety = 'MOVABLE'
                        backing = f_backing
                        backing_meta = {'ring_error': float(f_err), 'background_evidence': {'method': 'flush-card-body-backing/1'}}
                        flush_photo_used = True

        if safety == 'MOVE_UNSAFE' and (not allow_move_unsafe or crop.get('device_frame')):
            report.append({
                'id': rid,
                'state': 'SKIPPED',
                'move_safety': 'MOVE_UNSAFE',
                'bbox': [x, y, w, h],
                'seed_ids': crop['seed_ids'],
                **backing_meta,
                'backing_reason': backing_meta.get('reason'),
                'reason': 'MOVE_UNSAFE_DISABLED',
            })
            continue

        png = output / f'{rid}.png'
        upgraded = None
        if safety == 'MOVABLE' and allow_transparent_upgrade and not flush_photo_used and not crop.get('fixed_frame'):
            badge_inner = _detect_icon_in_tinted_circle(
                pixels, backing, foreground[y:y + h, x:x + w], crop.get('kinds', []),
            )
            if badge_inner is not None:
                (ix0, iy0, iw, ih), inner_backing = badge_inner
                sub_up = _try_transparent_upgrade(
                    pixels[iy0:iy0 + ih, ix0:ix0 + iw],
                    inner_backing,
                    2.0,
                    foreground[y + iy0:y + iy0 + ih, x + ix0:x + ix0 + iw],
                    crop.get('kinds', []),
                )
                if sub_up is not None:
                    sub_rgba, (stx, sty, stw, sth), sub_method, sub_ratio = sub_up
                    backing[iy0:iy0 + ih, ix0:ix0 + iw] = inner_backing
                    upgraded = (sub_rgba, [ix0 + stx, iy0 + sty, stw, sth], sub_method, sub_ratio)
            if upgraded is None:
                upgraded = _try_transparent_upgrade(
                    pixels, backing, float(backing_meta['ring_error']), foreground[y:y + h, x:x + w], crop.get('kinds', []),
                )

        if upgraded is not None:
            tight_rgba, (tx0, ty0, tw, th), upgrade_method, ratio = upgraded
            tx, ty = x + tx0, y + ty0
            Image.fromarray(tight_rgba).save(png)
            graph = fused_try
            full_rect = rectangle([tx, ty, tw, th], size)
            owner_mask = full_rect & ~foreground
            local_owner = owner_mask[ty:ty + th, tx:tx + tw]
            clean_local = local_owner | (contained_fg[ty:ty + th, tx:tx + tw] & (tight_rgba[..., 3] > 200))
            backing_sub = np.clip(np.rint(backing[ty0:ty0 + th, tx0:tx0 + tw]), 0, 255).astype(np.uint8)
            proof = output / f'{rid}-owner.npz'
            np.savez_compressed(proof, mask=owner_mask, restore_bbox=[tx, ty, tw, th], backing_patch=backing_sub)
            lower_comp[ty:ty + th, tx:tx + tw][clean_local] = backing_sub[clean_local]
            plate_write = clean_local & ~lower_covered[ty:ty + th, tx:tx + tw]
            plate[ty:ty + th, tx:tx + tw][plate_write] = backing_sub[plate_write]
            upgraded_mask |= owner_mask
            record = {
                'id': rid, 'kind': 'asset', 'region_crop': False, 'move_safety': 'MOVABLE',
                'bbox': [tx, ty, tw, th], 'restore_bbox': [tx, ty, tw, th], 'seed_ids': crop['seed_ids'],
                'payload': {
                    'method': upgrade_method, 'upgraded_from_region_crop': True,
                    'tight_transparent_ratio': ratio, **backing_meta,
                },
                'state': 'STAGED', 'quality': 'REVIEW_REQUIRED', 'human_acceptance': False,
                'source_sha256': graph.source_sha256, 'original_owner': 'source-plate',
                'complete_owner_scope': True,
                'mask_sha256': array_sha(owner_mask.astype('uint8')),
                'owner_evidence': {'file': proof.name, 'sha256': digest(proof), 'method': upgrade_method},
                'asset_reference': {'path': png.name, 'sha256': digest(png), 'source_sha256': graph.source_sha256,
                                    'bbox': [tx, ty, tw, th], 'crop_offset': [tx, ty]},
            }
            graph.transactions.append(record)
            graph.selected[rid] = record
            staged.append(rid)
            report.append({
                'id': rid, 'state': 'STAGED', 'move_safety': 'MOVABLE', 'region_crop': False,
                'upgraded_to_transparent_asset': True, 'transparent_upgrade_method': upgrade_method,
                'tight_transparent_ratio': ratio, 'bbox': [tx, ty, tw, th],
                'seed_ids': crop['seed_ids'], **backing_meta,
            })
            continue

        round_meta: dict = {}
        out_pixels = pixels
        if safety == 'MOVABLE' and backing is not None and device_owner is None:
            rounded = _detect_and_prepare_rounded_photo(pixels, backing)
            if rounded is not None:
                out_pixels, est_r = rounded
                round_meta = {'corner_radius_px': est_r, 'clip_geometry': 'roundRect'}

        alpha = (device_owner.astype(np.uint8) * 255 if device_owner is not None
                 else np.full((h, w), 255, np.uint8))
        Image.fromarray(np.dstack([out_pixels, alpha])).save(png)
        graph = fused_try
        record = {
            'id': rid, 'kind': 'asset', 'region_crop': True, 'move_safety': safety,
            'bbox': [x, y, w, h], 'restore_bbox': [x, y, w, h], 'seed_ids': crop['seed_ids'],
            'payload': {'method': METHOD, 'boundary_cuts_content': crop['cuts'], **round_meta, **backing_meta},
            'state': 'STAGED', 'quality': 'REVIEW_REQUIRED', 'human_acceptance': False,
            'source_sha256': graph.source_sha256, 'original_owner': 'source-plate',
            'complete_owner_scope': safety == 'MOVABLE',
            'asset_reference': {'path': png.name, 'sha256': digest(png), 'source_sha256': graph.source_sha256,
                                'bbox': [x, y, w, h], 'crop_offset': [x, y]},
        }
        if safety == 'MOVABLE':
            owner_mask = rectangle([x, y, w, h], size) & ~foreground
            local_owner = owner_mask[y:y + h, x:x + w]
            if device_owner is not None:
                local_owner &= device_owner
            clean_local = (local_owner if device_owner is not None
                           else local_owner | contained_fg[y:y + h, x:x + w])
            backing_u8 = np.clip(np.rint(backing), 0, 255).astype(np.uint8)
            proof = output / f'{rid}-owner.npz'
            np.savez_compressed(proof, mask=owner_mask, restore_bbox=[x, y, w, h], backing_patch=backing_u8)
            record['mask_sha256'] = array_sha(owner_mask.astype('uint8'))
            method = (backing_meta['background_evidence']['method'] if device_owner is not None
                      else 'region-crop-lateral-backing/1')
            record['owner_evidence'] = {'file': proof.name, 'sha256': digest(proof), 'method': method}
            lower_comp[y:y + h, x:x + w][clean_local] = backing_u8[clean_local]
            # Layer-algebra invariant: never overwrite canvas plate pixels underneath already-erased surfaces/containers,
            # while cleaning dark-blue caption islands under wholly-contained upper text/asset owners.
            plate_write = clean_local & ~lower_covered[y:y + h, x:x + w]
            plate[y:y + h, x:x + w][plate_write] = backing_u8[plate_write]
            movable_mask |= owner_mask
        graph.transactions.append(record)
        graph.selected[rid] = record
        staged.append(rid)
        report.append({'id': rid, 'state': 'STAGED', 'move_safety': safety, 'region_crop': True,
                       'bbox': [x, y, w, h], 'seed_ids': crop['seed_ids'], **backing_meta})

    if not staged:
        shutil.rmtree(output, ignore_errors=True)
        return baseline / 'slideir.json', [], report

    any_erased = (movable_mask | upgraded_mask).any()
    if any_erased:
        Image.fromarray(np.dstack([plate, plate_alpha])).save(output / plate_file.name)
        graph.raster_backing = {**rb, 'sha256': digest(output / plate_file.name)}
        if (output / 'ownership.npz').is_file() and (output / 'ownership.json').is_file():
            with np.load(output / 'ownership.npz', allow_pickle=False) as data:
                ownership, uncertain = data['ownership'].copy(), data['uncertain_text'].copy()
            if movable_mask.any():
                ownership[movable_mask] = 5
            if upgraded_mask.any():
                ownership[upgraded_mask] = 0
            np.savez_compressed(output / 'ownership.npz', ownership=ownership, uncertain_text=uncertain)
            meta = json.loads((output / 'ownership.json').read_text('utf8'))
            meta['map_sha256'] = digest(output / 'ownership.npz')
            if movable_mask.any():
                meta.setdefault('classes', {})['5'] = 'region_crop'
            write(output / 'ownership.json', meta)

    write(output / 'region-crops.json', {'schema': 'RegionCrops/1', 'method': METHOD, 'records': report})
    ir = adapt(graph, output)
    write(output / 'scene.json', graph.model_dump())
    write(output / 'slideir.json', ir)
    return output / 'slideir.json', staged, report
