"""Glyph-level ink masks for OCR lines.

An OCR box is a search window, not an erase region: it routinely contains an
adjacent icon or tile. Only connected ink that behaves like glyphs of this line
is erased; solid, oversized or off-colour end components are reported as icon
candidates and left in the page for segmentation.
"""
import unicodedata

import cv2
import numpy as np


def _hex(rgb):
    return ''.join(f'{int(v):02X}' for v in np.clip(np.rint(rgb), 0, 255))


def _lab(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def _surface(lab, ring):
    """Robust planar background in Lab from border-ring samples."""
    h, w = lab.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    design = np.stack([xx[ring], yy[ring], np.ones(int(ring.sum()))], 1).astype(np.float64)
    values = lab[ring].astype(np.float64)
    keep = np.ones(len(values), bool)
    coef = None
    for _ in range(3):
        if keep.sum() < 12:
            break
        coef, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        residual = np.linalg.norm(values-design@coef, axis=1)
        keep = residual <= max(4., 2.5*float(np.median(residual[keep])))
    if coef is None:
        return np.broadcast_to(np.median(values, 0), lab.shape).astype(np.float32), 99.
    plane = np.stack([xx, yy, np.ones_like(xx)], -1).reshape(-1, 3)@coef
    residual = np.linalg.norm(values-design@coef, axis=1)
    return plane.reshape(h, w, 3).astype(np.float32), float(np.percentile(residual, 80))


def _cluster_surface(lab, ring):
    """Fallback for a ring crossing several colours: the cluster owning most ring pixels."""
    samples = lab.reshape(-1, 3)
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, .5)
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(lab.shape[:2])
    owner = np.bincount(labels[ring], minlength=k).argmax()
    return np.broadcast_to(centers[owner], lab.shape).astype(np.float32)


def _otsu(values):
    values = np.clip(values, 0, 255).astype(np.uint8)
    if values.size < 16:
        return 20.
    threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return float(threshold)


def _stroke(mask):
    area = float(mask.sum())
    if area < 4:
        return 0.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return 2*area/max(perimeter, 1.)


def _units(text):
    return sum(.3 if c.isspace() else 1. if unicodedata.east_asian_width(c) in 'WF' else .55 for c in text)


def split_icon_text(text, ink, icons):
    """OCR often reads an adjacent icon as 1-3 characters. Attribute leading or
    trailing characters to the icon when the remaining text alone explains the
    measured glyph-ink width far better (one em ~ ink height for CJK lines)."""
    em = max(1., ink[3])
    lead = any(b[0]+b[2] <= ink[0]+2 for b in icons)
    trail = any(b[0] >= ink[0]+ink[2]-2 for b in icons)
    best, base = (0, 0), abs(_units(text)*em-ink[2])
    error = base
    def strippable(part):
        return all(not (unicodedata.east_asian_width(c) in 'WF' and c.isalnum()) for c in part)

    for k in range(0, 4 if lead else 1):
        for t in range(0, 4 if trail else 1):
            if k+t >= len(text) or not strippable(text[:k]) or not strippable(text[len(text)-t:]):
                continue
            e = abs(_units(text[k:len(text)-t])*em-ink[2])
            if e < error-1e-6:
                best, error = (k, t), e
    if best == (0, 0) or error > base-.5*em:
        return None
    k, t = best
    return {'lead': text[:k], 'native': text[k:len(text)-t], 'trail': text[len(text)-t:] if t else ''}


def _leading_icon(text, groups, h):
    """A low-confidence line whose first OCR characters are not CJK and whose
    leading glyph groups are explained by those characters' absence: the OCR
    read a same-colour icon as text (e.g. '8。' for a people icon)."""
    if len(groups) < 2 or len(text) < 4:
        return None
    k = 0
    while k < 2 and not (unicodedata.east_asian_width(text[k]) in 'WF' and text[k].isalnum()):
        k += 1
    if not k:
        return None
    first, second = groups[0], groups[1]
    gap = second['box'][0]-(first['box'][0]+first['box'][2])
    w, hh = first['box'][2:]
    if gap >= .2*h and .7*h <= hh <= 1.4*h and .6*h <= w <= 1.6*h and first['fill'] >= .25:
        return {'lead': text[:k], 'native': text[k:].lstrip(), 'trail': '', 'groups': 1}
    return None


def _strippable(c):
    return not (unicodedata.east_asian_width(c) in 'WF' and c.isalnum())


def _unread_end_icons(text, groups, icons, h, em):
    """An end mark the OCR never transcribed (a pale icon in front of a label) is
    still ink. Take it as an icon when the OCR text explains the ink width
    clearly better without it; symbols already attributed to icons are not text."""
    native = text.strip()
    if groups and any(m in icons for m in groups[-1]['members']):
        native = native.rstrip(''.join(c for c in native if _strippable(c)))
    if groups and any(m in icons for m in groups[0]['members']):
        native = native.lstrip(''.join(c for c in native if _strippable(c)))
    rest = [g for g in groups if not any(m in icons for m in g['members'])]
    if len(rest) < 2 or not native:
        return
    expected = _units(native)*em
    for end, inner in ((rest[0], rest[1]), (rest[-1], rest[-2])):
        gap = inner['box'][0]-(end['box'][0]+end['box'][2]) if end is rest[0] else end['box'][0]-(inner['box'][0]+inner['box'][2])
        if gap < .35*h or end['box'][2] > 1.6*h or max(end['box'][2:]) < .45*h:
            continue
        rest_now = [g for g in rest if not any(m in icons for m in g['members'])]
        width = rest_now[-1]['box'][0]+rest_now[-1]['box'][2]-rest_now[0]['box'][0]
        without = width-end['box'][2]-gap
        if abs(without-expected) < .5*em and abs(width-expected)-abs(without-expected) > .5*em:
            icons.update(end['members'])


def _glyph_groups(comps, h):
    """Merge components whose x-ranges overlap or nearly touch (radicals of one glyph)."""
    groups = []
    for c in sorted(comps, key=lambda c: c['box'][0]):
        x, y, w, hh = c['box']
        if groups and x <= groups[-1]['x1']+.08*h:
            g = groups[-1]
            g['x1'] = max(g['x1'], x+w); g['y0'] = min(g['y0'], y); g['y1'] = max(g['y1'], y+hh)
            g['members'].append(c['i']); g['area'] += c['area']; g['colors'].append((c['color'], c['area']))
        else:
            groups.append({'x0': x, 'x1': x+w, 'y0': y, 'y1': y+hh, 'members': [c['i']], 'area': c['area'],
                           'colors': [(c['color'], c['area'])]})
    for g in groups:
        g['box'] = [g['x0'], g['y0'], g['x1']-g['x0'], g['y1']-g['y0']]
        g['fill'] = g['area']/max(1, g['box'][2]*g['box'][3])
        g['color'] = np.average(np.array([c for c, _ in g['colors']]), axis=0,
                                weights=np.array([a for _, a in g['colors']], float))
    return groups


def analyze_line(rgb, line, text_mask, icon_mask):
    """Measure one OCR line; ORs its erase pixels into ``text_mask`` (page-sized bool).

    Returns a JSON-safe record. ``status`` other than ``ok`` means the line's
    pixels were not claimed and must stay in the background.
    """
    H, W = rgb.shape[:2]
    x, y, w, h = map(float, line['bbox'])
    record = {'id': line['id'], 'status': 'ok', 'ocr_bbox': [round(v, 2) for v in (x, y, w, h)],
              'icon_candidates': []}
    if w < 3 or h < 4:
        record['status'] = 'degenerate_box'
        return record
    pad = max(3, int(round(h*.18)))
    x0, y0 = max(0, int(np.floor(x))-pad), max(0, int(np.floor(y))-pad)
    x1, y1 = min(W, int(np.ceil(x+w))+pad), min(H, int(np.ceil(y+h))+pad)
    crop = rgb[y0:y1, x0:x1]
    lab = _lab(crop)
    ring = np.zeros(lab.shape[:2], bool)
    band = max(2, pad//2)
    ring[:band] = ring[-band:] = True
    ring[:, :band] = ring[:, -band:] = True
    surface, error = _surface(lab, ring)
    method = 'planar_ring'
    if error > 10:
        surface, method = _cluster_surface(lab, ring), 'ring_cluster'
    delta = np.linalg.norm(lab-surface, axis=2)
    bx0, by0 = int(np.floor(x))-x0, int(np.floor(y))-y0
    bx1, by1 = int(np.ceil(x+w))-x0, int(np.ceil(y+h))-y0
    inside = np.zeros_like(ring)
    inside[max(0, by0):by1, max(0, bx0):bx1] = True
    threshold = min(60., max(14., _otsu(delta[inside])*.8))
    record.update(background_method=method, background_error=round(error, 2), threshold=round(threshold, 2),
                  background_color=_hex(cv2.cvtColor(np.clip(np.median(surface[ring], 0), 0, 255)
                                                     .astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2RGB)[0, 0]))
    def extract(strong):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(strong.astype(np.uint8), 8)
        comps = []
        for i in range(1, count):
            sx, sy, sw, sh, area = (int(v) for v in stats[i])
            if area < 2:
                continue
            ox = max(0, min(sx+sw, bx1)-max(sx, bx0)); oy = max(0, min(sy+sh, by1)-max(sy, by0))
            if ox*oy == 0:
                continue
            member = labels == i
            if ox*oy < .5*sw*sh and (sx == 0 or sy == 0 or sx+sw == crop.shape[1] or sy+sh == crop.shape[0]):
                continue
            comps.append({'i': i, 'box': [sx, sy, sw, sh], 'area': area, 'fill': area/max(1, sw*sh),
                          'color': np.median(crop[member], 0)})
        return labels, comps

    labels, comps = extract(delta > threshold)
    keep = np.ones(delta.shape, bool)
    # A badge (number in a circle, label on a pill) ends up as one solid blob
    # covering the OCR box, with the glyphs as its holes: measure inside it.
    badge = next((c for c in comps if c['fill'] >= .5 and c['box'][2] >= .9*(bx1-bx0)
                  and c['box'][3] >= .9*(by1-by0)), None)
    if badge is not None:
        member = (labels == badge['i']).astype(np.uint8)
        contours, _ = cv2.findContours(member, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(member)
        cv2.drawContours(filled, contours, -1, 1, -1)
        keep = cv2.erode(filled, np.ones((3, 3), np.uint8), iterations=2) > 0
        badge_lab = np.median(lab[member > 0], 0)
        delta = np.where(keep, np.linalg.norm(lab-badge_lab, axis=2), 0)
        threshold = min(60., max(14., _otsu(delta[keep])*.8)) if keep.sum() >= 16 else 60.
        labels, comps = extract(delta > threshold)
        sx, sy, sw, sh = badge['box']
        record.update(badge={'bbox': [sx+x0, sy+y0, sw, sh], 'color': _hex(badge['color'])},
                      threshold=round(threshold, 2))
    if not comps:
        record['status'] = 'no_ink'
        return record
    glyph = np.median([c['box'][3] for c in comps if c['area'] >= 6] or [h*.8])
    groups = _glyph_groups(comps, h)
    core = [g for g in (groups[1:-1] if len(groups) >= 4 else groups) if g['fill'] < .82] or groups
    weights = np.array([g['box'][2] for g in core], float)
    main = np.average(np.array([g['color'] for g in core]), axis=0, weights=weights)
    main_lab = _lab(main.reshape(1, 1, 3))[0, 0]
    tall = np.median([g['box'][3] for g in groups])
    span = groups[-1]['box'][0]+groups[-1]['box'][2]-groups[0]['box'][0]
    icons = set()
    # Glyphs inside a line are never icons; only narrow end groups separated by
    # a gap that are off-colour, tile-solid or oversized go to segmentation.
    for ordered in (groups, groups[::-1]):
        outer = None
        for k in range(min(2, len(ordered)-1)):
            first, second = ordered[k], ordered[k+1]
            if first['box'][2] > .4*span or any(m in icons for m in first['members']):
                break
            gap = (second['box'][0]-(first['box'][0]+first['box'][2]) if ordered is groups
                   else first['box'][0]-(second['box'][0]+second['box'][2]))
            first_lab = _lab(first['color'].reshape(1, 1, 3))[0, 0]
            far = np.linalg.norm(first_lab-main_lab) > 35
            solid = first['fill'] >= .82 and min(first['box'][2:]) >= .55*h
            oversized = first['box'][3] > max(1.3*tall, 1.15*h)
            # A run of repeated marks ('>>') shares the colour of the outer one already taken.
            repeat = outer is not None and far and np.linalg.norm(first_lab-outer) < 20
            if max(first['box'][2:]) >= .45*h and (gap >= .35*h and far or gap >= .06*h and (solid or oversized or repeat)):
                icons.update(first['members'])
                outer = first_lab
            else:
                break
    _unread_end_icons(line['text'], groups, icons, h, glyph)
    lead_split = None
    if not icons and line.get('score', 1) < .95:
        lead_split = _leading_icon(line['text'], groups, h)
        if lead_split:
            for g in groups[:lead_split.pop('groups')]:
                icons.update(g['members'])
    text_ids = [c['i'] for c in comps if c['i'] not in icons]
    if not text_ids:
        record['status'] = 'icon_only'
    glyphs = np.isin(labels, text_ids)
    icon_pixels = np.isin(labels, list(icons))
    if icons:
        merged = cv2.dilate(icon_pixels.astype(np.uint8), np.ones((5, 5), np.uint8))
        n, _, s, _ = cv2.connectedComponentsWithStats(merged, 8)
        record['icon_candidates'] = [[int(s[j][0])+x0, int(s[j][1])+y0, int(s[j][2]), int(s[j][3])] for j in range(1, n)]
        icon_mask[y0:y1, x0:x1] |= cv2.dilate(icon_pixels.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if not text_ids:
        return record
    weak = delta > max(6., .35*threshold)
    fringe = weak & (cv2.dilate(glyphs.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    erase = cv2.dilate((glyphs | fringe).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    erase &= ~(cv2.dilate(icon_pixels.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    if badge is not None:
        erase &= cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ys, xs = np.where(glyphs)
    ink = [int(xs.min())+x0, int(ys.min())+y0, int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]
    if ink[3] > 1.25*h or ink[2]*ink[3] > 1.6*w*h:
        # The "ink" is a shape around the text (an undetected badge), not glyphs: leave it all in place.
        record.update(status='ink_exceeds_box', ink_bbox=ink)
        return record
    text_mask[y0:y1, x0:x1] |= erase
    core_pixels = glyphs & (delta > threshold*1.3)
    color = np.median(crop[core_pixels if core_pixels.sum() >= 8 else glyphs], 0)
    stroke = _stroke(glyphs)
    if lead_split:
        record['icon_text_split'] = lead_split
    elif record['icon_candidates']:
        split = split_icon_text(line['text'], ink, record['icon_candidates'])
        if split:
            record['icon_text_split'] = split
    record.update(ink_bbox=ink, color=_hex(color), stroke_px=round(stroke, 3),
                  bold_ratio=round(stroke/max(1, ink[3]), 4), glyph_height=round(float(glyph), 2),
                  erased_pixels=int(erase.sum()), contrast=round(float(np.median(delta[glyphs])), 2))
    return record


def _real(part):
    return sum(c.isalnum() for c in part)


def split_wide_gaps(rgb, lines, *, gap_em=.6):
    """OCR joins phrases set apart by a wide blank ('A    B') into one line and
    drops the blank. Split where an ink-free run of >= gap_em x OCR box height
    falls on a character boundary, so each phrase keeps its own position."""
    out = []
    for line in lines:
        text = line['text'].strip()
        x, y, w, h = (int(round(v)) for v in line['bbox'])
        crop = rgb[max(0, y):y+h, max(0, x):x+w]
        if len(text) < 4 or crop.size == 0 or h < 6:
            out.append(line); continue
        lab = _lab(crop)
        ring = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        ink = np.linalg.norm(lab-np.median(ring, 0), axis=2) > 25
        cols = np.where(ink.sum(0) > 0)[0]
        if len(cols) < 2:
            out.append(line); continue
        x0, x1 = int(cols[0]), int(cols[-1])+1
        empty = ~ink[:, x0:x1].any(0)
        runs, start = [], None
        for i, e in enumerate(np.append(empty, False)):
            if e and start is None:
                start = i
            elif not e and start is not None:
                if i-start >= gap_em*h:
                    runs.append((start, i))
                start = None
        units = [_units(c) for c in text]
        blank = sum(b-a for a, b in runs)
        scale = (x1-x0-blank)/max(sum(units), 1e-6)
        bounds = np.cumsum(units)*scale
        cuts, skipped = [], 0
        for a, b in runs:
            # The run swallows the missing blank, so the left phrase ends at its start.
            k = int(np.argmin(np.abs(bounds[:-1]-(a-skipped))))
            # Fullwidth punctuation carries its own blank half-em; that is not a phrase gap.
            punct = unicodedata.category(text[k])[0] == 'P' or unicodedata.category(text[k+1])[0] == 'P'
            if (not punct and abs(bounds[k]-(a-skipped)) <= .6*h
                    and _real(text[:k+1]) >= 2 and _real(text[k+1:]) >= 2):
                cuts.append((k+1, x0+a, x0+b))
            skipped += b-a
        if not cuts:
            out.append(line); continue
        edges = [(0, x+x0)] + [(k, x+bx) for k, _, bx in cuts]
        ends = [x+ax for _, ax, _ in cuts] + [x+x1]
        for (k, left), right, nxt in zip(edges, ends, [k for k, _, _ in cuts]+[len(text)]):
            part = text[k:nxt].strip()
            if part:
                bbox = [float(left), float(y), float(right-left), float(h)]
                out.append(line | {'text': part, 'bbox': bbox, 'split_from': line['text'],
                                   'polygon': [[left, y], [right, y], [right, y+h], [left, y+h]]})
    return out


def analyze_page(rgb, lines):
    H, W = rgb.shape[:2]
    text_mask = np.zeros((H, W), bool)
    icon_mask = np.zeros((H, W), bool)
    records = [analyze_line(rgb, line, text_mask, icon_mask) for line in lines]
    text_mask &= ~icon_mask
    return records, text_mask, icon_mask
