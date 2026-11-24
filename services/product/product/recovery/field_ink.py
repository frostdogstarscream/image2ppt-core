"""Match source glyphs to OCR characters before assigning an adjacent graphic or a second colour."""
import unicodedata

import numpy as np

from ..contracts import intersection


def _glyph_columns(source, text, bbox):
    # Equal advances give independent evidence of the character count. With
    # proportional text, a touching glyph, or a textured face, keep the OCR box.
    if (not 2 <= len(text) <= 160 or not any(ch.isalnum() for ch in text)
            or any(unicodedata.east_asian_width(ch) not in ('W', 'F') or ch.isspace() for ch in text)):
        return []
    x, y, w, h = map(round, bbox)
    if min(w, h) < 4 or min(x, y) < 0 or x+w > source.shape[1] or y+h > source.shape[0]:
        return []
    tile = source[y:y+h, x:x+w].astype(float)
    border = np.concatenate((tile[0], tile[-1]))
    paper = np.median(border, axis=0)
    if np.percentile(abs(border - paper).max(1), 85) > 25:
        return []
    contrast = abs(tile - paper).max(2)
    ink = contrast > 58
    columns = np.flatnonzero(ink.any(0))
    if not columns.size:
        return []
    gaps = np.diff(columns)
    breaks = np.flatnonzero(gaps > 1)
    if len(breaks) < len(text)-1:
        return []
    breaks = sorted(breaks[np.argsort(-gaps[breaks], kind='stable')[:len(text)-1]] + 1)
    glyphs = np.split(columns, breaks)
    letters = [(i, cols) for i, (ch, cols) in enumerate(zip(text, glyphs)) if ch.isalnum()]
    if len(letters) < 2:
        return []
    positions = np.array([i for i, _ in letters])
    centers = np.array([(cols[0]+cols[-1])/2 for _, cols in letters])
    pitch, origin = np.polyfit(positions, centers, 1)
    if (not .40*h <= pitch <= 1.8*h or np.max(abs(centers-origin-pitch*positions)) > .30*pitch
            or any(not .25*pitch <= cols[-1]-cols[0]+1 <= 1.20*pitch for _, cols in letters)):
        return []
    result = []
    for ch, cols in zip(text, glyphs):
        left, right = int(cols[0]), int(cols[-1])+1
        strength = contrast[:, left:right]
        active = ink[:, left:right]
        strong = active & (strength >= np.percentile(strength[active], 72))
        colours = tile[:, left:right][strong]
        rgb = np.median(colours, axis=0)
        if np.percentile(abs(colours-rgb).max(1), 85) > 42:
            return []
        result.append((ch, x+left, x+right, rgb))
    return result


def colour_runs(source, text, bbox):
    """Return text, source columns and ink colour only for a corroborated multicolour line."""
    glyphs = _glyph_columns(source, text, bbox)
    if not glyphs:
        return []
    palette = []
    indices = []
    for _, _, _, rgb in glyphs:
        distances = [abs(rgb-np.median(samples, axis=0)).max() for samples in palette]
        best = int(np.argmin(distances)) if distances else 0
        if not distances or distances[best] > 44:
            if len(palette) == 3:
                return []
            best = len(palette)
            palette.append([])
        palette[best].append(rgb)
        indices.append(best)
    if len(palette) < 2:
        return []
    colors = [np.median(samples, axis=0).round().astype('uint8') for samples in palette]
    # Require a repeatable colour on characters, not a single noisy pixel or punctuation mark.
    if any(sum(ch.isalnum() and k == i for (ch, *_), k in zip(glyphs, indices)) < 2
           for i in range(len(palette))):
        return []
    cuts = [0, *(i for i in range(1, len(text)) if indices[i] != indices[i-1]), len(text)]
    return [(text[a:b], glyphs[a][1], glyphs[b-1][2], bytes(colors[indices[a]]).hex().upper())
            for a, b in zip(cuts, cuts[1:])]


def separate_edge_graphic(source, line, proposals):
    """A compact detection may be outside the *text* even though OCR enclosed both."""
    x, y, w, h = map(round, line['bbox'])
    if not proposals or min(w,h) < 6 or min(x,y) < 0 or x+w > source.shape[1] or y+h > source.shape[0]:
        return None
    tile = source[y:y+h, x:x+w].astype(float)
    paper = np.median(np.concatenate((tile[0], tile[-1])), axis=0)
    contrast = abs(tile-paper).max(2)
    witnesses = {}
    for obj in proposals:
        bx, by, bw, bh = obj['bbox']
        if (not .45*h <= bh <= 1.5*h or not .45 <= bw/bh <= 1.8 or bw > .40*w
                or intersection(line['bbox'], obj['bbox']) < .85*bw*bh):
            continue
        at_left = bx+bw/2 < x+w/2
        outer_gap = abs(bx-x) if at_left else abs(x+w-bx-bw)
        if outer_gap > .35*h:
            continue
        edge = round(bx+bw-x if at_left else bx-x)
        a, b = (edge, min(w, edge+h)) if at_left else (max(0, edge-h), edge)
        if a < 0 or b > w or b <= a:
            continue
        quiet = contrast[:, a:b].max(0) < 22
        gaps = np.flatnonzero(np.diff(np.r_[False, quiet, False])).reshape(-1, 2)
        gaps = [(int(l+a),int(r+a)) for l,r in gaps if r-l >= max(2,.12*h)]
        if not gaps:
            continue
        start, end = gaps[0] if at_left else gaps[-1]
        if (start-edge if at_left else edge-end) > .15*h:
            continue
        left, right = (end, w) if at_left else (0, start)
        remaining = contrast[:, left:right] > 55
        yy, xx = np.where(remaining)
        if not len(xx):
            continue
        # Retain the original vertical OCR context for the glyph-count proof.
        glyphs = _glyph_columns(source, line['text'], [x+left,y,right-left,h])
        if not glyphs:
            continue
        pad = max(1, round(h * .05))
        x0, x1 = max(left, left+int(xx.min())-pad), min(right, left+int(xx.max())+1+pad)
        y0, y1 = max(0, int(yy.min())-pad), min(h, int(yy.max())+1+pad)
        box = [x+x0,y+y0,x1-x0,y1-y0]
        witnesses[tuple(box)] = {'bbox':box, 'graphic':{'id':obj['id'], 'bbox':list(obj['bbox'])}}
    return next(iter(witnesses.values())) if len(witnesses) == 1 else None
