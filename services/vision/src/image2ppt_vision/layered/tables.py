"""Ruled-grid tables: long horizontal and vertical rules that intersect as a lattice.

Borderless layouts that merely look tabular are not tables here; they stay as
native text over the repaired background.
"""
import cv2
import numpy as np


def _rules(lines, axis, min_length):
    kernel = np.ones((1, min_length), np.uint8) if axis == 0 else np.ones((min_length, 1), np.uint8)
    opened = cv2.morphologyEx(lines, cv2.MORPH_OPEN, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if axis == 0 and h <= 5:
            out.append((y+h/2, x, x+w))
        elif axis == 1 and w <= 5:
            out.append((x+w/2, y, y+h))
    return out


def _cluster(values, tol=4):
    values = sorted(values)
    groups = []
    for v in values:
        if groups and v-groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


def detect_tables(rgb, exclude=None):
    """Return lattice dicts ``{bbox, xs, ys}`` in source pixels."""
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    local = cv2.medianBlur(gray, 15).astype(np.int16)
    lines = (np.abs(gray.astype(np.int16)-local) > 18).astype(np.uint8)
    if exclude is not None:
        lines[exclude] = 0
    horizontal = _rules(lines, 0, max(40, W//25))
    vertical = _rules(lines, 1, max(30, H//25))
    if len(horizontal) < 3 or len(vertical) < 3:
        return []
    tables = []
    for hy, hx0, hx1 in horizontal:
        cols = [v for v in vertical if v[1]-4 <= hy <= v[2]+4 and hx0-4 <= v[0] <= hx1+4]
        if len(cols) < 3:
            continue
        x0, x1 = min(v[0] for v in cols), max(v[0] for v in cols)
        y0, y1 = min(v[1] for v in cols), max(v[2] for v in cols)
        if any(abs(t['bbox'][0]-x0) < 6 and abs(t['bbox'][1]-y0) < 6 for t in tables):
            continue
        rows = [h for h in horizontal if y0-4 <= h[0] <= y1+4 and h[1] <= x0+6 and h[2] >= x1-6]
        xs, ys = _cluster([v[0] for v in cols]), _cluster([h[0] for h in rows])
        if len(xs) < 3 or len(ys) < 3:
            continue
        hits = sum(any(abs(v[0]-x) <= 4 and v[1]-4 <= y <= v[2]+4 for v in cols) for x in xs for y in ys)
        if hits < .8*len(xs)*len(ys):
            continue
        tables.append({'bbox': [int(xs[0]), int(ys[0]), int(xs[-1]-xs[0])+1, int(ys[-1]-ys[0])+1],
                       'xs': xs, 'ys': ys})
    return tables


def fill_cells(rgb, table, lines, records):
    """Assign OCR lines to cells and measure each cell's fill colour."""
    xs, ys = table['xs'], table['ys']
    rows = [['' for _ in xs[:-1]] for _ in ys[:-1]]
    owned, styles, first = [], [], {}
    for line in sorted(lines, key=lambda t: (t['bbox'][1], t['bbox'][0])):
        x, y, w, h = line['bbox']
        cx, cy = x+w/2, y+h/2
        c = next((i for i in range(len(xs)-1) if xs[i] <= cx < xs[i+1]), None)
        r = next((i for i in range(len(ys)-1) if ys[i] <= cy < ys[i+1]), None)
        if c is None or r is None:
            continue
        rows[r][c] = (rows[r][c]+'\n' if rows[r][c] else '')+line['text']
        owned.append(line['id'])
        first.setdefault((r, c), line['id'])
    for r in range(len(ys)-1):
        for c in range(len(xs)-1):
            a, b = int(xs[c])+3, int(xs[c+1])-3
            top, bottom = int(ys[r])+3, int(ys[r+1])-3
            if b <= a or bottom <= top:
                continue
            fill = np.median(rgb[top:bottom, a:b].reshape(-1, 3), 0)
            style = {'row': r, 'column': c, 'fill': ''.join(f'{int(v):02X}' for v in fill)}
            record = records.get(first.get((r, c)), {})
            if record.get('color'):
                style['color'] = record['color']
            styles.append(style)
    return rows, owned, styles
