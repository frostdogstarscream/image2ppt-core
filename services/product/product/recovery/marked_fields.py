"""Locate independent caption fields using both OCR wording and the source separator."""

import re
import unicodedata

import numpy as np

from ..parallel_typography import measure_text_appearance


class FieldWidthMismatch(ValueError):
    """A uniform-width title field occupies a different number of source glyphs."""


def separated_fields(source, text, bbox):
    parts = re.fullmatch(r'([^|丨｜\n]+)([|丨｜])([^|丨｜\n]+)', text)
    if not parts or min(len(parts[1].strip()), len(parts[3].strip())) < 2:
        return []
    x, y, w, h = map(round, bbox)
    patch = source[y : y + h, x : x + w].astype(float)
    if min(w, h) < 8 or patch.shape[:2] != (h, w):
        return []
    background = np.median(patch.reshape(-1, 3), axis=0)
    ink = np.abs(patch - background).max(2) > 65
    transitions = np.diff(np.pad(ink.any(0).astype(int), 1))
    runs = list(zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)))
    matches = []
    for previous, (left, right), following in zip(runs, runs[1:], runs[2:]):
        if (
            right - left <= max(2, 0.10 * h)
            and ink[:, left:right].sum(0).max() >= 0.52 * h
            and min(left - previous[1], following[0] - right) >= max(3, 0.20 * h)
        ):
            matches.append((int(right), int(following[0])))
    if len(matches) != 1:
        return []
    end, start = matches[0]
    left_text, right_text = parts[1].strip(), parts[3].strip()
    if min(len(left_text), len(right_text)) >= 3 and all(
        unicodedata.east_asian_width(ch) in ('W', 'F') for ch in left_text + right_text
    ):
        left_ink = np.flatnonzero(ink[:, : end - 3].any(0))
        right_ink = np.flatnonzero(ink[:, start:].any(0))
        if left_ink.size and right_ink.size:
            pitch = (np.ptp(left_ink) + 1) / len(left_text)
            if abs((np.ptp(right_ink) + 1) / pitch - len(right_text)) > 0.65:
                raise FieldWidthMismatch('OCR character count and source field width disagree')
    boxes = [[x, y, end + 1, h], [x + start - 1, y, w - start + 1, h]]
    fields = []
    for value, box in zip((parts[1] + parts[2], parts[3]), boxes):
        appearance = measure_text_appearance(source, box)
        if not appearance:
            return []
        fields.append((value, box, appearance['color_hex']))
    return fields


def split_items(source, items, preserved):
    result = []
    for item in items:
        try:
            fields = (
                separated_fields(source, item['text'], item['bbox'])
                if len(item['ocr_line_ids']) == 1
                else []
            )
        except FieldWidthMismatch as exc:
            preserved.append(
                {
                    'id': item['id'],
                    'ocr_ids': item['ocr_line_ids'],
                    'bbox': item['bbox'],
                    'reason': str(exc),
                }
            )
            continue
        if not fields:
            result.append(item)
            continue
        for index, (value, box, color) in enumerate(fields):
            rgb = list(bytes.fromhex(color))
            result.append(
                {
                    **item,
                    'id': item['id'] + ('-detail' if index else ''),
                    'bbox': box,
                    'text': value,
                    'color_hex': color,
                    'rgb': rgb,
                }
            )
    return result


def striped_backing(source, box, protected, *, max_span: float = 18.0):
    """Fit a horizontal ramp independently on each scanline, with held-out columns."""
    x, y, w, h = box
    margin = 7
    if x < margin or x + w + margin > source.shape[1] or w < 16:
        return None
    left = source[y : y + h, x - margin : x].astype(float)
    right = source[y : y + h, x + w : x + w + margin].astype(float)
    blocked = protected[y : y + h]
    if blocked[:, x - margin : x].any() or blocked[:, x + w : x + w + margin].any():
        return None
    a = np.median(left[:, ::2], axis=1)[:, None, :]
    b = np.median(right[:, ::2], axis=1)[:, None, :]
    if np.max(np.abs(a - b)) > max_span:
        return None
    errors = np.concatenate((np.abs(left[:, 1::2] - a), np.abs(right[:, 1::2] - b)), axis=1).max(2)
    p95 = float(np.percentile(errors, 95))
    if p95 > 3 or errors.max() > 8:
        return None
    ramp = (np.arange(w) + 4) / (w + 7)
    reference = source.copy()
    reference[y : y + h, x : x + w] = np.rint(a + (b - a) * ramp[None, :, None]).astype('uint8')
    meta = {
        'method': 'scanline-lateral',
        'validation_p95_rgb': p95,
        'validation_pixels': int(errors.size),
        'margin': margin,
    }
    if max_span != 18.0:
        meta['max_span'] = float(max_span)
    return reference, meta
