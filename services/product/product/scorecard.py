"""Independent page scorecard for delivered PPTX files.

The scorecard inspects the delivered OOXML package directly (no SlideIR or
SceneGraph trust), optionally compares an actual Office render with the source
image, and optionally measures native text against an independent OCR
inventory. It produces measurements only; it never grants erase authority or
changes any product decision.

Ownership classes per visible pixel (top-most object in z order wins, transparent
picture pixels fall through):
  text        native text frame (frame area, not ink)
  shape       native non-text shape
  table/chart native graphic frame
  asset       picture with meaningful transparency (movable graphic)
  opaque_picture opaque picture smaller than the page (region crop or opaque container layer)
  plate       page-sized picture (source-plate / raster backing)
  background  slide background only
"""
from __future__ import annotations

from difflib import SequenceMatcher
import io
import json
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np
from PIL import Image

SCHEMA = 'PageScorecard/1'
NS = {
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'rel': 'http://schemas.openxmlformats.org/package/2006/relationships',
}
CLASSES = ('background', 'plate', 'opaque_picture', 'asset', 'shape', 'table', 'chart', 'text')
EDITABLE = ('opaque_picture', 'asset', 'shape', 'table', 'chart', 'text')
PLATE_AREA_FRACTION = 0.90
TRANSPARENT_FRACTION = 0.02


def _q(tag: str) -> str:
    prefix, name = tag.split(':')
    return '{%s}%s' % (NS[prefix], name)


def _resolve(base_folder: str, target: str) -> str:
    if target.startswith('/'):
        return posixpath.normpath(target.lstrip('/'))
    return posixpath.normpath(posixpath.join(base_folder, target))


def _slide_parts(archive: zipfile.ZipFile) -> list[str]:
    pres = ET.fromstring(archive.read('ppt/presentation.xml'))
    rels = ET.fromstring(archive.read('ppt/_rels/presentation.xml.rels'))
    targets = {r.get('Id'): r.get('Target') for r in rels.findall('rel:Relationship', NS)}
    parts = []
    id_list = pres.find('p:sldIdLst', NS)
    for sld in (list(id_list) if id_list is not None else []):
        target = targets[sld.get(_q('r:id'))]
        parts.append(_resolve('ppt', target))
    return parts


def _slide_size(archive: zipfile.ZipFile) -> tuple[int, int]:
    pres = ET.fromstring(archive.read('ppt/presentation.xml'))
    size = pres.find('p:sldSz', NS)
    return int(size.get('cx')), int(size.get('cy'))


def _part_rels(archive: zipfile.ZipFile, part: str) -> dict[str, str]:
    folder, name = posixpath.split(part)
    rel_file = posixpath.join(folder, '_rels', name + '.rels')
    if rel_file not in archive.namelist():
        return {}
    rels = ET.fromstring(archive.read(rel_file))
    return {r.get('Id'): _resolve(folder, r.get('Target'))
            for r in rels.findall('rel:Relationship', NS)}


def _xfrm(element) -> tuple[float, float, float, float] | None:
    xfrm = None
    for path in ('p:spPr/a:xfrm', 'p:xfrm', 'p:grpSpPr/a:xfrm'):
        xfrm = element.find(path, NS)
        if xfrm is not None:
            break
    if xfrm is None:
        return None
    off, ext = xfrm.find('a:off', NS), xfrm.find('a:ext', NS)
    if off is None or ext is None:
        return None
    return float(off.get('x')), float(off.get('y')), float(ext.get('cx')), float(ext.get('cy'))


def _text_of(element) -> str:
    body = element.find('p:txBody', NS)
    if body is None:
        return ''
    paragraphs = []
    for para in body.findall('a:p', NS):
        paragraphs.append(''.join(t.text or '' for t in para.iter(_q('a:t'))))
    return '\n'.join(paragraphs).strip()


def _walk(tree, transform):
    """Yield (element, kind, emu_bbox) in document (z) order, flattening groups."""
    sx, sy, ox, oy = transform
    for child in tree:
        tag = child.tag.split('}')[-1]
        if tag == 'grpSp':
            gx = child.find('p:grpSpPr/a:xfrm', NS)
            if gx is None:
                yield from _walk(child, transform)
                continue
            off, ext = gx.find('a:off', NS), gx.find('a:ext', NS)
            choff, chext = gx.find('a:chOff', NS), gx.find('a:chExt', NS)
            if None in (off, ext, choff, chext):
                yield from _walk(child, transform)
                continue
            kx = float(ext.get('cx')) / max(1.0, float(chext.get('cx')))
            ky = float(ext.get('cy')) / max(1.0, float(chext.get('cy')))
            nx = float(off.get('x')) - float(choff.get('x')) * kx
            ny = float(off.get('y')) - float(choff.get('y')) * ky
            yield from _walk(child, (sx * kx, sy * ky, ox + sx * nx, oy + sy * ny))
            continue
        if tag not in ('sp', 'pic', 'graphicFrame', 'cxnSp'):
            continue
        box = _xfrm(child)
        if box is None:
            continue
        x, y, w, h = box
        emu = (ox + sx * x, oy + sy * y, sx * w, sy * h)
        if tag == 'pic':
            yield child, 'picture', emu
        elif tag == 'graphicFrame':
            uri = ''
            data = child.find('a:graphic/a:graphicData', NS)
            if data is not None:
                uri = data.get('uri', '')
            yield child, 'table' if uri.endswith('/table') else 'chart', emu
        elif tag == 'cxnSp':
            yield child, 'shape', emu
        else:
            yield child, 'text' if _text_of(child) else 'shape', emu


def _picture_pixels(archive, rels, element, size_px):
    blip = element.find('p:blipFill/a:blip', NS)
    if blip is None:
        return None
    target = rels.get(blip.get(_q('r:embed')))
    if not target or target not in archive.namelist():
        return None
    image = Image.open(io.BytesIO(archive.read(target)))
    image.load()
    rgba = image.convert('RGBA')
    src = element.find('p:blipFill/a:srcRect', NS)
    if src is not None:
        w, h = rgba.size
        l, t = int(src.get('l', 0)), int(src.get('t', 0))
        r, b = int(src.get('r', 0)), int(src.get('b', 0))
        box = (round(w * l / 1e5), round(h * t / 1e5), round(w * (1 - r / 1e5)), round(h * (1 - b / 1e5)))
        if box[2] > box[0] and box[3] > box[1]:
            rgba = rgba.crop(box)
    tw, th = max(1, size_px[0]), max(1, size_px[1])
    return np.asarray(rgba.resize((tw, th), Image.BILINEAR))


def inspect_pptx(pptx: str | Path, canvas: tuple[int, int], slide: int = 1) -> dict:
    """Return z-ordered objects and a per-pixel ownership class map at canvas size."""
    width, height = canvas
    with zipfile.ZipFile(pptx) as archive:
        parts = _slide_parts(archive)
        if not 1 <= slide <= len(parts):
            raise ValueError(f'slide {slide} not present (slides={len(parts)})')
        part = parts[slide - 1]
        cx, cy = _slide_size(archive)
        kx, ky = width / cx, height / cy
        rels = _part_rels(archive, part)
        root = ET.fromstring(archive.read(part))
        tree = root.find('p:cSld/p:spTree', NS)
        owner = np.zeros((height, width), np.uint8)  # index into CLASSES; 0 background
        objects = []
        for z, (element, kind, emu) in enumerate(_walk(tree, (1.0, 1.0, 0.0, 0.0))):
            x0 = int(round(emu[0] * kx)); y0 = int(round(emu[1] * ky))
            x1 = int(round((emu[0] + emu[2]) * kx)); y1 = int(round((emu[1] + emu[3]) * ky))
            cnv = element.find('.//p:cNvPr', NS)
            name = cnv.get('name', '') if cnv is not None else ''
            record = {'z': z, 'name': name, 'bbox': [x0, y0, x1 - x0, y1 - y0]}
            cls = kind
            alpha = None
            if kind == 'picture':
                pixels = _picture_pixels(archive, rels, element, (x1 - x0, y1 - y0))
                area_fraction = max(0, x1 - x0) * max(0, y1 - y0) / float(width * height)
                transparent = 0.0
                if pixels is not None:
                    alpha = pixels[..., 3] > 8
                    transparent = float((pixels[..., 3] < 250).mean())
                if area_fraction >= PLATE_AREA_FRACTION:
                    cls = 'plate'
                elif transparent >= TRANSPARENT_FRACTION:
                    cls = 'asset'
                else:
                    cls = 'opaque_picture'
                record['transparent_fraction'] = round(transparent, 4)
            elif kind == 'text':
                record['text'] = _text_of(element)
            record['class'] = cls
            objects.append(record)
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(width, x1), min(height, y1)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            region = np.ones((cy1 - cy0, cx1 - cx0), bool)
            if alpha is not None:
                region = alpha[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            owner[cy0:cy1, cx0:cx1][region] = CLASSES.index(cls)
    return {'slide_count': len(parts), 'objects': objects, 'owner': owner}


def content_mask(source: np.ndarray, std_threshold: float = 6.0) -> np.ndarray:
    """Pixels carrying visible structure (edges, ink, texture) rather than flat fill."""
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (5, 5))
    var = cv2.blur(gray * gray, (5, 5)) - mean * mean
    return np.sqrt(np.maximum(var, 0)) > std_threshold


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    x = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda v: cv2.GaussianBlur(v, (11, 11), 1.5)
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    value = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(value.mean())


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorised CIEDE2000 (kL=kC=kH=1) for (N,3) Lab arrays."""
    L1, a1, b1 = lab1.T
    L2, a2, b2 = lab2.T
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    cbar = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt(cbar ** 7 / (cbar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(c1p * c2p == 0, 0, dh)
    dHp = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh) / 2)
    Lbar, Cbar = (L1 + L2) / 2, (c1p + c2p) / 2
    hsum = h1p + h2p
    hbar = np.where(np.abs(h1p - h2p) > 180, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2), hsum / 2)
    hbar = np.where(c1p * c2p == 0, hsum, hbar)
    t = (1 - 0.17 * np.cos(np.radians(hbar - 30)) + 0.24 * np.cos(np.radians(2 * hbar))
         + 0.32 * np.cos(np.radians(3 * hbar + 6)) - 0.20 * np.cos(np.radians(4 * hbar - 63)))
    sl = 1 + 0.015 * (Lbar - 50) ** 2 / np.sqrt(20 + (Lbar - 50) ** 2)
    sc, sh = 1 + 0.045 * Cbar, 1 + 0.015 * Cbar * t
    rt = (-2 * np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7))
          * np.sin(np.radians(60 * np.exp(-(((hbar - 275) / 25) ** 2)))))
    return np.sqrt((dLp / sl) ** 2 + (dCp / sc) ** 2 + (dHp / sh) ** 2 + rt * (dCp / sc) * (dHp / sh))


def fidelity(source: np.ndarray, render: np.ndarray, *, max_pixels: int = 400_000) -> dict:
    if render.shape != source.shape:
        render = cv2.resize(render, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA)
    diff = np.abs(source.astype(np.int16) - render.astype(np.int16))
    step = max(1, int(np.ceil(np.sqrt(source.shape[0] * source.shape[1] / max_pixels))))
    de = ciede2000(_lab(source[::step, ::step]), _lab(render[::step, ::step]))
    return {
        'mae_rgb': round(float(diff.mean()), 3),
        'p95_abs_rgb': round(float(np.percentile(diff.max(2), 95)), 3),
        'ssim': round(ssim(source, render), 4),
        'delta_e2000_mean': round(float(de.mean()), 3),
        'delta_e2000_p95': round(float(np.percentile(de, 95)), 3),
        'delta_e2000_gt10_fraction': round(float((de > 10).mean()), 4),
        'delta_e_sample_step': step,
    }


def _norm(text: str) -> str:
    return re.sub(r'\s+', '', text or '')


def _overlap(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = min(ax + aw, bx + bw) - max(ax, bx)
    h = min(ay + ah, by + bh) - max(ay, by)
    return max(0.0, w) * max(0.0, h)


def text_metrics(objects: list[dict], ocr_lines: list[dict], min_score: float = 0.5) -> dict:
    """Match independent OCR lines to native text frames by spatial overlap and wording."""
    natives = [o for o in objects if o['class'] == 'text']
    lines = [ln for ln in ocr_lines if _norm(ln.get('text')) and ln.get('score', 1.0) >= min_score]
    total_chars = sum(len(_norm(ln['text'])) for ln in lines)
    matched_chars = 0.0
    matched_lines = 0
    ious = []
    rows = []
    for ln in lines:
        ref = _norm(ln['text'])
        lb = [float(v) for v in ln['bbox']]
        best, best_ratio, best_iou = None, 0.0, 0.0
        for nat in natives:
            ov = _overlap(lb, nat['bbox'])
            if ov <= 0.3 * lb[2] * lb[3]:
                continue
            cand = _norm(nat['text'])
            # Partial match: a merged paragraph frame contains this OCR line.
            matcher = SequenceMatcher(None, ref, cand, autojunk=False)
            ratio = sum(block.size for block in matcher.get_matching_blocks()) / max(1, len(ref))
            if ratio > best_ratio:
                union = lb[2] * lb[3] + nat['bbox'][2] * nat['bbox'][3] - ov
                best, best_ratio, best_iou = nat, ratio, ov / max(1.0, union)
        if best is not None and best_ratio >= 0.5:
            matched_lines += 1
            matched_chars += best_ratio * len(ref)
            ious.append(best_iou)
        rows.append({'ocr_text': ln['text'], 'bbox': lb, 'matched_object': best['name'] if best else None,
                     'char_ratio': round(best_ratio, 3)})
    return {
        'reference': 'independent OCR lines (not a human inventory)',
        'ocr_lines': len(lines),
        'ocr_chars': total_chars,
        'native_text_objects': len(natives),
        'matched_lines': matched_lines,
        'line_recall': round(matched_lines / len(lines), 4) if lines else None,
        'char_recall': round(matched_chars / total_chars, 4) if total_chars else None,
        'mean_frame_iou': round(float(np.mean(ious)), 4) if ious else None,
        'unmatched': [r for r in rows if r['matched_object'] is None][:50],
    }


def score_page(source_path: str | Path, pptx_path: str | Path, *, render_path: str | Path | None = None,
               ocr: dict | None = None, slide: int = 1) -> dict:
    source = np.asarray(Image.open(source_path).convert('RGB'))
    height, width = source.shape[:2]
    inspected = inspect_pptx(pptx_path, (width, height), slide)
    owner = inspected['owner']
    content = content_mask(source)
    total = float(width * height)
    content_total = float(max(1, content.sum()))
    area = {c: round(float((owner == i).sum()) / total, 4) for i, c in enumerate(CLASSES)}
    content_share = {c: round(float(((owner == i) & content).sum()) / content_total, 4)
                     for i, c in enumerate(CLASSES)}
    counts: dict[str, int] = {}
    for obj in inspected['objects']:
        counts[obj['class']] = counts.get(obj['class'], 0) + 1
    result = {
        'schema': SCHEMA,
        'source': str(source_path),
        'pptx': str(pptx_path),
        'slide': slide,
        'image_size': [width, height],
        'object_counts': counts,
        'visible_area_fraction': area,
        'content_fraction': content_share,
        'editable_content_fraction': round(sum(content_share[c] for c in EDITABLE), 4),
        'native_content_fraction': round(sum(content_share[c] for c in ('shape', 'table', 'chart', 'text')), 4),
        'raster_plate_content_fraction': content_share['plate'],
        'notes': [
            'content_fraction is measured on source pixels with local structure; text is frame area, not ink.',
            'Measurements only: not human, WPS or publication acceptance.',
        ],
    }
    if render_path is not None:
        render = np.asarray(Image.open(render_path).convert('RGB'))
        result['fidelity'] = fidelity(source, render)
        result['render'] = str(render_path)
    if ocr is not None:
        result['text'] = text_metrics(inspected['objects'], ocr.get('lines', []))
    result['objects'] = inspected['objects']
    return result


def save_owner_preview(source_path: str | Path, pptx_path: str | Path, out_png: str | Path, slide: int = 1) -> None:
    """Colour-coded ownership overlay for human inspection."""
    source = np.asarray(Image.open(source_path).convert('RGB'))
    owner = inspect_pptx(pptx_path, (source.shape[1], source.shape[0]), slide)['owner']
    palette = np.array([[255, 255, 255], [220, 40, 40], [255, 160, 0], [40, 170, 60],
                        [60, 120, 230], [150, 80, 200], [0, 170, 170], [30, 30, 30]], np.float32)
    overlay = (0.45 * source.astype(np.float32) + 0.55 * palette[owner]).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_png)


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
