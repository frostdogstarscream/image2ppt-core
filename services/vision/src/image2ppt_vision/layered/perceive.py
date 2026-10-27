"""GPU perception worker for the layered pipeline (runs beside, never inside, production).

Usage (inside the vision image):
  python -m image2ppt_vision.layered.perceive pages <in_dir> <out_dir>
  python -m image2ppt_vision.layered.perceive ocr <out.json> <image> [...]

Every page directory receives the OCR, ink masks, text-free page, element
cutouts, cleaned background, residual OCR and a hash manifest. No model API is
called here.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import inpaint as repair
from .segment import classify, contrast_candidates, cutout, icon_fallbacks
from .tables import detect_tables, fill_cells
from .textmask import analyze_line, analyze_page, split_wide_gaps

OCR_MIN_SCORE = .5
SCHEMA = 'LayeredPerception/1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')


def _save_mask(path, mask):
    Image.fromarray(mask.astype(np.uint8)*255).save(path)


def plausible_text(hit):
    """A lone low-confidence CJK character is usually an icon read as a glyph."""
    text = hit['text'].strip()
    if not text or hit['score'] < OCR_MIN_SCORE:
        return False
    _, _, w, h = hit['bbox']
    # Slides are set horizontally; a doubtful tall narrow hit is a column of stacked icons.
    if len(text) >= 2 and h > 2.2*w and hit['score'] < .8:
        return False
    return not (len(text) == 1 and ord(text) > 0x2E80 and hit['score'] < .9)


def bullet_icon(hit, hits):
    """A lone letter directly left of a text line on the same row is a list
    bullet icon read as a glyph (a magnifier as 'Q'); digits stay, they badge."""
    text = hit['text'].strip()
    if len(text) != 1 or not text.isalpha() or ord(text) > 0x2E80:
        return False
    x, y, w, h = hit['bbox']
    for other in hits:
        if other is hit or len(other['text'].strip()) < 2:
            continue
        ox, oy, ow, oh = other['bbox']
        same_row = abs((oy+oh/2)-(y+h/2)) < .35*max(h, oh) and .6 < oh/max(h, 1) < 1.6
        if same_row and 0 <= ox-(x+w) <= 1.5*max(h, oh):
            return True
    return False


def _ocr(rgb):
    from ..models import recognize
    # Slides are upright; the angle classifier flips short digit strings ("09" -> "60").
    hits = [t for t in recognize(rgb, use_cls=False) if plausible_text(t)]
    return [t for t in hits if not bullet_icon(t, hits)]


def _sam(rgb):
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from ..models import model_root, sam_model
    model = sam_model(str(model_root()))
    h, w = rgb.shape[:2]
    scale = min(1., 1536/max(h, w))
    work = cv2.resize(rgb, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    with torch.inference_mode():
        generator = SAM2AutomaticMaskGenerator(model, points_per_side=48, points_per_batch=64,
                                               pred_iou_thresh=.8, stability_score_thresh=.9, crop_n_layers=0)
        results = generator.generate(work)
    return [(cv2.resize(r['segmentation'].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0,
             {'source': 'sam2', 'predicted_iou': float(r['predicted_iou']),
              'stability_score': float(r['stability_score'])}) for r in results]


def _lama():
    from ..models import lama_model, lama_path
    path = lama_path()
    return lama_model(str(path)) if path else None


def _overlaps(box, mask, fraction):
    x, y, w, h = (int(round(v)) for v in box)
    window = mask[max(0, y):y+h, max(0, x):x+w]
    return window.size and window.mean() >= fraction


def perceive(source, out, *, ocr=_ocr, sam=_sam, model=None):
    out = Path(out)
    (out/'assets').mkdir(parents=True, exist_ok=False)
    (out/'masks').mkdir()
    timings = {}
    clock = time.time()
    with Image.open(source) as image:
        rgb = np.asarray(image.convert('RGB')).copy()
    H, W = rgb.shape[:2]
    lines = split_wide_gaps(rgb, ocr(rgb))
    for i, line in enumerate(lines):
        line['id'] = f't{i:03d}'
    timings['ocr'] = time.time()-clock; clock = time.time()
    records, text_mask, icon_mask = analyze_page(rgb, lines)
    timings['textmask'] = time.time()-clock; clock = time.time()
    notext, text_method = repair.inpaint(rgb, text_mask, model)
    timings['inpaint_text'] = time.time()-clock; clock = time.time()
    tables = detect_tables(notext)
    table_mask = np.zeros((H, W), bool)
    by_id = {r['id']: r for r in records}
    for k, table in enumerate(tables):
        x, y, w, h = table['bbox']
        table_mask[max(0, y-3):y+h+3, max(0, x-3):x+w+3] = True
        table['id'] = f'tb{k:02d}'
        table['rows'], table['owned_text_ids'], table['cell_styles'] = fill_cells(notext, table, lines, by_id)
    masks = sam(notext) if sam else []
    timings['sam'] = time.time()-clock; clock = time.time()
    extra = contrast_candidates(notext, exclude=table_mask)
    elements, erase = classify(notext, [m for m in masks if not _overlaps(_box(m[0]), table_mask, .5)],
                               text_mask=text_mask, icon_mask=icon_mask, extra=extra)
    icons = [b for r in records for b in r['icon_candidates']]
    for fallback in icon_fallbacks(notext, icons, elements):
        fallback['id'] = f'el-{len(elements):03d}'
        elements.append(fallback)
        erase |= cv2.dilate(fallback['mask'].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    elements = [e for e in elements if not _overlaps(e['bbox'], table_mask, .5)]
    timings['segment'] = time.time()-clock; clock = time.time()
    background, bg_method = repair.inpaint(notext, erase | table_mask, model)
    timings['inpaint_elements'] = time.time()-clock; clock = time.time()
    # Residual OCR: text the first pass missed is recovered once; ghosts are reported.
    recovered = []
    erased_area = cv2.dilate((text_mask | erase).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    for hit in ocr(background) if ocr else []:
        if _overlaps(hit['bbox'], erased_area, .3):
            continue
        hit['id'] = f't{len(lines)+len(recovered):03d}'
        hit['recovered_from_background'] = True
        extra_mask, extra_icons = np.zeros((H, W), bool), np.zeros((H, W), bool)
        record = analyze_line(rgb, hit, extra_mask, extra_icons)
        if record['status'] != 'ok':
            continue
        background, _ = repair.inpaint(background, extra_mask & ~extra_icons, model)
        text_mask |= extra_mask & ~extra_icons
        lines.append(hit); records.append(record); recovered.append(hit['id'])
    residual = [{'text': h['text'], 'bbox': h['bbox'], 'score': h['score']} for h in (ocr(background) if ocr else [])]
    timings['residual_ocr'] = time.time()-clock
    Image.fromarray(rgb).save(out/'source.png')
    Image.fromarray(notext).save(out/'notext.png')
    Image.fromarray(background).save(out/'background.png')
    _save_mask(out/'masks/text.png', text_mask)
    _save_mask(out/'masks/icon.png', icon_mask)
    _save_mask(out/'masks/erase.png', erase | table_mask)
    overlay = rgb.copy()
    public = []
    for e in elements:
        x, y, w, h = e['bbox']
        cv2.rectangle(overlay, (x, y), (x+w-1, y+h-1), (230, 30, 30) if e['kind'] == 'native_shape' else (20, 160, 40), 1)
        item = {k: v for k, v in e.items() if k != 'mask'}
        if e['kind'] == 'raster':
            path = f'assets/{e["id"]}.png'
            Image.fromarray(cutout(notext, e), 'RGBA').save(out/path)
            item.update(asset=path, asset_sha256=_sha(out/path))
        public.append(item)
    for t in tables:
        x, y, w, h = t['bbox']
        cv2.rectangle(overlay, (x, y), (x+w-1, y+h-1), (40, 40, 220), 2)
    Image.fromarray(overlay).save(out/'elements-overlay.png')
    solid = repair.solid_color(background)
    result = {'schema': SCHEMA, 'input_sha256': _sha(Path(source)), 'source_sha256': _sha(out/'source.png'), 'image_size': [W, H],
              'ocr': [{k: line[k] for k in ('id', 'text', 'bbox', 'polygon', 'score', 'split_from') if k in line} |
                      ({'recovered_from_background': True} if line.get('recovered_from_background') else {})
                      for line in lines],
              'text': records, 'elements': public, 'tables': tables,
              'background': {'path': 'background.png', 'sha256': _sha(out/'background.png'), 'solid_color': solid,
                             'text_method': text_method, 'element_method': bg_method},
              'residual_ocr': residual, 'recovered_text_ids': recovered,
              'counts': {'ocr_lines': len(lines), 'ink_ok': sum(r['status'] == 'ok' for r in records),
                         'sam_masks': len(masks), 'contrast_candidates': len(extra),
                         'native_shapes': sum(e['kind'] == 'native_shape' for e in public),
                         'rasters': sum(e['kind'] == 'raster' for e in public), 'tables': len(tables)},
              'timings_s': {k: round(v, 3) for k, v in timings.items()}}
    result['files'] = {p.relative_to(out).as_posix(): _sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    _write(out/'perception.json', result)
    return result


def _box(mask):
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)] if len(xs) else [0, 0, 1, 1]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv.pop(0)
    if mode == 'ocr':
        target = Path(argv.pop(0))
        result = {}
        for name in argv:
            with Image.open(name) as image:
                result[Path(name).name] = _ocr(np.asarray(image.convert('RGB')).copy())
        _write(target, result)
        return 0
    if mode != 'pages':
        raise SystemExit('mode must be pages or ocr')
    source_dir, out_dir = Path(argv[0]), Path(argv[1])
    model = _lama()
    from ..models import LAMA_SHA256
    summary = {'schema': 'LayeredPerceptionRun/1', 'lama_sha256': LAMA_SHA256 if model else None, 'pages': {}}
    for page in sorted(source_dir.glob('*.png')):
        started = time.time()
        result = perceive(page, out_dir/page.stem, model=model)
        summary['pages'][page.stem] = {'seconds': round(time.time()-started, 2), 'counts': result['counts']}
        print(page.stem, json.dumps(result['counts']), flush=True)
    _write(out_dir/'run.json', summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
