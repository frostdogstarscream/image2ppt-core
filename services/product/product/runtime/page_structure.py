"""VLM page structure inventory & pixel-verified OCR proofreading (W1 + W5).

Produces `PageStructureInventory/1` (object inventory denominator + hierarchy +
form hints) and verifies 1-2 character VLM OCR typo fixes against source glyph
ink before SceneGraph fusion so misread lines become native editable text safely.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field

from ..contracts import intersection
from ..materialize import digest, write
from ..parallel_store import Store, key


class OCRCorrection(BaseModel):
    model_config = ConfigDict(extra='ignore')
    ocr_id: str
    corrected_text: str = Field(min_length=1, max_length=500)
    is_logo_brand: bool = False
    reason: str = ''


class InventoryElement(BaseModel):
    model_config = ConfigDict(extra='ignore')
    id: str
    category: Literal[
        'title', 'section_header', 'body_text',
        'card_shape', 'ribbon_shape', 'badge_shape', 'divider',
        'chart', 'table', 'icon', 'photo', 'logo', 'background_scenery',
    ]
    bbox: list[float] = Field(min_length=4, max_length=4)
    parent_id: str | None = None
    geometry_hint: str | None = None
    has_overlay_text: bool = False
    label: str = ''


class OCRVisualRole(BaseModel):
    ocr_id: str
    role: Literal['readable_text', 'graphic', 'uncertain']


class VLMPageParseReply(BaseModel):
    model_config = ConfigDict(extra='ignore')
    isolated_roles: list[OCRVisualRole] = Field(default_factory=list, max_length=100)
    ocr_corrections: list[OCRCorrection] = Field(default_factory=list, max_length=80)
    elements: list[InventoryElement] = Field(default_factory=list, max_length=200)


def _norm(s: str) -> str:
    return re.sub(r'\s+', '', s or '')


def _render_glyph_iou(
    source_crop: np.ndarray,
    text: str,
    font_file: str | None,
    font_index: int = 0,
) -> float | None:
    """Compare rendered glyph silhouette against source binarized ink crop."""
    h, w = source_crop.shape[:2]
    if h < 8 or w < 8 or not font_file or not Path(font_file).is_file():
        return None
    border = np.concatenate([
        source_crop[:2, :].reshape(-1, 3),
        source_crop[-2:, :].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    bg = np.median(border, axis=0)
    src_ink = np.abs(source_crop.astype(np.float32) - bg).max(2) > 22.0
    if int(src_ink.sum()) < 8:
        return None
    yy, xx = np.nonzero(src_ink)
    src_tight = src_ink[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    th, tw = src_tight.shape[:2]
    if th < 6 or tw < 6:
        return None

    try:
        font = ImageFont.truetype(font_file, max(10, int(round(th * 0.92))), index=font_index)
    except Exception:
        return None
    canvas = Image.new('L', (max(tw * 2, 64), max(th * 2, 32)), 0)
    draw = ImageDraw.Draw(canvas)
    draw.text((2, 2), text, fill=255, font=font)
    arr = np.asarray(canvas) > 64
    if not arr.any():
        return None
    ry, rx = np.nonzero(arr)
    rend_tight = arr[ry.min():ry.max() + 1, rx.min():rx.max() + 1].astype(np.uint8)
    resized = cv2.resize(rend_tight, (tw, th), interpolation=cv2.INTER_LINEAR) > 0
    # Allow 1px dilation for font family differences
    src_d = cv2.dilate(src_tight.astype('uint8'), np.ones((2, 2), np.uint8)) > 0
    ren_d = cv2.dilate(resized.astype('uint8'), np.ones((2, 2), np.uint8)) > 0
    inter = int((src_d & ren_d).sum())
    union = int((src_d | ren_d).sum())
    return inter / float(max(1, union))


def isolated_label_neighbors(lines):
    """Find square OCR fields next to a distinct longer label (keep numbers as text)."""
    matches = {}
    for line in lines:
        token = line.get('text', '').strip()
        if len(token) != 1 or not token.isalpha() or line.get('score', 0) < .85:
            continue
        x, y, w, h = line['bbox']
        if h < 8 or not .55 <= w/h <= 1.5:
            continue
        neighbors = [peer for peer in lines if len(peer.get('text', '').strip()) >= 2
                     and 0 <= peer['bbox'][0]-x-w <= 2.5*h
                     and abs(peer['bbox'][1]+peer['bbox'][3]/2-y-h/2) <= .55*h]
        if neighbors:
            matches[line['id']] = min(neighbors, key=lambda peer: peer['bbox'][0])
    return matches


def resolve_isolated_roles(source, ocr, layout, structure, fonts, neighbors):
    """Only credited textual readings may become editable words; abstentions keep source pixels."""
    judgments = {item['ocr_id']: item['role'] for item in structure.get('isolated_roles', [])}
    candidates = {}
    font = next((f for f in fonts.get('fonts', []) if f.get('cjk') and
                 ('雅黑' in f.get('family', '') or 'YaHei' in f.get('family', ''))), None)
    for line in ocr.get('lines', []):
        peer = neighbors.get(line['id'])
        if not peer:
            continue
        box = line['bbox']
        x, y, w, h = map(round, box)
        crop = source[y:y+h, x:x+w]
        if not crop.size:
            continue
        hints = [r for r in (layout or {}).get('regions', []) if r.get('label') == 'image'
                 and intersection(box, r['bbox']) >= .5*w*h and r['bbox'][2]*r['bbox'][3] <= 4*w*h]
        def ink_color(img):
            colors = img.reshape(-1, 3).astype(float)
            bg = np.median(colors, axis=0)
            contrast = np.abs(colors-bg).max(axis=1)
            ink = colors[contrast >= max(20, np.percentile(contrast, 80))]
            return np.median(ink, axis=0) if len(ink) else bg
        px, py, pw, ph = map(round, peer['bbox'])
        label = source[py:py+ph, px:px+pw]
        if not label.size:
            continue
        color_gap = float(np.abs(ink_color(crop)-ink_color(label)).max())
        similarity = _render_glyph_iou(crop, line['text'], font['file'], font.get('index', 0)) if font else None
        role = judgments.get(line['id'])
        oversized_marker = h >= 1.25 * ph
        ambiguous = bool(hints) or (color_gap > 42 and (oversized_marker or similarity is None or similarity < .72))
        if role == 'graphic' or (role != 'readable_text' and (ambiguous or role == 'uncertain')):
            candidates[line['id']] = dict(text=line['text'], bbox=box, neighbor_id=peer['id'],
                disposition='graphic' if role == 'graphic' or hints else 'unresolved_graphic_or_text',
                visual_role=role, layout_image=bool(hints), color_gap=round(color_gap, 2), glyph_iou=similarity)
    return candidates


def verify_and_apply_ocr_corrections(
    source: np.ndarray,
    ocr: dict,
    corrections: list[OCRCorrection],
    fonts: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Verify VLM OCR typo corrections against source pixels before updating `ocr`."""
    font_file = None
    font_index = 0
    if fonts and fonts.get('fonts'):
        cjk = next((f for f in fonts['fonts'] if f.get('cjk') and '雅黑' in f.get('family', '')), None)
        if not cjk:
            cjk = next((f for f in fonts['fonts'] if f.get('cjk')), None)
        if cjk:
            font_file = cjk.get('file')
            font_index = int(cjk.get('index', 0))

    h_img, w_img = source.shape[:2]
    lines_by_id = {ln['id']: ln for ln in ocr.get('lines', []) if 'id' in ln}
    applied: list[dict] = []

    for corr in corrections:
        ln = lines_by_id.get(corr.ocr_id)
        if ln is None:
            continue
        if corr.is_logo_brand:
            ln['is_logo'] = True
        raw_t = _norm(ln.get('text', ''))
        new_t = _norm(corr.corrected_text)
        if not new_t or new_t == raw_t:
            continue
        # Guard 1: Only localized 1-2 character OCR misreads (never alter numbers or percentages)
        if abs(len(new_t) - len(raw_t)) > 1:
            continue
        if re.findall(r'[\d%‰]+', new_t) != re.findall(r'[\d%‰]+', raw_t):
            continue
        sim = SequenceMatcher(None, raw_t, new_t, autojunk=False).ratio()
        min_sim = 0.50 if len(raw_t) <= 4 else 0.62
        if sim < min_sim:
            continue
        # Guard 2: Verify real ink strokes span the OCR bbox
        bx, by, bw, bh = [int(round(v)) for v in ln['bbox']]
        bx0, by0 = max(0, bx), max(0, by)
        bx1, by1 = min(w_img, bx + bw), min(h_img, by + bh)
        if bx1 - bx0 < 8 or by1 - by0 < 8:
            continue
        crop = source[by0:by1, bx0:bx1]
        border = np.concatenate([crop[:2, :].reshape(-1, 3), crop[-2:, :].reshape(-1, 3)], axis=0).astype(np.float32)
        bg = np.median(border, axis=0)
        ink = np.abs(crop.astype(np.float32) - bg).max(2) > 18.0
        if int(ink.sum()) < 10:
            continue
        xs = np.where(ink.any(axis=0))[0]
        if len(xs) == 0 or (xs.max() - xs.min() + 1) < 0.55 * (bx1 - bx0):
            continue

        # Guard 3: Silhouette sanity check (corrected glyph shape must not drop sharply below raw OCR)
        iou_raw = _render_glyph_iou(crop, raw_t, font_file, font_index)
        iou_new = _render_glyph_iou(crop, new_t, font_file, font_index)
        score_orig = float(ln.get('score', 1.0))
        if iou_raw is None or iou_new is None:
            continue
        cjk_only_diff = re.findall(r'[0-9A-Za-z%‰]+', new_t) == re.findall(r'[0-9A-Za-z%‰]+', raw_t)
        slack = 0.14 if (cjk_only_diff and iou_new >= 0.18) else (0.06 if score_orig < 0.97 else 0.015)
        if iou_new + slack < iou_raw:
            continue

        ln['ocr_raw_text'] = ln['text']
        ln['text'] = corr.corrected_text.strip()
        ln['score'] = max(score_orig, 0.92)
        ln['proofread_verified'] = True
        applied.append({
            'ocr_id': corr.ocr_id,
            'before': ln['ocr_raw_text'],
            'after': ln['text'],
            'similarity': round(sim, 3),
            'iou_before': round(iou_raw, 3) if iou_raw is not None else None,
            'iou_after': round(iou_new, 3) if iou_new is not None else None,
            'reason': corr.reason,
        })

    return ocr, applied


def build_deterministic_inventory(
    image_size: list[int],
    ocr: dict,
    layout: dict | None,
    assets: list[dict],
) -> list[dict]:
    """Construct baseline `PageStructureInventory/1` elements from OCR + Layout + LayerD."""
    w_img, h_img = image_size
    elements: list[dict] = []
    idx = 1

    for reg in (layout or {}).get('regions', []):
        lbl = reg.get('label', '')
        rb = [round(float(v), 1) for v in reg.get('bbox', [])]
        if len(rb) != 4 or float(reg.get('score', 0.0)) < 0.5:
            continue
        cat = None
        if lbl in ('header_image', 'seal') or (lbl == 'image' and rb[0] >= 0.78 * w_img and rb[1] <= 0.15 * h_img):
            cat = 'logo'
        elif lbl == 'chart':
            cat = 'chart'
        elif lbl == 'table':
            cat = 'table'
        elif lbl == 'image':
            # Distinguish larger rectangular photos vs compact icons
            aspect = rb[2] / max(1.0, rb[3])
            if rb[2] >= 150 and rb[3] >= 90 and 0.9 <= aspect <= 2.6:
                has_txt = any(
                    intersection(rb, ln['bbox']) >= 0.4 * max(1.0, ln['bbox'][2] * ln['bbox'][3])
                    for ln in ocr.get('lines', []) if ln.get('score', 0) >= 0.7
                )
                cat = 'photo'
                elements.append({
                    'id': f'inv-{idx:03d}',
                    'category': cat,
                    'bbox': rb,
                    'parent_id': None,
                    'geometry_hint': 'rect',
                    'has_overlay_text': has_txt,
                    'label': lbl,
                })
                idx += 1
                continue
            cat = 'icon'
        if cat:
            elements.append({
                'id': f'inv-{idx:03d}',
                'category': cat,
                'bbox': rb,
                'parent_id': None,
                'geometry_hint': None,
                'has_overlay_text': False,
                'label': lbl,
            })
            idx += 1

    # Add LayerD icons not already covered by layout regions
    for a in assets:
        ab = [round(float(v), 1) for v in a.get('bbox', [])]
        if len(ab) != 4:
            continue
        if any(intersection(ab, e['bbox']) >= 0.4 * min(ab[2] * ab[3], e['bbox'][2] * e['bbox'][3]) for e in elements):
            continue
        elements.append({
            'id': f'inv-{idx:03d}',
            'category': 'icon',
            'bbox': ab,
            'parent_id': None,
            'geometry_hint': None,
            'has_overlay_text': False,
            'label': a.get('kind', 'icon'),
        })
        idx += 1

    # Add non-logo OCR text items (identifying main title vs section headers vs body)
    for ln in ocr.get('lines', []):
        txt = _norm(ln.get('text', ''))
        lb = [round(float(v), 1) for v in ln.get('bbox', [])]
        if not txt or len(lb) != 4 or float(ln.get('score', 0)) < 0.75:
            continue
        is_logo = bool(ln.get('is_logo')) or (lb[0] >= 0.80 * w_img and lb[1] <= 0.16 * h_img and lb[2] <= 0.18 * w_img)
        if is_logo:
            continue
        cat = 'title' if (lb[1] <= 0.11 * h_img and lb[3] >= 28) else (
            'section_header' if lb[3] >= 24 else 'body_text'
        )
        elements.append({
            'id': f'inv-{idx:03d}',
            'category': cat,
            'bbox': lb,
            'parent_id': None,
            'geometry_hint': None,
            'has_overlay_text': False,
            'label': ln.get('text', ''),
            'ocr_id': ln.get('id'),
        })
        idx += 1

    return elements


def parse_page_structure(
    state,
    source_path: str | Path,
    ocr: dict,
    *,
    layout: dict | None = None,
    assets: list[dict] | None = None,
    fonts: dict | None = None,
    use_vlm: bool = True,
) -> dict:
    """Run deterministic + optional VLM page structure analysis & verified OCR proofreading."""
    from ..recovery.adaptive_policy import AdaptiveBudget
    from .provider import planning_actor

    source_file = Path(source_path)
    source = np.asarray(Image.open(source_file).convert('RGB'))
    h_img, w_img = source.shape[:2]
    src_sha = digest(source_file)

    det_elements = build_deterministic_inventory([w_img, h_img], ocr, layout, assets or [])
    vlm_status = 'SKIPPED'
    applied_corrections: list[dict] = []
    isolated_roles = []
    final_elements = det_elements

    request_file = state.job / 'request.json'
    request = json.loads(request_file.read_text('utf8')) if request_file.is_file() else {}
    page_count = max(1, len(request.get('sources') or [1]))
    budget = AdaptiveBudget(Store(state.job / 'state'), request.get('adaptive_budget'), page_count=page_count)

    if use_vlm:
        cache_dir = state.job / 'cache/page-structure'
        cache_dir.mkdir(parents=True, exist_ok=True)
        ocr_digest = key([{'id': r.get('id'), 'text': r.get('text'), 'bbox': r.get('bbox')} for r in ocr.get('lines', [])])
        cache_key = key({
            'schema': 'PageStructureCall/2', 'implementation': digest(__file__),
            'source_sha256': src_sha,
            'ocr_digest': ocr_digest,
            'model': request.get('agent_model') or 'default',
        })
        cache_file = cache_dir / f'{cache_key}.json'
        raw_reply = None
        if cache_file.is_file():
            raw_reply = json.loads(cache_file.read_text('utf8'))
            vlm_status = 'CACHED'
        else:
            try:
                actor = planning_actor(state.job)
                if not budget.reserve('model_calls', {'tool': 'parse_page_structure', 'key': cache_key}):
                    raise RuntimeError('previous page-structure response unknown')
                compact_ocr = [
                    {
                        'ocr_id': ln['id'],
                        'text': ln.get('text', ''),
                        'bbox': [round(float(v), 1) for v in ln.get('bbox', [])],
                        'score': round(float(ln.get('score', 0.0)), 3),
                    }
                    for ln in ocr.get('lines', [])
                    if _norm(ln.get('text', ''))
                ]
                system_prompt = (
                    'You are an accurate Chinese/English slide OCR proofreader and layout parser. '
                    'Inspect the slide image carefully against `ocr_lines`. '
                    '1) In `ocr_corrections`, report ONLY lines where OCR misread 1-2 characters '
                    '(e.g. 管理 vs 整理, 选代 vs 迭代, 瓷询 vs 咨询) or brand logo lettering (`is_logo_brand=true`). '
                    'Copy exact source characters from the image; never paraphrase. '
                    '2) In `elements`, list major visual containers/shapes (`card_shape`, `ribbon_shape`, `badge_shape`), '
                    '`chart`, `photo` (with `has_overlay_text`), `icon`, and `logo` with `[x,y,w,h]` pixel bboxes. '
                    '3) For each isolated_ocr_id compare the original symbol with the neighboring label. In isolated_roles return readable_text for real letters, graphic for pictograms, or uncertain. '
                    'Return strict JSON matching `output_schema`.'
                )
                attachments = [source_file]
                isolated = isolated_label_neighbors(ocr.get('lines', []))
                if isolated:
                    # Include enough of the adjacent wording to distinguish a leading icon
                    # from an independent number/letter. The page itself provides context.
                    tiles = [(ln, isolated[ln['id']]) for ln in ocr['lines'] if ln['id'] in isolated][:24]
                    board = Image.new('RGB', (720, ((len(tiles)+1)//2)*106), 'white')
                    draw = ImageDraw.Draw(board)
                    for i, (ln, peer) in enumerate(tiles):
                        x, y, w, h = map(round, ln['bbox'])
                        px, py, pw, ph = map(round, peer['bbox'])
                        xa, ya = max(0,x-h//2), max(0,min(y,py)-h//3)
                        crop = Image.fromarray(source[ya:min(h_img,max(y+h,py+ph)+h//3),
                                                      xa:min(w_img,px+pw+h//3)])
                        crop.thumbnail((350,82))
                        at = (360*(i%2),106*(i//2))
                        draw.text(at, str(ln['id']), fill='black')
                        board.paste(crop, (at[0],at[1]+18))
                    sheet = cache_dir/f'{cache_key}-isolated.png'
                    board.save(sheet)
                    attachments.append(sheet)
                raw_reply = actor.ask(
                    system_prompt,
                    {'image_size': [w_img, h_img], 'ocr_lines': compact_ocr, 'isolated_ocr_ids': list(isolated)},
                    attachments,
                    VLMPageParseReply,
                )
                write(cache_file, raw_reply)
                vlm_status = 'READY'
            except Exception as exc:
                vlm_status = f'UNAVAILABLE:{type(exc).__name__}'

        if raw_reply is not None:
            try:
                parsed = VLMPageParseReply.model_validate(raw_reply)
                isolated_roles = [role.model_dump() for role in parsed.isolated_roles]
                if parsed.ocr_corrections:
                    ocr, applied_corrections = verify_and_apply_ocr_corrections(
                        source, ocr, parsed.ocr_corrections, fonts=fonts
                    )
                    # Rebuild text inventory entries with updated verified wording
                    det_elements = build_deterministic_inventory([w_img, h_img], ocr, layout, assets or [])
                vlm_shapes = [
                    el.model_dump() for el in parsed.elements
                    if el.category in ('card_shape', 'ribbon_shape', 'badge_shape', 'chart', 'table', 'photo', 'logo', 'icon')
                    and all(v >= 0 for v in el.bbox[:2]) and el.bbox[2] > 8 and el.bbox[3] > 8
                ]
                # Merge non-duplicate VLM container/shape inventory items with deterministic items
                merged = list(det_elements)
                for vel in vlm_shapes:
                    vb = vel['bbox']
                    if not any(
                        e['category'] == vel['category']
                        and intersection(vb, e['bbox']) >= 0.5 * min(vb[2] * vb[3], e['bbox'][2] * e['bbox'][3])
                        for e in merged
                    ):
                        merged.append(vel)
                final_elements = merged
            except Exception as exc:
                vlm_status = f'PARSE_FALLBACK:{type(exc).__name__}'

    return {
        'schema': 'PageStructureInventory/1',
        'source_sha256': src_sha,
        'image_size': [w_img, h_img],
        'vlm_status': vlm_status,
        'isolated_roles': isolated_roles,
        'applied_ocr_corrections': applied_corrections,
        'elements': final_elements,
    }
