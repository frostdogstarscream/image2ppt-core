"""Monochrome solid-fill icon vectorization into native `custom` SlideIR shapes."""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..contracts import CustomCommand, CustomPath, Shape
from ..materialize import write
from ..parallel_contracts import SceneGraph
from ..parallel_rebuild import adapt


def _signed_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _rasterize_contours(
    outer_and_holes: list[tuple[np.ndarray, bool]],
    w: int,
    h: int,
    fg_rgb: np.ndarray,
    backing_crop: np.ndarray,
    ss: int = 4,
) -> np.ndarray:
    canvas = np.zeros((h * ss, w * ss), dtype=np.uint8)
    for pts, is_hole in outer_and_holes:
        scaled = np.rint(pts * float(ss) - 0.5).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [scaled], 0 if is_hole else 255)
    alpha = cv2.resize(canvas.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)[..., None]
    return np.clip(fg_rgb.astype(np.float32)[None, None, :] * alpha + backing_crop.astype(np.float32) * (1.0 - alpha), 0.0, 255.0)


def try_vectorize_monochrome_icon(
    rgba: np.ndarray,
    source_crop: np.ndarray,
    backing_crop: np.ndarray,
    bbox: list[float],
    shape_id: str,
    z: int,
) -> Shape | None:
    """Convert a uniform single-color icon RGBA into a native `geometry='custom'` `Shape` if pixel-faithful."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        return None
    h, w = rgba.shape[:2]
    if min(w, h) < 14 or max(w, h) > 135:
        return None
    if source_crop.shape[:2] != (h, w) or backing_crop.shape[:2] != (h, w):
        return None

    alpha = rgba[..., 3]
    fg_vis = alpha >= 32
    fg_frac = float(fg_vis.mean())
    if not (0.06 <= fg_frac <= 0.78):
        return None
    if all(float((edge > 128).mean()) >= 0.65 for edge in (alpha[0, :], alpha[-1, :], alpha[:, 0], alpha[:, -1])):
        return None

    strong = alpha >= 170
    if int(strong.sum()) < 24:
        return None
    fg_vals = rgba[..., :3][strong].astype(np.float32)
    fg_rgb = np.rint(np.median(fg_vals, axis=0)).astype(np.uint8)
    color_dev_p85 = float(np.percentile(np.abs(fg_vals - fg_rgb.astype(np.float32)).max(1), 85))
    if color_dev_p85 > 16.0:
        return None

    pad = 2
    alpha_pad = np.pad(alpha, pad, mode='constant', constant_values=0)
    up = cv2.resize(alpha_pad, ((w + 2 * pad) * 4, (h + 2 * pad) * 4), interpolation=cv2.INTER_LINEAR)
    mask = (up >= 120).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        return None
    hier = hierarchy[0]
    if len(contours) > 36:
        return None

    commands: list[CustomCommand] = []
    ordered_rings: list[tuple[np.ndarray, bool]] = []
    total_pts = 0

    for cid, cnt in enumerate(contours):
        area = abs(cv2.contourArea(cnt)) / 16.0
        if area < 4.5:
            continue
        is_hole = int(hier[cid][3]) >= 0
        eps = 1.8 if area < 200 else 2.2
        approx = (cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float64) + 0.5) / 4.0 - float(pad)
        approx[:, 0] = np.clip(approx[:, 0], 0.0, float(w))
        approx[:, 1] = np.clip(approx[:, 1], 0.0, float(h))
        if len(approx) < 3:
            continue
        # Ensure opposite winding for outer vs hole contours
        sa = _signed_area(approx)
        if (not is_hole and sa < 0) or (is_hole and sa > 0):
            approx = approx[::-1]
        ordered_rings.append((approx, is_hole))
        total_pts += len(approx) + 1

    if not ordered_rings or total_pts > 320:
        return None

    recomp = _rasterize_contours(ordered_rings, w, h, fg_rgb, backing_crop)
    err = np.abs(recomp - source_crop.astype(np.float32)).max(2)
    if float(err.mean()) > 4.8 or float(np.percentile(err, 95)) > 26.0:
        return None

    for pts, _ in ordered_rings:
        commands.append(CustomCommand(type='moveTo', x=round(float(pts[0, 0]), 2), y=round(float(pts[0, 1]), 2)))
        for pt in pts[1:]:
            commands.append(CustomCommand(type='lineTo', x=round(float(pt[0]), 2), y=round(float(pt[1]), 2)))
        commands.append(CustomCommand(type='close'))

    fill_hex = ''.join(f'{int(v):02X}' for v in fg_rgb[:3])
    return Shape(
        id=shape_id,
        bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
        z=int(z),
        geometry='custom',
        fill=fill_hex,
        line_color=fill_hex,
        line_width_pt=0.0,
        custom_paths=[CustomPath(width=float(w), height=float(h), commands=commands)],
    )


def upgrade_staged_icon_to_vector_shape(
    baseline_dir: str | Path,
    asset_id: str,
    output_dir: str | Path,
) -> tuple[Path, str] | None:
    """Replace an already-staged monochrome icon asset with a native `custom` `Shape` if faithful."""
    baseline = Path(baseline_dir).resolve()
    output = Path(output_dir).resolve()
    graph = SceneGraph.model_validate_json((baseline / 'scene.json').read_text('utf8'))
    rec = graph.selected.get(asset_id)
    if not rec or rec.get('kind') != 'asset' or rec.get('container_layer'):
        return None
    a_obj = next((a for a in graph.asset.objects if a.id == asset_id), None)
    is_crop = bool(rec.get('region_crop'))
    if a_obj is None:
        return None
    if not is_crop and a_obj.kind not in ('icon', 'decoration'):
        return None
    if is_crop and (a_obj.kind not in ('icon', 'decoration', 'illustration') or bool(a_obj.intrinsic_text)):
        return None
    if any(r.kind == 'intrinsic_text' and (r.source == asset_id or r.target == asset_id) for r in graph.relations):
        return None
    asset_ref = rec.get('asset_reference') or {}
    asset_path = baseline / asset_ref.get('path', '')
    owner_ref = rec.get('owner_evidence') or {}
    owner_path = baseline / owner_ref.get('file', '')
    if not asset_path.is_file() or not owner_path.is_file():
        return None

    rgba = np.array(Image.open(asset_path).convert('RGBA'))
    if is_crop and float((rgba[..., 3] < 24).mean()) < 0.12:
        return None
    source = np.array(Image.open(baseline / 'source.png').convert('RGB'))
    with np.load(owner_path, allow_pickle=False) as owner_data:
        backing_patch = owner_data['backing_patch']
        rx, ry, rw, rh = [int(v) for v in owner_data['restore_bbox']]
    ax, ay, aw, ah = [int(round(v)) for v in rec['bbox']]
    if (ax < rx or ay < ry or ax + aw > rx + rw or ay + ah > ry + rh
            or aw != rgba.shape[1] or ah != rgba.shape[0]):
        return None

    backing_crop = backing_patch[ay - ry:ay - ry + ah, ax - rx:ax - rx + aw]
    source_crop = source[ay:ay + ah, ax:ax + aw]

    used_ids = {x.id for x in [*graph.text.texts, *graph.asset.objects, *graph.surface.shapes]} | set(graph.selected)
    seq = 1
    while f's-v{seq:03d}' in used_ids:
        seq += 1
    sid = f's-v{seq:03d}'

    shape_obj = try_vectorize_monochrome_icon(
        rgba, source_crop, backing_crop, [float(ax), float(ay), float(aw), float(ah)], sid, 1
    )
    if shape_obj is None:
        return None

    if output.exists():
        shutil.rmtree(output, ignore_errors=True)
    shutil.copytree(baseline, output)

    # Replace asset entry with verified surface shape while keeping complete owner_evidence & mask
    graph.asset.objects = [a for a in graph.asset.objects if a.id != asset_id]
    graph.ownership.pop(asset_id, None)
    graph.relations = [
        r.model_copy(update={
            'source': sid if r.source == asset_id else r.source,
            'target': sid if r.target == asset_id else r.target,
        })
        for r in graph.relations
    ]
    graph.surface.shapes.append(shape_obj)
    graph.ownership[sid] = 'RESOLVED'

    rec_copy = dict(rec)
    rec_copy.pop('asset_reference', None)
    rec_copy.pop('region_crop', None)
    rec_copy['id'] = sid
    rec_copy['kind'] = 'surface'
    rec_copy['foreground_shape'] = True
    rec_copy['seed_ids'] = sorted({asset_id, *rec_copy.get('seed_ids', [])})
    rec_copy['payload'] = shape_obj.model_dump()
    rec_copy['method'] = 'monochrome-contour-vector-icon/1'
    graph.selected.pop(asset_id, None)
    graph.selected[sid] = rec_copy
    graph.transactions.append(rec_copy)

    stale_png = output / asset_ref['path']
    if stale_png.is_file():
        stale_png.unlink()

    if (output / 'ownership.npz').is_file() and (output / 'ownership.json').is_file():
        with np.load(output / 'ownership.npz', allow_pickle=False) as ow:
            ow_map, unc = ow['ownership'].copy(), ow['uncertain_text'].copy()
        with np.load(owner_path, allow_pickle=False) as owner_data:
            ow_mask = owner_data['mask'].astype(bool)
        ow_map[ow_mask] = 2
        np.savez_compressed(output / 'ownership.npz', ownership=ow_map, uncertain_text=unc)
        import json
        ow_meta = json.loads((output / 'ownership.json').read_text('utf8'))
        from ..materialize import digest
        ow_meta['map_sha256'] = digest(output / 'ownership.npz')
        write(output / 'ownership.json', ow_meta)

    ir = adapt(graph, output)
    write(output / 'scene.json', graph.model_dump())
    write(output / 'slideir.json', ir)
    return output / 'slideir.json', sid


def upgrade_crops_to_vector_shapes(
    baseline_dir: str | Path,
    staged_ids: list[str],
    work_dir: str | Path,
) -> tuple[Path, list[str]]:
    """Upgrade eligible tight cutout crops in `staged_ids` into native `custom` vector shapes."""
    cur_dir = Path(baseline_dir).resolve()
    out_root = Path(work_dir).resolve()
    updated_ids = list(staged_ids)
    for idx, oid in enumerate(list(staged_ids)):
        upgraded = upgrade_staged_icon_to_vector_shape(
            cur_dir, oid, out_root / f'vec-{idx:02d}-{oid}'
        )
        if upgraded is not None:
            vec_ir, vec_sid = upgraded
            cur_dir = vec_ir.parent
            updated_ids[idx] = vec_sid
    return cur_dir / 'slideir.json', updated_ids
