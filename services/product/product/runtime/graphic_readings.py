"""Attribute fragmentary OCR to an independently removable pictogram using its real edit receipt."""

from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts import intersection
from ..materialize import digest
from .late_fields import source_reading


def _read_bound(root, filename, sha):
    path = (root / filename).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or digest(path) != sha:
        raise ValueError('graphic evidence changed')
    return path


def _owned_fraction(box, owner_box, mask, alpha, source, hidden, deleted):
    x, y, w, h = map(int, box)
    left, top, width, height = owner_box
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > source.width or y + h > source.height:
        return None
    if intersection(box, owner_box) < 0.90 * w * h:
        return None
    crop = x, y, x + w, y + h
    background = np.asarray(deleted.crop(crop), dtype=float)
    # A separate label, a neighbouring stroke, or a textured picture exposed by deletion
    # prevents using this simple disappearance test.
    if np.max(np.abs(background - np.median(background, axis=(0, 1)))) > 30:
        return None
    strong = Image.fromarray(alpha).crop((x - left, y - top, x + w - left, y + h - top))
    supported = (np.asarray(strong) >= 32) & mask[y : y + h, x : x + w]
    source_ink = np.max(np.abs(np.asarray(source.crop(crop)) - background), axis=2) >= 40
    hidden_ink = np.max(np.abs(np.asarray(hidden.crop(crop)) - background), axis=2) >= 40
    joint = source_ink & hidden_ink
    if joint.sum() < 8 or not source_ink.any() or not hidden_ink.any():
        return None
    support = min((joint & supported).sum() / ink.sum() for ink in (source_ink, hidden_ink))
    fraction = joint.sum() / max(1, np.count_nonzero(alpha >= 32))
    # A standalone glyph that occupies the whole object remains a possible business field.
    if support < 0.98 or not 0.06 <= fraction <= 0.65:
        return None
    return {'owned_ink_fraction': round(float(support), 4), 'part_of_symbol_ink': round(float(fraction), 4)}


def _overlapping_parts(one, other):
    a, b = one['bbox'], other['bbox']
    overlap = intersection(a, b) / min(a[2]*a[3], b[2]*b[3])
    different_baselines = abs(a[1]+a[3]-b[1]-b[3]) > min(a[3], b[3]) * .35
    return one['text'] != other['text'] and .04 <= overlap < .30 and different_baselines


def bind_graphic_readings(
    audit, *, job, page, scene, backing, source_path, hidden_path, lines, editable_ids, residuals, directory
):
    observations = [
        row
        for row in audit['rows']
        if row['status'] == 'plate_only'
        and row['ocr_id'] is None
        and row.get('seen_on_named_plate') is False
        and len(row['text']) == 1
        and row['text'].isascii()
        and row['text'].isalpha()
        and not any(intersection(row['bbox'], line['bbox']) > 0 for line in lines)
    ]
    if len(observations) < 2 or audit['status'] != 'MEASURED':
        return
    kinds = {a['id']: a['kind'] for a in scene['asset']['objects']}
    receipts = {
        r['id']: r
        for r in residuals
        if r.get('residual_check_status') == 'PASS'
        and r.get('residual_pixels_confirmed') == 0
        and r.get('checked_owner_pixels', 0) > 0
    }
    assignments = {}
    if digest(source_path) != audit['source_sha256'] or digest(hidden_path) != audit['text_hidden_sha256']:
        return
    with Image.open(source_path) as original, Image.open(hidden_path) as hidden_file:
        source = original.convert('RGB')
        hidden = hidden_file.convert('RGB')
        for oid, obj in scene['selected'].items():
            if oid not in editable_ids or kinds.get(oid) != 'icon' or oid not in receipts:
                continue
            if (
                obj.get('move_safety') != 'MOVABLE'
                or not obj.get('complete_owner_scope')
                or not obj.get('badge_evidence_ids')
            ):
                continue
            ref = obj.get('asset_reference') or {}
            proof = obj.get('owner_evidence') or {}
            receipt = receipts[oid]
            if (
                proof.get('method') != 'measured-badge-ink'
                or proof.get('sha256') != receipt.get('reference_sha256')
                or obj.get('source_sha256') != audit['source_sha256']
                or ref.get('source_sha256') != audit['source_sha256']
                or receipt.get('source_sha256') != audit['source_sha256']
            ):
                continue
            box = list(map(int, obj['bbox']))
            candidates = [r for r in observations if intersection(r['bbox'], box) > 0]
            if len(candidates) < 2:
                continue
            try:
                png = _read_bound(backing, ref['path'], ref['sha256'])
                owner = _read_bound(backing, proof['file'], proof['sha256'])
                delete = _read_bound(
                    hidden_path.parent, f'{page:02d}-{oid}-delete.png', receipt['delete_sha256']
                )
                with np.load(owner, allow_pickle=False) as data, Image.open(png) as asset, Image.open(
                    delete
                ) as deleted_image:
                    if (
                        list(data['restore_bbox']) != box
                        or data['mask'].shape != source.size[::-1]
                        or asset.size != tuple(box[2:])
                        or asset.mode != 'RGBA'
                        or deleted_image.size != source.size
                    ):
                        continue
                    mask = data['mask'].astype(bool)
                    alpha = np.asarray(asset.getchannel('A'))
                    deleted = deleted_image.convert('RGB')
                    parts = []
                    for row in candidates:
                        metrics = _owned_fraction(row['bbox'], box, mask, alpha, source, hidden, deleted)
                        if metrics is None:
                            continue
                        work = directory / ('original-' + '-'.join(map(str, row['bbox'])))
                        work.mkdir(exist_ok=True, parents=True)
                        reading = source_reading(source, row['bbox'], work, job / 'cache')
                        # Strong original-crop evidence protects even a one-letter field.
                        if any(r['score'] >= 0.85 for r in reading['readings']):
                            continue
                        parts.append((row, dict(metrics, source_crop_sha256=reading['crop_sha256'])))
                    for row, metrics in parts:
                        # Independent lines and side-by-side labels can both be real letters.
                        if any(_overlapping_parts(row, other) for other, _ in parts):
                            assignments.setdefault(tuple(row['bbox']), []).append(
                                dict(
                                    metrics,
                                    object_id=oid,
                                    rgba_sha256=ref['sha256'],
                                    owner_sha256=proof['sha256'],
                                    delete_sha256=receipt['delete_sha256'],
                                )
                            )
            except (KeyError, OSError, ValueError, RuntimeError, ImportError):
                continue
    for row in observations:
        proofs = assignments.get(tuple(row['bbox']), [])
        if len(proofs) == 1:
            row.update(
                status='intrinsic_graphic', mark_ids=[proofs[0]['object_id']], graphic_ownership=proofs[0]
            )
