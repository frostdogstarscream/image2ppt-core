"""Transfer source-supported text strokes from bound card artwork on a private stage copy."""

import os
from subprocess import SubprocessError
import numpy as np
from PIL import Image

from ..materialize import digest
from ..parallel_rebuild import array_sha


def read_decoration_carriers(graph, group, baseline, directory, cache, lines):
    """Independently read single-seed cutouts fully inside a complete source field."""
    from ..contracts import intersection
    from ..parallel_branches import local_ocr, snapshot
    manifest = os.environ.get('IMAGE2PPT_OCR_MANIFEST')
    if not manifest:
        return {}
    words = {r['id']: r['text'] for r in lines}
    fields = {t.id: t for t in graph.text.texts if t.id in group['members']}
    seeds = {a.id for a in graph.asset.objects if a.kind == 'decoration' and not a.intrinsic_text}
    carriers = {}
    for oid, rec in graph.selected.items():
        ids = rec.get('seed_ids', [])
        if (len(ids) != 1 or ids[0] not in seeds or rec.get('move_safety') != 'MOVABLE'
                or not rec.get('payload', {}).get('upgraded_from_region_crop')):
            continue
        ref = rec['asset_reference']
        box = ref['bbox']
        matches = [t for t in fields.values() if intersection(t.bbox, box) >= box[2]*box[3]]
        if len(matches) != 1:
            continue
        field = matches[0]
        expected = ''.join(''.join(words.get(i, '').split()) for i in graph.text.members.get(field.id, []))
        if not expected or expected != ''.join(field.text.split()):
            continue
        path = (baseline/ref['path']).resolve()
        if not path.is_relative_to(baseline.resolve()) or digest(path) != ref['sha256']:
            raise ValueError('changed decoration candidate')
        with Image.open(path) as image:
            rgba = image.convert('RGBA')
        # Scale a copy for the local OCR detector. Only original pixels authorize transfer.
        sample = Image.new('RGBA', (rgba.width+12, rgba.height+12), 'white')
        sample.alpha_composite(rgba, (6,6))
        folder = directory/oid
        folder.mkdir(parents=True, exist_ok=True)
        crop = folder/'reading.png'
        sample.convert('RGB').resize((sample.width*2, sample.height*2), Image.Resampling.LANCZOS).save(crop)
        try:
            evidence = local_ocr(snapshot(crop, folder/'snapshots'), cache, manifest)
        except (OSError, RuntimeError, SubprocessError):
            # Missing secondary OCR cannot authorize removal of an existing picture.
            continue
        rows = evidence['lines']
        if len(rows) == 1:
            reading = ''.join(rows[0]['text'].split())
            if rows[0]['score'] >= .9 and sum(c.isalnum() for c in reading) >= 2 and expected.count(reading) == 1:
                carriers[oid] = {'field':field.id, 'seed':ids[0], 'reading':reading,
                                 'ocr_identity':evidence['identity'], 'asset_sha256':ref['sha256']}
    return carriers


def verify_text_cutout(directory, record, evidence, field, mask, box, clean):
    """Prove every visible mark on a removable cutout and the paint that will be revealed."""
    ref = record['asset_reference']
    owner = record['owner_evidence']
    path, proof = ((directory/n).resolve() for n in (ref['path'],owner['file']))
    if (not all(p.is_relative_to(directory.resolve()) for p in (path,proof))
            or evidence['asset_sha256'] != ref['sha256'] or digest(path) != ref['sha256']
            or digest(proof) != owner['sha256'] or record.get('seed_ids') != [evidence['seed']]
            or not record.get('payload', {}).get('upgraded_from_region_crop')
            or record.get('move_safety') != 'MOVABLE' or not evidence.get('ocr_identity')
            or not evidence['reading'] or evidence['reading'] not in ''.join(field.split())):
        raise ValueError('retired cutout source or reading changed')
    x,y,w,h = map(int,ref['bbox'])
    tx,ty,tw,th = box
    if x < tx or y < ty or x+w > tx+tw or y+h > ty+th:
        raise ValueError('cutout crosses text scope')
    with np.load(proof, allow_pickle=False) as stored:
        if list(stored['restore_bbox']) != [x,y,w,h]:
            raise ValueError('cutout restoration bounds changed')
        backing = stored['backing_patch'].astype(float)
        owned = stored['mask'][y:y+h,x:x+w].astype(bool)
    rgba = np.asarray(Image.open(path).convert('RGBA'))
    composite = backing + (rgba[:,:,:3] - backing)*(rgba[:,:,3:]/255)
    visible = abs(composite-backing).max(2) > 10
    author = mask[y:y+h,x:x+w] & owned
    desired = clean[y-ty:y-ty+h,x-tx:x-tx+w]
    with Image.open(directory/'source.png') as image:
        observed = np.asarray(image.convert('RGB').crop((x,y,x+w,y+h)))
    if (not visible.any() or (visible & ~author).any()
            or np.max(abs(observed[visible].astype(float) - composite[visible])) > 12
            or np.max(abs(desired[visible].astype(float) - backing[visible])) > 14):
        raise ValueError('cutout has independent artwork or an unproved backing')


def transfer_text_ink(directory, artwork, mask, text_box, clean_patch, source_sha):
    reference = artwork["asset_reference"]
    asset = (directory / reference["path"]).resolve()
    proof = (directory / artwork["owner_evidence"]["file"]).resolve()
    if not all(path.is_relative_to(directory.resolve()) for path in (asset, proof)):
        raise ValueError("artwork reference outside candidate")
    if (
        not artwork.get("retained_card_artwork")
        or reference.get("source_sha256") != source_sha
        or digest(asset) != reference["sha256"]
        or digest(proof) != artwork["owner_evidence"]["sha256"]
    ):
        raise ValueError("card artwork/source binding mismatch")
    with np.load(proof, allow_pickle=False) as stored:
        owner = stored["mask"].astype(bool)
        box = stored["restore_bbox"].copy()
        underlay = stored["backing_patch"].copy()
    if (mask & ~owner).any() or not mask.any() or list(box) != list(reference["bbox"]):
        raise ValueError("card artwork does not exclusively own the requested strokes")
    x, y, w, h = map(int, box)
    tx, ty, tw, th = text_box
    local = mask[y : y + h, x : x + w]
    if local.sum() != mask.sum() or mask[ty : ty + th, tx : tx + tw].sum() != mask.sum():
        raise ValueError("text strokes cross artwork or text boundary")
    rgba = np.array(Image.open(asset).convert("RGBA"))
    if rgba.shape != (h, w, 4) or underlay.shape != (h, w, 3):
        raise ValueError("card artwork dimensions changed")
    alpha = rgba[:, :, 3:][local].astype(float) / 255
    if alpha.min() < 0.95:
        raise ValueError("text is shared with a translucent parent")
    observed = np.array(Image.open(directory / "source.png").convert("RGB"))[mask]
    underneath = underlay[local].astype(float)
    composite = rgba[:, :, :3][local] * alpha + underneath * (1 - alpha)
    if np.abs(composite - observed).max() > 4:
        raise ValueError("retained artwork no longer reproduces original strokes")
    desired = clean_patch[mask[ty : ty + th, tx : tx + tw]]
    restored = (desired - underneath * (1 - alpha)) / alpha
    if restored.min() < -2 or restored.max() > 257:
        raise ValueError("clean paint cannot be represented by retained artwork")
    rgba[:, :, :3][local] = np.clip(np.rint(restored), 0, 255).astype("uint8")
    reveal = np.abs(desired.astype(float) - underneath).max(1) <= 3
    rgba[:, :, 3][local] = np.where(reveal, 0, rgba[:, :, 3][local])
    Image.fromarray(rgba).save(asset)
    owner &= ~mask
    np.savez_compressed(proof, mask=owner, restore_bbox=box, backing_patch=underlay)
    reference["sha256"] = digest(asset)
    artwork["mask_sha256"] = array_sha(owner.astype("uint8"))
    artwork["owner_evidence"]["sha256"] = digest(proof)
