"""Transfer source-supported text strokes from bound card artwork on a private stage copy."""

import numpy as np
from PIL import Image

from ..materialize import digest
from ..parallel_rebuild import array_sha


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
    Image.fromarray(rgba).save(asset)
    owner &= ~mask
    np.savez_compressed(proof, mask=owner, restore_bbox=box, backing_patch=underlay)
    reference["sha256"] = digest(asset)
    artwork["mask_sha256"] = array_sha(owner.astype("uint8"))
    artwork["owner_evidence"]["sha256"] = digest(proof)
