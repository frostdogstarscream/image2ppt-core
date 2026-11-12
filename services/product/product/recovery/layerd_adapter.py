"""Hash-verified LayerD prefix composites for background candidate generation."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from ..materialize import digest
from ..object_extract import load_page, recompose
from .background_adapter import RecoveryBlocked


@dataclass(frozen=True)
class LowerBackground:
    image: np.ndarray
    provenance: dict


def lower_backgrounds(source, layer_directory, write_mask, *, exploratory=False):
    """Enumerate actual lower stacks, without assuming the first layer is clean.

    Each prefix is a hypothesis. Source reconstruction and opaque coverage prove
    its origin, while background_adapter separately assesses its usefulness.
    """
    page = load_page(layer_directory)
    original = np.asarray(Image.open(page.source_path).convert('RGB'))
    if source.dtype != np.uint8 or source.shape != original.shape or not np.array_equal(source, original):
        raise RecoveryBlocked('LAYER_SOURCE_MISMATCH', 'LayerD source pixels differ from the target')
    if write_mask.dtype != bool or write_mask.shape != source.shape[:2] or not write_mask.any():
        raise RecoveryBlocked('INVALID_MASK', 'lower background requires a source-sized glyph mask')
    error = float(np.abs(recompose(page).astype(float) - source).mean())
    if not exploratory and (not np.isfinite(error) or error > 3):
        raise RecoveryBlocked('LAYER_RECOMPOSITION_MISMATCH', 'LayerD stack does not reconstruct the source')
    transparency = np.ones(source.shape[:2], dtype=np.float32)
    records = []
    for index, layer in enumerate(page.layers[:-1]):
        with Image.open(layer.path) as image:
            alpha = np.asarray(image.convert('RGBA'))[..., 3].astype(np.float32) / 255
        transparency *= 1 - alpha
        if not exploratory and not np.all(transparency[write_mask] == 0):
            continue
        rgb = recompose(replace(page, layers=page.layers[:index + 1]))
        rgb.setflags(write=False)
        records.append(LowerBackground(rgb, {
            'manifest_sha256': digest(Path(layer_directory) / 'manifest.json'),
            'source_file_sha256': digest(page.source_path),
            'layers': [{'file': item.file, 'sha256': item.sha256, 'z': item.z}
                       for item in page.layers if item.z <= layer.z],
            'below_z': layer.z + 1,
            'recomposition_mae': error,
            'mask_opaque_coverage': float(np.mean(transparency[write_mask] == 0)),
        }))
    return records
