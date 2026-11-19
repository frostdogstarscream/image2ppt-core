"""Centralized two-tier gate configuration: immutable hard safety vs soft visual tolerances."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / 'configs/fidelity_tolerance.json'

DEFAULT_TOLERANCE: dict = {
    'schema': 'FidelityTolerance/1',
    'profile': 'opencode_native_first_v1',
    'hard_gates': {
        'forbid_duplicate_text': True,
        'forbid_unknown_erasure': True,
        'forbid_residual_ink_ghost': True,
        'require_actual_office': True,
    },
    'fidelity': {
        'max_delta_e2000_p95': 10.0,
        'hard_max_delta_e2000_p95': 26.0,
        'min_ssim': 0.85,
        'hard_min_ssim': 0.72,
        'max_mae_rgb': 12.0,
    },
    'residual_gate': {
        'plate_backing_rgb': 8.0,
        'native_surface_backing_rgb': 26.0,
        'source_ink_contrast_rgb': 12.0,
    },
    'shape_fitting': {
        'solid_interior_p95_rgb': 13.5,
        'gradient_interior_p95_rgb': 16.5,
        'border_segment_max_rgb': 15.0,
        'max_unexplained_pixel_fraction': 0.035,
        'max_unexplained_component_min_side_px': 4,
        'max_unexplained_component_area_px': 20,
        'shadow_halo_pad_px': 7,
        'shadow_max_rgb_diff': 28.0,
    },
    'scorecard_gate': {
        'target_raster_plate_content_fraction': 0.15,
        'warn_raster_plate_content_fraction': 0.35,
        'max_succeeded_raster_plate_content_fraction': 0.85,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    # Hard gates can never be disabled through config overrides.
    merged['hard_gates'] = dict(DEFAULT_TOLERANCE['hard_gates'])
    return merged


@lru_cache(maxsize=8)
def _load_cached(path_str: str, mtime_ns: int) -> dict:
    del mtime_ns
    raw = json.loads(Path(path_str).read_text('utf-8'))
    return _deep_merge(DEFAULT_TOLERANCE, raw)


def load_tolerance_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path or os.environ.get('IMAGE2PPT_TOLERANCE_CONFIG') or DEFAULT_CONFIG_PATH)
    if not cfg_path.is_file():
        return _deep_merge(DEFAULT_TOLERANCE, {})
    return _load_cached(str(cfg_path.resolve()), cfg_path.stat().st_mtime_ns)
