"""LayerD-lower-first, Big-LaMa-optional background candidate generation.

No source-plate write occurs in this module. Caller must explicitly construct
an authorized region mask from resolved source ownership, and must validate
trial PPTX/Office edits and visual quality before committing any candidate.

This is a standalone r12 adapter to be wired into the r11 candidate engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class BackgroundCandidate:
    method: str
    image: np.ndarray = field(repr=False)
    write_mask: np.ndarray = field(repr=False)
    source_sha256: str
    mask_sha256: str
    diagnostics: dict
    status: str = "PRECHECK_PASS"  # Not an Office or release approval.


class RecoveryBlocked(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(detail)


def _sha(array: np.ndarray) -> str:
    # Include dtype and shape to bind the exact array interpretation.
    return sha256((str(array.dtype) + ":" + repr(array.shape)).encode() + array.tobytes()).hexdigest()


def _validate(source, write_mask, protected, lower):
    if not isinstance(source, np.ndarray) or source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise RecoveryBlocked("INVALID_SOURCE", "source must be HxWx3 uint8 RGB")
    h, w = source.shape[:2]
    for name, mask in (("write_mask", write_mask), ("protected", protected)):
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != (h, w):
            raise RecoveryBlocked("INVALID_MASK", f"{name} must be HxW bool")
    if not write_mask.any():
        raise RecoveryBlocked("EMPTY_WRITE_MASK", "no authorized pixels")
    if np.any(write_mask & protected):
        raise RecoveryBlocked("PROTECTED_NEIGHBOR", "write mask overlaps protected pixels")
    if lower is not None and (not isinstance(lower, np.ndarray) or lower.shape != source.shape or lower.dtype != np.uint8):
        raise RecoveryBlocked("INVALID_LAYER", "LayerD lower composite must match RGB source dimensions")


def _ring(write_mask: np.ndarray, protected: np.ndarray, radius: int = 8) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return (cv2.dilate(write_mask.astype(np.uint8), kernel) > 0) & ~write_mask & ~protected


def _high_frequency(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.6))


def _assess(source, repaired, write_mask, protected, *, context_reference=None, ring_mae_limit=3.0, ring_p95_limit=8.0,
            residual_fraction_limit=0.01) -> dict:
    sample = _ring(write_mask, protected)
    # Validate the proposed backing against untouched contextual source pixels.
    # Comparing the composited trial outside write_mask would always yield zero.
    context = repaired if context_reference is None else context_reference
    delta = np.abs(source.astype(np.int16) - context.astype(np.int16)).mean(axis=2)
    if not sample.any():
        return {"status": "UNKNOWN", "reason": "NO_CLEAN_CONTEXT_RING"}
    ring_mae = float(np.mean(delta[sample]))
    ring_p95 = float(np.percentile(delta[sample], 95))
    old_edges, new_edges = _high_frequency(source), _high_frequency(repaired)
    # The residual test is a diagnostic proxy, not independent OCR evidence.
    old_ink = write_mask & (old_edges > 12.0)
    if not old_ink.any():
        return {"status": "UNKNOWN", "reason": "NO_SOURCE_STROKE_EVIDENCE",
                "ring_mae": round(ring_mae, 4), "ring_p95": round(ring_p95, 4)}
    residual_fraction = float(np.mean(new_edges[old_ink] > 6.0))
    passed = (ring_mae <= ring_mae_limit and ring_p95 <= ring_p95_limit
              and residual_fraction <= residual_fraction_limit)
    return {"status": "PASS" if passed else "REVIEW_REQUIRED",
            "ring_mae": round(ring_mae, 4), "ring_p95": round(ring_p95, 4),
            "source_stroke_pixels": int(old_ink.sum()),
            "residual_fraction": round(residual_fraction, 5),
            "reference_note": "Old r7.1 thresholds are migration baselines, not production calibration"}


def generate_background_candidates(
    source: np.ndarray,
    write_mask: np.ndarray,
    protected: np.ndarray,
    *,
    lower_rgb: Optional[np.ndarray] = None,
    lower_provenance_verified: bool = False,
    inpaint: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    allow_inpaint_after_lower: bool = False,
) -> tuple[list[BackgroundCandidate], list[dict]]:
    """Return immutable-trial candidate copies and structured rejected attempts.

    ``lower_rgb`` must be a correctly composited visible underlayer (not merely
    layer index zero). Verify LayerD manifest/hash/alpha/z-order upstream.
    ``inpaint`` is the host-provided Big-LaMa adapter, called only when needed.

    A candidate is not a committed output even if diagnostic status says PASS.
    The caller must verify ownership, OCR residue, editing, and whole-page QA.
    """
    _validate(source, write_mask, protected, lower_rgb)
    source_sha, mask_sha = _sha(source), _sha(write_mask)
    results: list[BackgroundCandidate] = []
    issues: list[dict] = []
    if lower_rgb is not None:
        if not lower_provenance_verified:
            issues.append({"method": "layerd_lower", "code": "UNVERIFIED_LOWER_PROVENANCE"})
        else:
            trial = source.copy()
            trial[write_mask] = lower_rgb[write_mask]
            metrics = _assess(source, trial, write_mask, protected, context_reference=lower_rgb)
            if metrics["status"] == "PASS":
                trial.setflags(write=False)
                m = write_mask.copy(); m.setflags(write=False)
                results.append(BackgroundCandidate("layerd_lower", trial, m, source_sha, mask_sha, metrics))
            else:
                issues.append({"method": "layerd_lower", "code": "LOWER_NOT_VERIFIED", "metrics": metrics})
    if inpaint is not None and (allow_inpaint_after_lower or not results):
        # Separate full-frame context and its explicitly authorized mask.
        proposed = inpaint(source.copy(), write_mask.copy())
        if not isinstance(proposed, np.ndarray) or proposed.dtype != np.uint8 or proposed.shape != source.shape:
            raise RecoveryBlocked("INVALID_INPAINT_RESPONSE", "Big-LaMa returned invalid image")
        trial = source.copy()
        trial[write_mask] = proposed[write_mask]
        metrics = _assess(source, trial, write_mask, protected, context_reference=proposed)
        if metrics["status"] != "PASS":
            issues.append({"method": "big_lama", "code": "INPAINT_NEEDS_REVIEW", "metrics": metrics})
        # Keep as a trial even when visually uncertain, but never auto-approve.
        trial.setflags(write=False)
        m = write_mask.copy(); m.setflags(write=False)
        results.append(BackgroundCandidate("big_lama", trial, m, source_sha, mask_sha, metrics,
                                           status="PRECHECK_PASS" if metrics["status"] == "PASS" else "REVIEW_REQUIRED"))
    if not results:
        issues.append({"method": "source_fallback", "code": "NO_VALID_REPAIR_CANDIDATE"})
    return results, issues
