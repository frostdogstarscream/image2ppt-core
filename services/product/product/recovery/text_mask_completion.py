"""Source-derived text evidence. Search rectangles never grant erase authority."""
from dataclasses import dataclass

import cv2
import numpy as np

from ..object_extract import rectangle
from .background_adapter import RecoveryBlocked, _sha


@dataclass(frozen=True)
class TextMaskCompletion:
    evidence: dict
    masks: dict
    report: dict

    @classmethod
    def analyze(cls, source, original_mask, protected, bbox, backing, *, glyph_height):
        if (original_mask.dtype != bool or protected.dtype != bool
                or original_mask.shape != source.shape[:2] or protected.shape != original_mask.shape
                or backing.shape != source.shape or not original_mask.any()):
            raise RecoveryBlocked('INVALID_MASK', 'completion requires source-sized evidence')
        if (original_mask & protected).any():
            raise RecoveryBlocked('PROTECTED_NEIGHBOR', 'confirmed text overlaps protected peer')
        scope = rectangle(bbox, source.shape[1::-1])
        if (original_mask & ~scope).any():
            raise RecoveryBlocked('MASK_OUTSIDE_SCOPE', 'confirmed ink leaves text search scope')
        allowed = scope & ~protected
        delta = np.abs(source.astype(float) - backing.astype(float)).max(2)
        near_one = cv2.dilate(original_mask.astype('uint8'), np.ones((3, 3), np.uint8)) > 0
        # Include both dark AA and bright compression ringing. Every added pixel
        # needs measured source contrast and a bounded link to confirmed strokes.
        antialias = allowed & ~original_mask & near_one & (delta > 3)
        if not np.isfinite(glyph_height) or glyph_height <= 0:
            raise ValueError('source text glyph height must be positive')
        distance = cv2.distanceTransform((~original_mask).astype('uint8'), cv2.DIST_L2, 5)
        radius = min(12.0, max(3.0, glyph_height/2))
        signal = scope & (delta > 3)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(signal.astype('uint8'), 8)
        semantic = original_mask.copy()
        component_evidence = []
        for index in range(1, count):
            component = labels == index
            seeds = int((component & original_mask).sum())
            coverage = seeds / int(stats[index, cv2.CC_STAT_AREA])
            peer_conflict = bool((component & protected).any())
            extent = float(distance[component].max())
            accepted = coverage >= .35 and extent <= radius and not peer_conflict
            if accepted:
                semantic |= component
            component_evidence.append({'component': index, 'pixels': int(component.sum()),
                'confirmed_seed_fraction': coverage, 'max_seed_distance': extent,
                'protected_conflict': peer_conflict, 'accepted': accepted})
        possible = semantic & allowed & ~original_mask & ~antialias
        evidence = {'confirmed_text': original_mask.copy(), 'text_antialias': antialias,
                    'possible_text': possible, 'protected_peer': protected & scope}
        evidence['unknown'] = scope & ~np.logical_or.reduce(list(evidence.values()))
        masks = {'original': original_mask.copy(),
                 'bounded_expansion': original_mask | antialias,
                 'semantic_completion': original_mask | antialias | possible}
        unassigned = evidence['unknown'] & (delta > 6)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(unassigned.astype('uint8'), 8)
        components = [{'pixels': int(stats[i, cv2.CC_STAT_AREA]),
                       'bbox': [int(v) for v in stats[i, :4]]} for i in range(1, count)]
        report = {'schema': 'TextMaskCompletion/1', 'source_pixels_sha256': _sha(source),
                  'original_mask_sha256': _sha(original_mask), 'backing_sha256': _sha(backing),
                  'counts': {name: int(mask.sum()) for name, mask in evidence.items()},
                  'mask_counts': {name: int(mask.sum()) for name, mask in masks.items()},
                  'unassigned_contrast_pixels': int(unassigned.sum()),
                  'unassigned_components': components, 'max_completion_radius': radius,
                  'semantic_components': component_evidence,
                  'unknown_note': 'Includes unchanged low-contrast background; visible unassigned signal blocks staging',
                  'semantic_authority': 'resolved source text plus bounded source-contrast stroke connections'}
        for mask in [*evidence.values(), *masks.values()]:
            mask.setflags(write=False)
        return cls(evidence, masks, report)
