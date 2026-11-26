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
    def analyze(cls, source, original_mask, protected, bbox, backing, *, glyph_height, light_paragraph=False):
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
        halo_pixels = 0
        if light_paragraph:
            # Codec ringing can run through the pale backing between two OCR lines.
            # Require a short source-contrast path to ink, dense local stroke support
            # and an independently fitted light ground. Never cross protected ink.
            x, y, w, h = map(round, bbox)
            region = np.s_[y:y+h, x:x+w]
            seed = original_mask[region]
            strong = source[region][seed]
            ground = backing[region].astype(float)
            if np.percentile(ground, 5) >= 195 and np.median(ground[seed]-strong) > 90:
                radius_px = min(8, max(2, round(glyph_height*.34)))
                # Re-test components above the weak noise floor. A one-level
                # bridge can join a real stroke to an entire benign gradient.
                count, islands, stats, _ = cv2.connectedComponentsWithStats(
                    (delta[region] > 4).astype(np.uint8), 8)
                for i in range(1, count):
                    island = islands == i
                    if ((island & seed).sum() >= .35 * stats[i, cv2.CC_STAT_AREA]
                            and distance[region][island].max() <= radius
                            and not (protected[region] & island).any()):
                        semantic[region] |= island
                width = 2*radius_px+1
                kernel = np.ones((3,3), np.uint8)
                density = cv2.blur(seed.astype(np.float32), (width,width))
                peer = cv2.dilate(protected[region].astype(np.uint8), kernel) > 0
                fringe = ((delta[region] > 3) & (delta[region] <= 24) & (density >= .15)
                          & (distance[region] <= radius_px) & ~peer)
                grown = seed | antialias[region]
                for _ in range(radius_px):
                    grown |= (cv2.dilate(grown.astype(np.uint8), kernel) > 0) & fringe
                # A weaker second bright lobe can follow a dark printed edge.
                # Verify the four samples in order, including a return to ground.
                signed = source[region].astype(float) - ground
                bright = signed.mean(2)
                for axis in (0, 1):
                    for direction in (-1, 1):
                        def lag(array, count):
                            return np.roll(array, direction*count, axis=axis)
                        echo = (lag(seed,3) & (lag(bright,2) >= bright*1.5)
                                & (np.abs(lag(bright,1)) <= 3) & (bright >= 3)
                                & (signed.min(2) >= -2) & (delta[region] <= 12)
                                & ~(peer | lag(peer,1) | lag(peer,2) | lag(peer,3)))
                        boundary = [slice(None), slice(None)]
                        boundary[axis] = slice(0,3) if direction > 0 else slice(-3,None)
                        echo[tuple(boundary)] = False
                        semantic[region] |= echo
                extra = grown & fringe & ~semantic[region]
                halo_pixels = int(extra.sum())
                semantic[region] |= extra
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
                  'semantic_components': component_evidence, 'corroborated_halo_pixels': halo_pixels,
                  'unknown_note': 'Includes unchanged low-contrast background; visible unassigned signal blocks staging',
                  'semantic_authority': 'resolved source text plus bounded source-contrast stroke connections'}
        for mask in [*evidence.values(), *masks.values()]:
            mask.setflags(write=False)
        return cls(evidence, masks, report)
