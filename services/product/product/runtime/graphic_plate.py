"""Check identifiable artwork on the raster actually embedded in the edited slide."""
from collections import Counter

import cv2
import numpy as np
from PIL import Image

from ..materialize import digest
from .plate_audit import named_plate


def _signature(source, plate, box, text_mask, contour):
    x, y, w, h = map(round, box)
    height, width = source.shape[:2]
    if min(w, h) < 4 or x < 0 or y < 0 or x+w > width or y+h > height:
        return 'unknown', {}
    # Look for coherent, directional source edges over several parts of the object.
    # Keep a margin so that a rectangle cropped from unchanging scenery cannot
    # register an artificial edge along its own bounding box.
    margin = max(2, min(8, round(min(w, h)*.035)))
    x0, x1 = max(0,x-margin), min(width,x+w+margin)
    y0, y1 = max(0,y-margin), min(height,y+h+margin)
    yy, xx = np.ogrid[y0:y1,x0:x1]
    if contour:
        boundary = np.minimum(np.minimum(xx-x, x+w-1-xx), np.minimum(yy-y, y+h-1-yy))
        allowed = abs(boundary) < margin
    else:
        allowed = (xx >= x+1) & (xx < x+w-1) & (yy >= y+1) & (yy < y+h-1)
    allowed &= ~text_mask[y0:y1,x0:x1]
    def edges(image):
        crop = cv2.GaussianBlur(image[y0:y1,x0:x1], (3,3), 0)
        return np.stack([cv2.Sobel(crop, cv2.CV_32F, dx, dy, ksize=3, scale=.125)
                         for dx, dy in ((1,0),(0,1))], axis=-1)
    original, actual = edges(source), edges(plate)
    strength = np.sqrt((original*original).sum(axis=(-1,-2)))
    valid = allowed & (strength >= 8)
    if valid.sum() < max(12, .025*(w+h)):
        return 'unknown', {}
    norm = np.maximum(1, strength**2)
    agreement = (actual*original).sum(axis=(-1,-2)) / norm
    yy, xx = np.nonzero(valid)
    cells = np.clip((yy+y0-y)*3//h,0,2)*3 + np.clip((xx+x0-x)*3//w,0,2)
    counts = np.bincount(cells, minlength=9)
    matched = np.bincount(cells, weights=agreement[valid]>.52, minlength=9)
    reliable = counts >= max(3, valid.sum()*.015)
    coverage = float((agreement[valid]>.52).mean())
    support = int((reliable & (matched >= counts*.55)).sum())
    evidence = {'edge_agreement':round(coverage,3),'supporting_zones':support,
                'informative_zones':int(reliable.sum())}
    if support >= 2 and coverage >= .68:
        return 'retained', evidence
    if reliable.sum() >= 2 and coverage <= .12:
        return 'source_edges_removed', evidence
    return 'unknown', evidence


def measure(deck, page, snapshot, marks, plans, lines, selected, package_names, sha):
    report = {'pptx_sha256':sha, 'source_sha256':snapshot['source_sha256'], 'rows':[]}
    try:
        with Image.open(snapshot['path']) as image:
            original = np.asarray(image.convert('RGB'))
        if digest(snapshot['path']) != snapshot['source_sha256']:
            raise ValueError('source hash mismatch')
        plate = np.asarray(named_plate(deck, page))
        if plate.shape != original.shape:
            raise ValueError('plate and source dimensions differ')
        hidden = np.zeros(original.shape[:2], bool)
        height, width = hidden.shape
        for line in lines:
            x,y,w,h = map(round,line['bbox'])
            hidden[max(0,y-1):min(height,y+h+1), max(0,x-1):min(width,x+w+1)] = True
        owners = {}
        for oid, record in selected.items():
            if oid in package_names:
                for reference in [oid,record.get('mark_id'),*record.get('seed_ids',[]),
                                  *record.get('badge_evidence_ids',[])]:
                    if reference:
                        owners.setdefault(reference,[]).append(oid)
        marked = {m['id']:m for m in marks}
        targets = {}
        for plan in plans:
            oid = plan['id']
            mark = marked.get(oid,{})
            if plan['route']=='asset' and mark.get('label') in ('image','connected_diagram','icon','illustration'):
                targets[oid] = (mark['label'],mark['bbox'],False)
        for oid, record in selected.items():
            if record.get('kind') == 'surface' and oid in package_names:
                targets[oid] = ('surface',record['bbox'],True)
        for oid, (kind,box,contour) in targets.items():
            # A full-page layout detection carries no evidence of a single object.
            if box[2]*box[3] >= width*height*.9:
                status, evidence = 'unknown', {}
            else:
                status,evidence = _signature(original, plate, box, hidden, contour)
            report['rows'].append({'id':oid, 'kind':kind, 'bbox':box, 'status':status,
                                   'package_object_ids':sorted(set(owners.get(oid,[]))), **evidence})
        return {**report,'status':'MEASURED','counts':dict(Counter(r['status'] for r in report['rows']))}
    except (OSError, KeyError, ValueError, cv2.error) as exc:
        return {**report,'status':'UNKNOWN','reason':type(exc).__name__}
