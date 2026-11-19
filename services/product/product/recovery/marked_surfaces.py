"""Recover corroborated card geometry while retaining its remaining foreground."""

import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from ..contracts import Shape, intersection
from ..materialize import digest, write
from ..object_contracts import ClosedObject
from ..parallel_contracts import SceneGraph
from ..parallel_fusion import fuse_incremental_scene
from ..parallel_rebuild import adapt, array_sha
from ..parallel_store import key
from .container_stage import _discover_candidates, _ring_parent_patch
from .native_shapes import estimate_canvas_field, hex_rgb as _hex
from .marked_ownership import record_claims


def _fit(tile, parent):
    h, w = tile.shape[:2]
    if min(w, h) < 22:
        return None
    fill = np.median(tile[5:-5, 5:-5].reshape(-1, 3), axis=0)
    delta = np.abs(tile.astype(float) - fill).max(2)
    if np.percentile(delta[5:-5, 5:-5], 65) > 6 or np.abs(fill - parent).max() < 5:
        return None
    # Require the same distinct stroke on all four sides. A photo or an
    # enclosing text box alone provides no evidence of a native rectangle.
    colors = []
    for edge in (tile[:4], tile[-4:], tile[:, :4].transpose(1, 0, 2), tile[:, -4:].transpose(1, 0, 2)):
        trim = min(16, edge.shape[1] // 5)
        edge = edge[:, trim:-trim].astype(float)
        strength = np.minimum(abs(edge - fill).max(2), abs(edge - parent).max(2))
        if np.percentile(strength.max(0), 20) < 10:
            return None
        colors.append(np.median(edge[strength.argmax(0), np.arange(edge.shape[1])], axis=0))
    if np.ptp(colors, axis=0).max() > 65:
        return None
    return np.rint(fill).astype(np.uint8), np.median(colors, axis=0)


def stage_marked_surfaces(baseline, output, routes, skip_ids):
    from reconstruction import extract_rgba
    from ..runtime.marked_routes import iou

    baseline, output = Path(baseline), Path(output)
    graph = SceneGraph.model_validate_json((baseline / 'scene.json').read_text('utf8'))
    plate_ref = graph.raster_backing
    src = baseline / 'source.png'
    plate_path = baseline / plate_ref['path']
    if digest(src) != graph.source_sha256 or digest(plate_path) != plate_ref['sha256']:
        raise ValueError('stale card source or plate')
    plate = np.array(Image.open(plate_path).convert('RGBA'))
    canvas, _, scenery, flat = estimate_canvas_field(plate[..., :3])
    boxes = [b for k, b in _discover_candidates(plate[..., :3], canvas, scenery, is_flat_canvas=flat)
             if k == 'box']
    staged = []
    for field in routes:
        box = max(boxes, key=lambda b: iou(b, field['bbox']), default=None)
        if not box or iou(box, field['bbox']) < 0.80:
            continue
        x, y, w, h = box
        sid = 's-' + key([graph.source_sha256, 'marked-surface', box])[:16]
        aid = 'a-' + key([sid, 'retained-ink'])[:16]
        if sid in graph.selected or {field['id'], sid} & skip_ids:
            continue
        peers = [r for r in graph.selected.values() if intersection(box, r['bbox'])]
        from .layered_card import contained_peers, frame_layers
        layered = field.get('label') == 'card_frame'
        if not contained_peers(box, peers):
            continue
        if layered != any(r['kind']=='asset' for r in peers):
            continue
        patch = plate[y:y+h,x:x+w,:3].copy()
        ring = _ring_parent_patch(plate[..., :3], canvas, box)
        if ring is None or ring[2] > (8 if layered else 4) or scenery[y:y+h,x:x+w].any():
            continue
        parent_patch = np.clip(np.rint(ring[0]), 0, 255).astype(np.uint8)
        parent = np.clip(np.rint(ring[1]), 0, 255).astype(np.uint8)
        gradient = None
        if layered:
            ignored = [o.bbox for o in [*graph.text.texts,*graph.asset.objects]
                       if o.bbox[2]*o.bbox[3] < .65*w*h and intersection(box,o.bbox)]
            result = frame_layers(patch, parent_patch, box, peers, baseline, ignored)
            if result is None:
                continue
            fill, gradient, backing_field, owned, ink, rgba = result
            stroke = fill
            radius = min(12,h//4,w//4)
        else:
            fit = _fit(patch, parent)
            if fit is None:
                continue
            fill, stroke = fit
            # Preserve every interior pixel significantly different from the
            # measured fill in a transparent foreground. Exclude the proven edge.
            delta = abs(patch.astype(float) - fill).max(2)
            interior = np.zeros((h, w), bool)
            interior[4:-4, 4:-4] = True
            ink = interior & (delta > 6)
            # Anti-aliased rounded corners belong to the frame. Keep all other
            # interior ink, including text the reconstruction has yet to recover.
            radius = min(12, h // 4, w // 4)
            corners = np.zeros_like(ink)
            for ys in (slice(0, radius), slice(h - radius, h)):
                for xs in (slice(0, radius), slice(w - radius, w)):
                    corners[ys, xs] = True
            ink &= ~corners
            # An unknown foreground touching an edge invalidates this decomposition.
            edge = ~interior | corners
            from .container_stage import _segment_max_err

            # Curves can expose fully covered stroke pixels where the horizontal
            # one-pixel edge samples were antialiased. Test their colour direction.
            stronger = np.clip(fill + 2 * (stroke - fill), 0, 255)
            sample, paper, face = patch.astype(float), parent.astype(float), fill.astype(float)
            explain = np.minimum.reduce(
                [
                    _segment_max_err(sample, face, stroke),
                    _segment_max_err(sample, paper, stroke),
                    _segment_max_err(sample, face, stronger),
                    _segment_max_err(sample, paper, stronger),
                    _segment_max_err(sample, paper, face),
                ]
            )
            if np.max(explain[edge]) > 28:
                continue
            # A faint one-pixel colour transition along the measured frame is part
            # of that frame. Emitting it as a separate picture creates an occluded
            # owner beneath the label and a spurious selectable object.
            rim = np.ones_like(ink)
            rim[5:-5, 5:-5] = False
            ink &= ~(rim & (delta <= 14) & (explain <= 4))
            # Previously staged text already has its own backing proof.
            owned = np.zeros((h, w), bool)
            child_backings_match = True
            for r in peers:
                ref = r['owner_evidence']
                file = baseline / ref['file']
                if digest(file) != ref['sha256']:
                    raise ValueError('card child owner mismatch')
                with np.load(file, allow_pickle=False) as data:
                    owned |= data['mask'][y : y + h, x : x + w].astype(bool)
                    px,py,pw,ph=map(int,data['restore_bbox'])
                    child_mask=data['mask'][py:py+ph,px:px+pw].astype(bool)
                    # Deleting a child in Office will reveal this fill. Preserve its
                    # original backing if any proven pixel would exceed the gate.
                    if np.any(abs(data['backing_patch'][child_mask].astype(float)-fill)>8):
                        child_backings_match=False
                        break
            if not child_backings_match:
                continue
            ink &= ~owned
            if ink.any() and aid in skip_ids:
                continue
            rgba = extract_rgba(patch, np.broadcast_to(fill, patch.shape), ink)
            alpha = rgba[..., 3:4].astype(float) / 255
            if ink.any() and abs((rgba[..., :3] * alpha + fill * (1 - alpha) - patch)[ink]).max() > 2:
                continue
        shape = Shape(
            id=sid,
            bbox=box,
            z=len(graph.surface.shapes) + 1,
            geometry='roundRect',
            fill=_hex(fill),
            line_color=_hex(stroke),
            line_width_pt=0 if layered else 0.75,
            fill_gradient=gradient,
            corner_radius_px=float(radius),
        )
        assets = []
        if ink.any():
            yy, xx = np.where(ink)
            ix, iy, iw, ih = int(xx.min()), int(yy.min()), int(np.ptp(xx) + 1), int(np.ptp(yy) + 1)
            ibox = [x + ix, y + iy, iw, ih]
            assets.append(
                ClosedObject(
                    id=aid,
                    kind='decoration',
                    bbox=ibox,
                    layers=['source-plate.png'],
                    components=['foreground'],
                    parts=[{'name': 'foreground', 'bbox': ibox, 'layers': ['source-plate.png']}],
                    text_policy='exclude_independent_text',
                    intrinsic_text='',
                    replaces=[],
                    description='Measured foreground artwork' if layered else 'Retained card foreground pixels',
                    prompt_en='retained source foreground',
                )
            )
        fused = fuse_incremental_scene(
            baseline, base_graph=graph, added_shapes=[shape], added_assets=assets
        )
        if any(fused.ownership.get(k) != 'RESOLVED' for k in [sid, *[a.id for a in assets]]):
            continue
        if not staged:
            shutil.copytree(baseline, output)
        graph = fused
        footprint = np.ones_like(ink)
        if layered:
            yy, xx = np.ogrid[:h, :w]
            dx = np.maximum(radius-xx-.5, np.maximum(xx+.5-(w-radius), 0))
            dy = np.maximum(radius-yy-.5, np.maximum(yy+.5-(h-radius), 0))
            footprint = dx*dx+dy*dy <= radius*radius
        for identifier, kind, bbox, local_mask, backing in [
            (sid, 'surface', box, footprint & ~(ink | owned), parent_patch)
        ] + ([(aid, 'asset', ibox, (ink & ~owned)[iy:iy+ih, ix:ix+iw], backing_field[iy:iy+ih, ix:ix+iw] if layered else fill)] if assets else []):
            ax, ay, aw, ah = bbox
            mask = np.zeros(plate.shape[:2], bool)
            mask[ay : ay + ah, ax : ax + aw] = local_mask
            proof = output / (identifier + '-owner.npz')
            np.savez_compressed(
                proof, mask=mask, restore_bbox=bbox, backing_patch=np.broadcast_to(backing, (ah, aw, 3))
            )
            rec = {
                'id': identifier,
                'kind': kind,
                'bbox': bbox,
                'restore_bbox': bbox,
                'state': 'STAGED',
                'quality': 'REVIEW_REQUIRED',
                'human_acceptance': False,
                'source_sha256': graph.source_sha256,
                'original_owner': 'source-plate',
                'complete_owner_scope': True,
                'mask_sha256': array_sha(mask.astype('uint8')),
                'owner_evidence': {
                    'file': proof.name,
                    'sha256': digest(proof),
                    'method': 'four-sided-card-and-interior',
                },
                'method': 'marked-card/1',
                'mark_id': field['id'],
                'payload': shape.model_dump(),
            }
            if kind == 'asset':
                file = output / (aid + '.png')
                Image.fromarray(rgba[iy : iy + ih, ix : ix + iw]).save(file)
                rec['asset_reference'] = {
                    'path': file.name,
                    'sha256': digest(file),
                    'source_sha256': graph.source_sha256,
                    'bbox': bbox,
                }
                rec['payload'] = {'method': 'retained-card-foreground', 'background_parent_id': sid}
                if layered:
                    rec['container_layer'] = True
                    rec['retained_card_artwork'] = True
            elif len(peers) == 1 and h <= 2.3 * peers[0]['bbox'][3] and w >= 2.4 * h:
                # A single-label control can be moved as one native group.
                contained = [t.id for t in graph.text.texts if intersection(t.bbox, box)]
                if set(contained) == {peers[0]['id']}:
                    rec['move_members'] = [sid, peers[0]['id']] + ([aid] if assets else [])
            graph.selected[identifier] = rec
            graph.transactions.append(rec)
            staged.append(identifier)
        if layered:
            # Only replace material accounted for by the shape, retained artwork or proven children.
            replace = footprint | ink | owned
            plate[y:y+h,x:x+w,:3][replace] = parent_patch[replace]
        else:
            plate[y:y+h,x:x+w,:3] = parent_patch
    if not staged:
        return baseline / 'slideir.json', []
    Image.fromarray(plate).save(output / plate_path.name)
    graph.raster_backing = {**plate_ref, 'sha256': digest(output / plate_path.name)}
    record_claims(output, (graph.selected[k] for k in staged))
    write(output / 'scene.json', graph.model_dump())
    write(output / 'transactions.json', graph.transactions)
    write(output / 'slideir.json', adapt(graph, output))
    return output / 'slideir.json', staged
