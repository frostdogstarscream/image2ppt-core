"""Stage fully ruled tables with source-grid and OCR cell ownership evidence."""

import json
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image

from ..contracts import intersection
from ..materialize import digest, write
from ..object_contracts import ClosedObject
from ..object_unmix import backing_from_evidence
from ..parallel_contracts import SceneGraph
from ..parallel_fusion import fuse_incremental_scene
from ..parallel_rebuild import adapt, array_sha
from ..parallel_store import key
from .container_stage import _hex
from .marked_ownership import record_claims


def _lines(tile, axis):
    diff = np.abs(np.diff(tile.astype(float), axis=axis)).max(2)
    hits = np.flatnonzero((diff > 15).mean(axis=1 - axis) > 0.54)
    groups = np.split(hits, np.flatnonzero(np.diff(hits) > 4) + 1)
    return [int(round(float(np.median(g)))) + 1 for g in groups if len(g)]


def table_from_mark(image, mark, lines, fonts):
    from image2ppt_vision.table_reconstruction import verify_table
    from image2ppt_vision.plans import PlanValidationError
    from image2ppt_vision.layers import contains

    x, y, w, h = map(round, mark['bbox'])
    l, t = max(0, x - 9), max(0, y - 9)
    tile = image[t : y + h + 10, l : x + w + 10]
    ys = [t + i for i in _lines(tile, 0)]
    if len(ys) < 3 or len(ys) > 100:
        return None
    xs = [l + i for i in _lines(tile[ys[1] - t + 2 : ys[-1] - t - 2], 1)]
    if len(xs) < 3 or len(xs) > 32 or min(np.diff(ys)) < 14 or min(np.diff(xs)) < 22:
        return None
    box = [xs[0] - 2, ys[0] - 2, xs[-1] - xs[0] + 5, ys[-1] - ys[0] + 5]
    if min(box[:2]) < 0 or box[0] + box[2] > image.shape[1] or box[1] + box[3] > image.shape[0]:
        return None
    from ..runtime.marked_routes import iou

    if iou(box, mark['bbox']) < 0.85:
        return None
    xs[0], xs[-1] = box[0], box[0] + box[2]
    ys[0], ys[-1] = box[1], box[1] + box[3]
    owned = [v for v in lines if intersection(v['bbox'], box)]
    if any(not contains(box, v['bbox']) or v.get('score', 0) < 0.85 for v in owned):
        return None
    rows, styles = [], []
    for r, (top, bottom) in enumerate(zip(ys, ys[1:])):
        row = []
        for c, (left, right) in enumerate(zip(xs, xs[1:])):
            cell = [left, top, right - left, bottom - top]
            words = sorted(
                (v for v in owned if contains(cell, v['bbox'])), key=lambda v: (v['bbox'][1], v['bbox'][0])
            )
            if not words:
                return None
            patch = image[top:bottom, left:right].astype(float)
            support = np.ones(patch.shape[:2], bool)
            support[:6] = False
            support[-6:] = False
            support[:, :6] = False
            support[:, -6:] = False
            glyphs = []
            for v in words:
                a, b, w, h = map(round, v['bbox'])
                a -= left
                b -= top
                support[max(0, b - 2) : b + h + 2, max(0, a - 2) : a + w + 2] = False
                glyphs.append(patch[b : b + h, a : a + w].reshape(-1, 3))
            if support.sum() < 15:
                return None
            fill = np.median(patch[support], axis=0)
            unknown = support & (abs(patch - fill).max(2) > 24)
            _, _, stats, _ = cv2.connectedComponentsWithStats(unknown.astype('uint8'), 8)
            if any(area > 12 for area in stats[1:, -1]):
                return None
            pixels = np.concatenate(glyphs)
            contrast = abs(pixels - fill).max(1)
            fg = np.median(pixels[contrast >= np.percentile(contrast, 94)], axis=0)
            row.append(''.join(v['text'] for v in words))
            styles.append({'row': r, 'column': c, 'fill': _hex(fill), 'color': _hex(fg), 'bold': r == 0})
        rows.append(row)
    widths, heights = np.diff(xs).tolist(), np.diff(ys).tolist()
    plan = {
        'bbox': box,
        'rows': rows,
        'merges': [],
        'row_heights': heights,
        'column_widths': widths,
        'ocr_ids': [v['id'] for v in owned],
    }
    try:
        verify_table(plan, lines, {})
    except PlanValidationError:
        return None
    size = float(np.median([v['bbox'][3] for v in owned])) * 0.72 * 0.75
    return (
        box,
        owned,
        {
            'rows': rows,
            'column_tracks': [{'mode': 'fr', 'value': w} for w in widths],
            'row_heights': heights,
            'merges': [],
            'cell_styles': styles,
            'header_row': False,
            'banded_rows': False,
            'font': {'ea': fonts.ea, 'latin': fonts.latin, 'size_pt': size},
        },
    )


def stage_tables(baseline, output, fields, skip):
    from ..pipeline import compiler_runtime, ROOT

    _, node = compiler_runtime()
    check = subprocess.run(
        [
            str(node),
            '--input-type=module',
            '-e',
            "import {COMPILER} from './scripts/capabilities.mjs';console.log(COMPILER.native_tables)",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    if check.stdout.strip() != 'true':
        return baseline, [], set()
    graph = SceneGraph.model_validate_json((baseline / 'scene.json').read_text(encoding='utf8'))
    inv = json.loads((baseline / 'runtime-ocr-protection.json').read_text(encoding='utf8'))
    rb = graph.raster_backing
    source = baseline / 'source.png'
    backing = baseline / rb['path']
    if (
        digest(source) != graph.source_sha256
        or inv['source_sha256'] != graph.source_sha256
        or digest(backing) != rb['sha256']
    ):
        raise ValueError('stale table evidence')
    image = np.array(Image.open(source).convert('RGB'))
    plate = np.array(Image.open(backing).convert('RGBA'))
    staged = []
    members = set()
    for field in fields:
        tid = 'tbl-' + key([graph.source_sha256, field['id']])[:16]
        if tid in skip or field['id'] in skip or tid in graph.selected or not graph.text.styles:
            continue
        data = table_from_mark(image, field, inv['lines'], graph.text.styles[0])
        if not data:
            continue
        box, owned, payload = data
        if any(intersection(box, r['bbox']) for r in graph.selected.values()):
            continue
        x, y, w, h = box
        mask = np.zeros(image.shape[:2], bool)
        mask[y : y + h, x : x + w] = True
        bg, err, _ = backing_from_evidence(plate[..., :3], box, np.zeros_like(mask), graph.surface)
        if not np.isfinite(err) or err > 4:
            continue
        obj = ClosedObject(
            id=tid,
            kind='illustration',
            bbox=box,
            layers=['source-plate.png'],
            components=['table'],
            parts=[{'name': 'table', 'bbox': box, 'layers': ['source-plate.png']}],
            text_policy='exclude_independent_text',
            description='Source-verified table grid',
            prompt_en='independent native table',
        )
        fused = fuse_incremental_scene(baseline, base_graph=graph, added_assets=[obj])
        if fused.ownership[tid] != 'RESOLVED':
            continue
        if not staged:
            shutil.copytree(baseline, output)
        graph = fused
        proof = output / (tid + '-owner.npz')
        bg = np.rint(bg).clip(0, 255).astype(np.uint8)
        np.savez_compressed(proof, mask=mask, restore_bbox=box, backing_patch=bg)
        lids = {v['id'] for v in owned}
        covered = [k for k, v in graph.text.members.items() if v and set(v) <= lids]
        rec = {
            'id': tid,
            'kind': 'table',
            'bbox': box,
            'restore_bbox': box,
            'payload': payload,
            'mark_id': field['id'],
            'covered_text_ids': covered,
            'cell_evidence': owned,
            'source_sha256': graph.source_sha256,
            'method': 'ocr-verified-grid/1',
            'state': 'STAGED',
            'quality': 'REVIEW_REQUIRED',
            'human_acceptance': False,
            'complete_owner_scope': True,
            'original_owner': 'source-plate',
            'mask_sha256': array_sha(mask.astype('uint8')),
            'owner_evidence': {
                'file': proof.name,
                'sha256': digest(proof),
                'method': 'source-grid-and-lateral-backing',
            },
        }
        graph.selected[tid] = rec
        graph.transactions.append(rec)
        staged.append(tid)
        members.update(covered)
        plate[y : y + h, x : x + w, :3] = bg
    if staged:
        Image.fromarray(plate).save(output / backing.name)
        graph.raster_backing = {**rb, 'sha256': digest(output / backing.name)}
        record_claims(output, (graph.selected[k] for k in staged))
        write(output / 'scene.json', graph.model_dump())
        write(output / 'transactions.json', graph.transactions)
        write(output / 'slideir.json', adapt(graph, output))
    return output if staged else baseline, staged, members
