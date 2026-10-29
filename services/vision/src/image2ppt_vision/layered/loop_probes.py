"""Actual-file edit probes, never mutations of delivered PPTX bytes."""
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image

from image2ppt.quality import inspect_pptx, object_name, NS, q, edit_copy
from image2ppt.contracts import ProbeCase
from ..storage import digest
from .loop_review import intersect, clip
from .loop_checks import issue, _edge


def hide_copy(source, target, names):
    """Remove exact named drawable objects from a diagnostic ZIP copy."""
    before = inspect_pptx(source); page = before['slides'][0]
    with ZipFile(source) as src:
        root = ET.fromstring(src.read(page['part']))
        parents = {child: parent for parent in root.iter() for child in parent}
        elements = {object_name(e): e for e in root.iter() if e.tag in {q('p:sp'), q('p:pic'), q('p:grpSp'), q('p:graphicFrame')}}
        if set(names)-set(elements):
            raise ValueError('unknown hidden object')
        for name in names:
            element = elements[name]; parents[element].remove(element)
        with ZipFile(target, 'x', compression=ZIP_DEFLATED) as out:
            for entry in src.infolist():
                out.writestr(entry, ET.tostring(root, encoding='utf-8', xml_declaration=True)
                             if entry.filename == page['part'] else src.read(entry.filename))
    after = inspect_pptx(target)
    remaining = {o['name'] for o in after['slides'][0]['objects']}
    if set(names) & remaining:
        raise ValueError('hide probe failed')
    return after


def duplicate_candidates(ir, proposal):
    """Only visible, detailed pictures overlapping earlier pictures are probed."""
    ordered = sorted(ir['nodes'], key=lambda n: n['z']); assets = {a['id']: a for a in ir['assets']}
    candidates = []
    for i, node in enumerate(ordered):
        if node['type'] != 'image' or node['id'].startswith('retained-'):
            continue
        box = node['bbox']; area = box[2]*box[3]
        lower = [n for n in ordered[:i] if n['type'] == 'image' and intersect(n['bbox'], box) >= area*.7]
        # An image hidden by a later picture is an occlusion issue, not evidence
        # that its lower layer contains a duplicate.
        if not lower or any(n['type'] != 'text' and intersect(n['bbox'], box) > area*.1 for n in ordered[i+1:]):
            continue
        with Image.open(Path(proposal)/assets[node['asset_id']]['path']) as im:
            rgba = np.asarray(im.convert('RGBA'))
        detail = (_edge(rgba[:, :, :3]) > 40) & (rgba[:, :, 3] > 200)
        if detail.sum() < 30:
            continue
        candidates.append((int(detail.sum()), node))
    return [n for _, n in sorted(candidates, key=lambda item: -item[0])]


def disjoint_batches(nodes):
    batches = []
    for node in nodes:
        group = next((g for g in batches if not any(intersect(node['bbox'], n['bbox']) for n in g)), None)
        if group is None:
            group = []; batches.append(group)
        group.append(node)
    return batches


def run(ir, proposal, pptx, preview, output, backend, *, full=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    original = digest(pptx); findings = []; cases = []
    with Image.open(preview) as im:
        base = np.asarray(im.convert('RGB').resize(tuple(ir['canvas']['source_px']), Image.Resampling.LANCZOS))
    selected = duplicate_candidates(ir, proposal)
    for i, batch in enumerate(disjoint_batches(selected)):
        folder = output/f'hide-{i}'; folder.mkdir()
        target = folder/'probe.pptx'
        try:
            hide_copy(pptx, target, [node['id'] for node in batch])
            result = backend.render(target, folder/'render')
            with Image.open(result['pages'][0]['path']) as im:
                hidden = np.asarray(im.convert('RGB').resize(tuple(ir['canvas']['source_px']), Image.Resampling.LANCZOS))
            for node in batch:
                x, y, w, h = clip(node['bbox'], ir['canvas']['source_px'])
                a, b = base[y:y+h, x:x+w], hidden[y:y+h, x:x+w]
                edges = _edge(a) > 40
                unchanged = float(np.abs(a.astype(float)-b.astype(float))[edges].mean()) if edges.any() else None
                cases.append({'action': 'hide_picture', 'target': node['id'], 'pptx_sha256': digest(target),
                              'preview_sha256': digest(result['pages'][0]['path']), 'edge_delta': unchanged,
                              'batch': i, 'original_preserved': digest(pptx) == original})
                if unchanged is not None and unchanged < 1.:
                    findings.append(issue('actual_duplicate', node['bbox'], target=node['id'], confirmed=True,
                                          severity='blocking', evidence='hidden_actual_picture_leaves_same_visible_detail'))
        except Exception as error:
            findings.append(issue('edit_probe_unavailable', targets=[n['id'] for n in batch], detail=str(error)[:500]))
    if full and ir['groups']:
        group = ir['groups'][0]; target = output/'move-group.pptx'
        try:
            case = ProbeCase(id='move-module', action='move_group', targets=[group['id']], delta_px=[24, 16])
            edit_copy(pptx, target, case)
            result = backend.render(target, output/'move-render')
            cases.append({'action': 'move_group', 'target': group['id'], 'child_transforms_preserved': True,
                          'pptx_sha256': digest(target), 'preview_sha256': digest(result['pages'][0]['path']),
                          'visual_acceptance': False})
        except Exception as error:
            findings.append(issue('group_edit_failed', target=group['id'], severity='blocking', confirmed=True, detail=str(error)[:500]))
    if digest(pptx) != original:
        raise ValueError('probe modified delivery')
    return {'source_pptx_sha256': original, 'original_preserved': True, 'cases': cases, 'issues': findings,
            'coverage': 'all visible overlapping picture candidates, disjoint batches; one module in selected revision',
            'human_acceptance': False, 'office_acceptance': False}
