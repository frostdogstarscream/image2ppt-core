"""Source-bound R12 background trials enter the existing private r11 stage."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageFont
from fontTools.ttLib import TTFont

from ..materialize import digest, font_gate, write
from ..parallel_candidate_contracts import ReconstructionCandidate, PreflightResult, transition
from ..parallel_candidate_engine import CHECKS, font_bindings
from ..parallel_candidate_stage import stage
from ..parallel_contracts import SceneGraph
from ..parallel_fusion import glyph_evidence
from ..parallel_rebuild import background_model
from ..parallel_regions import regions
from ..parallel_store import key
from ..parallel_text_candidates import peer_protection
from .diagnostics import assess_trial
from .gradient_backing import fit_gradient
from .text_mask_completion import TextMaskCompletion


def font_line_height(record, size):
    with TTFont(record['file'], fontNumber=record['index'], lazy=True) as face:
        hhea, os2 = face['hhea'], face['OS/2']
        units = max(hhea.ascent-hhea.descent+hhea.lineGap,
                    os2.usWinAscent+os2.usWinDescent,
                    os2.sTypoAscender-os2.sTypoDescender+os2.sTypoLineGap)
        return size*units/face['head'].unitsPerEm


def native_frame(text, styles, fonts, source_size, protected, scope, *, minimum_size=None):
    """Fit installed font bearings/leading within this local replacement scope."""
    used = {text.style, *[span.style for span in text.spans]}
    records = {(font['family'], font['bold']): font for font in fonts['fonts']}
    bearing = 0
    line_height = 0
    advance_width = text.bbox[2]
    for identifier in used:
        style = styles[identifier]
        family = style.ea if any(ord(c) > 255 for c in text.text) else style.latin
        record = records[family, style.bold]
        face = ImageFont.truetype(record['file'], max(1, round(style.size_px)), index=record['index'])
        bearing = max(bearing, max(0, face.getbbox(text.text.split('\n')[0])[1]))
        advance_width=max(advance_width,*[math.ceil(face.getlength(line))+1 for line in text.text.split('\n')])
        for name in (style.ea, style.latin):
            line_height = max(line_height, font_line_height(records[name, style.bold], style.size_px))
    payload = text.model_dump()
    spacing = text.line_spacing_px or styles[text.style].size_px*1.2
    metric_height = (len(text.text.split('\n'))-1)*spacing+math.ceil(line_height)+1
    payload['bbox'] = [*text.bbox[:2],advance_width,max(text.bbox[3]+bearing, metric_height)]
    if minimum_size is not None:
        payload['bbox'][2]=max(payload['bbox'][2],minimum_size[0])
        payload['bbox'][3]=max(payload['bbox'][3],minimum_size[1])
    from ..object_extract import rectangle
    grown = rectangle(payload['bbox'], source_size)
    # A transparent textbox's font-leading allowance is not an erase mask.
    # Keep its origin fixed and validate canvas/peer bounds independently.
    left,top,width,height=payload['bbox']
    if ((grown & protected).any() or min(left,top)<0 or left+width>source_size[0]
            or top+height>source_size[1]):
        raise ValueError('native font frame allowance conflicts with canvas or protected peers')
    return payload


def stage_background(trial_directory, graph, baseline_directory, fonts, output, *, typography=None, frame_minimum=None):
    """Recompute all local prechecks, then atomically expose one trial directory.

    Failed trials leave the baseline and already staged regions untouched.
    PRECHECK_PASS grants staging only. Office and final independent QA remain.
    """
    trial_directory, baseline_directory, output = map(Path, (trial_directory, baseline_directory, output))
    if output.exists():
        raise ValueError('stage output already exists')
    hashes = json.loads((trial_directory/'hashes.json').read_text('utf8'))
    required = {'source.png', 'mask.png', 'raw_inpaint.png', 'written_back.png',
                'difference_map.png', 'residual_report.json'}
    if not required <= hashes.keys():
        raise ValueError('incomplete trial evidence')
    for name, sha in hashes.items():
        if Path(name).name != name or digest(trial_directory/name) != sha:
            raise ValueError('trial evidence binding mismatch')
    saved = json.loads((trial_directory/'residual_report.json').read_text('utf8'))
    if (saved['status'] != 'PRECHECK_PASS' or saved.get('scene_graph_sha256') != key(graph.model_dump())
            or saved['source_sha256'] != graph.source_sha256):
        raise ValueError('trial is unapproved or stale')
    source_path = baseline_directory/'source.png'
    if digest(source_path) != graph.source_sha256:
        raise ValueError('baseline source binding mismatch')
    source = np.array(Image.open(source_path).convert('RGB'))
    if not np.array_equal(source, np.array(Image.open(trial_directory/'source.png').convert('RGB'))):
        raise ValueError('trial source pixels changed')
    text = next(t for t in graph.text.texts if t.id == saved['object_id'])
    if graph.ownership[text.id] != 'RESOLVED':
        raise ValueError('unresolved text content')
    region = regions(graph)[1][text.id]
    if region['member_ids'] != [text.id]:
        raise ValueError('indivisible region cannot be split')
    styles = {s.id: s for s in graph.text.styles}
    protected = peer_protection(text, graph, source, np.zeros(source.shape[:2], bool),baseline_directory=baseline_directory)
    x, y, w, h = map(round, text.bbox)
    box = [x-3, y-3, w+6, h+6]
    bx, by, bw, bh = box
    context_protected=peer_protection(text,graph,source,np.zeros(source.shape[:2],bool),purpose='context',baseline_directory=baseline_directory)
    backing, _ = background_model(source, box, context_protected)
    ink, _ = glyph_evidence(source, box, styles[text.style].color, backing)
    original = np.zeros(source.shape[:2], bool)
    original[by:by+bh, bx:bx+bw] = ink
    reference, fit_report = fit_gradient(source, box, context_protected)
    completion = TextMaskCompletion.analyze(source, original, protected, box, reference,
                                             glyph_height=styles[text.style].size_px)
    selected = np.array(Image.open(trial_directory/'mask.png').convert('L'))
    if saved['mask_variant']=='source_authorized_subset':
        if not np.isin(selected,[0,255]).all() or selected.shape!=source.shape[:2]:
            raise ValueError('invalid source-authorized subset mask')
        wanted=selected>0
        if not wanted.any() or (wanted & ~completion.masks['semantic_completion']).any():
            raise ValueError('trial mask contains pixels without source authorization')
    else:
        wanted = completion.masks[saved['mask_variant']]
    if not np.isin(selected, [0, 255]).all() or not np.array_equal(selected > 0, wanted):
        raise ValueError('trial mask is not source-authorized')
    raw = np.array(Image.open(trial_directory/'raw_inpaint.png').convert('RGB'))
    context_width=(saved.get('provenance') or {}).get('background_context_width',14)
    if type(context_width) is not int or not 8<=context_width<=128:
        raise ValueError('invalid source context width')
    if context_width!=14:
        reference,fit_report=fit_gradient(source,box,context_protected,context=context_width)
    if saved['method'] == 'gradient' and not np.array_equal(raw, reference):
        raise ValueError('gradient raw backing binding mismatch')
    report, written, _ = assess_trial(source, wanted, raw, protected, box, reference, fit_report, {})
    if report['status'] != 'PRECHECK_PASS':
        raise ValueError('background recheck failed')
    if not np.array_equal(written, np.array(Image.open(trial_directory/'written_back.png').convert('RGB'))):
        raise ValueError('trial writeback mismatch')
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = {'source': graph.source_sha256, 'scene': key(graph.model_dump()), 'text': text.id,
                'typography':typography,'frame_minimum':frame_minimum,
                'trial_hashes': hashes, 'fonts': key(fonts), 'font_files': font_bindings(graph, fonts),
                'implementation_sha256': digest(__file__)}
    cid = 'c-'+key(identity)
    candidate = ReconstructionCandidate(candidate_id=cid, source_sha256=graph.source_sha256,
        scene_sha256=key(graph.model_dump()), region_id=region['region_id'], object_ids=[text.id],
        representation='NATIVE_TEXT', strategy='R12-background-'+saved['method'],
        provenance={'version': 'r12-regional-background/1', 'input_binding': key(identity)})
    # The existing stage reads relative assets and writes private SceneGraph/IR.
    # It runs inside an owned temporary tree, so an exception cannot expose a
    # partial region or roll back an unrelated baseline transaction.
    with TemporaryDirectory(prefix='r12-stage-', dir=output.parent) as temporary:
        folder = Path(temporary)/'candidate'
        folder.mkdir()
        np.savez_compressed(folder/'pixels.npz', mask=wanted,
                            patch=written[by:by+bh, bx:bx+bw], bbox=box)
        proposed=text.model_copy(deep=True)
        definitions=[]
        if typography:
            if text.spans:raise ValueError('font proposal cannot overwrite emphasis spans')
            style=styles[text.style].model_copy(update=typography)
            style.id=text.style[:40]+'-r12-'+key(style.model_dump())[:12]
            definitions=[style.model_dump()]
            styles={**styles,style.id:style};proposed.style=style.id
        frame_options={'minimum_size':frame_minimum['size']} if frame_minimum else {}
        text_payload = native_frame(proposed, styles, fonts, source.shape[1::-1], protected, box,**frame_options)
        fitted=proposed.model_copy(update={'bbox':text_payload['bbox']})
        fitted_styles={fitted.style,*[span.style for span in fitted.spans]}
        font_gate(SimpleNamespace(styles=[styles[i] for i in fitted_styles],texts=[fitted]),fonts)
        payload = {'id': text.id, 'kind': 'text', 'bbox': box, 'payload': text_payload,
                   'source_text_definition':text.model_dump(),'style_definitions':definitions,
                   'office_frame_feedback':frame_minimum,
                   'native_layout': {'original_bbox': text.bbox, 'proposed_bbox': text_payload['bbox'],
                                     'reason': 'installed EA/Latin font bearings and leading; source origin and line pitch retained'},
                   'method': 'r12-'+saved['method'], 'owner_method': 'source-connected-strokes+lateral-context/r12-1',
                   'recovery_evidence': {'trial_path': str(trial_directory.resolve()), 'hashes': hashes}}
        write(folder/'payload.json', payload)
        refs = {name: digest(folder/name) for name in ('pixels.npz', 'payload.json')}
        candidate = transition(candidate.model_copy(update={'references': refs, 'hard_gate': 'PASS'}), 'PRECHECK_PASS')
        preflight = PreflightResult(candidate_id=cid, status='PASS', checks={name: 'PASS' for name in CHECKS},
                                    unknown_pixels=0, evidence_hashes=list(hashes.values()))
        write(folder/'candidate.json', candidate.model_dump(by_alias=True))
        write(folder/'preflight.json', preflight.model_dump(by_alias=True))
        staged = Path(temporary)/'region'
        stage(folder, graph, baseline_directory, staged)
        write(staged/'r12-background-preflight.json', {'candidate': candidate.model_dump(by_alias=True),
              'preflight': preflight.model_dump(by_alias=True), 'residual_recheck': report,
              'completion': completion.report, 'trial_hashes': hashes, 'published': False})
        staged.rename(output)
    return output/'slideir.json'
