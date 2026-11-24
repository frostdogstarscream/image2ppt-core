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
from .background_adapter import RecoveryBlocked
from .gradient_backing import fit_gradient
from .text_mask_completion import TextMaskCompletion


class NativeFrameConflict(ValueError):
    """The measured frame has no room between the canvas and neighboring objects."""


def font_line_height(record, size):
    with TTFont(record['file'], fontNumber=record['index'], lazy=True) as face:
        hhea, os2 = face['hhea'], face['OS/2']
        units = max(hhea.ascent-hhea.descent+hhea.lineGap,
                    os2.usWinAscent+os2.usWinDescent,
                    os2.sTypoAscender-os2.sTypoDescender+os2.sTypoLineGap)
        return size*units/face['head'].unitsPerEm


def _find_non_colliding_frame(text, payload_bbox, spacing, max_font_px, max_glyph_bottom, max_line_len, source_size, protected, minimum_size):
    left, top, width, height = payload_bbox
    lines_n = len(text.text.split('\n'))
    if min(left, top) < 0 or left + text.bbox[2] > source_size[0] or top + text.bbox[3] > source_size[1]:
        return None
    min_gate_w = max(int(math.ceil(text.bbox[2])), max_line_len - 2)
    min_gate_h = max(
        int(math.ceil(text.bbox[3])),
        int(math.ceil((lines_n - 1) * spacing + max_font_px - 3)),
        (max_glyph_bottom + 1 if lines_n == 1 else int(math.ceil(text.bbox[3]))),
    )
    if minimum_size is not None:
        min_gate_w = max(min_gate_w, int(math.ceil(minimum_size[0])))
        min_gate_h = max(min_gate_h, int(math.ceil(minimum_size[1])))
    max_w = min(int(math.ceil(width)), int(math.floor(source_size[0] - left)))
    max_h = min(int(math.ceil(height)), int(math.floor(source_size[1] - top)))
    ix0 = max(0, int(round(left)))
    iy0 = max(0, int(round(top)))
    max_w = min(max_w, source_size[0] - ix0)
    max_h = min(max_h, source_size[1] - iy0)
    if max_w < min_gate_w or max_h < min_gate_h:
        return None
    from ..object_extract import rectangle
    sub = protected[iy0:iy0 + max_h, ix0:ix0 + max_w].astype(np.int32)
    csum = np.pad(sub, ((1, 0), (1, 0)), mode='constant').cumsum(axis=0).cumsum(axis=1)
    best = None
    best_cost = None
    for cand_w in range(max_w, min_gate_w - 1, -1):
        for cand_h in range(max_h, min_gate_h - 1, -1):
            if csum[cand_h, cand_w] == 0:
                cost = (max_w - cand_w) * 2 + (max_h - cand_h)
                if best_cost is None or cost < best_cost:
                    cand_box = [left, top, float(cand_w), float(cand_h)]
                    if (cand_box[0] + cand_box[2] <= source_size[0]
                            and cand_box[1] + cand_box[3] <= source_size[1]
                            and not (rectangle(cand_box, source_size) & protected).any()):
                        best = cand_box
                        best_cost = cost
                break
    return best


def native_frame(text, styles, fonts, source_size, protected, scope, *, minimum_size=None, exploratory=False, fit_origin=False):
    """Fit installed font bearings/leading within this local replacement scope."""
    used = {text.style, *[span.style for span in text.spans]}
    records = {(font['family'], font['bold']): font for font in fonts['fonts']}
    bearing = 0
    line_height = 0
    max_glyph_bottom = 0
    max_line_len = 0
    max_font_px = styles[text.style].size_px
    advance_width = text.bbox[2]
    lines = text.text.split('\n')
    for identifier in used:
        style = styles[identifier]
        max_font_px = max(max_font_px, style.size_px)
        family = style.ea if any(ord(c) > 255 for c in text.text) else style.latin
        record = records[family, style.bold]
        face = ImageFont.truetype(record['file'], max(1, round(style.size_px)), index=record['index'])
        bearing = max(bearing, max(0, face.getbbox(lines[0])[1]))
        max_glyph_bottom = max(max_glyph_bottom, *[face.getbbox(line)[3] for line in lines])
        max_line_len = max(max_line_len, *[math.ceil(face.getlength(line)) for line in lines])
        advance_width = max(advance_width, *[math.ceil(face.getlength(line)) + 1 for line in lines])
        for name in (style.ea, style.latin):
            line_height = max(line_height, font_line_height(records[name, style.bold], style.size_px))
    payload = text.model_dump()
    spacing = text.line_spacing_px or max_font_px*1.2
    metric_height = (len(lines)-1)*spacing+math.ceil(line_height)+1
    payload['bbox'] = [*text.bbox[:2],advance_width,max(text.bbox[3]+bearing, metric_height)]
    if minimum_size is not None:
        payload['bbox'][2]=max(payload['bbox'][2],minimum_size[0])
        payload['bbox'][3]=max(payload['bbox'][3],minimum_size[1])
    from ..object_extract import rectangle
    grown = rectangle(payload['bbox'], source_size)
    # A transparent textbox's font-leading allowance is not an erase mask.
    # Keep its origin fixed and validate canvas/peer bounds independently.
    left,top,width,height=payload['bbox']
    if not exploratory and ((grown & protected).any() or min(left,top)<0 or left+width>source_size[0]
            or top+height>source_size[1]):
        clamped = _find_non_colliding_frame(
            text, payload['bbox'], spacing, max_font_px, max_glyph_bottom, max_line_len,
            source_size, protected, minimum_size)
        # An OCR box begins at visible ink. The native font also needs top bearing.
        # Move only that transparent frame, leaving the source mask and peer masks fixed.
        if clamped is None and fit_origin and len(lines) == 1:
            for dy in range(1, min(bearing, math.ceil(max_font_px * .22)) + 1):
                proposed = [left, top-dy, width, height+dy]
                clamped = _find_non_colliding_frame(
                    text, proposed, spacing, max_font_px, max_glyph_bottom, max_line_len,
                    source_size, protected, minimum_size)
                if clamped is not None:
                    break
        if clamped is None:
            raise NativeFrameConflict('native font frame allowance conflicts with canvas or protected peers')
        payload['bbox'] = clamped
    return payload


def _fit_measured_line(text, styles, fonts, source_size, protected, mask, scope, frame_options, *, office_style=None):
    """Fit the transparent Office frame around owned ink without altering the erase mask."""
    if text.spans or '\n' in text.text or text.align != 'left' or not mask.any() or (mask & protected).any():
        raise NativeFrameConflict('native font frame allowance conflicts with canvas or protected peers')
    style = styles[text.style]
    family = style.ea if any(ord(c) > 255 for c in text.text) else style.latin
    record = next(f for f in fonts['fonts'] if (f['family'],f['bold']) == (family,style.bold))
    yy, xx = np.where(mask)
    ink_left, ink_top, ink_right, ink_bottom = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
    original_size = style.size_px
    face = ImageFont.truetype(record['file'], max(1,round(original_size)), index=record['index'])
    l, t, r, b = face.getbbox(text.text)
    # Only try sizes corroborated by the observed strokes. Never shrink to an
    # arbitrary size just to squeeze a long label past a neighbour.
    scale = min((ink_right-ink_left)/max(1,r-l), (ink_bottom-ink_top)/max(1,b-t))
    measured_size = max(4, round(original_size * min(1.0,scale)))
    sizes = [original_size]
    if .72*original_size <= measured_size < original_size:
        sizes.append(measured_size)
    minimum = frame_options.get('minimum_size')
    reference_size = 0
    if minimum and office_style and all(office_style.get(k) == getattr(style,k) for k in ('ea','latin','bold')):
        reference_size = office_style.get('size_px', 0)
    safe_scale = 0

    def fit_at_size(size, options, *, measure_room=False):
        nonlocal safe_scale
        fitted = text.model_copy(deep=True)
        definition = None
        fitted_styles = styles
        if size != original_size:
            definition = style.model_copy(update={'size_px':size})
            definition.id = style.id[:40]+'-ink-'+key(definition.model_dump())[:12]
            fitted_styles = {**styles, definition.id:definition}
            fitted.style = definition.id
        face = ImageFont.truetype(record['file'], max(1,round(size)), index=record['index'])
        l, t, r, b = face.getbbox(text.text)
        for left in dict.fromkeys([round(text.bbox[0]), ink_left-l, ink_right-r]):
            if abs(left+l-ink_left) > .22*size:
                continue
            for top in dict.fromkeys([round(text.bbox[1]), ink_top-t]):
                fitted.bbox = [left, top, max(1,ink_right-left), max(1,ink_bottom-top)]
                try:
                    payload = native_frame(fitted, fitted_styles, fonts, source_size, protected, scope, **options)
                    return fitted, payload, definition
                except NativeFrameConflict:
                    if not measure_room:
                        continue
                try:
                    room = native_frame(fitted, fitted_styles, fonts, source_size, protected, scope,
                                        **{**options, 'minimum_size':None})['bbox'][2:]
                except NativeFrameConflict:
                    continue
                ratio = min((a-1)/max(1,b-1) for a,b in zip(room,minimum))
                safe_scale = max(safe_scale, ratio)

    for size in sizes:
        result = fit_at_size(size, frame_options, measure_room=reference_size == size)
        if result:
            return result
    # Office measured the previous typeface on the replacement string. A
    # smaller font can use a proportionally smaller transparent frame, but
    # it must still fit the source strokes and pass the next real edit probe.
    adjusted = math.floor(reference_size * min(1, safe_scale))
    lower_limit = max(.72*reference_size, .8*min(original_size, measured_size))
    if reference_size and lower_limit <= adjusted < reference_size:
        scaled = [math.ceil((v-1)*adjusted/reference_size)+1 for v in minimum]
        result = fit_at_size(adjusted, {**frame_options, 'minimum_size':scaled})
        if result:
            return result
    raise NativeFrameConflict('native font frame allowance conflicts with canvas or protected peers')


def stage_background(trial_directory, graph, baseline_directory, fonts, output, *, typography=None, frame_minimum=None, fit_origin=False):
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
    from .region_candidates import _safe_background_model
    backing, _ = _safe_background_model(source, box, context_protected)
    ink, _ = glyph_evidence(source, box, styles[text.style].color, backing)
    original = np.zeros(source.shape[:2], bool)
    original[by:by+bh, bx:bx+bw] = ink
    reference_ctx = (saved.get('provenance') or {}).get('background_context_width', 14)
    if type(reference_ctx) is not int or not 8 <= reference_ctx <= 128:
        raise ValueError('invalid source context width')
    try:
        reference, fit_report = fit_gradient(source, box, context_protected, context=reference_ctx)
    except RecoveryBlocked:
        reference, fit_report = fit_gradient(source, box, context_protected, context=reference_ctx, allow_vertical=True)
    if ((saved.get('provenance') or {}).get('fit') or {}).get('method') == 'scanline-lateral':
        from .marked_fields import striped_backing
        stripe = striped_backing(source,box,context_protected)
        if stripe is None:
            raise ValueError('scanline backing no longer corroborated')
        reference, fit_report = stripe
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
    if saved['method'] == 'gradient' and not np.array_equal(raw, reference):
        raise ValueError('gradient raw backing binding mismatch')
    report, written, _ = assess_trial(source, wanted, raw, protected, box, reference, fit_report,
                                      {'method': saved['method'], 'provenance': saved.get('provenance')})
    if report['status'] != 'PRECHECK_PASS':
        raise ValueError('background recheck failed')
    if not np.array_equal(written, np.array(Image.open(trial_directory/'written_back.png').convert('RGB'))):
        raise ValueError('trial writeback mismatch')
    return stage_text_pixels(graph, baseline_directory, fonts, output, text=text, region=region,
                             source=source, box=box, wanted=wanted, written=written, protected=protected,
                             method=saved['method'], report=report, completion_report=completion.report,
                             trial_directory=trial_directory, hashes=hashes, typography=typography,
                             frame_minimum=frame_minimum, fit_origin=fit_origin)


def stage_text_pixels(graph, baseline_directory, fonts, output, *, text, region, source, box, wanted,
                      written, protected, method, report, completion_report, trial_directory, hashes,
                      typography=None, frame_minimum=None, fit_origin=False, fit_ink_frame=False,
                      source_artwork=None, text_cutouts=None):
    """Stage a checked source-owner mask and corresponding editable field as one transaction."""
    styles = {s.id: s for s in graph.text.styles}
    bx, by, bw, bh = box
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = {'source': graph.source_sha256, 'scene': key(graph.model_dump()), 'text': text.id,
                'typography':typography,'frame_minimum':frame_minimum,'fit_origin':fit_origin,'fit_ink_frame':fit_ink_frame,
                'trial_hashes': hashes, 'fonts': key(fonts), 'font_files': font_bindings(graph, fonts),
                'implementation_sha256': digest(__file__)}
    cid = 'c-'+key(identity)
    candidate = ReconstructionCandidate(candidate_id=cid, source_sha256=graph.source_sha256,
        scene_sha256=key(graph.model_dump()), region_id=region['region_id'], object_ids=[text.id],
        representation='NATIVE_TEXT', strategy='R12-background-'+method,
        provenance={'version': 'r12-regional-background/1', 'input_binding': key(identity)})
    # The existing stage reads relative assets and writes private SceneGraph/IR.
    # It runs inside an owned temporary tree, so an exception cannot expose a
    # partial region or roll back an unrelated baseline transaction.
    with TemporaryDirectory(prefix='r12-stage-', dir=output.parent) as temporary:
        folder = Path(temporary)/'candidate'
        folder.mkdir()
        # The OCR frame positions text. A verified stroke may extend beyond it.
        ys, xs = np.nonzero(wanted)
        if not len(xs):
            raise ValueError('empty text owner mask')
        left, top = min(bx, int(xs.min())), min(by, int(ys.min()))
        right, bottom = max(bx+bw, int(xs.max())+1), max(by+bh, int(ys.max())+1)
        np.savez_compressed(folder/'pixels.npz', mask=wanted, patch=written[top:bottom, left:right],
                            bbox=[left, top, right-left, bottom-top])
        proposed=text.model_copy(deep=True)
        definitions=[]
        if typography:
            if text.spans:raise ValueError('font proposal cannot overwrite emphasis spans')
            style=styles[text.style].model_copy(update=typography)
            style.id=text.style[:40]+'-r12-'+key(style.model_dump())[:12]
            definitions=[style.model_dump()]
            styles={**styles,style.id:style};proposed.style=style.id
        frame_options={'minimum_size':frame_minimum['size']} if frame_minimum else {}
        if fit_origin:
            frame_options['fit_origin'] = True
        try:
            text_payload = native_frame(proposed, styles, fonts, source.shape[1::-1], protected, box, **frame_options)
        except NativeFrameConflict:
            if not fit_ink_frame:
                raise
            proposed, text_payload, measured_style = _fit_measured_line(
                proposed, styles, fonts, source.shape[1::-1], protected, wanted, box, frame_options,
                office_style=(frame_minimum or {}).get('measured_style'))
            if measured_style is not None:
                styles[measured_style.id] = measured_style
                definitions.append(measured_style.model_dump())
        if fit_origin and proposed.align == 'left' and any(k!=text.id and ids==graph.text.members[text.id]
                                                         for k,ids in graph.text.members.items()):
            # Reserve transparent editing room only along a measured blank run.
            # No source pixel or peer geometry changes when the frame grows.
            fx,fy,fw,fh = map(round,text_payload['bbox'])
            x0 = fx+fw
            room = source[fy:fy+fh,x0:x0+round(styles[text.style].size_px*2)].astype(float)
            blocked = protected[fy:fy+fh,x0:x0+room.shape[1]]
            if room.shape[1]>=4:
                variation = abs(room-np.median(room[:,:4],axis=1,keepdims=True)).max((0,2))
                # An isolated low-contrast antialias pixel may lie just beyond
                # the OCR extent. A repeated edge or distinct ink stops growth.
                gradient = np.percentile(abs(np.diff(room,axis=1,prepend=room[:,:1])).max(2),90,axis=0)
                clean = (variation<=12) & (gradient<=8) & ~blocked.any(0)
                text_payload['bbox'][2] += int(np.cumprod(clean).sum())
        fitted=proposed.model_copy(update={'bbox':text_payload['bbox']})
        fitted_styles={fitted.style,*[span.style for span in fitted.spans]}
        font_gate(SimpleNamespace(styles=[styles[i] for i in fitted_styles],texts=[fitted]),fonts)
        payload = {'id': text.id, 'kind': 'text', 'bbox': box, 'payload': text_payload,
                   'source_text_definition':text.model_dump(),'style_definitions':definitions,
                   'office_frame_feedback':frame_minimum,
                   'native_layout': {'original_bbox': text.bbox, 'proposed_bbox': text_payload['bbox'],
                                     'reason': 'installed EA/Latin font bearings and leading; source origin and line pitch retained'},
                   'method': 'r12-'+method, 'owner_method': method if method.startswith('region-') else 'source-connected-strokes+lateral-context/r12-1',
                   'recovery_evidence': {'trial_path': str(trial_directory.resolve()), 'hashes': hashes}}
        if source_artwork:
            payload['source_artwork'] = source_artwork
        if text_cutouts:
            payload['text_cutouts'] = text_cutouts
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
              'completion': completion_report, 'trial_hashes': hashes, 'published': False})
        staged.rename(output)
    return output/'slideir.json'
