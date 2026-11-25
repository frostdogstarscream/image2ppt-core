"""Independent stateless delivery audit & form-aware scorecard gate (W6 + W7).

Evaluates the compiled PPTX and actual PowerPoint render independently of the
executing OpenCode Agent session:
  1) Runs `PageScorecard/1` directly on OOXML + Office render + OCR inventory;
  2) Computes form-aware proper-representation fraction (treating framed photos
     and brand logos delivered as movable image modules as proper PPT form, distinct
     from unrecovered cards/texts) and `PageStructureInventory/1` recall;
  3) Runs a stateless VLM visual audit (isolated session ID, zero confirmation bias)
     capable of catching missing text/objects, ghosting, and uneditable titles.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import intersection
from ..materialize import digest, write
from ..parallel_store import Store, key
from .. import scorecard
from ..tolerance_config import load_tolerance_config


class AuditDefect(BaseModel):
    model_config = ConfigDict(extra='ignore')
    severity: Literal['critical', 'minor']
    category: Literal[
        'missing_content', 'duplicate_ghosting', 'text_corruption',
        'uneditable_title_or_card', 'font_or_shade_deviation', 'missing_text', 'incorrect_group', 'other',
    ]
    description: str = Field(min_length=1, max_length=1000)
    bbox: list[float] | None = None
    bbox_space: Literal['source_px_xywh','normalized_xywh'] | None = None
    mark_numbers: list[int] = Field(default_factory=list, max_length=30)
    edit_action: Literal['replace_text','move_group','recolor','delete','edit_cell','edit_chart','inspect'] = 'inspect'
    text_impact: Literal['appearance_only','content_obscured','missing_strokes','wrong_or_missing_content','unknown'] = 'unknown'


class VLMIndependentAuditReply(BaseModel):
    model_config = ConfigDict(extra='ignore')
    passed: bool
    defects: list[AuditDefect] = Field(default_factory=list, max_length=50)
    summary: str = Field(default='', max_length=2000)


def classify_audit(reply):
    critical, minor = [], []
    for defect in reply.defects:
        row = defect.model_dump()
        if defect.text_impact in ('content_obscured', 'missing_strokes', 'wrong_or_missing_content'):
            row['severity'] = 'critical'
        elif defect.category == 'font_or_shade_deviation' and defect.text_impact == 'appearance_only':
            row['severity'] = 'minor'
        if row['severity'] == 'critical':
            critical.append(row)
        else:
            minor.append(row)
    only_cosmetic = bool(minor) and all(
        r['category'] == 'font_or_shade_deviation' and r['text_impact'] == 'appearance_only' for r in minor
    )
    return critical, minor, not critical and (reply.passed or only_cosmetic)


def compute_inventory_and_form_metrics(
    sc: dict,
    inventory: dict | None,
) -> dict:
    """Measure production inventory coverage separately from frozen reference recall."""
    elements = [e for e in (inventory or {}).get('elements', []) if e.get('category') != 'background_scenery']
    objects = list(sc.get('objects', []))
    objects.extend({**cell, 'class': 'text'} for obj in sc.get('objects', []) for cell in obj.get('cells', []))
    kinds = {
        'title': ('text',), 'section_header': ('text',), 'body_text': ('text',),
        'card_shape': ('shape',), 'ribbon_shape': ('shape',), 'badge_shape': ('shape',),
        'divider': ('shape',), 'chart': ('chart',), 'table': ('table',),
        'photo': ('asset', 'opaque_picture'), 'logo': ('asset', 'opaque_picture'),
        'icon': ('asset', 'shape'),
    }
    pairs = []
    detected = set()
    for i, el in enumerate(elements):
        for j, obj in enumerate(objects):
            if obj.get('class') in ('plate', 'background', 'duplicate_picture'):
                continue
            eb, ob = el['bbox'], obj['bbox']
            overlap = intersection(eb, ob)
            union = eb[2]*eb[3] + ob[2]*ob[3] - overlap
            if union <= 0 or overlap / union < 0.5:
                continue
            detected.add(i)
            if obj.get('class') not in kinds.get(el.get('category'), ()):
                continue
            if obj['class'] == 'text':
                expected = ''.join((el.get('text') or el.get('label') or '').split())
                if not expected or expected != ''.join(obj.get('text', '').split()):
                    continue
            pairs.append((overlap / union, i, j))
    assigned, used = {}, set()
    for _, i, j in sorted(pairs, reverse=True):
        if i not in assigned and j not in used:
            assigned[i] = objects[j]['name']
            used.add(j)
    rows = [{**el, 'detected': i in detected, 'correct_form': i in assigned,
             'office_object': assigned.get(i), 'specified_edit': 'UNKNOWN'} for i, el in enumerate(elements)]
    titles = [r for r in rows if r['category'] == 'title']
    fraction = len(assigned)/len(rows) if rows else None
    return {'inventory_available': bool(rows), 'total_items': len(rows) if rows else None,
            'matched_items': len(assigned) if rows else None, 'inventory_coverage': fraction,
            'recall': None, 'proper_form_object_fraction': fraction,
            'main_title_editable': all(r['correct_form'] for r in titles) if titles else None,
            'objects': rows, 'missing_items': [r for r in rows if not r['correct_form']][:25]}


def _stateless_vlm_audit(
    state,
    source_path: Path,
    render_path: Path,
    owner_preview_path: Path,
    sc_summary: dict,
    *, deck_path: Path, edit_panels: list[Path], numbered_image: Path | None = None,
) -> dict:
    """Call VLM in a fresh stateless session (separate from Agent conversation) to audit visual fidelity."""
    from ..recovery.adaptive_policy import AdaptiveBudget
    from .provider import agent_identity, model_identity, planning_actor
    import os

    request = json.loads((state.job / 'request.json').read_text('utf8'))
    budget = AdaptiveBudget(Store(state.job / 'state'), request.get('adaptive_budget'))
    from .provider import GO_MODEL_ALIASES
    builder_provider, builder_model = agent_identity(request)
    provider = os.environ.get('IMAGE2PPT_REVIEW_PROVIDER') or ('moma' if builder_provider=='deepseek' else 'opencode_go')
    requested = os.environ.get('IMAGE2PPT_REVIEW_MODEL') or ('qwen/qwen3.8-max' if provider=='moma' else 'hy4')
    review_model = GO_MODEL_ALIASES.get(requested, requested) if provider == 'opencode_go' else requested
    session_file = state.job/'runtime/session.json'
    builder_session = json.loads(session_file.read_text('utf8'))['id'] if session_file.is_file() else None
    from . import provider as model_provider, review_control, review_targets
    audit_key = key({
        'schema': 'StatelessVisualAudit/5', 'provider':provider, 'model': review_model,
        'builder':[builder_provider,builder_model,builder_session],
        'policy': key(load_tolerance_config()),
        'review_code':digest(__file__), 'model_policy':digest(model_provider.__file__),
        'controls':digest(review_control.__file__), 'target_policy':digest(review_targets.__file__),
        'scorecard':key(sc_summary), 'deck':digest(deck_path),
        'edit_panels': [digest(p) for p in edit_panels],
        'source_sha256': digest(source_path), 'numbered_image':digest(numbered_image) if numbered_image else None,
        'render_sha256': digest(render_path),
        'owner_sha256': digest(owner_preview_path),
    })
    cache_dir = state.job / 'cache/independent-audit'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'{audit_key}.json'
    if cache_file.is_file():
        return {**json.loads(cache_file.read_text('utf8')), 'status': 'CACHED'}

    try:
        if model_identity(review_model) == model_identity(builder_model):
            raise ValueError('independent model required')
        if provider == 'moma' and builder_provider in ('deepseek', 'moma', 'opencode_go'):
            from .moma_review import MomaReviewer
            reviewer = MomaReviewer(review_model)
        elif provider == builder_provider == 'opencode_go':
            reviewer = planning_actor(state.job)
            reviewer.session = f'independent-audit-{uuid.uuid4().hex}'
            reviewer.model = review_model
        else:
            raise ValueError('independent reviewer not configured')
        if not reviewer.session or reviewer.session in (builder_session, state.job.name):
            raise ValueError('independent session required')
        system_prompt = (
            'You are an independent PowerPoint reconstruction QA auditor (separate from the builder agent). '
            'The first three images are: (1) original source slide, (2) actual PowerPoint rendered slide, '
            'and (3) pixel-ownership map (red=unrecovered source-plate, blue=native shape, dark=native text, '
            'green=transparent asset, yellow=movable picture crop), plus objective scorecard metrics. '
            'Additional panels show actual Office edits. Check original positions for old text and neighbors for damage. '
            'Check strictly for CRITICAL defects (`severity="critical"`): '
            'missing/deleted text or graphics in image (2) vs (1), corrupted/wrong text wording, or severe '
            'double-rendered ghosting. Subtle font-weight, shadow omission, or minor gradient shade differences '
            'are `severity="minor"` (`font_or_shade_deviation`) and MUST NOT fail the audit (`passed=true`). '
            'Set `passed=false` ONLY when a `severity="critical"` defect exists. '
            'Set text_impact=appearance_only only if all content remains legible and intact. '
            'Report obscured words, missing glyph strokes and wrong/missing content with their respective '
            'text_impact even when font or colour is the apparent cause. Use unknown if uncertain.'
        )
        if numbered_image:
            system_prompt += (
                ' The fourth image shows numbered marks on the same source. Other panels are Office edit evidence. '
                'For each defect give mark_numbers from this page and the intended edit_action. '
                'Use incorrect_group when edit evidence shows members left behind. '
                'An unmatched new business word is missing_text. For an unnumbered finding give its bbox '
                'as [left,top,width,height] and bbox_space=source_px_xywh in the original source dimensions. '
                'Report unknown locations without a box. Do not infer a successful edit from an unedited image.'
            )
        control = review_control.verify_reviewer(reviewer,cache_dir/'controls'/provider,budget)
        verify_model = provider == 'moma' and builder_provider in ('deepseek', 'moma')
        if not control['passed'] or (verify_model and
                model_identity(control.get('reported_model')) != model_identity(review_model)):
            return {'status':'CONTROL_UNVERIFIED','passed':False,'control':control,'model':review_model,
                    'critical_defects':[],'minor_deviations':[]}
        if not budget.reserve('model_calls', {'tool':'audit_delivery','key':audit_key}):
            raise RuntimeError('previous independent audit response unknown')
        inputs = [source_path, render_path, owner_preview_path]
        if numbered_image:
            inputs.append(numbered_image)
        raw = reviewer.ask(system_prompt, {'scorecard': sc_summary}, [*inputs, *edit_panels], VLMIndependentAuditReply)
        reported = getattr(reviewer, 'reported_model', None)
        actual_id = model_identity(reported)
        if actual_id == model_identity(builder_model) or (verify_model
                and actual_id != model_identity(review_model)):
            return {'status':'MODEL_IDENTITY_UNVERIFIED', 'passed':False, 'provider':provider, 'model':review_model,
                    'reported_model':reported, 'builder_model':builder_model, 'session':reviewer.session,
                    'usage':reviewer.last_usage, 'receipt':audit_key, 'critical_defects':[], 'minor_deviations':[]}
        parsed = VLMIndependentAuditReply.model_validate(raw)
        critical, minor, passed = classify_audit(parsed)
        verdict = {
            'status': 'READY', 'provider':provider, 'model': review_model, 'reported_model':reviewer.reported_model, 'usage':reviewer.last_usage, 'session': reviewer.session, 'receipt': audit_key, 'control':control,
            'builder_model':builder_model, 'builder_provider':builder_provider, 'builder_session':builder_session,
            'passed': passed,
            'critical_defects': critical,
            'minor_deviations': minor,
            'summary': parsed.summary,
        }
        write(cache_file, verdict)
        return verdict
    except Exception as exc:
        # Keep only a normalized provider status, never the arbitrary response body.
        import re
        status = re.fullmatch(r'(?:opencode|moma)_http_(\d{3})', str(exc))
        return {
            'model': review_model, 'provider_http_status': int(status.group(1)) if status else None,
            'status': f'UNAVAILABLE:{type(exc).__name__}',
            'passed': False,
            'critical_defects': [],
            'minor_deviations': [],
            'summary': f'Independent audit unavailable ({type(exc).__name__}).',
        }


def run_delivery_audit(
    service,
    state,
    current: dict,
    directory: Path,
    *,
    use_vlm: bool = True,
) -> dict:
    """Compute per-page `PageScorecard/1`, inventory recall, tolerance gates, and stateless VLM audit."""
    tol = load_tolerance_config()
    fid_tol = tol.get('fidelity', {})
    sc_tol = tol.get('scorecard_gate', {})

    composition = service.composition(current)
    pptx_path = Path(composition['artifacts']['pptx'])
    deck_hash = digest(pptx_path)
    office = composition.get('office') or {}
    office_dir = Path(office['office_directory']) if office.get('office_directory') else None

    directory.mkdir(parents=True, exist_ok=True)
    page_audits: list[dict] = []
    all_critical: list[dict] = []
    all_warnings: list[str] = []

    for idx, page_row in enumerate(current['pages'], 1):
        src_path = Path(service.source(state, idx))
        if digest(src_path) != page_row['snapshot']['source_sha256']:
            src_path = Path(page_row['snapshot']['path'])
        render_path = (office_dir / f'page-{idx:02d}-before.png') if office_dir else None
        if render_path is not None and not render_path.is_file():
            render_path = None

        ocr_data = None
        ocr_file = current.get('ocr', {}).get(str(idx))
        if ocr_file and Path(ocr_file).is_file():
            ocr_data = json.loads(Path(ocr_file).read_text('utf8'))

        inv_data = None
        inv_file = current.get('page_structure', {}).get(str(idx))
        if inv_file and Path(inv_file).is_file():
            inv_data = json.loads(Path(inv_file).read_text('utf8'))

        sc = scorecard.score_page(
            src_path, pptx_path, render_path=render_path, ocr=ocr_data, slide=idx
        )
        owner_png = directory / f'owner-preview-{idx:02d}.png'
        scorecard.save_owner_preview(src_path, pptx_path, owner_png, slide=idx)
        inv_metrics = compute_inventory_and_form_metrics(sc, inv_data)
        sc['inventory'] = inv_metrics
        sc_file = directory / f'scorecard-{idx:02d}.json'
        scorecard.write_json(sc_file, sc)

        # Evaluate soft vs hard fidelity & plate thresholds
        fid = sc.get('fidelity') or {}
        plate_frac = float(sc.get('raster_plate_content_fraction', 1.0))
        de_p95 = float(fid.get('delta_e2000_p95', 0.0))
        ssim_val = float(fid.get('ssim', 1.0))

        if fid:
            if ssim_val < float(fid_tol.get('hard_min_ssim', 0.72)):
                all_critical.append({
                    'page': idx,
                    'category': 'severe_visual_divergence',
                    'description': f'SSIM {ssim_val:.4f} fell below hard floor {fid_tol.get("hard_min_ssim", 0.72)}',
                })
            elif de_p95 > float(fid_tol.get('max_delta_e2000_p95', 10.0)) or ssim_val < float(fid_tol.get('min_ssim', 0.85)):
                all_warnings.append(
                    f'page {idx}: soft fidelity deviation (SSIM={ssim_val:.4f}, dE00_p95={de_p95:.2f}) recorded under native-first tolerance'
                )

        if plate_frac > float(sc_tol.get('warn_raster_plate_content_fraction', 0.35)):
            all_warnings.append(
                f'page {idx}: {plate_frac:.1%} structured content still in source-plate (target <= {sc_tol.get("target_raster_plate_content_fraction", 0.15):.0%})'
            )

        compact_sc = {
            'page': idx,
            'object_counts': sc.get('object_counts'),
            'native_editable_content_fraction': sc.get('native_editable_content_fraction'),
            'movable_raster_content_fraction': sc.get('movable_raster_content_fraction'),
            'raster_plate_content_fraction': sc.get('raster_plate_content_fraction'),
            'fidelity': fid,
            'non_logo_line_recall': (sc.get('text') or {}).get('non_logo_line_recall'),
            'unmatched_ocr': [
                u for u in (sc.get('text') or {}).get('unmatched', []) if not u.get('is_logo')
            ][:10],
            'main_title_editable': inv_metrics.get('main_title_editable'),
            'missing_inventory_items': inv_metrics.get('missing_items', [])[:10],
        }

        edit_panels = [state.evidence(r['id'])[0] for r in composition.get('required_evidence', [])
                       if r['role'] == 'edit_panel' and Path(state.evidence(r['id'])[0]).parent.name == f'panels-{idx:02d}']
        numbered_image = None
        rows = []
        size = page_row['snapshot'].get('image_size')
        inventory_file = current.get('experiment_marks', {}).get(str(idx))
        if use_vlm and render_path is not None and inventory_file:
            from PIL import Image
            from .page_reading import mark_page
            inventory = json.loads(Path(inventory_file).read_text('utf8'))
            if inventory.get('source_sha256') != digest(src_path):
                raise ValueError('review inventory belongs to a different source')
            rows = inventory['rows']
            with Image.open(src_path) as original:
                size = list(original.size)
                numbered_image = directory/f'review-marks-{idx:02d}.png'
                mark_page(original.convert('RGB'), [(r['mark'],r) for r in rows]).save(numbered_image)
            compact_sc.update(source_size=size, numbered_inventory_sha256=digest(inventory_file))
        vlm_verdict = (
            _stateless_vlm_audit(state, src_path, render_path, owner_png, compact_sc, deck_path=pptx_path,
                                 edit_panels=edit_panels, numbered_image=numbered_image)
            if (use_vlm and render_path is not None) else {
                'status': 'SKIPPED', 'passed': False, 'critical_defects': [], 'minor_deviations': [], 'summary': '',
            }
        )
        if numbered_image:
            from .review_targets import locate
            structured = []
            for defect in vlm_verdict.get('critical_defects', []):
                bound = locate(defect, rows, size)
                fingerprint = key([idx,inventory['source_sha256'],bound['category'],bound.get('edit_action'),
                                   sorted(bound['mark_ids']),bound['bbox'],bound.get('candidate_ids')])
                structured.append({**bound, 'page':idx, 'finding_id':fingerprint[:24],
                                   'source_sha256':inventory['source_sha256'], 'deck_sha256':deck_hash})
            vlm_verdict = {**vlm_verdict, 'critical_defects':structured}
        for cd in vlm_verdict.get('critical_defects', []):
            all_critical.append({'page': idx, **cd})

        owner_ev = state.register(owner_png, 'owner_preview')
        sc_ev = state.register(sc_file, 'scorecard')
        page_audits.append({
            'page': idx,
            'scorecard_file': str(sc_file),
            'owner_preview_file': str(owner_png),
            'scorecard_evidence_id': sc_ev['id'],
            'owner_preview_evidence_id': owner_ev['id'],
            'summary': compact_sc,
            'inventory': inv_metrics,
            'vlm_audit': vlm_verdict,
        })

    audit_passed = bool(page_audits) and not all_critical and all(p['vlm_audit']['passed'] for p in page_audits)
    report = {
        'schema': 'IndependentDeliveryAudit/1',
        'tolerance_profile': tol.get('profile'),
        'composition_sha256': key(composition),
        'passed': audit_passed,
        'critical_defects': all_critical,
        'warnings': all_warnings,
        'pages': page_audits,
    }
    write(directory / 'independent-audit.json', report)
    current['independent_audit'] = report
    return report
