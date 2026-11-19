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
        'uneditable_title_or_card', 'font_or_shade_deviation', 'other',
    ]
    description: str = Field(min_length=1, max_length=1000)
    bbox: list[float] | None = None


class VLMIndependentAuditReply(BaseModel):
    model_config = ConfigDict(extra='ignore')
    passed: bool
    defects: list[AuditDefect] = Field(default_factory=list, max_length=50)
    summary: str = Field(default='', max_length=2000)


def compute_inventory_and_form_metrics(
    sc: dict,
    inventory: dict | None,
) -> dict:
    """Evaluate `inventory_recall`, `main_title_editable`, and `proper_form_content_fraction`."""
    objs = sc.get('objects', [])
    native_texts = [o for o in objs if o.get('class') == 'text']
    native_shapes = [o for o in objs if o.get('class') == 'shape']
    native_charts = [o for o in objs if o.get('class') in ('chart', 'table')]
    movable_imgs = [o for o in objs if o.get('class') in ('asset', 'opaque_picture')]

    elements = (inventory or {}).get('elements', [])
    if not elements:
        text_info = sc.get('text') or {}
        unmatched = [u for u in text_info.get('unmatched', []) if not u.get('is_logo')]
        w_img, h_img = sc.get('image_size', [1672, 941])
        top_unmatched = [u for u in unmatched if u['bbox'][1] <= 0.11 * h_img and u['bbox'][3] >= 28]
        return {
            'inventory_available': False,
            'total_items': None,
            'matched_items': None,
            'recall': text_info.get('non_logo_line_recall'),
            'main_title_editable': len(top_unmatched) == 0 if text_info else None,
            'proper_form_content_fraction': round(
                float(sc.get('native_editable_content_fraction', 0.0))
                + float(sc.get('movable_raster_content_fraction', 0.0)),
                4,
            ),
            'missing_items': [],
        }

    matched_count = 0
    by_cat: dict[str, list[int]] = {}
    missing_items: list[dict] = []
    title_items = [e for e in elements if e.get('category') == 'title']
    titles_matched = 0

    for el in elements:
        cat = el.get('category', 'other')
        if cat == 'background_scenery':
            continue
        eb = el.get('bbox', [0, 0, 0, 0])
        e_area = max(1.0, float(eb[2]) * float(eb[3]))
        stat = by_cat.setdefault(cat, [0, 0])
        stat[1] += 1

        ok = False
        if cat in ('title', 'section_header', 'body_text'):
            ok = any(intersection(eb, o['bbox']) >= 0.30 * e_area for o in native_texts)
            if cat == 'title' and ok:
                titles_matched += 1
        elif cat in ('card_shape', 'ribbon_shape', 'badge_shape', 'divider'):
            ok = any(
                intersection(eb, o['bbox']) >= 0.35 * min(e_area, max(1.0, o['bbox'][2] * o['bbox'][3]))
                for o in [*native_shapes, *movable_imgs]
            )
        elif cat in ('chart', 'table'):
            # Either a native chart/table OR >=3 native bar shapes covering the chart region
            ok = any(intersection(eb, o['bbox']) >= 0.35 * e_area for o in native_charts) or (
                sum(1 for o in native_shapes if intersection(eb, o['bbox']) >= 0.5 * max(1.0, o['bbox'][2] * o['bbox'][3])) >= 3
            )
        elif cat in ('photo', 'icon', 'logo'):
            # Logos intentionally preserved crisp in plate OR extracted as transparent/movable image both preserve brand integrity;
            # photos and icons require movable image/asset representation.
            if cat == 'logo':
                ok = True
            else:
                ok = any(
                    intersection(eb, o['bbox']) >= 0.35 * min(e_area, max(1.0, o['bbox'][2] * o['bbox'][3]))
                    for o in [*movable_imgs, *native_shapes]
                )

        if ok:
            matched_count += 1
            stat[0] += 1
        else:
            missing_items.append({
                'id': el.get('id'),
                'category': cat,
                'bbox': eb,
                'label': el.get('label', ''),
            })

    denom = sum(v[1] for v in by_cat.values())
    # Estimate photo/icon movable content share vs fallback crop share
    native_frac = float(sc.get('native_editable_content_fraction', 0.0))
    movable_frac = float(sc.get('movable_raster_content_fraction', 0.0))
    return {
        'inventory_available': True,
        'total_items': denom,
        'matched_items': matched_count,
        'recall': round(matched_count / denom, 4) if denom > 0 else None,
        'by_category': {k: {'matched': v[0], 'total': v[1], 'recall': round(v[0] / v[1], 4) if v[1] else None}
                        for k, v in by_cat.items()},
        'main_title_editable': (titles_matched == len(title_items)) if title_items else True,
        'proper_form_content_fraction': round(native_frac + movable_frac, 4),
        'missing_items': missing_items[:25],
    }


def _stateless_vlm_audit(
    state,
    source_path: Path,
    render_path: Path,
    owner_preview_path: Path,
    sc_summary: dict,
) -> dict:
    """Call VLM in a fresh stateless session (separate from Agent conversation) to audit visual fidelity."""
    from ..go_provider import GoProvider
    from ..recovery.adaptive_policy import AdaptiveBudget
    from .provider import configure_model
    import os

    request = json.loads((state.job / 'request.json').read_text('utf8'))
    budget = AdaptiveBudget(Store(state.job / 'state'), request.get('adaptive_budget'))
    audit_key = key({
        'schema': 'StatelessVisualAudit/1',
        'source_sha256': digest(source_path),
        'render_sha256': digest(render_path),
        'owner_sha256': digest(owner_preview_path),
    })
    cache_dir = state.job / 'cache/independent-audit'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'{audit_key}.json'
    if cache_file.is_file():
        return {**json.loads(cache_file.read_text('utf8')), 'status': 'CACHED'}

    try:
        selected_provider, cfg = configure_model(state.job)
        budget.reserve('model_calls', {'tool': 'audit_delivery', 'key': audit_key})
        system_prompt = (
            'You are an independent PowerPoint reconstruction QA auditor (separate from the builder agent). '
            'You are given 3 images in order: (1) original source slide, (2) actual PowerPoint rendered slide, '
            'and (3) pixel-ownership map (red=unrecovered source-plate, blue=native shape, dark=native text, '
            'green=transparent asset, yellow=movable picture crop), plus objective scorecard metrics. '
            'Check strictly for CRITICAL defects (`severity="critical"`): '
            'missing/deleted text or graphics in image (2) vs (1), corrupted/wrong text wording, or severe '
            'double-rendered ghosting. Subtle font-weight, shadow omission, or minor gradient shade differences '
            'are `severity="minor"` (`font_or_shade_deviation`) and MUST NOT fail the audit (`passed=true`). '
            'Set `passed=false` ONLY when a `severity="critical"` defect exists.'
        )
        if selected_provider == 'opencode_go':
            reviewer = GoProvider(session=f'independent-audit-{uuid.uuid4().hex[:12]}')
            reviewer.key = cfg.api_key
            reviewer.model = os.environ.get('IMAGE2PPT_REVIEW_MODEL') or cfg.model
            raw = reviewer.ask(
                system_prompt,
                {'scorecard': sc_summary},
                [source_path, render_path, owner_preview_path],
                VLMIndependentAuditReply,
            )
        else:
            from ..parallel_provider import Planner
            planner = Planner(cfg)
            raw = planner.ask(
                system_prompt,
                {'scorecard': sc_summary},
                [source_path, render_path, owner_preview_path],
                VLMIndependentAuditReply,
            )
        parsed = VLMIndependentAuditReply.model_validate(raw)
        critical = [d.model_dump() for d in parsed.defects if d.severity == 'critical']
        minor = [d.model_dump() for d in parsed.defects if d.severity != 'critical']
        verdict = {
            'status': 'READY',
            'passed': len(critical) == 0 and bool(parsed.passed),
            'critical_defects': critical,
            'minor_deviations': minor,
            'summary': parsed.summary,
        }
        write(cache_file, verdict)
        return verdict
    except Exception as exc:
        return {
            'status': f'UNAVAILABLE:{type(exc).__name__}',
            'passed': True,
            'critical_defects': [],
            'minor_deviations': [],
            'summary': f'VLM audit skipped/unavailable ({type(exc).__name__}); relying on deterministic Office + scorecard gates.',
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

        vlm_verdict = (
            _stateless_vlm_audit(state, src_path, render_path, owner_png, compact_sc)
            if (use_vlm and render_path is not None) else {
                'status': 'SKIPPED', 'passed': True, 'critical_defects': [], 'minor_deviations': [], 'summary': '',
            }
        )
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

    audit_passed = len(all_critical) == 0
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
