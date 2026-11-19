---
name: image2ppt-runtime
description: Reconstruct uploaded slide images into maximally native-editable PowerPoint decks using deterministic backbone perception/staging, targeted exception repair, and independent scorecard/visual audit.
---

# Image2PPT Native-First Autonomous Runtime

Reconstruct uploaded slide screenshot(s) into a maximally native-editable PowerPoint (`result.pptx`) within the job budget:
- **Primary goal**: Maximize native editable objects (`text`, `shape` with solid/gradient/preset geometries `roundRect`/`pentagon`/`chevron`/`ellipse`, `chart`, `table`) and clean movable assets/photos, leaving `source-plate` with clean background only.
- **Two-tier gates**:
  - **Hard safety gates (mandatory)**: Exact source wording & numbers, zero duplicate overlaid text, zero unknown-pixel bbox erasure, zero source-ink ghosting, and actual PowerPoint verification (`PASS`).
  - **Soft visual tolerances**: Minor font-family/weight, shadow-omission, or gradient-shade deviations within configured tolerance belong in `minor_deviations`, never in `findings` and never as an excuse to drop recoverable cards/headings back to `source-plate`.

## Autonomous Reconstruction Flow (`tool_profile=all`)

1. **Deterministic Backbone & Scene Check**:
   - Call `get_job_context` and `get_evidence` on the source image(s).
   - Call `prepare_page_evidence(page=...)` once per page (runs local OCR, LayerD, PP-DocLayoutV3, VLM OCR typo proofreading & structure inventory, style calibration, and initializes the scene).
   - If `scene_relations.logo_lockup_candidates` lists brand lettering next to a logo (e.g., `中国移动` / `China Mobile`) or intrinsic text inside an illustration, remove those uncommitted draft text IDs via `update_scene(remove_object_ids=[...])` so brand marks stay crisp. Keep `prepare_page_evidence` assets intact.
   - Call `batch_reconstruct_page(page=..., include_assets=true, include_containers=true)` and immediately commit the returned `VALIDATED` candidate via `commit_candidate(candidate_id=..., expected_version=...)`.

2. **Targeted Exception Surgery & Photo/Icon Module Finalization**:
   - Inspect `failed_regions` from `batch_reconstruct_page`. Only spend bounded surgery turns on high-value recoverable headings/body texts (`max_repair_attempts >= 1`) that were not yet committed; skip `max_repair_attempts == 0` items (such as brand logos).
   - Call `crop_region(page=...)` (auto mode: omits `object_ids`/`bbox`) to stage framed photos and remaining isolated graphics as movable modules AND automatically upgrade newly cleaned enclosing cards into native `Shape`s (`s-*`). If `STAGED_LOCAL`, commit via `commit_candidate(candidate_id=..., expected_version=...)`.

3. **Independent Audit, Visual Review & Finalization**:
   - Call `compile_pptx()` (no `candidate_id`) and `validate_office()`.
   - Inspect `validate_office.scorecard_audit` (or call `audit_delivery()`) and call `get_evidence` on **every** ID in `required_evidence` (source, `office_preview`, `edit_panel`s, and `hard_checks`).
   - If `audit_delivery` or visual inspection reveals a critical defect (`missing_content`, `duplicate_ghosting`, or an easily fixable collision), repair it locally and re-run `compile_pptx()` -> `validate_office()`; otherwise proceed to review.
   - Call `record_visual_review` (`checked` must include `content`, `completeness`, `layout`, `residual`, `background`, `editing`; record subtle font/gradient/shadow differences or retained raster regions in `minor_deviations` / `observations`, and reserve `findings` strictly for unresolved critical defects where `passed=false`), then call `finalize_job`.

When `tool_profile=basic` omits batch shortcuts, compose atomic tools directly (`run_ocr`, `list_fonts`, `decompose_layers`, `create_scene`, `fit_background`, `complete_text_mask`, `create_candidate`, `commit_candidate`).

## Key Tool Rules

- Tool names carry the `image2ppt_` prefix in OpenCode. Supply a unique `idempotency_key` on every new mutating call; reuse the exact key only when reconciling an uncertain transport response.
- Candidate IDs (`c-...`) are distinct from scene object IDs (`t-...`, `a-...`, `s-...`, `r-...`). `commit_candidate` requires `expected_version`.
- Never describe a raster-only fallback (`NO_EDITABLE_RECONSTRUCTION`) as editable reconstruction success. Report native editable objects, movable photo/icon modules, and any retained `source-plate` areas accurately.
