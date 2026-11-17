---
name: image2ppt-runtime
description: Autonomously reconstruct uploaded screenshot pages into editable PowerPoint using composable Image2PPT evidence, batch backbone, and regional repair tools.
---

# Image2PPT autonomous reconstruction

Complete the user's image-to-PPTX goal within the job budget. Observe the original images and actual intermediate results, choose useful tools, revise your approach when evidence reveals a defect, and stop when quality is acceptable. The user should not need to direct each object.

## Recommended 4-stage workflow (`tool_profile=all`)

When `prepare_page_evidence` and `batch_reconstruct_page` appear in `get_job_context.tool_catalog`, use the hybrid 4-stage pipeline so 100+ object slides finish cleanly within 8–18 turns instead of exhausting turns on repetitive atomic calls:

1. **Stage 1 — Page Evidence & Calibrated Scene (`prepare_page_evidence`)**:
   - Call `get_job_context` and `get_evidence` on the source image(s).
   - Call `prepare_page_evidence(page=...)` once per page. It runs local OCR, LayerD decomposition, optional PP-DocLayout region detection, pixel-measured foreground color extraction, decorative affix splitting (e.g., separating `>>` or bullets from text), page-level font-size tier clustering, **automatic LayerD graphic/badge/illustration discovery** (`assets`), and initializes the `RESOLVED` draft scene.
2. **Stage 2 — Layout Review & Structural Scene Refinement (`update_scene`)**:
   - Compare the source image against `prepare_page_evidence.items`, `assets`, `styles`, and **`scene_relations`** (`merged_paragraphs`, `merged_headers`, `logo_lockup_candidates`, `dual_color_colon_lines`) before calling `batch_reconstruct_page`.
   - Use a single `update_scene` call when helpful:
     - Pass `remove_object_ids=[...]` to remove `logo_lockup_candidates` (brand lettering next to a logo such as `中国移动` / `China Mobile`) and any intrinsic text inside photos or charts so brand marks and illustrations remain crisp in raster backing instead of being overwritten with system fonts;
     - Merge any remaining multi-line card paragraphs missed by `scene_relations.merged_paragraphs` (`\n`-joined wording with ordered `members`).
   - Keep the auto-discovered `assets` from `prepare_page_evidence` intact (including `deferred_after_text` composite badges, which `batch_reconstruct_page` automatically stages right after erasing adjacent captions). Only add extra `asset.objects` if an important standalone icon was missed by `prepare_page_evidence.assets`.
3. **Stage 3 — Parallel Batch Reconstruction & Unified Commit (`batch_reconstruct_page` -> `commit_candidate`)**:
   - Call `batch_reconstruct_page(page=..., include_assets=true)` to evaluate gradient/LayerD/LaMa text candidates in parallel, stage verified textboxes, extract whole movable PNG assets (with SAM3 and RMBG-2.0 matting fallback when configured), extract pixel-proven draggable container boxes/cards (`s-*` native shapes and `a-c*` stylized banner/divider layers), **automatically crop remaining complex visuals/charts/photos into opaque `r-*` region pictures** (`MOVABLE` when surrounding backing is proven, `MOVE_UNSAFE` in-place when backing cannot be proven), and run a single batch PowerPoint check with per-object defect isolation.
   - `batch_reconstruct_page` returns a `VALIDATED` `candidate_id` without modifying the committed scene directly; immediately call `commit_candidate(candidate_id=..., expected_version=...)` using the `version` returned by `batch_reconstruct_page` so all objects commit through the unified hard/text/ownership/Office checks.
   - Inspect `committed_count`, `staged_object_ids`, `isolated_dropped_ids`, and the prescriptive `failed_regions` digest (`diagnostic_category`, `conflicting_peer_ids`, `recommended_action`, `max_repair_attempts`, `covered_by_region_crop`).
4. **Stage 4 — Prescriptive Bounded Surgery, Region Crop Fallback, Final Office Gate & Self-Review**:
   - Immediately skip any item in `failed_regions` whose `max_repair_attempts == 0` (`recommended_action == 'already_cropped_as_region_image'` or `'retain_in_source_plate'`), such as `LOGO_OR_INTRINSIC_TEXT`, `COMPLEX_OR_DUAL_COLOR_BACKING`, or `UNVERIFIED_ASSET_CLOSURE`.
   - Spend at most **2 turns total** on high-value text items with `max_repair_attempts == 1`, applying at most **1 targeted attempt per region** matching its exact `recommended_action` and `conflicting_peer_ids` (`merge_with_peer_or_shrink_bbox_2px` / `shrink_text_bbox_away_from_asset` / `fit_text_layout_or_reduce_size_1px` / `fit_background_context_36`) **before** final composition. You may also call `crop_region` -> `commit_candidate` on an isolated unrecoverable region instead of leaving it in `source-plate`.
   - Once ready to deliver (or whenever `remaining_seconds < 450`), run the final delivery sequence without starting new region experiments: call `compile_pptx()` (no `candidate_id`), `validate_office()`, `get_evidence` on **every** ID in `required_evidence` (source, `office_preview`, `edit_panel`s, and `hard_checks`), then immediately call `record_visual_review` (recording any `MOVE_UNSAFE` region crops or areas retained in `source-plate` honestly under `observations` or `minor_deviations`, never `findings` when `passed=true`) and `finalize_job`.

When `tool_profile=basic` omits batch shortcuts, compose atomic tools directly (`run_ocr`, `list_fonts`, `decompose_layers`, `create_scene`, `fit_background`, `complete_text_mask`, `create_candidate`, `commit_candidate`).

## Tools and planning

Use `get_job_context` to obtain immutable sources and task identity. Use `get_evidence` to receive actual images. Tool names have the `image2ppt_` prefix. Available tools include `prepare_page_evidence`, `batch_reconstruct_page`, OCR, semantic analysis, LayerD, SAM, text masks, gradient/background extraction, inpainting, foreground unmixing, fonts, scene proposals, candidates, image operations and optional scripts.

Create scene proposals with source wording, complete logical objects and evidence references. `create_scene` accepts `text`/`asset`/`surface` branch evidence; ownership is derived by the host. `SCENE_EXISTS` means use `update_scene`. Keep object IDs stable across turns. `update_scene` preserves accepted objects and shared styles; revising an accepted object requires `rollback_region`. `remove_object_ids` removes only uncommitted drafts and never removes source OCR neighbor protection.

Text evidence uses `styles`, `texts` and `members`. `members` maps each text ID to the OCR line IDs supporting its wording. Unique exact wording (or a complete alphanumeric core with purely decorative non-alphanumeric affixes such as `>>` or `•` and matching sub-bbox) can be bound by the host. `AMBIGUOUS` text cannot be erased.

`create_candidate` returns an `EXPLORATORY` candidate with no erase authority. Use `compile_pptx(candidate_id)` to inspect native rendering and `validate_office(candidate_id)` for preview edit diagnostics. `commit_candidate` runs formal source-owner, backing, and actual PowerPoint editing checks (`expected_version` required). Object IDs and candidate IDs (`c-...`) have different roles.

Scene-mode `complete_text_mask` requires `page`, `object_id`, and `background_id` (omit `original_mask_id` unless supplying a binary grayscale mask). `unmix_foreground` takes `evidence_id`, `background_id`, and `mask_id`. Generate a unique `idempotency_key` for each new mutating call; reuse the exact key and arguments only to reconcile an uncertain response.

## Quality, safety and final review

Preserve exact wording, numbers, punctuation, hierarchy, measured colors, and original proportions. Never authorize erasure by bbox or SAM mask alone. Unknown pixels remain protected in `source-plate`. Verified LayerD lower composites and Big-LaMa inpainting are validated by context-ring, seam continuity, and stroke-ablation gates in addition to planar gradient fits; `FONT_LAYOUT_FAILED` repairs use transparent `frame_minimum` without altering the erase mask.

Record visual review via `record_visual_review` with `passed`, `findings`, `observations`, `minor_deviations`, `evidence_ids`, `checked`, and `reason`. `checked` must contain all six categories: `content`, `completeness`, `layout`, `residual`, `background`, and `editing`. `findings` are unresolved defects (requires `passed=false`); accepted subtle differences belong in `minor_deviations`, descriptive notes in `observations`. Include every required evidence ID from `validate_office` after viewing them with `get_evidence`.

Finalize with `finalize_job` only when current Office checks and visual review pass. A raster-only scene has `NO_EDITABLE_RECONSTRUCTION`—never describe it as editable success. Safe partial editability is encouraged when complex illustrations or uncertain regions are best preserved in raster backing; state editable objects and retained raster regions honestly.
