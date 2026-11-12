---
name: image2ppt-runtime
description: Autonomously reconstruct uploaded screenshot pages into editable PowerPoint using composable Image2PPT evidence and reconstruction tools.
---

# Image2PPT autonomous reconstruction

Complete the user's image-to-PPTX goal within the job budget. Observe the original images and actual intermediate results, choose useful tools, revise your approach when evidence reveals a defect, and stop when quality is acceptable. The user should not need to direct each object.

## Tools and planning

Use get_job_context to obtain immutable sources and task identity. Use get_evidence to receive actual images. Tool names have the image2ppt_ prefix. Available tools include OCR, semantic analysis, LayerD, SAM, text masks, gradient/background extraction, inpainting, foreground unmixing, fonts, scene proposals, candidates, image operations and optional scripts. Tools can be combined in the order justified by the image. The full pipeline and regional rebuild tools are optional shortcuts.

Create scene proposals with source wording, complete logical objects and evidence references. create_scene accepts text/asset/surface branch evidence. Ownership is derived by the host. Inspect returned ownership and conflicts. You may start with only the evidence relevant to your chosen object. A source-backed page with unprocessed objects is permitted.

After committing an object, update_scene can add or revise other draft objects. It preserves accepted definitions and shared styles. Revising an accepted object requires dependency-local rollback. Preserve a source text search scope corroborated by OCR. Transparent native frame expansion is a separate layout proposal.

Text evidence uses styles, texts and members. members maps each text ID to the independent OCR line IDs that support its exact wording. The host can bind a unique exact OCR line with matching geometry. AMBIGUOUS text cannot be erased. Match source words and geometry, then inspect the ownership diagnostics when unresolved.

A layout result is a proposal. A mask or fitted background is evidence. A candidate exists only when create_candidate or a reconstruction tool returns a candidate ID. commit_candidate accepts that returned candidate ID after validate_candidate succeeds. Object IDs and candidate IDs have different roles.

Tool inputs are evidence IDs and scene/object references. Generate a unique idempotency_key for each new operation. Reuse the same key and arguments to reconcile an uncertain response. Only committing requires expected_version. Read current version before submitting a candidate. Do not invent results, model calls, file paths, object counts or acceptance.

## Quality and exploration

Preserve exact wording, numbers, punctuation, hierarchy and the original proportions. Never authorize erasure by bbox or SAM mask alone. Unknown pixels remain protected. Candidate preparation and exploratory scripts do not grant ownership. The host independently validates masks, source_owner, neighboring objects, background and actual edit behavior.

Prefer existing reliable evidence and low-cost tools. A failed attempt should lead to new evidence or a changed strategy. Do not repeat the same failure with unchanged inputs. Mild font, italic, spacing, antialiasing and subtle background differences may be accepted. Obvious residuals, missing foreground, wrong text, damaged neighbors and unusable editing must be corrected or safely reverted.

Read the specific failed checks. BACKGROUND_REFERENCE_UNRELIABLE concerns independent source sampling. A different raw repair does not change that reference proof. Obtain genuinely different source evidence or continue an independent object. FONT_LAYOUT_FAILED may include Office-measured transparent frame extents. Frame repair does not require changing the source text search bbox or erase Mask.

Make progress toward the user's requested coverage. A useful verified object can establish the method before spending budget fitting many uncommitted objects. When the requested deliverable is satisfied, finish its checks and stop. Preserve other content as raster when further recovery is unjustified.

Read the task budget and remaining_seconds in tool responses. Account for compilation, final Office checks, viewing evidence, self-review and finalization before adding more objects. Once the requested minimum coverage is verified, prioritize completing acceptance. Further coverage must fit the remaining budget together with its required checks.

Observe candidates before accepting them. Tools return evidence IDs for images and receipts. Use get_evidence for close inspection. You may use run_image_script only when its OS sandbox is available; outputs remain unverified candidates. Do not modify product code or manually construct final SlideIR/PPTX. The host's scene adapter and fixed compiler produce deliveries.

Evidence references include names and roles. Choose the source_crop and written_crop images for local comparisons when they answer the question. Use full-page images for page context and final review. This keeps repeated image observations focused.

## Final result

Compile the chosen scene and run validate_office. Inspect every required source, Office preview, edit panel and hard-check receipt using get_evidence. Visually examine text, completeness, layout, old residuals, background and actual edits. If defects remain, inspect, repair or rollback the affected region, then recompile and recheck the changed result.

Record visual review with passed, findings, minor_deviations, evidence_ids, checked and reason. checked must contain content, completeness, layout, residual, background and editing. The same Agent performs this review. Go is not used on this path. A provided-image receipt records transport provision; give a genuine visual judgment.

Finalize only with current hard checks and visual evidence. Safe partial editability is allowed. Report editable objects, retained raster regions, unresolved issues and why further work stopped. Unavailable Office or vision evidence remains UNKNOWN. Do not claim complete recall without an independent reference denominator.
