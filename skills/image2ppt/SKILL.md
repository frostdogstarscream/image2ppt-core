---
name: image2ppt
description: Develop, debug, run and validate the Image2PPT parallel_pipeline automatic screenshot-to-editable-PPT product. Use for its text, asset and surface branches, local reconstruction failures, editability coverage and actual PPTX acceptance.
---

# Image2PPT automatic product engineering

The primary workflow is the r10 automatic product: `product.parallel_pipeline.run`. Codex develops and diagnoses the product; the backend performs planning and reconstruction. **Do not hand-author SlideIR, manually substitute a page plan, or manually assemble a better-looking deck in place of automatic product planning.** Such an output cannot count as an automatic regression result. Diagnostic fixtures and explicitly requested manual comparisons must be labeled and kept separate.

Preserve source wording, numbers, hierarchy, layout, colors and proportions. Preserve originals, historical outputs, existing services and unrelated worktree changes. Use the existing fixed compiler and public SlideIR contract.

## Locate and run the real entrypoint

1. Identify the actual Git worktree, branch, dirty changes and task directory. Read that worktree's `AGENTS.md` and `services/product/PARALLEL_R10.md`. If `services/product/product/parallel_pipeline.py` is absent, locate the established product worktree; do not bootstrap from the legacy manual workflow or invent a replacement implementation.
2. Read [automatic product development and acceptance](references/parallel-product.md). For compiler diagnostics only, read [core tooling](references/core.md) and [SlideIR](references/slideir.md); those documents describe downstream capabilities, not the automatic planning entrypoint.
3. Use `services/product/launch.py` for the upload service, `services/product/run_local.py` for the same pipeline locally, or `services/product/run_parallel_baseline.py` for a frozen regression. Inspect their current arguments and environment requirements. `prepare_parallel_rebuild.py` is an explicitly labeled cached-evidence reconstruction replay, not proof of fresh-upload planning.
4. Default to `parallel_review`. `parallel_shadow` is diagnostic. `parallel_auto` remains gated; changing instructions, passing unit tests or receiving one Go PASS never opens it. Do not invoke legacy serial `run()` or a whole-page VisualPage plan as an automatic fallback.

## Engineering invariants

- Three branches share one immutable source: text supplies OCR/content/typography evidence; assets supply complete logical objects and original RGBA/layer evidence; surfaces supply backing, cards and simple geometry. MoMA plans and repairs; Go independently reviews. LayerD is shared per input; SAM is a candidate generator. A model reply is evidence, not permission to erase source pixels.
- SceneGraph owns the fused scene. Resolve intrinsic logo lettering, independent labels, parts, backing and occlusions before rebuilding. Keep unresolved evidence explicit. Only the scene adapter emits public SlideIR; branches must not independently publish or delete other branches' content.
- Target local object/region transactions. Font sharing is not a pixel dependency; an unrelated failed object must not veto all page surfaces. Split safe subregions and retain their dependencies. Indivisible parts and overlapping ownership require atomic replacement. These are development requirements: inspect code and receipts before claiming they are implemented.
- Keep source-plate as safe backing until source ownership, replacement and backing repair are proved. Never overlay new text on uncleared old text, erase by bbox, hide the whole plate to inflate editability, or cut an uncertain object at a region boundary. Raster islands require separate validation before activation.
- Preserve verified stable asset bytes unless changed source/ownership requires revalidation. Maintain input/version/hash-bound caches, durable external-call reservations and task-level generation commits. An unknown paid response must not be automatically resent as the same request.
- Measure visual integrity and editability separately. Native-node count or the existence of one date textbox does not prove a reconstructed page. Missing independent denominators produce UNKNOWN/null recall. Raster fallback is safe delivery, not editable success.
- Validate the final PPTX in actual PowerPoint: object edits, movement/scaling/deletion, save/reopen, sibling stability and outside-transaction pixels. Check target-region residuals separately; an outside-pixel PASS alone is insufficient. Go must review the bound source/final preview and available edit evidence; unavailable QA remains UNKNOWN.

## Verification and reporting

Run focused product/planner/orchestrator tests for behavioral changes and `./image2ppt.ps1 test` for protocol changes. Compiler/OOXML changes additionally require the synthetic demo, `scripts/check_demo.mjs`, actual-file previews and retained reports. Synthetic tests establish behavior, not real-slide accuracy or human acceptance. A documentation-only change does not require model calls or rebuilding customer decks.

Inspect real source/final previews and report tested versus untested gates. Keep `human_acceptance`, WPS checks, automatic quality, staged transactions and publication distinct. Deliver PPTX, whole-page previews and a short honest explanation when generation is requested; retain detailed diagnostics internally unless requested. State partial editability and full-raster pages without claiming all objects are independently editable.

## Historical manual workflow and maintenance

The previous Codex visual planning/hand-written SlideIR procedure is preserved in [historical manual workflow](references/legacy-manual-workflow.md). Its code, fixed compilation and replay abilities remain available. It is not the default product route, and its approximate mean-pixel-difference threshold of 10 is not a universal r10 acceptance threshold.

Edit `skills/image2ppt/`, then use `exports/sync_codex_skill.py` to sync/check `.agents/skills/image2ppt/` in the intended worktree. Re-export with `exports/build_source_package.py` and `exports/package_skill_entries.py`; inspect their manifests. Claude has its own maintained source, `skills/image2ppt-claude/`: never overwrite or derive it from this skill. Skill ZIPs and the fixed-core source export are not a complete automatic backend deployment.
