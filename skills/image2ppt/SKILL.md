---
name: image2ppt
description: Develop, debug and validate the OpenCode image-to-editable-PowerPoint runtime. Use for upload, field recognition, object reconstruction, independent review and actual PowerPoint edit checks.
---

# Image2PPT automatic product engineering

The primary automatic product mainline is the OpenCode runtime (`orchestrator=opencode`, `/api/v1/jobs`, `services/product/opencode_client.py`, and `skills/image2ppt-runtime/SKILL.md`), targeting maximally native-editable output (`text`, `shape` with solid/gradient/preset geometry, `chart`, `table`, and independent image modules) under the two-tier safety/tolerance config (`configs/fidelity_tolerance.json`). The r10/R12 pipeline (`product.parallel_pipeline.run`) is retained as a frozen comparison baseline. Codex develops and diagnoses the product; the backend performs planning and reconstruction. **Do not hand-author SlideIR, manually substitute a page plan, or manually assemble a better-looking deck in place of automatic product planning.** Such an output cannot count as an automatic regression result. Diagnostic fixtures and explicitly requested manual comparisons must be labeled and kept separate.

Preserve source wording, numbers, hierarchy, layout, colors and proportions. Preserve originals, historical outputs, existing services and unrelated worktree changes. Use the existing fixed compiler and public SlideIR contract.

## Locate and run the real entrypoint

1. Identify the Git worktree, branch, local changes and task directory. Read that checkout's `AGENTS.md`, `services/product/OPENCODE_RUNTIME.md` and `skills/image2ppt-runtime/SKILL.md`. If the directory lacks the runtime client or service, locate the product worktree that contains them and confirm the service revision used by the job.
2. Read [automatic product development and acceptance](references/parallel-product.md). The [core tooling](references/core.md) and [SlideIR](references/slideir.md) references cover downstream compiler diagnostics.
3. Submit fresh images through OpenCode runtime at `POST /api/v1/jobs`. Read the host-prepared evidence and Office baseline before incremental work. Use `run_local.py` and the r10/R12 pipeline only for frozen comparisons.

## Engineering invariants

- The host prepares OCR, text, asset and surface evidence from one immutable source and checks a deterministic baseline in Office. The OpenCode Agent uses numbered questions to plan repairs. Model roles are deployment-configured. Verify the builder and independent reviewer from the job receipts. LayerD is shared per input and optional segmentation provides candidates. Pixel ownership is established by the host.
- SceneGraph owns the fused scene. Resolve intrinsic logo lettering, independent labels, parts, backing and occlusions before rebuilding. Keep unresolved evidence explicit. Only the scene adapter emits public SlideIR; branches must not independently publish or delete other branches' content.
- Target local object/region transactions. Font sharing is not a pixel dependency; an unrelated failed object must not veto all page surfaces. Split safe subregions and retain their dependencies. Indivisible parts and overlapping ownership require atomic replacement. These are development requirements: inspect code and receipts before claiming they are implemented.
- Keep source-plate as safe backing until source ownership, replacement and backing repair are proved. Never overlay new text on uncleared old text, erase by bbox, hide the whole plate to inflate editability, or cut an uncertain object at a region boundary. Raster islands require separate validation before activation.
- Preserve verified stable asset bytes unless changed source/ownership requires revalidation. Maintain input/version/hash-bound caches, durable external-call reservations and task-level generation commits. An unknown paid response must not be automatically resent as the same request.
- Measure visual integrity and editability separately. Native-node count or the existence of one date textbox does not prove a reconstructed page. Missing independent denominators produce UNKNOWN/null recall. Raster fallback is safe delivery, not editable success.
- Validate the selected PPTX in actual PowerPoint. Replace one business field without taking a neighboring separator. Move a card with one selection and check its intended text and icon also moved. Recolor its base, then delete an independent photo and check the exposed color and remaining title. For applicable tables and charts, edit a cell or datum. Save and reopen each tested copy, then inspect the old location and adjacent content. Record the specified edit and the correctness of the initial wording separately. Preserve ambiguous source readings for confirmation.
- A complex image fallback must have one identified, independently movable object. Count residual business text on the background as unfinished. A full-page raster is ineligible for editable delivery. Bind independent visual review to the source, final preview and edit evidence and report its actual provider, model and status.

## Verification and reporting

Run focused product/planner/orchestrator tests for behavioral changes and `./image2ppt.ps1 test` for protocol changes. Compiler/OOXML changes additionally require the synthetic demo, `scripts/check_demo.mjs`, actual-file previews and retained reports. Synthetic tests establish behavior, not real-slide accuracy or human acceptance. A documentation-only change does not require model calls or rebuilding customer decks.

Inspect real source/final previews and report tested versus untested gates. Distinguish a host baseline, an Office-checked working copy and a selected download. Report the specific edit results, visual quality, independent review and human/WPS status. When the product holds publication, give the reason and the diagnostic files actually listed by the artifact API. Label any separately shared research PPTX with its origin.

## Historical manual workflow and maintenance

The previous Codex visual planning/hand-written SlideIR procedure is preserved in [historical manual workflow](references/legacy-manual-workflow.md). Its code, fixed compilation and replay abilities remain available. It is not the default product route, and its approximate mean-pixel-difference threshold of 10 is not a universal r10 acceptance threshold.

Edit `skills/image2ppt/`, then use `exports/sync_codex_skill.py` to sync/check `.agents/skills/image2ppt/` in the intended worktree. Re-export with `exports/build_source_package.py` and `exports/package_skill_entries.py`; inspect their manifests. Claude has its own maintained source, `skills/image2ppt-claude/`: never overwrite or derive it from this skill. Skill ZIPs and the fixed-core source export are not a complete automatic backend deployment.
