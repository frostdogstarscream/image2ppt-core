---
name: image2ppt
description: Develop and validate the Image2PPT upload service, OpenCode and direct MoMA runtimes, frozen parallel pipeline, editable reconstruction and actual PowerPoint results.
---

# Image2PPT automatic product engineering

The upload worker calls `product.parallel_pipeline.run`. Use the recorded `orchestrator` and `visual_review_backend` in the uploaded job's request to identify which route that entrypoint dispatched.

| Route | Who chooses the reconstruction | What to check |
| --- | --- | --- |
| OpenCode, default for `POST /api/v1/jobs` | OpenCode Session using `skills/image2ppt-runtime/SKILL.md`, the shared reconstruction policy and MCP tools | For host-prepared jobs, inspect the baseline manifest and Office receipts. Check the Session review and the final delivery audit against the actual PPTX. |
| Direct MoMA, explicit `orchestrator=editable_reconstruction` | MoMA using `runtime/editable_runner.py`, the shared reconstruction policy and in-process tools | Inspect the job's `runtime/editable-turns`, pending candidate IDs, Office receipts, MoMA review and delivery audit. This path starts from the uploaded images. |
| Frozen r10/R12, `orchestrator=pipeline` | The text, asset and surface pipeline with MoMA planning | Use the comparison entrypoints or the legacy `/api/jobs` default when `IMAGE2PPT_DEFAULT_ORCHESTRATOR` does not select OpenCode. Check Office and the independent Go review in the frozen pipeline. |

The shared agent rules live in `skills/image2ppt-runtime/reconstruction-policy.md`. This developer Skill guides engineering and diagnosis. Do not hand-author SlideIR or assemble a deck to substitute for the upload worker's automatic result. Label manual comparisons and diagnostic fixtures separately.

Preserve source wording, numbers, hierarchy, layout, colors and proportions. Preserve originals, historical outputs, existing services and unrelated worktree changes. Use the existing fixed compiler and public SlideIR contract.

## Locate and run the real entrypoint

1. Identify the actual Git worktree, branch, dirty changes and task directory. Read that worktree's `AGENTS.md` and `services/product/PARALLEL_R10.md`. If `services/product/product/parallel_pipeline.py` is absent, locate the established product worktree; do not bootstrap from the legacy manual workflow or invent a replacement implementation.
2. Read [automatic product development and acceptance](references/parallel-product.md). For compiler diagnostics only, read [core tooling](references/core.md) and [SlideIR](references/slideir.md); those documents describe downstream capabilities, not the automatic planning entrypoint.
3. Submit fresh images through `POST /api/v1/jobs` and select the requested route. Use `run_local.py` and the r10/R12 pipeline for baseline comparisons. Inspect the request, candidate state and final artifact hashes in the same job directory.

## Engineering invariants

- Text, asset and surface evidence share one immutable source. Text supplies OCR, content and typography. Asset evidence covers complete logical objects and original RGBA or layers. Surface evidence covers backing, cards and simple geometry. Reuse LayerD per input. SAM can propose an outline. Source-pixel ownership must justify any erasure.
- SceneGraph owns the fused scene. Resolve intrinsic logo lettering, independent labels, parts, backing and occlusions before rebuilding. Keep unresolved evidence explicit. Only the scene adapter emits public SlideIR; branches must not independently publish or delete other branches' content.
- Target local object/region transactions. Font sharing is not a pixel dependency; an unrelated failed object must not veto all page surfaces. Split safe subregions and retain their dependencies. Indivisible parts and overlapping ownership require atomic replacement. These are development requirements: inspect code and receipts before claiming they are implemented.
- Keep source-plate as safe backing until source ownership, replacement and backing repair are proved. Never overlay new text on uncleared old text, erase by bbox, hide the whole plate to inflate editability, or cut an uncertain object at a region boundary. Raster islands require separate validation before activation.
- Preserve verified stable asset bytes unless changed source/ownership requires revalidation. Maintain input/version/hash-bound caches, durable external-call reservations and task-level generation commits. An unknown paid response must not be automatically resent as the same request.
- Measure visual integrity and editability separately. Native-node count or the existence of one date textbox does not prove a reconstructed page. Missing independent denominators produce UNKNOWN/null recall. Raster fallback is safe delivery, not editable success.
- Validate the final PPTX in actual PowerPoint by editing a copy, saving and reopening it. Inspect the complete vacated site, siblings and pixels outside each transaction. Bind the route's Office and visual review receipts to the final file and edit previews. Record unavailable checks as UNKNOWN.

## Verification and reporting

Run focused product/planner/orchestrator tests for behavioral changes and `./image2ppt.ps1 test` for protocol changes. Compiler/OOXML changes additionally require the synthetic demo, `scripts/check_demo.mjs`, actual-file previews and retained reports. Synthetic tests establish behavior, not real-slide accuracy or human acceptance. A documentation-only change does not require model calls or rebuilding customer decks.

Inspect real source/final previews and report tested versus untested gates. Keep `human_acceptance`, WPS checks, automatic quality, staged transactions and publication distinct. Deliver PPTX, whole-page previews and a short honest explanation when generation is requested; retain detailed diagnostics internally unless requested. State partial editability and full-raster pages without claiming all objects are independently editable.

## Historical manual workflow and maintenance

The previous Codex visual planning/hand-written SlideIR procedure is preserved in [historical manual workflow](references/legacy-manual-workflow.md). Its code, fixed compilation and replay abilities remain available. It is not the default product route, and its approximate mean-pixel-difference threshold of 10 is not a universal r10 acceptance threshold.

Edit `skills/image2ppt/`, then use `exports/sync_codex_skill.py` to sync/check `.agents/skills/image2ppt/` in the intended worktree. Re-export with `exports/build_source_package.py` and `exports/package_skill_entries.py`; inspect their manifests. Claude has its own maintained source, `skills/image2ppt-claude/`: never overwrite or derive it from this skill. Skill ZIPs and the fixed-core source export are not a complete automatic backend deployment.
