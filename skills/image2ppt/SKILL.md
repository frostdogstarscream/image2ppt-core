---
name: image2ppt
description: Diagnose and improve automatic reconstruction of uploaded images into editable PowerPoint slides.
---

# Image2PPT product engineering

Choose a change the user expects to make on a previously unseen slide. Open the delivered PowerPoint and try it. A program can report a hundred successful deletes while the title the user wants to change still lives in the background.

Use the uploaded source, the selected PPTX, its embedded raster and a save/reopen edit to locate the first point at which that change was lost. Follow the object from perception and planning into the scene, compiled file and edit receipt. Inspect the actual service revision and upload route if they affect the diagnosis. An instruction in this skill is a claim to test. If a rule rewards the wrong result, fix that rule in the product and correct the documentation.

## What counts as an edit

- Replace a complete business field or number in its own textbox. Its neighbor and divider should stay put. Record the ability to edit separately from confidence in an AI-generated glyph. Preserve disputed numbers and units for confirmation.
- Recolor a card and move it with its intended children in one selection. Delete an independent photo after recoloring and inspect the newly exposed surface. A complex diagram or photo may be one movable object. A slide flattened into a picture cannot qualify.
- Inspect a saved deletion against the actual visible lower layers. Also inspect the raster inside the same PPTX. A small difference from an estimated fill is a diagnostic measurement. A readable old word, lost content or damaged neighbor is an edit failure. Normal viewing size is the visual reference. Minor font, shadow and gradient differences are acceptable.
- Preserve unknown source pixels while repairing the known owner. Sharing a font does not join two objects' erase regions. Restore or reject each independent part separately when possible. Keep original inputs, historical outputs, unrelated worktrees and verified asset bytes. SceneGraph owns the relationships and its adapter emits the public SlideIR. Use the fixed compiler.

## Running and reporting

Fresh OpenCode jobs enter through `POST /api/v1/jobs` using `services/product/opencode_client.py`. The host prepares a baseline, the Agent can address omissions, and the service lists the selected files. Read [the engineering reference](references/parallel-product.md) for the commands and evidence relevant to the failure. The frozen r10/R12 comparison uses `product.parallel_pipeline.run` and `services/product/run_local.py`. Label source-evidence replay, fixtures and manual comparisons by their actual origin. A manually assembled deck cannot establish automatic reconstruction.

For every claim identify the checked file and the actual edit. Report visual quality, editable completion, Office evidence, independent model verdict and human/WPS acceptance separately. With no independent reference, recall is unknown. Partial editable copies should identify usable edits and remaining defects. The download list and selection receipt identify what the user actually received.

Run focused tests for changed behavior and `./image2ppt.ps1 test` for protocol changes. See the [core](references/core.md) and [SlideIR](references/slideir.md) references for compiler diagnostics. Bind model calls and caches to the job, preserve durable reservations and keep credentials out of source and logs.

Maintain this file in `skills/image2ppt/`. Sync its generated copy with `exports/sync_codex_skill.py`, verify with `--check` and validate both exports. The [historical hand-planning workflow](references/legacy-manual-workflow.md) remains available for labeled comparisons. Claude has its own maintained skill.
