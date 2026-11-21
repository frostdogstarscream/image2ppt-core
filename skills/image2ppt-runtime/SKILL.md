---
name: image2ppt-runtime
description: Reconstruct uploaded slide images into maximally native-editable PowerPoint decks using deterministic backbone perception/staging, targeted exception repair, and independent scorecard/visual audit.
---

# Host-precomputed Image2PPT workflow

1. **Stage 1 — Read the host evidence and deterministic baseline.** Start with `get_job_context`. The host has already run perception for every page, compiled the baseline deck and recorded its actual Office checks and scorecard. Read a numbered question block and its image with `get_evidence`. The original is available by evidence ID. Use `prepare_page_evidence(page)` only to request a new measurement.
2. **Stage 2 — Resolve open questions.** Inspect a question block and its numbered image on each page with open issues before final review. Prioritize changes that make a field independently editable or restore an entire image or container. Use `plan_marked_page` to select up to 30 numbered questions at once. Supply only mark, route, role and movement partners (`move_with` mark numbers). Use the provided roles. For a low-confidence native text issue, call `review_numbered_text` with the listed object IDs. The host reads the original crop and checks the proposed reading against the glyphs. Commit a returned candidate for Office validation. Leave uncertain pixels in the source plate. Preserve intrinsic screenshot lettering within its image. If the evidence supports no improvement, record the reasons in the visual review.
3. **Stage 3 — Incrementally rebuild.** Call `batch_reconstruct_page` with the affected block number or issue object IDs. Commit each VALIDATED candidate using its candidate ID and explicit version. Read the returned failure counts before retrying. Inspect another block when a decision still has a source-backed improvement.
4. **Stage 4 — Review and deliver.** Compile, validate Office, fetch the required evidence in small batches, review the rendered changes and finalize. Check content, completeness, layout, residual, background and editing. The host compares the resulting file with its frozen baseline before delivery. Report uncertainty honestly.

Use numbered issues to repair each affected page in at most two rounds. Read the independent audit and the actual PowerPoint edit panels. Stop when no newly validated edits can be committed. Missing independent review leaves the job in REVIEW_REQUIRED.

An independently movable complex object may remain an image. Keep source identities through retries. Surface plans can include an optional outline hint (roundRect, homePlate or chevron). The host measures the outline and owns all replacement masks. A new upload with only page-sized raster content produces diagnostic results without a downloadable deck.
