---
name: image2ppt-runtime
description: Reconstruct uploaded slide images into maximally native-editable PowerPoint decks using deterministic backbone perception/staging, targeted exception repair, and independent scorecard/visual audit.
---

# OpenCode image reconstruction

Use `get_job_context` to see the uploaded originals, available evidence and budget. When the host has prepared a baseline, inspect its Office checks and scorecard, then investigate the open questions. Request a new measurement with `prepare_page_evidence` when needed.

For pages with numbered issues, inspect the issue image and inventory with `get_evidence`. Use `plan_marked_page` for up to 30 marks at a time, supplying the mark, route, role and movement partners. Use `review_numbered_text` when a crop contains uncertain characters. Use `batch_reconstruct_page` on the affected block or object IDs when that tool fits the task. On host-prepared uploads, keep numbered-issue repairs within the existing two rounds per affected page.

Compile committed objects, obtain the required Office evidence and review the rendered edits. The host compares the final candidate with its baseline when one exists. Inspect the delivery audit and use `finalize_job` to record the outcome. The runtime appends the shared candidate and edit-evidence policy to these instructions.
