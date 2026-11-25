# OpenCode product acceptance and frozen parallel comparisons

## Entrypoints and evidence

Run from the product worktree. The primary automatic mainline is the OpenCode runtime (`services/product/OPENCODE_RUNTIME.md`, `/api/v1/jobs`, `services/product/opencode_client.py`, and `skills/image2ppt-runtime/SKILL.md`), with two-tier hard-safety vs soft-fidelity gates configured in `configs/fidelity_tolerance.json`. `services/product/product/parallel_pipeline.py` (`PARALLEL_R10.md`) is retained as a frozen comparison baseline. Inspect CLI `--help` before adapting commands. Typical PowerShell entrypoints are:

```powershell
# OpenCode mainline, using the profile for the intended service and its active tunnel:
python services/product/opencode_client.py --profile .runtime/opencode-client.dpapi submit C:/slides/page.png --tool-profile all
# Frozen r10/R12 comparison baseline:
python services/product/launch.py --port 8770
python services/product/run_local.py --image C:/slides/page.png --output outputs/new-job --pipeline-mode parallel_review --object-extraction auto --prompt-key
python services/product/run_parallel_baseline.py <baseline.json> <new-output-directory> --prompt-keys
```

Use a new job/output directory; preserve live services and prior outputs. Keys use the existing password UI, hidden prompt or authorized process environment, never documentation, source, logs or committed configuration. Reuse existing authorization without inventing additional approval steps. Do not read other applications' credential stores. Use the configured providers; the skill does not authorize a new provider or unrelated uploads.

The local OCR manifest must verify its three ONNX files; task processing does not silently download them. LayerD and SAM share a physical-GPU lock across processes, including loading and inference. Keep established CPU/API/Office/compiler limits and bounded object repair budgets. Do not terminate neighboring products to free resources.

## Frozen r10/R12 barriers and local transactions

| Stage | Responsibility | Boundary |
| --- | --- | --- |
| Text perception | Local OCR, MoMA content/roles/emphasis, installed font candidates | OCR boxes locate text; they are not ink masks. Content agreement, placement and erase evidence are distinct. Low score alone need not reject corroborated content; disagreement cannot silently rewrite text. |
| Asset perception | Complete semantic objects, parts/exclusions, intrinsic lettering, original LayerD RGBA | No early flattening of all SAM masks, no assumed white-background removal, no stable PNG resampling. |
| Surface perception | Page backing, cards, separators, simple geometry and complex-raster candidates | A proposed rectangle does not authorize clearing everything it intersects. |
| Barrier 1 | SceneGraph ownership and relations | RESOLVED/AMBIGUOUS/UNOWNED remain explicit. Sharing a style or intersecting bboxes is not semantic ownership. |
| Constrained reconstruction | Candidate text, Alpha/Unmix/SAM assets, native or raster backing | Keep real white content, holes, soft Alpha and whole disconnected parts. Only proven source pixels may be cleaned. |
| Barrier 2 | Local replacement closure and staging | Include all required old owners, backing patches and occlusions. An indivisible replacement commits or reverts together. Independent subregions continue. |
| Compile and QA | Fixed compiler, actual PowerPoint and independent Go | Bind exact scene, package, preview and QA hashes before task-level generation publication. |

A region can contain editable text over a retained raster backing when its original ink is safely removed. A failed card conversion need not revert independent text transactions. Conversely, a new card cannot cover uncleared lettering. Cross-region graphics require explicit dependencies or a larger indivisible transaction, not clipping. Shared typography must remain coordinated without becoming a page-wide candidate acceptance group.

These are target engineering rules, not proof of implementation. The r10 baseline still has page-wide surface and shared-style fallback behavior; the local-transaction repair must demonstrate its replacement in code and real receipts. Never describe a planned gate or a newly added unintegrated module as active.

## Source-plate and failure diagnosis

First preserve the source plate and clean only proven object ink. Later local RasterTile work is a separate shadow experiment with coordinate, alpha, overlap and backing evidence. Neither a source plate's presence nor its removal proves editing quality.

For an OpenCode upload, start with `request.json`, `host-baseline/manifest.json`, the corresponding baseline scorecard, the actual Office checks, `result.json` and `runtime/delivery-selection.json`. Follow the object IDs through the page's numbered evidence, plans, scene, independent audit and final PowerPoint file. Verify the selected file hash against the artifact list. When selection is empty, distinguish the diagnostic preview and any internal candidate file.

For a frozen parallel run, inspect `provenance.json`, `page-*/barrier-1.json`, the final IR paths in `result.json`, their `scene.json`, `transactions.json`, ownership maps, `quality-round-*.json`, `qa.json`, Office reports and SQLite call/commit records. Bind diagnoses to the actual job evidence and record absent evidence as UNKNOWN.

Classify content conflict, erase/backing uncertainty, incomplete assets, local surface closure, compilation, unavailable QA and fallback separately. Repair only the affected dependencies. Preserve successful asset bytes and known raw model replies. Keep unknown pixels in safe backing. For editable delivery an image fallback covers a single identified object, and its move/delete must leave clean backing and intact neighbors. A full-raster page may be retained as a diagnostic.

## Visual integrity and editability

Report these separately, including per-page values where available:

- Actual native text boxes, native shapes and independent asset nodes; distinguish node counts from complete semantic objects.
- Text/asset/surface recall only against a source-bound independent reference inventory. Model self-reported object totals are candidate counts, not reference truth. Missing denominators or unverified completeness mean null/UNKNOWN.
- Editable-object ratio and union of verified ownership pixels, with QA status. Do not add overlapping bbox areas or count retained raster as native area.
- Full-raster page numbers, partial pages, fallback reason counts and important unresolved objects.
- Visual integrity, target residual checks, actual Office result, independent visual review (provider and model used), release qualification, human and WPS checks independently. Frozen r10/R12 uses Go for its independent review.

An object-count improvement does not establish quality. A visually faithful whole-page screenshot does not establish reconstruction. Keep legacy `reconstructed_pages` explicitly labeled as pages with any retained native/asset candidate if compatibility requires it; add honest completion fields instead of silently changing its meaning. Engineering targets such as 95% text recall are not achieved values or substitutes for independent references.

The historical mean pixel difference near 10 is a sample-specific manual heuristic, not the sole automatic gate. Use source-bound object-type metrics and calibrated renderer tolerance alongside wording, completeness, residual and editability checks. Do not loosen thresholds globally to pass a failing page.

## Real PowerPoint edits and independent visual review

Use copies of the selected PPTX, or name the internal candidate and its hash if no file was selected. Test the edit at the level the user will select it. For a field, replacing the entire textbox should change only that word or number and leave the separator alone. For a card, one selection should move the intended label and icon together. After changing a card fill, remove its independent photo and inspect the exposed fill and title. For applicable tables and charts, verify an edited cell or data point. Save, close and reopen to check persistence, old locations and neighbors. Record an unconfirmed original reading separately from a successful text edit.

Check inside the deleted object's proven owner region against independently justified backing and legitimate neighboring layers. Save before/delete/move renders, owner evidence and reference hashes; report confirmed residual pixels and PASS/FAIL/UNKNOWN. Do not use a candidate's own mask/recomposition as its only correctness reference. An outside-pixel count of zero cannot prove no old cloud half or text remains.

On the OpenCode mainline the configured independent model compares the source, actual Office render and available edit panels. Check the independent model identity and its receipt for the exact tested composition. A transport failure leaves review unavailable. A changed composition requires a matching receipt. Frozen r10/R12 uses its historical Go review and page-level fallback rules.

Keep `parallel_auto` closed until all separately defined release gates are evidenced. Documentation edits, synthetic tests and one provider verdict never change that state. `parallel_review` can deliver a safe partial result with an honest brief explanation; no manual deck may be substituted as an automatic success.
