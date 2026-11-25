# Parallel product development and acceptance

## Entrypoints and evidence

Run from the product worktree. Read the route table in [the engineering Skill](../SKILL.md) and the job's `request.json`. The versioned upload API defaults to OpenCode. Explicit `orchestrator=editable_reconstruction` selects the direct MoMA runner. Both use the shared runtime decision and edit-evidence policy. The r10/R12 comparison path runs through the same dispatcher. Inspect CLI `--help` before adapting commands.

```powershell
# Primary OpenCode mainline (against 4090 / local tunnel):
.\scripts\start_4090_bridge.ps1
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

Inspect the actual job: `request.json`, `provenance.json`, `page-*/barrier-1.json`, the final IR paths in `result.json`, their `scene.json`, `transactions.json`, ownership maps, `quality-round-*.json`, `qa.json`, Office reports and SQLite call/commit records. Do not infer page-specific causes solely from a PPTX node count. Bind any new region/coverage diagnosis to these files and their hashes; mark absent evidence UNKNOWN.

Classify content conflict, erase/backing uncertainty, incomplete assets, local surface closure, compilation, unavailable QA and page fallback separately. Repair only the affected dependency subgraph. Keep successful asset bytes and known raw model replies; unknown responses are not permission for identical paid retries. Source fallback is permitted for safe delivery but never counts as extraction success.

## Visual integrity and editability

Report these separately, including per-page values where available:

- Actual native text boxes, native shapes and independent asset nodes; distinguish node counts from complete semantic objects.
- Text/asset/surface recall only against a source-bound independent reference inventory. Model self-reported object totals are candidate counts, not reference truth. Missing denominators or unverified completeness mean null/UNKNOWN.
- Editable-object ratio and union of verified ownership pixels, with QA status. Do not add overlapping bbox areas or count retained raster as native area.
- Full-raster page numbers, partial pages, fallback reason counts and important unresolved objects.
- Visual integrity, residual checks, Office result, Go result, release qualification and human acceptance independently.

An object-count improvement does not establish quality. A visually faithful whole-page screenshot does not establish reconstruction. Keep legacy `reconstructed_pages` explicitly labeled as pages with any retained native/asset candidate if compatibility requires it; add honest completion fields instead of silently changing its meaning. Engineering targets such as 95% text recall are not achieved values or substitutes for independent references.

The historical mean pixel difference near 10 is a sample-specific manual heuristic, not the sole automatic gate. Use source-bound object-type metrics and calibrated renderer tolerance alongside wording, completeness, residual and editability checks. Do not loosen thresholds globally to pass a failing page.

## Actual PowerPoint and route-specific visual review

Use copies of the final PPTX. For staged target objects, test movement/scaling, deletion and save/reopen, checking effective text/run styles, other objects and pixels outside the authorized transaction. For text changes, also verify content editing when claimed. Preserve original output and user-open Office documents.

Check the deleted object's proven pixels against independently justified backing and neighboring layers. Inspect the whole vacated footprint and a margin in the actual render after moving the entire icon clear. Where foreground and supporting shape are independently editable, inspect the foreground removed with the backing in place and then both removed. Keep the original, before/edited renders, action, object IDs and hashes together. Record untested actions and missing independent background evidence. The observation area can include neighboring pixels. Source modification remains limited to its proven owners.

In frozen r10/R12, Go compares the source and the final render with the actual edit panels when available. For OpenCode and direct MoMA, check the recorded runtime visual review, its evidence and the delivery audit. Verify each result's PPTX hash and tested actions. Record PowerPoint, WPS and human checks separately.

Keep `parallel_auto` closed until all separately defined release gates are evidenced. Documentation edits, synthetic tests and one provider verdict never change that state. `parallel_review` can deliver a safe partial result with an honest brief explanation; no manual deck may be substituted as an automatic success.
