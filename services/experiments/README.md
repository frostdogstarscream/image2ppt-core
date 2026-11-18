# Six-model comparison over OpenSSH

All six experimental inference paths execute on the RTX 4090 host. The local Windows machine submits page/crop PNGs on SSH stdin, verifies input and artifact hashes, and runs PowerPoint. Go hy4 is the external planning model. CPU inference (including VTracer) also runs on the remote host. LayerD uses the existing authenticated remote loopback API; model credentials stay with that service.

The tested experiment root is `F:/i2oc/six-model-comparison`. Install the pinned artifacts with `prepare_models.py <root>/weights` and copy this directory's `worker.py` to that root. The tested isolated packages are doclayout-yolo 0.0.4, pymatting 1.1.16 and vtracer 0.6.15 under `deps`, with numpy 1.26.4 under `compat` for DocLayout-YOLO and ViTMatte/PyMatting. Existing PyTorch, Transformers, Ultralytics, PaddleOCR, and LayerD installations supply the underlying runtimes. Keep experiment dependencies isolated from the live services.

Set `IMAGE2PPT_EXPERIMENT_CONFIG` on the experiment product host to a JSON object mapping `v3`, `pp-l`, `doclayout`, `omni`, `vitmatte`, `local-layerd` and `vtracer` to OpenSSH argv arrays. For example:

```json
{
  "omni": ["ssh", "-o", "BatchMode=yes", "research-4090", "F:/i2oc/venv/Scripts/python.exe F:/i2oc/six-model-comparison/worker.py F:/i2oc/six-model-comparison --device cuda:0"],
  "timeout": 1250
}
```

Use the same command for doclayout, vitmatte, local-layerd and vtracer. For v3 and pp-l use the existing layout-venv interpreter with `--device cpu` and the remote PaddleX cache. Configure the SSH host alias with the correct wired interface and existing identity. Layout-L was tested with oneDNN disabled. Weight file size and upstream SHA for LFS files are verified on acquisition; per-file SHA-256 and source revision are persisted.

Submit `experiment_profile` through `POST /api/v1/jobs` or `python services/product/opencode_client.py ... submit --experiment-profile <name>`. Use OpenCode Go hy4. `run_comparison.py --url <local-api> --jobs <job-root> --out <run-output> --images page1.png page2.png --holdout p08.png --holdout p07.png` runs the matched matrix using the service access token in the environment. It preserves admissions, the 45-minute snapshot for still-running jobs, and terminal stage results. A missing result or absent model receipt is unknown. Count only independently tested PPTX edits as successful.

## Deployment smoke test

OpenSSH inference on real source pixels produced 50 PP-DocLayout-L boxes and 55 DocLayout-YOLO boxes on the first page. OmniParser ran on the full page and cropped windows; counts before deduplication depend on the selected windows. A local crop from that page passed through the remote LayerD service, ViTMatte plus PyMatting, and CPU VTracer, returning layer PNGs, an RGBA image and SVG. These receipts verify model execution, not editability.

## Reproduce the two-page evaluation

On the Windows PowerPoint host, use the product Python environment with pywin32. Keep the job directory and its original artifact paths available. Supply the two independent field inventories (`reference-fields-1.json` and `reference-fields-2.json`) and a private UTF-8 JSON array of edit cases. Each case contains the page number, target label, `[left, top, width, height]` on the 1672 × 941 source, action (`text`, `move`, `delete`, or `fill`), and replacement text. For example, a synthetic entry is `[1, "Target", [100, 80, 64, 28], "text", "Changed"]`. Use fresh output directories outside the checkout.

```powershell
python services/experiments/evaluate_two_pages.py --job $combinedJob --references $references --cases $editCases --out $combinedOut
python services/experiments/evaluate_two_pages.py --job $baselineJob --references $references --cases $editCases --out $baselineOut
python services/experiments/compare_deliveries.py $combinedOut $baselineOut
```

The evaluator checks the composition digest, locates the corresponding Office checks, and probes the six specified actions on private copies of the deck. If a terminal run has only committed scenes, the evaluator labels the independently compiled deck as diagnostic. The two evaluation reports contain field matches and model decisions. The comparison includes the number of differing PowerPoint render pixels for each page.

## Paired business-page result

On implementation `6ced62b`, the combined and baseline runs each scored **0/6** prescribed edits. Their PowerPoint exports at source resolution matched pixel for pixel on both pages. The six target operations covered a numeric field, a labeled role box, a software screenshot, an independently editable project name, a table field and a card fill.

| Page | Matched native fields | Native share of structured content | Verified movable picture share | Duplicate share | RGB MAE vs. source | SSIM |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 28/93 | 30.55% | 1.99% | 0% | 8.230 | 0.8625 |
| 2 | 9/74 | 18.30% | 0% | 0% | 5.168 | 0.9242 |

All six remote workers returned model receipts. Hy4 selected a software screenshot and two tables. The batch reconstruction tool checked that both marked plans existed, then processed the existing SceneGraph without receiving the selected object IDs in either batch call. No independently deletable screenshot or native table survived in the deck. The backing-fit strategy ranked first on all eight ranked asset trials. All eight SVG candidates failed the contrast check.

At 45 minutes the combined job had composed a two-page deck with 42 and 15 committed objects. Its SHA-256 was `3befe29bb8dfff9f3c917b477762959b9aae82e89d2a281997625457887d2051`. Office passed 114 move/scale or delete records, 86 text replacement records and 57 residual checks on those bytes. Both agents subsequently ended in `REVIEW_REQUIRED` after `compaction.failed`. Human and WPS acceptance are unverified. This comparison covers the baseline and combined configuration on these two pages. It does not rank the six individual groups.

## Executing the marked plans (2026-11-18)

**Recorded-plan replay: 6/6 prescribed edits in desktop PowerPoint. Fresh combined upload: no deck.** The replay reused the two pages' original Hy4 marks, plans and text trials, applied the new product reconstruction stages, and incorporated frame/exclusion feedback from a preceding Office pass. The independent edit cases were used after compilation. Production did not receive their answers. This experiment measures the reconstruction changes on fixed planning evidence. The separate fresh upload stopped at the first Hy4 inference with HTTP 429 (`provider.quota`), before either page had a plan. Its terminal status was `REVIEW_REQUIRED` and QA was `UNKNOWN`.

| Page | Actual edit after save/reopen | Actual PowerPoint target | Changed pixels outside expected region |
| --- | --- | --- | --- |
| 1 | Replace numeric field | Text box | 0 |
| 1 | Move role and label together | One native group | 0 |
| 1 | Delete dashboard screenshot | Movable picture with verified backing | 0 |
| 2 | Change the project name independently | Separate native text box | 0 |
| 2 | Replace the industry field | Left native table, row 3, column 3 | 0 |
| 2 | Recolor industry card | Native shape | 0 |

| Page | Matched native fields | Native structured-content area | Verified movable area | Duplicate area | Raster plate area | RGB MAE | SSIM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 29/93 | 32.05% | 3.96% | 0% | 63.99% | 8.482 | 0.8550 |
| 2 | 43/74 | 45.04% | 5.51% | 0% | 49.45% | 11.568 | 0.7834 |

All area shares use structured source content as the denominator. Native cells count toward field recall. The checked deck SHA-256 is `b35cf1c11f0e3e49b8fddcd0f6d2f695b225fe5e50f189e7bd48e95a8b788334`. On those bytes Office passed 140 move/scale or delete records, 82 text replacement records and 70 residual checks. The numeric field had a recorded LayerD-backed expansion trial in the original run and was listed among its first-pass failures. The replay committed the recovered text. A different card title with a suspected missing OCR glyph stayed source-backed. Both pages fell in whole-page SSIM against the paired baseline. Page 2's editable tables have left-aligned text and dark rules where the source has centered text and pale blue rules. Human and WPS acceptance are pending.

Independent OpenSSH calls on the 4090 host also produced the following receipts. Counts are detector outputs before planner selection. The asset runs do not establish which method contributed to the edit score.

| Worker | Real input | Output |
| --- | --- | --- |
| PP-DocLayout-L | Both source pages | 50 and 24 boxes |
| DocLayout-YOLO | Both source pages | 55 and 36 boxes |
| OmniParser V2 detection | Both pages with local windows | 152 and 179 boxes |
| ViTMatte-Small + PyMatting | 28 × 28 padded source crop and recorded foreground mask | RGBA |
| Local LayerD | 168 × 168 padded dashboard crop | Four layers |
| VTracer | 12 × 12 transparent icon | SVG |

Run the replay against a retained job with source-bound artifacts. Each output directory must be new. Set the optional repair argument to the preceding replay directory only when it contains hash-matched Office checks and their location in `office-check-path.txt`. The two-page evaluator joins mark and plan evidence, SceneGraph owners, actual slide object names and save/reopen probe results in `field_evidence`.

```powershell
python services/experiments/replay_marked.py $job $newOut --repair-from $previousOfficeReplay
$deck = (Get-ChildItem "$newOut/composition" -Recurse -Filter deck.pptx).FullName
python services/experiments/probe_fields.py $deck "$newOut/edits" --cases $editCases
```

